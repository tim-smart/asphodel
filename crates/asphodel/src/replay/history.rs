//! Real-history replay: `asphodel replay --corpus <file> --mode
//! live|replay|fast`.
//!
//! The corpus, the cassette, the probes file and the report all live under
//! the private dir, and are refused anywhere else. The LLM for `live` and
//! `fast` is built as `serve` builds it, from `[llm]` in `--config` and
//! the environment, or from `ASPHODEL_LLM_SCRIPT` in tests; `replay` never
//! calls one. The real models run with ONNX Runtime's intra-op threads
//! pinned (`--onnx-threads`, default 1), or `ASPHODEL_MODELS=fake` runs the
//! fakes.

use std::collections::BTreeSet;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context as _, anyhow, bail};
use asphodel_core::config::{Deployment, LLM_API_KEY_ENV, LlmAuth, Secret};
use asphodel_core::extraction::guidance_hash;
use asphodel_core::models::{
    CodexResponses, FakeLlm, LlmClient, LlmSettings, OpenAiCompatible, TokenStore,
};
use asphodel_core::store::bank::BankIdentity;
use asphodel_core::{Clock, Models, Service, SimulatedClock, SystemClock, Tuning, VERSION};
use jiff::tz::TimeZone;

use super::cassette::Recorder;
use super::engine::{Engine, Llm, Settings};
use super::report::{Call1, Flags, Report};
use super::scenario::Group;
use super::timeline::Timeline;
use super::{Finished, NO_DEADLINE, SHADOW_FILE, failure};
use crate::cli::{RefreshMode, ReplayArgs, ReplayMode};

pub(super) fn execute(args: &ReplayArgs) -> anyhow::Result<Finished> {
    let corpus_arg = args
        .corpus
        .as_deref()
        .ok_or_else(|| anyhow!("--corpus is required"))?;
    let mode = args
        .mode
        .ok_or_else(|| anyhow!("--mode live|replay|fast is required with --corpus"))?;
    if args.self_test && mode == ReplayMode::Live {
        bail!(
            "--self-test can't run in live mode: live measures latency and records as it goes, so two runs can't be identical. Record first, then self-test with --mode replay or fast"
        );
    }
    if args.no_cache && mode != ReplayMode::Live {
        bail!("--no-cache re-records, so it goes with --mode live");
    }
    if args.refresh.is_some() && mode != ReplayMode::Fast {
        bail!(
            "--refresh says how fast mode answers refreshes; live and replay answer them by request"
        );
    }
    let refresh = args.refresh.unwrap_or(RefreshMode::Recorded);

    let dir = super::private_dir(args.replay_dir.as_deref())?;
    let _lock = super::lock_private_dir(&dir)?;
    let corpus_path = super::inside_private(&dir, corpus_arg, "the corpus")?;
    let corpus = super::corpus::load(&corpus_path)?;
    let stem = corpus_path
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .filter(|stem| !stem.is_empty())
        .unwrap_or_else(|| "history".into());
    let cassette_path = match &args.cassette {
        Some(path) => super::inside_private(&dir, path, "the cassette")?,
        None => {
            let cassettes = dir.join("cassettes");
            super::refuse_symlink(&cassettes)?;
            std::fs::create_dir_all(&cassettes)
                .with_context(|| format!("creating {}", cassettes.display()))?;
            cassettes.join(format!("{stem}.jsonl"))
        }
    };
    let report_path = match &args.report {
        Some(path) => super::inside_private(&dir, path, "the report")?,
        None => super::default_report(&dir, &format!("{stem}-{}", mode_name(mode)))?,
    };
    let aggregate_path = args
        .aggregate
        .as_deref()
        .map(|path| super::aggregate_path(path, &dir))
        .transpose()?;
    let labelling_path = args
        .labelling
        .as_deref()
        .map(|path| super::inside_private(&dir, path, "the labelling material"))
        .transpose()?;
    let labels_path = args
        .labels
        .as_deref()
        .map(|path| super::inside_private(&dir, path, "the labels file"))
        .transpose()?;
    let probes_path = args
        .probes
        .as_deref()
        .map(|path| super::inside_private(&dir, path, "the probes file"))
        .transpose()?;
    if let Some(path) = &labelling_path {
        let occupied = [
            Some(&corpus_path),
            Some(&cassette_path),
            probes_path.as_ref(),
            Some(&report_path),
            aggregate_path.as_ref(),
            labels_path.as_ref(),
        ];
        if occupied.into_iter().flatten().any(|other| path == other) {
            bail!(
                "--labelling {} collides with an input or output of this run",
                path.display()
            );
        }
    }
    let labelled = match &labels_path {
        Some(path) => super::labelling::labelled_queries(path)?,
        None => BTreeSet::new(),
    };
    let shadow_path = dir.join(SHADOW_FILE);
    super::refuse_symlink(&shadow_path)?;

    let fake = super::fake_models_requested()?;
    let threads = args.onnx_threads.or(NonZeroUsize::new(1));
    let models = if fake {
        Models::fake()
    } else {
        super::load_models(args.model_dir.as_deref(), threads)
            .context("a real-history replay needs the real models in ASPHODEL_MODEL_DIR")?
    };
    let group = if fake { Group::Ci } else { Group::Models };
    let probes = match &probes_path {
        Some(path) => super::probes::load(path, group)?,
        None => Vec::new(),
    };
    let tuning = super::layered_tuning(args, None, &stem, fake)?;
    let latency = args
        .latency
        .as_deref()
        .map(|text| super::scenario::duration(text).map_err(|error| anyhow!("--latency: {error}")))
        .transpose()?;
    // `replay` never calls the LLM, whatever is configured.
    let live = match mode {
        ReplayMode::Replay => None,
        ReplayMode::Live | ReplayMode::Fast => live_client(&tuning, args, &dir)?,
    };
    if mode == ReplayMode::Live && live.is_none() {
        bail!(
            "live mode needs an LLM: set [llm] in --config and {LLM_API_KEY_ENV}, or log in with `asphodel llm login` and pass --token-dir"
        );
    }

    let timeline =
        Timeline::from_corpus(&corpus, probes).map_err(|message| anyhow!("{message}"))?;
    let start = timeline
        .earliest()
        .ok_or_else(|| anyhow!("the corpus {} has no events", corpus_path.display()))?;
    let header = &corpus.header;
    let timezone = TimeZone::get(&header.timezone)
        .with_context(|| format!("the corpus's timezone {:?}", header.timezone))?;
    let identity = BankIdentity {
        owner_name: header.owner.name.clone(),
        owner_platform_ids: header.owner.platform_ids.clone(),
        assistant_name: header.assistant.clone(),
        timezone: Some(header.timezone.clone()),
    };
    let kind = mode_name(mode);
    let refresh_name = match mode {
        ReplayMode::Fast => match refresh {
            RefreshMode::Live => "live",
            RefreshMode::Recorded => "recorded",
            RefreshMode::Off => "off",
        },
        _ => "request",
    };

    let mut material = None;
    let finished = super::deliver(args, &report_path, aggregate_path.as_deref(), || {
        let clock = Arc::new(SimulatedClock::new(start));
        let store = super::open_store(&dir, Arc::clone(&clock) as Arc<dyn Clock>)?;
        store.check_fingerprint(&tuning.deletion_fingerprint())?;
        let service = Service::with_models(
            Arc::clone(&clock) as Arc<dyn Clock>,
            store,
            tuning.clone(),
            super::clone_models(&models),
        )?
        .with_reranker_deadline(NO_DEADLINE);
        service.ensure_bank_with_models(&header.bank, &identity)?;
        super::create_models(&service, &header.bank, &header.models)?;
        let recorder = Recorder::open(
            &cassette_path,
            mode,
            refresh,
            args.no_cache,
            live.clone(),
            tuning.llm.language.clone(),
            Arc::clone(&clock),
        )?
        .with_guidance(guidance_hash(tuning.extraction.guidance.as_deref()));
        let settings = Settings {
            bank: header.bank.clone(),
            timezone: timezone.clone(),
            latency: latency.unwrap_or(jiff::SignedDuration::ZERO),
            latency_from_cassette: latency.is_none(),
            until: args.until,
            labelling: labelling_path.as_ref().map(|_| labelled.clone()),
        };
        let engine = Engine::new(
            &service,
            Arc::clone(&clock),
            &timeline,
            &tuning,
            settings,
            Llm::Recorded(&recorder, mode),
        )
        .map_err(failure)?;
        let mut outcome = engine.run().map_err(failure)?;
        material = outcome.material.take();
        let purged_then_re_mentioned =
            super::write_shadow(&service, &tuning, &shadow_path, &outcome)?;
        Ok(Report {
            kind,
            scenario: stem.clone(),
            group: if fake { "fake" } else { "models" },
            version: VERSION,
            git_sha: option_env!("ASPHODEL_GIT_SHA"),
            corpus_hash: Some(corpus.hash.clone()),
            cassette_hash: Some(recorder.completed_hash()?),
            tuning: tuning.clone(),
            call1: Call1::of(&tuning),
            flags: Flags {
                latency_ms: latency
                    .map(|latency| u64::try_from(latency.as_millis()).unwrap_or(0))
                    .unwrap_or(0),
                until: args.until,
                refresh: refresh_name,
                mode: Some(kind),
                no_cache: args.no_cache,
                self_test: args.self_test,
                onnx_threads: (!fake).then(|| threads.map(NonZeroUsize::get)).flatten(),
            },
            probes: outcome.probes,
            purges_per_day: outcome.purges_per_day,
            purged_then_re_mentioned,
            fade_outs_per_week: outcome.fade_outs_per_week,
            bands_per_week: outcome.bands_per_week,
            extraction_lag: outcome.extraction_lag,
            refresh_calls_per_day: outcome.refresh_calls_per_day,
            injected_tokens: outcome.injected_tokens,
            profile_tokens: outcome.profile_tokens,
            call2_rate: outcome.call2_rate,
            agenda_lines_per_day: outcome.agenda_lines_per_day,
            significance_histogram: outcome.significance_histogram,
            kind_histogram: outcome.kind_histogram,
            memories: outcome.memories,
            llm: outcome.llm,
        })
    })?;
    if let (Some(path), Some(material)) = (&labelling_path, material) {
        let mut json = serde_json::to_vec_pretty(&material)?;
        json.push(b'\n');
        super::write_file(path, &json)
            .with_context(|| format!("writing the labelling material to {}", path.display()))?;
    }
    Ok(finished)
}

fn mode_name(mode: ReplayMode) -> &'static str {
    match mode {
        ReplayMode::Live => "live",
        ReplayMode::Replay => "replay",
        ReplayMode::Fast => "fast",
    }
}

/// The LLM `live` and `fast` call on a miss, built as `serve` builds its
/// own: the scripted fake from `ASPHODEL_LLM_SCRIPT` for tests, else `[llm]`
/// with the API key from the environment or the ChatGPT login under
/// `--token-dir`. `None` when nothing is configured.
fn live_client(
    tuning: &Tuning,
    args: &ReplayArgs,
    dir: &Path,
) -> anyhow::Result<Option<Arc<dyn LlmClient>>> {
    if let Some(path) = std::env::var_os(crate::serve::LLM_SCRIPT_ENV) {
        let script = std::fs::read_to_string(&path).with_context(|| {
            format!(
                "reading {} {}",
                crate::serve::LLM_SCRIPT_ENV,
                Path::new(&path).display()
            )
        })?;
        let model = tuning
            .llm
            .model
            .as_deref()
            .unwrap_or(crate::serve::FAKE_LLM_MODEL);
        tracing::warn!(
            "{} is set: live calls go to a scripted fake LLM, not a real one",
            crate::serve::LLM_SCRIPT_ENV
        );
        return Ok(Some(Arc::new(FakeLlm::from_script(model, &script)?)));
    }
    let deployment = Deployment {
        listen: String::new(),
        data_dir: None,
        config: args.config.clone(),
        allow_network_fs: false,
        model_dir: args.model_dir.clone(),
        token: None,
        llm_api_key: Secret::from_env(LLM_API_KEY_ENV),
    };
    let Some(settings) = LlmSettings::from_config(tuning, &deployment)? else {
        return Ok(None);
    };
    let client: Arc<dyn LlmClient> = match settings.auth {
        LlmAuth::ApiKey => Arc::new(OpenAiCompatible::new(settings)),
        LlmAuth::Chatgpt => {
            let token_dir: PathBuf = args.token_dir.clone().unwrap_or_else(|| dir.to_owned());
            // The login's expiry is in world time, so the live client reads
            // the system clock: the one place replay may.
            let mut client = CodexResponses::new(
                settings,
                TokenStore::open(&token_dir),
                Arc::new(SystemClock) as Arc<dyn Clock>,
            );
            if let Ok(issuer) = std::env::var(crate::cli::LLM_ISSUER_ENV) {
                client = client.with_issuer(&issuer);
            }
            Arc::new(client)
        }
    };
    Ok(Some(client))
}
