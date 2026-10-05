//! `asphodel report rescore`: the labelling material's recall pools, and
//! with `--refresh-queries` its refresh pools, scored again against another
//! rerank query (`docs/replay.md`, "Rescoring fixed pools").
//!
//! Each recall sample is found in the corpus by its session and time, and
//! its rerank query rebuilt from the corpus as replay builds it: the message
//! (after a short follow-up borrowed the previous one), or the conversation
//! query. The sample's candidates are scored with the reranker against it
//! and listed by logit, highest first, ties in the order the material gave.
//! Every sample and candidate id, memory and sentence is kept, so labels
//! written for the material apply to what this writes, and call 2's lists
//! are copied unchanged.
//!
//! With `--refresh-queries`, a TOML table of facet heading to another
//! query, every refresh sample whose facet has that heading, of any model,
//! is scored against that query instead, which becomes its `rerank_query`.
//! Its `query` stays, so facet labels still match, and its candidates stay
//! in the order the refresh ranked them, with the strength, score, rank,
//! citation, what took them and their handle as the refresh recorded them:
//! only the logits change, and the sample is marked reranked. A heading no
//! refresh sample has is refused, since it can only be a typo. The other
//! refresh samples, and all of them without the option, are copied
//! unchanged.
//!
//! The pools are fixed and recall samples are ordered by logit alone, so
//! this compares rerank queries; it says nothing about prefetch's final
//! ranking, which adds strength, state confidence and phase. Everything read and written
//! is history, so all of it stays under the private dir and no error
//! quotes it.

use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::path::Path;

use anyhow::{Context as _, anyhow, bail};
use asphodel_core::config::RerankQuery;
use asphodel_core::models::{Models, Reranker};
use asphodel_core::retrieval::{clean_query, conversation_query, effective_query};

use super::labelling::{Material, load_material};
use super::timeline::{Timeline, Turn};
use crate::cli::RescoreArgs;

/// Runs `asphodel report rescore`: exit 0 once the material is written, or
/// exit 2 when the inputs are refused or don't parse, writing nothing.
pub fn run(args: RescoreArgs) -> anyhow::Result<()> {
    if let Err(error) = execute(&args) {
        eprintln!("error: {error:#}");
        std::process::exit(2)
    }
    Ok(())
}

fn execute(args: &RescoreArgs) -> anyhow::Result<()> {
    let dir = super::private_dir(args.replay_dir.as_deref())?;
    let material_path = super::inside_private(&dir, &args.material, "the material")?;
    let corpus_path = super::inside_private(&dir, &args.corpus, "the corpus")?;
    let out_path = super::inside_private(&dir, &args.out, "the rescored material")?;
    let queries_path = args
        .refresh_queries
        .as_deref()
        .map(|path| super::inside_private(&dir, path, "the refresh queries"))
        .transpose()?;
    if out_path == material_path
        || out_path == corpus_path
        || queries_path.as_ref() == Some(&out_path)
    {
        bail!(
            "--out {} collides with an input of this run",
            out_path.display()
        );
    }
    let mut material = load_material(&material_path)?;
    let refresh_queries = match &queries_path {
        Some(path) => load_refresh_queries(path, &material)?,
        None => BTreeMap::new(),
    };
    let corpus = super::corpus::load(&corpus_path)?;
    let timeline =
        Timeline::from_corpus(&corpus, Vec::new()).map_err(|message| anyhow!(message))?;

    let mode = args.rerank_query.into();
    let mut queries = Vec::with_capacity(material.recall.len());
    for sample in &material.recall {
        let turn = timeline
            .turns
            .iter()
            .find(|turn| turn.session == sample.session && turn.at == sample.at)
            .ok_or_else(|| {
                anyhow!(
                    "sample {} has no prefetch in the corpus {} at its session and time; rescore the material with the corpus its run replayed",
                    sample.sample,
                    corpus_path.display()
                )
            })?;
        queries.push(rerank_query(turn, mode));
    }

    let models = if super::fake_models_requested()? {
        Models::fake()
    } else {
        let threads = args.onnx_threads.or(NonZeroUsize::new(1));
        super::load_models(args.model_dir.as_deref(), threads)
            .context("rescoring needs the real models in ASPHODEL_MODEL_DIR")?
    };
    rescore(&mut material, queries, models.reranker.as_ref())?;
    rescore_refresh(&mut material, &refresh_queries, models.reranker.as_ref())?;

    let mut json = serde_json::to_vec_pretty(&material)?;
    json.push(b'\n');
    super::write_file(&out_path, &json)
        .with_context(|| format!("writing the rescored material to {}", out_path.display()))
}

/// The refresh queries file: facet heading to query, every heading one
/// a refresh sample of `material` has and every query nonblank. Errors
/// name the file and never quote it.
fn load_refresh_queries(
    path: &Path,
    material: &Material,
) -> anyhow::Result<BTreeMap<String, String>> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let queries: BTreeMap<String, String> =
        toml::from_str(&text).map_err(|error| super::toml_error(path, &text, &error))?;
    let unknown = queries
        .keys()
        .filter(|heading| !material.refresh.iter().any(|s| &s.facet == *heading))
        .count();
    if unknown > 0 {
        bail!(
            "{} names {unknown} facet heading(s) no refresh sample of the material has",
            path.display()
        );
    }
    let blank = queries.values().filter(|q| q.trim().is_empty()).count();
    if blank > 0 {
        bail!("{} gives {blank} facet(s) a blank query", path.display());
    }
    Ok(queries
        .into_iter()
        .map(|(heading, query)| (heading, query.trim().to_owned()))
        .collect())
}

/// Scores each refresh sample whose heading `queries` names against that
/// query, keeping its candidates' order and everything the refresh
/// recorded of them but the logit.
fn rescore_refresh(
    material: &mut Material,
    queries: &BTreeMap<String, String>,
    reranker: &dyn Reranker,
) -> anyhow::Result<()> {
    for sample in &mut material.refresh {
        let Some(query) = queries.get(&sample.facet) else {
            continue;
        };
        if !sample.candidates.is_empty() {
            let documents: Vec<&str> = sample
                .candidates
                .iter()
                .map(|candidate| candidate.sentence.as_str())
                .collect();
            let logits = reranker
                .rerank(query, &documents)
                .with_context(|| format!("reranking sample {}", sample.sample))?;
            if logits.len() != documents.len() {
                bail!(
                    "the reranker scored {} of sample {}'s {} candidates",
                    logits.len(),
                    sample.sample,
                    documents.len()
                );
            }
            for (candidate, logit) in sample.candidates.iter_mut().zip(logits) {
                candidate.logit = Some(f64::from(logit));
            }
        }
        sample.rerank_query = query.clone();
        sample.reranked = true;
    }
    Ok(())
}

/// The query replay's prefetch would have reranked `turn` against.
fn rerank_query(turn: &Turn, mode: RerankQuery) -> String {
    let message = clean_query(&turn.user);
    let previous = turn.previous_query.as_deref().map(clean_query);
    match mode {
        RerankQuery::Message => effective_query(&message, previous.as_deref()),
        RerankQuery::Conversation => conversation_query(
            &message,
            previous.as_deref(),
            turn.previous_reply.as_deref(),
        ),
    }
}

/// Scores each recall sample's candidates against its query and sorts them
/// by logit, highest first; the sort is stable, so ties keep their order.
fn rescore(
    material: &mut Material,
    queries: Vec<String>,
    reranker: &dyn Reranker,
) -> anyhow::Result<()> {
    for (sample, query) in material.recall.iter_mut().zip(queries) {
        if !sample.candidates.is_empty() {
            let documents: Vec<&str> = sample
                .candidates
                .iter()
                .map(|candidate| candidate.sentence.as_str())
                .collect();
            let logits = reranker
                .rerank(&query, &documents)
                .with_context(|| format!("reranking sample {}", sample.sample))?;
            if logits.len() != documents.len() {
                bail!(
                    "the reranker scored {} of sample {}'s {} candidates",
                    logits.len(),
                    sample.sample,
                    documents.len()
                );
            }
            for (candidate, logit) in sample.candidates.iter_mut().zip(logits) {
                candidate.score = f64::from(logit);
            }
            sample
                .candidates
                .sort_by(|a, b| b.score.total_cmp(&a.score));
        }
        sample.rerank_query = Some(query);
    }
    Ok(())
}
