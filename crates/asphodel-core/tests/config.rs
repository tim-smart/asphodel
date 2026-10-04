//! Configuration validation, layering and deletion fingerprints. Every
//! setting is daemon-wide, a bank never overrides tuning, and the values that
//! decide an irreversible deletion are fingerprinted.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use asphodel_core::config::{
    ConfigError, DeletionInputs, Deployment, Layer, ResolvedConfig, Secret, Tuning,
    deletion_fingerprint,
};
use asphodel_core::constants::{self, FixedConstants, Significance};
use jiff::civil::Time;

fn load(text: &str) -> Result<Tuning, ConfigError> {
    Tuning::from_toml(text)
}

fn invalid_keys(text: &str) -> Vec<String> {
    match load(text) {
        Err(ConfigError::Invalid(values)) => values.into_iter().map(|v| v.key).collect(),
        other => panic!("expected out-of-range values, got {other:?}"),
    }
}

fn assert_rejected(text: &str) {
    assert!(load(text).is_err(), "accepted:\n{text}");
}

fn layers(texts: &[&str]) -> Result<Tuning, ConfigError> {
    let layers: Vec<Layer<'_>> = texts
        .iter()
        .map(|text| Layer {
            origin: "layer",
            text,
        })
        .collect();
    Tuning::from_layers(&layers)
}

fn deployment(token: Option<&str>, llm_api_key: Option<&str>) -> Deployment {
    Deployment {
        listen: "127.0.0.1:7720".into(),
        data_dir: Some("/var/lib/asphodel".into()),
        config: Some("/etc/asphodel/tuning.toml".into()),
        allow_network_fs: false,
        model_dir: None,
        token: token.map(Secret::new),
        llm_api_key: llm_api_key.map(Secret::new),
    }
}

/// A temporary directory removed even when an assertion unwinds.
struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "asphodel-config-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn file(&self, name: &str, text: &str) -> PathBuf {
        let path = self.0.join(name);
        std::fs::write(&path, text).unwrap();
        path
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

// Fixed constants

#[test]
fn fixed_constants_are_not_tuning_keys() {
    // Calibrated strength constants are fixed in code: none of these is ever
    // in a config struct.
    for (section, key) in [
        ("strength", "s"),
        ("strength", "tau"),
        ("clock", "full_speed_window"),
        ("clock", "full_speed_window_hours"),
        ("recall", "tau"),
        ("recall", "fading_cutoff"),
        ("injection", "reranker_deadline"),
        ("injection", "reranker_deadline_ms"),
        ("purge", "n0"),
    ] {
        assert_rejected(&format!("[{section}]\n{key} = 1\n"));
    }
    let json = serde_json::to_value(Tuning::default()).unwrap();
    let text = json.to_string();
    for name in ["\"tau\"", "\"n0\"", "\"d_max\"", "reranker_deadline"] {
        assert!(!text.contains(name), "tuning serialises {name}");
    }
}

// Tuning: defaults and loading

#[test]
fn defaults_are_valid() {
    Tuning::default().validate().unwrap();
}

#[test]
fn undated_days_defaults_to_thirty_and_accepts_overrides() {
    let days =
        |tuning: &Tuning| serde_json::to_value(tuning).unwrap()["agenda"]["undated_days"].clone();
    assert_eq!(days(&Tuning::default()), 30);
    assert_eq!(days(&load("").unwrap()), 30);
    assert_eq!(days(&load("[agenda]\nundated_days = 10\n").unwrap()), 10);
    assert_eq!(days(&load("[agenda]\nundated_days = 1\n").unwrap()), 1);
}

#[test]
fn undated_days_must_be_at_least_one() {
    assert_eq!(
        invalid_keys("[agenda]\nundated_days = 0\n"),
        ["agenda.undated_days"]
    );
}

#[test]
fn an_agenda_update_budget_too_small_for_the_date_alone_is_rejected() {
    // The date-only update is about 18 tokens, so 1 can't hold it.
    for budget in [0, 1, 39] {
        assert_eq!(
            invalid_keys(&format!("[agenda]\nupdate_budget = {budget}\n")),
            ["agenda.update_budget"],
            "{budget}"
        );
    }
    for budget in [40, 200] {
        assert_eq!(
            load(&format!("[agenda]\nupdate_budget = {budget}\n"))
                .unwrap()
                .agenda
                .update_budget,
            budget
        );
    }
}

#[test]
fn a_refresh_recalls_up_to_six_facets_of_twenty_within_ninety_or_a_hundred_with_cited() {
    // Up to six facets of 20 candidates each, at most 90 after removing
    // duplicates, and room up to 100 for the memories the model cites.
    let defaults = serde_json::to_value(Tuning::default().mental_models).unwrap();
    for (key, value) in [
        ("max_facets", 6),
        ("facet_budget", 20),
        ("input_budget", 90),
        ("input_budget_with_cited", 100),
    ] {
        assert_eq!(defaults[key], value, "{key}");
    }
    let set = load("[mental_models]\nmax_facets = 4\nfacet_budget = 15\n").unwrap();
    let set = serde_json::to_value(set.mental_models).unwrap();
    assert_eq!(
        (&set["max_facets"], &set["facet_budget"]),
        (&4.into(), &15.into())
    );
    for key in ["max_facets", "facet_budget"] {
        assert_eq!(
            invalid_keys(&format!("[mental_models]\n{key} = 0\n")),
            [format!("mental_models.{key}")]
        );
    }
}

#[test]
fn no_file_and_an_empty_file_give_the_defaults() {
    assert_eq!(Tuning::load(None).unwrap(), Tuning::default());
    assert_eq!(load("").unwrap(), Tuning::default());
    assert_eq!(load("# nothing set\n").unwrap(), Tuning::default());
    let empty_sections = "[clock]\n[purge]\n[recall]\n[injection]\n[reconcile]\n[agenda]\n\
                          [mental_models]\n[sessions]\n[llm]\n[strength]\n\
                          [strength.significance]\n";
    assert_eq!(load(empty_sections).unwrap(), Tuning::default());
}

#[test]
fn a_full_file_sets_every_value() {
    let t = load(
        r#"
        [clock]
        quiet_rate = 0.25

        [purge]
        delta = 0.5
        source_horizon_days = 120

        [recall]
        strong_cutoff = 0.4

        [injection]
        cap = 6
        token_budget = 400
        reranker_floors = { "jina-reranker-v1-turbo-en:int8" = -2.5 }

        [reconcile]
        embedding_floors = { "bge-small-en-v1.5:int8" = 0.82 }

        [agenda]
        horizon_days = 10
        overdue_days = 45
        dated_lines = 12
        routines = 3
        undated_tasks = 2

        [mental_models]
        budget = 1000
        profile_max_tokens = 600
        trigger_level = "major"
        refresh_debounce_minutes = 10
        refresh_max_delay_minutes = 60
        sweep_time = "03:15"
        input_budget = 50
        input_budget_with_cited = 65

        [sessions]
        mapping_expiry_days = 14

        [ranking]
        w_s_inject = 0.8
        w_s_recall = 0.3
        phase_bonus = 1.5
        phase_penalty = 0.5
        relevance_scales = { "jina-reranker-v1-turbo-en:int8" = 1.0 }

        [llm]
        model = "some-model:q4_K_M"
        endpoint = "https://llm.example/v1"

        [strength.significance]
        trivial = 0.05
        minor = 0.2
        notable = 0.45
        major = 0.65
        critical = 0.85
        "#,
    )
    .unwrap();

    assert_eq!(t.clock.quiet_rate, 0.25);
    assert_eq!(t.purge.delta, Some(0.5));
    assert_eq!(t.purge.source_horizon_days, 120);
    assert_eq!(t.recall.strong_cutoff, 0.4);
    assert_eq!(t.injection.cap, 6);
    assert_eq!(t.injection.token_budget, 400);
    assert_eq!(
        t.injection.reranker_floors["jina-reranker-v1-turbo-en:int8"],
        -2.5
    );
    assert_eq!(t.reconcile.embedding_floors["bge-small-en-v1.5:int8"], 0.82);
    assert_eq!(t.agenda.horizon_days, 10);
    assert_eq!(t.agenda.overdue_days, 45);
    assert_eq!(t.agenda.dated_lines, 12);
    assert_eq!(t.agenda.routines, 3);
    assert_eq!(t.agenda.undated_tasks, 2);
    assert_eq!(t.mental_models.budget, 1000);
    assert_eq!(t.mental_models.profile_max_tokens, 600);
    assert_eq!(t.mental_models.trigger_level, Significance::Major);
    assert_eq!(t.mental_models.refresh_debounce_minutes, 10);
    assert_eq!(t.mental_models.refresh_max_delay_minutes, 60);
    assert_eq!(t.mental_models.sweep_time, Time::constant(3, 15, 0, 0));
    assert_eq!(t.mental_models.input_budget, 50);
    assert_eq!(t.mental_models.input_budget_with_cited, 65);
    assert_eq!(t.sessions.mapping_expiry_days, 14);
    assert_eq!(t.ranking.w_s_inject, 0.8);
    assert_eq!(t.ranking.w_s_recall, 0.3);
    assert_eq!(t.ranking.phase_bonus, 1.5);
    assert_eq!(t.ranking.phase_penalty, 0.5);
    assert_eq!(
        t.ranking.relevance_scales["jina-reranker-v1-turbo-en:int8"],
        1.0
    );
    assert_eq!(t.llm.model.as_deref(), Some("some-model:q4_K_M"));
    assert_eq!(t.llm.endpoint.as_deref(), Some("https://llm.example/v1"));
    assert_eq!(t.strength.significance.trivial, 0.05);
    assert_eq!(t.strength.significance.minor, 0.2);
    assert_eq!(t.strength.significance.notable, 0.45);
    assert_eq!(t.strength.significance.major, 0.65);
    assert_eq!(t.strength.significance.critical, 0.85);
}

#[test]
fn resolved_json_shows_a_null_delta_as_null() {
    // TOML spells a null δ "never"; JSON must still show null.
    let mut t = Tuning::default();
    t.purge.delta = None;
    let json = serde_json::to_value(&t).unwrap();
    assert!(json["purge"]["delta"].is_null());
    let json: serde_json::Value =
        serde_json::from_str(&serde_json::to_string(&t).unwrap()).unwrap();
    assert!(json["purge"]["delta"].is_null());
}

#[test]
fn load_reads_a_file_and_fails_on_a_missing_or_invalid_one() {
    let dir = TestDir::new();
    let path = dir.file("tuning.toml", "[clock]\nquiet_rate = 0.3\n");
    assert_eq!(Tuning::load(Some(&path)).unwrap().clock.quiet_rate, 0.3);

    assert!(matches!(
        Tuning::load(Some(&dir.0.join("missing.toml"))),
        Err(ConfigError::Read { .. })
    ));
    let unknown = dir.file("unknown.toml", "[clock]\nquiet_rat = 0.3\n");
    assert!(Tuning::load(Some(&unknown)).is_err());
    let range = dir.file("range.toml", "[injection]\ncap = 0\n");
    assert!(matches!(
        Tuning::load(Some(&range)),
        Err(ConfigError::Invalid(_))
    ));
}

#[test]
fn a_tuning_written_as_toml_loads_back_unchanged() {
    // The replay overrides file has exactly the shape of
    // `Tuning`, so a resolved tuning written out must load back as itself,
    // including a null δ, which TOML can't spell as null.
    let mut changed = load(
        "[injection.reranker_floors]\nm = -1.0\n[llm]\nmodel = \"m\"\n\
         endpoint = \"http://localhost:8080\"\n[mental_models]\nsweep_time = \"05:30\"\n",
    )
    .unwrap();
    changed.clock.quiet_rate = 0.2;
    changed.strength.significance.trivial = 0.05;
    let mut never = Tuning::default();
    never.purge.delta = None;
    for tuning in [Tuning::default(), changed, never] {
        let text = toml::to_string(&tuning).unwrap();
        assert_eq!(load(&text).unwrap(), tuning, "round trip of:\n{text}");
    }
}

// Tuning: rejection

#[test]
fn unknown_keys_are_rejected_in_every_section() {
    for section in [
        "clock",
        "purge",
        "recall",
        "injection",
        "reconcile",
        "agenda",
        "mental_models",
        "sessions",
        "ranking",
        "llm",
    ] {
        assert_rejected(&format!("[{section}]\nnot_a_setting = 1\n"));
    }
}

#[test]
fn secrets_bank_identity_and_deployment_settings_are_not_tuning_keys() {
    // Secrets come from the environment only, and bank settings,
    // deployment and tuning never share a key.
    for text in [
        "[llm]\napi_key = \"sk-123\"\n",
        "[llm]\nkey = \"sk-123\"\n",
        "token = \"t\"\n",
        "[clock]\ntimezone = \"Pacific/Auckland\"\n",
        "owner = \"Tim\"\n",
        "assistant = \"Hermes\"\n",
        "listen = \"127.0.0.1:7720\"\n",
        "data_dir = \"/data\"\n",
        "allow_network_fs = true\n",
        "model_dir = \"/models\"\n",
    ] {
        assert_rejected(text);
    }
}

#[test]
fn delta_must_be_finite_and_not_negative() {
    // δ is null or ≥ 0.
    assert_eq!(invalid_keys("[purge]\ndelta = -0.5\n"), ["purge.delta"]);
    assert_rejected("[purge]\ndelta = nan\n");
    assert_rejected("[purge]\ndelta = -inf\n");
    assert_eq!(
        load("[purge]\ndelta = 0.0\n").unwrap().purge.delta,
        Some(0.0)
    );
}

#[test]
fn quiet_rate_must_keep_bank_time_moving() {
    // Bank time slows down when a bank is quiet: it never stops
    // or runs backwards.
    for value in ["0.0", "-0.1", "nan"] {
        assert_rejected(&format!("[clock]\nquiet_rate = {value}\n"));
    }
    load("[clock]\nquiet_rate = 0.05\n").unwrap();
}

#[test]
fn llm_concurrency_defaults_to_ten() {
    assert_eq!(Tuning::default().llm.concurrency, 10);
    assert_eq!(load("").unwrap().llm.concurrency, 10);
    assert_eq!(load("[llm]\n").unwrap().llm.concurrency, 10);
}

#[test]
fn llm_concurrency_accepts_explicit_overrides_and_is_at_least_one() {
    let concurrency =
        |tuning: &Tuning| serde_json::to_value(tuning).unwrap()["llm"]["concurrency"].clone();
    assert_eq!(concurrency(&load("[llm]\nconcurrency = 1\n").unwrap()), 1);
    assert_eq!(concurrency(&load("[llm]\nconcurrency = 5\n").unwrap()), 5);
    for value in ["0", "-1", "1.5"] {
        assert_rejected(&format!("[llm]\nconcurrency = {value}\n"));
    }
}

#[test]
fn llm_language_is_unset_by_default_and_must_not_be_empty() {
    // Unset keeps the language inferred from the text; set, it's the
    // language every claim is written in.
    assert_eq!(Tuning::default().llm.language, None);
    let tuning = load("[llm]\nlanguage = \"English\"\n").unwrap();
    assert_eq!(tuning.llm.language.as_deref(), Some("English"));
    // `GET /v1/config` shows the resolved value.
    let config = ResolvedConfig::new(tuning, deployment(None, None));
    let json = serde_json::to_value(&config).unwrap();
    assert_eq!(json["tuning"]["llm"]["language"], "English");

    for value in ["\"\"", "\"  \"", "\"\\t\""] {
        assert_eq!(
            invalid_keys(&format!("[llm]\nlanguage = {value}\n")),
            ["llm.language"],
            "{value}"
        );
    }
}

#[test]
fn extraction_guidance_is_unset_by_default_and_must_not_be_empty() {
    let guidance =
        |tuning: &Tuning| serde_json::to_value(tuning).unwrap()["extraction"]["guidance"].clone();
    // Unset, call 1's prompt is the fixed one.
    assert!(guidance(&Tuning::default()).is_null());
    // Set, it can run over several lines.
    let tuning =
        load("[extraction]\nguidance = \"\"\"\nSkip build logs.\nKeep release dates.\n\"\"\"\n")
            .unwrap();
    assert_eq!(guidance(&tuning), "Skip build logs.\nKeep release dates.\n");
    // `GET /v1/config` shows the resolved value.
    let config = ResolvedConfig::new(tuning, deployment(None, None));
    let json = serde_json::to_value(&config).unwrap();
    assert_eq!(
        json["tuning"]["extraction"]["guidance"],
        "Skip build logs.\nKeep release dates.\n"
    );
    // The overrides file replaces it, as replay's A/B runs need.
    let layered = layers(&[
        "[extraction]\nguidance = \"Skip build logs.\"\n",
        "[extraction]\nguidance = \"Keep release dates.\"\n",
    ])
    .unwrap();
    assert_eq!(guidance(&layered), "Keep release dates.");

    for value in ["\"\"", "\"  \"", "\"\\n\\t\""] {
        assert_eq!(
            invalid_keys(&format!("[extraction]\nguidance = {value}\n")),
            ["extraction.guidance"],
            "{value}"
        );
    }
    assert_rejected("[extraction]\nprompt = \"Skip build logs.\"\n");
}

#[test]
fn the_rerank_query_is_the_conversation_by_default_and_can_be_the_message() {
    let rerank_query = |tuning: &Tuning| {
        serde_json::to_value(tuning).unwrap()["injection"]["rerank_query"].clone()
    };
    // Unset, the reranker scores against the conversation.
    assert_eq!(rerank_query(&Tuning::default()), "conversation");
    assert_eq!(rerank_query(&layers(&[""]).unwrap()), "conversation");
    // The overrides file switches it, as replay's A/B runs need.
    let layered = layers(&["", "[injection]\nrerank_query = \"message\"\n"]).unwrap();
    assert_eq!(rerank_query(&layered), "message");
    assert_rejected("[injection]\nrerank_query = \"thread\"\n");
}

#[test]
fn embedding_floors_must_be_cosines() {
    // floors inside the model's score range.
    for value in ["1.5", "-1.01", "nan", "inf"] {
        assert_rejected(&format!("[reconcile.embedding_floors]\nm = {value}\n"));
    }
    for value in ["-1.0", "0.0", "0.85", "1.0"] {
        load(&format!("[reconcile.embedding_floors]\nm = {value}\n")).unwrap();
    }
}

#[test]
fn relevance_scales_must_be_positive_and_finite() {
    // The scale divides the reranker logit into relevance, so zero, a
    // negative or a non-finite one is rejected.
    assert!(Tuning::default().ranking.relevance_scales.is_empty());
    for value in ["0.0", "-0.0", "-1.0", "nan", "inf", "-inf"] {
        assert_eq!(
            invalid_keys(&format!("[ranking.relevance_scales]\nm = {value}\n")),
            ["ranking.relevance_scales.\"m\""],
            "{value}"
        );
    }
    for value in ["0.25", "1.0", "3.2"] {
        let t = load(&format!("[ranking.relevance_scales]\nm = {value}\n")).unwrap();
        assert_eq!(
            t.ranking.relevance_scales["m"],
            value.parse::<f64>().unwrap()
        );
    }
    assert_eq!(
        invalid_keys("[ranking.relevance_scales]\n\"\" = 1.0\n"),
        ["ranking.relevance_scales"]
    );
}

#[test]
fn strong_cutoff_must_sit_above_the_faded_boundary() {
    // The faded/fading boundary is τ by definition, and only the
    // strong cut-off is tunable, so it can't fall to or below τ.
    assert_rejected(&format!("[recall]\nstrong_cutoff = {}\n", constants::TAU));
    assert_rejected("[recall]\nstrong_cutoff = -2.0\n");
}

// Strength: significance values

#[test]
fn significance_values_default_to_the_calibrated_levels() {
    let significance = Tuning::default().strength.significance;
    assert_eq!(significance.trivial, 0.0);
    assert_eq!(significance.minor, 0.2);
    assert_eq!(significance.notable, 0.5);
    assert_eq!(significance.major, 0.7);
    assert_eq!(significance.critical, 0.9);
}

#[test]
fn a_significance_override_sets_only_the_levels_it_names() {
    let t = load("[strength.significance]\ntrivial = 0.02\nminor = 0.25\n").unwrap();
    let significance = t.strength.significance;
    assert_eq!(significance.trivial, 0.02);
    assert_eq!(significance.minor, 0.25);
    assert_eq!(significance.notable, 0.5);
    assert_eq!(significance.major, 0.7);
    assert_eq!(significance.critical, 0.9);

    // The inline-table spelling is the same key.
    let inline = load("[strength]\nsignificance = { trivial = 0.1 }\n").unwrap();
    assert_eq!(inline.strength.significance.trivial, 0.1);
    assert_eq!(inline.strength.significance.minor, 0.2);
}

#[test]
fn significance_values_must_lie_between_0_and_1() {
    assert_eq!(
        invalid_keys("[strength.significance]\ntrivial = -0.1\n"),
        ["strength.significance.trivial"]
    );
    assert_eq!(
        invalid_keys("[strength.significance]\ncritical = 1.1\n"),
        ["strength.significance.critical"]
    );
    assert_rejected("[strength.significance]\nnotable = nan\n");
    assert_rejected("[strength.significance]\ncritical = inf\n");
    load("[strength.significance]\ntrivial = 0.0\n").unwrap();
}

#[test]
fn significance_values_must_increase_with_the_level() {
    // A higher level never counts for less than a lower one: equal or
    // swapped values are both rejected.
    for text in [
        "[strength.significance]\nminor = 0.0\n",
        "[strength.significance]\nnotable = 0.2\n",
        "[strength.significance]\nmajor = 0.9\ncritical = 0.7\n",
    ] {
        let keys = invalid_keys(text);
        assert!(
            keys.iter().all(|k| k.starts_with("strength.significance.")),
            "{keys:?} for:\n{text}"
        );
    }
}

#[test]
fn kept_significance_is_not_a_tuning_key() {
    // A kept memory never fades, so its significance stays fixed in code.
    assert_rejected("[strength.significance]\nkept = 0.95\n");
    assert_rejected("[strength]\nsignificance_kept = 0.95\n");
    assert_rejected("[strength.significance]\ntrival = 0.05\n");
}

#[test]
fn every_invalid_value_is_reported_together() {
    let keys = invalid_keys(
        "[purge]\ndelta = -1.0\nsource_horizon_days = 0\n[injection]\ncap = 0\n\
         [reconcile.embedding_floors]\nm = 2.0\n",
    );
    for expected in ["purge.delta", "purge.source_horizon_days", "injection.cap"] {
        assert!(keys.iter().any(|k| k == expected), "{expected} in {keys:?}");
    }
    assert!(
        keys.iter()
            .any(|k| k.starts_with("reconcile.embedding_floors"))
    );
}

// Floors for the configured models

#[test]
fn a_missing_floor_for_a_configured_model_is_an_error() {
    // A missing floor for a configured model stops the daemon.
    let t = Tuning::default();
    assert!(
        t.check_floors("bge-small-en-v1.5:int8", "jina-reranker-v1-turbo-en:int8")
            .is_err()
    );

    let only_embedding =
        load("[reconcile.embedding_floors]\n\"bge-small-en-v1.5:int8\" = 0.8\n").unwrap();
    assert!(
        only_embedding
            .check_floors("bge-small-en-v1.5:int8", "jina-reranker-v1-turbo-en:int8")
            .is_err()
    );

    let only_reranker =
        load("[injection.reranker_floors]\n\"jina-reranker-v1-turbo-en:int8\" = -1.0\n").unwrap();
    assert!(
        only_reranker
            .check_floors("bge-small-en-v1.5:int8", "jina-reranker-v1-turbo-en:int8")
            .is_err()
    );
}

#[test]
fn only_a_floor_for_the_exact_model_string_counts() {
    // Int8 and fp32 give different scores, so there's no fallback.
    let t = load(
        "[reconcile.embedding_floors]\n\"bge-small-en-v1.5:int8\" = 0.8\n\
         [injection.reranker_floors]\n\"jina-reranker-v1-turbo-en:int8\" = -1.0\n\
         [ranking.relevance_scales]\n\"jina-reranker-v1-turbo-en:int8\" = 1.0\n",
    )
    .unwrap();
    t.check_floors("bge-small-en-v1.5:int8", "jina-reranker-v1-turbo-en:int8")
        .unwrap();
    assert!(
        t.check_floors("bge-small-en-v1.5:fp32", "jina-reranker-v1-turbo-en:int8")
            .is_err()
    );
    assert!(
        t.check_floors("bge-small-en-v1.5:int8", "jina-reranker-v1-turbo-en:fp32")
            .is_err()
    );
    assert!(
        t.check_floors("bge-small-en-v1.5", "jina-reranker-v1-turbo-en")
            .is_err()
    );
    assert!(
        t.check_floors("BGE-small-en-v1.5:int8", "jina-reranker-v1-turbo-en:int8")
            .is_err()
    );
}

// Layering for replay

const PRODUCTION: &str = "[clock]\nquiet_rate = 0.2\n[purge]\ndelta = 1.5\n\
                          [injection.reranker_floors]\nprod = -1.0\n";

#[test]
fn layers_apply_defaults_then_production_then_overrides() {
    // Replay layers code defaults, then the production file,
    // then the overrides.
    let t = layers(&[PRODUCTION, "[purge]\ndelta = 0.75\n"]).unwrap();
    assert_eq!(t.purge.delta, Some(0.75)); // override
    assert_eq!(t.clock.quiet_rate, 0.2); // production
    assert_eq!(t.purge.source_horizon_days, 90); // default
    assert_eq!(t.injection.reranker_floors["prod"], -1.0);
}

#[test]
fn an_override_can_switch_purge_off_and_on() {
    let off = layers(&[PRODUCTION, "[purge]\ndelta = \"never\"\n"]).unwrap();
    assert_eq!(off.purge.delta, None);
    let on = layers(&["[purge]\ndelta = \"never\"\n", "[purge]\ndelta = 2.0\n"]).unwrap();
    assert_eq!(on.purge.delta, Some(2.0));
}

#[test]
fn an_override_can_add_a_floor_for_another_model() {
    let t = layers(&[PRODUCTION, "[injection.reranker_floors]\nother = -3.0\n"]).unwrap();
    assert_eq!(t.injection.reranker_floors["prod"], -1.0);
    assert_eq!(t.injection.reranker_floors["other"], -3.0);
}

#[test]
fn an_unknown_key_in_any_layer_is_rejected() {
    assert!(layers(&[PRODUCTION, "[purge]\ndleta = 0.5\n"]).is_err());
    assert!(layers(&["[clock]\nquiet_rat = 0.2\n", "[clock]\nquiet_rate = 0.3\n"]).is_err());
}

#[test]
fn an_override_can_lower_one_significance_level_over_production() {
    let production = "[strength.significance]\ntrivial = 0.1\nminor = 0.25\n";
    let t = layers(&[production, "[strength.significance]\ntrivial = 0.05\n"]).unwrap();
    assert_eq!(t.strength.significance.trivial, 0.05); // override
    assert_eq!(t.strength.significance.minor, 0.25); // production
    assert_eq!(t.strength.significance.notable, 0.5); // default
}

#[test]
fn significance_order_is_checked_on_the_layered_result() {
    // Production sets minor below trivial; the override lowers trivial
    // below minor, so the merged result is in order.
    let production = "[strength.significance]\ntrivial = 0.1\nminor = 0.08\n";
    assert!(layers(&[production]).is_err());
    let t = layers(&[production, "[strength.significance]\ntrivial = 0.04\n"]).unwrap();
    assert_eq!(t.strength.significance.trivial, 0.04);
    assert_eq!(t.strength.significance.minor, 0.08);
}

#[test]
fn layered_result_is_validated_as_a_whole() {
    assert!(matches!(
        layers(&[PRODUCTION, "[purge]\ndelta = -1.0\n"]),
        Err(ConfigError::Invalid(_))
    ));
    assert!(layers(&[PRODUCTION, "[injection]\ncap = 0\n"]).is_err());
    assert!(layers(&["[injection]\ncap = 0\n", "[injection]\ncap = 3\n"]).is_ok());
}

// Redaction and the resolved config

#[test]
fn the_resolved_config_redacts_the_token_and_the_llm_key() {
    let config = ResolvedConfig::new(
        Tuning::default(),
        deployment(Some("tok-abc123"), Some("sk-live-xyz789")),
    );
    let json = serde_json::to_string(&config).unwrap();
    let debug = format!("{config:?}");
    let pretty = serde_json::to_string_pretty(&config).unwrap();
    for text in [&json, &debug, &pretty] {
        assert!(!text.contains("tok-abc123"), "token leaked: {text}");
        assert!(!text.contains("sk-live-xyz789"), "key leaked: {text}");
        assert!(!text.contains("abc123"));
        assert!(!text.contains("xyz789"));
    }
    assert_eq!(
        config.deployment.token.as_ref().unwrap().expose(),
        "tok-abc123"
    );
}

#[test]
fn secret_from_env_reads_the_variable() {
    // A variable name unique to this test, so parallel tests can't race.
    const NAME: &str = "ASPHODEL_CONFIG_TEST_SECRET_FROM_ENV";
    // SAFETY: no other code reads or writes this variable.
    unsafe { std::env::set_var(NAME, "from-env") };
    assert_eq!(Secret::from_env(NAME).unwrap().expose(), "from-env");
    unsafe { std::env::set_var(NAME, "") };
    assert!(Secret::from_env(NAME).is_none());
    unsafe { std::env::remove_var(NAME) };
    assert!(Secret::from_env(NAME).is_none());
}

#[test]
fn the_resolved_config_carries_tuning_deployment_constants_and_purge_state() {
    // `GET /v1/config` returns the resolved config, the fixed
    // constants and the purge-pause state.
    let tuning = load("[clock]\nquiet_rate = 0.3\n").unwrap();
    let config = ResolvedConfig::new(tuning.clone(), deployment(Some("t"), None));
    assert_eq!(config.tuning, tuning);
    assert_eq!(config.constants, FixedConstants::current());
    assert_eq!(config.version, asphodel_core::VERSION);
    assert_eq!(config.deletion_fingerprint, tuning.deletion_fingerprint());

    let json = serde_json::to_value(&config).unwrap();
    for key in [
        "tuning",
        "deployment",
        "constants",
        "deletion_fingerprint",
        "purge",
    ] {
        assert!(!json[key].is_null(), "missing {key} in {json}");
    }
    assert_eq!(json["tuning"]["clock"]["quiet_rate"], 0.3);
    assert_eq!(json["constants"]["tau"], constants::TAU);
    assert_eq!(json["deployment"]["listen"], "127.0.0.1:7720");
    assert_eq!(
        json["deletion_fingerprint"],
        tuning.deletion_fingerprint().as_str()
    );
}

// The deletion fingerprint

fn base_inputs() -> DeletionInputs {
    DeletionInputs::new(&Tuning::default())
}

/// Each fingerprinted input, changed on its own.
fn single_input_changes() -> Vec<(&'static str, DeletionInputs)> {
    let base = base_inputs();
    let mut changes = Vec::new();
    macro_rules! change {
        ($name:literal, |$i:ident| $body:expr) => {{
            let mut $i = base.clone();
            $body;
            changes.push(($name, $i));
        }};
    }
    change!("s", |i| i.s = 2.4);
    change!("tau", |i| i.tau = -0.8);
    change!("a", |i| i.a = 0.36);
    change!("c", |i| i.c = 0.25);
    change!("d_max", |i| i.d_max = 1.5);
    change!("g", |i| i.g = 0.7);
    change!("n0", |i| i.n0 = 17.0);
    change!("min_access_age_days", |i| i.min_access_age_days = 0.02);
    change!("floor_spacing_days", |i| i.floor_spacing_days = 2.0);
    change!("weight_created", |i| i.weight_created = 1.1);
    change!("weight_used", |i| i.weight_used = 0.9);
    change!("weight_mentioned_again", |i| i.weight_mentioned_again =
        1.25);
    change!("weight_confirmed", |i| i.weight_confirmed = 2.5);
    change!("weight_window_close", |i| i.weight_window_close = 0.5);
    change!("full_speed_window_secs", |i| i.full_speed_window_secs =
        12 * 3600);
    for level in 0..5 {
        let mut i = base.clone();
        i.significance[level] += 0.05;
        changes.push(("significance", i));
    }
    change!("significance_kept", |i| i.significance_kept = 0.95);
    change!("quiet_rate", |i| i.quiet_rate = 0.2);
    change!("delta", |i| i.delta = Some(0.5));
    change!("delta null", |i| i.delta = None);
    change!("overdue_days", |i| i.overdue_days = 31);
    change!("source_horizon_days", |i| i.source_horizon_days = 91);
    changes
}

#[test]
fn the_fingerprint_is_printable_for_an_ack() {
    // `asphodel purge ack --hash <h>` quotes it.
    let fingerprint = Tuning::default().deletion_fingerprint();
    let text = fingerprint.to_string();
    assert!(!text.is_empty());
    assert_eq!(text, fingerprint.as_str());
    assert!(text.chars().all(|c| c.is_ascii_alphanumeric()), "{text}");
    assert_eq!(
        serde_json::to_value(&fingerprint).unwrap(),
        serde_json::json!(text)
    );
}

#[test]
fn every_fingerprinted_input_changes_the_fingerprint_on_its_own() {
    let base = deletion_fingerprint(&base_inputs());
    for (name, inputs) in single_input_changes() {
        assert_ne!(
            deletion_fingerprint(&inputs),
            base,
            "changing {name} left the fingerprint unchanged"
        );
    }
}

#[test]
fn swapping_two_equal_shaped_inputs_changes_the_fingerprint() {
    let mut a = base_inputs();
    a.overdue_days = 30;
    a.source_horizon_days = 90;
    let mut b = base_inputs();
    b.overdue_days = 90;
    b.source_horizon_days = 30;
    assert_ne!(deletion_fingerprint(&a), deletion_fingerprint(&b));
}

#[test]
fn null_delta_differs_from_every_number() {
    let mut never = base_inputs();
    never.delta = None;
    let never = deletion_fingerprint(&never);
    for value in [0.0, 1.0, f64::MAX] {
        let mut i = base_inputs();
        i.delta = Some(value);
        assert_ne!(deletion_fingerprint(&i), never, "δ = {value}");
    }
}

#[test]
fn each_fingerprinted_tuning_value_changes_the_fingerprint() {
    // The fingerprinted tuning values: `quiet_rate`, `delta`,
    // `agenda.overdue_days` and `purge.source_horizon_days`.
    let base = Tuning::default().deletion_fingerprint();
    for text in [
        "[clock]\nquiet_rate = 0.11\n",
        "[purge]\ndelta = 1.01\n",
        "[purge]\ndelta = \"never\"\n",
        "[purge]\nsource_horizon_days = 89\n",
        "[agenda]\noverdue_days = 29\n",
        "[strength.significance]\ntrivial = 0.05\n",
        "[strength.significance]\ncritical = 0.95\n",
    ] {
        assert_ne!(
            load(text).unwrap().deletion_fingerprint(),
            base,
            "fingerprint ignored:\n{text}"
        );
    }
}

#[test]
fn setting_a_fingerprinted_value_to_its_default_keeps_the_fingerprint() {
    let base = Tuning::default().deletion_fingerprint();
    let explicit = load(
        "[clock]\nquiet_rate = 0.1\n[purge]\ndelta = 1.0\nsource_horizon_days = 90\n\
         [agenda]\noverdue_days = 30\n\
         [strength.significance]\ntrivial = 0.0\nminor = 0.2\nnotable = 0.5\nmajor = 0.7\n\
         critical = 0.9\n",
    )
    .unwrap();
    assert_eq!(explicit.deletion_fingerprint(), base);
    // An integer δ means the same value as its float.
    assert_eq!(
        load("[purge]\ndelta = 1\n").unwrap().deletion_fingerprint(),
        base
    );
}

#[test]
fn a_significance_change_is_named_by_its_tuning_key() {
    // `purge plan` names what changed. Significance is tuning now, so it's
    // reported under its key rather than as a code constant.
    let stored = DeletionInputs::new(&Tuning::default());
    let tuned = DeletionInputs::new(&load("[strength.significance]\ntrivial = 0.05\n").unwrap());
    assert_eq!(tuned.changed_from(&stored), ["strength.significance"]);
}

#[test]
fn excluded_tuning_values_leave_the_fingerprint_alone() {
    // Only the fingerprinted values decide an irreversible deletion; a
    // change to anything else must not pause purge.
    let base = Tuning::default().deletion_fingerprint();
    for text in [
        "[recall]\nstrong_cutoff = 0.5\n",
        "[injection]\ncap = 3\n",
        "[injection]\ntoken_budget = 900\n",
        "[injection.reranker_floors]\nm = -1.0\n",
        "[reconcile.embedding_floors]\nm = 0.8\n",
        "[agenda]\nhorizon_days = 14\n",
        "[agenda]\nundated_days = 60\n",
        "[agenda]\ndated_lines = 20\n",
        "[agenda]\nroutines = 0\n",
        "[agenda]\nundated_tasks = 9\n",
        "[mental_models]\nbudget = 1200\n",
        "[mental_models]\nprofile_max_tokens = 300\n",
        "[mental_models]\ntrigger_level = \"major\"\n",
        "[mental_models]\nrefresh_debounce_minutes = 2\n",
        "[mental_models]\nrefresh_max_delay_minutes = 90\n",
        "[mental_models]\nsweep_time = \"02:00\"\n",
        "[mental_models]\ninput_budget = 40\n",
        "[mental_models]\ninput_budget_with_cited = 120\n",
        "[mental_models]\nmax_facets = 4\n",
        "[mental_models]\nfacet_budget = 10\n",
        "[sessions]\nmapping_expiry_days = 7\n",
        "[llm]\nmodel = \"other-model\"\n",
        "[llm]\nendpoint = \"https://other.example/v1\"\n",
        "[llm]\nconcurrency = 5\n",
    ] {
        assert_eq!(
            load(text).unwrap().deletion_fingerprint(),
            base,
            "fingerprint changed for:\n{text}"
        );
    }
}

// LLM base URLs: validate syntax without DNS or network access.

fn endpoint_toml(endpoint: &str) -> String {
    format!("[llm]\nendpoint = '{endpoint}'\n")
}

#[test]
fn llm_endpoint_rejects_anything_but_a_plain_http_base_url() {
    for endpoint in [
        // No host, or no usable port.
        "http://",
        "https://",
        "https://:8080",
        "http://localhost:65536",
        "http://localhost:port",
        "http://localhost:0",
        // Malformed hosts.
        "http://exa mple.com",
        "http://[::1",
        "http://[::g]/",
        // Spellings URL parsing would silently repair.
        "http:localhost:8080",
        "http:/localhost",
        "http:\\localhost",
        " https://llm.example/v1",
        "https://llm.example/v1 ",
        // Not http or https.
        "ftp://host",
        "file:///tmp/x",
        "localhost:8080",
        "//host/v1",
        "",
        // Credentials, queries and fragments can carry secrets.
        "https://user:pass@llm.example/v1",
        "https://llm.example/v1?key=abc",
        "https://llm.example/v1#x",
    ] {
        assert_eq!(
            invalid_keys(&endpoint_toml(endpoint)),
            ["llm.endpoint"],
            "accepted {endpoint:?}"
        );
    }
}

#[test]
fn llm_endpoint_accepts_base_urls_without_rewriting() {
    for endpoint in [
        "http://localhost:11434",
        "http://127.0.0.1:8080/v1",
        "http://[::1]:8080/v1",
        "https://llm.example/v1",
        "https://llm.example/v1/",
        "https://llm.example:443",
        "http://llm.internal.svc.cluster.local:8000/openai/v1",
    ] {
        let tuning = load(&endpoint_toml(endpoint)).unwrap();
        assert_eq!(tuning.llm.endpoint.as_deref(), Some(endpoint));
    }
}

#[test]
fn llm_endpoint_error_does_not_echo_credentials() {
    let password = "recognisable-endpoint-password-7913";
    let endpoint = format!("https://user:{password}@llm.example/v1");
    let error = load(&endpoint_toml(&endpoint)).unwrap_err();
    assert!(
        matches!(&error, ConfigError::Invalid(values) if values.iter().any(|value| value.key == "llm.endpoint"))
    );
    assert!(!error.to_string().contains(password));
    assert!(!error.to_string().contains(&endpoint));
}
