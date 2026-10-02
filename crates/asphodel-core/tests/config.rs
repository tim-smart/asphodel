//! The configuration surface, checked against ADR 0008, ADR 0009 and the
//! resolutions they came from: "Strength model" (TIM-91), "Retrieval and
//! ranking" (TIM-93), "Mental models" (TIM-95, with ADR 0007), "Deletion
//! policy" (TIM-97) and "Configuration surface" (TIM-98).

use std::path::{Path, PathBuf};
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
    // ADR 0009: none of these is ever in a config struct.
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
fn no_file_and_an_empty_file_give_the_defaults() {
    assert_eq!(Tuning::load(None).unwrap(), Tuning::default());
    assert_eq!(load("").unwrap(), Tuning::default());
    assert_eq!(load("# nothing set\n").unwrap(), Tuning::default());
    let empty_sections = "[clock]\n[purge]\n[recall]\n[injection]\n[reconcile]\n[agenda]\n\
                          [mental_models]\n[sessions]\n[llm]\n";
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

        [llm]
        model = "some-model:q4_K_M"
        endpoint = "https://llm.example/v1"
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
    assert_eq!(t.llm.model.as_deref(), Some("some-model:q4_K_M"));
    assert_eq!(t.llm.endpoint.as_deref(), Some("https://llm.example/v1"));
}

#[test]
fn delta_can_be_null() {
    // ADR 0008: δ is nullable, and null means never purge. TOML has no
    // null, so this uses the file spelling the implementation chose.
    let t = load("[purge]\ndelta = \"never\"\n").unwrap();
    assert_eq!(t.purge.delta, None);

    let mut never = Tuning::default();
    never.purge.delta = None;
    never.validate().unwrap();
}

#[test]
fn resolved_json_shows_a_null_delta_as_null() {
    let mut t = Tuning::default();
    t.purge.delta = None;
    let json = serde_json::to_value(&t).unwrap();
    assert!(json["purge"]["delta"].is_null());
    let json: serde_json::Value =
        serde_json::from_str(&serde_json::to_string(&t).unwrap()).unwrap();
    assert!(json["purge"]["delta"].is_null());
    for delta in [0.0, 0.5, 1.0] {
        t.purge.delta = Some(delta);
        assert_eq!(serde_json::to_value(&t).unwrap()["purge"]["delta"], delta);
        let json: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&t).unwrap()).unwrap();
        assert_eq!(json["purge"]["delta"], delta);
    }
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
    // ADR 0009: the replay overrides file has exactly the shape of
    // `Tuning`, so a resolved tuning written out must load back as itself.
    let mut changed = load(
        "[injection.reranker_floors]\nm = -1.0\n[llm]\nmodel = \"m\"\n\
         endpoint = \"http://localhost:8080\"\n[mental_models]\nsweep_time = \"05:30\"\n",
    )
    .unwrap();
    changed.clock.quiet_rate = 0.2;
    for tuning in [Tuning::default(), changed] {
        let text = toml::to_string(&tuning).unwrap();
        assert_eq!(load(&text).unwrap(), tuning, "round trip of:\n{text}");
    }
}

#[test]
fn a_never_delta_written_as_toml_loads_back_as_never() {
    let mut tuning = Tuning::default();
    tuning.purge.delta = None;
    let text = toml::to_string(&tuning).expect("a null δ serialises to TOML");
    assert_eq!(
        load(&text).unwrap().purge.delta,
        None,
        "a null δ came back as a number from:\n{text}"
    );
}

#[test]
fn numeric_deltas_written_as_toml_load_back_as_numbers() {
    let mut tuning = Tuning::default();
    for delta in [0.0, 0.5, 1.0] {
        tuning.purge.delta = Some(delta);
        let text = toml::to_string(&tuning).unwrap();
        let value: toml::Value = toml::from_str(&text).unwrap();
        assert_eq!(value["purge"]["delta"].as_float(), Some(delta));
        assert_eq!(load(&text).unwrap(), tuning, "round trip of:\n{text}");
    }
}

// Tuning: rejection

#[test]
fn unknown_sections_are_rejected() {
    for section in ["strength", "ranking_typo", "bank", "deployment", "secrets"] {
        assert_rejected(&format!("[{section}]\n"));
    }
    assert_rejected("quiet_rate = 0.1\n");
}

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
    // ADR 0009: secrets come from the environment only, and bank settings,
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
fn wrong_types_are_rejected() {
    for text in [
        "[clock]\nquiet_rate = \"fast\"\n",
        "[purge]\ndelta = true\n",
        "[purge]\ndelta = \"sometimes\"\n",
        "[purge]\nsource_horizon_days = 1.5\n",
        "[injection]\ncap = -1\n",
        "[injection]\nreranker_floors = 1.0\n",
        "[injection.reranker_floors]\nm = \"high\"\n",
        "[mental_models]\ntrigger_level = \"huge\"\n",
        "[mental_models]\nsweep_time = \"25:00\"\n",
        "[clock]\nquiet_rate = [0.1]\n",
        "clock = 0.1\n",
    ] {
        assert_rejected(text);
    }
}

#[test]
fn malformed_toml_is_rejected() {
    assert!(matches!(
        load("[clock\nquiet_rate = 0.1\n"),
        Err(ConfigError::Parse { .. })
    ));
}

#[test]
fn delta_must_be_finite_and_not_negative() {
    // TIM-98: δ is null or ≥ 0.
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
    // Bank time slows down when a bank is quiet (ADR 0004): it never stops
    // or runs backwards.
    for value in ["0.0", "-0.1", "nan"] {
        assert_rejected(&format!("[clock]\nquiet_rate = {value}\n"));
    }
    load("[clock]\nquiet_rate = 0.05\n").unwrap();
}

#[test]
fn injection_cap_must_be_positive() {
    // TIM-98: cap > 0.
    assert_eq!(invalid_keys("[injection]\ncap = 0\n"), ["injection.cap"]);
    load("[injection]\ncap = 1\n").unwrap();
}

#[test]
fn in_context_idle_days_must_be_positive() {
    // TIM-93, amended by TIM-109: at least 1, like mapping_expiry_days.
    assert_eq!(
        invalid_keys("[sessions]\nin_context_idle_days = 0\n"),
        ["sessions.in_context_idle_days"]
    );
    let t = load("[sessions]\nin_context_idle_days = 1\n").unwrap();
    assert_eq!(t.sessions.in_context_idle_days, 1);
}

#[test]
fn ranking_constants_are_tuning_keys() {
    // TIM-98 places w_s for each mode and the phase bonus and penalty in
    // [ranking].
    let t = load(
        "[ranking]\nw_s_inject = 0.8\nw_s_recall = 0.3\nphase_bonus = 1.5\nphase_penalty = 0.5\n",
    )
    .unwrap();
    assert_eq!(t.ranking.w_s_inject, 0.8);
    assert_eq!(t.ranking.w_s_recall, 0.3);
    assert_eq!(t.ranking.phase_bonus, 1.5);
    assert_eq!(t.ranking.phase_penalty, 0.5);
}

#[test]
fn source_horizon_must_be_positive() {
    assert_rejected("[purge]\nsource_horizon_days = 0\n");
}

#[test]
fn embedding_floors_must_be_cosines() {
    // TIM-98: floors inside the model's score range.
    for value in ["1.5", "-1.01", "nan", "inf"] {
        assert_rejected(&format!("[reconcile.embedding_floors]\nm = {value}\n"));
    }
    for value in ["-1.0", "0.0", "0.85", "1.0"] {
        load(&format!("[reconcile.embedding_floors]\nm = {value}\n")).unwrap();
    }
}

#[test]
fn reranker_floors_must_be_finite() {
    for value in ["nan", "inf", "-inf"] {
        assert_rejected(&format!("[injection.reranker_floors]\nm = {value}\n"));
    }
    load("[injection.reranker_floors]\nm = -7.25\n").unwrap();
}

#[test]
fn strong_cutoff_must_sit_above_the_faded_boundary() {
    // ADR 0009: the faded/fading boundary is τ by definition, and only the
    // strong cut-off is tunable, so it can't fall to or below τ.
    assert_rejected(&format!("[recall]\nstrong_cutoff = {}\n", constants::TAU));
    assert_rejected("[recall]\nstrong_cutoff = -2.0\n");
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
    // ADR 0009: a missing floor for a configured model stops the daemon.
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
         [injection.reranker_floors]\n\"jina-reranker-v1-turbo-en:int8\" = -1.0\n",
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

#[test]
fn an_embedding_floor_does_not_stand_in_for_a_reranker_floor() {
    let t = load("[reconcile.embedding_floors]\nshared = 0.8\n").unwrap();
    assert!(t.check_floors("shared", "shared").is_err());
}

// Layering for replay

const PRODUCTION: &str = "[clock]\nquiet_rate = 0.2\n[purge]\ndelta = 1.5\n\
                          [injection.reranker_floors]\nprod = -1.0\n";

#[test]
fn layers_apply_defaults_then_production_then_overrides() {
    // ADR 0009: replay layers code defaults, then the production file,
    // then the overrides.
    let t = layers(&[PRODUCTION, "[purge]\ndelta = 0.75\n"]).unwrap();
    assert_eq!(t.purge.delta, Some(0.75)); // override
    assert_eq!(t.clock.quiet_rate, 0.2); // production
    assert_eq!(t.purge.source_horizon_days, 90); // default
    assert_eq!(t.injection.reranker_floors["prod"], -1.0);
}

#[test]
fn production_alone_is_run_a() {
    assert_eq!(layers(&[PRODUCTION]).unwrap(), load(PRODUCTION).unwrap());
    assert_eq!(layers(&[]).unwrap(), Tuning::default());
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
fn a_secret_never_shows_in_debug_or_json() {
    let secret = Secret::new("hunter2-token");
    assert!(!format!("{secret:?}").contains("hunter2"));
    assert!(!serde_json::to_string(&secret).unwrap().contains("hunter2"));
    assert_eq!(secret.expose(), "hunter2-token");
}

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
fn redaction_holds_for_secrets_that_look_like_other_values() {
    for value in ["127.0.0.1:7720", "true", "null", "{\"a\":1}", "[redacted]x"] {
        let config = ResolvedConfig::new(Tuning::default(), deployment(Some(value), Some(value)));
        let json = serde_json::to_value(&config).unwrap();
        assert_ne!(json["deployment"]["token"], serde_json::json!(value));
        assert_ne!(json["deployment"]["llm_api_key"], serde_json::json!(value));
    }
}

#[test]
fn absent_secrets_stay_absent() {
    let config = ResolvedConfig::new(Tuning::default(), deployment(None, None));
    let json = serde_json::to_value(&config).unwrap();
    assert!(json["deployment"]["token"].is_null());
    assert!(json["deployment"]["llm_api_key"].is_null());
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
    // ADR 0009: `GET /v1/config` returns the resolved config, the fixed
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
fn the_fingerprint_is_deterministic() {
    assert_eq!(
        deletion_fingerprint(&base_inputs()),
        deletion_fingerprint(&base_inputs())
    );
    assert_eq!(
        Tuning::default().deletion_fingerprint(),
        load("").unwrap().deletion_fingerprint()
    );
}

#[test]
fn the_fingerprint_is_printable_for_an_ack() {
    // ADR 0010: `asphodel purge ack --hash <h>` quotes it.
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
fn deletion_inputs_come_from_the_constants_and_the_tuning() {
    let tuning = load(
        "[clock]\nquiet_rate = 0.3\n[purge]\ndelta = 0.5\nsource_horizon_days = 60\n\
         [agenda]\noverdue_days = 21\n",
    )
    .unwrap();
    let i = DeletionInputs::new(&tuning);
    assert_eq!(i.s, constants::S);
    assert_eq!(i.tau, constants::TAU);
    assert_eq!(i.a, constants::A);
    assert_eq!(i.c, constants::C);
    assert_eq!(i.d_max, constants::D_MAX);
    assert_eq!(i.g, constants::G);
    assert_eq!(i.n0, constants::N0);
    assert_eq!(i.min_access_age_days, constants::MIN_ACCESS_AGE_DAYS);
    assert_eq!(i.floor_spacing_days, constants::FLOOR_SPACING_DAYS);
    assert_eq!(i.weight_created, constants::WEIGHT_CREATED);
    assert_eq!(i.weight_used, constants::WEIGHT_USED);
    assert_eq!(i.weight_mentioned_again, constants::WEIGHT_MENTIONED_AGAIN);
    assert_eq!(i.weight_confirmed, constants::WEIGHT_CONFIRMED);
    assert_eq!(i.weight_window_close, constants::WEIGHT_WINDOW_CLOSE);
    assert_eq!(
        i.full_speed_window_secs,
        constants::FULL_SPEED_WINDOW.as_secs()
    );
    assert_eq!(i.significance, Significance::ALL.map(Significance::value));
    assert_eq!(i.significance_kept, constants::SIGNIFICANCE_KEPT);
    assert_eq!(i.quiet_rate, 0.3);
    assert_eq!(i.delta, Some(0.5));
    assert_eq!(i.overdue_days, 21);
    assert_eq!(i.source_horizon_days, 60);
    assert_eq!(tuning.deletion_fingerprint(), deletion_fingerprint(&i));
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
fn different_single_changes_give_different_fingerprints() {
    let fingerprints: Vec<_> = single_input_changes()
        .into_iter()
        .map(|(name, inputs)| (name, deletion_fingerprint(&inputs)))
        .collect();
    for (i, (a_name, a)) in fingerprints.iter().enumerate() {
        for (b_name, b) in &fingerprints[i + 1..] {
            assert_ne!(a, b, "{a_name} and {b_name} collide");
        }
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

    let mut c = base_inputs();
    c.weight_mentioned_again = 2.0;
    c.weight_confirmed = 1.5;
    assert_ne!(
        deletion_fingerprint(&c),
        deletion_fingerprint(&base_inputs())
    );
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
    // ADR 0009: `quiet_rate`, `delta`, `agenda.overdue_days` and
    // `purge.source_horizon_days`.
    let base = Tuning::default().deletion_fingerprint();
    for text in [
        "[clock]\nquiet_rate = 0.11\n",
        "[purge]\ndelta = 1.01\n",
        "[purge]\ndelta = \"never\"\n",
        "[purge]\nsource_horizon_days = 89\n",
        "[agenda]\noverdue_days = 29\n",
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
         [agenda]\noverdue_days = 30\n",
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
fn excluded_tuning_values_leave_the_fingerprint_alone() {
    // Only the values ADR 0009 lists decide an irreversible deletion; a
    // change to anything else must not pause purge.
    let base = Tuning::default().deletion_fingerprint();
    for text in [
        "[recall]\nstrong_cutoff = 0.5\n",
        "[injection]\ncap = 3\n",
        "[injection]\ntoken_budget = 900\n",
        "[injection.reranker_floors]\nm = -1.0\n",
        "[reconcile.embedding_floors]\nm = 0.8\n",
        "[agenda]\nhorizon_days = 14\n",
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
        "[mental_models]\ninput_budget_with_cited = 80\n",
        "[sessions]\nmapping_expiry_days = 7\n",
        "[llm]\nmodel = \"other-model\"\n",
        "[llm]\nendpoint = \"https://other.example/v1\"\n",
    ] {
        assert_eq!(
            load(text).unwrap().deletion_fingerprint(),
            base,
            "fingerprint changed for:\n{text}"
        );
    }
}

#[test]
fn deployment_and_secrets_leave_the_fingerprint_alone() {
    let a = ResolvedConfig::new(Tuning::default(), deployment(Some("a"), Some("b")));
    let mut other = deployment(None, None);
    other.listen = "0.0.0.0:9000".into();
    other.data_dir = Some(Path::new("/elsewhere").into());
    other.allow_network_fs = true;
    other.model_dir = Some("/models".into());
    other.config = None;
    let b = ResolvedConfig::new(Tuning::default(), other);
    assert_eq!(a.deletion_fingerprint, b.deletion_fingerprint);
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
        "http://localhost:99999",
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
        "ws://host",
        "file:///tmp/x",
        "localhost:8080",
        "//host/v1",
        "",
        // Credentials, queries and fragments can carry secrets.
        "https://user:pass@llm.example/v1",
        "https://key@llm.example",
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
