//! Configuration validation, layering and deletion fingerprints. Every
//! setting is daemon-wide, a bank never overrides tuning, and the values that
//! decide an irreversible deletion are fingerprinted.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use asphodel_core::config::{ConfigError, Layer, PurgePause};
use asphodel_core::constants;
use asphodel_core::store::OpenOptions;
use asphodel_core::{Service, SimulatedClock, Store, Tuning};
use serde_json::{Value, json};

fn load(text: &str) -> Result<Tuning, ConfigError> {
    Tuning::from_toml(text)
}

fn invalid_keys(error: ConfigError) -> Vec<String> {
    match error {
        ConfigError::Invalid(values) => values.into_iter().map(|v| v.key).collect(),
        other => panic!("expected out-of-range values, got {other:?}"),
    }
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

/// Asserts every leaf of `expected` has the same value at its path in
/// `actual`.
fn assert_contains(actual: &Value, expected: &Value, path: &str) {
    match expected.as_object() {
        Some(map) => {
            for (key, value) in map {
                assert_contains(&actual[key], value, &format!("{path}.{key}"));
            }
        }
        None => assert_eq!(actual, expected, "{path}"),
    }
}

/// A temporary directory removed even when an assertion unwinds.
struct TestDir(PathBuf);

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn no_file_and_an_empty_file_give_the_defaults() {
    assert_eq!(Tuning::load(None).unwrap(), Tuning::default());
    let empty_sections = "[clock]\n[purge]\n[recall]\n[injection]\n[reconcile]\n[agenda]\n\
                          [mental_models]\n[sessions]\n[llm]\n[strength]\n\
                          [strength.significance]\n";
    for text in ["", "# nothing set\n", empty_sections] {
        assert_eq!(load(text).unwrap(), Tuning::default(), "{text}");
    }
}

#[test]
fn a_full_file_sets_every_value() {
    let text = r#"
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
        rerank_query = "message"
        reranker_floors = { "jina-reranker-v1-turbo-en:int8" = -2.5 }

        [reconcile]
        embedding_floors = { "bge-small-en-v1.5:int8" = 0.82 }

        [agenda]
        horizon_days = 10
        overdue_days = 45
        undated_days = 1
        dated_lines = 12
        routines = 3
        undated_tasks = 2
        update_budget = 150

        [mental_models]
        budget = 1000
        profile_max_tokens = 600
        trigger_level = "major"
        refresh_debounce_minutes = 10
        refresh_max_delay_minutes = 60
        sweep_time = "03:15"
        max_facets = 4
        facet_budget = 15
        input_budget = 50
        input_budget_with_cited = 65

        [sessions]
        mapping_expiry_days = 14

        [ranking]
        w_s_inject = 0.8
        w_s_recall = 0.3
        phase_bonus = 1.5
        phase_penalty = 0.5
        relevance_scales = { "jina-reranker-v1-turbo-en:int8" = 0.25 }

        [llm]
        model = "some-model:q4_K_M"
        endpoint = "https://llm.example/v1"
        concurrency = 1
        language = "English"

        [extraction]
        guidance = "Skip build logs.\nKeep release dates.\n"

        [strength.significance]
        trivial = 0.05
        minor = 0.2
        notable = 0.45
        major = 0.65
        critical = 0.85
        "#;
    let t = load(text).unwrap();

    // Every value reads back through the serialised form, the shape of the
    // replay overrides file. jiff writes the sweep time with seconds.
    let mut file = serde_json::to_value(toml::from_str::<toml::Table>(text).unwrap()).unwrap();
    file["mental_models"]["sweep_time"] = json!("03:15:00");
    assert_contains(&serde_json::to_value(&t).unwrap(), &file, "");

    // A partial table sets only the levels it names, in either spelling.
    let default = Tuning::default().strength.significance;
    for text in [
        "[strength.significance]\ntrivial = 0.02\n",
        "[strength]\nsignificance = { trivial = 0.02 }\n",
    ] {
        let significance = load(text).unwrap().strength.significance;
        assert_eq!(significance.trivial, 0.02, "{text}");
        assert_eq!(significance.minor, default.minor);
        assert_eq!(significance.critical, default.critical);
    }
}

#[test]
fn values_at_the_edge_of_their_range_load_unchanged() {
    for text in [
        "[purge]\ndelta = 0.0\n",
        "[clock]\nquiet_rate = 0.05\n",
        "[agenda]\nupdate_budget = 40\n",
        "[strength.significance]\ntrivial = 0.0\n",
        "[reconcile.embedding_floors]\nm = -1.0\n",
        "[reconcile.embedding_floors]\nm = 1.0\n",
    ] {
        load(text).unwrap_or_else(|e| panic!("{text}: {e}"));
    }
    // LLM base URLs are checked for syntax only, and kept as written.
    for endpoint in [
        "http://localhost:11434",
        "http://127.0.0.1:8080/v1",
        "http://[::1]:8080/v1",
        "https://llm.example/v1/",
        "https://llm.example:443",
        "http://llm.internal.svc.cluster.local:8000/openai/v1",
    ] {
        let tuning = load(&format!("[llm]\nendpoint = '{endpoint}'\n")).unwrap();
        assert_eq!(tuning.llm.endpoint.as_deref(), Some(endpoint));
    }
}

#[test]
fn a_resolved_tuning_writes_back_as_toml_and_json() {
    // The replay overrides file has exactly the shape of `Tuning`, so a
    // resolved tuning written out must load back as itself, including a
    // null δ, which TOML spells "never" and JSON still shows as null.
    let mut changed = load(
        "[injection.reranker_floors]\nm = -1.0\n[llm]\nmodel = \"m\"\n\
         endpoint = \"http://localhost:8080\"\n[mental_models]\nsweep_time = \"05:30\"\n",
    )
    .unwrap();
    changed.clock.quiet_rate = 0.2;
    changed.strength.significance.trivial = 0.05;
    let mut never = Tuning::default();
    never.purge.delta = None;
    for tuning in [Tuning::default(), changed, never.clone()] {
        let text = toml::to_string(&tuning).unwrap();
        assert_eq!(load(&text).unwrap(), tuning, "round trip of:\n{text}");
    }
    assert!(serde_json::to_value(&never).unwrap()["purge"]["delta"].is_null());
}

#[test]
fn unknown_keys_and_values_of_the_wrong_kind_are_rejected() {
    let mut texts: Vec<String> =
        "clock purge recall injection reconcile agenda mental_models sessions ranking llm"
            .split(' ')
            .map(|section| format!("[{section}]\nnot_a_setting = 1\n"))
            .collect();
    texts.extend(
        [
            // Calibrated strength constants are fixed in code.
            "[strength]\ns = 1\n",
            "[strength]\ntau = 1\n",
            "[clock]\nfull_speed_window = 1\n",
            "[clock]\nfull_speed_window_hours = 1\n",
            "[recall]\ntau = 1\n",
            "[recall]\nfading_cutoff = 1\n",
            "[injection]\nreranker_deadline = 1\n",
            "[injection]\nreranker_deadline_ms = 1\n",
            "[purge]\nn0 = 1\n",
            // A kept memory never fades, so its significance is fixed too.
            "[strength.significance]\nkept = 0.95\n",
            "[strength]\nsignificance_kept = 0.95\n",
            "[strength.significance]\ntrival = 0.05\n",
            // Secrets come from the environment only, and bank settings,
            // deployment and tuning never share a key.
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
            // Guidance adds to call 1's prompt; it can't replace it.
            "[extraction]\nprompt = \"Skip build logs.\"\n",
            "[injection]\nrerank_query = \"thread\"\n",
            "[llm]\nconcurrency = -1\n",
            "[llm]\nconcurrency = 1.5\n",
        ]
        .map(String::from),
    );
    for text in &texts {
        assert!(
            matches!(load(text), Err(ConfigError::Parse { .. })),
            "accepted:\n{text}"
        );
    }
}

#[test]
fn out_of_range_values_are_reported_together_by_key() {
    // Each value is written under its key's table, and only that key is
    // reported.
    for (key, value) in [
        ("agenda.undated_days", "0"),
        // The update must hold at least its date line.
        ("agenda.update_budget", "0"),
        ("agenda.update_budget", "39"),
        // δ is "never" or a finite number of at least 0.
        ("purge.delta", "-0.5"),
        ("purge.delta", "nan"),
        ("purge.delta", "-inf"),
        // Bank time slows down when a bank is quiet: it never stops or runs
        // backwards.
        ("clock.quiet_rate", "0.0"),
        ("clock.quiet_rate", "-0.1"),
        ("clock.quiet_rate", "nan"),
        ("llm.concurrency", "0"),
        ("mental_models.max_facets", "0"),
        ("mental_models.facet_budget", "0"),
        ("llm.language", "\"\""),
        ("llm.language", "\"\\t \""),
        ("extraction.guidance", "\"\""),
        ("extraction.guidance", "\"\\n\\t\""),
        // Floors are cosines.
        ("reconcile.embedding_floors.\"m\"", "1.5"),
        ("reconcile.embedding_floors.\"m\"", "nan"),
        // The scale divides the reranker logit into relevance.
        ("ranking.relevance_scales.\"m\"", "0.0"),
        ("ranking.relevance_scales.\"m\"", "-1.0"),
        ("ranking.relevance_scales.\"m\"", "inf"),
        // The faded/fading boundary is τ by definition, so the strong
        // cut-off can't fall to or below it.
        ("recall.strong_cutoff", &constants::TAU.to_string()),
        // Significance lies in [0, 1] and rises strictly with the level.
        ("strength.significance.trivial", "-0.1"),
        ("strength.significance.critical", "1.1"),
        ("strength.significance.notable", "nan"),
        ("strength.significance.minor", "0.0"),
    ] {
        let (table, name) = key.rsplit_once('.').unwrap();
        let text = format!("[{table}]\n{name} = {value}\n");
        assert_eq!(
            invalid_keys(load(&text).unwrap_err()),
            [key],
            "for:\n{text}"
        );
    }

    let reported = |text: &str| {
        let mut keys = invalid_keys(load(text).unwrap_err());
        keys.sort();
        keys
    };
    let unnamed = "[ranking.relevance_scales]\n\"\" = 1.0\n";
    assert_eq!(reported(unnamed), ["ranking.relevance_scales"]);
    let out_of_order = "[strength.significance]\nmajor = 0.9\ncritical = 0.7\n";
    assert_eq!(reported(out_of_order), ["strength.significance.critical"]);
    let several = "[purge]\ndelta = -1.0\nsource_horizon_days = 0\n[injection]\ncap = 0\n\
                   [reconcile.embedding_floors]\nm = 2.0\n";
    assert_eq!(
        reported(several),
        [
            "injection.cap",
            "purge.delta",
            "purge.source_horizon_days",
            "reconcile.embedding_floors.\"m\"",
        ]
    );
}

#[test]
fn llm_endpoint_must_be_a_plain_http_base_url_and_is_never_echoed() {
    for endpoint in [
        // No host, or no usable port.
        "http://",
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
        "https://user:recognisable-password@llm.example/v1",
        "https://llm.example/v1?key=recognisable-key",
        "https://llm.example/v1#x",
    ] {
        let error = load(&format!("[llm]\nendpoint = '{endpoint}'\n")).unwrap_err();
        let message = error.to_string();
        assert!(!message.contains("recognisable"), "{message}");
        assert_eq!(invalid_keys(error), ["llm.endpoint"], "{endpoint:?}");
    }
}

#[test]
fn a_configured_model_needs_floors_under_its_exact_id() {
    // A missing floor for a configured model stops the daemon. Int8 and fp32
    // give different scores, so there's no fallback to a similar id.
    let (embedder, reranker) = ("bge-small-en-v1.5:int8", "jina-reranker-v1-turbo-en:int8");
    let embedding = format!("[reconcile.embedding_floors]\n\"{embedder}\" = 0.8\n");
    let gate = format!("[injection.reranker_floors]\n\"{reranker}\" = -1.0\n");
    let scale = format!("[ranking.relevance_scales]\n\"{reranker}\" = 1.0\n");
    for partial in [
        String::new(),
        embedding.clone(),
        gate.clone(),
        format!("{embedding}{gate}"),
    ] {
        assert!(
            load(&partial)
                .unwrap()
                .check_floors(embedder, reranker)
                .is_err()
        );
    }

    let full = load(&format!("{embedding}{gate}{scale}")).unwrap();
    full.check_floors(embedder, reranker).unwrap();
    for (embedder, reranker) in [
        ("bge-small-en-v1.5:fp32", reranker),
        (embedder, "jina-reranker-v1-turbo-en:fp32"),
        ("bge-small-en-v1.5", "jina-reranker-v1-turbo-en"),
        ("BGE-small-en-v1.5:int8", reranker),
    ] {
        assert!(full.check_floors(embedder, reranker).is_err());
    }
}

#[test]
fn later_layers_win_key_by_key_and_the_merged_result_is_validated() {
    // Replay layers code defaults, then the production file, then the
    // overrides.
    let production = "[clock]\nquiet_rate = 0.2\n[purge]\ndelta = 1.5\n\
                      [injection.reranker_floors]\nprod = -1.0\n\
                      [strength.significance]\ntrivial = 0.1\nminor = 0.25\n";
    let t = layers(&[
        production,
        "[purge]\ndelta = 0.75\n[injection.reranker_floors]\nother = -3.0\n\
         [strength.significance]\ntrivial = 0.05\n",
    ])
    .unwrap();
    let default = Tuning::default();
    let expected = json!({
        "clock": {"quiet_rate": 0.2},
        "purge": {"delta": 0.75, "source_horizon_days": default.purge.source_horizon_days},
        "injection": {"reranker_floors": {"prod": -1.0, "other": -3.0}},
        "strength": {"significance": {
            "trivial": 0.05, "minor": 0.25, "notable": default.strength.significance.notable,
        }},
    });
    assert_contains(&serde_json::to_value(&t).unwrap(), &expected, "");

    // An override switches purge off and on, and replaces guidance and the
    // rerank query, as replay's A/B runs need.
    let delta = |texts: &[&str]| layers(texts).unwrap().purge.delta;
    assert_eq!(delta(&[production, "[purge]\ndelta = \"never\"\n"]), None);
    assert_eq!(
        delta(&["[purge]\ndelta = \"never\"\n", "[purge]\ndelta = 2.0\n"]),
        Some(2.0)
    );
    let json = serde_json::to_value(
        layers(&[
            "[extraction]\nguidance = \"Skip build logs.\"\n",
            "[extraction]\nguidance = \"Keep release dates.\"\n\
             [injection]\nrerank_query = \"message\"\n",
        ])
        .unwrap(),
    )
    .unwrap();
    assert_eq!(json["extraction"]["guidance"], "Keep release dates.");
    assert_eq!(json["injection"]["rerank_query"], "message");

    // Range checks run on the merged result, so a later layer can fix a
    // combination an earlier one left invalid, or break a valid one.
    let out_of_order = "[strength.significance]\ntrivial = 0.1\nminor = 0.08\n";
    assert!(layers(&[out_of_order]).is_err());
    let fixed = layers(&[out_of_order, "[strength.significance]\ntrivial = 0.04\n"]).unwrap();
    assert_eq!(fixed.strength.significance.minor, 0.08);
    assert!(layers(&["[injection]\ncap = 0\n", "[injection]\ncap = 3\n"]).is_ok());
    assert!(matches!(
        layers(&[production, "[purge]\ndelta = -1.0\n"]),
        Err(ConfigError::Invalid(_))
    ));

    // Every layer must be valid on its own shape.
    assert!(layers(&[production, "[purge]\ndleta = 0.5\n"]).is_err());
    assert!(layers(&["[clock]\nquiet_rat = 0.2\n", "[clock]\nquiet_rate = 0.3\n"]).is_err());
}

/// Starts a daemon's store on the tuning `from`, restarts it on `to`, and
/// returns the purge state and what `purge plan` says changed.
fn restart(from: &str, to: &str) -> (PurgePause, Vec<String>) {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    let dir =
        TestDir(std::env::temp_dir().join(format!("asphodel-config-{}-{n}", std::process::id())));
    std::fs::create_dir_all(&dir.0).unwrap();
    let clock = Arc::new(SimulatedClock::new("2026-03-01T12:00:00Z".parse().unwrap()));
    let open = |tuning: Tuning| {
        let store = Store::open(&dir.0, OpenOptions::default(), clock.clone()).unwrap();
        let pause = store
            .check_fingerprint(&tuning.deletion_fingerprint())
            .unwrap();
        Service::open(clock.clone(), store, tuning).with_purge_pause(pause)
    };
    drop(open(load(from).unwrap()));
    let tuning = load(to).unwrap();
    let service = open(tuning.clone());
    let plan = service.purge_plan().unwrap();
    assert_eq!(plan.current, tuning.deletion_fingerprint());
    (service.purge_pause(), plan.changed)
}

#[test]
fn only_a_fingerprinted_value_pauses_purge_and_the_plan_names_it() {
    for (text, changed) in [
        ("[clock]\nquiet_rate = 0.11\n", "clock.quiet_rate"),
        ("[purge]\ndelta = 1.01\n", "purge.delta"),
        ("[purge]\ndelta = \"never\"\n", "purge.delta"),
        (
            "[purge]\nsource_horizon_days = 89\n",
            "purge.source_horizon_days",
        ),
        ("[agenda]\noverdue_days = 29\n", "agenda.overdue_days"),
        (
            "[strength.significance]\ntrivial = 0.05\n",
            "strength.significance",
        ),
        (
            "[strength.significance]\ncritical = 0.95\n",
            "strength.significance",
        ),
    ] {
        let (pause, names) = restart("", text);
        assert!(matches!(pause, PurgePause::Paused { .. }), "{text}");
        assert_eq!(names, [changed], "{text}");
    }

    // A change to anything else, or a fingerprinted value set to what it
    // already was, must not pause purge.
    for text in [
        toml::to_string(&Tuning::default()).unwrap(),
        "[recall]\nstrong_cutoff = 0.5\n".into(),
        "[injection]\ncap = 3\ntoken_budget = 900\n\
         [injection.reranker_floors]\nm = -1.0\n"
            .into(),
        "[reconcile.embedding_floors]\nm = 0.8\n".into(),
        "[agenda]\nhorizon_days = 14\nundated_days = 60\ndated_lines = 20\nroutines = 0\n\
         undated_tasks = 9\n"
            .into(),
        "[mental_models]\nbudget = 1200\nprofile_max_tokens = 300\ntrigger_level = \"major\"\n\
         refresh_debounce_minutes = 2\nrefresh_max_delay_minutes = 90\n\
         sweep_time = \"02:00\"\ninput_budget = 40\ninput_budget_with_cited = 80\n\
         max_facets = 4\nfacet_budget = 10\n"
            .into(),
        "[sessions]\nmapping_expiry_days = 7\n".into(),
        "[llm]\nmodel = \"other-model\"\nendpoint = \"https://other.example/v1\"\n\
         concurrency = 5\n"
            .into(),
    ] {
        assert_eq!(restart("", &text), (PurgePause::Running, vec![]), "{text}");
    }
    // An integer δ means the same value as its float.
    assert_eq!(
        restart("[purge]\ndelta = 2.0\n", "[purge]\ndelta = 2\n"),
        (PurgePause::Running, vec![])
    );
}
