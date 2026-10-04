//! Shared support for the real-history replay tests: a temporary private
//! directory, the binary as a process, and the synthetic Hermes history.
//! Not every test file uses every helper.
#![allow(dead_code)]

pub mod hermes;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{Value, json};

/// A temporary directory removed even when an assertion unwinds. It's
/// under the system temp dir, so never inside a git working tree.
pub struct TestDir(PathBuf);

impl TestDir {
    pub fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "asphodel-history-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    pub fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }

    /// The private directory (`ASPHODEL_REPLAY_DIR`).
    pub fn private(&self) -> PathBuf {
        let path = self.path("private");
        fs::create_dir_all(&path).unwrap();
        path
    }

    /// A path under the private directory, its parent created.
    pub fn private_path(&self, name: &str) -> PathBuf {
        let path = self.private().join(name);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        path
    }

    /// A file under the private directory.
    pub fn private_file(&self, name: &str, text: &str) -> PathBuf {
        let path = self.private_path(name);
        fs::write(&path, text).unwrap();
        path
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// `asphodel <args>` with the private dir set, on the fake models, with no
/// real models and no LLM unless the test adds one.
pub fn asphodel(dir: &TestDir) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_asphodel"));
    command
        .env("ASPHODEL_REPLAY_DIR", dir.private())
        .env("ASPHODEL_MODELS", "fake")
        .env_remove("ASPHODEL_MODEL_DIR")
        .env_remove("ASPHODEL_LLM_SCRIPT")
        .env_remove("ASPHODEL_LLM_API_KEY")
        .env_remove("ASPHODEL_CONFIG");
    command
}

pub fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

pub fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

/// Exit 0, or the stderr in the message.
pub fn assert_ok(output: &Output) {
    assert_eq!(
        output.status.code(),
        Some(0),
        "stdout: {}\nstderr: {}",
        stdout(output),
        stderr(output)
    );
}

/// Exit 2 with `word` in stderr.
pub fn assert_refused(output: &Output, word: &str) {
    assert_eq!(output.status.code(), Some(2), "stderr: {}", stderr(output));
    assert!(
        stderr(output).to_lowercase().contains(&word.to_lowercase()),
        "stderr should name {word:?}: {}",
        stderr(output)
    );
}

/// `asphodel import` of `state_db` with the test manifest, the corpus to
/// `out` under the private dir.
pub fn import(dir: &TestDir, state_db: &Path, out: &Path) -> Output {
    import_with(dir, state_db, out, hermes::MANIFEST, &[])
}

/// `asphodel import` with `manifest` as the manifest's text and `extra`
/// after the other flags.
pub fn import_with(
    dir: &TestDir,
    state_db: &Path,
    out: &Path,
    manifest: &str,
    extra: &[&str],
) -> Output {
    let manifest = dir.private_file("manifest.toml", manifest);
    asphodel(dir)
        .arg("import")
        .arg("--state-db")
        .arg(state_db)
        .arg("--manifest")
        .arg(manifest)
        .arg("--out")
        .arg(out)
        .args(extra)
        .output()
        .unwrap()
}

/// The private dir holding [`hermes::small_history`] imported to
/// `corpus/main.jsonl`; returns the corpus path.
pub fn imported_small_history(dir: &TestDir) -> PathBuf {
    let state_db = dir.private_path("state.db");
    hermes::small_history(&state_db);
    let corpus = dir.private_path("corpus/main.jsonl");
    assert_ok(&import(dir, &state_db, &corpus));
    corpus
}

/// A script for the live stand-in (`ASPHODEL_LLM_SCRIPT`): every call 1
/// reply claims [`hermes::HOME_SENTENCE`] quoting [`hermes::HOME_QUOTE`],
/// so the claim survives only in the turn that says it and the rest of the
/// history extracts nothing, and no call 2 or refresh is ever needed.
pub fn live_script(dir: &TestDir) -> PathBuf {
    let reply = json!({
        "claims": [{
            "content": hermes::HOME_SENTENCE,
            "kind": "fact",
            "quote": hermes::HOME_QUOTE,
            "significance": "notable",
            "remember_this": false,
            "changes_something": false,
            "valid_from": null,
            "valid_until": null,
            "window_confidence": "high",
            "until_event": null,
            "due_at": null,
            "volatility": null,
            "recurrence_text": null,
            "recurrence_rrule": null,
            "recurrence_start": null,
            "entities": []
        }],
        "used_injected_ids": []
    });
    let steps: Vec<Value> = (0..32).map(|_| json!({ "reply": reply })).collect();
    let path = dir.path("llm-script.json");
    fs::write(&path, serde_json::to_vec(&steps).unwrap()).unwrap();
    path
}

/// The test manifest with one mental model of its own beside the "User
/// profile" every bank is seeded with. The seeded profile takes 500 of the
/// default 800-token budget, so this one stays within the rest.
pub fn model_manifest(question: &str) -> String {
    format!(
        "{}\n[[model]]\nname = \"home\"\nquestion = \"{question}\"\nkinds = [\"fact\"]\nmax_tokens = 100\n",
        hermes::MANIFEST
    )
}

/// The question [`imported_with_a_model`] gives its model.
pub const HOME_QUESTION: &str = "Where does the user live?";

/// The private dir holding [`hermes::small_history`] imported with
/// [`model_manifest`] to `corpus/main.jsonl`; returns the corpus path.
pub fn imported_with_a_model(dir: &TestDir) -> PathBuf {
    let state_db = dir.private_path("state.db");
    hermes::small_history(&state_db);
    let corpus = dir.private_path("corpus/main.jsonl");
    assert_ok(&import_with(
        dir,
        &state_db,
        &corpus,
        &model_manifest(HOME_QUESTION),
        &[],
    ));
    corpus
}

/// A sentence of a refresh's write, citing memory handles.
pub fn said(text: &str, cites: &[&str]) -> Value {
    json!({ "text": text, "cites": cites })
}

/// A refresh's write: the whole summary, one section holding `sentences`.
pub fn write_reply(sentences: Vec<Value>) -> Value {
    json!({ "sections": [{ "heading": "Home", "sentences": sentences }] })
}

/// A refresh's plan: one facet recalling [`HOME_QUESTION`], so `home`
/// selects as one retrieval for its question would.
pub fn home_facets() -> Value {
    json!([{ "heading": "Home", "query": HOME_QUESTION }])
}

/// A script whose every step answers any call: call 1 reads `claims` and
/// `used_injected_ids` (the home claim, as [`live_script`]), a refresh's
/// plan reads `facets`, and its write reads `sections`, one sentence citing
/// `m1`, the only memory the history makes. No reply type refuses the
/// others' fields, so the order calls come in doesn't matter.
pub fn universal_script(dir: &TestDir) -> PathBuf {
    let reply = json!({
        "claims": [{
            "content": hermes::HOME_SENTENCE,
            "kind": "fact",
            "quote": hermes::HOME_QUOTE,
            "significance": "notable",
            "remember_this": false,
            "changes_something": false,
            "valid_from": null,
            "valid_until": null,
            "window_confidence": "high",
            "until_event": null,
            "due_at": null,
            "volatility": null,
            "recurrence_text": null,
            "recurrence_rrule": null,
            "recurrence_start": null,
            "entities": []
        }],
        "used_injected_ids": [],
        "facets": home_facets(),
        "sections": write_reply(vec![said("Tim lives in Auckland.", &["m1"])])["sections"]
    });
    let steps: Vec<Value> = (0..64).map(|_| json!({ "reply": reply })).collect();
    let path = dir.path("universal-script.json");
    fs::write(&path, serde_json::to_vec(&steps).unwrap()).unwrap();
    path
}

/// The call 1 claim [`live_script`] makes, as JSON.
pub fn home_claim() -> Value {
    claim(hermes::HOME_SENTENCE, hermes::HOME_QUOTE, "fact")
}

/// A call 1 claim of `kind`, notable, with nothing else set.
pub fn claim(content: &str, quote: &str, kind: &str) -> Value {
    json!({
        "content": content,
        "kind": kind,
        "quote": quote,
        "significance": "notable",
        "remember_this": false,
        "changes_something": false,
        "valid_from": null,
        "valid_until": null,
        "window_confidence": "high",
        "until_event": null,
        "due_at": null,
        "volatility": null,
        "recurrence_text": null,
        "recurrence_rrule": null,
        "recurrence_start": null,
        "entities": []
    })
}

/// [`live_script`] with every step taking `delay_ms` to answer, so a
/// `live` run measures that latency. The reply also reads as a plan and
/// as an empty write, so the seeded profile's refreshes succeed.
pub fn delayed_script(dir: &TestDir, delay_ms: u64) -> PathBuf {
    let reply = json!({
        "claims": [home_claim()],
        "used_injected_ids": [],
        "facets": home_facets(),
        "sections": []
    });
    let steps: Vec<Value> = (0..64)
        .map(|_| json!({ "reply": reply, "delay_ms": delay_ms }))
        .collect();
    let path = dir.path(&format!("delayed-script-{delay_ms}.json"));
    fs::write(&path, serde_json::to_vec(&steps).unwrap()).unwrap();
    path
}

/// A script whose every step answers any call with `claims`, the home
/// facet and a write of `sentences`, as [`universal_script`] does.
pub fn script_answering_everything(
    dir: &TestDir,
    name: &str,
    claims: Vec<Value>,
    sentences: Vec<Value>,
) -> PathBuf {
    let reply = json!({
        "claims": claims,
        "used_injected_ids": [],
        "facets": home_facets(),
        "sections": write_reply(sentences)["sections"]
    });
    let steps: Vec<Value> = (0..64).map(|_| json!({ "reply": reply })).collect();
    let path = dir.path(&format!("{name}.json"));
    fs::write(&path, serde_json::to_vec(&steps).unwrap()).unwrap();
    path
}

/// Replaces the cassette with `records`, one per line.
pub fn write_cassette(dir: &TestDir, records: &[Value]) {
    let mut text = String::new();
    for record in records {
        text.push_str(&serde_json::to_string(record).unwrap());
        text.push('\n');
    }
    fs::write(dir.private_path("cassettes/main.jsonl"), text).unwrap();
}

/// A script for the `judge_used` top-up: every step
/// judges that the reply relied on nothing.
pub fn judge_script(dir: &TestDir) -> PathBuf {
    let steps: Vec<Value> = (0..32)
        .map(|_| json!({ "reply": { "used": [] } }))
        .collect();
    let path = dir.path("judge-script.json");
    fs::write(&path, serde_json::to_vec(&steps).unwrap()).unwrap();
    path
}

/// The cassette's records, in file order.
pub fn cassette_records(dir: &TestDir) -> Vec<Value> {
    fs::read_to_string(dir.private_path("cassettes/main.jsonl"))
        .expect("a cassette was recorded")
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).expect("every cassette line is JSON"))
        .collect()
}

/// Real-history probes: opaque ids, memories matched
/// by a regex on the sentence. Every one passes on the small history.
pub const PASSING_PROBES: &str = r#"
[[probe]]
id = "p001"
at = "2026-01-05T12:00:00Z"
kind = "exists"
memory = "lives in Auckland"

[[probe]]
id = "p002"
at = "2026-01-08T12:00:00Z"
kind = "recall_finds"
memory = "lives in Auckland"
query = "Tim lives in Auckland"
"#;

/// The passing probes plus one that fails: the memory isn't absent.
pub const PROBES_WITH_A_FAILURE: &str = r#"
[[probe]]
id = "p001"
at = "2026-01-05T12:00:00Z"
kind = "exists"
memory = "lives in Auckland"

[[probe]]
id = "p002"
at = "2026-01-08T12:00:00Z"
kind = "recall_finds"
memory = "lives in Auckland"
query = "Tim lives in Auckland"

[[probe]]
id = "p003"
at = "2026-01-09T12:00:00Z"
kind = "absent"
memory = "lives in Auckland"
"#;

/// One real-history `asphodel replay` run.
pub struct Run {
    pub output: Output,
    pub report_path: PathBuf,
}

impl Run {
    pub fn report(&self) -> Value {
        let bytes = fs::read(&self.report_path).unwrap_or_else(|_| {
            panic!(
                "no report at {}: {}",
                self.report_path.display(),
                stderr(&self.output)
            )
        });
        serde_json::from_slice(&bytes).expect("the report is JSON")
    }

    pub fn report_bytes(&self) -> Vec<u8> {
        fs::read(&self.report_path).unwrap()
    }
}

/// `asphodel replay --corpus <corpus> --mode <mode> --cassette <private
/// cassette> --probes <private probes> --report <report>` plus `extra`.
/// `script` makes the live stand-in available; without it no LLM is.
pub fn replay_history(
    dir: &TestDir,
    corpus: &Path,
    mode: &str,
    probes: &str,
    report: &str,
    script: Option<&Path>,
    extra: &[&str],
) -> Run {
    let probes = dir.private_file("probes.toml", probes);
    let report_path = dir.private_path(&format!("reports/{report}.json"));
    let mut command = asphodel(dir);
    command
        .arg("replay")
        .arg("--corpus")
        .arg(corpus)
        .arg("--mode")
        .arg(mode)
        .arg("--cassette")
        .arg(dir.private_path("cassettes/main.jsonl"))
        .arg("--probes")
        .arg(probes)
        .arg("--report")
        .arg(&report_path)
        .args(extra);
    if let Some(script) = script {
        command.env("ASPHODEL_LLM_SCRIPT", script);
    }
    Run {
        output: command.output().unwrap(),
        report_path,
    }
}

/// A `live` run on the small history with the stand-in, which records the
/// cassette every later mode reads.
pub fn record(dir: &TestDir, corpus: &Path) -> Run {
    let script = live_script(dir);
    let run = replay_history(
        dir,
        corpus,
        "live",
        PASSING_PROBES,
        "live",
        Some(&script),
        &[],
    );
    assert_ok(&run.output);
    run
}
