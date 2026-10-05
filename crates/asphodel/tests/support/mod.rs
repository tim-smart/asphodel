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

    /// A file outside the private directory.
    pub fn file(&self, name: &str, text: &str) -> PathBuf {
        let path = self.path(name);
        fs::write(&path, text).unwrap();
        path
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

    /// `value` as pretty JSON in a file under the private directory.
    pub fn private_json(&self, name: &str, value: &Value) -> PathBuf {
        self.private_file(name, &serde_json::to_string_pretty(value).unwrap())
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

/// Exit 2, `sentinel` in neither stream, and `words` in stderr.
pub fn assert_refused_without(output: &Output, sentinel: &str, words: &[&str]) {
    let (out, err) = (stdout(output), stderr(output));
    assert_eq!(output.status.code(), Some(2), "stderr: {err}");
    assert!(!out.contains(sentinel), "stdout echoes the input: {out}");
    assert!(!err.contains(sentinel), "stderr echoes the input: {err}");
    for word in words {
        assert!(err.contains(word), "stderr should name {word:?}: {err}");
    }
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
    import_paths(dir, state_db, &manifest, out, extra)
}

/// `asphodel import` with the given manifest and `state.db` paths, as
/// they are.
pub fn import_paths(
    dir: &TestDir,
    state_db: &Path,
    manifest: &Path,
    out: &Path,
    extra: &[&str],
) -> Output {
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

/// The history `build` writes to a fresh private `state.db`, imported to
/// `corpus/main.jsonl`; returns the corpus path.
pub fn import_history(dir: &TestDir, build: impl FnOnce(&Path) -> hermes::StateDb) -> PathBuf {
    let state_db = dir.private_path("state.db");
    drop(build(&state_db));
    let corpus = dir.private_path("corpus/main.jsonl");
    assert_ok(&import(dir, &state_db, &corpus));
    corpus
}

/// The private dir holding [`hermes::small_history`] imported to
/// `corpus/main.jsonl`; returns the corpus path.
pub fn imported_small_history(dir: &TestDir) -> PathBuf {
    import_history(dir, hermes::small_history)
}

/// Writes `steps` as a script for the live stand-in
/// (`ASPHODEL_LLM_SCRIPT`).
pub fn script_steps(dir: &TestDir, name: &str, steps: &[Value]) -> PathBuf {
    let path = dir.path(&format!("{name}.json"));
    fs::write(&path, serde_json::to_vec(steps).unwrap()).unwrap();
    path
}

/// A script answering every call, up to more than any test makes, with
/// `reply`.
pub fn script(dir: &TestDir, name: &str, reply: Value) -> PathBuf {
    script_steps(dir, name, &vec![json!({ "reply": reply }); 256])
}

/// A script for the live stand-in: every call 1 reply claims
/// [`hermes::HOME_SENTENCE`] quoting [`hermes::HOME_QUOTE`], so the claim
/// survives only in the turn that says it and the rest of the history
/// extracts nothing.
pub fn live_script(dir: &TestDir) -> PathBuf {
    let reply = json!({ "claims": [home_claim()], "used_injected_ids": [] });
    script(dir, "llm-script", reply)
}

/// The test manifest with one mental model of its own beside the "User
/// profile" every bank is seeded with, within the budget the profile
/// leaves.
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
    let manifest = model_manifest(HOME_QUESTION);
    assert_ok(&import_with(dir, &state_db, &corpus, &manifest, &[]));
    corpus
}

/// A sentence of a refresh's write, citing memory handles.
pub fn said(text: &str, cites: &[&str]) -> Value {
    json!({ "text": text, "cites": cites })
}

/// A refresh's write: the whole answer, one section joining `sentences`,
/// citing every handle they cite. With no sentence it has no text, which a
/// refresh refuses, so it writes nothing.
pub fn write_reply(sentences: &[Value]) -> Value {
    let texts: Vec<&str> = sentences
        .iter()
        .map(|sentence| sentence["text"].as_str().unwrap())
        .collect();
    let mut cites: Vec<&Value> = Vec::new();
    for cite in sentences
        .iter()
        .flat_map(|s| s["cites"].as_array().unwrap())
    {
        if !cites.contains(&cite) {
            cites.push(cite);
        }
    }
    let sections = if texts.is_empty() {
        json!([])
    } else {
        json!([{ "heading": "Home", "text": texts.join(" ") }])
    };
    json!({ "sections": sections, "cites": cites })
}

/// A script whose every step answers any call: call 1 with the home claim
/// (as [`live_script`]), and a refresh's write with one sentence citing
/// `m1`, the only memory the history makes.
pub fn universal_script(dir: &TestDir) -> PathBuf {
    let sentences = vec![said("Tim lives in Auckland.", &["m1"])];
    script_answering_everything(dir, "universal-script", vec![home_claim()], sentences)
}

/// The call 1 claim [`live_script`] makes, as JSON.
pub fn home_claim() -> Value {
    claim(hermes::HOME_SENTENCE, hermes::HOME_QUOTE, "fact")
}

/// A call 1 claim of `kind`, notable, with nothing else set.
pub fn claim(content: &str, quote: &str, kind: &str) -> Value {
    json!({
        "content": content, "kind": kind, "quote": quote, "significance": "notable",
        "remember_this": false, "changes_something": false, "valid_from": null,
        "valid_until": null, "window_confidence": "high", "until_event": null, "due_at": null,
        "volatility": null, "recurrence_text": null, "recurrence_rrule": null,
        "recurrence_start": null, "entities": []
    })
}

/// The reply of [`script_answering_everything`]: call 1 reads `claims`, a
/// refresh's plan reads `facets` (one recalling [`HOME_QUESTION`]) and its
/// write reads `sections` and `cites`, from `sentences`. With no sentence
/// the write is one citing `m1`, since a refresh refuses a write with
/// nothing in it and would try again every half hour; any refresh that
/// writes lists at least one memory. No reply type refuses the others'
/// fields, so the order calls come in doesn't matter.
pub fn reply_to_everything(claims: Vec<Value>, sentences: Vec<Value>) -> Value {
    let quiet = [said("Tim has a history here.", &["m1"])];
    let sentences = if sentences.is_empty() {
        &quiet[..]
    } else {
        &sentences[..]
    };
    let mut reply = write_reply(sentences);
    reply["claims"] = json!(claims);
    reply["used_injected_ids"] = json!([]);
    reply["facets"] = json!([{ "heading": "Home", "query": HOME_QUESTION }]);
    reply
}

/// [`live_script`] with every step taking `delay_ms` to answer, so a
/// `live` run measures that latency. The reply also reads as a plan and a
/// write, so the seeded profile's refreshes succeed.
pub fn delayed_script(dir: &TestDir, delay_ms: u64) -> PathBuf {
    let reply = reply_to_everything(vec![home_claim()], vec![]);
    let steps = vec![json!({ "reply": reply, "delay_ms": delay_ms }); 64];
    script_steps(dir, &format!("delayed-script-{delay_ms}"), &steps)
}

/// A script whose every step answers any call with
/// [`reply_to_everything`].
pub fn script_answering_everything(
    dir: &TestDir,
    name: &str,
    claims: Vec<Value>,
    sentences: Vec<Value>,
) -> PathBuf {
    script(dir, name, reply_to_everything(claims, sentences))
}

/// The cassette every real-history run here records to and reads from.
pub fn cassette_path(dir: &TestDir) -> PathBuf {
    dir.private_path("cassettes/main.jsonl")
}

/// The cassette's bytes.
pub fn cassette_bytes(dir: &TestDir) -> Vec<u8> {
    fs::read(cassette_path(dir)).expect("a cassette was recorded")
}

/// Replaces the cassette with `records`, one per line.
pub fn write_cassette(dir: &TestDir, records: &[Value]) {
    let text: String = records.iter().map(|record| format!("{record}\n")).collect();
    fs::write(cassette_path(dir), text).unwrap();
}

/// The cassette's records, in file order.
pub fn cassette_records(dir: &TestDir) -> Vec<Value> {
    String::from_utf8(cassette_bytes(dir))
        .unwrap()
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).expect("every cassette line is JSON"))
        .collect()
}

/// One `[[probe]]` table, `fields` its lines after `kind`.
pub fn probe(id: &str, at: &str, kind: &str, fields: &str) -> String {
    format!("\n[[probe]]\nid = \"{id}\"\nat = \"{at}\"\nkind = \"{kind}\"\n{fields}\n")
}

/// The probe field naming the home memory by its sentence.
pub const HOME_MEMORY: &str = "memory = \"lives in Auckland\"";

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

    /// Exit 0 with the report it wrote.
    pub fn ok(&self) -> Value {
        assert_ok(&self.output);
        self.report()
    }
}

/// `asphodel replay --corpus <corpus> --mode <mode> --cassette <private
/// cassette> --probes <private probes>` with the report to a fresh file
/// under `reports/`, plus `extra`. `script` makes the live stand-in
/// available; without it no LLM is.
pub fn replay_history(
    dir: &TestDir,
    corpus: &Path,
    mode: &str,
    probes: &str,
    script: Option<&Path>,
    extra: &[&str],
) -> Run {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let report = format!("run-{}", NEXT.fetch_add(1, Ordering::Relaxed));
    replay_history_to(dir, corpus, mode, probes, &report, script, extra)
}

/// [`replay_history`] with the report to `reports/<report>.json`.
pub fn replay_history_to(
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
        .args(["--mode", mode])
        .arg("--cassette")
        .arg(cassette_path(dir))
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
    let run = replay_history(dir, corpus, "live", PASSING_PROBES, Some(&script), &[]);
    assert_ok(&run.output);
    run
}

/// `toml` in an overrides file of its own under the private dir; returns
/// its path for `--overrides`.
pub fn overrides(dir: &TestDir, toml: &str) -> String {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let name = format!("overrides/{}.toml", NEXT.fetch_add(1, Ordering::Relaxed));
    dir.private_file(&name, toml).to_str().unwrap().to_owned()
}

/// What a run simulated, without what identifies the run: the mode
/// (`kind`), the flags it was invoked with, where its LLM replies came
/// from (`llm`) and the cassette hash it started from.
pub fn simulation(report: &Value) -> Value {
    let mut report = report.clone();
    let object = report.as_object_mut().expect("the report is an object");
    for key in ["kind", "flags", "llm", "cassette_hash"] {
        object.remove(key);
    }
    report
}

/// The probe `id` in a report.
pub fn probe_in<'a>(report: &'a Value, id: &str) -> &'a Value {
    report["probes"]
        .as_array()
        .expect("the report lists probes")
        .iter()
        .find(|probe| probe["id"] == id)
        .unwrap_or_else(|| panic!("no probe {id}: {report}"))
}

/// The JSON file at `path`.
pub fn read_json(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).expect("the file is written")).expect("it is JSON")
}
