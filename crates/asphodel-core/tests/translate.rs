//! Translating a memory into `[llm] language` (`Service::translate_memory`).
//!
//! A translation is a new memory that supersedes the one named, the way a
//! refinement does, so the chain carries strength, accesses and provenance
//! over. The other two tests race a translation against the bank's other
//! writers. Every memory here is made by extracting a turn.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use asphodel_core::Service;
use asphodel_core::clock::SimulatedClock;
use asphodel_core::config::Tuning;
use asphodel_core::extraction::Call2Input;
use asphodel_core::ingest::Turn;
use asphodel_core::inspect::MemoryView;
use asphodel_core::models::{
    Embedder, FakeEmbedder, FakeEmbedderV2, FakeLlm, FakeReranker, LlmClient, ModelError, Models,
};
use asphodel_core::store::bank::BankIdentity;
use asphodel_core::store::{DB_FILE, OpenOptions, Store};
use asphodel_core::translate::{TranslateError, Translation};
use jiff::Timestamp;
use serde_json::{Value, json};
use uuid::Uuid;

// Fixtures

const START: &str = "2026-10-01T07:00:00Z";

/// When the clock starts, so memories can be said before [`START`].
const EARLIER: &str = "2026-09-01T00:00:00Z";
const TZ: &str = "Pacific/Auckland";
const BANK: &str = "main";
const MODEL: &str = "fake-llm";

/// What the turn a memory comes from says around it. The LLM is never
/// shown it.
const PASSAGE: &str = "Passage text the translation must never see.";

const RUSSIAN: &str = "Сэм пьёт чай каждое утро.";
const ENGLISH: &str = "Sam drinks tea every morning.";

fn at(text: &str) -> Timestamp {
    text.parse().unwrap()
}

/// A temporary directory removed even when an assertion unwinds.
struct TestDir(PathBuf);

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn turn(session: &str, message_at: Timestamp, user: &str) -> Turn {
    Turn {
        session_id: session.into(),
        message_at,
        timezone: Some(TZ.into()),
        user_text: user.into(),
        assistant_text: "Noted.".into(),
        author: None,
        platform: Some("cli".into()),
        recall_id: None,
        forget_requested: false,
    }
}

/// Call 1's claim of `content`, quoting all of it: a minor fact with no
/// times and no entities. The nullable fields it may leave out are left out.
fn claim(content: &str) -> Value {
    json!({
        "content": content,
        "kind": "fact",
        "quote": content,
        "significance": "minor",
        "remember_this": false,
        "changes_something": false,
        "window_confidence": "high",
        "entities": [],
    })
}

/// Call 2's reply giving the first claim `labels` against their memories.
fn call2(input: &Call2Input, labels: &[(Uuid, &str)]) -> Value {
    let labels: Vec<Value> = labels
        .iter()
        .map(|(memory, label)| {
            let neighbour = input.neighbours.iter().find(|n| n.memory == *memory);
            json!({"neighbour": neighbour.expect("a neighbour").handle, "label": label})
        })
        .collect();
    json!({"claims": [{"claim": input.claims[0].handle, "labels": labels}]})
}

/// The LLM's reply for a translation.
fn llm(sentence: &str) -> FakeLlm {
    FakeLlm::scripted(MODEL, vec![json!({ "sentence": sentence })])
}

/// A service on the fakes with one bank, [`BANK`]. Field order matters: the
/// service drops before the directory it lives in.
struct Harness {
    service: Service,
    clock: Arc<SimulatedClock>,
    dir: TestDir,
}

impl Harness {
    fn new(language: Option<&str>) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = TestDir(std::env::temp_dir().join(format!(
            "asphodel-translate-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        )));
        let clock = Arc::new(SimulatedClock::new(at(EARLIER)));
        let h = Self::open(dir, clock, language, Models::fake());
        let identity = BankIdentity {
            owner_name: Some("Tim".into()),
            assistant_name: Some("Hermes".into()),
            timezone: Some(TZ.into()),
            ..BankIdentity::default()
        };
        h.service.ensure_bank_with_models(BANK, &identity).unwrap();
        h
    }

    /// The same store under a restarted daemon translating into English and
    /// serving with `models`.
    fn restart(self, models: Models) -> Self {
        drop(self.service);
        Self::open(self.dir, self.clock, Some("English"), models)
    }

    fn open(dir: TestDir, clock: Arc<SimulatedClock>, lang: Option<&str>, models: Models) -> Self {
        let mut toml = format!(
            "[injection.reranker_floors]\n\"{}\" = 0.0\n\
             [ranking.relevance_scales]\n\"{0}\" = 1.0\n\
             [reconcile.embedding_floors]\n\"{}\" = 0.5\n\"{}\" = 0.5\n",
            FakeReranker::MODEL_ID,
            FakeEmbedder::MODEL_ID,
            FakeEmbedderV2::MODEL_ID,
        );
        if let Some(language) = lang {
            toml.push_str(&format!("[llm]\nlanguage = \"{language}\"\n"));
        }
        let store =
            Store::open(&dir.0.join("data"), OpenOptions::default(), clock.clone()).unwrap();
        let tuning = Tuning::from_toml(&toml).unwrap();
        let service = Service::with_models(clock.clone(), store, tuning, models).unwrap();
        Self {
            service,
            clock,
            dir,
        }
    }

    fn db(&self) -> PathBuf {
        self.dir.0.join("data").join(DB_FILE)
    }

    /// The owner says `user` at `message_at`, and the turn is extracted at
    /// once with call 1 finding `claims` and judging `used` used, and call 2
    /// giving the first claim `labels` against their memories. Returns the
    /// new memories.
    fn says(
        &self,
        message_at: &str,
        user: &str,
        claims: Vec<Value>,
        used: &[Uuid],
        labels: &[(Uuid, &str)],
    ) -> Vec<Uuid> {
        let service = &self.service;
        self.clock.set(at(message_at));
        let said = turn("fixtures", at(message_at), user);
        service.ingest_turn(BANK, &said).unwrap();
        let lease = service.claim_chunk(BANK).unwrap().expect("queued");
        let input = service.call1_input(&lease, used).unwrap();
        let handles: Vec<&str> = used
            .iter()
            .map(|id| {
                let memory = input.in_context.iter().find(|m| m.memory == *id);
                memory.expect("in context").handle.as_str()
            })
            .collect();
        let call1 = json!({"claims": claims, "used_injected_ids": handles});
        let mut replies = vec![call1.clone()];
        if let Some(input) = service.call2_input(&lease, &call1, used).unwrap() {
            replies.push(call2(&input, labels));
        }
        let llm = FakeLlm::scripted(MODEL, replies);
        service.extract_chunk(lease, &llm, used).unwrap().memories
    }

    /// A state said on 1 September inside [`PASSAGE`], with a window, a
    /// volatility and an entity linked by its first word, then used on 10
    /// September and confirmed on 20 September: three separate occasions.
    fn memory(&self, content: &str) -> Uuid {
        let surface = content.split_whitespace().next().unwrap();
        let mut state = claim(content);
        state["kind"] = json!("state");
        state["valid_from"] = json!({"at": "2026-09-01T00:00", "precision": "day"});
        state["window_confidence"] = json!("low");
        state["volatility"] = json!("months");
        state["entities"] = json!([
            {"entity": null, "new_name": "Sam", "new_kind": "person", "surface_form": surface}
        ]);
        let passage = format!("{PASSAGE} {content}");
        let memory = self.says(EARLIER, &passage, vec![state], &[], &[])[0];
        self.says("2026-09-10T00:00:00Z", "Thanks.", vec![], &[memory], &[]);
        let confirmed = [(memory, "confirmed")];
        let again = vec![claim(content)];
        self.says("2026-09-20T00:00:00Z", content, again, &[], &confirmed);
        self.clock.set(at(START));
        memory
    }

    /// The memories call 2 would be shown for a new claim of `sentence`.
    /// Its vector must be near theirs for call 2 to run at all.
    fn neighbours(&self, sentence: &str) -> Vec<Uuid> {
        let service = &self.service;
        let probe = turn("probe", at(START), sentence);
        service.ingest_turn(BANK, &probe).unwrap();
        let lease = service.claim_chunk(BANK).unwrap().expect("queued");
        let call1 = json!({"claims": [claim(sentence)], "used_injected_ids": []});
        match service.call2_input(&lease, &call1, &[]).unwrap() {
            Some(input) => input.neighbours.iter().map(|n| n.memory).collect(),
            None => Vec::new(),
        }
    }

    fn show(&self, memory: Uuid) -> MemoryView {
        self.service.show_memory(BANK, &memory.to_string()).unwrap()
    }

    fn translate(&self, memory: Uuid, llm: &dyn LlmClient) -> Result<Translation, TranslateError> {
        self.service
            .translate_memory(BANK, &memory.to_string(), llm)
    }
}

/// The new head of a translation.
fn translated(outcome: Result<Translation, TranslateError>) -> Uuid {
    match outcome {
        Ok(Translation::Translated { to, .. }) => to,
        other => panic!("expected a translation, got {other:?}"),
    }
}

// Supersession

#[test]
fn a_translation_supersedes_the_memory_and_keeps_its_strength_accesses_and_provenance() {
    let h = Harness::new(None);
    let original = h.memory(RUSSIAN);
    h.service
        .set_significance(BANK, &original.to_string(), Some("major"))
        .unwrap();
    let asked = llm(ENGLISH);

    // Without a target language there's nothing to translate into.
    let refused = h.translate(original, &asked);
    let unset = matches!(refused, Err(TranslateError::LanguageUnset));
    assert!(unset, "{refused:?}");
    assert!(asked.requests().is_empty());

    let h = h.restart(Models::fake());
    let before = h.show(original);
    let head = translated(h.translate(original, &asked));

    // The LLM is asked for the target language and shown the sentence, never
    // the passage it came from.
    let requests = asked.requests();
    assert_eq!(requests.len(), 1);
    let sent = format!("{}\n{}", requests[0].system, requests[0].user);
    assert!(sent.contains("English") && sent.contains(RUSSIAN), "{sent}");
    assert!(!sent.contains(PASSAGE), "{sent}");

    // A refinement: one chain, the original superseded but not retracted.
    let after = h.show(head);
    assert_eq!(after.sentence, ENGLISH);
    assert_eq!(after.chain.head, head);
    assert_eq!(after.chain.members.len(), 2);
    let old = after
        .chain
        .members
        .iter()
        .find(|member| member.id == original)
        .expect("the original is in the new head's chain");
    assert_eq!(old.superseded_by, Some(head));
    assert!(!old.retracted && !old.hidden);

    // Strength is unchanged at the same instant: the same significance, the
    // same accesses, and no new access for the translation itself.
    assert_eq!(after.strength, before.strength);
    assert_eq!(after.strength.occasions, 3);
    assert_eq!(after.accesses.len(), before.accesses.len());
    for (inherited, own) in after.accesses.iter().zip(&before.accesses) {
        assert_eq!(
            (&inherited.kind, inherited.at, inherited.turn),
            (&own.kind, own.at, own.turn)
        );
        assert_eq!(inherited.inherited_from, Some(original));
    }

    // Provenance and everything but the sentence carry over, and the new
    // sentence is what's embedded: a claim of it finds the head.
    assert_eq!(after.source, before.source);
    assert_eq!(after.observed_at, before.observed_at);
    assert_eq!(after.kind, before.kind);
    assert_eq!(after.window, before.window);
    assert_eq!(after.significance, before.significance);
    assert_eq!(after.entities, before.entities);
    assert!(!after.entities.is_empty());

    // Naming an id again writes nothing and asks the LLM nothing: the old
    // id is refused with its head, and the head is already in the language.
    let again = llm(ENGLISH);
    let repeated = h.translate(original, &again);
    assert!(
        matches!(repeated, Err(TranslateError::Superseded { head: named }) if named == head),
        "{repeated:?}"
    );
    assert_eq!(
        h.translate(head, &again).unwrap(),
        Translation::AlreadyInLanguage {
            memory: head,
            language: "English".into(),
        }
    );
    assert!(again.requests().is_empty());
    assert_eq!(h.show(head).chain.members.len(), 2);
    assert!(h.neighbours(ENGLISH).contains(&head));
}

// Concurrency with the bank's other writers

/// [`FakeEmbedderV2`] behind a gate, for a re-embed. Its first call is the
/// job's step: while it runs, a memory is inserted below the job's cursor,
/// so the step never reaches it and the swap has a tail to embed. The
/// second call is the swap's tail, embedded after the swap has read what's
/// unstaged and before its transaction: it says so on `paused` and waits on
/// `open` to be let go.
struct Gate {
    db: PathBuf,
    calls: AtomicUsize,
    paused: Sender<()>,
    open: Mutex<Receiver<()>>,
}

impl Embedder for Gate {
    fn model_id(&self) -> &str {
        FakeEmbedderV2.model_id()
    }

    fn dimensions(&self) -> usize {
        FakeEmbedderV2.dimensions()
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, ModelError> {
        match self.calls.fetch_add(1, Ordering::SeqCst) {
            0 => {
                rusqlite::Connection::open(&self.db)
                    .unwrap()
                    .execute(
                        "INSERT INTO memories (id, uuid, bank_id, content, kind, significance,
                                               chunk_id, source_start, source_end, observed_at,
                                               window_confidence, created_at, updated_at)
                         SELECT -1, ?1, bank_id, 'Sam walks to work.', kind, significance,
                                chunk_id, source_start, source_end, observed_at,
                                window_confidence, created_at, updated_at
                         FROM memories LIMIT 1",
                        [Uuid::from_u128(0x7a).to_string()],
                    )
                    .unwrap();
            }
            1 => {
                let _ = self.paused.send(());
                let _ = self.open.lock().unwrap().recv();
            }
            _ => {}
        }
        FakeEmbedderV2.embed(texts)
    }
}

#[test]
fn a_translation_during_a_reembed_swap_is_embedded_with_the_new_model() {
    // The swap embeds its tail outside the store lock, between reading what's
    // unstaged and its transaction. A translation committed in that gap is in
    // neither, so it has to wait for the swap and be embedded with the model
    // the bank has after it.
    let h = Harness::new(Some("English"));
    let original = h.memory(RUSSIAN);
    let (paused, on_pause) = mpsc::channel();
    let (open, on_open) = mpsc::channel();
    let gate = Gate {
        db: h.db(),
        calls: AtomicUsize::new(0),
        paused,
        open: Mutex::new(on_open),
    };
    let models = Models {
        embedder: Arc::new(gate),
        reranker: Arc::new(FakeReranker),
    };
    let mut h = h.restart(models);
    // The bank was recorded under the old model, which the re-embed replaces.
    let previous = Arc::new(FakeEmbedder);
    h.service = h.service.with_previous_embedder(previous).unwrap();
    h.service.start_reembed(BANK).unwrap();

    let (outcome, reembedded) = std::thread::scope(|scope| {
        let reembed = scope.spawn(|| h.service.run_reembed(BANK));
        on_pause
            .recv_timeout(Duration::from_secs(30))
            .expect("the swap reaches its tail");
        let translate = scope.spawn(|| h.translate(original, &llm(ENGLISH)));
        // Give the translation a moment to commit. One that has to wait for
        // the bank never does, so carry on either way.
        let deadline = Instant::now() + Duration::from_millis(500);
        while h.show(original).chain.head == original && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        open.send(()).unwrap();
        let reembedded = reembed.join().unwrap();
        (translate.join().unwrap(), reembedded)
    });

    reembedded.unwrap();
    let head = translated(outcome);
    assert!(
        h.neighbours(ENGLISH).contains(&head),
        "the translation keeps a vector from a model the bank no longer uses"
    );
}

#[test]
fn a_translation_cant_commit_while_an_extraction_is_prepared() {
    // Extraction plans a refinement of the memory, then a translation of the
    // same memory arrives before the plan commits. Committing it there would
    // leave the plan to overwrite its supersession, splitting the chain. It
    // waits for the bank instead, and once the wait runs out it gives up
    // without writing and stops holding back the queue.
    let h = Harness::new(Some("English"));
    let before_work = "Sam drinks tea every morning before work.";
    let original = h.says(EARLIER, before_work, vec![claim(before_work)], &[], &[])[0];
    h.clock.set(at(START));

    let green = "Sam drinks green tea every morning before work.";
    let said = turn("s1", at("2026-10-01T06:30:00Z"), green);
    h.service.ingest_turn(BANK, &said).unwrap();
    let call1 = json!({"claims": [claim(green)], "used_injected_ids": []});
    let claimed = h.service.next_extraction(BANK).unwrap();
    let claimed = claimed.expect("the turn is queued");
    let (lease, in_context) = (claimed.lease, &claimed.in_context);
    let input = h.service.call2_input(&lease, &call1, in_context).unwrap();
    let input = input.expect("the memory is a neighbour, so call 2 runs");
    let call2 = call2(&input, &[(original, "refines")]);
    let extractor = FakeLlm::scripted(MODEL, vec![call1, call2]);
    let prepared = h
        .service
        .prepare_extraction(lease, &extractor, in_context)
        .unwrap();

    let next = turn("s2", at("2026-10-01T06:31:00Z"), "Sam walks to work.");
    h.service.ingest_turn(BANK, &next).unwrap();

    let busy = h.translate(original, &llm(ENGLISH));
    assert!(matches!(busy, Err(TranslateError::Busy)), "{busy:?}");
    assert_eq!(h.show(original).chain.head, original, "it wrote nothing");

    let refinement = h.service.commit_extraction(prepared).unwrap().memories[0];
    assert_eq!(h.show(original).chain.head, refinement);
    assert!(
        h.service.next_extraction(BANK).unwrap().is_some(),
        "the queue hands out the next chunk"
    );
}
