//! Extraction: turning a chunk into claims, reconciling them with what's
//! stored, and committing the result.
//!
//! A leased chunk ([`crate::queue`]) goes through five steps:
//!
//! 1. **Input.** Code assembles what call 1 sees: the chunk's text, its
//!    context (up to [`CONTEXT_TURNS`] earlier turns of the session, or the
//!    text before a document chunk), the reference date and a calendar strip
//!    in the source's timezone, the speaker, the entity candidates found
//!    through the alias FTS, and the in-context memories ("Extraction:
//!    significance, validity windows and supersession", TIM-92).
//! 2. **Call 1.** One structured-output call ([`call1_request`]) returns the
//!    claims and the `used` verdicts.
//! 3. **Checks in code.** A claim whose quote isn't in the chunk is dropped.
//!    Times become instants at the start of their unit in the source's
//!    timezone, fields that don't belong to a kind are dropped, a named
//!    weekday that doesn't match the date lowers window confidence, an RRULE
//!    is kept only when it parses and recurs, and remember-this keeps a
//!    memory only from the owner's own message (TIM-92, TIM-94 decision 1).
//! 4. **Reconciliation.** Code finds each claim's nearest stored memories,
//!    and when one clears the floor or a claim signals a change, call 2
//!    ([`call2_request`]) labels the claims against them
//!    ([`reconcile`](self::reconcile), TIM-92, ADR 0005). Call 1's reply is
//!    saved on the chunk first, so a failed call 2 is retried from it.
//! 5. **Commit.** One transaction writes the new memories and their
//!    vectors, entity links, new entities and aliases as logged edits, the
//!    neighbours ended, retracted or refined and their logged edits, the
//!    `created`, `mentioned_again`, `confirmed` and `used` accesses, and
//!    marks the chunk extracted, dropping call 1's saved reply (TIM-90,
//!    ADR 0001, ADR 0008).
//!
//! A failure anywhere before the commit writes nothing but the queue's
//! count of the failed attempt and call 1's saved reply, so the chunk is
//! retried in place.

mod call2;
mod claims;
mod commit;
mod input;
mod prompt;
mod reconcile;

#[cfg(test)]
mod gap_tests;

/// A test's hook into [`commit_prepared`] at the last moment before the
/// commit takes the store for its writes: whatever it does to the store,
/// the commit must see (TIM-117 review, the lock gap). Test builds only.
#[cfg(test)]
pub(crate) mod gap {
    use std::cell::RefCell;

    use crate::store::Store;

    type Hook = Box<dyn FnOnce(&Store)>;

    thread_local! {
        static HOOK: RefCell<Option<Hook>> = const { RefCell::new(None) };
    }

    /// Runs `hook` once, at the next commit on this thread.
    pub(crate) fn set(hook: impl FnOnce(&Store) + 'static) {
        HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
    }

    pub(super) fn run(store: &Store) {
        if let Some(hook) = HOOK.with(|slot| slot.borrow_mut().take()) {
            hook(store);
        }
    }
}

use jiff::Timestamp;
use jiff::civil::Date;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use uuid::Uuid;

use crate::config::Tuning;
use crate::models::{Embedder, LlmClient, LlmError, ModelError};
use crate::queue::{self, ChunkError, Failure, Lease, Leases, QueueError, SourceKind};
use crate::store::{Store, StoreError, VectorIndex};
use crate::system_prompt::BlockEntry;

pub use call2::call2_request;
pub(crate) use input::{entities_named, phrase, survivor};
pub use prompt::call1_request;

/// Call 1's template name and version, which replay's cassette keys include
/// (TIM-96, decision 4).
pub const CALL1_TEMPLATE: &str = "extract_claims";
pub const CALL1_VERSION: u32 = 2;

/// Call 2's template name and version, which replay's cassette keys include
/// (TIM-96, decision 4).
pub const CALL2_TEMPLATE: &str = "reconcile_claims";
pub const CALL2_VERSION: u32 = 2;

/// Neighbours kept per claim after fusing vector search and BM25 (TIM-92,
/// "top 5 per claim, fused"). A flagged claim's entity-linked open tasks and
/// current states come on top.
pub const NEIGHBOURS_PER_CLAIM: usize = 5;

/// Neighbours shown for the whole chunk, at most (TIM-92, "capped at about
/// 40 per unit").
pub const NEIGHBOUR_CAP: usize = 40;

/// The edit log kinds reconciliation writes, each on the row of the memory
/// it changed (`edits.memory_id`), with ids and times in `details` and never
/// content (TIM-90, "every edit logged").
pub const EDIT_ENDED: &str = "memory_ended";
pub const EDIT_RETRACTED: &str = "memory_retracted";
pub const EDIT_REFINED: &str = "memory_refined";
pub const EDIT_SIGNIFICANCE_RAISED: &str = "significance_raised";
/// A remember-this on a neighbour sets the owner's significance to kept.
pub const EDIT_KEPT: &str = "memory_kept";
/// The memory that ended another was retracted or refined, so the ended
/// memory's `ended_by` and `valid_until` follow the successor (TIM-92,
/// "Reopening", as amended by TIM-108).
pub const EDIT_END_REPOINTED: &str = "end_repointed";
/// The memory that ended another was denied, so the ended memory is open
/// again: its `valid_until` and `ended_by` are cleared (TIM-92, "Reopening",
/// as amended by TIM-108).
pub const EDIT_END_CLEARED: &str = "end_cleared";

/// Earlier clean turns of the session given as context (TIM-92, "up to 3
/// previous clean turns").
pub const CONTEXT_TURNS: usize = 3;

/// The context turns' total size in characters. Over it, they're clipped
/// oldest first: characters come off the start of the oldest turn, and a turn
/// clipped to nothing is left out (TIM-92).
pub const CONTEXT_CHARS: usize = 6_000;

/// How much of the document before a chunk it gets as context, in characters
/// (TIM-92, "the last few hundred characters of the previous chunk"). It's
/// taken from the source text, so a chunk skipped as seen in an earlier
/// version of the document still provides it.
pub const PREVIOUS_CHUNK_CHARS: usize = 400;

/// Entity candidates found by alias search, at most (TIM-92, "capped at about
/// 30 per unit, ranked by how many memories link to them"). `user`,
/// `assistant` and the speaker come on top, since search never has to find
/// them.
pub const ENTITY_CANDIDATE_CAP: usize = 30;

/// Linked memory sentences shown with a candidate, strongest first.
pub const CANDIDATE_MEMORIES: usize = 3;

/// The calendar strip: the reference date and this many days either side
/// (TIM-92, "three weeks either side").
pub const CALENDAR_DAYS: i64 = 21;

/// Everything call 1 is given for one chunk. Handles are short ids local to
/// the call (`e1`, `e2`, … for candidates, `m1`, `m2`, … for in-context
/// memories and `n1`, `n2`, … for mental model entries), which the reply
/// refers back to; they're cheaper and harder to garble than UUIDs.
#[derive(Debug, Clone, PartialEq)]
pub struct Call1Input {
    pub chunk: Uuid,
    pub source_kind: SourceKind,
    /// The chunk's text: the unit claims are extracted from and quotes are
    /// checked against. For a turn, the user message,
    /// [`TURN_SEPARATOR`](crate::ingest::TURN_SEPARATOR) and the reply.
    pub text: String,
    /// Where the assistant's reply starts in `text`, in characters. `None`
    /// for a document.
    pub reply_start: Option<usize>,
    /// The source's `observed_at`.
    pub observed_at: Timestamp,
    /// The source's timezone.
    pub timezone: String,
    /// The local date relative times resolve against: the message time's
    /// date in the source's timezone, or a document's reference date. `None`
    /// for a document whose reference date isn't exact, whose relative dates
    /// aren't resolved (TIM-92).
    pub reference_date: Option<Date>,
    /// The reference date and [`CALENDAR_DAYS`] either side, in order. The
    /// prompt renders each as `YYYY-MM-DD Weekday`. Empty when
    /// `reference_date` is `None`.
    pub calendar: Vec<Date>,
    /// Whose words the user message is; `None` for a document. "I" and "me"
    /// resolve to them (TIM-94, decision 1).
    pub speaker: Option<SpeakerRef>,
    /// Context only, oldest first: never quoted from, never extracted from on
    /// its own.
    pub context: Vec<String>,
    /// `user`, `assistant` and the speaker first, then the entities whose
    /// aliases appear in the text or its context.
    pub candidates: Vec<Candidate>,
    /// The in-context memories call 1 judges `used` against. Always empty
    /// for a document, which has no reply to have used anything.
    pub in_context: Vec<InContextMemory>,
    /// The mental model entries of the block the turn's session held, each
    /// with the in-context memories it cites. A reply that relied on one is
    /// `used` on every memory it cites (TIM-95, decision 4). Always empty
    /// for a document.
    pub entries: Vec<InContextEntry>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerRef {
    /// The speaker's handle among the candidates.
    pub handle: String,
    pub entity: Uuid,
    pub name: String,
    /// Whether the speaker is the owner, the seeded `user`.
    pub owner: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    pub handle: String,
    pub entity: Uuid,
    pub name: String,
    pub kind: EntityKind,
    pub aliases: Vec<String>,
    /// Up to [`CANDIDATE_MEMORIES`] sentences of memories linked to the
    /// entity, strongest first, leaving out hidden and retracted ones.
    pub memories: Vec<String>,
}

/// An entity's kind, as the `entities.kind` column holds it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EntityKind {
    Person,
    Place,
    Organisation,
    Project,
    Thing,
}

impl EntityKind {
    pub const ALL: [EntityKind; 5] = [
        EntityKind::Person,
        EntityKind::Place,
        EntityKind::Organisation,
        EntityKind::Project,
        EntityKind::Thing,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            EntityKind::Person => "person",
            EntityKind::Place => "place",
            EntityKind::Organisation => "organisation",
            EntityKind::Project => "project",
            EntityKind::Thing => "thing",
        }
    }

    fn parse(text: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.as_str() == text)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct InContextMemory {
    pub handle: String,
    pub memory: Uuid,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct InContextEntry {
    pub handle: String,
    pub entry: Uuid,
    pub text: String,
    /// The handles of the in-context memories it cites.
    pub cites: Vec<String>,
}

/// What a claim does to a neighbour (TIM-92, CONTEXT.md "Reconciliation").
/// `Retracts` is a corrected version of the neighbour, such as a reschedule;
/// `Denies` says the neighbour didn't happen or isn't true. Both retract it,
/// and differ only in what happens to anything it had ended (the TIM-92
/// amendment from TIM-108).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Label {
    MentionedAgain,
    Confirmed,
    Refines,
    Retracts,
    Denies,
    Ends,
}

impl Label {
    pub const ALL: [Label; 6] = [
        Label::MentionedAgain,
        Label::Confirmed,
        Label::Refines,
        Label::Retracts,
        Label::Denies,
        Label::Ends,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Label::MentionedAgain => "mentioned_again",
            Label::Confirmed => "confirmed",
            Label::Refines => "refines",
            Label::Retracts => "retracts",
            Label::Denies => "denies",
            Label::Ends => "ends",
        }
    }
}

/// Everything call 2 is given for one chunk. Handles are short ids local to
/// the call, as in call 1: `c1`, `c2`, … for the claims in reply order, and
/// `n1`, `n2`, … for the neighbours.
///
/// The reply is `{"claims": [{"claim": "c1", "labels": [{"neighbour": "n1",
/// "label": "ends"}]}]}`. A claim left out, or given no labels, is new. A
/// label naming a handle that isn't in the input is ignored, as call 1
/// ignores an unknown entity handle.
#[derive(Debug, Clone, PartialEq)]
pub struct Call2Input {
    pub chunk: Uuid,
    /// The claims that passed call 1's checks, in reply order.
    pub claims: Vec<ReconcileClaim>,
    /// Every claim's neighbours, each once, at most [`NEIGHBOUR_CAP`].
    pub neighbours: Vec<NeighbourMemory>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReconcileClaim {
    pub handle: String,
    /// The claim's index in call 1's reply, as [`Dropped::claim`] counts it.
    pub claim: usize,
    pub content: String,
    /// The source's `observed_at`, which decides direction.
    pub observed_at: Timestamp,
    /// `changes_something` or `remember_this`: the claim gets the wider
    /// candidate set.
    pub flagged: bool,
    /// The handles of the neighbours found for this claim, best first.
    pub neighbours: Vec<String>,
}

/// A stored memory shown to call 2. Faded and ended memories are shown,
/// retracted ones aren't, and a hit on a superseded memory shows the head of
/// its chain instead (TIM-92).
#[derive(Debug, Clone, PartialEq)]
pub struct NeighbourMemory {
    pub handle: String,
    pub memory: Uuid,
    pub content: String,
    pub kind: crate::strength::Kind,
    pub observed_at: Timestamp,
    /// Whether another memory has already ended it. Code rejects a newer
    /// claim's labels on it.
    pub ended: bool,
}

/// What one successful extraction committed.
#[derive(Debug, Clone, PartialEq)]
pub struct Extracted {
    pub chunk: Uuid,
    /// The new memories, in claim order, without the dropped claims and
    /// without those reconciliation turned into accesses or nothing.
    pub memories: Vec<Uuid>,
    /// The in-context memories call 1 judged used, each once.
    pub used: Vec<Uuid>,
    /// Entities created for proposed new entities.
    pub entities_created: Vec<Uuid>,
    /// Claims code dropped, by their index in the reply.
    pub dropped: Vec<Dropped>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dropped {
    pub claim: usize,
    pub reason: DropReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DropReason {
    /// The quote is empty or isn't in the chunk's text. A passage that's
    /// only in the context doesn't count (TIM-92).
    QuoteNotFound,
    /// The content is empty after trimming.
    EmptyContent,
    /// A task quoted from the assistant's reply with neither a due date nor
    /// an until-event (TIM-92: the assistant's task has "a due date or an
    /// until-event beyond the current turn").
    AssistantTaskUndated,
}

/// Why an extraction didn't commit. No variant carries the prompt, the reply
/// or a claim (ADR 0010).
///
/// The chunk's `last_error_kind` names the cause: `llm_transport`,
/// `llm_timeout`, `llm_status` (with the status), `llm_no_content`,
/// `llm_not_json`, `llm_refused`, `llm_backend`, `invalid_reply`,
/// `embedding`, `search` or `commit`. A call 2 failure records the same
/// `llm_*` kinds as call 1.
#[derive(Debug, thiserror::Error)]
pub enum ExtractError {
    /// The LLM can't be used at all: not configured, conflicting settings,
    /// no valid login, or a usage limit. Not the chunk's fault, so nothing is
    /// counted and the chunk stays at the head of the queue.
    #[error("the extraction queue holds: {error}")]
    Held { error: LlmError },

    /// Call 1 failed, and the queue counted it.
    #[error("call 1 failed: {error}")]
    Call1 { error: LlmError, failure: Failure },

    /// The reply came back but doesn't fit the schema: a missing field, or a
    /// value outside an enum such as a significance above `critical`.
    #[error("call 1's reply is invalid: {reason}")]
    InvalidReply {
        reason: &'static str,
        failure: Failure,
    },

    /// Call 2 failed, and the queue counted it. Call 1's reply stays saved
    /// on the chunk, so the retry starts at call 2.
    #[error("call 2 failed: {error}")]
    Call2 { error: LlmError, failure: Failure },

    /// [`Service::call2_input`](crate::Service::call2_input) was given a
    /// call 1 reply that doesn't fit its schema. Nothing is counted.
    #[error("the call 1 reply given is invalid: {reason}")]
    Rejected { reason: &'static str },

    /// Embedding failed while previewing call 2's input. Nothing is counted.
    #[error("embedding the claims failed: {error}")]
    Model { error: ModelError },

    /// Searching the store for the claims' neighbours failed, and the queue
    /// counted it, so a fault that recurs reaches the retry cap rather than
    /// holding the bank's queue. Call 1's reply isn't saved yet, so the retry
    /// starts at call 1.
    #[error("searching for the claims' neighbours failed: {error}")]
    Search { error: StoreError, failure: Failure },

    /// Embedding the new memories failed, and the queue counted it.
    #[error("embedding the new memories failed: {error}")]
    Embedding { error: ModelError, failure: Failure },

    /// The service was built without models, so there's nothing to embed
    /// with. Nothing is counted.
    #[error("no models are loaded, so nothing can be extracted")]
    NoModels,

    /// The bank's recorded embedding model isn't loaded, so its claims
    /// can't be compared with its memories or given vectors that fit them
    /// (ADR 0010). Nothing is counted: the queue holds until a re-embed
    /// moves the bank to the daemon's model.
    #[error(
        "the bank records embedding model {model}, which this daemon doesn't carry; run `asphodel reembed --bank` to move it"
    )]
    ModelUnavailable { model: String },

    /// The commit transaction failed and was rolled back, and the queue
    /// counted it, so a fault that recurs reaches the retry cap rather than
    /// holding the bank's queue.
    #[error("committing the chunk failed: {error}")]
    Commit { error: StoreError, failure: Failure },

    #[error(transparent)]
    Queue(#[from] QueueError),

    #[error(transparent)]
    Store(#[from] StoreError),
}

impl ExtractError {
    /// What the queue did with the chunk, or `None` when the attempt wasn't
    /// counted.
    pub fn failure(&self) -> Option<Failure> {
        match self {
            ExtractError::Call1 { failure, .. }
            | ExtractError::Call2 { failure, .. }
            | ExtractError::Search { failure, .. }
            | ExtractError::InvalidReply { failure, .. }
            | ExtractError::Embedding { failure, .. }
            | ExtractError::Commit { failure, .. } => Some(*failure),
            ExtractError::Held { .. }
            | ExtractError::Rejected { .. }
            | ExtractError::Model { .. }
            | ExtractError::NoModels
            | ExtractError::ModelUnavailable { .. }
            | ExtractError::Queue(_)
            | ExtractError::Store(_) => None,
        }
    }
}

impl From<rusqlite::Error> for ExtractError {
    fn from(error: rusqlite::Error) -> Self {
        ExtractError::Store(StoreError::Sqlite(error))
    }
}

/// Call 1's input for the leased chunk. Reads only.
pub(crate) fn call1_input(
    store: &Store,
    leases: &Leases,
    tuning: &Tuning,
    lease: &Lease,
    in_context: &[uuid::Uuid],
) -> Result<Call1Input, ExtractError> {
    queue::check_held(leases, lease)?;
    let conn = store.connection();
    let (input, _) = input::assemble(&conn, tuning, store.now(), lease, in_context, &[])?;
    Ok(input)
}

/// Call 2's input for the leased chunk when call 1 replied `reply`, or
/// `None` when call 2 won't run. Reads only.
pub(crate) fn call2_input(
    store: &Store,
    leases: &Leases,
    tuning: &Tuning,
    embedder: &dyn Embedder,
    lease: &Lease,
    reply: &Value,
    in_context: &[Uuid],
) -> Result<Option<Call2Input>, ExtractError> {
    queue::check_held(leases, lease)?;
    let (input, unit) = {
        let conn = store.connection();
        input::assemble(&conn, tuning, store.now(), lease, in_context, &[])?
    };
    let checked =
        claims::check(reply, &input, &unit).map_err(|reason| ExtractError::Rejected { reason })?;
    let vectors =
        embed(store, embedder, &checked).map_err(|error| ExtractError::Model { error })?;
    let conn = store.connection();
    let search = reconcile::search(
        &conn,
        floor(tuning, embedder),
        &input,
        &unit,
        &checked,
        &vectors,
    )?;
    Ok(search.map(|search| search.input))
}

/// A leased chunk extracted up to its commit: call 1's checked claims,
/// their embeddings, the neighbours call 2 was shown with its labels, and
/// the plan. The lease is still held. [`commit_prepared`] writes it.
///
/// Production runs the two halves back to back. Replay runs the first
/// when the worker claims the chunk, so the LLM calls are measured, and
/// the second a latency later (TIM-96, decision 3).
pub struct Prepared {
    lease: Lease,
    input: Call1Input,
    unit: input::Unit,
    checked: claims::Checked,
    vectors: Vec<Vec<f32>>,
    search: Option<reconcile::Search>,
    labels: Vec<call2::ClaimLabels>,
    plan: reconcile::Plan,
}

impl Prepared {
    /// The lease the chunk is held under.
    pub fn lease(&self) -> &Lease {
        &self.lease
    }
}

/// Runs call 1 on the leased chunk, or takes its saved reply, reconciles
/// the claims with call 2 when they land near something stored, and commits
/// the result.
#[allow(clippy::too_many_arguments)]
pub(crate) fn extract(
    store: &Store,
    leases: &Leases,
    tuning: &Tuning,
    embedder: &dyn Embedder,
    lease: Lease,
    llm: &dyn LlmClient,
    in_context: &[Uuid],
    entries: &[BlockEntry],
) -> Result<Extracted, ExtractError> {
    let prepared = prepare(
        store, leases, tuning, embedder, lease, llm, in_context, entries,
    )?;
    commit_prepared(store, leases, prepared)
}

/// The first half of [`extract`]: everything up to the commit, the LLM
/// calls included.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare(
    store: &Store,
    leases: &Leases,
    tuning: &Tuning,
    embedder: &dyn Embedder,
    lease: Lease,
    llm: &dyn LlmClient,
    in_context: &[Uuid],
    entries: &[BlockEntry],
) -> Result<Prepared, ExtractError> {
    queue::check_held(leases, &lease)?;
    let (input, mut unit, saved) = {
        let conn = store.connection();
        let (input, unit) =
            input::assemble(&conn, tuning, store.now(), &lease, in_context, entries)?;
        let saved: Option<String> = conn.query_row(
            "SELECT call1_output FROM chunks WHERE id = ?1",
            [lease.chunk_id()],
            |row| row.get(0),
        )?;
        (input, unit, saved)
    };

    // A saved reply means call 2 failed last time: resume from it, with the
    // handles call 1 was given, rather than pay for call 1 again (TIM-92).
    let saved = saved.and_then(|saved| restore(&saved, &mut unit));
    let resumed = saved.is_some();
    let reply = match saved {
        Some(reply) => reply,
        None => match llm.complete(&call1_request(&input)) {
            Ok(response) => response.json,
            Err(error) => {
                let Some(chunk_error) = chunk_error(&error) else {
                    tracing::warn!(chunk = %input.chunk, %error, "the extraction queue holds");
                    return Err(ExtractError::Held { error });
                };
                let failure = queue::fail(store, leases, lease, chunk_error)?;
                return Err(ExtractError::Call1 { error, failure });
            }
        },
    };

    let checked = match claims::check(&reply, &input, &unit) {
        Ok(checked) => checked,
        Err(reason) => {
            let failure = queue::fail(store, leases, lease, INVALID_REPLY)?;
            return Err(ExtractError::InvalidReply { reason, failure });
        }
    };

    let vectors = match embed(store, embedder, &checked) {
        Ok(vectors) => vectors,
        Err(error) => {
            let failure = queue::fail(store, leases, lease, EMBEDDING)?;
            return Err(ExtractError::Embedding { error, failure });
        }
    };

    // The connection is released before the failure is counted.
    let searched = {
        let conn = store.connection();
        reconcile::search(
            &conn,
            floor(tuning, embedder),
            &input,
            &unit,
            &checked,
            &vectors,
        )
    };
    let search = match searched {
        Ok(search) => search,
        Err(error) => {
            let failure = queue::fail(store, leases, lease, SEARCH)?;
            return Err(ExtractError::Search {
                error: StoreError::Sqlite(error),
                failure,
            });
        }
    };
    let (labels, plan) = match &search {
        None => (Vec::new(), reconcile::Plan::all_new(checked.memories.len())),
        Some(search) => {
            if !resumed {
                save(store, &lease, &reply, &unit)?;
            }
            let labels = match llm.complete(&call2_request(&search.input)) {
                Ok(response) => match call2::parse(&response.json) {
                    Ok(labels) => labels,
                    Err(reason) => {
                        let failure = queue::fail(store, leases, lease, INVALID_REPLY)?;
                        return Err(ExtractError::InvalidReply { reason, failure });
                    }
                },
                Err(error) => {
                    let Some(chunk_error) = chunk_error(&error) else {
                        tracing::warn!(chunk = %input.chunk, %error, "the extraction queue holds");
                        return Err(ExtractError::Held { error });
                    };
                    let failure = queue::fail(store, leases, lease, chunk_error)?;
                    return Err(ExtractError::Call2 { error, failure });
                }
            };
            let plan = reconcile::plan(search, &input, &unit, &checked, &labels);
            (labels, plan)
        }
    };
    Ok(Prepared {
        lease,
        input,
        unit,
        checked,
        vectors,
        search,
        labels,
        plan,
    })
}

/// The second half of [`extract`]: writes a prepared chunk. A neighbour
/// that went between the halves, purged or forgotten, is planned without:
/// call 2's labels on it are dropped, so a claim that would have ended,
/// refined or restated it is new instead, as if it had never been stored.
///
/// The check, the replanning and the writes share one hold on the store
/// and one transaction, so a sweep or an erase on another thread can't
/// delete a neighbour between the check and the writes (TIM-117 review).
pub(crate) fn commit_prepared(
    store: &Store,
    leases: &Leases,
    prepared: Prepared,
) -> Result<Extracted, ExtractError> {
    let Prepared {
        lease,
        input,
        unit,
        checked,
        vectors,
        search,
        labels,
        plan,
    } = prepared;
    queue::check_held(leases, &lease)?;

    // The last moment before the commit takes the store, where a test can
    // change it (TIM-117 review, the lock gap).
    #[cfg(test)]
    gap::run(store);

    let committed = {
        let mut conn = store.connection();
        (|| -> Result<(Extracted, reconcile::Plan), StoreError> {
            let tx = conn.transaction()?;
            let plan = match &search {
                Some(search) => {
                    without_vanished(&tx, search, &input, &unit, &checked, &labels)?.unwrap_or(plan)
                }
                None => plan,
            };
            let neighbours = search.as_ref().map(|search| search.neighbours.as_slice());
            let extracted = commit::commit(
                &tx,
                store,
                &lease,
                &input,
                &unit,
                &checked,
                &vectors,
                &plan,
                neighbours.unwrap_or_default(),
            )?;
            tx.commit()?;
            Ok((extracted, plan))
        })()
    };
    match committed {
        Ok((extracted, plan)) => {
            tracing::info!(
                chunk = %extracted.chunk,
                memories = extracted.memories.len(),
                used = extracted.used.len(),
                dropped = extracted.dropped.len(),
                entities_created = extracted.entities_created.len(),
                reconciled = search.is_some(),
                accesses = plan.accesses.len(),
                edits = plan.edits.len(),
                "extracted a chunk"
            );
            Ok(extracted)
        }
        Err(error) => {
            // A commit that fails every time must still reach the retry cap
            // rather than hold the bank's queue for ever. The store's hold
            // was released above, so the failure can be counted.
            let failure = queue::fail(store, leases, lease, COMMIT)?;
            Err(ExtractError::Commit { error, failure })
        }
    }
}

/// The plan again without call 2's labels on neighbours that are no longer
/// in the store, read through the commit's transaction; `None` when every
/// neighbour is still there.
fn without_vanished(
    conn: &rusqlite::Connection,
    search: &reconcile::Search,
    input: &Call1Input,
    unit: &input::Unit,
    checked: &claims::Checked,
    labels: &[call2::ClaimLabels],
) -> Result<Option<reconcile::Plan>, StoreError> {
    let gone = vanished(conn, &search.neighbours)?;
    if gone.is_empty() {
        return Ok(None);
    }
    let gone_handles: std::collections::BTreeSet<&str> = search
        .input
        .neighbours
        .iter()
        .zip(&search.neighbours)
        .filter(|(_, neighbour)| gone.contains(&neighbour.id))
        .map(|(shown, _)| shown.handle.as_str())
        .collect();
    let kept: Vec<call2::ClaimLabels> = labels
        .iter()
        .map(|(claim, claim_labels)| {
            let still: Vec<_> = claim_labels
                .iter()
                .filter(|(neighbour, _)| !gone_handles.contains(neighbour.as_str()))
                .cloned()
                .collect();
            (claim.clone(), still)
        })
        .collect();
    tracing::info!(
        chunk = %input.chunk,
        neighbours = gone.len(),
        "neighbours went between call 2 and the commit; their labels are dropped"
    );
    Ok(Some(reconcile::plan(search, input, unit, checked, &kept)))
}

/// The ids among `neighbours` that are no longer in the store.
fn vanished(
    conn: &rusqlite::Connection,
    neighbours: &[reconcile::Neighbour],
) -> Result<Vec<i64>, StoreError> {
    let mut statement = conn.prepare("SELECT 1 FROM memories WHERE id = ?1")?;
    let mut gone = Vec::new();
    for neighbour in neighbours {
        if !statement.exists([neighbour.id])? {
            gone.push(neighbour.id);
        }
    }
    Ok(gone)
}

/// The reconcile floor for the embedder's exact model. The service refuses
/// to open without one (ADR 0009), so a missing floor never runs call 2 on
/// similarity alone.
fn floor(tuning: &Tuning, embedder: &dyn Embedder) -> f64 {
    tuning
        .reconcile
        .embedding_floors
        .get(embedder.model_id())
        .copied()
        .unwrap_or(f64::INFINITY)
}

/// One vector per claim. Every vector must fit the index before the commit
/// starts, so a bad embedder is an embedding failure rather than a failed
/// commit.
fn embed(
    store: &Store,
    embedder: &dyn Embedder,
    checked: &claims::Checked,
) -> Result<Vec<Vec<f32>>, ModelError> {
    let contents: Vec<&str> = checked
        .memories
        .iter()
        .map(|memory| memory.content.as_str())
        .collect();
    let width = store.vectors().dimensions();
    let vectors = embedder.embed(&contents)?;
    if vectors.len() == contents.len() && vectors.iter().all(|vector| vector.len() == width) {
        Ok(vectors)
    } else {
        Err(ModelError::Inference {
            model: embedder.model_id().to_owned(),
            reason: format!(
                "returned vectors that don't fit the index: one {width}-wide vector per text"
            ),
        })
    }
}

/// Saves call 1's reply on the chunk before call 2 runs, with the handles
/// it was given, so a retry resumes from it. The commit drops it (ADR 0008).
fn save(store: &Store, lease: &Lease, reply: &Value, unit: &input::Unit) -> Result<(), StoreError> {
    let saved = json!({
        "reply": reply,
        "entity_boundary": unit.entity_boundary,
        "candidates": unit.candidates,
        "in_context": unit
            .in_context
            .iter()
            .map(|(handle, (id, uuid))| (handle.clone(), json!([id, uuid.to_string()])))
            .collect::<serde_json::Map<_, _>>(),
        "entries": unit
            .entries
            .iter()
            .map(|(handle, cites)| {
                let cites: Vec<Value> = cites
                    .iter()
                    .map(|(id, uuid)| json!([id, uuid.to_string()]))
                    .collect();
                (handle.clone(), Value::Array(cites))
            })
            .collect::<serde_json::Map<_, _>>(),
    });
    store.connection().execute(
        "UPDATE chunks SET call1_output = ?1 WHERE id = ?2",
        (saved.to_string(), lease.chunk_id()),
    )?;
    Ok(())
}

/// Call 1's saved reply, with `unit` given back the handles call 1 saw.
/// `None` when there's nothing usable saved, so call 1 runs again.
fn restore(saved: &str, unit: &mut input::Unit) -> Option<Value> {
    let mut saved: Value = serde_json::from_str(saved).ok()?;
    let entity_boundary = saved.get("entity_boundary")?.as_i64()?;
    let candidates = saved
        .get("candidates")?
        .as_object()?
        .iter()
        .map(|(handle, id)| Some((handle.clone(), id.as_i64()?)))
        .collect::<Option<_>>()?;
    let in_context = saved
        .get("in_context")?
        .as_object()?
        .iter()
        .map(|(handle, pair)| {
            let id = pair.get(0)?.as_i64()?;
            let uuid = pair.get(1)?.as_str()?.parse().ok()?;
            Some((handle.clone(), (id, uuid)))
        })
        .collect::<Option<_>>()?;
    // A reply saved before version 6 has no entries: none were shown.
    let entries = match saved.get("entries") {
        Some(entries) => entries
            .as_object()?
            .iter()
            .map(|(handle, cites)| {
                let cites = cites
                    .as_array()?
                    .iter()
                    .map(|pair| {
                        let id = pair.get(0)?.as_i64()?;
                        let uuid = pair.get(1)?.as_str()?.parse().ok()?;
                        Some((id, uuid))
                    })
                    .collect::<Option<Vec<_>>>()?;
                Some((handle.clone(), cites))
            })
            .collect::<Option<_>>()?,
        None => Default::default(),
    };
    let reply = saved.get_mut("reply")?.take();
    unit.entity_boundary = entity_boundary;
    unit.candidates = candidates;
    unit.in_context = in_context;
    unit.entries = entries;
    Some(reply)
}

const INVALID_REPLY: ChunkError = ChunkError {
    kind: "invalid_reply",
    status: None,
};

const SEARCH: ChunkError = ChunkError {
    kind: "search",
    status: None,
};

const EMBEDDING: ChunkError = ChunkError {
    kind: "embedding",
    status: None,
};

const COMMIT: ChunkError = ChunkError {
    kind: "commit",
    status: None,
};

/// How a failed call is recorded on the chunk, or `None` when the queue
/// holds instead: the LLM can't be used at all, which is no fault of the
/// chunk's and would otherwise burn every queued chunk to failed.
fn chunk_error(error: &LlmError) -> Option<ChunkError> {
    let (kind, status) = match error {
        LlmError::NotConfigured { .. }
        | LlmError::Conflicting { .. }
        | LlmError::LoginRequired
        | LlmError::UsageLimited { .. } => return None,
        LlmError::Transport { .. } => ("llm_transport", None),
        LlmError::Timeout => ("llm_timeout", None),
        LlmError::Status { status } => ("llm_status", Some(*status)),
        LlmError::NoContent => ("llm_no_content", None),
        LlmError::NotJson { .. } => ("llm_not_json", None),
        LlmError::Refused => ("llm_refused", None),
        LlmError::Backend { .. } => ("llm_backend", None),
    };
    Some(ChunkError { kind, status })
}
