//! Extraction: turning a chunk into claims, reconciling them with what's
//! stored, and committing the result.
//!
//! A leased chunk ([`crate::queue`]) goes through five steps:
//!
//! 1. **Input.** Code assembles what call 1 sees: the chunk's text, its
//!    context (up to [`CONTEXT_TURNS`] earlier turns of the session, or the
//!    text before a document chunk), the reference date and a calendar strip
//!    in the source's timezone, the speaker, the entity candidates found
//!    through the alias FTS, the in-context memories, and for a turn the
//!    bank's nearest upcoming events. Memories cited by the session's
//!    prompt block are shown under their own handles, never as a copy of
//!    the block's prose; `used` credits the memory the reply relied on.
//! 2. **Call 1.** One structured-output call ([`call1_request`]) returns the
//!    claims and the `used` verdicts.
//! 3. **Checks in code.** A claim whose quote isn't in the chunk is dropped.
//!    Times become instants at the start of their unit in the source's
//!    timezone, fields that don't belong to a kind are dropped, a named
//!    weekday that doesn't match the date lowers window confidence, an RRULE
//!    is kept only when it parses and recurs, and remember-this keeps a
//!    memory only from the owner's own message.
//! 4. **Reconciliation.** Code finds each claim's nearest stored memories,
//!    and when one clears the floor or a claim signals a change, call 2
//!    ([`call2_request`]) labels the claims against them
//!    ([`reconcile`](self::reconcile)). Call 1's reply is saved on the
//!    chunk first, so a failed call 2 is retried from it.
//! 5. **Commit.** One transaction writes the new memories and their
//!    vectors, entity links, new entities and aliases as logged edits, the
//!    neighbours ended, retracted or refined and their logged edits, the
//!    `created`, `mentioned_again`, `confirmed` and `used` accesses, and
//!    marks the chunk extracted, dropping call 1's saved reply so no claim
//!    text outlives a purge or forget.
//!
//! A failure anywhere before the commit writes nothing but the queue's
//! count of the failed attempt and call 1's saved reply, so the chunk is
//! retried in place.
//!
//! With `[llm] concurrency` above 1 a bank has several chunks in flight,
//! each searching before the others commit. Each commits in the order it
//! was handed out ([`crate::queue`]), and the commit checks, under the
//! store, what its search could have missed: a memory committed since that
//! clears the floor for one of its claims, an edit since on a neighbour
//! call 2 was shown, or any new memory when a claim is flagged. A chunk
//! that missed something is [`Committed::Stale`]: it searches and runs
//! call 2 again from the same call 1, which isn't counted against it, so a
//! repeat is still an access on the memory it repeats.

mod call2;
mod claims;
mod commit;
mod input;
mod prompt;
mod reconcile;

use jiff::Timestamp;
use jiff::civil::Date;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::config::Tuning;
use crate::models::{Embedder, LlmClient, LlmError, ModelError};
use crate::queue::{self, ChunkError, Failure, Lease, Leases, QueueError, SourceKind};
use crate::store::{Store, StoreError, VectorIndex};

pub use call2::call2_request;
pub(crate) use input::{entities_named, phrase, survivor};
pub use prompt::call1_request;

/// Call 1's template name and version, which replay's cassette keys include.
pub const CALL1_TEMPLATE: &str = "extract_claims";
pub const CALL1_VERSION: u32 = 17;

/// The hash call 1's template carries for `[extraction] guidance`:
/// lower-case hex SHA-256 of the text as the prompt inserts it, trimmed.
pub fn guidance_hash(guidance: Option<&str>) -> Option<String> {
    guidance.map(|guidance| crate::chunking::hex(&Sha256::digest(guidance.trim().as_bytes())))
}

/// Call 2's template name and version, which replay's cassette keys include.
pub const CALL2_TEMPLATE: &str = "reconcile_claims";
pub const CALL2_VERSION: u32 = 5;

/// The top five neighbours per claim after fusing vector search and BM25.
/// A flagged claim's entity-linked open tasks and current states come on top.
pub const NEIGHBOURS_PER_CLAIM: usize = 5;

/// Neighbours shown for the whole chunk, capped at 40.
pub const NEIGHBOUR_CAP: usize = 40;

/// The edit log kinds reconciliation writes, each on the row of the memory
/// it changed (`edits.memory_id`), with ids and times in `details` and never
/// content.
pub const EDIT_ENDED: &str = "memory_ended";
pub const EDIT_RETRACTED: &str = "memory_retracted";
pub const EDIT_REFINED: &str = "memory_refined";
pub const EDIT_SIGNIFICANCE_RAISED: &str = "significance_raised";
/// A remember-this on a neighbour sets the owner's significance to kept.
pub const EDIT_KEPT: &str = "memory_kept";
/// The memory that ended another was retracted or refined, so the ended
/// memory's `ended_by` and `valid_until` follow the successor.
pub const EDIT_END_REPOINTED: &str = "end_repointed";
/// The memory that ended another was denied, so the ended memory is open
/// again: its `valid_until` and `ended_by` are cleared.
pub const EDIT_END_CLEARED: &str = "end_cleared";

/// Up to three earlier clean turns of the session given as context.
pub const CONTEXT_TURNS: usize = 3;

/// The context turns' total size in characters. Over it, they're clipped
/// oldest first: characters come off the start of the oldest turn, and a turn
/// clipped to nothing is left out.
pub const CONTEXT_CHARS: usize = 6_000;

/// How much of the document before a chunk it gets as context, in characters.
/// It includes the last few hundred characters before the chunk, taken from
/// the source text, so a chunk skipped as seen in an earlier
/// version of the document still provides it.
pub const PREVIOUS_CHUNK_CHARS: usize = 400;

/// Up to 30 entity candidates found by alias search, ranked by how many
/// memories link to them. `user`, `assistant` and the speaker come on top,
/// since search never has to find them.
pub const ENTITY_CANDIDATE_CAP: usize = 30;

/// Linked memory sentences shown with a candidate, strongest first.
pub const CANDIDATE_MEMORIES: usize = 3;

/// The bank's upcoming events shown with a turn, nearest first.
pub const UPCOMING_EVENTS: usize = 5;

/// The calendar strip: the reference date and three weeks either side.
pub const CALENDAR_DAYS: i64 = 21;

/// Everything call 1 is given for one chunk. Handles are short ids local to
/// the call (`e1`, `e2`, … for candidates and `m1`, `m2`, … for in-context
/// memories), which the reply refers back to; they're cheaper and harder to
/// garble than UUIDs.
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
    /// aren't resolved.
    pub reference_date: Option<Date>,
    /// The reference date and [`CALENDAR_DAYS`] either side, in order. The
    /// prompt renders each as `YYYY-MM-DD Weekday`. Empty when
    /// `reference_date` is `None`.
    pub calendar: Vec<Date>,
    /// Whose words the user message is; `None` for a document. "I" and "me"
    /// resolve to them.
    pub speaker: Option<SpeakerRef>,
    /// Context only, oldest first: never quoted from, never extracted from on
    /// its own.
    pub context: Vec<String>,
    /// Source-local time anchors aligned with `context` for earlier turns.
    /// Empty for document context, which uses the document reference date.
    /// These survive clipping of their passage and are never quote text.
    pub context_times: Vec<ContextTurnTime>,
    /// `user`, `assistant` and the speaker first, then the entities whose
    /// aliases appear in the text or its context.
    pub candidates: Vec<Candidate>,
    /// The in-context memories call 1 judges `used` against. Always empty
    /// for a document, which has no reply to have used anything.
    pub in_context: Vec<InContextMemory>,
    /// Up to [`UPCOMING_EVENTS`] of the bank's events that are still
    /// upcoming, from any session, nearest first, leaving out any already
    /// in context. They ground a task's window on a dated occasion the
    /// current text only names. They have no handles, so the reply can't
    /// credit them `used`. Always empty for a document.
    pub upcoming: Vec<UpcomingEvent>,
    /// `[llm] language`: the language every claim is written in. `None`
    /// writes each in the language of the passage it quotes.
    pub language: Option<String>,
    /// `[extraction] guidance`, added after the fixed rules. `None` sends
    /// them alone.
    pub guidance: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ContextTurnTime {
    pub observed_at: Timestamp,
    pub timezone: String,
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
pub struct UpcomingEvent {
    pub memory: Uuid,
    pub content: String,
}

/// What a claim does to a neighbour (CONTEXT.md "Reconciliation").
/// `Retracts` is a corrected version of the neighbour, such as a reschedule;
/// `Denies` says the neighbour didn't happen or isn't true. Both retract it,
/// and differ only in what happens to anything it had ended.
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
    pub due_at: Option<Timestamp>,
    pub valid_from: Option<Timestamp>,
    pub valid_until: Option<Timestamp>,
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
/// its chain instead.
#[derive(Debug, Clone, PartialEq)]
pub struct NeighbourMemory {
    pub handle: String,
    pub memory: Uuid,
    pub content: String,
    pub due_at: Option<Timestamp>,
    pub valid_from: Option<Timestamp>,
    pub valid_until: Option<Timestamp>,
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
    /// The new memories among `memories` that call 2 labelled a repeat of a
    /// neighbour, made chain heads instead because they matter at least
    /// `reconcile.promotion_gap` significance levels more than it.
    pub promoted: Vec<Uuid>,
    /// Restatements written: one per absorbed newer claim and memory it was
    /// credited to.
    pub restatements: usize,
    /// Refinements between kinds that can't be versions of each other, in
    /// label order, and whether `reconcile.kind_guard` rejected each.
    pub kind_mismatches: Vec<KindMismatch>,
}

/// What made reconciliation treat a claim as a refinement of a neighbour.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RefineCause {
    /// Call 2 labelled it `refines`.
    Explicit,
    /// A repeat label on a claim that supplies a new or changed date.
    DatePromoted,
    /// A repeat label on a claim that matters `reconcile.promotion_gap`
    /// levels more than the neighbour.
    WeightPromoted,
}

/// A refinement between a claim and a neighbour of kinds that can't be
/// versions of each other: any two kinds that differ, except a task or
/// recurring claim on a task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KindMismatch {
    /// The claim's index in call 1's reply.
    pub claim: usize,
    /// The memory the claim made, if it made one.
    pub memory: Option<Uuid>,
    pub neighbour: Uuid,
    pub claim_kind: crate::strength::Kind,
    pub neighbour_kind: crate::strength::Kind,
    pub cause: RefineCause,
    /// The claim was said before the neighbour.
    pub older: bool,
    /// `reconcile.kind_guard` dropped the label. When it's off, the
    /// refinement went ahead as call 2 or the promotion asked.
    pub rejected: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dropped {
    pub claim: usize,
    pub reason: DropReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DropReason {
    /// The quote is empty or isn't in the chunk's text. A passage that's
    /// only in the context doesn't count.
    QuoteNotFound,
    /// The content is empty after trimming.
    EmptyContent,
    /// A task quoted from the assistant's reply with neither a due date nor
    /// an until-event (beyond the current turn).
    AssistantTaskUndated,
}

/// Why an extraction didn't commit. No variant carries the prompt, the reply
/// or a claim, since content is only ever logged at `trace`.
///
/// The chunk's `last_error_kind` names the cause: `llm_transport`,
/// `llm_timeout`, `llm_status` (with the status), `llm_no_content`,
/// `llm_not_json`, `llm_refused`, `llm_backend`, `invalid_reply`,
/// `embedding`, `search` or `commit`. A call 2 failure records the same
/// `llm_*` kinds as call 1.
#[derive(Debug, thiserror::Error)]
pub enum ExtractError {
    /// The LLM can't be used at all: not configured, conflicting settings,
    /// no valid login, a usage limit, or a 429 that said when to retry. Not the chunk's fault, so nothing is
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
    /// can't be compared with its memories or given vectors that fit them.
    /// Nothing is counted: the queue holds until a re-embed moves the bank
    /// to the daemon's model.
    #[error(
        "the bank records embedding model {model}, which this daemon doesn't carry; run `asphodel reembed --bank` to move it"
    )]
    ModelUnavailable { model: String },

    /// The commit transaction failed and was rolled back, and the queue
    /// counted it, so a fault that recurs reaches the retry cap rather than
    /// holding the bank's queue.
    #[error("committing the chunk failed: {error}")]
    Commit { error: StoreError, failure: Failure },

    /// Another chunk of the bank committed something since this one's
    /// search that it must reconcile against, and the caller can't run
    /// call 2 again. Nothing is counted: the lease is released and the
    /// chunk is extracted again.
    #[error("the chunk's neighbours changed since its search")]
    Stale,

    /// The chunk's document was removed while it was queued or in flight.
    /// Nothing from it is written; the chunk leaves the queue unextracted
    /// and nothing is counted.
    #[error("the chunk's document was removed")]
    SourceRemoved,

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
            | ExtractError::Stale
            | ExtractError::SourceRemoved
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
    let (input, _) = input::assemble(&conn, tuning, store.now(), lease, in_context)?;
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
        input::assemble(&conn, tuning, store.now(), lease, in_context)?
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
/// the second a latency later.
pub struct Prepared {
    lease: Lease,
    input: Call1Input,
    unit: input::Unit,
    /// Call 1's reply, saved on the chunk before call 2 runs.
    reply: Value,
    /// Whether the reply is saved on the chunk already.
    saved: bool,
    checked: claims::Checked,
    vectors: Vec<Vec<f32>>,
    /// The reconcile floor the search used.
    floor: f64,
    /// `strength.corroborate_used`, read with the rest of the tuning.
    corroborate_used: bool,
    /// `[reconcile]`'s rules for the plan, read with the rest of the tuning.
    rules: reconcile::Rules,
    /// The bank as the search saw it.
    snapshot: reconcile::Snapshot,
    search: Option<reconcile::Search>,
    labels: Vec<call2::ClaimLabels>,
    plan: reconcile::Plan,
    /// The model that answered call 2, when it ran.
    call2_model: Option<String>,
}

impl Prepared {
    /// The lease the chunk is held under.
    pub fn lease(&self) -> &Lease {
        &self.lease
    }

    /// What call 1 was given.
    pub fn call1_input(&self) -> &Call1Input {
        &self.input
    }

    /// What call 2 was given, or `None` when it didn't run.
    pub fn call2_input(&self) -> Option<&Call2Input> {
        self.search.as_ref().map(|search| &search.input)
    }
}

impl std::fmt::Debug for Prepared {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Prepared")
            .field("chunk", &self.lease.chunk)
            .field("claims", &self.checked.memories.len())
            .field("reconciled", &self.search.is_some())
            .finish_non_exhaustive()
    }
}

/// What [`commit_prepared`] did.
#[derive(Debug)]
pub enum Committed {
    /// The chunk is committed.
    Extracted(Extracted),
    /// Another chunk of the bank committed since this one's search
    /// something it must reconcile against. Nothing was written; run
    /// [`redo`] and commit again.
    Stale(Box<Prepared>),
}

/// A claim and the neighbours call 2 was shown for it, for replay's
/// labelling material.
/// Flagged claims bypass the vector floor, and BM25 neighbours are not
/// filtered by it, so these lists may include below-floor candidates.
/// Thresholding their scores does not predict what another reconcile
/// floor would retain.
#[derive(Debug, Clone, PartialEq)]
pub struct Call2List {
    /// The chunk the claim came from.
    pub chunk: Uuid,
    /// The claim's index in call 1's reply, which with the chunk names
    /// the claim across runs, as it names the memory a claim makes.
    pub ordinal: usize,
    /// The claim's sentence.
    pub claim: String,
    pub candidates: Vec<Call2Candidate>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Call2Candidate {
    pub memory: Uuid,
    pub sentence: String,
    /// The claim's cosine similarity to it, which the reconcile floor
    /// compares; `None` when its vector has gone since.
    pub similarity: Option<f64>,
}

/// The lists call 2 was shown for a prepared chunk, one per claim; empty
/// when call 2 didn't run. Only what call 2 was shown: neighbours below
/// the floor never reach it, so they aren't here either.
pub(crate) fn call2_lists(
    store: &Store,
    prepared: &Prepared,
) -> Result<Vec<Call2List>, StoreError> {
    let Some(search) = &prepared.search else {
        return Ok(Vec::new());
    };
    let conn = store.connection();
    Ok(reconcile::shown_lists(&conn, search, &prepared.vectors)?)
}

/// Runs call 1 on the leased chunk, or takes its saved reply, reconciles
/// the claims with call 2 when they land near something stored, and commits
/// the result, reconciling again as often as another chunk of the bank
/// commits something it must see first.
#[allow(clippy::too_many_arguments)]
pub(crate) fn extract(
    store: &Store,
    leases: &Leases,
    tuning: &Tuning,
    embedder: &dyn Embedder,
    lease: Lease,
    llm: &dyn LlmClient,
    in_context: &[Uuid],
) -> Result<Extracted, ExtractError> {
    let mut prepared = prepare(store, leases, tuning, embedder, lease, llm, in_context)?;
    loop {
        match commit_prepared(store, leases, prepared)? {
            Committed::Extracted(extracted) => return Ok(extracted),
            Committed::Stale(stale) => prepared = redo(store, leases, llm, *stale)?,
        }
    }
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
) -> Result<Prepared, ExtractError> {
    queue::check_held(leases, &lease)?;
    if queue::source_removed(&store.connection(), lease.chunk_id())? {
        discard(store, &lease)?;
        return Err(ExtractError::SourceRemoved);
    }
    let (input, mut unit, saved) = {
        let conn = store.connection();
        let (input, unit) = input::assemble(&conn, tuning, store.now(), &lease, in_context)?;
        let saved: Option<String> = conn.query_row(
            "SELECT call1_output FROM chunks WHERE id = ?1",
            [lease.chunk_id()],
            |row| row.get(0),
        )?;
        (input, unit, saved)
    };

    // A saved reply means call 2 failed last time: resume from it, with the
    // handles call 1 was given, rather than pay for call 1 again.
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

    let claims = checked.memories.len();
    reconcile_claims(
        store,
        leases,
        llm,
        Prepared {
            lease,
            input,
            unit,
            reply,
            saved: resumed,
            checked,
            vectors,
            floor: floor(tuning, embedder),
            corroborate_used: tuning.strength.corroborate_used,
            rules: reconcile::Rules::of(tuning),
            snapshot: reconcile::Snapshot::default(),
            search: None,
            labels: Vec::new(),
            plan: reconcile::Plan::all_new(claims),
            call2_model: None,
        },
    )
}

/// Searches and runs call 2 again for a chunk [`commit_prepared`] found
/// stale, from the same call 1. The redo itself isn't counted against the
/// chunk; a call 2 that fails is, as it would be the first time.
pub(crate) fn redo(
    store: &Store,
    leases: &Leases,
    llm: &dyn LlmClient,
    prepared: Prepared,
) -> Result<Prepared, ExtractError> {
    tracing::info!(
        chunk = %prepared.input.chunk,
        "another chunk committed near this one since its search; reconciling again"
    );
    reconcile_claims(store, leases, llm, prepared)
}

/// What call 2 would be given if `prepared` searched again now, or `None`
/// when it wouldn't run. Reads only.
pub(crate) fn redo_input(
    store: &Store,
    prepared: &Prepared,
) -> Result<Option<Call2Input>, StoreError> {
    let conn = store.connection();
    Ok(reconcile::search(
        &conn,
        prepared.floor,
        &prepared.input,
        &prepared.unit,
        &prepared.checked,
        &prepared.vectors,
    )?
    .map(|search| search.input))
}

/// Finds the claims' neighbours, noting what the bank held at that moment,
/// and runs call 2 when they land near something stored.
fn reconcile_claims(
    store: &Store,
    leases: &Leases,
    llm: &dyn LlmClient,
    prepared: Prepared,
) -> Result<Prepared, ExtractError> {
    let Prepared {
        lease,
        input,
        unit,
        reply,
        mut saved,
        checked,
        vectors,
        floor,
        corroborate_used,
        rules,
        ..
    } = prepared;
    // The connection is released before the failure is counted.
    let searched = {
        let conn = store.connection();
        reconcile::snapshot(&conn, unit.bank_id, leases.commits(unit.bank_id)).and_then(
            |snapshot| {
                reconcile::search(&conn, floor, &input, &unit, &checked, &vectors)
                    .map(|search| (snapshot, search))
            },
        )
    };
    let (snapshot, search) = match searched {
        Ok(searched) => searched,
        Err(error) => {
            let failure = queue::fail(store, leases, lease, SEARCH)?;
            return Err(ExtractError::Search {
                error: StoreError::Sqlite(error),
                failure,
            });
        }
    };
    let (labels, plan, call2_model) = match &search {
        None => (
            Vec::new(),
            reconcile::Plan::all_new(checked.memories.len()),
            None,
        ),
        Some(search) => {
            if !saved {
                save(store, &lease, &reply, &unit)?;
                saved = true;
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
            let plan = reconcile::plan(search, &input, &unit, &checked, &labels, rules);
            (labels, plan, Some(llm.model().to_string()))
        }
    };
    Ok(Prepared {
        lease,
        input,
        unit,
        reply,
        saved,
        checked,
        vectors,
        floor,
        corroborate_used,
        rules,
        snapshot,
        search,
        labels,
        plan,
        call2_model,
    })
}

/// The second half of [`extract`]: writes a prepared chunk, once every
/// chunk of its bank handed out before it has committed or failed. A
/// neighbour that went between the halves, purged or forgotten, is planned
/// without: call 2's labels on it are dropped, so a claim that would have
/// ended, refined or restated it is new instead, as if it had never been
/// stored. When another chunk of the bank committed since the search
/// something this one must reconcile against, nothing is written and the
/// chunk comes back [`Committed::Stale`].
///
/// The checks, the replanning and the writes share one hold on the store
/// and one transaction, so a sweep or an erase on another thread can't
/// delete a neighbour between the check and the writes.
pub(crate) fn commit_prepared(
    store: &Store,
    leases: &Leases,
    prepared: Prepared,
) -> Result<Committed, ExtractError> {
    queue::check_held(leases, &prepared.lease)?;
    leases.wait_turn(&prepared.lease);

    let bank_id = prepared.unit.bank_id;
    let committed = {
        let mut conn = store.connection();
        (|| -> Result<Commit, StoreError> {
            let tx = conn.transaction()?;
            // The document was removed since the chunk was handed out:
            // nothing it found may be written.
            if queue::source_removed(&tx, prepared.lease.chunk_id())? {
                queue::drop_removed(&tx, &prepared.lease)?;
                tx.commit()?;
                return Ok(Commit::Removed);
            }
            if leases.commits(bank_id) != prepared.snapshot.commits
                && reconcile::stale(
                    &tx,
                    bank_id,
                    &prepared.snapshot,
                    prepared.floor,
                    &prepared.checked,
                    &prepared.vectors,
                    prepared.search.as_ref(),
                )?
            {
                return Ok(Commit::Stale);
            }
            let plan = match &prepared.search {
                Some(search) => without_vanished(
                    &tx,
                    search,
                    &prepared.input,
                    &prepared.unit,
                    &prepared.checked,
                    &prepared.labels,
                    prepared.rules,
                )?
                .unwrap_or_else(|| prepared.plan.clone()),
                None => prepared.plan.clone(),
            };
            let neighbours = prepared
                .search
                .as_ref()
                .map(|search| search.neighbours.as_slice());
            let extracted = commit::commit(
                &tx,
                store,
                &prepared.lease,
                &prepared.input,
                &prepared.unit,
                &prepared.checked,
                &prepared.vectors,
                &plan,
                neighbours.unwrap_or_default(),
                prepared.corroborate_used,
                prepared.call2_model.as_deref(),
            )?;
            tx.commit()?;
            leases.committed(bank_id);
            Ok(Commit::Written(Box::new((extracted, plan))))
        })()
    };
    match committed {
        Ok(Commit::Written(written)) => {
            let (extracted, plan) = *written;
            tracing::info!(
                chunk = %extracted.chunk,
                memories = extracted.memories.len(),
                used = extracted.used.len(),
                dropped = extracted.dropped.len(),
                entities_created = extracted.entities_created.len(),
                reconciled = prepared.search.is_some(),
                accesses = plan.accesses.len(),
                edits = plan.edits.len(),
                "extracted a chunk"
            );
            Ok(Committed::Extracted(extracted))
        }
        Ok(Commit::Stale) => Ok(Committed::Stale(Box::new(prepared))),
        Ok(Commit::Removed) => {
            tracing::info!(
                chunk = %prepared.input.chunk,
                "the chunk's document was removed; its extraction is discarded"
            );
            Err(ExtractError::SourceRemoved)
        }
        Err(error) => {
            // A commit that fails every time must still reach the retry cap
            // rather than hold the bank's queue for ever. The store's hold
            // was released above, so the failure can be counted.
            let failure = queue::fail(store, leases, prepared.lease, COMMIT)?;
            Err(ExtractError::Commit { error, failure })
        }
    }
}

/// What one try at [`commit_prepared`]'s transaction did.
enum Commit {
    Written(Box<(Extracted, reconcile::Plan)>),
    /// Another chunk committed something this one must see first.
    Stale,
    /// The chunk's document was removed; it left the queue.
    Removed,
}

/// Takes a leased chunk whose document was removed off the queue.
fn discard(store: &Store, lease: &Lease) -> Result<(), ExtractError> {
    let mut conn = store.connection();
    let tx = conn.transaction()?;
    queue::drop_removed(&tx, lease)?;
    tx.commit()?;
    tracing::info!(chunk = %lease.chunk, "the chunk's document was removed; it isn't extracted");
    Ok(())
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
    rules: reconcile::Rules,
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
    Ok(Some(reconcile::plan(
        search, input, unit, checked, &kept, rules,
    )))
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
/// to open without one, so a missing floor never runs call 2 on similarity
/// alone.
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
/// it was given, so a retry resumes from it. The commit drops it, so no
/// claim text outlives a purge or forget.
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
    let reply = saved.get_mut("reply")?.take();
    unit.entity_boundary = entity_boundary;
    unit.candidates = candidates;
    unit.in_context = in_context;
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
/// chunk's and would otherwise burn every queued chunk to failed, or the
/// daemon stopped the call while it waited to retry.
fn chunk_error(error: &LlmError) -> Option<ChunkError> {
    let (kind, status) = match error {
        LlmError::NotConfigured { .. }
        | LlmError::Conflicting { .. }
        | LlmError::LoginRequired
        | LlmError::UsageLimited { .. }
        | LlmError::RateLimited { .. }
        | LlmError::Stopped => return None,
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
