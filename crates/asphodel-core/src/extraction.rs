//! Extraction call 1: turning a chunk into claims and committing them.
//!
//! A leased chunk ([`crate::queue`]) goes through four steps:
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
//! 4. **Commit.** One transaction writes the memories and their vectors,
//!    entity links, new entities and aliases as logged edits, the `created`
//!    and `used` accesses, and marks the chunk extracted (TIM-90, ADR 0001).
//!
//! Reconciliation (call 2) comes in a later stage, so for now every claim
//! that survives the checks becomes a new memory. A failure anywhere before
//! the commit writes nothing but the queue's count of the failed attempt, so
//! the chunk is retried in place.

mod claims;
mod commit;
mod input;
mod prompt;

use jiff::Timestamp;
use jiff::civil::Date;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::config::Tuning;
use crate::models::{Embedder, LlmClient, LlmError, ModelError};
use crate::queue::{self, ChunkError, Failure, Lease, Leases, QueueError, SourceKind};
use crate::store::{Store, StoreError, VectorIndex};

pub use prompt::call1_request;

/// Call 1's template name and version, which replay's cassette keys include
/// (TIM-96, decision 4).
pub const CALL1_TEMPLATE: &str = "extract_claims";
pub const CALL1_VERSION: u32 = 1;

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

/// What one successful extraction committed.
#[derive(Debug, Clone, PartialEq)]
pub struct Extracted {
    pub chunk: Uuid,
    /// The new memories, in claim order, without the dropped claims.
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
/// `embedding` or `commit`.
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

    /// Embedding the new memories failed, and the queue counted it.
    #[error("embedding the new memories failed: {error}")]
    Embedding { error: ModelError, failure: Failure },

    /// The service was built without models, so there's nothing to embed
    /// with. Nothing is counted.
    #[error("no models are loaded, so nothing can be extracted")]
    NoModels,

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
            | ExtractError::InvalidReply { failure, .. }
            | ExtractError::Embedding { failure, .. }
            | ExtractError::Commit { failure, .. } => Some(*failure),
            ExtractError::Held { .. }
            | ExtractError::NoModels
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

/// Runs call 1 on the leased chunk and commits what it found.
pub(crate) fn extract(
    store: &Store,
    leases: &Leases,
    tuning: &Tuning,
    embedder: &dyn Embedder,
    lease: Lease,
    llm: &dyn LlmClient,
    in_context: &[Uuid],
) -> Result<Extracted, ExtractError> {
    queue::check_held(leases, &lease)?;
    let (input, unit) = {
        let conn = store.connection();
        input::assemble(&conn, tuning, store.now(), &lease, in_context)?
    };

    let request = call1_request(&input);
    let response = match llm.complete(&request) {
        Ok(response) => response,
        Err(error) => {
            let Some(chunk_error) = chunk_error(&error) else {
                tracing::warn!(chunk = %input.chunk, %error, "the extraction queue holds");
                return Err(ExtractError::Held { error });
            };
            let failure = queue::fail(store, leases, lease, chunk_error)?;
            return Err(ExtractError::Call1 { error, failure });
        }
    };

    let checked = match claims::check(&response.json, &input, &unit) {
        Ok(checked) => checked,
        Err(reason) => {
            let failure = queue::fail(store, leases, lease, INVALID_REPLY)?;
            return Err(ExtractError::InvalidReply { reason, failure });
        }
    };

    let contents: Vec<&str> = checked
        .memories
        .iter()
        .map(|memory| memory.content.as_str())
        .collect();
    // Every vector must fit the index before the commit starts, so a bad
    // embedder is an embedding failure rather than a failed commit.
    let width = store.vectors().dimensions();
    let vectors = match embedder.embed(&contents) {
        Ok(vectors)
            if vectors.len() == contents.len()
                && vectors.iter().all(|vector| vector.len() == width) =>
        {
            vectors
        }
        Ok(_) => {
            let error = ModelError::Inference {
                model: embedder.model_id().to_owned(),
                reason: format!(
                    "returned vectors that don't fit the index: one {width}-wide vector per text"
                ),
            };
            let failure = queue::fail(store, leases, lease, EMBEDDING)?;
            return Err(ExtractError::Embedding { error, failure });
        }
        Err(error) => {
            let failure = queue::fail(store, leases, lease, EMBEDDING)?;
            return Err(ExtractError::Embedding { error, failure });
        }
    };

    match commit::commit(store, &lease, &input, &unit, &checked, &vectors) {
        Ok(extracted) => {
            tracing::info!(
                chunk = %extracted.chunk,
                memories = extracted.memories.len(),
                used = extracted.used.len(),
                dropped = extracted.dropped.len(),
                entities_created = extracted.entities_created.len(),
                "extracted a chunk"
            );
            Ok(extracted)
        }
        Err(error) => {
            // A commit that fails every time must still reach the retry cap
            // rather than hold the bank's queue for ever.
            let failure = queue::fail(store, leases, lease, COMMIT)?;
            Err(ExtractError::Commit { error, failure })
        }
    }
}

const INVALID_REPLY: ChunkError = ChunkError {
    kind: "invalid_reply",
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
