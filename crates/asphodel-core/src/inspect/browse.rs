//! What the dashboard lists: each bank's counts, its memories, and its
//! sources with their chunks.
//!
//! Like the rest of [`crate::inspect`] these read only. Above all they never
//! go through recall, so browsing writes no access and no recall row, and
//! looking at a memory never strengthens it. A memory's fade and purge are
//! the projections `memory show` gives ([`super::outlook`]), computed for
//! every matching row before paging when the sort or a filter needs them.

use std::collections::{BTreeMap, BTreeSet};

use jiff::Timestamp;
use jiff::tz::TimeZone;
use rusqlite::types::Value as Sql;
use rusqlite::{Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::{Gone, InspectError, Outlook, Projected, SignificanceView, outlook, parse};
use crate::agenda::has_ended;
use crate::config::{PurgePause, Tuning};
use crate::constants::TAU;
use crate::entities::resolve;
use crate::ingest::find_bank;
use crate::queue::SourceKind;
use crate::retrieval::PhaseFilter;
use crate::store::strength::{StrengthLoader, memory_kind, significance_value, window, world_time};
use crate::store::{Store, timestamp};
use crate::strength::Phase;

/// The most rows one page holds.
pub const MAX_LIST: usize = 200;

/// A page's rows when the query doesn't say.
pub const DEFAULT_LIST: usize = 50;

/// The significance levels a memory can have, lowest first, and `kept`.
const LEVELS: [&str; 6] = ["trivial", "minor", "notable", "major", "critical", "kept"];

/// Where a memory stands. One status per memory, the first that holds in
/// this order: forgetting, retracted, superseded, ended, live.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryStatus {
    /// Recall, the agenda and refreshes can see it.
    Live,
    /// A later memory refined it; the chain's head is what recall shows.
    Superseded,
    /// Another memory ended it, or its window has closed.
    Ended,
    /// Invalidated: a later memory or the owner said it never held.
    Retracted,
    /// Forgotten and hidden; its erase hasn't run yet.
    Forgetting,
}

/// How many memories have each status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
pub struct StatusCounts {
    pub live: usize,
    pub superseded: usize,
    pub ended: usize,
    pub retracted: usize,
    pub forgetting: usize,
}

impl StatusCounts {
    fn add(&mut self, status: MemoryStatus) {
        match status {
            MemoryStatus::Live => self.live += 1,
            MemoryStatus::Superseded => self.superseded += 1,
            MemoryStatus::Ended => self.ended += 1,
            MemoryStatus::Retracted => self.retracted += 1,
            MemoryStatus::Forgetting => self.forgetting += 1,
        }
    }
}

/// The order of the memory list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemorySort {
    /// Newest first.
    #[default]
    Created,
    /// The soonest to fade first; those already faded lead, and those that
    /// never fade come last.
    Fade,
    /// The strongest first.
    Strength,
}

/// The fade filter, by the projection if the memory isn't used again.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Fading {
    /// Below τ now: recall leaves it out.
    Faded,
    /// Recallable now, and fades within 7 bank days.
    Week,
    /// Recallable now, and fades within 30 bank days.
    Month,
    /// Never fades: kept, or the lasting floor holds it up.
    Never,
}

/// What `GET /v1/banks/{bank}/memories` takes. Every field narrows the list;
/// none of them is required.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct MemoryQuery {
    pub status: Option<MemoryStatus>,
    /// `fact`, `event`, `state`, `task` or `recurring`.
    pub kind: Option<String>,
    /// Words the sentence must all contain, each as a prefix.
    pub q: Option<String>,
    /// An entity by id, name or alias; memories linked to it or to an
    /// entity merged into it.
    pub entity: Option<String>,
    /// The effective significance: the owner's setting, else extraction's.
    pub significance: Option<String>,
    /// Only kept memories, or only those not kept.
    pub kept: Option<bool>,
    pub source_kind: Option<SourceKind>,
    pub document_id: Option<String>,
    pub session_id: Option<String>,
    /// When it was said, from (inclusive) and to (exclusive).
    pub observed_from: Option<Timestamp>,
    pub observed_to: Option<Timestamp>,
    pub phase: PhaseFilter,
    pub fading: Option<Fading>,
    pub sort: MemorySort,
    /// The `next_cursor` of the page before.
    pub cursor: Option<String>,
    /// At most [`MAX_LIST`]; [`DEFAULT_LIST`] when absent.
    pub limit: Option<usize>,
}

/// One row of the memory list.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MemorySummary {
    pub id: Uuid,
    pub sentence: String,
    pub kind: String,
    pub phase: Option<Phase>,
    pub status: MemoryStatus,
    pub significance: SignificanceView,
    pub observed_at: Timestamp,
    pub created_at: Timestamp,
    /// The source and chunk it rests on.
    pub source: Uuid,
    pub source_kind: SourceKind,
    pub document_id: Option<String>,
    pub session_id: Option<String>,
    pub superseded_by: Option<Uuid>,
    pub ended_by: Option<Uuid>,
    /// Strength now; recall leaves out anything below τ.
    pub strength: f64,
    pub recallable: bool,
    /// When strength falls below τ if it isn't used again. `None` never;
    /// zero bank days means it already has.
    pub fade: Option<Projected>,
    /// When its chain can be purged, as `memory show` projects it.
    pub purge: Option<Projected>,
}

/// One page of the memory list.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MemoryPage {
    pub memories: Vec<MemorySummary>,
    /// Every memory that matches, across pages.
    pub total: usize,
    /// The matches by status, counting every filter but `status`, for the
    /// filter's choices.
    pub statuses: StatusCounts,
    pub next_cursor: Option<String>,
    /// The daemon's now, which strength and the projections are taken at.
    pub as_of: Timestamp,
}

/// What `GET /v1/banks/{bank}/sources` takes.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct SourceQuery {
    pub kind: Option<SourceKind>,
    pub document_id: Option<String>,
    pub session_id: Option<String>,
    /// Words the document id, session id, stored text or reply must all
    /// contain, each as a prefix. Text that's gone isn't searched.
    pub q: Option<String>,
    /// Only sources whose text is gone, or only those whose text is kept.
    pub gone: Option<bool>,
    pub cursor: Option<String>,
    pub limit: Option<usize>,
}

/// One row of the source list.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SourceSummary {
    pub id: Uuid,
    pub kind: SourceKind,
    pub document_id: Option<String>,
    pub session_id: Option<String>,
    pub message_at: Option<Timestamp>,
    pub observed_at: Timestamp,
    pub ingested_at: Timestamp,
    pub chunks: usize,
    /// Chunks waiting or in flight.
    pub queued: usize,
    pub failed: usize,
    /// Visible memories resting on its chunks.
    pub memories: usize,
    /// Why its text is gone, when it is.
    pub gone: Option<Gone>,
    pub secret_kinds: Vec<String>,
}

/// One page of the source list, newest ingest first.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SourcePage {
    pub sources: Vec<SourceSummary>,
    pub total: usize,
    pub next_cursor: Option<String>,
}

/// `GET /v1/banks/{bank}/sources/{source}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SourceDetail {
    pub id: Uuid,
    pub kind: SourceKind,
    pub document_id: Option<String>,
    pub session_id: Option<String>,
    pub message_at: Option<Timestamp>,
    pub observed_at: Timestamp,
    pub ingested_at: Timestamp,
    pub reference_date: Option<String>,
    pub timezone: String,
    pub platform: Option<String>,
    pub author_name: Option<String>,
    /// The stored text: a turn's user message, or the document.
    pub text: Option<String>,
    /// A turn's assistant reply.
    pub reply: Option<String>,
    pub gone: Option<Gone>,
    pub secret_kinds: Vec<String>,
    /// Every version of the document, this one included, oldest first.
    /// Empty for a turn.
    pub versions: Vec<SourceVersion>,
    pub chunks: Vec<ChunkView>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SourceVersion {
    pub id: Uuid,
    pub ingested_at: Timestamp,
    pub gone: Option<Gone>,
}

/// Where a chunk is in extraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ChunkState {
    Queued,
    InFlight,
    Extracted,
    /// It reached the retry cap; `chunks/retry` puts it back.
    Failed,
    /// It left the queue unextracted: its source was removed.
    Dropped,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ChunkView {
    pub id: Uuid,
    pub position: i64,
    pub heading_path: Option<String>,
    /// Character offsets into the source text.
    pub start: i64,
    pub end: i64,
    pub state: ChunkState,
    pub error_count: u32,
    pub error_kind: Option<String>,
    pub failed_at: Option<Timestamp>,
    /// The visible memories resting on it, in rowid order.
    pub memories: Vec<Uuid>,
    /// The visible memories it mentioned again.
    pub mentions: Vec<Uuid>,
}

/// `GET /v1/banks`: one bank and its counts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BankOverview {
    pub name: String,
    pub owner_name: Option<String>,
    pub assistant_name: Option<String>,
    pub timezone: String,
    pub created_at: Timestamp,
    pub turns: i64,
    pub last_turn_at: Option<Timestamp>,
    pub memories: StatusCounts,
    pub kept: usize,
    /// Memories by kind, forgetting ones left out.
    pub kinds: BTreeMap<String, usize>,
    /// Memories by effective significance, forgetting ones left out.
    pub significance: BTreeMap<String, usize>,
    pub sources: SourceCounts,
    pub chunks: ChunkCounts,
    pub models: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
pub struct SourceCounts {
    pub turns: usize,
    /// Distinct document ids that still have a version not removed.
    pub documents: usize,
    pub document_versions: usize,
    /// Sources whose text is gone: swept, removed, or never stored.
    pub tombstoned: usize,
    pub removed: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
pub struct ChunkCounts {
    pub queued: usize,
    pub failed: usize,
}

/// The bank's memory rows, before strength.
struct Row {
    id: i64,
    uuid: Uuid,
    sentence: String,
    kind: String,
    significance: String,
    owner: Option<String>,
    observed_at: Timestamp,
    created_at: Timestamp,
    hidden: bool,
    invalidated: bool,
    superseded_by: Option<Uuid>,
    ended_by: Option<Uuid>,
    valid_until: Option<crate::strength::WorldTime>,
    timezone: String,
    source: Uuid,
    source_kind: SourceKind,
    document_id: Option<String>,
    session_id: Option<String>,
}

impl Row {
    fn status(&self, now: Timestamp) -> MemoryStatus {
        status(
            self.hidden,
            self.invalidated,
            self.superseded_by.is_some(),
            self.ended_by.is_some(),
            self.valid_until,
            &self.timezone,
            now,
        )
    }
}

/// The first status that holds, in [`MemoryStatus`]'s order.
fn status(
    hidden: bool,
    invalidated: bool,
    superseded: bool,
    ended: bool,
    valid_until: Option<crate::strength::WorldTime>,
    timezone: &str,
    now: Timestamp,
) -> MemoryStatus {
    let tz = TimeZone::get(timezone).unwrap_or(TimeZone::UTC);
    if hidden {
        MemoryStatus::Forgetting
    } else if invalidated {
        MemoryStatus::Retracted
    } else if superseded {
        MemoryStatus::Superseded
    } else if ended || has_ended(valid_until, &tz, now) {
        MemoryStatus::Ended
    } else {
        MemoryStatus::Live
    }
}

fn source_kind(text: &str) -> SourceKind {
    if text == "turn" {
        SourceKind::Turn
    } else {
        SourceKind::Document
    }
}

fn source_kind_text(kind: SourceKind) -> &'static str {
    match kind {
        SourceKind::Turn => "turn",
        SourceKind::Document => "document",
    }
}

/// The page `cursor` and `limit` ask for: its offset and size.
fn page(cursor: Option<&str>, limit: Option<usize>) -> Result<(usize, usize), InspectError> {
    let offset = match cursor.map(str::trim).filter(|cursor| !cursor.is_empty()) {
        Some(cursor) => cursor.parse().map_err(|_| InspectError::InvalidQuery {
            reason: "the cursor isn't one a page gave",
        })?,
        None => 0,
    };
    Ok((offset, limit.unwrap_or(DEFAULT_LIST).clamp(1, MAX_LIST)))
}

fn next_cursor(offset: usize, size: usize, total: usize) -> Option<String> {
    (offset + size < total).then(|| (offset + size).to_string())
}

/// The FTS5 query for `text`: every word, each as a quoted prefix.
fn match_all(text: &str) -> Option<String> {
    let words: Vec<String> = text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(|word| format!("\"{}\"*", word.to_lowercase()))
        .collect();
    (!words.is_empty()).then(|| words.join(" "))
}

/// The memory list.
pub(crate) fn memories(
    store: &Store,
    tuning: &Tuning,
    pause: &PurgePause,
    bank: &str,
    query: &MemoryQuery,
) -> Result<MemoryPage, InspectError> {
    let now = store.now();
    let (offset, size) = page(query.cursor.as_deref(), query.limit)?;
    let conn = store.connection();
    let (bank_id, _) = find_bank(&conn, bank)?.ok_or(InspectError::UnknownBank)?;

    let mut filters = vec!["m.bank_id = ?".to_string()];
    let mut params: Vec<Sql> = vec![Sql::Integer(bank_id)];
    if let Some(kind) = query.kind.as_deref().map(str::trim) {
        if memory_kind(kind).is_none() {
            return Err(InspectError::InvalidQuery {
                reason: "unknown kind; give fact, event, state, task or recurring",
            });
        }
        filters.push("m.kind = ?".into());
        params.push(Sql::Text(kind.into()));
    }
    if let Some(text) = query.q.as_deref() {
        match match_all(text) {
            Some(search) => {
                filters.push(
                    "m.id IN (SELECT rowid FROM memories_fts WHERE memories_fts MATCH ?)".into(),
                );
                params.push(Sql::Text(search));
            }
            None if !text.trim().is_empty() => {
                return Err(InspectError::InvalidQuery {
                    reason: "the search has no words in it",
                });
            }
            None => {}
        }
    }
    if let Some(entity) = query.entity.as_deref() {
        let found = resolve(&conn, bank_id, entity)?;
        filters.push(
            "m.id IN (SELECT me.memory_id FROM memory_entities me
                      JOIN entities e ON e.id = me.entity_id
                      WHERE e.id = ? OR e.merged_into = ?)"
                .into(),
        );
        params.push(Sql::Integer(found.id));
        params.push(Sql::Integer(found.id));
    }
    if let Some(level) = query.significance.as_deref().map(str::trim) {
        if !LEVELS.contains(&level) {
            return Err(InspectError::InvalidQuery {
                reason: "unknown significance; give trivial, minor, notable, major, critical or kept",
            });
        }
        filters.push("COALESCE(m.owner_significance, m.significance) = ?".into());
        params.push(Sql::Text(level.into()));
    }
    match query.kept {
        Some(true) => filters.push("m.owner_significance IS 'kept'".into()),
        Some(false) => filters.push("m.owner_significance IS NOT 'kept'".into()),
        None => {}
    }
    if let Some(kind) = query.source_kind {
        filters.push("s.kind = ?".into());
        params.push(Sql::Text(source_kind_text(kind).into()));
    }
    if let Some(document) = &query.document_id {
        filters.push("s.document_id = ?".into());
        params.push(Sql::Text(document.clone()));
    }
    if let Some(session) = &query.session_id {
        filters.push("s.session_id = ?".into());
        params.push(Sql::Text(session.clone()));
    }
    if let Some(from) = query.observed_from {
        filters.push("m.observed_at >= ?".into());
        params.push(Sql::Integer(crate::store::micros(from)));
    }
    if let Some(to) = query.observed_to {
        filters.push("m.observed_at < ?".into());
        params.push(Sql::Integer(crate::store::micros(to)));
    }

    let rows: Vec<Row> = {
        let mut statement = conn.prepare(&format!(
            "SELECT m.id, m.uuid, m.content, m.kind, m.significance, m.owner_significance,
                    m.observed_at, m.created_at, m.hidden_at IS NOT NULL,
                    m.invalidated_at IS NOT NULL, n.uuid, e.uuid, m.valid_until,
                    m.valid_until_precision, s.timezone, s.uuid, s.kind, s.document_id,
                    s.session_id
             FROM memories m
             JOIN chunks c ON c.id = m.chunk_id JOIN sources s ON s.id = c.source_id
             LEFT JOIN memories n ON n.id = m.superseded_by
             LEFT JOIN memories e ON e.id = m.ended_by
             WHERE {}
             ORDER BY m.id",
            filters.join(" AND ")
        ))?;
        statement
            .query_map(rusqlite::params_from_iter(params), |row| {
                Ok(Row {
                    id: row.get(0)?,
                    uuid: parse(&row.get::<_, String>(1)?),
                    sentence: row.get(2)?,
                    kind: row.get(3)?,
                    significance: row.get(4)?,
                    owner: row.get(5)?,
                    observed_at: timestamp(row.get(6)?),
                    created_at: timestamp(row.get(7)?),
                    hidden: row.get(8)?,
                    invalidated: row.get(9)?,
                    superseded_by: row.get::<_, Option<String>>(10)?.map(|u| parse(&u)),
                    ended_by: row.get::<_, Option<String>>(11)?.map(|u| parse(&u)),
                    valid_until: world_time(row.get(12)?, row.get(13)?),
                    timezone: row.get(14)?,
                    source: parse(&row.get::<_, String>(15)?),
                    source_kind: source_kind(&row.get::<_, String>(16)?),
                    document_id: row.get(17)?,
                    session_id: row.get(18)?,
                })
            })?
            .collect::<Result<_, _>>()?
    };

    // Status and phase are cheap, so every row gets them.
    let mut candidates: Vec<(Row, MemoryStatus, Option<Phase>)> = Vec::new();
    for row in rows {
        let phase = window(&conn, row.id)?.map(|(window, tz)| window.phase(&tz, now));
        if query.phase != PhaseFilter::Any && !phase.is_some_and(|phase| query.phase.admits(phase))
        {
            continue;
        }
        let status = row.status(now);
        candidates.push((row, status, phase));
    }
    let wanted = |status: MemoryStatus| query.status.is_none_or(|wanted| wanted == status);

    // Strength and the projections cost a few queries a row, so they're
    // taken for every match only when the sort or the fade filter needs
    // them, and otherwise for the page alone.
    let loader = StrengthLoader::new(&conn, bank_id, tuning, now)?;
    let summarise = |(row, status, phase): (Row, MemoryStatus, Option<Phase>)| {
        let strength = loader.strength(&conn, row.id)?.value;
        let Outlook { fade, purge, .. } = outlook(&conn, tuning, pause, &loader, row.id, now)?;
        let effective = row
            .owner
            .clone()
            .unwrap_or_else(|| row.significance.clone());
        Ok::<_, rusqlite::Error>(MemorySummary {
            id: row.uuid,
            sentence: row.sentence,
            kind: row.kind,
            phase,
            status,
            significance: SignificanceView {
                value: significance_value(&effective, &tuning.strength.significance),
                extracted: row.significance,
                owner: row.owner,
                effective,
            },
            observed_at: row.observed_at,
            created_at: row.created_at,
            source: row.source,
            source_kind: row.source_kind,
            document_id: row.document_id,
            session_id: row.session_id,
            superseded_by: row.superseded_by,
            ended_by: row.ended_by,
            strength,
            recallable: strength >= TAU,
            fade,
            purge,
        })
    };

    // The status counts cover every filter but `status`. The fade filter
    // needs each candidate's projection before they can be counted, so it
    // takes them for every status; without it, the projections are taken
    // for the matches only when the sort needs them, and otherwise for the
    // page alone.
    let mut statuses = StatusCounts::default();
    let (memories, total) = if let Some(fading) = query.fading {
        let mut all = candidates
            .into_iter()
            .map(summarise)
            .collect::<Result<Vec<_>, _>>()?;
        all.retain(|memory| fades(memory, fading));
        for memory in &all {
            statuses.add(memory.status);
        }
        all.retain(|memory| wanted(memory.status));
        sort(&mut all, query.sort);
        let total = all.len();
        (all.into_iter().skip(offset).take(size).collect(), total)
    } else {
        for (_, status, _) in &candidates {
            statuses.add(*status);
        }
        let mut matched: Vec<_> = candidates
            .into_iter()
            .filter(|(_, status, _)| wanted(*status))
            .collect();
        let total = matched.len();
        let memories = if query.sort == MemorySort::Created {
            // Newest first: the rows came oldest first, rowid breaking ties.
            matched.sort_by(|a, b| {
                b.0.created_at
                    .cmp(&a.0.created_at)
                    .then(b.0.id.cmp(&a.0.id))
            });
            matched
                .into_iter()
                .skip(offset)
                .take(size)
                .map(summarise)
                .collect::<Result<Vec<_>, _>>()?
        } else {
            let mut all = matched
                .into_iter()
                .map(summarise)
                .collect::<Result<Vec<_>, _>>()?;
            sort(&mut all, query.sort);
            all.into_iter().skip(offset).take(size).collect()
        };
        (memories, total)
    };
    Ok(MemoryPage {
        next_cursor: next_cursor(offset, memories.len(), total),
        memories,
        total,
        statuses,
        as_of: now,
    })
}

/// Puts summarised rows in `sort`'s order.
fn sort(all: &mut [MemorySummary], sort: MemorySort) {
    match sort {
        MemorySort::Created => all.sort_by_key(|memory| std::cmp::Reverse(memory.created_at)),
        MemorySort::Fade => all.sort_by(|a, b| {
            let days = |memory: &MemorySummary| {
                memory
                    .fade
                    .as_ref()
                    .map_or(f64::INFINITY, |fade| fade.bank_days)
            };
            days(a).total_cmp(&days(b))
        }),
        MemorySort::Strength => all.sort_by(|a, b| b.strength.total_cmp(&a.strength)),
    }
}

fn fades(memory: &MemorySummary, fading: Fading) -> bool {
    let within = |days: f64| {
        memory.recallable
            && memory
                .fade
                .as_ref()
                .is_some_and(|fade| fade.bank_days <= days)
    };
    match fading {
        Fading::Faded => !memory.recallable,
        Fading::Week => within(7.0),
        Fading::Month => within(30.0),
        Fading::Never => memory.fade.is_none(),
    }
}

/// A source's row, and why its text is gone.
struct SourceRow {
    id: i64,
    uuid: Uuid,
    kind: SourceKind,
    document_id: Option<String>,
    session_id: Option<String>,
    message_at: Option<Timestamp>,
    observed_at: Timestamp,
    ingested_at: Timestamp,
    gone: Option<Gone>,
    secret_kinds: Vec<String>,
}

const SOURCE_COLUMNS: &str = "s.id, s.uuid, s.kind, s.document_id, s.session_id, s.message_at,
     s.observed_at, s.ingested_at, s.text IS NULL, s.tombstoned_at, s.tombstone_reason,
     s.removed_at IS NOT NULL, s.secret_kinds";

fn source_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<SourceRow> {
    let text_gone: bool = row.get(8)?;
    let tombstoned_at = row.get::<_, Option<i64>>(9)?.map(timestamp);
    let reason: Option<String> = row.get(10)?;
    let removed: bool = row.get(11)?;
    let gone = if removed {
        Some(Gone::Removed)
    } else if reason.as_deref() == Some("forget_requested") {
        Some(Gone::ForgetRequested)
    } else if text_gone {
        Some(Gone::Swept { at: tombstoned_at })
    } else {
        None
    };
    Ok(SourceRow {
        id: row.get(0)?,
        uuid: parse(&row.get::<_, String>(1)?),
        kind: source_kind(&row.get::<_, String>(2)?),
        document_id: row.get(3)?,
        session_id: row.get(4)?,
        message_at: row.get::<_, Option<i64>>(5)?.map(timestamp),
        observed_at: timestamp(row.get(6)?),
        ingested_at: timestamp(row.get(7)?),
        gone,
        secret_kinds: row
            .get::<_, Option<String>>(12)?
            .and_then(|kinds| serde_json::from_str(&kinds).ok())
            .unwrap_or_default(),
    })
}

/// What the source detail shows beyond its list row.
struct Stored {
    reference_date: Option<String>,
    timezone: String,
    platform: Option<String>,
    author_name: Option<String>,
    text: Option<String>,
    reply: Option<String>,
}

fn count(conn: &Connection, sql: &str, id: i64) -> Result<usize, rusqlite::Error> {
    let n: i64 = conn.query_row(sql, [id], |row| row.get(0))?;
    Ok(usize::try_from(n).unwrap_or(0))
}

/// The source list.
pub(crate) fn sources(
    store: &Store,
    bank: &str,
    query: &SourceQuery,
) -> Result<SourcePage, InspectError> {
    let (offset, size) = page(query.cursor.as_deref(), query.limit)?;
    let conn = store.connection();
    let (bank_id, _) = find_bank(&conn, bank)?.ok_or(InspectError::UnknownBank)?;
    let mut filters = vec!["s.bank_id = ?".to_string()];
    let mut params: Vec<Sql> = vec![Sql::Integer(bank_id)];
    if let Some(kind) = query.kind {
        filters.push("s.kind = ?".into());
        params.push(Sql::Text(source_kind_text(kind).into()));
    }
    if let Some(document) = &query.document_id {
        filters.push("s.document_id = ?".into());
        params.push(Sql::Text(document.clone()));
    }
    if let Some(session) = &query.session_id {
        filters.push("s.session_id = ?".into());
        params.push(Sql::Text(session.clone()));
    }
    if let Some(text) = query.q.as_deref() {
        match match_all(text) {
            Some(search) => {
                filters.push(
                    "s.id IN (SELECT rowid FROM sources_fts WHERE sources_fts MATCH ?)".into(),
                );
                params.push(Sql::Text(search));
            }
            None if !text.trim().is_empty() => {
                return Err(InspectError::InvalidQuery {
                    reason: "the search has no words in it",
                });
            }
            None => {}
        }
    }
    if let Some(gone) = query.gone {
        let test = "(s.text IS NULL OR s.removed_at IS NOT NULL)";
        filters.push(if gone {
            test.into()
        } else {
            format!("NOT {test}")
        });
    }
    let filter = filters.join(" AND ");
    let total: i64 = conn.query_row(
        &format!("SELECT COUNT(*) FROM sources s WHERE {filter}"),
        rusqlite::params_from_iter(params.iter()),
        |row| row.get(0),
    )?;
    params.push(Sql::Integer(size as i64));
    params.push(Sql::Integer(offset as i64));
    let rows: Vec<SourceRow> = {
        let mut statement = conn.prepare(&format!(
            "SELECT {SOURCE_COLUMNS} FROM sources s WHERE {filter}
             ORDER BY s.ingested_at DESC, s.id DESC LIMIT ? OFFSET ?"
        ))?;
        statement
            .query_map(rusqlite::params_from_iter(params), source_row)?
            .collect::<Result<_, _>>()?
    };
    let mut sources = Vec::with_capacity(rows.len());
    for row in rows {
        sources.push(SourceSummary {
            chunks: count(
                &conn,
                "SELECT COUNT(*) FROM chunks WHERE source_id = ?1",
                row.id,
            )?,
            queued: count(
                &conn,
                "SELECT COUNT(*) FROM extraction_queue q JOIN chunks c ON c.id = q.chunk_id
                 WHERE c.source_id = ?1",
                row.id,
            )?,
            failed: count(
                &conn,
                "SELECT COUNT(*) FROM chunks WHERE source_id = ?1 AND failed_at IS NOT NULL",
                row.id,
            )?,
            memories: count(
                &conn,
                "SELECT COUNT(*) FROM memories m JOIN chunks c ON c.id = m.chunk_id
                 WHERE c.source_id = ?1 AND m.hidden_at IS NULL",
                row.id,
            )?,
            id: row.uuid,
            kind: row.kind,
            document_id: row.document_id,
            session_id: row.session_id,
            message_at: row.message_at,
            observed_at: row.observed_at,
            ingested_at: row.ingested_at,
            gone: row.gone,
            secret_kinds: row.secret_kinds,
        });
    }
    let total = usize::try_from(total).unwrap_or(0);
    Ok(SourcePage {
        next_cursor: next_cursor(offset, sources.len(), total),
        sources,
        total,
    })
}

/// One source, its text, its versions and its chunks. `out` is the queue
/// rows the bank has out on leases now.
pub(crate) fn source(
    store: &Store,
    out: impl Fn(i64) -> BTreeSet<i64>,
    bank: &str,
    id: &str,
) -> Result<SourceDetail, InspectError> {
    let conn = store.connection();
    let (bank_id, _) = find_bank(&conn, bank)?.ok_or(InspectError::UnknownBank)?;
    let uuid = id
        .trim()
        .parse::<Uuid>()
        .map_err(|_| InspectError::UnknownSource)?;
    let row = conn
        .query_row(
            &format!("SELECT {SOURCE_COLUMNS} FROM sources s WHERE s.uuid = ?1 AND s.bank_id = ?2"),
            (uuid.to_string(), bank_id),
            source_row,
        )
        .optional()?
        .ok_or(InspectError::UnknownSource)?;
    let stored = conn.query_row(
        "SELECT reference_date, timezone, platform, author_name, text, reply
         FROM sources WHERE id = ?1",
        [row.id],
        |row| {
            Ok(Stored {
                reference_date: row.get(0)?,
                timezone: row.get(1)?,
                platform: row.get(2)?,
                author_name: row.get(3)?,
                text: row.get(4)?,
                reply: row.get(5)?,
            })
        },
    )?;

    let versions = match &row.document_id {
        Some(document) => {
            let mut statement = conn.prepare(&format!(
                "SELECT {SOURCE_COLUMNS} FROM sources s
                 WHERE s.bank_id = ?1 AND s.kind = 'document' AND s.document_id = ?2
                 ORDER BY s.ingested_at, s.id"
            ))?;
            statement
                .query_map((bank_id, document), source_row)?
                .map(|version| {
                    version.map(|version| SourceVersion {
                        id: version.uuid,
                        ingested_at: version.ingested_at,
                        gone: version.gone,
                    })
                })
                .collect::<Result<_, _>>()?
        }
        None => Vec::new(),
    };

    let out = out(bank_id);
    let chunks = {
        let mut statement = conn.prepare(
            "SELECT c.id, c.uuid, c.position, c.heading_path, c.start_offset, c.end_offset,
                    c.extracted_at IS NOT NULL, c.error_count, c.last_error_kind, c.failed_at,
                    (SELECT q.id FROM extraction_queue q WHERE q.chunk_id = c.id)
             FROM chunks c WHERE c.source_id = ?1 ORDER BY c.position",
        )?;
        let rows: Vec<(i64, ChunkView)> = statement
            .query_map([row.id], |chunk| {
                let extracted: bool = chunk.get(6)?;
                let failed_at = chunk.get::<_, Option<i64>>(9)?.map(timestamp);
                let queued: Option<i64> = chunk.get(10)?;
                let state = match queued {
                    Some(queue_id) if out.contains(&queue_id) => ChunkState::InFlight,
                    Some(_) => ChunkState::Queued,
                    None if failed_at.is_some() => ChunkState::Failed,
                    None if extracted => ChunkState::Extracted,
                    None => ChunkState::Dropped,
                };
                Ok((
                    chunk.get(0)?,
                    ChunkView {
                        id: parse(&chunk.get::<_, String>(1)?),
                        position: chunk.get(2)?,
                        heading_path: chunk.get(3)?,
                        start: chunk.get(4)?,
                        end: chunk.get(5)?,
                        state,
                        error_count: u32::try_from(chunk.get::<_, i64>(7)?).unwrap_or(u32::MAX),
                        error_kind: chunk.get(8)?,
                        failed_at,
                        memories: Vec::new(),
                        mentions: Vec::new(),
                    },
                ))
            })?
            .collect::<Result<_, _>>()?;
        let mut resting = conn.prepare(
            "SELECT uuid FROM memories WHERE chunk_id = ?1 AND hidden_at IS NULL ORDER BY id",
        )?;
        let mut mentioned = conn.prepare(
            "SELECT DISTINCT m.uuid, m.id FROM mention_passages p
             JOIN memories m ON m.id = p.memory_id
             WHERE p.chunk_id = ?1 AND m.hidden_at IS NULL ORDER BY m.id",
        )?;
        let mut chunks = Vec::with_capacity(rows.len());
        for (chunk_id, mut chunk) in rows {
            chunk.memories = resting
                .query_map([chunk_id], |row| row.get::<_, String>(0))?
                .map(|uuid| uuid.map(|u| parse(&u)))
                .collect::<Result<_, _>>()?;
            chunk.mentions = mentioned
                .query_map([chunk_id], |row| row.get::<_, String>(0))?
                .map(|uuid| uuid.map(|u| parse(&u)))
                .collect::<Result<_, _>>()?;
            chunks.push(chunk);
        }
        chunks
    };

    Ok(SourceDetail {
        id: row.uuid,
        kind: row.kind,
        document_id: row.document_id,
        session_id: row.session_id,
        message_at: row.message_at,
        observed_at: row.observed_at,
        ingested_at: row.ingested_at,
        reference_date: stored.reference_date,
        timezone: stored.timezone,
        platform: stored.platform,
        author_name: stored.author_name,
        text: stored.text,
        reply: stored.reply,
        gone: row.gone,
        secret_kinds: row.secret_kinds,
        versions,
        chunks,
    })
}

/// Every bank and its counts, in creation order.
pub(crate) fn banks(store: &Store) -> Result<Vec<BankOverview>, InspectError> {
    let now = store.now();
    let conn = store.connection();
    #[allow(clippy::type_complexity)]
    let banks: Vec<(
        i64,
        String,
        Option<String>,
        Option<String>,
        String,
        i64,
        i64,
        Option<i64>,
    )> = {
        let mut statement = conn.prepare(
            "SELECT id, name, owner_name, assistant_name, timezone, created_at, turns,
                    last_turn_at
             FROM banks ORDER BY id",
        )?;
        statement
            .query_map([], |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                    row.get(6)?,
                    row.get(7)?,
                ))
            })?
            .collect::<Result<_, _>>()?
    };
    let mut overviews = Vec::with_capacity(banks.len());
    for (bank_id, name, owner_name, assistant_name, timezone, created_at, turns, last_turn_at) in
        banks
    {
        let mut memories = StatusCounts::default();
        let mut kinds: BTreeMap<String, usize> = BTreeMap::new();
        let mut significance: BTreeMap<String, usize> = BTreeMap::new();
        let mut kept = 0;
        {
            let mut statement = conn.prepare(
                "SELECT m.kind, COALESCE(m.owner_significance, m.significance),
                        m.hidden_at IS NOT NULL, m.invalidated_at IS NOT NULL,
                        m.superseded_by IS NOT NULL, m.ended_by IS NOT NULL, m.valid_until,
                        m.valid_until_precision, s.timezone
                 FROM memories m
                 JOIN chunks c ON c.id = m.chunk_id JOIN sources s ON s.id = c.source_id
                 WHERE m.bank_id = ?1",
            )?;
            let mut rows = statement.query([bank_id])?;
            while let Some(row) = rows.next()? {
                let kind: String = row.get(0)?;
                let level: String = row.get(1)?;
                let hidden: bool = row.get(2)?;
                let status = status(
                    hidden,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                    world_time(row.get(6)?, row.get(7)?),
                    &row.get::<_, String>(8)?,
                    now,
                );
                memories.add(status);
                if hidden {
                    continue;
                }
                kept += usize::from(level == "kept");
                *kinds.entry(kind).or_default() += 1;
                *significance.entry(level).or_default() += 1;
            }
        }
        let mut sources = SourceCounts::default();
        {
            let mut statement = conn.prepare(
                "SELECT kind, document_id, tombstoned_at IS NOT NULL, removed_at IS NOT NULL
                 FROM sources WHERE bank_id = ?1",
            )?;
            let mut rows = statement.query([bank_id])?;
            let mut documents = BTreeSet::new();
            while let Some(row) = rows.next()? {
                let kind: String = row.get(0)?;
                let document: Option<String> = row.get(1)?;
                let tombstoned: bool = row.get(2)?;
                let removed: bool = row.get(3)?;
                sources.tombstoned += usize::from(tombstoned);
                sources.removed += usize::from(removed);
                if kind == "turn" {
                    sources.turns += 1;
                } else {
                    sources.document_versions += 1;
                    if !removed && let Some(document) = document {
                        documents.insert(document);
                    }
                }
            }
            sources.documents = documents.len();
        }
        let chunks = ChunkCounts {
            queued: count(
                &conn,
                "SELECT COUNT(*) FROM extraction_queue WHERE bank_id = ?1 AND kind = 'chunk'",
                bank_id,
            )?,
            failed: count(
                &conn,
                "SELECT COUNT(*) FROM chunks WHERE bank_id = ?1 AND failed_at IS NOT NULL",
                bank_id,
            )?,
        };
        overviews.push(BankOverview {
            name,
            owner_name,
            assistant_name,
            timezone,
            created_at: timestamp(created_at),
            turns,
            last_turn_at: last_turn_at.map(timestamp),
            memories,
            kept,
            kinds,
            significance,
            sources,
            chunks,
            models: count(
                &conn,
                "SELECT COUNT(*) FROM mental_models WHERE bank_id = ?1",
                bank_id,
            )?,
        });
    }
    Ok(overviews)
}
