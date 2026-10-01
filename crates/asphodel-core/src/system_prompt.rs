//! The block `system_prompt_block()` returns ("Mental models", TIM-95,
//! decisions 4, 6 and 8 and the amendment's decision 2; "API surface and
//! Hermes transport", TIM-94, decision 8, as amended).
//!
//! It holds the agenda, every enabled model's entries, and one pointer line
//! with the build time. Building it costs queries only, never an LLM call,
//! so the plugin's 2 s fetch never waits on a refresh.
//!
//! - **The cache.** One block per bank, in memory, rebuilt lazily on the
//!   next fetch once it's cleared. It's cleared when a model completes a
//!   refresh or is edited, when a memory the agenda would list is written,
//!   ended, retracted, kept or unkept, when any memory is ended or
//!   retracted (an entry citing it would otherwise be served until the
//!   next refresh), and when the bank-local day rolls over.
//! - **Memories win.** An entry is rendered only while every memory it
//!   cites is current: not retracted, forgotten or ended. An entry citing a
//!   state whose confidence is below 0.9 shows its age, as injection does.
//! - **In context.** A fetch with a session id persists which block the
//!   session holds and the memories it lists or cites (`session_blocks`),
//!   and they join the session's in-context set: injection skips them, and
//!   extraction checks a reply against them for `used`. Hermes freezes the
//!   block per session and restores it after a restart without asking
//!   again, so the mapping lives in the store and expires after
//!   `sessions.mapping_expiry_days` without a turn.

use std::collections::HashMap;
use std::sync::Mutex;

use jiff::civil::Date;
use jiff::tz::TimeZone;
use jiff::{SignedDuration, Timestamp};
use rusqlite::{Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::config::Tuning;
use crate::mental_models::{load_entries, load_models};
use crate::retrieval::candidates::Cleanup;
use crate::retrieval::format;
use crate::store::{Store, micros, timestamp};

/// One built block. `id` changes exactly when the content is rebuilt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Block {
    pub id: Uuid,
    pub built_at: Timestamp,
    pub text: String,
    /// The memories the agenda lists, in the order it lists them.
    pub agenda: Vec<Uuid>,
    /// The memories the rendered entries cite, each once.
    pub cited: Vec<Uuid>,
}

impl Block {
    /// Everything the block puts in a session's context.
    pub fn in_context(&self) -> Vec<Uuid> {
        let mut ids = self.agenda.clone();
        for id in &self.cited {
            if !ids.contains(id) {
                ids.push(*id);
            }
        }
        ids
    }
}

/// Each bank's block, with the local date it was built for. Each clear
/// bumps the bank's generation, so a build that raced a clear isn't cached.
#[derive(Debug, Default)]
pub(crate) struct Blocks {
    inner: Mutex<HashMap<i64, Cached>>,
}

#[derive(Debug, Default)]
struct Cached {
    block: Option<(Block, Date)>,
    generation: u64,
}

impl Blocks {
    /// The bank's cached block if it was built for `today`, or the
    /// generation to cache a new one under.
    pub(crate) fn get(&self, bank_id: i64, today: Date) -> Result<Block, u64> {
        let mut blocks = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let cached = blocks.entry(bank_id).or_default();
        match &cached.block {
            Some((block, date)) if *date == today => Ok(block.clone()),
            _ => Err(cached.generation),
        }
    }

    /// Caches `block`, unless the bank was cleared since `generation`.
    pub(crate) fn put(&self, bank_id: i64, today: Date, block: Block, generation: u64) {
        let mut blocks = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let cached = blocks.entry(bank_id).or_default();
        if cached.generation == generation {
            cached.block = Some((block, today));
        }
    }

    /// Clears the bank's block, so the next fetch rebuilds it.
    pub(crate) fn invalidate(&self, bank_id: i64) {
        let mut blocks = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let cached = blocks.entry(bank_id).or_default();
        cached.block = None;
        cached.generation += 1;
    }
}

/// The pointer line, with the build time (TIM-94, decision 8; TIM-95,
/// decision 8 and the amendment). Sessions are frozen, so it says how to
/// reach anything added since.
fn pointer(now: Timestamp, tz: &TimeZone) -> String {
    format!(
        "Built {}; memories win: use memory_recall for history and detail, and for anything \
         added since, call memory_recall with phase upcoming.",
        now.to_zoned(tz.clone()).strftime("%a %-d %b %H:%M")
    )
}

/// Builds the bank's block at `now`.
pub(crate) fn build(
    store: &Store,
    tuning: &Tuning,
    bank_id: i64,
    tz: &TimeZone,
) -> Result<Block, rusqlite::Error> {
    let now = store.now();
    let conn = store.connection();
    let agenda = crate::agenda::build(&conn, tuning, bank_id, tz, now)?;
    let mut sections: Vec<String> = Vec::new();

    let mut lines = Vec::new();
    if !agenda.dated.is_empty() || agenda.agenda.folded > 0 {
        lines.push(format!(
            "Agenda for {}",
            now.to_zoned(tz.clone()).strftime("%a %-d %b")
        ));
        lines.extend(agenda.dated.iter().cloned());
        if agenda.agenda.folded > 0 {
            lines.push(format!(
                "- and {} more dated item{}",
                agenda.agenda.folded,
                if agenda.agenda.folded == 1 { "" } else { "s" }
            ));
        }
    }
    if !agenda.routines.is_empty() {
        lines.push("Routines".to_owned());
        lines.extend(agenda.routines.iter().cloned());
    }
    if !agenda.undated_tasks.is_empty() {
        lines.push("Open tasks".to_owned());
        lines.extend(agenda.undated_tasks.iter().cloned());
    }
    if !lines.is_empty() {
        sections.push(lines.join("\n"));
    }

    let mut cited: Vec<Uuid> = Vec::new();
    for model in load_models(&conn, bank_id)?
        .into_iter()
        .filter(|model| model.enabled)
    {
        let mut lines = Vec::new();
        for entry in load_entries(&conn, model.id)? {
            let Some(line) = entry_line(&conn, tuning, bank_id, now, &entry)? else {
                continue;
            };
            lines.push(line);
            for (_, uuid) in &entry.cites {
                if !cited.contains(uuid) {
                    cited.push(*uuid);
                }
            }
        }
        // An empty model renders nothing, not even a header (decision 5).
        if !lines.is_empty() {
            sections.push(format!("{}\n{}", model.name, lines.join("\n")));
        }
    }
    sections.push(pointer(now, tz));

    Ok(Block {
        id: store.new_id(),
        built_at: now,
        text: sections.join("\n\n"),
        agenda: agenda.agenda.listed(),
        cited,
    })
}

/// An entry's line, or `None` when any memory it cites is retracted,
/// forgotten, ended or gone (decision 6).
fn entry_line(
    conn: &Connection,
    tuning: &Tuning,
    bank_id: i64,
    now: Timestamp,
    entry: &crate::mental_models::StoredEntry,
) -> Result<Option<String>, rusqlite::Error> {
    if entry.cites.is_empty() {
        return Ok(None);
    }
    let mut statement = conn.prepare_cached(
        "SELECT invalidated_at IS NULL AND hidden_at IS NULL AND ended_by IS NULL
                AND (valid_until IS NULL OR valid_until > ?2)
         FROM memories WHERE id = ?1",
    )?;
    for (memory, _) in &entry.cites {
        let current: Option<bool> = statement
            .query_row((memory, micros(now)), |row| row.get(0))
            .optional()?;
        if current != Some(true) {
            return Ok(None);
        }
    }
    let keep_all = |_: &crate::retrieval::candidates::Candidate| true;
    let mut cleanup = Cleanup::new(conn, bank_id, tuning.clock.quiet_rate, now, &keep_all)?;
    let ids: Vec<i64> = entry.cites.iter().map(|(memory, _)| *memory).collect();
    cleanup.list(&ids)?;
    let age = cleanup
        .take(&ids)
        .iter()
        .find_map(|candidate| format::state_age(candidate, now));
    Ok(Some(match age {
        Some(age) => format!("- {} [{age}]", entry.text),
        None => format!("- {}", entry.text),
    }))
}

/// Records that `session` holds `block`, with what it puts in context. A
/// new fetch replaces the mapping and starts its expiry over.
pub(crate) fn map_session(
    conn: &Connection,
    bank_id: i64,
    session: &str,
    block: &Block,
    now: Timestamp,
) -> Result<(), rusqlite::Error> {
    let ids = serde_json::to_string(&block.in_context()).unwrap_or_else(|_| "[]".into());
    conn.execute(
        "INSERT INTO session_blocks (bank_id, session_id, block_id, cited, built_at, last_turn_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)
         ON CONFLICT (bank_id, session_id) DO UPDATE
           SET block_id = excluded.block_id, cited = excluded.cited,
               built_at = excluded.built_at, last_turn_at = excluded.last_turn_at",
        (
            bank_id,
            session,
            block.id.to_string(),
            ids,
            micros(block.built_at),
            micros(now),
        ),
    )?;
    Ok(())
}

/// What the session's block put in context, unless the mapping expired, in
/// which case it's deleted.
pub(crate) fn mapped(
    conn: &Connection,
    bank_id: i64,
    session: &str,
    now: Timestamp,
    expiry: SignedDuration,
) -> Result<Vec<Uuid>, rusqlite::Error> {
    let found: Option<(String, i64)> = conn
        .query_row(
            "SELECT cited, last_turn_at FROM session_blocks
             WHERE bank_id = ?1 AND session_id = ?2",
            (bank_id, session),
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()?;
    let Some((ids, last_turn_at)) = found else {
        return Ok(Vec::new());
    };
    if expired(timestamp(last_turn_at), now, expiry) {
        unmap_session(conn, bank_id, session)?;
        return Ok(Vec::new());
    }
    Ok(serde_json::from_str(&ids).unwrap_or_default())
}

/// A turn arrived in the session: its mapping's expiry starts over.
pub(crate) fn touch_session(
    conn: &Connection,
    bank_id: i64,
    session: &str,
    now: Timestamp,
) -> Result<(), rusqlite::Error> {
    conn.execute(
        "UPDATE session_blocks SET last_turn_at = MAX(last_turn_at, ?3)
         WHERE bank_id = ?1 AND session_id = ?2",
        (bank_id, session, micros(now)),
    )?;
    Ok(())
}

/// Drops the session's mapping, as `on_session_switch` asks.
pub(crate) fn unmap_session(
    conn: &Connection,
    bank_id: i64,
    session: &str,
) -> Result<(), rusqlite::Error> {
    conn.execute(
        "DELETE FROM session_blocks WHERE bank_id = ?1 AND session_id = ?2",
        (bank_id, session),
    )?;
    Ok(())
}

/// Deletes every mapping past its expiry.
pub(crate) fn expire_mappings(
    conn: &Connection,
    now: Timestamp,
    expiry: SignedDuration,
) -> Result<usize, rusqlite::Error> {
    let cutoff = now.checked_sub(expiry).unwrap_or(Timestamp::MIN);
    conn.execute(
        "DELETE FROM session_blocks WHERE last_turn_at <= ?1",
        [micros(cutoff)],
    )
}

fn expired(last_turn_at: Timestamp, now: Timestamp, expiry: SignedDuration) -> bool {
    last_turn_at
        .checked_add(expiry)
        .is_ok_and(|expires| now >= expires)
}
