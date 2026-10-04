//! The block `system_prompt_block()` returns.
//!
//! It opens with memory usage guidance and the build time, followed by the
//! agenda and every enabled model's entries. Building it costs queries only,
//! never an LLM call, so the plugin's 2 s fetch never waits on a refresh.
//!
//! - **The budget.** The whole text stays within `mental_models.budget`
//!   tokens, which the agenda and every model share. The guidance is
//!   always there. The agenda is laid out first and whole, and folds
//!   only when it and the guidance alone are over: undated tasks, then
//!   routines, least-ranked first, then dated lines in the agenda's fold
//!   order. The models fill what's left, oldest first, each with its
//!   entries in stored order until the next doesn't fit. What the block
//!   lists, cites and keeps by id is only what it rendered.
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
//! - **The fallback.** Every built block is kept by id (`prompt_blocks`),
//!   with what it lists and cites and its rendered entries. A plugin that
//!   got a block without a session id sends the id with its first prefetch,
//!   and the session is mapped to that block then, even if the cache has
//!   rebuilt since. The block's entries are also what a turn's snapshot
//!   takes, so call 1 is shown the entries the session could see.

use std::collections::HashMap;
use std::sync::Mutex;

use jiff::civil::Date;
use jiff::tz::TimeZone;
use jiff::{SignedDuration, Timestamp};
use rusqlite::{Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::agenda::{Agenda, Built};
use crate::config::Tuning;
use crate::mental_models::{load_entries, load_models};
use crate::retrieval::candidates::Cleanup;
use crate::retrieval::estimate_tokens;
use crate::retrieval::format;
use crate::store::strength::world_time;
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

/// A rendered entry as the block held it, with the memories it cites. It's
/// what call 1 is shown, by handle, for a turn in a session holding the
/// block.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockEntry {
    pub entry: Uuid,
    pub text: String,
    pub cites: Vec<Uuid>,
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

/// Memory usage guidance, with the build time. Sessions are frozen, so it says
/// how to reach anything added since.
fn guidance(built_at: &str) -> String {
    format!(
        "## Long-term memory (Asphodel)\n\
         Memories from past conversations are saved automatically, so you never need to save anything. \
         Relevant ones arrive with user messages inside `<memory-context>`. \
         The lists below were built {}, and a memory beats them where they disagree. \
         Before saying you don't know or don't remember something, call memory_recall \
         (with phase \"upcoming\" for plans added since). \
         Use it ahead of session_search unless you need exact wording.",
        built_at
    )
}

fn agenda_heading(date: &str) -> String {
    format!("Agenda for {date}")
}

fn fold_summary(count: usize) -> String {
    format!(
        "- and {count} more dated item{}",
        if count == 1 { "" } else { "s" }
    )
}

/// Reserve the mandatory guidance and the agenda after every dated item folds.
/// English abbreviated weekdays/months have three characters; a two-digit day
/// and the largest representable fold count cover any date and bank contents.
/// Measure the same renderers and separators as the block, not a token constant.
pub(crate) fn minimum_budget() -> usize {
    estimate_tokens(&format!(
        "{}\n\n{}\n{}",
        guidance("Wed 30 Sep 23:59"),
        agenda_heading("Wed 30 Sep"),
        fold_summary(usize::MAX)
    ))
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
    let budget = tuning.mental_models.budget as usize;
    let guidance = guidance(
        &now.to_zoned(tz.clone())
            .strftime("%a %-d %b %H:%M")
            .to_string(),
    );
    // The agenda and every enabled model share the budget, with the guidance
    // always kept. Each try lays the sections out as they'd be rendered
    // and measures the whole text.
    let fits = |sections: &[String]| -> bool {
        let mut text = guidance.clone();
        if !sections.is_empty() {
            text.push_str("\n\n");
            text.push_str(&sections.join("\n\n"));
        }
        estimate_tokens(&text) <= budget
    };

    // The agenda first, whole when it fits: its dated lines are chosen by
    // time so an item can't drop out on the day it matters. Only when the
    // agenda and the guidance alone are over the budget does it fold, the
    // least-ranked undated task first, then the least-ranked routine, then
    // dated lines in the agenda's own fold order.
    let mut shown = Shown {
        dated: vec![true; agenda.dated.len()],
        folded: agenda.agenda.folded,
        routines: agenda.routines.len(),
        undated_tasks: agenda.undated_tasks.len(),
    };
    let mut fold = agenda.dated_fold.iter();
    let mut agenda_section = shown.section(&agenda, now, tz);
    while !agenda_section
        .as_ref()
        .is_none_or(|section| fits(std::slice::from_ref(section)))
    {
        if shown.undated_tasks > 0 {
            shown.undated_tasks -= 1;
        } else if shown.routines > 0 {
            shown.routines -= 1;
        } else if let Some(position) = fold.next() {
            shown.dated[*position] = false;
            shown.folded += 1;
        } else {
            break;
        }
        agenda_section = shown.section(&agenda, now, tz);
    }
    let mut sections: Vec<String> = agenda_section.into_iter().collect();

    // Then each enabled model, oldest first, in what's left: its entries in
    // stored order until the next one doesn't fit. Only what's rendered is
    // recorded as cited, and so put in a session's context.
    let mut cited: Vec<Uuid> = Vec::new();
    let mut entries: Vec<BlockEntry> = Vec::new();
    for model in load_models(&conn, bank_id)?
        .into_iter()
        .filter(|model| model.enabled)
    {
        let mut lines: Vec<String> = Vec::new();
        for entry in load_entries(&conn, model.id)? {
            let Some(line) = entry_line(&conn, tuning, bank_id, now, &entry)? else {
                continue;
            };
            let mut with = lines.clone();
            with.push(line);
            let mut tried = sections.clone();
            tried.push(format!("{}\n{}", model.name, with.join("\n")));
            if !fits(&tried) {
                break;
            }
            lines = with;
            for (_, uuid) in &entry.cites {
                if !cited.contains(uuid) {
                    cited.push(*uuid);
                }
            }
            entries.push(BlockEntry {
                entry: entry.uuid,
                text: entry.text.clone(),
                cites: entry.cites.iter().map(|(_, uuid)| *uuid).collect(),
            });
        }
        // An empty model renders nothing, not even a header.
        if !lines.is_empty() {
            sections.push(format!("{}\n{}", model.name, lines.join("\n")));
        }
    }
    sections.insert(0, guidance);

    let block = Block {
        id: store.new_id(),
        built_at: now,
        text: sections.join("\n\n"),
        agenda: shown.listed(&agenda.agenda),
        cited,
    };
    // Kept by id, for the plugin that sends it with its first prefetch and
    // for the entries a turn's snapshot takes.
    conn.execute(
        "INSERT INTO prompt_blocks (uuid, bank_id, in_context, entries, built_at)
         VALUES (?1, ?2, ?3, ?4, ?5)",
        (
            block.id.to_string(),
            bank_id,
            serde_json::to_string(&block.in_context()).unwrap_or_else(|_| "[]".into()),
            serde_json::to_string(&entries).unwrap_or_else(|_| "[]".into()),
            micros(now),
        ),
    )?;
    Ok(block)
}

/// How much of the agenda the block shows: which dated lines, how many
/// fold into a count, and how many of the routines and undated tasks, best
/// first.
struct Shown {
    dated: Vec<bool>,
    folded: usize,
    routines: usize,
    undated_tasks: usize,
}

impl Shown {
    /// The agenda's section, or `None` when it lists nothing.
    fn section(&self, agenda: &Built, now: Timestamp, tz: &TimeZone) -> Option<String> {
        let mut lines = Vec::new();
        let dated: Vec<&String> = agenda
            .dated
            .iter()
            .zip(&self.dated)
            .filter_map(|(line, shown)| shown.then_some(line))
            .collect();
        if !dated.is_empty() || self.folded > 0 {
            lines.push(agenda_heading(
                &now.to_zoned(tz.clone()).strftime("%a %-d %b").to_string(),
            ));
            lines.extend(dated.into_iter().cloned());
            if self.folded > 0 {
                lines.push(fold_summary(self.folded));
            }
        }
        if self.routines > 0 {
            lines.push("Routines".to_owned());
            lines.extend(agenda.routines[..self.routines].iter().cloned());
        }
        if self.undated_tasks > 0 {
            lines.push("Open tasks".to_owned());
            lines.extend(agenda.undated_tasks[..self.undated_tasks].iter().cloned());
        }
        (!lines.is_empty()).then(|| lines.join("\n"))
    }

    /// The memories the section lists, in the order it lists them.
    fn listed(&self, agenda: &Agenda) -> Vec<Uuid> {
        agenda
            .dated
            .iter()
            .zip(&self.dated)
            .filter_map(|(id, shown)| shown.then_some(*id))
            .chain(agenda.routines.iter().take(self.routines).copied())
            .chain(
                agenda
                    .undated_tasks
                    .iter()
                    .take(self.undated_tasks)
                    .copied(),
            )
            .collect()
    }
}

/// An entry's line, or `None` when any memory it cites is retracted,
/// forgotten, ended or gone.
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
        "SELECT m.invalidated_at IS NULL AND m.hidden_at IS NULL, m.valid_until,
                m.valid_until_precision, s.timezone
         FROM memories m JOIN chunks c ON c.id = m.chunk_id JOIN sources s ON s.id = c.source_id
         WHERE m.id = ?1",
    )?;
    for (memory, _) in &entry.cites {
        let found: Option<(bool, Option<i64>, Option<String>, String)> = statement
            .query_row([memory], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
            })
            .optional()?;
        let Some((shown, until, precision, timezone)) = found else {
            return Ok(None);
        };
        let tz = TimeZone::get(&timezone).unwrap_or(TimeZone::UTC);
        if !shown || crate::agenda::has_ended(world_time(until, precision), &tz, now) {
            return Ok(None);
        }
    }
    let keep_all = |_: &crate::retrieval::candidates::Candidate| true;
    let mut cleanup = Cleanup::new(conn, bank_id, tuning, now, &keep_all)?;
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

/// What the session's block put in context, or `None` when the session has
/// no mapping. An expired mapping is deleted and counts as none.
pub(crate) fn mapped(
    conn: &Connection,
    bank_id: i64,
    session: &str,
    now: Timestamp,
    expiry: SignedDuration,
) -> Result<Option<Vec<Uuid>>, rusqlite::Error> {
    let found: Option<(String, i64)> = conn
        .query_row(
            "SELECT cited, last_turn_at FROM session_blocks
             WHERE bank_id = ?1 AND session_id = ?2",
            (bank_id, session),
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()?;
    let Some((ids, last_turn_at)) = found else {
        return Ok(None);
    };
    if expired(timestamp(last_turn_at), now, expiry) {
        unmap_session(conn, bank_id, session)?;
        return Ok(None);
    }
    Ok(Some(serde_json::from_str(&ids).unwrap_or_default()))
}

/// The block-id fallback: when Hermes gave no session
/// id at `system_prompt_block()` time, the plugin sends the block's id with
/// its first prefetch, and the session is mapped to that block then. Only a
/// block of the same bank counts, an unknown id maps nothing, and a session
/// that already has a mapping keeps it. Returns whether it mapped.
pub(crate) fn map_held_block(
    conn: &Connection,
    bank_id: i64,
    session: &str,
    block: Uuid,
    now: Timestamp,
) -> Result<bool, rusqlite::Error> {
    let mapped = conn.execute(
        "INSERT INTO session_blocks (bank_id, session_id, block_id, cited, built_at, last_turn_at)
         SELECT bank_id, ?2, uuid, in_context, built_at, ?4 FROM prompt_blocks
         WHERE uuid = ?3 AND bank_id = ?1
         ON CONFLICT (bank_id, session_id) DO NOTHING",
        (bank_id, session, block.to_string(), micros(now)),
    )?;
    Ok(mapped > 0)
}

/// The entries of the block the session holds, for the turn's snapshot.
/// Empty when the session has no live mapping or its block is gone.
pub(crate) fn mapped_entries(
    conn: &Connection,
    bank_id: i64,
    session: &str,
    now: Timestamp,
    expiry: SignedDuration,
) -> Result<Vec<BlockEntry>, rusqlite::Error> {
    let found: Option<(String, i64)> = conn
        .query_row(
            "SELECT b.entries, s.last_turn_at FROM session_blocks s
             JOIN prompt_blocks b ON b.uuid = s.block_id AND b.bank_id = s.bank_id
             WHERE s.bank_id = ?1 AND s.session_id = ?2",
            (bank_id, session),
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()?;
    match found {
        Some((entries, last_turn_at)) if !expired(timestamp(last_turn_at), now, expiry) => {
            Ok(serde_json::from_str(&entries).unwrap_or_default())
        }
        _ => Ok(Vec::new()),
    }
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
    let mappings = conn.execute(
        "DELETE FROM session_blocks WHERE last_turn_at <= ?1",
        [micros(cutoff)],
    )?;
    // A block no session holds is only waiting for a prefetch to name it,
    // which a plugin does at once or not at all.
    conn.execute(
        "DELETE FROM prompt_blocks WHERE built_at <= ?1
           AND NOT EXISTS (SELECT 1 FROM session_blocks s
                           WHERE s.bank_id = prompt_blocks.bank_id
                             AND s.block_id = prompt_blocks.uuid)",
        [micros(cutoff)],
    )?;
    Ok(mappings)
}

fn expired(last_turn_at: Timestamp, now: Timestamp, expiry: SignedDuration) -> bool {
    last_turn_at
        .checked_add(expiry)
        .is_ok_and(|expires| now >= expires)
}
