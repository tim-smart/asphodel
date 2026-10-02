//! The corpus: what `asphodel import` writes and real-history `replay` and
//! `bench` read (TIM-96, decision 1). JSON lines under the private dir: a
//! header, then one event per line in time order. Its SHA-256 is the
//! corpus hash every report embeds (decision 6), so the same history
//! always writes the same bytes.
//!
//! The corpus carries the turns' text, so it never leaves
//! `ASPHODEL_REPLAY_DIR`. It never carries the system prompt or
//! `api_content`, which the importer doesn't read (decision 8).

use std::fs;
use std::path::Path;

use anyhow::{Context as _, bail};
use jiff::Timestamp;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::scenario::{Author, ModelSection};
use super::timeline::SessionClass;

/// The format version in the header.
pub const VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Header {
    /// [`VERSION`].
    pub corpus: u32,
    pub bank: String,
    pub timezone: String,
    pub owner: Owner,
    pub assistant: Option<String>,
    /// The manifest's mental models, created in the bank before the first
    /// event. In the header so the corpus hash covers them; left out when
    /// there are none.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub models: Vec<ModelSection>,
    /// The Hermes schema version the importer checked.
    pub hermes_schema_version: i64,
    pub counts: Counts,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Owner {
    pub name: Option<String>,
    #[serde(default)]
    pub platform_ids: Vec<String>,
}

/// What the import found, counts only, so it can be shared.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Counts {
    pub primary_sessions: u64,
    pub cron_sessions: u64,
    pub subagent_sessions_skipped: u64,
    pub turns: u64,
    pub prefetch_only_turns: u64,
    pub compactions: u64,
    pub summary_rows_skipped: u64,
    pub tool_rows_skipped: u64,
    pub inactive_rows_skipped: u64,
    pub multimodal_rows: u64,
    pub memory_blocks_stripped: u64,
    pub image_parts_dropped: u64,
    pub backfills_stripped: u64,
    pub non_owner_turns: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "lowercase")]
pub enum Event {
    /// At the user message's time.
    Prefetch {
        at: Timestamp,
        session: String,
        class: SessionClass,
        query: String,
        previous_query: Option<String>,
        platform: Option<String>,
    },
    /// At the final assistant reply's time; `message_at` names its
    /// prefetch.
    Sync {
        at: Timestamp,
        session: String,
        message_at: Timestamp,
        user: String,
        assistant: String,
        author: Option<Author>,
        platform: Option<String>,
    },
    /// Compaction: the session's in-context set is cleared.
    Clear { at: Timestamp, session: String },
}

impl Event {
    pub fn at(&self) -> Timestamp {
        match self {
            Event::Prefetch { at, .. } | Event::Sync { at, .. } | Event::Clear { at, .. } => *at,
        }
    }
}

#[derive(Debug)]
pub struct Corpus {
    pub header: Header,
    pub events: Vec<Event>,
    /// SHA-256 of the file, hex.
    pub hash: String,
}

/// Reads and checks a corpus file.
pub fn load(path: &Path) -> anyhow::Result<Corpus> {
    let bytes = fs::read(path).with_context(|| format!("reading the corpus {}", path.display()))?;
    let hash = hex(&Sha256::digest(&bytes));
    let text = std::str::from_utf8(&bytes)
        .with_context(|| format!("the corpus {} isn't UTF-8", path.display()))?;
    let mut lines = text.lines().filter(|line| !line.trim().is_empty());
    let Some(first) = lines.next() else {
        bail!("the corpus {} is empty", path.display());
    };
    let header: Header = serde_json::from_str(first)
        .with_context(|| format!("the first line of {} isn't a corpus header", path.display()))?;
    if header.corpus != VERSION {
        bail!(
            "the corpus {} is version {}, and this build reads version {VERSION}",
            path.display(),
            header.corpus
        );
    }
    let mut events = Vec::new();
    for (number, line) in lines.enumerate() {
        let event: Event = serde_json::from_str(line)
            .with_context(|| format!("line {} of {} isn't an event", number + 2, path.display()))?;
        events.push(event);
    }
    Ok(Corpus {
        header,
        events,
        hash,
    })
}

/// Writes the header and the events through a fresh file in the same
/// directory and renames it into place. Fails if the destination is a
/// symlink.
pub fn write(path: &Path, header: &Header, events: &[Event]) -> anyhow::Result<()> {
    let mut out = Vec::new();
    serde_json::to_writer(&mut out, header)?;
    out.push(b'\n');
    for event in events {
        serde_json::to_writer(&mut out, event)?;
        out.push(b'\n');
    }
    super::write_file(path, &out)
}

/// Hex of a digest.
pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// SHA-256 of `text`, hex.
pub fn sha256(text: &str) -> String {
    hex(&Sha256::digest(text.as_bytes()))
}
