//! The timeline the engine runs: turns, documents, clears and probes in
//! simulated time, built from a scripted scenario (`docs/replay.md`) or from a
//! corpus `asphodel import` wrote. The engine reads only this, so nothing in it
//! knows whether the history is scripted or real.

use std::collections::BTreeMap;

use jiff::Timestamp;
use serde::{Deserialize, Serialize};

use super::corpus::{Corpus, Event};
use super::scenario::{Author, Claim, Clear, Document, Probe, Scenario, duration};

/// A session's class: cron sessions get prefetch only and are reported apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SessionClass {
    Primary,
    Cron,
}

/// One user message and the assistant's reply. Prefetch runs at `at`,
/// `sync_turn` at `reply_at`, unless the turn is prefetch only.
#[derive(Debug, Clone)]
pub struct Turn {
    pub at: Timestamp,
    pub reply_at: Timestamp,
    pub session: String,
    pub user: String,
    pub assistant: String,
    /// The plugin's previous prefetch query for the session, which a short
    /// follow-up borrows.
    pub previous_query: Option<String>,
    /// The assistant's reply to the previous message, when it came before
    /// this one.
    pub previous_reply: Option<String>,
    pub author: Option<Author>,
    pub platform: Option<String>,
    pub class: SessionClass,
    /// A cron turn, or one Hermes never answered: prefetch runs and
    /// nothing is ingested.
    pub prefetch_only: bool,
    /// Scripted claims; empty for real history.
    pub claims: Vec<Claim>,
    /// Scripted `used` labels; empty for real history.
    pub used: Vec<String>,
}

/// How a probe's `memory` names a memory: a claim label in a scenario, or a
/// regex over sentences in real history.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Matching {
    Label,
    Sentence,
}

#[derive(Debug)]
pub struct Timeline {
    pub turns: Vec<Turn>,
    pub documents: Vec<Document>,
    pub clears: Vec<Clear>,
    pub probes: Vec<Probe>,
    pub matching: Matching,
}

impl Timeline {
    /// A scenario's events, with its chatter expanded into turns.
    pub fn from_scenario(scenario: &Scenario) -> Result<Self, String> {
        let mut turns: Vec<Turn> = scenario
            .turns
            .iter()
            .map(|turn| Turn {
                at: turn.at,
                reply_at: turn.reply_at.unwrap_or(turn.at),
                session: turn.session.clone(),
                user: turn.user.clone(),
                assistant: turn.assistant.clone(),
                previous_query: None,
                previous_reply: None,
                author: turn.author.clone(),
                platform: turn.platform.clone(),
                class: SessionClass::Primary,
                prefetch_only: false,
                claims: turn.claims.clone(),
                used: turn.used.clone(),
            })
            .collect();
        for chatter in &scenario.chatter {
            let every =
                duration(&chatter.every).map_err(|error| format!("chatter interval: {error}"))?;
            let session = chatter.session.clone().unwrap_or_else(|| "chatter".into());
            let mut at = chatter.from;
            for n in 1..=chatter.count {
                turns.push(Turn {
                    at,
                    reply_at: at,
                    session: session.clone(),
                    user: format!("Just checking in ({n})."),
                    assistant: "Hello again.".into(),
                    previous_query: None,
                    previous_reply: None,
                    author: None,
                    platform: None,
                    class: SessionClass::Primary,
                    prefetch_only: false,
                    claims: Vec::new(),
                    used: Vec::new(),
                });
                at = at
                    .checked_add(every)
                    .map_err(|_| "chatter runs past the end of time".to_string())?;
            }
        }
        Ok(Self {
            turns,
            documents: scenario.documents.clone(),
            clears: scenario.clears.clone(),
            probes: scenario.probes.clone(),
            matching: Matching::Label,
        })
    }

    /// A corpus's events: a prefetch and its sync make one turn, a prefetch on
    /// its own a prefetch-only turn, and a clear a clear. A prefetch that has
    /// a previous query takes the reply to the session's previous turn, if
    /// it was synced by then. Probes come from the private probes file.
    pub fn from_corpus(corpus: &Corpus, probes: Vec<Probe>) -> Result<Self, String> {
        let mut turns: Vec<Turn> = Vec::new();
        let mut clears = Vec::new();
        let mut last: BTreeMap<&str, usize> = BTreeMap::new();
        for event in &corpus.events {
            match event {
                Event::Prefetch {
                    at,
                    session,
                    class,
                    query,
                    previous_query,
                    platform,
                } => {
                    let previous_reply = previous_query
                        .as_ref()
                        .and_then(|_| last.get(session.as_str()))
                        .map(|&index| &turns[index])
                        .filter(|turn| !turn.prefetch_only && !turn.assistant.is_empty())
                        .map(|turn| turn.assistant.clone());
                    last.insert(session, turns.len());
                    turns.push(Turn {
                        at: *at,
                        reply_at: *at,
                        session: session.clone(),
                        user: query.clone(),
                        assistant: String::new(),
                        previous_query: previous_query.clone(),
                        previous_reply,
                        author: None,
                        platform: platform.clone(),
                        class: *class,
                        prefetch_only: true,
                        claims: Vec::new(),
                        used: Vec::new(),
                    });
                }
                Event::Sync {
                    at,
                    session,
                    message_at,
                    user,
                    assistant,
                    author,
                    platform,
                } => {
                    let Some(turn) = turns
                        .iter_mut()
                        .rev()
                        .find(|turn| &turn.session == session && turn.at == *message_at)
                    else {
                        return Err(format!(
                            "a sync at {at} in session {session:?} has no prefetch at {message_at}"
                        ));
                    };
                    if !turn.prefetch_only {
                        return Err(format!(
                            "session {session:?} has two syncs for the prefetch at {message_at}"
                        ));
                    }
                    if *at < turn.at {
                        return Err(format!(
                            "a sync at {at} in session {session:?} is before its prefetch at {message_at}"
                        ));
                    }
                    turn.reply_at = *at;
                    turn.user = user.clone();
                    turn.assistant = assistant.clone();
                    turn.author = author.clone();
                    turn.platform = platform.clone();
                    turn.prefetch_only = false;
                }
                Event::Clear { at, session } => clears.push(Clear {
                    at: *at,
                    session: session.clone(),
                }),
            }
        }
        Ok(Self {
            turns,
            documents: Vec::new(),
            clears,
            probes,
            matching: Matching::Sentence,
        })
    }

    /// The earliest event or probe, where the clock starts.
    pub fn earliest(&self) -> Option<Timestamp> {
        self.times().min()
    }

    /// The latest event or probe, where the run ends unless `--until`
    /// extends it.
    pub fn latest(&self) -> Option<Timestamp> {
        self.times().max()
    }

    fn times(&self) -> impl Iterator<Item = Timestamp> + '_ {
        self.turns
            .iter()
            .flat_map(|turn| [turn.at, turn.reply_at])
            .chain(self.documents.iter().map(|document| document.at))
            .chain(self.clears.iter().map(|clear| clear.at))
            .chain(self.probes.iter().map(|probe| probe.at))
    }
}
