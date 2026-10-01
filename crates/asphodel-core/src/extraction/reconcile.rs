//! Reconciliation: finding each claim's nearest stored memories and
//! turning call 2's labels into a plan for the commit (TIM-92, ADR 0005).
//!
//! **Neighbours.** Each claim is searched two ways within its bank: vector
//! search on its embedding and BM25 over memory content. The two ranked
//! lists are fused by reciprocal rank and cut to [`NEIGHBOURS_PER_CLAIM`].
//! A hit on a superseded memory shows the head of its chain instead, and a
//! chain whose head is retracted shows nothing. Faded, ended and hidden
//! memories are shown: a re-mention strengthens what's there (ADR 0008),
//! and a chunk queued before a forget reconciles against the hidden memory
//! so it's erased with it (ADR 0010). A claim that changes something or
//! asks to be remembered is flagged: its vector hits aren't held to the
//! floor, and the open tasks and current states linked to its entities come
//! on top. The chunk shows at most [`NEIGHBOUR_CAP`] neighbours, each once,
//! filled best rank first across claims.
//!
//! **When call 2 runs.** Only the vector floor for the embedding model
//! decides: a claim whose vector search finds a neighbour at or above it,
//! or any flagged claim. BM25 hits and the entity-linked set only fill out
//! the candidates once call 2 is running anyway. Call 2 never runs with no
//! neighbours to show.
//!
//! **Labels.** Code decides direction from `observed_at`, ties broken by
//! the later `ingested_at` and then the source's rowid, never from the LLM.
//! A newer claim's labels apply as given, except that any label on a
//! neighbour already ended or retracted is rejected. An older claim's
//! `mentioned_again` and `confirmed` still write the access, even on an
//! ended neighbour; its `ends` creates it already ended by the neighbour;
//! its `retracts`, `denies` and `refines` create nothing. A claim left with
//! no labels is new.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use jiff::Timestamp;
use jiff::tz::TimeZone;
use rusqlite::Connection;
use uuid::Uuid;

use super::claims::{Checked, Link, NewMemory, Precision, Stamp, start_of_day};
use super::input::{Unit, phrase};
use super::{
    Call1Input, Call2Input, Label, NEIGHBOUR_CAP, NEIGHBOURS_PER_CLAIM, NeighbourMemory,
    ReconcileClaim,
};
use crate::constants::Significance;
use crate::store::{SqliteVec, VectorError, VectorIndex, timestamp};
use crate::strength::{Chains, Kind, Link as ChainLink};

/// Hits each retriever takes at first. Superseded versions collapse into
/// their chain's head and retracted memories drop out, so when fewer than
/// [`NEIGHBOURS_PER_CLAIM`] distinct memories are left, the retriever asks
/// for twice as many, until it has them or runs out (the TIM-108 review).
const HITS_PER_RETRIEVER: usize = NEIGHBOURS_PER_CLAIM * 4;

/// Reciprocal rank fusion's constant: a hit at rank r scores 1 / (k + r).
/// "Retrieval and ranking" (TIM-93) fuses with the same step, without its
/// strength, phase and confidence weighting, which reconciliation doesn't
/// use (TIM-92).
const RRF_K: f64 = 60.0;

/// A neighbour as the plan needs it.
#[derive(Debug, Clone)]
pub(super) struct Neighbour {
    pub id: i64,
    pub uuid: Uuid,
    pub content: String,
    pub kind: Kind,
    pub observed_at: Timestamp,
    /// Who said it later wins a tie on `observed_at` (TIM-92).
    pub ingested_at: Timestamp,
    pub source_id: i64,
    /// The document it came from, if it came from one.
    pub document_id: Option<String>,
    /// Its source's timezone, which its day is in.
    pub tz: TimeZone,
    pub valid_from: Option<Stamp>,
    pub significance: Significance,
    pub owner_significance: Option<String>,
    /// Already ended by another memory.
    pub ended: bool,
}

/// Call 2's input and the neighbours behind its handles, in the same order.
pub(super) struct Search {
    pub input: Call2Input,
    pub neighbours: Vec<Neighbour>,
}

/// Finds each claim's neighbours and decides whether call 2 runs. `vectors`
/// are the claims' embeddings, one per checked memory.
pub(super) fn search(
    conn: &Connection,
    floor: f64,
    input: &Call1Input,
    unit: &Unit,
    checked: &Checked,
    vectors: &[Vec<f32>],
) -> Result<Option<Search>, rusqlite::Error> {
    if checked.memories.is_empty() {
        return Ok(None);
    }
    let mut chains = Chains::new(&chain_links(conn, unit.bank_id)?);
    let mut loaded: BTreeMap<i64, Option<Neighbour>> = BTreeMap::new();
    let mut runs = false;
    let mut per_claim: Vec<Vec<i64>> = Vec::with_capacity(checked.memories.len());

    for (memory, vector) in checked.memories.iter().zip(vectors) {
        let mut vector_ranked = Vec::new();
        let mut limit = HITS_PER_RETRIEVER;
        loop {
            let hits = nearest(conn, unit.bank_id, vector, limit)?;
            vector_ranked.clear();
            // Hits come nearest first, so once one is below the floor every
            // later one is too.
            let mut below_floor = false;
            for hit in &hits {
                let similarity = 1.0 - f64::from(hit.distance);
                if !memory.flagged && similarity < floor {
                    below_floor = true;
                    break;
                }
                let Some(head) = shown(conn, &mut chains, &mut loaded, hit.memory_id)? else {
                    continue;
                };
                if similarity >= floor {
                    runs = true;
                }
                if !vector_ranked.contains(&head) {
                    vector_ranked.push(head);
                }
            }
            if below_floor || hits.len() < limit || vector_ranked.len() >= NEIGHBOURS_PER_CLAIM {
                break;
            }
            limit *= 2;
        }
        let mut bm25_ranked = Vec::new();
        let mut limit = HITS_PER_RETRIEVER;
        loop {
            let hits = bm25(conn, unit.bank_id, &memory.content, limit)?;
            bm25_ranked.clear();
            for hit in &hits {
                if let Some(head) = shown(conn, &mut chains, &mut loaded, *hit)?
                    && !bm25_ranked.contains(&head)
                {
                    bm25_ranked.push(head);
                }
            }
            if hits.len() < limit || bm25_ranked.len() >= NEIGHBOURS_PER_CLAIM {
                break;
            }
            limit *= 2;
        }
        let mut found = fuse(&[&vector_ranked, &bm25_ranked]);
        found.truncate(NEIGHBOURS_PER_CLAIM);
        if memory.flagged {
            runs = true;
            for id in linked_open(conn, unit.bank_id, memory)? {
                if !found.contains(&id) && shown(conn, &mut chains, &mut loaded, id)?.is_some() {
                    found.push(id);
                }
            }
        }
        per_claim.push(found);
    }
    if !runs {
        return Ok(None);
    }

    // The unit cap, filled best rank first across claims so no claim loses
    // its nearest neighbour to another's fifth.
    let mut order: Vec<i64> = Vec::new();
    let deepest = per_claim.iter().map(Vec::len).max().unwrap_or(0);
    for rank in 0..deepest {
        for found in &per_claim {
            if let Some(&id) = found.get(rank)
                && !order.contains(&id)
                && order.len() < NEIGHBOUR_CAP
            {
                order.push(id);
            }
        }
    }
    if order.is_empty() {
        return Ok(None);
    }

    let handle_of: BTreeMap<i64, String> = order
        .iter()
        .enumerate()
        .map(|(index, id)| (*id, format!("n{}", index + 1)))
        .collect();
    let neighbours: Vec<Neighbour> = order
        .iter()
        .map(|id| loaded[id].clone().expect("a shown neighbour was loaded"))
        .collect();
    let claims = checked
        .memories
        .iter()
        .zip(&per_claim)
        .enumerate()
        .map(|(index, (memory, found))| ReconcileClaim {
            handle: format!("c{}", index + 1),
            claim: memory.claim,
            content: memory.content.clone(),
            observed_at: input.observed_at,
            flagged: memory.flagged,
            neighbours: found
                .iter()
                .filter_map(|id| handle_of.get(id).cloned())
                .collect(),
        })
        .collect();
    let shown_neighbours = neighbours
        .iter()
        .map(|neighbour| NeighbourMemory {
            handle: handle_of[&neighbour.id].clone(),
            memory: neighbour.uuid,
            content: neighbour.content.clone(),
            kind: neighbour.kind,
            observed_at: neighbour.observed_at,
            ended: neighbour.ended,
        })
        .collect();
    Ok(Some(Search {
        input: Call2Input {
            chunk: input.chunk,
            claims,
            neighbours: shown_neighbours,
        },
        neighbours,
    }))
}

fn chain_links(conn: &Connection, bank_id: i64) -> Result<Vec<ChainLink>, rusqlite::Error> {
    let mut statement = conn.prepare_cached(
        "SELECT id, superseded_by, ended_by FROM memories
         WHERE bank_id = ?1 AND superseded_by IS NOT NULL",
    )?;
    statement
        .query_map([bank_id], |row| {
            Ok(ChainLink {
                id: row.get(0)?,
                superseded_by: row.get(1)?,
                ended_by: row.get(2)?,
            })
        })?
        .collect()
}

fn nearest(
    conn: &Connection,
    bank_id: i64,
    vector: &[f32],
    limit: usize,
) -> Result<Vec<crate::store::Neighbour>, rusqlite::Error> {
    SqliteVec
        .nearest(conn, bank_id, vector, limit)
        .map_err(|error| match error {
            VectorError::Sqlite(error) => error,
            other => rusqlite::Error::ToSqlConversionFailure(Box::new(other)),
        })
}

/// BM25 over memory content within the bank, best first: any of the
/// claim's words, each as a quoted phrase so nothing is read as syntax.
fn bm25(
    conn: &Connection,
    bank_id: i64,
    content: &str,
    limit: usize,
) -> Result<Vec<i64>, rusqlite::Error> {
    let words: BTreeSet<String> = content
        .split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_lowercase)
        .collect();
    if words.is_empty() {
        return Ok(Vec::new());
    }
    let query = words
        .iter()
        .map(|word| phrase(word))
        .collect::<Vec<_>>()
        .join(" OR ");
    let mut statement = conn.prepare_cached(
        "SELECT m.id FROM memories_fts f JOIN memories m ON m.id = f.rowid
         WHERE memories_fts MATCH ?1 AND m.bank_id = ?2
         ORDER BY f.rank, m.id
         LIMIT ?3",
    )?;
    statement
        .query_map(
            (query, bank_id, i64::try_from(limit).unwrap_or(i64::MAX)),
            |row| row.get(0),
        )?
        .collect()
}

/// Reciprocal rank fusion of ranked lists, best first, each id once. Ties
/// go to the lower rowid, so the order is stable.
fn fuse(lists: &[&Vec<i64>]) -> Vec<i64> {
    let mut scores: BTreeMap<i64, f64> = BTreeMap::new();
    for list in lists {
        let mut seen = BTreeSet::new();
        let mut rank = 0usize;
        for id in list.iter() {
            if !seen.insert(*id) {
                continue;
            }
            rank += 1;
            *scores.entry(*id).or_default() += 1.0 / (RRF_K + rank as f64);
        }
    }
    let mut fused: Vec<(i64, f64)> = scores.into_iter().collect();
    fused.sort_by(|(left_id, left), (right_id, right)| {
        right.total_cmp(left).then(left_id.cmp(right_id))
    });
    fused.into_iter().map(|(id, _)| id).collect()
}

/// The open tasks and current states linked to the claim's known
/// entities, newest first (TIM-92, the flagged claim's wider set).
fn linked_open(
    conn: &Connection,
    bank_id: i64,
    memory: &NewMemory,
) -> Result<Vec<i64>, rusqlite::Error> {
    let mut statement = conn.prepare_cached(
        "SELECT m.id FROM memory_entities me JOIN memories m ON m.id = me.memory_id
         WHERE me.entity_id = ?1 AND m.bank_id = ?2 AND m.kind IN ('task', 'state')
           AND m.valid_until IS NULL AND m.ended_by IS NULL
           AND m.invalidated_at IS NULL AND m.superseded_by IS NULL
         ORDER BY m.observed_at DESC, m.id DESC
         LIMIT ?3",
    )?;
    let mut found = Vec::new();
    for link in &memory.links {
        let Link::Known { entity, .. } = link else {
            continue;
        };
        let ids: Vec<i64> = statement
            .query_map((entity, bank_id, NEIGHBOURS_PER_CLAIM as i64), |row| {
                row.get(0)
            })?
            .collect::<Result<_, _>>()?;
        for id in ids {
            if !found.contains(&id) && found.len() < NEIGHBOURS_PER_CLAIM {
                found.push(id);
            }
        }
    }
    Ok(found)
}

/// The memory a hit on `id` shows: the head of its chain, unless that head
/// is retracted. Loads and caches the head.
fn shown(
    conn: &Connection,
    chains: &mut Chains,
    loaded: &mut BTreeMap<i64, Option<Neighbour>>,
    id: i64,
) -> Result<Option<i64>, rusqlite::Error> {
    let head = chains.head(id);
    if let std::collections::btree_map::Entry::Vacant(entry) = loaded.entry(head) {
        entry.insert(load(conn, head)?);
    }
    Ok(loaded[&head].as_ref().map(|neighbour| neighbour.id))
}

/// A memory as a neighbour, or `None` when it's retracted.
fn load(conn: &Connection, id: i64) -> Result<Option<Neighbour>, rusqlite::Error> {
    let row = conn.query_row(
        "SELECT m.uuid, m.content, m.kind, m.observed_at, m.valid_from, m.valid_from_precision,
                m.significance, m.owner_significance, m.ended_by IS NOT NULL,
                m.invalidated_at IS NOT NULL, s.id, s.ingested_at, s.document_id, s.timezone
         FROM memories m JOIN chunks c ON c.id = m.chunk_id JOIN sources s ON s.id = c.source_id
         WHERE m.id = ?1",
        [id],
        |row| {
            let valid_from = match (
                row.get::<_, Option<i64>>(4)?,
                row.get::<_, Option<String>>(5)?,
            ) {
                (Some(at), Some(precision)) => {
                    Precision::parse(&precision).map(|precision| Stamp {
                        at: timestamp(at),
                        precision,
                    })
                }
                _ => None,
            };
            let retracted: bool = row.get(9)?;
            Ok((
                retracted,
                Neighbour {
                    id,
                    uuid: row
                        .get::<_, String>(0)?
                        .parse()
                        .expect("a stored uuid parses"),
                    content: row.get(1)?,
                    kind: kind(&row.get::<_, String>(2)?),
                    observed_at: timestamp(row.get(3)?),
                    ingested_at: timestamp(row.get(11)?),
                    source_id: row.get(10)?,
                    document_id: row.get(12)?,
                    tz: TimeZone::get(&row.get::<_, String>(13)?).unwrap_or(TimeZone::UTC),
                    valid_from,
                    significance: level(&row.get::<_, String>(6)?),
                    owner_significance: row.get(7)?,
                    ended: row.get(8)?,
                },
            ))
        },
    )?;
    let (retracted, neighbour) = row;
    Ok((!retracted).then_some(neighbour))
}

fn kind(text: &str) -> Kind {
    match text {
        "event" => Kind::Event,
        "state" => Kind::State,
        "task" => Kind::Task,
        "recurring" => Kind::Recurring,
        _ => Kind::Fact,
    }
}

pub(super) fn level(text: &str) -> Significance {
    match text {
        "trivial" => Significance::Trivial,
        "minor" => Significance::Minor,
        "notable" => Significance::Notable,
        "major" => Significance::Major,
        _ => Significance::Critical,
    }
}

/// What becomes of one checked claim.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum Fate {
    /// A new memory.
    New,
    /// A new memory that's already ended: an older claim that a newer
    /// neighbour ends (TIM-92).
    NewEnded {
        by: i64,
        until: Stamp,
        low_confidence: bool,
    },
    /// No memory: the claim is accesses on its neighbours, or an older
    /// retraction or refinement.
    Absorbed,
}

/// What a newer claim does to a neighbour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Edit {
    Ends,
    Retracts,
    /// A retraction that also reopens whatever the neighbour had ended.
    Denies,
    Refines,
}

/// Everything the labels decided, for the commit to write.
#[derive(Debug, Default)]
pub(super) struct Plan {
    /// One per checked claim, in order.
    pub fates: Vec<Fate>,
    /// Edits on neighbours, in label order: the checked claim's index, the
    /// neighbour and the edit.
    pub edits: Vec<(usize, i64, Edit)>,
    /// Accesses on neighbours, the strongest kind each.
    pub accesses: BTreeMap<i64, Label>,
    /// Significance raises from `mentioned_again`, the larger each, for
    /// neighbours whose significance the owner hasn't set.
    pub raises: BTreeMap<i64, Significance>,
    /// Neighbours the owner asked to remember through a mention.
    pub keeps: BTreeSet<i64>,
}

impl Plan {
    /// Every claim new, as when call 2 doesn't run.
    pub fn all_new(claims: usize) -> Self {
        Self {
            fates: vec![Fate::New; claims],
            ..Self::default()
        }
    }
}

/// Turns call 2's labels into a plan. `labels` is by claim handle and
/// neighbour handle, as [`super::call2::parse`] returns it.
pub(super) fn plan(
    search: &Search,
    input: &Call1Input,
    unit: &Unit,
    checked: &Checked,
    labels: &[super::call2::ClaimLabels],
) -> Plan {
    let neighbour_index: BTreeMap<&str, usize> = search
        .input
        .neighbours
        .iter()
        .enumerate()
        .map(|(index, neighbour)| (neighbour.handle.as_str(), index))
        .collect();
    let mut by_claim: BTreeMap<&str, Vec<(usize, Label)>> = BTreeMap::new();
    for (claim, claim_labels) in labels {
        let entry = by_claim.entry(claim.as_str()).or_default();
        for (neighbour, label) in claim_labels {
            if let Some(&index) = neighbour_index.get(neighbour.as_str()) {
                entry.push((index, *label));
            }
        }
    }

    let document_id: Option<String> = unit.document_id.clone();
    let claim_key = (input.observed_at, unit.ingested_at, unit.source_id);
    let mut ended: Vec<bool> = search.neighbours.iter().map(|n| n.ended).collect();
    let mut retracted = vec![false; search.neighbours.len()];
    let mut plan = Plan::default();

    for (index, (memory, claim)) in checked
        .memories
        .iter()
        .zip(&search.input.claims)
        .enumerate()
    {
        let mut changed = false;
        let mut older_nothing = false;
        let mut older_end: Option<usize> = None;
        let mut mentions: Vec<(usize, Label)> = Vec::new();
        for &(n, label) in by_claim.get(claim.handle.as_str()).into_iter().flatten() {
            if retracted[n] {
                continue;
            }
            let neighbour = &search.neighbours[n];
            let newer = claim_key.cmp(&(
                neighbour.observed_at,
                neighbour.ingested_at,
                neighbour.source_id,
            )) != Ordering::Less;
            match label {
                Label::MentionedAgain | Label::Confirmed => {
                    if newer && ended[n] {
                        continue;
                    }
                    mentions.push((n, label));
                }
                Label::Ends | Label::Retracts | Label::Denies | Label::Refines if ended[n] => {}
                Label::Ends if newer => {
                    plan.edits.push((index, neighbour.id, Edit::Ends));
                    ended[n] = true;
                    changed = true;
                }
                Label::Retracts if newer => {
                    plan.edits.push((index, neighbour.id, Edit::Retracts));
                    retracted[n] = true;
                    changed = true;
                }
                Label::Denies if newer => {
                    plan.edits.push((index, neighbour.id, Edit::Denies));
                    retracted[n] = true;
                    changed = true;
                }
                Label::Refines if newer => {
                    plan.edits.push((index, neighbour.id, Edit::Refines));
                    retracted[n] = true;
                    changed = true;
                }
                Label::Ends => {
                    older_end.get_or_insert(n);
                }
                Label::Retracts | Label::Denies | Label::Refines => older_nothing = true,
            }
        }

        for &(n, label) in &mentions {
            let neighbour = &search.neighbours[n];
            // TIM-92: a later version of the same document repeating itself
            // isn't an independent mention.
            let same_document = document_id.is_some() && neighbour.document_id == document_id;
            if same_document {
                continue;
            }
            let kind = plan.accesses.entry(neighbour.id).or_insert(label);
            if label == Label::Confirmed {
                *kind = Label::Confirmed;
            }
            if label == Label::MentionedAgain
                && neighbour.owner_significance.is_none()
                && memory.significance > neighbour.significance
            {
                let raise = plan
                    .raises
                    .entry(neighbour.id)
                    .or_insert(memory.significance);
                *raise = (*raise).max(memory.significance);
            }
        }

        let fate = if changed {
            Fate::New
        } else if older_nothing {
            Fate::Absorbed
        } else if let Some(n) = older_end {
            let neighbour = &search.neighbours[n];
            let (until, low_confidence) =
                end_at(neighbour.valid_from, neighbour.observed_at, &neighbour.tz);
            Fate::NewEnded {
                by: neighbour.id,
                until,
                low_confidence,
            }
        } else if !mentions.is_empty() {
            Fate::Absorbed
        } else {
            Fate::New
        };
        // TIM-92: remember-this goes on the neighbour when the claim is
        // only a mention, and on the new memory otherwise.
        if fate == Fate::Absorbed && memory.kept {
            plan.keeps
                .extend(mentions.iter().map(|&(n, _)| search.neighbours[n].id));
        }
        plan.fates.push(fate);
    }
    plan
}

/// Where a memory ends when another ends it: the ending memory's start, or,
/// with none, the day it was said with low confidence (TIM-92, "in both
/// directions"). The day is the start of that day in the source's timezone,
/// at day precision, as for an event with no stated time.
pub(super) fn end_at(
    valid_from: Option<Stamp>,
    observed_at: Timestamp,
    tz: &TimeZone,
) -> (Stamp, bool) {
    match valid_from {
        Some(stamp) => (stamp, false),
        None => (
            Stamp {
                at: start_of_day(observed_at, tz).unwrap_or(observed_at),
                precision: Precision::Day,
            },
            true,
        ),
    }
}
