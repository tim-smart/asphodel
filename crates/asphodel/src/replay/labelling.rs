//! The labelling material and the precision curve (`docs/replay.md`, "Labelling
//! and the precision curve").
//!
//! The reranker gate floor and the reconcile similarity floor are set from
//! Tim's labels. `asphodel replay --labelling FILE` writes the material to
//! label: recall candidates at [`SAMPLED_TURNS`] turns, scored with the
//! reranker logit the gate compares and including those it turned away, and
//! every candidate list call 2 was shown, scored with cosine similarity. Call
//! 2's lists hold only what call 2 was shown. Flagged claims bypass the vector
//! floor, and BM25 neighbours are not filtered by it, so candidates can score
//! below the reconcile floor. The call-2 curve is precision by score threshold
//! over observed candidates, not a prediction of what a raised or lowered
//! reconcile floor would retain.
//!
//! `asphodel report precision --labels L --material M` reads the labels
//! and prints the curve: at each labelled score, the relevant fraction of
//! the labelled candidates at or above it. Labels are keyed by what they
//! judge, so they score another run's material over the same corpus: a
//! recall label by the cleaned query and the memory, a call 2 label by the
//! claim's chunk and ordinal and the neighbour. A file in the old form, a
//! table of candidate id to relevance, is read against the material it was
//! written for, and `--convert` rewrites it keyed. The curve is numbers
//! only. Every file is history, so all must be inside the private dir, and
//! no error quotes any.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use anyhow::{Context as _, bail};
use asphodel_core::extraction::Call2List;
use asphodel_core::retrieval::ScoredPrefetch;
use jiff::Timestamp;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::cli::PrecisionArgs;

/// How many synced turns the recall material samples.
pub const SAMPLED_TURNS: usize = 50;

/// The material's format version. Version 2 added the claim's chunk and
/// ordinal to call 2's samples.
const VERSION: u32 = 2;

#[derive(Debug, Serialize, Deserialize)]
pub struct Material {
    pub version: u32,
    pub recall: Vec<RecallSample>,
    pub call2: Vec<Call2Sample>,
}

/// A sampled turn's prefetch: the query the reranker scored against, the
/// message it was cleaned from, and its candidates in ranked order.
#[derive(Debug, Serialize, Deserialize)]
pub struct RecallSample {
    pub sample: String,
    pub at: Timestamp,
    pub session: String,
    /// The cleaned query, which calibration uses.
    pub query: String,
    /// The message as Hermes sent it. Material written before it was
    /// recorded has none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_query: Option<String>,
    pub candidates: Vec<Candidate>,
}

/// A claim and the neighbours call 2 was shown for it.
#[derive(Debug, Serialize, Deserialize)]
pub struct Call2Sample {
    pub sample: String,
    pub at: Timestamp,
    /// The claim's chunk and its index in call 1's reply, which call 2
    /// labels are keyed by. Material written before version 2 has neither.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunk: Option<Uuid>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ordinal: Option<usize>,
    pub claim: String,
    pub candidates: Vec<Candidate>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Candidate {
    /// Unique in the material: what a label names.
    pub id: String,
    pub memory: Uuid,
    pub score: f64,
    pub sentence: String,
}

/// A turn's prefetch, before sampling.
struct Turn {
    at: Timestamp,
    session: String,
    query: String,
    raw_query: String,
    candidates: Vec<(Uuid, String, f64)>,
}

/// What a run collects for the material while it runs. Collecting reads
/// what the prefetches and call 2 already computed, so it changes nothing
/// the run simulates.
pub struct Collector {
    /// The queries that already have recall labels, which sampling prefers.
    labelled: BTreeSet<String>,
    turns: Vec<Turn>,
    call2: Vec<(Timestamp, Call2List)>,
}

impl Collector {
    pub fn new(labelled: BTreeSet<String>) -> Self {
        Self {
            labelled,
            turns: Vec::new(),
            call2: Vec::new(),
        }
    }

    /// A synced turn's prefetch. A candidate the reranker didn't score has
    /// no logit to label against and is left out; replay never skips the
    /// reranker, so there are none in practice.
    pub fn turn(&mut self, at: Timestamp, session: &str, scored: &ScoredPrefetch) {
        let shown = &scored.candidates;
        self.turns.push(Turn {
            at,
            session: session.to_owned(),
            query: scored.query.clone(),
            raw_query: scored.raw_query.clone(),
            candidates: shown
                .iter()
                .filter_map(|candidate| {
                    Some((
                        candidate.memory,
                        candidate.sentence.clone(),
                        candidate.logit?,
                    ))
                })
                .collect(),
        });
    }

    /// The lists call 2 was shown for one chunk, at the claim.
    pub fn call2(&mut self, at: Timestamp, lists: Vec<Call2List>) {
        self.call2.extend(lists.into_iter().map(|list| (at, list)));
    }

    /// The material: [`SAMPLED_TURNS`] turns, or every turn when there are
    /// fewer, and every call 2 list, with ids assigned in order. Turns whose
    /// queries already have labels come first, so a re-recorded run reuses
    /// as many labels as it can: spread evenly over the labelled turns when
    /// there are more of them than the sample holds, and otherwise all of
    /// them with the rest spread evenly over the other turns. Without
    /// labels that's an even spread over the run.
    pub fn finish(self) -> Material {
        let (labelled, other): (Vec<usize>, Vec<usize>) = (0..self.turns.len())
            .partition(|&index| self.labelled.contains(&self.turns[index].query));
        let mut picked = spread(&labelled, SAMPLED_TURNS);
        picked.extend(spread(&other, SAMPLED_TURNS - picked.len()));
        let recall = self
            .turns
            .into_iter()
            .enumerate()
            .filter(|(index, _)| picked.contains(index))
            .enumerate()
            .map(|(number, (_, turn))| {
                let sample = format!("r{}", number + 1);
                RecallSample {
                    candidates: candidates(&sample, turn.candidates),
                    sample,
                    at: turn.at,
                    session: turn.session,
                    query: turn.query,
                    raw_query: Some(turn.raw_query),
                }
            })
            .collect();
        let call2 = self
            .call2
            .into_iter()
            .enumerate()
            .map(|(number, (at, list))| {
                let sample = format!("c{}", number + 1);
                let scored = list
                    .candidates
                    .into_iter()
                    .filter_map(|candidate| {
                        Some((candidate.memory, candidate.sentence, candidate.similarity?))
                    })
                    .collect();
                Call2Sample {
                    candidates: candidates(&sample, scored),
                    sample,
                    at,
                    chunk: Some(list.chunk),
                    ordinal: Some(list.ordinal),
                    claim: list.claim,
                }
            })
            .collect();
        Material {
            version: VERSION,
            recall,
            call2,
        }
    }
}

/// `count` of `indices` spread evenly over them in order (index
/// `i × n / count` of `n`), or all of them when there are no more.
fn spread(indices: &[usize], count: usize) -> BTreeSet<usize> {
    let total = indices.len();
    if total <= count {
        return indices.iter().copied().collect();
    }
    (0..count)
        .map(|index| indices[index * total / count])
        .collect()
}

fn candidates(sample: &str, scored: Vec<(Uuid, String, f64)>) -> Vec<Candidate> {
    scored
        .into_iter()
        .enumerate()
        .map(|(index, (memory, sentence, score))| Candidate {
            id: format!("{sample}.{}", index + 1),
            memory,
            score,
            sentence,
        })
        .collect()
}

// Labels.

/// A labels file: keyed by what each label judges, or the old form of
/// candidate ids, which only the material it was written for can read.
enum Labels {
    Keyed(Keyed),
    Ids(BTreeMap<String, bool>),
}

/// Labels keyed by what they judge, so they carry over to another run's
/// material over the same corpus.
#[derive(Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Keyed {
    #[serde(default)]
    recall: Vec<RecallLabel>,
    #[serde(default)]
    call2: Vec<Call2Label>,
}

/// Whether a memory is worth injecting for a query.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RecallLabel {
    /// The cleaned query, as the material's `query`.
    query: String,
    memory: Uuid,
    relevant: bool,
}

/// Whether a neighbour is about the same thing as a claim, named by its
/// chunk and its index in call 1's reply.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Call2Label {
    chunk: Uuid,
    ordinal: usize,
    memory: Uuid,
    relevant: bool,
}

type RecallKey = (String, Uuid);
type Call2Key = (Uuid, usize, Uuid);

/// Keyed labels as lookups. The same judgement given twice is one label;
/// given twice with different answers it's refused, naming the entries by
/// number only.
struct Lookup {
    recall: BTreeMap<RecallKey, bool>,
    call2: BTreeMap<Call2Key, bool>,
}

impl Lookup {
    fn new(keyed: Keyed, path: &Path) -> anyhow::Result<Self> {
        fn insert<K: Ord>(
            map: &mut BTreeMap<K, (usize, bool)>,
            key: K,
            entry: usize,
            relevant: bool,
            list: &str,
            path: &Path,
        ) -> anyhow::Result<()> {
            match map.get(&key) {
                Some(&(first, before)) if before != relevant => bail!(
                    "{} labels the same judgement both ways: {list} labels {first} and {entry}",
                    path.display()
                ),
                Some(_) => Ok(()),
                None => {
                    map.insert(key, (entry, relevant));
                    Ok(())
                }
            }
        }
        let mut recall = BTreeMap::new();
        for (index, label) in keyed.recall.into_iter().enumerate() {
            let key = (label.query, label.memory);
            insert(&mut recall, key, index + 1, label.relevant, "recall", path)?;
        }
        let mut call2 = BTreeMap::new();
        for (index, label) in keyed.call2.into_iter().enumerate() {
            let key = (label.chunk, label.ordinal, label.memory);
            insert(&mut call2, key, index + 1, label.relevant, "call 2", path)?;
        }
        fn answers<K: Ord>(map: BTreeMap<K, (usize, bool)>) -> BTreeMap<K, bool> {
            map.into_iter()
                .map(|(key, (_, relevant))| (key, relevant))
                .collect()
        }
        Ok(Self {
            recall: answers(recall),
            call2: answers(call2),
        })
    }
}

/// The queries `path`'s recall labels judge, which `replay --labels`
/// samples first. Labels of candidate ids name nothing in a new run, so
/// they're refused until converted.
pub fn labelled_queries(path: &Path) -> anyhow::Result<BTreeSet<String>> {
    match load_labels(path)? {
        Labels::Keyed(keyed) => Ok(keyed.recall.into_iter().map(|label| label.query).collect()),
        Labels::Ids(_) => bail!(
            "{} labels candidate ids, which name nothing in a new run; convert it with `asphodel report precision --convert` first",
            path.display()
        ),
    }
}

fn load_labels(path: &Path) -> anyhow::Result<Labels> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let error = |error: toml::de::Error| super::toml_error(path, &text, &error);
    let table: toml::Table = toml::from_str(&text).map_err(error)?;
    if table.contains_key("recall") || table.contains_key("call2") {
        toml::from_str(&text).map(Labels::Keyed).map_err(error)
    } else {
        toml::from_str(&text).map(Labels::Ids).map_err(error)
    }
}

fn load_material(path: &Path) -> anyhow::Result<Material> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&text)
        .map_err(|error| super::json_error(path, error.line(), "labelling material", &error))
}

// The precision curve.

#[derive(Debug, Serialize)]
struct Curves {
    recall: Curve,
    call2: Curve,
    /// What `--convert` wrote.
    #[serde(skip_serializing_if = "Option::is_none")]
    converted: Option<Conversion>,
}

#[derive(Debug, Serialize)]
struct Curve {
    /// Candidates a label scored.
    labelled: u64,
    /// Candidates no label scored.
    unlabelled: u64,
    /// Labels that scored a candidate.
    matched: u64,
    /// Labels that found no candidate.
    unmatched: u64,
    curve: Vec<Point>,
}

#[derive(Debug, Serialize)]
struct Point {
    floor: f64,
    kept: u64,
    relevant: u64,
    precision: f64,
}

/// What converting labels of candidate ids wrote, and what it left out.
#[derive(Debug, Default, Serialize)]
struct Conversion {
    recall: u64,
    call2: u64,
    /// Call 2 labels on material from before version 2, which doesn't
    /// record the claim's chunk and ordinal, so they have no key.
    dropped_call2: u64,
    /// Labels left out because another label judged the same thing the
    /// other way: the same query and memory in two samples, say.
    conflicting: u64,
}

/// Runs `asphodel report precision`: the curve on stdout and exit 0, or
/// exit 2 when the inputs are refused or don't parse.
pub fn run(args: PrecisionArgs) -> anyhow::Result<()> {
    match execute(&args) {
        Ok(curves) => {
            println!("{}", serde_json::to_string_pretty(&curves)?);
            Ok(())
        }
        Err(error) => {
            eprintln!("error: {error:#}");
            std::process::exit(2)
        }
    }
}

fn execute(args: &PrecisionArgs) -> anyhow::Result<Curves> {
    let dir = super::private_dir(args.replay_dir.as_deref())?;
    let labels_path = super::inside_private(&dir, &args.labels, "the labels file")?;
    let material_path = super::inside_private(&dir, &args.material, "the material")?;
    let convert_path = args
        .convert
        .as_deref()
        .map(|path| super::inside_private(&dir, path, "the converted labels"))
        .transpose()?;
    if let Some(path) = &convert_path
        && (path == &labels_path || path == &material_path)
    {
        bail!(
            "--convert {} would overwrite an input; write the converted labels to a new file",
            path.display()
        );
    }
    let labels = load_labels(&labels_path)?;
    let material = load_material(&material_path)?;

    match labels {
        Labels::Keyed(keyed) => {
            if convert_path.is_some() {
                bail!(
                    "--convert rewrites labels of candidate ids, and {} is keyed already",
                    labels_path.display()
                );
            }
            let lookup = Lookup::new(keyed, &labels_path)?;
            Ok(keyed_curves(&material, &lookup))
        }
        Labels::Ids(ids) => {
            let located = candidates_by_id(&material);
            let unknown = ids
                .keys()
                .filter(|id| !located.contains_key(id.as_str()))
                .count();
            if unknown > 0 {
                bail!(
                    "{} labels {unknown} candidate id(s) that {} doesn't hold; label the material the labels were written for",
                    labels_path.display(),
                    material_path.display()
                );
            }
            let mut curves = id_curves(&material, &ids);
            if let Some(path) = &convert_path {
                let (keyed, conversion) = convert(&located, &ids);
                let text = toml::to_string(&keyed).context("writing the converted labels")?;
                super::write_file(path, text.as_bytes()).with_context(|| {
                    format!("writing the converted labels to {}", path.display())
                })?;
                curves.converted = Some(conversion);
            }
            Ok(curves)
        }
    }
}

/// A candidate and the sample it's in.
enum Located<'a> {
    Recall(&'a RecallSample, &'a Candidate),
    Call2(&'a Call2Sample, &'a Candidate),
}

fn candidates_by_id(material: &Material) -> BTreeMap<&str, Located<'_>> {
    let recall = material.recall.iter().flat_map(|sample| {
        sample
            .candidates
            .iter()
            .map(move |candidate| (candidate.id.as_str(), Located::Recall(sample, candidate)))
    });
    let call2 = material.call2.iter().flat_map(|sample| {
        sample
            .candidates
            .iter()
            .map(move |candidate| (candidate.id.as_str(), Located::Call2(sample, candidate)))
    });
    recall.chain(call2).collect()
}

/// Labels of candidate ids against the material they name. Each names a
/// candidate the material holds, so none is unmatched.
fn id_curves(material: &Material, ids: &BTreeMap<String, bool>) -> Curves {
    let scored = |candidates: &mut dyn Iterator<Item = &Candidate>| {
        let scored: Vec<(f64, Option<bool>)> = candidates
            .map(|candidate| (candidate.score, ids.get(&candidate.id).copied()))
            .collect();
        let matched = scored.iter().filter(|(_, label)| label.is_some()).count() as u64;
        curve(&scored, matched, 0)
    };
    Curves {
        recall: scored(&mut material.recall.iter().flat_map(|sample| &sample.candidates)),
        call2: scored(&mut material.call2.iter().flat_map(|sample| &sample.candidates)),
        converted: None,
    }
}

/// Keyed labels against any material: a candidate is labelled when a label
/// judges its sample's query or claim and its memory. Call 2 samples from
/// before version 2 have no key, so no label reaches them.
fn keyed_curves(material: &Material, lookup: &Lookup) -> Curves {
    let mut hits = BTreeSet::new();
    let recall: Vec<(f64, Option<bool>)> = material
        .recall
        .iter()
        .flat_map(|sample| {
            sample
                .candidates
                .iter()
                .map(move |candidate| (candidate.score, (sample.query.clone(), candidate.memory)))
        })
        .map(|(score, key)| {
            let label = lookup.recall.get(&key).copied();
            if label.is_some() {
                hits.insert(key);
            }
            (score, label)
        })
        .collect();
    let matched = hits.len() as u64;
    let recall = curve(&recall, matched, lookup.recall.len() as u64 - matched);

    let mut hits = BTreeSet::new();
    let call2: Vec<(f64, Option<bool>)> = material
        .call2
        .iter()
        .flat_map(|sample| {
            let claim = sample.chunk.zip(sample.ordinal);
            sample.candidates.iter().map(move |candidate| {
                let key = claim.map(|(chunk, ordinal)| (chunk, ordinal, candidate.memory));
                (candidate.score, key)
            })
        })
        .map(|(score, key)| {
            let label = key.and_then(|key| {
                let label = lookup.call2.get(&key).copied();
                if label.is_some() {
                    hits.insert(key);
                }
                label
            });
            (score, label)
        })
        .collect();
    let matched = hits.len() as u64;
    let call2 = curve(&call2, matched, lookup.call2.len() as u64 - matched);

    Curves {
        recall,
        call2,
        converted: None,
    }
}

/// Labels of candidate ids rewritten keyed, through the material they name.
/// A call 2 label on material that predates the claim's key is dropped, and
/// so are labels judging one key both ways; both are counted.
fn convert(
    located: &BTreeMap<&str, Located<'_>>,
    ids: &BTreeMap<String, bool>,
) -> (Keyed, Conversion) {
    let mut conversion = Conversion::default();
    let mut recall: BTreeMap<RecallKey, Vec<bool>> = BTreeMap::new();
    let mut call2: BTreeMap<Call2Key, Vec<bool>> = BTreeMap::new();
    for (id, &relevant) in ids {
        match located.get(id.as_str()) {
            Some(Located::Recall(sample, candidate)) => recall
                .entry((sample.query.clone(), candidate.memory))
                .or_default()
                .push(relevant),
            Some(Located::Call2(sample, candidate)) => match sample.chunk.zip(sample.ordinal) {
                Some((chunk, ordinal)) => call2
                    .entry((chunk, ordinal, candidate.memory))
                    .or_default()
                    .push(relevant),
                None => conversion.dropped_call2 += 1,
            },
            None => {}
        }
    }
    let mut keep = |answers: Vec<bool>| -> Option<bool> {
        let first = answers[0];
        if answers.iter().all(|&answer| answer == first) {
            Some(first)
        } else {
            conversion.conflicting += answers.len() as u64;
            None
        }
    };
    let recall: Vec<RecallLabel> = recall
        .into_iter()
        .filter_map(|((query, memory), answers)| {
            keep(answers).map(|relevant| RecallLabel {
                query,
                memory,
                relevant,
            })
        })
        .collect();
    let call2: Vec<Call2Label> = call2
        .into_iter()
        .filter_map(|((chunk, ordinal, memory), answers)| {
            keep(answers).map(|relevant| Call2Label {
                chunk,
                ordinal,
                memory,
                relevant,
            })
        })
        .collect();
    conversion.recall = recall.len() as u64;
    conversion.call2 = call2.len() as u64;
    (Keyed { recall, call2 }, conversion)
}

/// One point per distinct score among the labelled candidates, in
/// ascending order. A candidate is kept at a floor when it scores at or
/// above it. For recall this matches the gate comparison; for call 2 it
/// is only a score threshold over observed candidates, not a prediction
/// of retention under another reconcile floor.
fn curve(scored: &[(f64, Option<bool>)], matched: u64, unmatched: u64) -> Curve {
    let labelled: Vec<(f64, bool)> = scored
        .iter()
        .filter_map(|&(score, label)| Some((score, label?)))
        .collect();
    let unlabelled = (scored.len() - labelled.len()) as u64;
    let mut floors: Vec<f64> = labelled.iter().map(|(score, _)| *score).collect();
    floors.sort_by(f64::total_cmp);
    floors.dedup();
    let curve = floors
        .into_iter()
        .map(|floor| {
            let kept: Vec<bool> = labelled
                .iter()
                .filter(|(score, _)| *score >= floor)
                .map(|(_, relevant)| *relevant)
                .collect();
            let relevant = kept.iter().filter(|relevant| **relevant).count() as u64;
            let kept = kept.len() as u64;
            Point {
                floor,
                kept,
                relevant,
                precision: relevant as f64 / kept as f64,
            }
        })
        .collect();
    Curve {
        labelled: labelled.len() as u64,
        unlabelled,
        matched,
        unmatched,
        curve,
    }
}
