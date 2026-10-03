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
//! `asphodel report precision --labels L --material M` reads a TOML table
//! of candidate id to relevance and prints the curve: at each labelled
//! score, the relevant fraction of the labelled candidates at or above it.
//! The curve is numbers only. Both files are history, so both must be
//! inside the private dir, and no error quotes either.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use anyhow::{Context as _, bail};
use asphodel_core::extraction::Call2List;
use asphodel_core::retrieval::GateCandidate;
use jiff::Timestamp;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::cli::PrecisionArgs;

/// How many synced turns the recall material samples.
pub const SAMPLED_TURNS: usize = 50;

/// The material's format version.
const VERSION: u32 = 1;

#[derive(Debug, Serialize, Deserialize)]
pub struct Material {
    pub version: u32,
    pub recall: Vec<RecallSample>,
    pub call2: Vec<Call2Sample>,
}

/// A sampled turn's prefetch: the query the reranker scored against, and
/// its candidates in ranked order.
#[derive(Debug, Serialize, Deserialize)]
pub struct RecallSample {
    pub sample: String,
    pub at: Timestamp,
    pub session: String,
    pub query: String,
    pub candidates: Vec<Candidate>,
}

/// A claim and the neighbours call 2 was shown for it.
#[derive(Debug, Serialize, Deserialize)]
pub struct Call2Sample {
    pub sample: String,
    pub at: Timestamp,
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
    candidates: Vec<(Uuid, String, f64)>,
}

/// What a run collects for the material while it runs. Collecting reads
/// what the prefetches and call 2 already computed, so it changes nothing
/// the run simulates.
#[derive(Default)]
pub struct Collector {
    turns: Vec<Turn>,
    call2: Vec<(Timestamp, Call2List)>,
}

impl Collector {
    /// A synced turn's prefetch. A candidate the reranker didn't score has
    /// no logit to label against and is left out; replay never skips the
    /// reranker, so there are none in practice.
    pub fn turn(&mut self, at: Timestamp, session: &str, query: &str, shown: &[GateCandidate]) {
        self.turns.push(Turn {
            at,
            session: session.to_owned(),
            query: query.to_owned(),
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

    /// The material: [`SAMPLED_TURNS`] turns spread evenly over the run, or
    /// every turn when there are fewer, and every call 2 list, with ids
    /// assigned in order.
    pub fn finish(self) -> Material {
        let total = self.turns.len();
        let picked: BTreeSet<usize> = if total <= SAMPLED_TURNS {
            (0..total).collect()
        } else {
            (0..SAMPLED_TURNS)
                .map(|index| index * total / SAMPLED_TURNS)
                .collect()
        };
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

// The precision curve.

#[derive(Debug, Serialize)]
struct Curves {
    recall: Curve,
    call2: Curve,
}

#[derive(Debug, Serialize)]
struct Curve {
    labelled: u64,
    unlabelled: u64,
    curve: Vec<Point>,
}

#[derive(Debug, Serialize)]
struct Point {
    floor: f64,
    kept: u64,
    relevant: u64,
    precision: f64,
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
    let labels = load_labels(&labels_path)?;
    let material = load_material(&material_path)?;

    let ids: BTreeSet<&str> = material
        .recall
        .iter()
        .flat_map(|sample| &sample.candidates)
        .chain(material.call2.iter().flat_map(|sample| &sample.candidates))
        .map(|candidate| candidate.id.as_str())
        .collect();
    let unknown = labels
        .keys()
        .filter(|id| !ids.contains(id.as_str()))
        .count();
    if unknown > 0 {
        bail!(
            "{} labels {unknown} candidate id(s) that {} doesn't hold; label the material the labels were written for",
            labels_path.display(),
            material_path.display()
        );
    }

    Ok(Curves {
        recall: curve(
            material.recall.iter().flat_map(|sample| &sample.candidates),
            &labels,
        ),
        call2: curve(
            material.call2.iter().flat_map(|sample| &sample.candidates),
            &labels,
        ),
    })
}

fn load_labels(path: &Path) -> anyhow::Result<BTreeMap<String, bool>> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    toml::from_str(&text).map_err(|error| super::toml_error(path, &text, &error))
}

fn load_material(path: &Path) -> anyhow::Result<Material> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&text)
        .map_err(|error| super::json_error(path, error.line(), "labelling material", &error))
}

/// One point per distinct score among the labelled candidates, in
/// ascending order. A candidate is kept at a floor when it scores at or
/// above it. For recall this matches the gate comparison; for call 2 it
/// is only a score threshold over observed candidates, not a prediction
/// of retention under another reconcile floor.
fn curve<'a>(
    candidates: impl Iterator<Item = &'a Candidate>,
    labels: &BTreeMap<String, bool>,
) -> Curve {
    let mut labelled: Vec<(f64, bool)> = Vec::new();
    let mut unlabelled = 0;
    for candidate in candidates {
        match labels.get(&candidate.id) {
            Some(&relevant) => labelled.push((candidate.score, relevant)),
            None => unlabelled += 1,
        }
    }
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
        curve,
    }
}
