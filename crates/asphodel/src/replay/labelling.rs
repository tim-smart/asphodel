//! The labelling material and the precision curve (`docs/replay.md`, "Labelling
//! and the precision curve").
//!
//! The reranker gate floor and the reconcile similarity floor are set from
//! Tim's labels. `asphodel replay --labelling FILE` writes the material to
//! label: recall candidates at [`SAMPLED_TURNS`] turns, scored with the
//! reranker logit the gate compares and including those it turned away,
//! every candidate list call 2 was shown, scored with cosine similarity, and
//! every facet's whole pool at [`SAMPLED_REFRESHES`] refreshes, scored with
//! the raw reranker logit, with what took or cut each candidate. Call
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
//! written for, and `--convert` rewrites it keyed. For recall it also
//! counts the relevant candidates in each sample's top [`TOP`] by logit,
//! the same way for either form of label. For refreshes it adds, per facet
//! query, the curve, and at each floor each sample's input size and the
//! share of cited memories still kept. The output is numbers only. Every
//! file is history, so all must be inside the private dir, and
//! no error quotes any.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use anyhow::{Context as _, bail};
use asphodel_core::extraction::Call2List;
use asphodel_core::mental_models::ScoredRefresh;
use asphodel_core::retrieval::{ScoredPrefetch, Taken};
use asphodel_core::store::bank::PROFILE_NAME;
use jiff::Timestamp;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::cli::PrecisionArgs;

/// How many synced turns the recall material samples.
pub const SAMPLED_TURNS: usize = 50;

/// How many refreshes the refresh material samples.
pub const SAMPLED_REFRESHES: usize = 10;

/// How many of a sample's candidates by logit `top8` looks at: the
/// injection cap.
const TOP: usize = 8;

/// The material's format version. Version 2 added the claim's chunk and
/// ordinal to call 2's samples, version 3 the refresh samples.
const VERSION: u32 = 3;

#[derive(Debug, Serialize, Deserialize)]
pub struct Material {
    pub version: u32,
    pub recall: Vec<RecallSample>,
    pub call2: Vec<Call2Sample>,
    /// Material before version 3 has none.
    #[serde(default)]
    pub refresh: Vec<RefreshSample>,
}

/// A sampled turn's prefetch: the query the retrievers searched, the query
/// the reranker scored against, the message they were cleaned from, and its
/// candidates in ranked order.
#[derive(Debug, Serialize, Deserialize)]
pub struct RecallSample {
    pub sample: String,
    pub at: Timestamp,
    pub session: String,
    /// The cleaned query the retrievers searched.
    pub query: String,
    /// The query the reranker scored against, which calibration uses: the
    /// same as `query` unless the run reranked against the conversation.
    /// Material written before it was recorded has none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rerank_query: Option<String>,
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

/// One facet of a sampled refresh: its query and its whole pool before the
/// facet's budget cut it, in score order.
#[derive(Debug, Serialize, Deserialize)]
pub struct RefreshSample {
    pub sample: String,
    pub at: Timestamp,
    pub model: String,
    pub facet: String,
    /// The facet's place in the model's plan, from 0.
    pub facet_index: usize,
    /// The facet's query as the plan holds it, which facet labels are keyed
    /// by.
    pub query: String,
    /// The query the reranker scored against.
    pub rerank_query: String,
    /// Whether the reranker scored the facet. When it didn't, every logit
    /// is null and the scores are strength alone.
    pub reranked: bool,
    pub candidates: Vec<RefreshCandidate>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RefreshCandidate {
    /// Unique in the material.
    pub id: String,
    pub memory: Uuid,
    pub sentence: String,
    /// The raw reranker logit, or null when the facet missed the reranker.
    pub logit: Option<f64>,
    pub strength: f64,
    /// The combined score the facet ranked by.
    pub score: f64,
    /// The place in the facet's score order, from 1.
    pub rank: usize,
    /// Whether the model cited the memory when the refresh began.
    pub cited: bool,
    pub taken: TakenBy,
    /// The handle the memory reached the write under, under whichever
    /// facet took it, or null when it isn't in the selection.
    pub input: Option<String>,
}

/// What took a refresh candidate: the facet's budget, the cited fill past
/// it, or neither.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TakenBy {
    Budget,
    Cited,
    Cut,
}

impl From<Taken> for TakenBy {
    fn from(taken: Taken) -> Self {
        match taken {
            Taken::Budget => TakenBy::Budget,
            Taken::Cited => TakenBy::Cited,
            Taken::Cut => TakenBy::Cut,
        }
    }
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
    rerank_query: String,
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
    refreshes: Vec<ScoredRefresh>,
}

impl Collector {
    pub fn new(labelled: BTreeSet<String>) -> Self {
        Self {
            labelled,
            turns: Vec::new(),
            call2: Vec::new(),
            refreshes: Vec::new(),
        }
    }

    /// The refreshes that made a selection, in the order they ran.
    pub fn refreshes(&mut self, scored: Vec<ScoredRefresh>) {
        self.refreshes.extend(scored);
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
            rerank_query: scored.rerank_query.clone(),
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
    ///
    /// The refresh samples are every facet of [`SAMPLED_REFRESHES`]
    /// refreshes, or of every refresh when there are fewer, in the order
    /// they ran: the seeded profile's spread evenly first, then the other
    /// models' spread over what's left.
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
                    rerank_query: Some(turn.rerank_query),
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
        let (profile, other): (Vec<usize>, Vec<usize>) = (0..self.refreshes.len())
            .partition(|&index| self.refreshes[index].model == PROFILE_NAME);
        let mut picked = spread(&profile, SAMPLED_REFRESHES);
        picked.extend(spread(&other, SAMPLED_REFRESHES - picked.len()));
        let mut refresh = Vec::new();
        for (_, scored) in self
            .refreshes
            .into_iter()
            .enumerate()
            .filter(|(index, _)| picked.contains(index))
        {
            for (facet_index, facet) in scored.facets.into_iter().enumerate() {
                let sample = format!("f{}", refresh.len() + 1);
                let candidates = facet
                    .pool
                    .candidates
                    .into_iter()
                    .enumerate()
                    .map(|(index, candidate)| RefreshCandidate {
                        id: format!("{sample}.{}", index + 1),
                        memory: candidate.memory,
                        sentence: candidate.sentence,
                        logit: candidate.logit,
                        strength: candidate.strength,
                        score: candidate.score,
                        rank: index + 1,
                        cited: candidate.cited,
                        taken: candidate.taken.into(),
                        input: candidate.input,
                    })
                    .collect();
                refresh.push(RefreshSample {
                    sample,
                    at: scored.at,
                    model: scored.model.clone(),
                    facet: facet.heading,
                    facet_index,
                    query: facet.query,
                    rerank_query: facet.pool.rerank_query,
                    reranked: facet.pool.reranked,
                    candidates,
                });
            }
        }
        Material {
            version: VERSION,
            recall,
            call2,
            refresh,
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
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    facet: Vec<FacetLabel>,
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

/// Whether a memory belongs under a refresh facet's heading, named by the
/// facet's query.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FacetLabel {
    /// The facet's query, as the material's `query`.
    query: String,
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
    facet: BTreeMap<RecallKey, bool>,
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
        let mut facet = BTreeMap::new();
        for (index, label) in keyed.facet.into_iter().enumerate() {
            let key = (label.query, label.memory);
            insert(&mut facet, key, index + 1, label.relevant, "facet", path)?;
        }
        fn answers<K: Ord>(map: BTreeMap<K, (usize, bool)>) -> BTreeMap<K, bool> {
            map.into_iter()
                .map(|(key, (_, relevant))| (key, relevant))
                .collect()
        }
        Ok(Self {
            recall: answers(recall),
            call2: answers(call2),
            facet: answers(facet),
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
    if ["recall", "call2", "facet"]
        .iter()
        .any(|list| table.contains_key(*list))
    {
        toml::from_str(&text).map(Labels::Keyed).map_err(error)
    } else {
        toml::from_str(&text).map(Labels::Ids).map_err(error)
    }
}

pub(crate) fn load_material(path: &Path) -> anyhow::Result<Material> {
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
    refresh: RefreshCurves,
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
    /// For recall only.
    #[serde(skip_serializing_if = "Option::is_none")]
    top8: Option<Top>,
}

/// The refresh samples under facet labels: the curve pooled over every
/// facet and for each facet query, then at each of the pooled curve's
/// floors each sample's input size and the share of each refresh's cited
/// memories that would stay. A null logit never passes a floor.
#[derive(Debug, Serialize)]
struct RefreshCurves {
    #[serde(flatten)]
    pooled: Curve,
    facets: Vec<FacetCurve>,
    inputs: Vec<SampleInputs>,
    cited: Vec<CitedPoint>,
}

/// One facet query's curve, named by the first sample with that query and
/// the facet's place in its plan, since the query itself is history.
#[derive(Debug, Serialize)]
struct FacetCurve {
    sample: String,
    facet_index: usize,
    #[serde(flatten)]
    curve: Curve,
}

/// A sample's input size: the candidates its facet's budget took, and of
/// those, how many score at or above each floor.
#[derive(Debug, Serialize)]
struct SampleInputs {
    sample: String,
    budget: u64,
    sizes: Vec<InputSize>,
}

#[derive(Debug, Serialize)]
struct InputSize {
    floor: f64,
    size: u64,
}

/// Over every sampled refresh, the memories its model cited and how many of
/// them score at or above the floor under at least one of its facets.
#[derive(Debug, Serialize)]
struct CitedPoint {
    floor: f64,
    cited: u64,
    retained: u64,
    share: f64,
}

/// The candidates labelled relevant that rank in their sample's top [`TOP`]
/// by logit, out of every candidate labelled relevant.
#[derive(Debug, Serialize)]
struct Top {
    found: u64,
    relevant: u64,
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
        let scored: Vec<(Option<f64>, Option<bool>)> = candidates
            .map(|candidate| (Some(candidate.score), ids.get(&candidate.id).copied()))
            .collect();
        let matched = scored.iter().filter(|(_, label)| label.is_some()).count() as u64;
        curve(&scored, matched, 0)
    };
    let mut recall = scored(&mut material.recall.iter().flat_map(|sample| &sample.candidates));
    recall.top8 = Some(top(&material.recall, |_, candidate| {
        ids.get(&candidate.id).copied()
    }));
    Curves {
        recall,
        call2: scored(&mut material.call2.iter().flat_map(|sample| &sample.candidates)),
        refresh: refresh_curves(&material.refresh, &BTreeMap::new()),
        converted: None,
    }
}

/// Keyed labels against any material: a candidate is labelled when a label
/// judges its sample's query or claim and its memory. Call 2 samples from
/// before version 2 have no key, so no label reaches them.
fn keyed_curves(material: &Material, lookup: &Lookup) -> Curves {
    let mut hits = BTreeSet::new();
    let recall: Vec<(Option<f64>, Option<bool>)> = material
        .recall
        .iter()
        .flat_map(|sample| {
            sample.candidates.iter().map(move |candidate| {
                (
                    Some(candidate.score),
                    (sample.query.clone(), candidate.memory),
                )
            })
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
    let mut recall = curve(&recall, matched, lookup.recall.len() as u64 - matched);
    recall.top8 = Some(top(&material.recall, |sample, candidate| {
        lookup
            .recall
            .get(&(sample.query.clone(), candidate.memory))
            .copied()
    }));

    let mut hits = BTreeSet::new();
    let call2: Vec<(Option<f64>, Option<bool>)> = material
        .call2
        .iter()
        .flat_map(|sample| {
            let claim = sample.chunk.zip(sample.ordinal);
            sample.candidates.iter().map(move |candidate| {
                let key = claim.map(|(chunk, ordinal)| (chunk, ordinal, candidate.memory));
                (Some(candidate.score), key)
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
        refresh: refresh_curves(&material.refresh, &lookup.facet),
        converted: None,
    }
}

/// The refresh samples under `labels`, keyed by the facet's query and the
/// memory. A label matches every candidate with that memory under a sample
/// with that query, in every sampled refresh.
fn refresh_curves(samples: &[RefreshSample], labels: &BTreeMap<RecallKey, bool>) -> RefreshCurves {
    let key = |sample: &RefreshSample, candidate: &RefreshCandidate| {
        (sample.query.clone(), candidate.memory)
    };
    let scored = |samples: &[&RefreshSample], hits: &mut BTreeSet<RecallKey>| {
        let mut scored: Vec<(Option<f64>, Option<bool>)> = Vec::new();
        for sample in samples {
            for candidate in &sample.candidates {
                let key = key(sample, candidate);
                let label = labels.get(&key).copied();
                if label.is_some() {
                    hits.insert(key);
                }
                scored.push((candidate.logit, label));
            }
        }
        scored
    };

    let all: Vec<&RefreshSample> = samples.iter().collect();
    let mut hits = BTreeSet::new();
    let pooled = scored(&all, &mut hits);
    let matched = hits.len() as u64;
    let pooled = curve(&pooled, matched, labels.len() as u64 - matched);

    // Facet queries in the order they first appear.
    let mut queries: Vec<&RefreshSample> = Vec::new();
    for sample in samples {
        if !queries.iter().any(|first| first.query == sample.query) {
            queries.push(sample);
        }
    }
    let facets = queries
        .into_iter()
        .map(|first| {
            let under: Vec<&RefreshSample> = samples
                .iter()
                .filter(|sample| sample.query == first.query)
                .collect();
            let mut hits = BTreeSet::new();
            let scored = scored(&under, &mut hits);
            let matched = hits.len() as u64;
            let labelled = labels.keys().filter(|(query, _)| *query == first.query);
            let unmatched = labelled.count() as u64 - matched;
            FacetCurve {
                sample: first.sample.clone(),
                facet_index: first.facet_index,
                curve: curve(&scored, matched, unmatched),
            }
        })
        .collect();

    let floors: Vec<f64> = pooled.curve.iter().map(|point| point.floor).collect();
    let passes = |logit: Option<f64>, floor: f64| logit.is_some_and(|logit| logit >= floor);
    let inputs = samples
        .iter()
        .map(|sample| {
            let taken: Vec<Option<f64>> = sample
                .candidates
                .iter()
                .filter(|candidate| candidate.taken == TakenBy::Budget)
                .map(|candidate| candidate.logit)
                .collect();
            SampleInputs {
                sample: sample.sample.clone(),
                budget: taken.len() as u64,
                sizes: floors
                    .iter()
                    .map(|&floor| InputSize {
                        floor,
                        size: taken.iter().filter(|&&logit| passes(logit, floor)).count() as u64,
                    })
                    .collect(),
            }
        })
        .collect();

    // A refresh is its samples with one time and model.
    let mut refreshes: BTreeMap<(Timestamp, &str), Vec<&RefreshSample>> = BTreeMap::new();
    for sample in samples {
        refreshes
            .entry((sample.at, sample.model.as_str()))
            .or_default()
            .push(sample);
    }
    let cited: Vec<(BTreeSet<Uuid>, BTreeMap<Uuid, f64>)> = refreshes
        .values()
        .map(|samples| {
            let mut cited = BTreeSet::new();
            let mut best: BTreeMap<Uuid, f64> = BTreeMap::new();
            for candidate in samples.iter().flat_map(|sample| &sample.candidates) {
                if candidate.cited {
                    cited.insert(candidate.memory);
                }
                if let Some(logit) = candidate.logit {
                    let kept = best.entry(candidate.memory).or_insert(logit);
                    *kept = kept.max(logit);
                }
            }
            (cited, best)
        })
        .collect();
    let total: u64 = cited.iter().map(|(cited, _)| cited.len() as u64).sum();
    let cited = if total == 0 {
        Vec::new()
    } else {
        floors
            .iter()
            .map(|&floor| {
                let retained = cited
                    .iter()
                    .map(|(cited, best)| {
                        cited
                            .iter()
                            .filter(|memory| passes(best.get(memory).copied(), floor))
                            .count() as u64
                    })
                    .sum::<u64>();
                CitedPoint {
                    floor,
                    cited: total,
                    retained,
                    share: retained as f64 / total as f64,
                }
            })
            .collect()
    };

    RefreshCurves {
        pooled,
        facets,
        inputs,
        cited,
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
    let keyed = Keyed {
        recall,
        call2,
        facet: Vec::new(),
    };
    (keyed, conversion)
}

/// Each sample's candidates ranked by logit, highest first and ties in the
/// material's order, with `label` giving each candidate's label. Unlabelled
/// candidates take places but aren't counted.
fn top(samples: &[RecallSample], label: impl Fn(&RecallSample, &Candidate) -> Option<bool>) -> Top {
    let mut found = 0;
    let mut relevant = 0;
    for sample in samples {
        let mut ranked: Vec<&Candidate> = sample.candidates.iter().collect();
        ranked.sort_by(|a, b| b.score.total_cmp(&a.score));
        for (rank, candidate) in ranked.into_iter().enumerate() {
            if label(sample, candidate) == Some(true) {
                relevant += 1;
                if rank < TOP {
                    found += 1;
                }
            }
        }
    }
    Top { found, relevant }
}

/// One point per distinct score among the labelled candidates, in
/// ascending order. A candidate is kept at a floor when it scores at or
/// above it, and one with no score, a refresh candidate the reranker
/// missed, never is. For recall this matches the gate comparison; for call
/// 2 it is only a score threshold over observed candidates, not a
/// prediction of retention under another reconcile floor.
fn curve(scored: &[(Option<f64>, Option<bool>)], matched: u64, unmatched: u64) -> Curve {
    let labelled: Vec<(Option<f64>, bool)> = scored
        .iter()
        .filter_map(|&(score, label)| Some((score, label?)))
        .collect();
    let unlabelled = (scored.len() - labelled.len()) as u64;
    let mut floors: Vec<f64> = labelled.iter().filter_map(|(score, _)| *score).collect();
    floors.sort_by(f64::total_cmp);
    floors.dedup();
    let curve = floors
        .into_iter()
        .map(|floor| {
            let kept: Vec<bool> = labelled
                .iter()
                .filter(|(score, _)| score.is_some_and(|score| score >= floor))
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
        top8: None,
    }
}
