//! Compare call 1 counts and kind multisets without exporting private text.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use anyhow::{anyhow, bail};
use asphodel_core::extraction::CALL1_TEMPLATE;
use serde_json::{Value, json};

use super::cassette::{ChunkKey, Record};
use crate::cli::ClaimsAgreementArgs;

// These are extraction metadata, never included in the output or errors.
#[derive(PartialEq, Eq)]
struct Metadata {
    version: u32,
    guidance: Option<String>,
    model: String,
    language: Option<String>,
}

type Kinds = BTreeMap<&'static str, usize>;
type Chunks = BTreeMap<ChunkKey, Kinds>;

pub fn run(args: ClaimsAgreementArgs) -> anyhow::Result<()> {
    match execute(&args) {
        Ok(report) => {
            println!("{}", serde_json::to_string_pretty(&report)?);
            Ok(())
        }
        Err(error) => {
            eprintln!("error: {error}");
            std::process::exit(2)
        }
    }
}

fn execute(args: &ClaimsAgreementArgs) -> anyhow::Result<Value> {
    // Do not forward path or parser errors: either can contain private text.
    let dir = super::private_dir(args.replay_dir.as_deref())
        .map_err(|_| anyhow!("claims agreement needs a valid private replay directory"))?;
    let _lock = super::lock_private_dir(&dir)
        .map_err(|_| anyhow!("cannot lock the private replay directory"))?;
    let mut metadata = None;
    let mut load = |given: &Path, side: &str| {
        let path = super::inside_private(&dir, &dir.join(given), "cassette").map_err(|_| {
            anyhow!("{side} cassette must be a file inside the private replay directory")
        })?;
        read(&path, side, &mut metadata)
    };
    let serial = load(&args.serial, "serial")?;
    let primed = load(&args.primed, "primed")?;
    let mut compared = 0;
    let mut count_agreement = 0;
    let mut kind_agreement = 0;
    let mut serial_kinds = Kinds::new();
    let mut primed_kinds = Kinds::new();
    for (chunk, a) in &serial {
        let Some(b) = primed.get(chunk) else { continue };
        compared += 1;
        count_agreement += usize::from(total(a) == total(b));
        kind_agreement += usize::from(a == b);
        for (kinds, totals) in [(a, &mut serial_kinds), (b, &mut primed_kinds)] {
            for (kind, count) in kinds {
                *totals.entry(kind).or_default() += count;
            }
        }
    }
    Ok(json!({
        "chunks": {
            "serial": serial.len(), "primed": primed.len(), "compared": compared,
            "serial_only": serial.len() - compared, "primed_only": primed.len() - compared
        },
        "agreement": {"claim_count": count_agreement, "kind_multiset": kind_agreement},
        "claims": {"serial": total(&serial_kinds), "primed": total(&primed_kinds)},
        "kinds": {"serial": serial_kinds, "primed": primed_kinds}
    }))
}

fn total(kinds: &Kinds) -> usize {
    kinds.values().sum()
}

fn read(path: &Path, side: &str, metadata: &mut Option<Metadata>) -> anyhow::Result<Chunks> {
    let file = File::open(path).map_err(|_| anyhow!("cannot read {side} cassette"))?;
    let mut chunks = Chunks::new();
    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line_number = index + 1;
        let line = line.map_err(|_| anyhow!("cannot read {side} cassette line {line_number}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let record: Record = serde_json::from_str(&line)
            .map_err(|_| anyhow!("invalid {side} cassette record at line {line_number}"))?;
        if record.template.name != CALL1_TEMPLATE {
            continue;
        }
        let Some(chunk) = record.chunk else { continue };
        if record.template != record.request.template {
            bail!("inconsistent extraction metadata in {side} cassette at line {line_number}");
        }
        let current = Metadata {
            version: record.template.version,
            guidance: record.template.guidance,
            model: record.model,
            language: record.language,
        };
        if let Some(expected) = metadata.as_ref() {
            if expected != &current {
                bail!("incompatible extraction metadata in {side} cassette at line {line_number}");
            }
        } else {
            *metadata = Some(current);
        }
        if chunks.contains_key(&chunk) {
            bail!("duplicate call 1 chunk in {side} cassette at line {line_number}");
        }
        let claims = record
            .response
            .json
            .get("claims")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow!("invalid claims in {side} cassette at line {line_number}"))?;
        let mut kinds = Kinds::new();
        for claim in claims {
            // Never export an arbitrary response string as a JSON key.
            let kind = match claim.get("kind").and_then(Value::as_str) {
                Some("fact") => "fact",
                Some("preference") => "preference",
                Some("event") => "event",
                Some("state") => "state",
                Some("task") => "task",
                Some("recurring") => "recurring",
                _ => bail!("invalid claim kind in {side} cassette at line {line_number}"),
            };
            *kinds.entry(kind).or_default() += 1;
        }
        chunks.insert(chunk, kinds);
    }
    Ok(chunks)
}
