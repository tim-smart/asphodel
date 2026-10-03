//! The real-history probes file: the same kinds as a scenario's probes, with
//! opaque ids and `memory` a regex over sentences, since there are no claim
//! labels in real history. Tim writes it after reading a private report, and it
//! lives under the private dir.

use std::collections::BTreeSet;
use std::path::Path;

use anyhow::{Context as _, bail};
use regex::Regex;
use serde::Deserialize;

use super::scenario::{Group, PROBE_SESSION_PREFIX, Probe};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct File {
    #[serde(default, rename = "probe")]
    probes: Vec<Probe>,
}

/// Reads and checks the probes file: ids unique, regexes valid, and no
/// probe that needs the real models when `group` is the fakes.
pub fn load(path: &Path, group: Group) -> anyhow::Result<Vec<Probe>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading the probes file {}", path.display()))?;
    let file: File =
        toml::from_str(&text).map_err(|error| super::toml_error(path, &text, &error))?;
    let mut errors = Vec::new();
    let mut ids = BTreeSet::new();
    for (index, probe) in file.probes.iter().enumerate() {
        let id = probe.id(index);
        if !ids.insert(id.clone()) {
            errors.push(format!("the probe id {id:?} is used twice"));
        }
        if id.starts_with(PROBE_SESSION_PREFIX) {
            errors.push(format!(
                "the probe id {id:?} can't start with {PROBE_SESSION_PREFIX:?}"
            ));
        }
        // The regex error quotes the pattern, which is about Tim's
        // memories, so only `trace` sees it.
        if let Err(error) = Regex::new(probe.check.memory()) {
            tracing::trace!(probe = %id, %error, "a probe's memory regex doesn't parse");
            errors.push(format!("probe {id}: its memory regex doesn't parse"));
        }
        if probe.check.needs_models() && group == Group::Ci {
            errors.push(format!(
                "probe {id} needs the real models, which this run doesn't have"
            ));
        }
        if let super::scenario::Check::FadedAt { between, .. } = &probe.check {
            if between[0] > between[1] {
                errors.push(format!("probe {id} has its range backwards"));
            }
            if probe.at < between[1] {
                errors.push(format!(
                    "probe {id} at {} is before the end of its range {}",
                    probe.at, between[1]
                ));
            }
        }
    }
    if !errors.is_empty() {
        bail!("{}:\n{}", path.display(), errors.join("\n"));
    }
    Ok(file.probes)
}
