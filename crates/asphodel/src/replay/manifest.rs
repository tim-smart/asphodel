//! The private manifest (TIM-96, decision 1): what `state.db` doesn't hold.
//! The timezone, the owner's platform ids, the owner's and assistant's
//! names, and the names of the other speakers. Its `[[model]]` tables,
//! in a scenario's shape, are the mental models a corpus run creates
//! before the first event, since `state.db` holds none. It lives under the
//! private dir and nothing in it is ever written anywhere but the corpus
//! header.

use std::path::Path;

use anyhow::{Context as _, bail};
use jiff::tz::TimeZone;
use serde::Deserialize;

use super::scenario::ModelSection;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    /// An IANA timezone name: the bank's, and every turn's.
    pub timezone: String,
    /// The bank the history replays into; `main` when absent.
    #[serde(default = "default_bank")]
    pub bank: String,
    #[serde(default)]
    pub assistant: Option<String>,
    pub owner: Owner,
    /// The non-owner speakers a `[Name] ` prefix can name. Any other prefix
    /// is the owner's turn.
    #[serde(default, rename = "speaker")]
    pub speakers: Vec<Speaker>,
    /// The fence Hermes puts injected memory in, inside multimodal content.
    /// Defaults to `<memory-context>` and `</memory-context>`.
    #[serde(default)]
    pub memory_block: Option<MemoryBlock>,
    /// The mental models to create in the bank before the first event.
    #[serde(default, rename = "model")]
    pub models: Vec<ModelSection>,
}

fn default_bank() -> String {
    "main".into()
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Owner {
    #[serde(default)]
    pub name: Option<String>,
    /// `<platform>:<id>`, as the bank's config takes them.
    #[serde(default)]
    pub platform_ids: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Speaker {
    pub name: String,
    /// `<platform>:<id>`.
    pub id: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryBlock {
    pub start: String,
    pub end: String,
}

impl Default for MemoryBlock {
    fn default() -> Self {
        Self {
            start: "<memory-context>".into(),
            end: "</memory-context>".into(),
        }
    }
}

impl Manifest {
    /// The speaker a prefix names, if it's a non-owner one.
    pub fn speaker(&self, name: &str) -> Option<&Speaker> {
        self.speakers.iter().find(|speaker| speaker.name == name)
    }

    pub fn memory_block(&self) -> MemoryBlock {
        self.memory_block.clone().unwrap_or_default()
    }
}

/// Reads and checks the manifest.
pub fn load(path: &Path) -> anyhow::Result<Manifest> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading the manifest {}", path.display()))?;
    let manifest: Manifest =
        toml::from_str(&text).map_err(|error| super::toml_error(path, &text, &error))?;
    TimeZone::get(&manifest.timezone)
        .with_context(|| format!("the manifest's timezone {:?}", manifest.timezone))?;
    if manifest.bank.is_empty() {
        bail!("the manifest's bank name is empty");
    }
    let mut names = std::collections::BTreeSet::new();
    for speaker in &manifest.speakers {
        if speaker.name.trim().is_empty() || speaker.id.trim().is_empty() {
            bail!("a manifest speaker has an empty name or id");
        }
        if !names.insert(speaker.name.as_str()) {
            bail!("the manifest lists one speaker's name twice");
        }
        if manifest.owner.name.as_deref() == Some(speaker.name.as_str()) {
            bail!("the manifest lists the owner as a speaker too");
        }
    }
    let mut models = std::collections::BTreeSet::new();
    for model in &manifest.models {
        if model.name.trim().is_empty() || model.question.trim().is_empty() {
            bail!("a manifest model has an empty name or question");
        }
        if !models.insert(model.name.as_str()) {
            bail!("the manifest lists the model {:?} twice", model.name);
        }
    }
    if let Some(block) = &manifest.memory_block
        && (block.start.is_empty() || block.end.is_empty())
    {
        bail!("the manifest's memory_block start and end can't be empty");
    }
    Ok(manifest)
}
