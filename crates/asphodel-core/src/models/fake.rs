//! Deterministic stand-ins for the embedding model and the reranker.
//!
//! The scripted replay scenarios and the unit tests run on these, so they
//! have to behave the same in every process: no randomness, no state, and a
//! hash that doesn't change between Rust versions.

use super::{Embedder, ModelError, Reranker, normalise};
use crate::store::vector::EMBEDDING_DIMENSIONS;

/// A deterministic stand-in for the embedding model: a bag of words hashed
/// into [`EMBEDDING_DIMENSIONS`] buckets and normalised. Texts that share
/// words are closer, word order doesn't matter, and the same text always
/// gives the same vector.
#[derive(Debug, Clone, Copy, Default)]
pub struct FakeEmbedder;

impl FakeEmbedder {
    /// Distinct from the real id, so floors keyed on the real model never
    /// apply to the fake by accident.
    pub const MODEL_ID: &'static str = "fake-embedder:v1";
}

impl Embedder for FakeEmbedder {
    fn model_id(&self) -> &str {
        Self::MODEL_ID
    }

    fn dimensions(&self) -> usize {
        EMBEDDING_DIMENSIONS
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, ModelError> {
        Ok(texts.iter().map(|text| embed_one(text)).collect())
    }
}

/// A second deterministic embedder with its own id, which hashes words
/// into other buckets: a stand-in for a changed embedding model, so a
/// re-embed can be driven on fakes (ADR 0010). `ASPHODEL_MODELS=fake-v2`
/// serves with it and carries [`FakeEmbedder`] for banks recorded under it.
#[derive(Debug, Clone, Copy, Default)]
pub struct FakeEmbedderV2;

impl FakeEmbedderV2 {
    pub const MODEL_ID: &'static str = "fake-embedder:v2";
}

impl Embedder for FakeEmbedderV2 {
    fn model_id(&self) -> &str {
        Self::MODEL_ID
    }

    fn dimensions(&self) -> usize {
        EMBEDDING_DIMENSIONS
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, ModelError> {
        Ok(texts
            .iter()
            .map(|text| embed_seeded(text, b"v2:"))
            .collect())
    }
}

fn embed_one(text: &str) -> Vec<f32> {
    embed_seeded(text, b"")
}

/// The bag of words, each word hashed after `seed`.
fn embed_seeded(text: &str, seed: &[u8]) -> Vec<f32> {
    let mut vector = vec![0.0f32; EMBEDDING_DIMENSIONS];
    let mut any = false;
    for word in words(text) {
        let hash = fnv1a(&[seed, word.as_bytes()].concat());
        let bucket = (hash % EMBEDDING_DIMENSIONS as u64) as usize;
        let sign = if (hash >> 32) & 1 == 0 { 1.0 } else { -1.0 };
        vector[bucket] += sign;
        any = true;
    }
    if !any {
        // An empty text still gets a unit vector, so norms hold everywhere.
        vector[0] = 1.0;
    }
    normalise(&mut vector);
    vector
}

/// A deterministic stand-in for the reranker: the logit is the number of
/// distinct query words found in the document, minus one half, so a
/// document sharing nothing with the query scores below zero and one
/// sharing more scores higher.
#[derive(Debug, Clone, Copy, Default)]
pub struct FakeReranker;

impl FakeReranker {
    pub const MODEL_ID: &'static str = "fake-reranker:v1";
}

impl Reranker for FakeReranker {
    fn model_id(&self) -> &str {
        Self::MODEL_ID
    }

    fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<f32>, ModelError> {
        let query: std::collections::BTreeSet<String> = words(query).collect();
        Ok(documents
            .iter()
            .map(|document| {
                let found: std::collections::BTreeSet<String> = words(document)
                    .filter(|word| query.contains(word))
                    .collect();
                found.len() as f32 - 0.5
            })
            .collect())
    }
}

/// Lowercase runs of letters and digits.
fn words(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_lowercase)
}

/// FNV-1a, 64-bit: stable across platforms and Rust versions, unlike
/// `DefaultHasher`.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}
