//! The local models and the LLM client, behind traits.
//!
//! Three boundaries, each a trait so tests and the replay harness can stand
//! in for it:
//!
//! - [`Embedder`] over bge-small-en-v1.5 int8 ("Rust storage and search
//!   stack", TIM-89);
//! - [`Reranker`] over jina-reranker-v1-turbo-en int8 ("Retrieval and
//!   ranking", TIM-93, decision 3);
//! - [`LlmClient`] over any OpenAI-compatible endpoint, with structured JSON
//!   output. Replay's recording and cassette modes wrap it ("Replay
//!   harness", TIM-96, decision 4).
//!
//! The ONNX models load from a [`ModelDir`] that `asphodel models fetch`
//! fills from the [`manifest`]. Nothing downloads at runtime, and a missing
//! or corrupt file fails before ONNX Runtime is touched ("API surface and
//! Hermes transport", TIM-94, decision 4). The fakes are deterministic and
//! live here rather than in tests, because the scripted replay scenarios
//! run on them under `cargo test` (TIM-96, decision 2).

mod dir;
mod fake;
mod fetch;
mod llm;
mod manifest;
mod onnx;

use std::num::NonZeroUsize;
use std::sync::Arc;

use crate::store::bank::ModelIds;

pub use dir::{ModelDir, ModelError};
pub use fake::{FakeEmbedder, FakeReranker};
pub use fetch::{FetchError, FetchFailure, FetchReport, Fetcher, HttpFetcher, fetch_models};
pub use llm::{
    FakeLlm, LlmClient, LlmError, LlmRequest, LlmResponse, LlmSettings, LlmUsage, OpenAiCompatible,
    Template,
};
pub use manifest::{
    EMBEDDING_MODEL_ID, MODEL_FILES, ModelFile, ModelSpec, RERANKER_MODEL_ID, manifest,
};
pub use onnx::{OnnxEmbedder, OnnxReranker};

/// How to run the ONNX models.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ModelOptions {
    /// Intra-op threads for ONNX Runtime. `None` leaves it to the runtime;
    /// replay pins it (TIM-96, decision 3) through `--onnx-threads` /
    /// `ASPHODEL_ONNX_THREADS`.
    pub threads: Option<NonZeroUsize>,
}

/// Turns text into vectors.
pub trait Embedder: Send + Sync {
    /// The exact model string, as the floors and the bank row use it.
    fn model_id(&self) -> &str;

    /// The width of every vector: [`EMBEDDING_DIMENSIONS`] for bge-small.
    ///
    /// [`EMBEDDING_DIMENSIONS`]: crate::store::vector::EMBEDDING_DIMENSIONS
    fn dimensions(&self) -> usize;

    /// One unit-length vector per text, in the texts' order. An empty slice
    /// gives an empty vec. Implementations take `&self`, so a model that
    /// needs `&mut` sits behind a mutex.
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, ModelError>;
}

/// Scores documents against a query.
pub trait Reranker: Send + Sync {
    /// The exact model string, as the floors and the bank row use it.
    fn model_id(&self) -> &str;

    /// One relevance logit per document, in the documents' order (not
    /// sorted: the caller keeps its ids). Higher is more relevant, and the
    /// injection gate is a floor on this value (TIM-93, decision 9). An
    /// empty slice gives an empty vec.
    fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<f32>, ModelError>;
}

/// The pair the daemon runs. Shared, so handlers and the extraction worker
/// use one loaded copy.
#[derive(Clone)]
pub struct Models {
    pub embedder: Arc<dyn Embedder>,
    pub reranker: Arc<dyn Reranker>,
}

impl Models {
    /// Loads both models from `dir` against the [`manifest`]. Every file is
    /// checked for presence, then for its checksum, before ONNX Runtime is
    /// touched, so a missing or corrupt file fails fast and names the path.
    /// It never downloads and never creates the dir.
    pub fn load(dir: &ModelDir, options: &ModelOptions) -> Result<Self, ModelError> {
        let manifest = manifest();
        for spec in &manifest {
            for file in &spec.files {
                let path = dir.file(spec, &file.name);
                if !path.is_file() {
                    return Err(ModelError::MissingFile {
                        model: spec.id.clone(),
                        path,
                    });
                }
            }
        }
        let embedding = dir.read_verified(&manifest[0])?;
        let reranking = dir.read_verified(&manifest[1])?;
        let embedder = OnnxEmbedder::new(&manifest[0].id, embedding, options)?;
        let reranker = OnnxReranker::new(&manifest[1].id, reranking, options)?;
        Ok(Self {
            embedder: Arc::new(embedder),
            reranker: Arc::new(reranker),
        })
    }

    /// The deterministic fakes, for tests and the scripted scenarios.
    pub fn fake() -> Self {
        Self {
            embedder: Arc::new(FakeEmbedder),
            reranker: Arc::new(FakeReranker),
        }
    }

    /// What a bank created under these models records.
    pub fn ids(&self) -> ModelIds {
        ModelIds {
            embedding: self.embedder.model_id().to_string(),
            reranker: self.reranker.model_id().to_string(),
        }
    }
}

impl std::fmt::Debug for Models {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Models")
            .field("embedder", &self.embedder.model_id())
            .field("reranker", &self.reranker.model_id())
            .finish()
    }
}

/// Scales `vector` to unit length. A zero vector is left alone.
pub(crate) fn normalise(vector: &mut [f32]) {
    let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in vector.iter_mut() {
            *value /= norm;
        }
    }
}
