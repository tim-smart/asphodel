//! The local models and the LLM client, behind traits.
//!
//! Three boundaries, each a trait so tests and the replay harness can stand
//! in for it:
//!
//! - [`Embedder`] over bge-small-en-v1.5 int8;
//! - [`Reranker`] over ms-marco-MiniLM-L-6-v2 int8;
//! - [`LlmClient`] over any OpenAI-compatible endpoint, with structured JSON
//!   output. Replay's recording and cassette modes wrap it.
//!
//! The ONNX models load from a [`ModelDir`] that `asphodel models fetch`
//! fills from the [`manifest`]. Nothing downloads at runtime, and a missing
//! or corrupt file fails before ONNX Runtime is touched. The fakes are
//! deterministic and live here rather than in tests, because the scripted
//! replay scenarios
//! run on them under `cargo test`.

mod chatgpt;
mod dir;
mod fake;
mod fetch;
mod gate;
mod llm;
mod manifest;
mod onnx;
pub(crate) mod write;

use std::num::NonZeroUsize;
use std::sync::Arc;

use crate::store::bank::ModelIds;

pub use crate::config::LlmAuth;
pub use chatgpt::{
    AUTH_ISSUER, CLIENT_ID, CODEX_ENDPOINT, ChatgptTokens, CodexResponses, DeviceCode, LlmStatus,
    LoginError, ORIGINATOR, REFRESH_WINDOW, TOKEN_FILE, TOKEN_LOCK_FILE, TokenError, TokenLock,
    TokenStore, device_code_login,
};
pub use dir::{ModelDir, ModelError};
pub use fake::{FakeEmbedder, FakeEmbedderV2, FakeReranker};
pub use fetch::{FetchError, FetchFailure, FetchReport, Fetcher, HttpFetcher, fetch_models};
pub use gate::LlmGate;
pub use llm::{
    FakeLlm, LlmClient, LlmError, LlmRequest, LlmResponse, LlmSettings, LlmUsage, OpenAiCompatible,
    ScriptError, ScriptStep, ScriptedFailure, Template,
};
pub use manifest::{
    EMBEDDING_MODEL_ID, MODEL_FILES, ModelFile, ModelSpec, RERANKER_MODEL_ID, manifest,
};
pub use onnx::{OnnxEmbedder, OnnxReranker};

/// How to run the ONNX models.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ModelOptions {
    /// Intra-op threads for ONNX Runtime. `None` leaves it to the runtime;
    /// replay pins it through `--onnx-threads` /
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
    /// injection gate is a floor on this value. An
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

/// The embedder a bank whose recorded model is `recorded` is served with:
/// that model when it's the daemon's own or one of the `previous` models
/// it carries during a change. `None` when the daemon
/// doesn't carry it: another model's vectors aren't comparable with the
/// bank's, so the bank is refused until a re-embed moves it.
pub(crate) fn serving<'a>(
    models: &'a Models,
    previous: &'a [Arc<dyn Embedder>],
    recorded: &str,
) -> Option<&'a dyn Embedder> {
    if models.embedder.model_id() == recorded {
        return Some(models.embedder.as_ref());
    }
    previous
        .iter()
        .find(|embedder| embedder.model_id() == recorded)
        .map(|embedder| embedder.as_ref())
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
