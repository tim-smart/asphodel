//! The real models, run by fastembed on ONNX Runtime.
//!
//! Both are loaded as user-defined models from bytes the manifest has
//! already verified, with fastembed's Hugging Face download feature
//! compiled out. ONNX Runtime itself is loaded at runtime from
//! `ORT_DYLIB_PATH` (ort's `load-dynamic`), which the nix shell and the
//! image point at nixpkgs' library (TIM-89).

use std::sync::Mutex;

use fastembed::{
    InitOptionsUserDefined, Pooling, QuantizationMode, RerankInitOptionsUserDefined, TextEmbedding,
    TextRerank, TokenizerFiles, UserDefinedEmbeddingModel, UserDefinedRerankingModel,
};

use super::{Embedder, ModelError, ModelOptions, Reranker, normalise};
use crate::store::vector::EMBEDDING_DIMENSIONS;

/// bge-small's context; longer inputs are truncated.
const MAX_LENGTH: usize = 512;

/// bge-small-en-v1.5 int8 behind fastembed.
pub struct OnnxEmbedder {
    model_id: String,
    model: Mutex<TextEmbedding>,
}

impl OnnxEmbedder {
    /// `files` are the manifest's five files, in manifest order.
    pub(crate) fn new(
        model_id: &str,
        files: Vec<Vec<u8>>,
        options: &ModelOptions,
    ) -> Result<Self, ModelError> {
        let (onnx, tokenizer) = split(files);
        // bge pools on [CLS] (fastembed's own setting for the model). The
        // graph is dynamically quantised, so texts are embedded one at a
        // time (see `embed`), and fastembed's batching guard isn't needed.
        let model = UserDefinedEmbeddingModel::new(onnx, tokenizer)
            .with_pooling(Pooling::Cls)
            .with_quantization(QuantizationMode::None);
        let mut init = InitOptionsUserDefined::new().with_max_length(MAX_LENGTH);
        if let Some(threads) = options.threads {
            init = init.with_intra_threads(threads.get());
        }
        let model = TextEmbedding::try_new_from_user_defined(model, init).map_err(|error| {
            ModelError::Load {
                model: model_id.to_string(),
                reason: error.to_string(),
            }
        })?;
        Ok(Self {
            model_id: model_id.to_string(),
            model: Mutex::new(model),
        })
    }
}

impl Embedder for OnnxEmbedder {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn dimensions(&self) -> usize {
        EMBEDDING_DIMENSIONS
    }

    /// One text per session run. With dynamic quantisation the activation
    /// range is set per batch, so a text's vector would depend on what it
    /// was batched with; one at a time, a text always gets the same vector,
    /// which the reconcile floor relies on.
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, ModelError> {
        let mut model = self
            .model
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        texts
            .iter()
            .map(|text| {
                let mut vectors =
                    model
                        .embed([*text], None)
                        .map_err(|error| ModelError::Inference {
                            model: self.model_id.clone(),
                            reason: error.to_string(),
                        })?;
                let mut vector = vectors.pop().ok_or_else(|| ModelError::Inference {
                    model: self.model_id.clone(),
                    reason: "no embedding came back".into(),
                })?;
                if vector.len() != EMBEDDING_DIMENSIONS {
                    return Err(ModelError::Inference {
                        model: self.model_id.clone(),
                        reason: format!(
                            "embedding has {} dimensions, expected {EMBEDDING_DIMENSIONS}",
                            vector.len()
                        ),
                    });
                }
                normalise(&mut vector);
                Ok(vector)
            })
            .collect()
    }
}

/// jina-reranker-v1-turbo-en int8 behind fastembed.
pub struct OnnxReranker {
    model_id: String,
    model: Mutex<TextRerank>,
}

impl OnnxReranker {
    /// `files` are the manifest's five files, in manifest order.
    pub(crate) fn new(
        model_id: &str,
        files: Vec<Vec<u8>>,
        options: &ModelOptions,
    ) -> Result<Self, ModelError> {
        let (onnx, tokenizer) = split(files);
        let model = UserDefinedRerankingModel::new(onnx, tokenizer);
        let mut init = RerankInitOptionsUserDefined::new().with_max_length(MAX_LENGTH);
        if let Some(threads) = options.threads {
            init = init.with_intra_threads(threads.get());
        }
        let model = TextRerank::try_new_from_user_defined(model, init).map_err(|error| {
            ModelError::Load {
                model: model_id.to_string(),
                reason: error.to_string(),
            }
        })?;
        Ok(Self {
            model_id: model_id.to_string(),
            model: Mutex::new(model),
        })
    }
}

impl Reranker for OnnxReranker {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<f32>, ModelError> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }
        let mut model = self
            .model
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let results = model
            .rerank(query, documents, false, None)
            .map_err(|error| ModelError::Inference {
                model: self.model_id.clone(),
                reason: error.to_string(),
            })?;
        // fastembed sorts by score; put the logits back in the documents'
        // order so the caller keeps its ids.
        let mut scores = vec![f32::NAN; documents.len()];
        for result in results {
            if let Some(slot) = scores.get_mut(result.index) {
                *slot = result.score;
            }
        }
        if scores.iter().any(|score| score.is_nan()) {
            return Err(ModelError::Inference {
                model: self.model_id.clone(),
                reason: "a document came back without a score".into(),
            });
        }
        Ok(scores)
    }
}

/// Splits the manifest's five files into the graph and the tokenizer set.
fn split(files: Vec<Vec<u8>>) -> (Vec<u8>, TokenizerFiles) {
    let mut files = files.into_iter();
    let onnx = files.next().expect("the manifest lists model.onnx first");
    let tokenizer = TokenizerFiles {
        tokenizer_file: files.next().expect("tokenizer.json"),
        config_file: files.next().expect("config.json"),
        special_tokens_map_file: files.next().expect("special_tokens_map.json"),
        tokenizer_config_file: files.next().expect("tokenizer_config.json"),
    };
    (onnx, tokenizer)
}
