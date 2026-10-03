//! The manifest: the one place the models' files, their sources and their
//! checksums are written down. `asphodel models fetch` fills the model dir
//! from it, and [`Models::load`] checks the dir against it.
//!
//! Both models are pinned to a Hugging Face revision, so a repository
//! update can't change what the daemon runs, and every file carries the
//! SHA-256 of its bytes, verified against the repositories' LFS metadata
//! and a download: bge-small on 2026-10-01, the reranker on 2026-10-03.
//!
//! [`Models::load`]: super::Models::load

/// bge-small-en-v1.5, int8, 384 dimensions. The exact string,
/// quantisation included, keys the reconcile floor and is what a bank
/// records: int8 and fp32 give different scores.
pub const EMBEDDING_MODEL_ID: &str = "bge-small-en-v1.5:int8";

/// ms-marco-MiniLM-L-6-v2, int8. Keys the gate floor and the relevance
/// scale.
pub const RERANKER_MODEL_ID: &str = "ms-marco-MiniLM-L-6-v2:int8";

/// The files every model needs: the ONNX graph and the four tokenizer
/// files fastembed loads a user-defined model from.
pub const MODEL_FILES: [&str; 5] = [
    "model.onnx",
    "tokenizer.json",
    "config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
];

/// `Xenova/bge-small-en-v1.5`, the ONNX export of `BAAI/bge-small-en-v1.5`.
const EMBEDDING_REPO: &str = "Xenova/bge-small-en-v1.5";
const EMBEDDING_REVISION: &str = "ea104dacec62c0de699686887e3f920caeb4f3e3";

/// `Xenova/ms-marco-MiniLM-L-6-v2`, the ONNX export of
/// `cross-encoder/ms-marco-MiniLM-L-6-v2`. Its tokenizer is bge-small's, so
/// `tokenizer.json` and `special_tokens_map.json` share bge-small's
/// checksums.
const RERANKER_REPO: &str = "Xenova/ms-marco-MiniLM-L-6-v2";
const RERANKER_REVISION: &str = "a09144355adeed5f58c8ed011d209bf8ee5a1fec";

/// One file of a model: where it goes under the model's directory, where
/// `models fetch` gets it, and the SHA-256 of its bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelFile {
    pub name: String,
    pub url: String,
    /// Lowercase hex, 64 characters.
    pub sha256: String,
}

/// One model: its id, the directory it lives in under the model dir (no
/// `:`, so the layout is the same on every filesystem), and its files.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSpec {
    pub id: String,
    pub dir: String,
    pub files: Vec<ModelFile>,
}

/// The two models the daemon runs, embedding first, then reranker.
pub fn manifest() -> Vec<ModelSpec> {
    vec![
        spec(
            EMBEDDING_MODEL_ID,
            "bge-small-en-v1.5-int8",
            EMBEDDING_REPO,
            EMBEDDING_REVISION,
            [
                // 34,014,426 bytes: dynamic int8 (`model_quantized.onnx`).
                "6c9c6101a956d62dfb5e7190c538226c0c5bb9cb27b651234b6df063ee7dbfe4",
                "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
                "fa73f90bf92c8cace1fbcb709626306f2bdbc9ea3e5b5f94b440df9b6aa56350",
                "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3",
                "9261e7d79b44c8195c1cada2b453e55b00aeb81e907a6664974b4d7776172ab3",
            ],
        ),
        spec(
            RERANKER_MODEL_ID,
            "ms-marco-MiniLM-L-6-v2-int8",
            RERANKER_REPO,
            RERANKER_REVISION,
            [
                // 23,143,499 bytes: dynamic int8 (`model_quantized.onnx`).
                "e9d8ebf845c413e981c175bfe49a3bfa9b3dcce2a3ba54875ee5df5a58639fbe",
                "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
                "d827779a72d27ae68cf878a6fc2e954542663fe21ca515d9f4783fc96be2d37e",
                "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3",
                "0b29c7bfc889e53b36d9dd3e686dd4300f6525110eaa98c76a5dafceb2029f53",
            ],
        ),
    ]
}

/// The path of each of [`MODEL_FILES`] in a Hugging Face repository. The
/// int8 graph is the `onnx/model_quantized.onnx` export; the tokenizer
/// files sit at the root.
const REPO_PATHS: [&str; 5] = [
    "onnx/model_quantized.onnx",
    "tokenizer.json",
    "config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
];

fn spec(id: &str, dir: &str, repo: &str, revision: &str, sha256: [&str; 5]) -> ModelSpec {
    let files = MODEL_FILES
        .iter()
        .zip(REPO_PATHS)
        .zip(sha256)
        .map(|((name, path), sha256)| ModelFile {
            name: (*name).to_string(),
            url: format!("https://huggingface.co/{repo}/resolve/{revision}/{path}"),
            sha256: sha256.to_string(),
        })
        .collect();
    ModelSpec {
        id: id.to_string(),
        dir: dir.to_string(),
        files,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_manifest_is_well_formed() {
        let manifest = manifest();
        assert_eq!(manifest.len(), 2);
        assert_eq!(manifest[0].id, EMBEDDING_MODEL_ID);
        assert_eq!(manifest[1].id, RERANKER_MODEL_ID);
        for spec in &manifest {
            assert!(!spec.dir.contains([':', '/']), "{}", spec.dir);
            let names: Vec<_> = spec.files.iter().map(|f| f.name.as_str()).collect();
            assert_eq!(names, MODEL_FILES);
            for file in &spec.files {
                assert_eq!(file.sha256.len(), 64, "{}", file.name);
                assert!(
                    file.sha256
                        .bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
                    "{}",
                    file.sha256
                );
                assert!(
                    file.url.starts_with("https://huggingface.co/"),
                    "{}",
                    file.url
                );
                assert!(file.url.contains("/resolve/"), "{}", file.url);
            }
        }
        let urls: std::collections::BTreeSet<_> = manifest
            .iter()
            .flat_map(|s| s.files.iter().map(|f| &f.url))
            .collect();
        assert_eq!(urls.len(), 10, "every file has its own URL");
        assert_ne!(manifest[0].dir, manifest[1].dir, "the models share a dir");
    }
}
