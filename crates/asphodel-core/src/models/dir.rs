//! Where the models live, and what goes wrong with them.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use super::manifest::ModelSpec;

/// The model dir (TIM-94, decision 4). `ASPHODEL_MODEL_DIR` overrides it,
/// and the default is the XDG cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelDir {
    path: PathBuf,
}

impl ModelDir {
    /// Resolves the dir from an explicit override (the flag or
    /// `ASPHODEL_MODEL_DIR`), else `$XDG_CACHE_HOME/asphodel/models`, else
    /// `$HOME/.cache/asphodel/models`. A relative `XDG_CACHE_HOME` is
    /// ignored, as the XDG spec says. The binary reads the environment;
    /// this takes values so tests never set process-wide variables.
    pub fn resolve(
        override_dir: Option<&Path>,
        xdg_cache_home: Option<&Path>,
        home: Option<&Path>,
    ) -> Result<Self, ModelError> {
        if let Some(dir) = override_dir {
            return Ok(Self::at(dir));
        }
        if let Some(cache) = xdg_cache_home.filter(|cache| cache.is_absolute()) {
            return Ok(Self::at(cache.join("asphodel").join("models")));
        }
        if let Some(home) = home {
            return Ok(Self::at(
                home.join(".cache").join("asphodel").join("models"),
            ));
        }
        Err(ModelError::NoModelDir)
    }

    /// A dir at a known path, for tests and `models fetch --model-dir`.
    pub fn at(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// `<dir>/<spec.dir>/<name>`.
    pub fn file(&self, spec: &ModelSpec, name: &str) -> PathBuf {
        self.path.join(&spec.dir).join(name)
    }

    /// Reads every file of `spec`, in manifest order, checking each one's
    /// bytes against the manifest. A missing file is
    /// [`ModelError::MissingFile`]; wrong bytes are [`ModelError::Checksum`].
    pub(crate) fn read_verified(&self, spec: &ModelSpec) -> Result<Vec<Vec<u8>>, ModelError> {
        spec.files
            .iter()
            .map(|file| {
                let path = self.file(spec, &file.name);
                let bytes = std::fs::read(&path).map_err(|_| ModelError::MissingFile {
                    model: spec.id.clone(),
                    path: path.clone(),
                })?;
                if sha256_hex(&bytes) != file.sha256 {
                    return Err(ModelError::Checksum {
                        model: spec.id.clone(),
                        path,
                    });
                }
                Ok(bytes)
            })
            .collect()
    }
}

/// The lowercase hex SHA-256 of `bytes`, as the manifest writes it.
pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Why the models couldn't be found, loaded or run. No variant carries
/// text that was embedded or reranked (TIM-96, decision 8).
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("no model dir: set ASPHODEL_MODEL_DIR, XDG_CACHE_HOME or HOME")]
    NoModelDir,

    /// A file from the manifest isn't there.
    #[error("{model} is missing {}: run `asphodel models fetch`", path.display())]
    MissingFile { model: String, path: PathBuf },

    /// A file is there but its bytes don't match the manifest.
    #[error("{model} file {} doesn't match its checksum: run `asphodel models fetch`", path.display())]
    Checksum { model: String, path: PathBuf },

    /// ONNX Runtime or the tokenizer refused the files.
    #[error("loading {model}: {reason}")]
    Load { model: String, reason: String },

    /// A call into a loaded model failed.
    #[error("running {model}: {reason}")]
    Inference { model: String, reason: String },
}
