//! `asphodel models fetch`: filling the model dir from the manifest.

use std::path::PathBuf;
use std::time::Duration;

use super::dir::{ModelDir, sha256_hex};
use super::manifest::ModelSpec;

/// Fetches one URL. `models fetch` uses [`HttpFetcher`]; tests use a map
/// of canned bytes.
pub trait Fetcher: Send + Sync {
    fn fetch(&self, url: &str) -> Result<Vec<u8>, FetchError>;
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum FetchError {
    #[error("transport: {0}")]
    Transport(String),
    #[error("HTTP {0}")]
    Status(u16),
}

/// What one `models fetch` did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FetchReport {
    /// Files written, in manifest order.
    pub fetched: Vec<PathBuf>,
    /// Files already present with the right checksum, in manifest order.
    pub skipped: Vec<PathBuf>,
}

#[derive(Debug, thiserror::Error)]
pub enum FetchFailure {
    #[error("fetching {url}: {error}")]
    Fetch { url: String, error: FetchError },

    /// The bytes came back but don't match the manifest. Nothing is
    /// written.
    #[error("{url} doesn't match the checksum in the manifest")]
    Checksum { url: String },

    #[error("writing {}: {error}", path.display())]
    Io {
        path: PathBuf,
        #[source]
        error: std::io::Error,
    },
}

/// Fills `dir` from `specs`, one file at a time in manifest order. A file
/// already present with the right checksum is skipped without a fetch. A
/// fetched file is checked against its checksum, then written to a temp
/// name and renamed into place, so a crash or a bad download never leaves
/// a partial file at the final path. The first failure stops the run; what
/// was written stays, so the next run resumes.
pub fn fetch_models(
    dir: &ModelDir,
    specs: &[ModelSpec],
    fetcher: &dyn Fetcher,
) -> Result<FetchReport, FetchFailure> {
    let mut report = FetchReport::default();
    for spec in specs {
        for file in &spec.files {
            let path = dir.file(spec, &file.name);
            if let Ok(bytes) = std::fs::read(&path)
                && sha256_hex(&bytes) == file.sha256
            {
                report.skipped.push(path);
                continue;
            }
            let bytes = fetcher
                .fetch(&file.url)
                .map_err(|error| FetchFailure::Fetch {
                    url: file.url.clone(),
                    error,
                })?;
            if sha256_hex(&bytes) != file.sha256 {
                return Err(FetchFailure::Checksum {
                    url: file.url.clone(),
                });
            }
            write_atomically(&path, &bytes)?;
            report.fetched.push(path);
        }
    }
    Ok(report)
}

/// Writes `bytes` to `path` through a temp file in the same directory and
/// a rename, creating the directory if needed.
fn write_atomically(path: &std::path::Path, bytes: &[u8]) -> Result<(), FetchFailure> {
    let io = |error| FetchFailure::Io {
        path: path.to_path_buf(),
        error,
    };
    let parent = path.parent().expect("a model file has a directory");
    std::fs::create_dir_all(parent).map_err(io)?;
    let mut temp = path.as_os_str().to_owned();
    temp.push(format!(".part-{}", std::process::id()));
    let temp = PathBuf::from(temp);
    let written = std::fs::write(&temp, bytes).and_then(|()| std::fs::rename(&temp, path));
    if written.is_err() {
        let _ = std::fs::remove_file(&temp);
    }
    written.map_err(io)
}

/// The HTTPS fetcher `asphodel models fetch` uses.
pub struct HttpFetcher {
    agent: ureq::Agent,
}

impl HttpFetcher {
    /// The largest file the manifest lists is under 40 MB; this leaves room
    /// for a bigger model later without reading without bound.
    const LIMIT: u64 = 512 * 1024 * 1024;

    pub fn new() -> Self {
        let config = ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(600)))
            .http_status_as_error(false)
            .build();
        Self {
            agent: config.into(),
        }
    }
}

impl Default for HttpFetcher {
    fn default() -> Self {
        Self::new()
    }
}

impl Fetcher for HttpFetcher {
    fn fetch(&self, url: &str) -> Result<Vec<u8>, FetchError> {
        let mut response = self
            .agent
            .get(url)
            .call()
            .map_err(|error| FetchError::Transport(error.to_string()))?;
        let status = response.status().as_u16();
        if !(200..300).contains(&status) {
            return Err(FetchError::Status(status));
        }
        response
            .body_mut()
            .with_config()
            .limit(Self::LIMIT)
            .read_to_vec()
            .map_err(|error| FetchError::Transport(error.to_string()))
    }
}
