//! First-run setup: the tuning file and secrets kept in the data dir, and
//! the wizard that writes them.
//!
//! Without `--config`, the daemon reads its tuning file from
//! [`CONFIG_FILE`] in the data dir. When that isn't there either, it binds,
//! answers `/v1/health` with `setup: true`, and waits for `POST /v1/setup`
//! to write it. The dashboard's wizard makes that call. Once the file is
//! written, startup goes on exactly as a restart would read it.
//!
//! Setup asks for no code, token or credentials, and nothing guards it:
//! anyone who can reach an unconfigured daemon can complete setup, choosing
//! its LLM and receiving the token it makes. Keep the daemon private until
//! setup is done. Setup runs once; after it, `POST /v1/setup` answers 409,
//! and every other route needs the token, when one is configured, as
//! before.
//!
//! The secrets the wizard keeps (the bearer token and the LLM's API key) go
//! in [`SECRETS_FILE`], mode 0600. The environment wins over each of them.

use std::fs::{self, File, OpenOptions};
use std::io::{ErrorKind, Read, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::Context;
use asphodel_core::Tuning;
use asphodel_core::config::{Deployment, LLM_API_KEY_ENV, LlmAuth, Secret, TOKEN_ENV};
use asphodel_core::models::LlmSettings;
use axum::Json;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;
use tracing::info;

use super::Shared;
use super::api::ApiError;

/// The tuning file in the data dir, read when `--config` isn't given.
pub(crate) const CONFIG_FILE: &str = "asphodel.toml";

/// The secrets the wizard keeps, as TOML: `token` and `llm_api_key`.
pub(crate) const SECRETS_FILE: &str = "secrets.toml";

/// The tuning file in `data_dir`, if there is one.
pub(crate) fn data_dir_config(data_dir: &Path) -> Option<PathBuf> {
    let path = data_dir.join(CONFIG_FILE);
    path.exists().then_some(path)
}

/// The secrets kept in the data dir. Each is `None` when the file or the
/// key isn't there.
#[derive(Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct StoredSecrets {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) token: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) llm_api_key: Option<String>,
}

impl StoredSecrets {
    /// Reads [`SECRETS_FILE`] from `data_dir`. A missing file is no secrets;
    /// an unreadable or malformed one stops the daemon, and the error never
    /// quotes the file.
    pub(crate) fn read(data_dir: &Path) -> anyhow::Result<Self> {
        let path = data_dir.join(SECRETS_FILE);
        let text = match fs::read_to_string(&path) {
            Ok(text) => text,
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(Self::default()),
            Err(error) => {
                return Err(error).with_context(|| format!("reading {}", path.display()));
            }
        };
        toml::from_str(&text).map_err(|_| {
            anyhow::anyhow!(
                "{} isn't valid: it takes `token` and `llm_api_key`, both strings",
                path.display()
            )
        })
    }

    /// A stored secret, with an empty one counting as absent.
    fn secret(value: &Option<String>) -> Option<Secret> {
        value
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(Secret::new)
    }

    pub(crate) fn token(&self) -> Option<Secret> {
        Self::secret(&self.token)
    }

    pub(crate) fn llm_api_key(&self) -> Option<Secret> {
        Self::secret(&self.llm_api_key)
    }

    fn write(&self, data_dir: &Path) -> anyhow::Result<()> {
        let text = toml::to_string(self).context("writing the secrets")?;
        write_private(&data_dir.join(SECRETS_FILE), text.as_bytes())
    }
}

/// A secret from the environment, else from the data dir.
pub(crate) fn secret(env: &str, stored: Option<Secret>) -> Option<Secret> {
    Secret::from_env(env).or(stored)
}

/// The floors the wizard writes for the models the daemon runs. They are
/// starting values, written into the file for the operator to recalibrate;
/// the daemon itself still has no fallback.
#[derive(Debug, Clone)]
pub(crate) struct StartingFloors {
    /// Each embedding model and its reconcile floor.
    pub(crate) embedding: Vec<(&'static str, f64)>,
    /// The reranker, its gate floor and its relevance scale.
    pub(crate) reranker: (&'static str, f64, f64),
    /// The embedding models whose floor is a placeholder nobody has
    /// calibrated yet. The wizard warns about them before setup, and the
    /// file marks them.
    pub(crate) uncalibrated: Vec<&'static str>,
}

impl StartingFloors {
    /// Each placeholder floor as the tuning file sets it.
    fn uncalibrated(&self) -> Vec<String> {
        let quote = |value: &str| toml::Value::String(value.to_owned()).to_string();
        self.embedding
            .iter()
            .filter(|(model, _)| self.uncalibrated.contains(model))
            .map(|(model, floor)| {
                format!("reconcile.embedding_floors.{} = {floor:?}", quote(model))
            })
            .collect()
    }
}

/// Setup waiting for the wizard.
pub(crate) struct Pending {
    data_dir: PathBuf,
    /// Whether a token already exists, from the environment or the data dir.
    token_configured: bool,
    /// Whether the daemon listens off loopback, so needs a token.
    needs_token: bool,
    env_llm_api_key: Option<Secret>,
    floors: StartingFloors,
    done: oneshot::Sender<()>,
}

impl Pending {
    /// The pending setup, and what resolves once it's done.
    pub(crate) fn new(
        data_dir: &Path,
        needs_token: bool,
        floors: StartingFloors,
    ) -> anyhow::Result<(Self, oneshot::Receiver<()>)> {
        fs::create_dir_all(data_dir)
            .with_context(|| format!("creating the data dir {}", data_dir.display()))?;
        let stored = StoredSecrets::read(data_dir)?;
        let token_configured = Secret::from_env(TOKEN_ENV).is_some() || stored.token().is_some();
        let (done, finished) = oneshot::channel();
        Ok((
            Self {
                data_dir: data_dir.to_owned(),
                token_configured,
                needs_token,
                env_llm_api_key: Secret::from_env(LLM_API_KEY_ENV),
                floors,
                done,
            },
            finished,
        ))
    }

    /// Validates `request`, and writes the secrets and the tuning file. The
    /// tuning file goes last: it's what says
    /// setup is done, so a failure before it leaves setup to try again.
    fn complete(&self, request: SetupRequest) -> Result<SetupDone, ApiError> {
        let llm = request.llm.map(LlmChoice::normalized).transpose()?;
        if let Some(llm) = &llm
            && llm.api_key.is_some()
        {
            if llm.auth == LlmAuth::Chatgpt {
                return Err(bad_request(
                    "a ChatGPT subscription doesn't use an API key; leave it out and run `asphodel llm login` after setup",
                ));
            }
            if self.env_llm_api_key.is_some() {
                return Err(bad_request(format!(
                    "{LLM_API_KEY_ENV} is set and overrides any key given here, so leave the key out"
                )));
            }
        }
        let text = tuning_text(llm.as_ref(), &self.floors);
        let tuning = Tuning::from_toml(&text).map_err(|error| bad_request(error.to_string()))?;
        let mut stored = StoredSecrets::read(&self.data_dir).map_err(ApiError::internal)?;
        let api_key = llm.as_ref().and_then(|llm| llm.api_key.clone());
        let deployment = Deployment {
            listen: String::new(),
            data_dir: Some(self.data_dir.clone()),
            config: None,
            allow_network_fs: false,
            model_dir: None,
            token: None,
            llm_api_key: self
                .env_llm_api_key
                .clone()
                .or_else(|| api_key.clone().map(Secret::new))
                .or_else(|| stored.llm_api_key()),
        };
        LlmSettings::from_config(&tuning, &deployment)
            .map_err(|error| bad_request(error.to_string()))?;

        let token = (self.needs_token && !self.token_configured)
            .then(|| random_hex(32))
            .transpose()
            .map_err(ApiError::internal)?;
        if token.is_some() || api_key.is_some() {
            stored.token = token.clone().or(stored.token);
            stored.llm_api_key = api_key.or(stored.llm_api_key);
            stored.write(&self.data_dir).map_err(ApiError::internal)?;
        }
        let config = self.data_dir.join(CONFIG_FILE);
        write_file(&config, text.as_bytes(), 0o644).map_err(ApiError::internal)?;
        info!(config = %config.display(), "setup wrote the tuning file");
        Ok(SetupDone {
            data_dir: self.data_dir.clone(),
            config,
            secrets: self.data_dir.join(SECRETS_FILE),
            token,
            llm_login: llm.is_some_and(|llm| llm.auth == LlmAuth::Chatgpt),
        })
    }
}

/// The setup slot the HTTP handlers share: `Some` while setup waits.
pub(crate) type Slot = Mutex<Option<Pending>>;

/// `GET /v1/setup`: whether setup is waiting, and what it will need. No
/// token: it says nothing the health check doesn't.
#[derive(Debug, Serialize)]
pub(crate) struct SetupState {
    needed: bool,
    /// A token already exists, so setup won't make one.
    token_configured: bool,
    /// Setup makes a token, since the daemon listens off loopback.
    makes_token: bool,
    /// `ASPHODEL_LLM_API_KEY` is set, so setup takes no key.
    llm_api_key_from_env: bool,
    /// The floors setup would write that are placeholders, not calibrated,
    /// as the tuning file sets them.
    uncalibrated: Vec<String>,
}

/// `POST /v1/setup`'s body. `llm` left out sets the LLM up later: chunks
/// wait on the queue until it's configured.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SetupRequest {
    #[serde(default)]
    llm: Option<LlmChoice>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LlmChoice {
    #[serde(default)]
    auth: LlmAuth,
    #[serde(default)]
    endpoint: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    reasoning_effort: Option<String>,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    api_key: Option<String>,
}

impl LlmChoice {
    /// Trims every field, an empty one counting as left out, and checks the
    /// model is there.
    fn normalized(self) -> Result<Self, ApiError> {
        let trim = |value: Option<String>| {
            value
                .map(|value| value.trim().to_owned())
                .filter(|value| !value.is_empty())
        };
        let choice = Self {
            auth: self.auth,
            endpoint: trim(self.endpoint),
            model: trim(self.model),
            reasoning_effort: trim(self.reasoning_effort),
            language: trim(self.language),
            api_key: trim(self.api_key),
        };
        if choice.model.is_none() {
            return Err(bad_request("llm.model is required"));
        }
        if choice.auth == LlmAuth::ApiKey && choice.endpoint.is_none() {
            return Err(bad_request("llm.endpoint is required with an API key"));
        }
        Ok(choice)
    }
}

/// What setup did. `token` is the bearer token it made, shown this once;
/// `None` when one was already configured or none is needed.
#[derive(Debug, Serialize)]
pub(crate) struct SetupDone {
    data_dir: PathBuf,
    config: PathBuf,
    secrets: PathBuf,
    token: Option<String>,
    /// The LLM is a ChatGPT subscription, which waits for
    /// `asphodel llm login`.
    llm_login: bool,
}

pub(crate) async fn state(State(app): State<Shared>) -> Json<SetupState> {
    let slot = app
        .setup
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    Json(match slot.as_ref() {
        Some(pending) => SetupState {
            needed: true,
            token_configured: pending.token_configured,
            makes_token: pending.needs_token && !pending.token_configured,
            llm_api_key_from_env: pending.env_llm_api_key.is_some(),
            uncalibrated: pending.floors.uncalibrated(),
        },
        None => SetupState {
            needed: false,
            token_configured: false,
            makes_token: false,
            llm_api_key_from_env: false,
            uncalibrated: Vec::new(),
        },
    })
}

/// `POST /v1/setup`: writes the config and lets startup go on. It needs no
/// token: whoever reaches an unconfigured daemon first sets it up. 409 once
/// setup is done or when it was never needed.
pub(crate) async fn complete(
    State(app): State<Shared>,
    body: Result<Json<SetupRequest>, JsonRejection>,
) -> Result<Json<SetupDone>, ApiError> {
    let Json(body) = body?;
    let app = std::sync::Arc::clone(&app);
    tokio::task::spawn_blocking(move || {
        let mut slot = app
            .setup
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        // An earlier request may already have finished setup.
        let pending = slot.as_ref().ok_or_else(|| {
            ApiError::new(
                StatusCode::CONFLICT,
                "setup is already done; to change a setting, edit the tuning file in the data dir and restart",
            )
        })?;
        let done = pending.complete(body)?;
        if let Some(pending) = slot.take() {
            let _ = pending.done.send(());
        }
        Ok(Json(done))
    })
    .await
    .map_err(ApiError::internal)?
}

/// The tuning file the wizard writes: the LLM, if one was chosen, and the
/// starting floors for the models the daemon runs.
fn tuning_text(llm: Option<&LlmChoice>, floors: &StartingFloors) -> String {
    let quote = |value: &str| toml::Value::String(value.to_owned()).to_string();
    let mut text = String::from(
        "# Written by the setup wizard. To change a setting, edit this file and\n\
         # restart the daemon. docs/operations.md, \"The tuning file\", lists every key.\n",
    );
    match llm {
        Some(llm) => {
            text.push_str("\n[llm]\n");
            let auth = match llm.auth {
                LlmAuth::ApiKey => "api_key",
                LlmAuth::Chatgpt => "chatgpt",
            };
            text.push_str(&format!("auth = {}\n", quote(auth)));
            for (key, value) in [
                ("endpoint", &llm.endpoint),
                ("model", &llm.model),
                ("reasoning_effort", &llm.reasoning_effort),
                ("language", &llm.language),
            ] {
                if let Some(value) = value {
                    text.push_str(&format!("{key} = {}\n", quote(value)));
                }
            }
            if llm.auth == LlmAuth::ApiKey {
                text.push_str(&format!(
                    "# The API key is in {SECRETS_FILE} beside this file, or {LLM_API_KEY_ENV}.\n"
                ));
            }
        }
        None => {
            text.push_str("\n# No LLM yet: chunks wait on the queue until [llm] is set here.\n")
        }
    }
    text.push_str(
        "\n# Starting floors for the models this daemon runs. Recalibrate them in\n\
         # replay (docs/replay.md, \"Labelling and the precision curve\").\n\
         [reconcile.embedding_floors]\n",
    );
    for (model, floor) in &floors.embedding {
        if floors.uncalibrated.contains(model) {
            text.push_str(
                "# A placeholder nobody has calibrated yet. It decides which claims\n\
                 # reconcile against an existing memory, so calibrate it before relying\n\
                 # on the store.\n",
            );
        }
        text.push_str(&format!("{} = {floor:?}\n", quote(model)));
    }
    let (reranker, floor, scale) = floors.reranker;
    text.push_str(&format!(
        "\n[injection.reranker_floors]\n{} = {floor:?}\n\n[ranking.relevance_scales]\n{} = {scale:?}\n",
        quote(reranker),
        quote(reranker)
    ));
    text
}

/// `bytes` random bytes from the kernel, as lowercase hex.
fn random_hex(bytes: usize) -> anyhow::Result<String> {
    let mut buffer = vec![0u8; bytes];
    File::open("/dev/urandom")
        .and_then(|mut random| random.read_exact(&mut buffer))
        .context("reading /dev/urandom")?;
    Ok(buffer.iter().map(|byte| format!("{byte:02x}")).collect())
}

/// Writes `path` readable by its owner only.
fn write_private(path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    write_file(path, bytes, 0o600)
}

/// Writes `path` through a temp file and a rename, so a crash never leaves
/// half a file, with `mode` set when the file is made.
fn write_file(path: &Path, bytes: &[u8], mode: u32) -> anyhow::Result<()> {
    let temp = path.with_extension("tmp");
    let _ = fs::remove_file(&temp);
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(mode)
        .open(&temp)
        .with_context(|| format!("creating {}", temp.display()))?;
    file.write_all(bytes)
        .and_then(|()| file.sync_all())
        .with_context(|| format!("writing {}", temp.display()))?;
    fs::rename(&temp, path).with_context(|| format!("writing {}", path.display()))?;
    // The rename is durable only once the directory is synced, so a file
    // written before another is on disk before it, across a power loss too.
    let dir = path.parent().unwrap_or(Path::new("."));
    File::open(dir)
        .and_then(|dir| dir.sync_all())
        .with_context(|| format!("syncing {}", dir.display()))?;
    Ok(())
}

fn bad_request(message: impl Into<String>) -> ApiError {
    ApiError::new(StatusCode::BAD_REQUEST, message)
}
