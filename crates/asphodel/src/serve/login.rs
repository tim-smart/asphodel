//! The setup wizard's ChatGPT login: `asphodel llm login`'s device-code
//! flow, run by the daemon.
//!
//! `POST /v1/setup/llm-login` starts the flow on a blocking thread and
//! answers with the verification URL and the code once the issuer hands
//! them out; the thread then waits for the owner's approval and saves the
//! tokens in the data dir, where the LLM client reads them before every
//! call. `GET` reads how the login stands.
//!
//! Like setup, starting a login needs no token, so it is only offered while
//! setup waits: anyone who can reach an unconfigured daemon can already
//! point it at their own endpoint. Once setup is done, `POST` answers 409
//! and `asphodel llm login` is the way to log in again. A login in flight
//! outlives setup, and `GET` keeps answering without a token. It never
//! shows the code: after setup, a reader who could approve it with their
//! own account would have the daemon send memories there. A login in
//! flight stops with the daemon, so shutdown never waits for the owner.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::Context;
use asphodel_core::Clock;
use asphodel_core::clock::StopSleeper;
use asphodel_core::models::{
    AUTH_ISSUER, DeviceCode, LOGIN_WAIT, LoginError, TokenStore, device_code_login_within,
};
use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use serde::Serialize;
use tokio::sync::oneshot;
use tracing::{info, warn};

use super::Shared;
use super::api::ApiError;

/// `ASPHODEL_LLM_LOGIN_WAIT_MS=<ms>` replaces the 15 minutes a login waits
/// for approval, so an integration test can see one expire. It is for
/// integration tests, environment only, never in `--help`.
pub(crate) const LLM_LOGIN_WAIT_ENV: &str = "ASPHODEL_LLM_LOGIN_WAIT_MS";

/// The wizard's login and where it runs.
pub(crate) struct Login {
    data_dir: PathBuf,
    issuer: String,
    /// How long a login waits for approval.
    wait: Duration,
    clock: Arc<dyn Clock>,
    /// What a login waits on between polls; stopped with the daemon.
    sleeper: StopSleeper,
    state: Mutex<Phase>,
    /// Held while a login starts, so two requests can't start two.
    starting: tokio::sync::Mutex<()>,
}

#[derive(Debug, Clone)]
enum Phase {
    None,
    Pending(DeviceCode),
    Done,
    Expired,
    Failed(String),
}

impl Login {
    /// Reads the issuer and wait overrides from the environment; a wait
    /// that isn't a number stops the daemon before it binds.
    pub(crate) fn new(data_dir: PathBuf, clock: Arc<dyn Clock>) -> anyhow::Result<Self> {
        let issuer =
            std::env::var(crate::cli::LLM_ISSUER_ENV).unwrap_or_else(|_| AUTH_ISSUER.to_string());
        let wait = match std::env::var_os(LLM_LOGIN_WAIT_ENV) {
            None => LOGIN_WAIT,
            Some(value) => {
                let wait = value
                    .to_str()
                    .and_then(|value| value.parse().ok())
                    .map(Duration::from_millis)
                    .with_context(|| {
                        format!("{LLM_LOGIN_WAIT_ENV} isn't a whole number of milliseconds")
                    })?;
                warn!("{LLM_LOGIN_WAIT_ENV} is set: a login waits {wait:?} for approval");
                wait
            }
        };
        Ok(Self {
            data_dir,
            issuer,
            wait,
            clock,
            sleeper: StopSleeper::default(),
            state: Mutex::new(Phase::None),
            starting: tokio::sync::Mutex::new(()),
        })
    }

    /// Stops a login in flight, and any started later, at its next poll.
    pub(crate) fn stop(&self) {
        self.sleeper.stop();
    }

    fn phase(&self) -> Phase {
        self.state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .clone()
    }

    fn set(&self, phase: Phase) {
        *self
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner()) = phase;
    }

    /// Runs the flow to its end on this thread, sending the code on `shown`
    /// once the issuer hands it out.
    fn run(&self, shown: oneshot::Sender<DeviceCode>) {
        let store = TokenStore::open(&self.data_dir);
        let mut shown = Some(shown);
        let mut show = |code: &DeviceCode| {
            self.set(Phase::Pending(code.clone()));
            if let Some(shown) = shown.take() {
                let _ = shown.send(code.clone());
            }
        };
        let result = device_code_login_within(
            &self.issuer,
            &store,
            self.clock.as_ref(),
            self.wait,
            &self.sleeper,
            &mut show,
        );
        match result {
            Ok(_) => {
                info!(token_file = %store.path().display(), "the wizard's login is done");
                self.set(Phase::Done);
            }
            Err(LoginError::Stopped) => {
                info!("the wizard's login stopped with the daemon");
                self.set(Phase::Failed(LoginError::Stopped.to_string()));
            }
            Err(LoginError::Expired) => {
                warn!("the wizard's login wasn't approved in time");
                self.set(Phase::Expired);
            }
            Err(error) => {
                warn!(%error, "the wizard's login failed");
                self.set(Phase::Failed(error.to_string()));
            }
        }
    }
}

/// `GET /v1/setup/llm-login`'s body. `error` says why a login failed.
#[derive(Debug, Serialize)]
pub(crate) struct LoginState {
    state: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

/// `POST /v1/setup/llm-login`'s body: what the owner opens and enters.
#[derive(Debug, Serialize)]
pub(crate) struct LoginStarted {
    state: &'static str,
    verification_url: String,
    user_code: String,
}

impl From<DeviceCode> for LoginStarted {
    fn from(code: DeviceCode) -> Self {
        Self {
            state: "pending",
            verification_url: code.verification_url,
            user_code: code.user_code,
        }
    }
}

/// `GET /v1/setup/llm-login`: how the wizard's login stands, before and
/// after setup. No token, and never the code.
pub(crate) async fn state(State(app): State<Shared>) -> Json<LoginState> {
    let (state, error) = match app.login.phase() {
        Phase::None => ("none", None),
        Phase::Pending(_) => ("pending", None),
        Phase::Done => ("done", None),
        Phase::Expired => ("expired", None),
        Phase::Failed(error) => ("failed", Some(error)),
    };
    Json(LoginState { state, error })
}

/// `POST /v1/setup/llm-login`: starts a login and answers with its code, or
/// with the code of the one already waiting. 409 once setup is done; 502
/// when the issuer hands out no code.
pub(crate) async fn start(State(app): State<Shared>) -> Result<Json<LoginStarted>, ApiError> {
    let _starting = app.login.starting.lock().await;
    if !app.setting_up() {
        return Err(ApiError::new(
            StatusCode::CONFLICT,
            "setup is done; to log in again, run `asphodel llm login`",
        ));
    }
    if let Phase::Pending(code) = app.login.phase() {
        return Ok(Json(code.into()));
    }
    let (shown, code) = oneshot::channel();
    let login = Arc::clone(&app);
    tokio::task::spawn_blocking(move || login.login.run(shown));
    match code.await {
        Ok(code) => Ok(Json(code.into())),
        // The flow ended before the issuer handed out a code.
        Err(_) => Err(match app.login.phase() {
            Phase::Failed(error) => ApiError::new(StatusCode::BAD_GATEWAY, error),
            _ => ApiError::new(StatusCode::BAD_GATEWAY, "the issuer handed out no code"),
        }),
    }
}
