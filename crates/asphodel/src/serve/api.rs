//! The HTTP API under `/v1` (TIM-94, decision 10). JSON bodies are the
//! service layer's own types, and every handler is a thin call into
//! [`Service`] on a blocking thread, the same calls the replay harness makes
//! (TIM-96, decision 3).
//!
//! `/v1/health` answers before the daemon is ready, with 503, so a
//! supervisor can gate readiness on it. Every other route answers 503 until
//! the daemon is ready, and needs the bearer token when one is configured.
//! After SIGTERM, health and ingest answer 503 while the daemon drains.

use std::io::Read;
use std::sync::Arc;

use asphodel_core::agenda::Agenda;
use asphodel_core::config::Secret;
use asphodel_core::erase::{ForgetError, ForgetRequest, Forgotten};
use asphodel_core::ingest::{Document, IngestError, Ingested, Outcome, Turn};
use asphodel_core::keep::{KeepError, Kept, MemoryIds, Unkept};
use asphodel_core::mental_models::{Model, ModelEdit, ModelError, ModelSpec, Outcome as Refreshed};
use asphodel_core::operations::{
    Audit, AuditError, AuditList, BACKED_UP_AT_HEADER, Backup, BackupError, LENGTH_HEADER,
    SHA256_HEADER, Status,
};
use asphodel_core::queue::{ChunkList, QueueError, Retried, RetryRequest};
use asphodel_core::retrieval::{Prefetch, PrefetchRequest, Recall, RecallError, RecallRequest};
use asphodel_core::store::StoreError;
use asphodel_core::store::bank::{Bank, BankError, BankIdentity};
use asphodel_core::sweep::{PurgeAck, PurgeError, PurgePlan};
use asphodel_core::system_prompt::Block;
use asphodel_core::{Health, ResolvedConfig, Service};
use axum::body::{Body, Bytes};
use axum::extract::rejection::{JsonRejection, QueryRejection};
use axum::extract::{DefaultBodyLimit, Path, Query, Request, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, patch, post, put};
use axum::{Json, Router};
use http_body_util::channel::Channel;
use serde::{Deserialize, Serialize};
use tracing::warn;

use super::{App, Shared};

/// The largest request body: a document of a few megabytes, with room to
/// spare. Anything larger is refused before it's read.
const BODY_LIMIT: usize = 16 * 1024 * 1024;

pub(crate) fn router(app: Shared) -> Router {
    let authorized = Router::new()
        .route("/v1/config", get(config))
        .route("/v1/banks/{bank}", put(put_bank))
        .route("/v1/banks/{bank}/turns", post(turns))
        .route("/v1/banks/{bank}/documents", post(documents))
        .route("/v1/banks/{bank}/prefetch", post(prefetch))
        .route("/v1/banks/{bank}/recall", post(recall))
        .route("/v1/banks/{bank}/forget", post(forget))
        .route("/v1/banks/{bank}/keep", post(keep))
        .route("/v1/banks/{bank}/unkeep", post(unkeep))
        .route(
            "/v1/banks/{bank}/sessions/{session}/clear",
            post(clear_session),
        )
        .route("/v1/banks/{bank}/chunks", get(chunks))
        .route("/v1/banks/{bank}/chunks/retry", post(retry_chunks))
        .route("/v1/banks/{bank}/system-prompt", get(system_prompt))
        .route("/v1/banks/{bank}/agenda", get(agenda))
        .route(
            "/v1/banks/{bank}/models",
            get(list_models).post(create_model),
        )
        .route("/v1/banks/{bank}/models/{model}", patch(edit_model))
        .route(
            "/v1/banks/{bank}/models/{model}/refresh",
            post(refresh_model),
        )
        .route("/v1/banks/{bank}/purges", get(purges))
        .route("/v1/banks/{bank}/forgets", get(forgets))
        .route("/v1/banks/{bank}/sweeps", get(sweeps))
        .route("/v1/banks/{bank}/recalls", get(recalls))
        .route("/v1/backup", post(backup))
        .route("/v1/status", get(status))
        .route("/v1/purge/plan", get(purge_plan))
        .route("/v1/purge/ack", post(purge_ack))
        .route_layer(middleware::from_fn_with_state(Arc::clone(&app), authorize));
    Router::new()
        .route("/v1/health", get(health))
        .merge(authorized)
        .fallback(not_found)
        .layer(DefaultBodyLimit::max(BODY_LIMIT))
        .with_state(app)
}

/// An error as the API returns it: a status and `{"error": "..."}`. No
/// message carries content (ADR 0010): the service's errors name kinds and
/// ids only.
#[derive(Debug)]
pub(crate) struct ApiError {
    status: StatusCode,
    message: String,
}

#[derive(Serialize)]
struct ErrorBody<'a> {
    error: &'a str,
}

impl ApiError {
    fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }

    fn not_ready() -> Self {
        Self::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "the daemon is starting: the store and models aren't ready yet",
        )
    }

    fn draining() -> Self {
        Self::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "the daemon is shutting down and accepts no more ingest",
        )
    }

    fn internal(error: impl std::fmt::Display) -> Self {
        warn!(%error, "request failed");
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, error.to_string())
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let mut response = (
            self.status,
            Json(ErrorBody {
                error: &self.message,
            }),
        )
            .into_response();
        if self.status == StatusCode::UNAUTHORIZED {
            response
                .headers_mut()
                .insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
        }
        response
    }
}

impl From<JsonRejection> for ApiError {
    fn from(rejection: JsonRejection) -> Self {
        Self::new(rejection.status(), rejection.body_text())
    }
}

impl From<QueryRejection> for ApiError {
    fn from(rejection: QueryRejection) -> Self {
        Self::new(rejection.status(), rejection.body_text())
    }
}

impl From<StoreError> for ApiError {
    fn from(error: StoreError) -> Self {
        Self::internal(error)
    }
}

impl From<IngestError> for ApiError {
    fn from(error: IngestError) -> Self {
        match error {
            IngestError::UnknownBank => Self::new(StatusCode::NOT_FOUND, error.to_string()),
            IngestError::InvalidTimezone => Self::new(StatusCode::BAD_REQUEST, error.to_string()),
            IngestError::Store(error) => error.into(),
        }
    }
}

impl From<BankError> for ApiError {
    fn from(error: BankError) -> Self {
        match error {
            BankError::EmptyName | BankError::InvalidTimezone => {
                Self::new(StatusCode::BAD_REQUEST, error.to_string())
            }
            BankError::NoModels => Self::new(StatusCode::SERVICE_UNAVAILABLE, error.to_string()),
            BankError::Store(error) => error.into(),
        }
    }
}

impl From<RecallError> for ApiError {
    fn from(error: RecallError) -> Self {
        match error {
            RecallError::UnknownBank => Self::new(StatusCode::NOT_FOUND, error.to_string()),
            RecallError::InvertedRange => Self::new(StatusCode::BAD_REQUEST, error.to_string()),
            RecallError::NoModels => Self::new(StatusCode::SERVICE_UNAVAILABLE, error.to_string()),
            RecallError::Model { .. } => Self::internal(error),
            RecallError::Store(error) => error.into(),
        }
    }
}

impl From<KeepError> for ApiError {
    fn from(error: KeepError) -> Self {
        match error {
            KeepError::UnknownBank => Self::new(StatusCode::NOT_FOUND, error.to_string()),
            KeepError::TooMany { .. } => Self::new(StatusCode::BAD_REQUEST, error.to_string()),
            KeepError::Store(error) => error.into(),
        }
    }
}

impl From<ForgetError> for ApiError {
    fn from(error: ForgetError) -> Self {
        match error {
            ForgetError::UnknownBank => Self::new(StatusCode::NOT_FOUND, error.to_string()),
            ForgetError::TooMany { .. } => Self::new(StatusCode::BAD_REQUEST, error.to_string()),
            ForgetError::Store(error) => error.into(),
        }
    }
}

impl From<PurgeError> for ApiError {
    fn from(error: PurgeError) -> Self {
        match error {
            PurgeError::HashMismatch => Self::new(StatusCode::CONFLICT, error.to_string()),
            PurgeError::Store(error) => error.into(),
        }
    }
}

impl From<BackupError> for ApiError {
    fn from(error: BackupError) -> Self {
        match error {
            BackupError::Corrupt { .. } => Self::internal(error),
            BackupError::Store(error) => error.into(),
        }
    }
}

impl From<AuditError> for ApiError {
    fn from(error: AuditError) -> Self {
        match error {
            AuditError::UnknownBank => Self::new(StatusCode::NOT_FOUND, error.to_string()),
            AuditError::Store(error) => error.into(),
        }
    }
}

impl From<QueueError> for ApiError {
    fn from(error: QueueError) -> Self {
        match error {
            QueueError::UnknownBank => Self::new(StatusCode::NOT_FOUND, error.to_string()),
            QueueError::NotHeld { .. } => Self::new(StatusCode::CONFLICT, error.to_string()),
            QueueError::Store(error) => error.into(),
        }
    }
}

impl From<ModelError> for ApiError {
    fn from(error: ModelError) -> Self {
        match error {
            ModelError::UnknownBank | ModelError::UnknownModel => {
                Self::new(StatusCode::NOT_FOUND, error.to_string())
            }
            ModelError::DuplicateName => Self::new(StatusCode::CONFLICT, error.to_string()),
            ModelError::OverBudget { .. }
            | ModelError::UnknownEntity
            | ModelError::Invalid { .. } => {
                Self::new(StatusCode::UNPROCESSABLE_ENTITY, error.to_string())
            }
            ModelError::Retrieval(error) => error.into(),
            ModelError::Store(error) => error.into(),
        }
    }
}

impl App {
    /// Runs `call` on the service on a blocking thread: every service call
    /// touches SQLite, and recall runs the models.
    async fn call<T, E>(
        &self,
        call: impl FnOnce(&Service) -> Result<T, E> + Send + 'static,
    ) -> Result<T, ApiError>
    where
        T: Send + 'static,
        E: Into<ApiError> + Send + 'static,
    {
        let service = Arc::clone(&self.ready().ok_or_else(ApiError::not_ready)?.service);
        tokio::task::spawn_blocking(move || call(&service).map_err(Into::into))
            .await
            .map_err(ApiError::internal)?
    }

    /// Refuses ingest once SIGTERM has arrived (TIM-94, decision 3).
    fn accepting_ingest(&self) -> Result<(), ApiError> {
        if self.draining() {
            Err(ApiError::draining())
        } else {
            Ok(())
        }
    }

    /// Wakes `bank`'s extraction worker after something was queued.
    fn wake(&self, bank: &str) {
        if let Some(workers) = self.ready().and_then(|ready| ready.workers.as_ref()) {
            workers.wake(bank);
        }
    }
}

/// Checks the bearer token, when the daemon has one. Off loopback it always
/// has one (TIM-94, decision 2); on loopback it's optional, and checked
/// when set.
async fn authorize(
    State(app): State<Shared>,
    request: Request,
    next: Next,
) -> Result<Response, ApiError> {
    if let Some(token) = &app.token {
        let given = request
            .headers()
            .get(header::AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.strip_prefix("Bearer "));
        if !given.is_some_and(|given| token_matches(token, given)) {
            return Err(ApiError::new(
                StatusCode::UNAUTHORIZED,
                "a bearer token is required: set ASPHODEL_TOKEN on the client",
            ));
        }
    }
    Ok(next.run(request).await)
}

/// Compares the token in time independent of where the first difference
/// is, so the comparison doesn't leak a prefix.
fn token_matches(token: &Secret, given: &str) -> bool {
    let expected = token.expose().as_bytes();
    let given = given.trim().as_bytes();
    let mut difference = expected.len() ^ given.len();
    for (index, byte) in expected.iter().enumerate() {
        difference |= usize::from(byte ^ given.get(index).copied().unwrap_or(0));
    }
    difference == 0
}

async fn not_found() -> ApiError {
    ApiError::new(StatusCode::NOT_FOUND, "no such route")
}

/// `GET /v1/health`: 503 while the store migrates and the models load, and
/// again once the daemon is draining; 200 with the version once ready.
async fn health(State(app): State<Shared>) -> (StatusCode, Json<Health>) {
    match app.ready() {
        Some(ready) if !app.draining() => (StatusCode::OK, Json(ready.service.health())),
        _ => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(Health::starting(app.clock.now())),
        ),
    }
}

/// `GET /v1/config`: the resolved config, secrets redacted (ADR 0009), with
/// the purge state as it is now, after any ack.
async fn config(State(app): State<Shared>) -> Result<Json<ResolvedConfig>, ApiError> {
    let ready = app.ready().ok_or_else(ApiError::not_ready)?;
    let mut config = ready.config.clone();
    config.purge = ready.service.purge_pause();
    Ok(Json(config))
}

/// `GET /v1/purge/plan`: which fingerprinted values changed and what the
/// sweep would delete now (ADR 0010). It deletes nothing.
async fn purge_plan(State(app): State<Shared>) -> Result<Json<PurgePlan>, ApiError> {
    let plan = app.call(|service| service.purge_plan()).await?;
    Ok(Json(plan))
}

/// `POST /v1/purge/ack`: acknowledges the running daemon's deletion
/// fingerprint, which the body must quote; 409 for any other hash. 204.
async fn purge_ack(
    State(app): State<Shared>,
    body: Result<Json<PurgeAck>, JsonRejection>,
) -> Result<StatusCode, ApiError> {
    let Json(PurgeAck { hash }) = body?;
    app.call(move |service| service.purge_ack(&hash)).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// How much of a backup is read and sent at a time.
const BACKUP_CHUNK: usize = 64 * 1024;

/// `POST /v1/backup`: an online backup of the store, integrity-checked,
/// streamed with its SHA-256 and length in headers (ADR 0010). The copy is
/// unlinked from the data dir before the first byte goes, so a stream that
/// stops part way leaves nothing behind. The completion is recorded for
/// `status` just before the last bytes are sent.
async fn backup(State(app): State<Shared>) -> Result<Response, ApiError> {
    let service = Arc::clone(&app.ready().ok_or_else(ApiError::not_ready)?.service);
    let Backup {
        mut file,
        sha256,
        length,
        backed_up_at,
    } = app.call(|service| service.backup()).await?;
    let (mut sender, body) = Channel::<Bytes, std::io::Error>::new(4);
    let runtime = tokio::runtime::Handle::current();
    tokio::task::spawn_blocking(move || {
        let mut buffer = vec![0; BACKUP_CHUNK];
        let mut sent = 0;
        loop {
            let read = match file.read(&mut buffer) {
                Ok(0) => return,
                Ok(read) => read,
                Err(error) => {
                    warn!(%error, "reading the backup copy failed; the stream stops short");
                    sender.abort(error);
                    return;
                }
            };
            sent += read as u64;
            if sent == length
                && let Err(error) = service.record_backup()
            {
                warn!(%error, "recording the backup's completion failed");
            }
            let chunk = Bytes::copy_from_slice(&buffer[..read]);
            if runtime.block_on(sender.send_data(chunk)).is_err() {
                // The client went away.
                return;
            }
        }
    });
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/vnd.sqlite3")
        .header(header::CONTENT_LENGTH, length)
        .header(SHA256_HEADER, sha256)
        .header(LENGTH_HEADER, length)
        .header(BACKED_UP_AT_HEADER, backed_up_at.to_string())
        .body(Body::new(body))
        .map_err(ApiError::internal)
}

/// `GET /v1/status`: queue depth, failures, the purge pause with both
/// hashes, the last sweep, the pre-migration copy and the last backup, and
/// what needs attention (ADR 0010). Always 200: `asphodel status` decides
/// its exit code from `attention`.
async fn status(State(app): State<Shared>) -> Result<Json<Status>, ApiError> {
    let status = app.call(|service| service.status()).await?;
    Ok(Json(status))
}

#[derive(Debug, Default, Deserialize)]
struct ListQuery {
    /// How many rows, newest first.
    #[serde(default)]
    limit: Option<usize>,
}

/// One audit list of a bank (TIM-99, decision 7).
async fn audit(
    app: Shared,
    bank: String,
    query: Result<Query<ListQuery>, QueryRejection>,
    list: AuditList,
) -> Result<Json<Audit>, ApiError> {
    let Query(query) = query?;
    let audit = app
        .call(move |service| service.audit(&bank, list, query.limit))
        .await?;
    Ok(Json(audit))
}

/// `GET /v1/banks/{bank}/purges[?limit=N]`: ids only.
async fn purges(
    State(app): State<Shared>,
    Path(bank): Path<String>,
    query: Result<Query<ListQuery>, QueryRejection>,
) -> Result<Json<Audit>, ApiError> {
    audit(app, bank, query, AuditList::Purges).await
}

/// `GET /v1/banks/{bank}/forgets[?limit=N]`: ids and turn keys only.
async fn forgets(
    State(app): State<Shared>,
    Path(bank): Path<String>,
    query: Result<Query<ListQuery>, QueryRejection>,
) -> Result<Json<Audit>, ApiError> {
    audit(app, bank, query, AuditList::Forgets).await
}

/// `GET /v1/banks/{bank}/sweeps[?limit=N]`: counts only.
async fn sweeps(
    State(app): State<Shared>,
    Path(bank): Path<String>,
    query: Result<Query<ListQuery>, QueryRejection>,
) -> Result<Json<Audit>, ApiError> {
    audit(app, bank, query, AuditList::Sweeps).await
}

/// `GET /v1/banks/{bank}/recalls[?limit=N]`: the one list with content, its
/// queries, until the sweep clears them.
async fn recalls(
    State(app): State<Shared>,
    Path(bank): Path<String>,
    query: Result<Query<ListQuery>, QueryRejection>,
) -> Result<Json<Audit>, ApiError> {
    audit(app, bank, query, AuditList::Recalls).await
}

/// `PUT /v1/banks/{bank}`: creates the bank or merges the identity into it
/// (TIM-94, decision 7). 201 when it created the bank.
async fn put_bank(
    State(app): State<Shared>,
    Path(bank): Path<String>,
    body: Result<Json<BankIdentity>, JsonRejection>,
) -> Result<(StatusCode, Json<Bank>), ApiError> {
    let Json(identity) = body?;
    let bank = app
        .call(move |service| service.ensure_bank_with_models(&bank, &identity))
        .await?;
    let status = if bank.created {
        StatusCode::CREATED
    } else {
        StatusCode::OK
    };
    Ok((status, Json(bank)))
}

/// `POST /v1/banks/{bank}/turns`: `sync_turn`.
async fn turns(
    State(app): State<Shared>,
    Path(bank): Path<String>,
    body: Result<Json<Turn>, JsonRejection>,
) -> Result<Json<Ingested>, ApiError> {
    app.accepting_ingest()?;
    let Json(turn) = body?;
    let name = bank.clone();
    let ingested = app
        .call(move |service| service.ingest_turn(&name, &turn))
        .await?;
    if ingested.outcome == Outcome::Stored && ingested.chunks_queued > 0 {
        app.wake(&bank);
    }
    Ok(Json(ingested))
}

/// `POST /v1/banks/{bank}/documents`: `asphodel ingest`.
async fn documents(
    State(app): State<Shared>,
    Path(bank): Path<String>,
    body: Result<Json<Document>, JsonRejection>,
) -> Result<Json<Ingested>, ApiError> {
    app.accepting_ingest()?;
    let Json(document) = body?;
    let name = bank.clone();
    let ingested = app
        .call(move |service| service.ingest_document(&name, &document))
        .await?;
    if ingested.chunks_queued > 0 {
        app.wake(&bank);
    }
    Ok(Json(ingested))
}

/// `POST /v1/banks/{bank}/prefetch`: the injection for a user message.
async fn prefetch(
    State(app): State<Shared>,
    Path(bank): Path<String>,
    body: Result<Json<PrefetchRequest>, JsonRejection>,
) -> Result<Json<Prefetch>, ApiError> {
    let Json(request) = body?;
    let prefetch = app
        .call(move |service| service.prefetch(&bank, &request))
        .await?;
    Ok(Json(prefetch))
}

/// `POST /v1/banks/{bank}/recall`: `memory_recall` and `asphodel recall`.
async fn recall(
    State(app): State<Shared>,
    Path(bank): Path<String>,
    body: Result<Json<RecallRequest>, JsonRejection>,
) -> Result<Json<Recall>, ApiError> {
    let Json(request) = body?;
    let recall = app
        .call(move |service| service.recall(&bank, &request))
        .await?;
    Ok(Json(recall))
}

/// `POST /v1/banks/{bank}/keep`: `memory_keep`.
async fn keep(
    State(app): State<Shared>,
    Path(bank): Path<String>,
    body: Result<Json<MemoryIds>, JsonRejection>,
) -> Result<Json<Kept>, ApiError> {
    let Json(MemoryIds { ids }) = body?;
    let kept = app.call(move |service| service.keep(&bank, &ids)).await?;
    Ok(Json(kept))
}

/// `POST /v1/banks/{bank}/forget`: `memory_forget`, owner-only in the
/// plugin, with the session its request turn will arrive in. Returns every
/// id the erase removes. The erase runs at once when nothing was queued
/// before it, and otherwise the bank's worker or housekeeping runs it
/// behind those chunks (ADR 0010).
async fn forget(
    State(app): State<Shared>,
    Path(bank): Path<String>,
    body: Result<Json<ForgetRequest>, JsonRejection>,
) -> Result<Json<Forgotten>, ApiError> {
    let Json(request) = body?;
    let name = bank.clone();
    let forgotten = app
        .call(move |service| {
            let forgotten = service.forget_request(&name, &request)?;
            if !forgotten.forgotten.is_empty() {
                service.erase_next(&name)?;
            }
            Ok::<_, ApiError>(forgotten)
        })
        .await?;
    app.wake(&bank);
    Ok(Json(forgotten))
}

/// `POST /v1/banks/{bank}/unkeep`: `memory_unkeep`.
async fn unkeep(
    State(app): State<Shared>,
    Path(bank): Path<String>,
    body: Result<Json<MemoryIds>, JsonRejection>,
) -> Result<Json<Unkept>, ApiError> {
    let Json(MemoryIds { ids }) = body?;
    let unkept = app.call(move |service| service.unkeep(&bank, &ids)).await?;
    Ok(Json(unkept))
}

/// `POST /v1/banks/{bank}/sessions/{id}/clear`: Hermes'
/// `on_session_switch` on compression, reset or rewind. 204.
async fn clear_session(
    State(app): State<Shared>,
    Path((bank, session)): Path<(String, String)>,
) -> Result<StatusCode, ApiError> {
    app.call(move |service| service.clear_session(&bank, &session))
        .await?;
    Ok(StatusCode::NO_CONTENT)
}

#[derive(Debug, Default, Deserialize)]
struct ChunksQuery {
    /// Only the failed chunks.
    #[serde(default)]
    failed: bool,
}

/// `GET /v1/banks/{bank}/chunks[?failed=true]`: the bank's queue and its
/// failed chunks, or only the failed ones.
async fn chunks(
    State(app): State<Shared>,
    Path(bank): Path<String>,
    query: Result<Query<ChunksQuery>, QueryRejection>,
) -> Result<Json<ChunkList>, ApiError> {
    let Query(query) = query?;
    let chunks = app
        .call(move |service| service.chunks(&bank, query.failed))
        .await?;
    Ok(Json(chunks))
}

/// `POST /v1/banks/{bank}/chunks/retry`: puts failed chunks back on the
/// queue, the ones named or all of them.
async fn retry_chunks(
    State(app): State<Shared>,
    Path(bank): Path<String>,
    body: Result<Json<RetryRequest>, JsonRejection>,
) -> Result<Json<Retried>, ApiError> {
    let Json(request) = body?;
    let name = bank.clone();
    let retried = app
        .call(move |service| service.retry_chunks(&name, request.chunks.as_deref()))
        .await?;
    if !retried.retried.is_empty() {
        app.wake(&bank);
    }
    Ok(Json(retried))
}

#[derive(Debug, Default, Deserialize)]
struct SystemPromptQuery {
    /// The Hermes session the block is for. The daemon records which block
    /// the session holds, and what it lists or cites joins the session's
    /// in-context set (TIM-95, decision 4).
    #[serde(default)]
    session_id: Option<String>,
}

/// `GET /v1/banks/{bank}/system-prompt[?session_id=...]`:
/// `system_prompt_block()`. Built from the agenda and the models' entries
/// with queries only, never an LLM call.
async fn system_prompt(
    State(app): State<Shared>,
    Path(bank): Path<String>,
    query: Result<Query<SystemPromptQuery>, QueryRejection>,
) -> Result<Json<Block>, ApiError> {
    let Query(query) = query?;
    let block = app
        .call(move |service| service.system_prompt(&bank, query.session_id.as_deref()))
        .await?;
    Ok(Json(block))
}

/// `GET /v1/banks/{bank}/agenda`: the agenda as the block would list it.
async fn agenda(
    State(app): State<Shared>,
    Path(bank): Path<String>,
) -> Result<Json<Agenda>, ApiError> {
    let agenda = app.call(move |service| service.agenda(&bank)).await?;
    Ok(Json(agenda))
}

/// `GET /v1/banks/{bank}/models`: `asphodel model list`.
async fn list_models(
    State(app): State<Shared>,
    Path(bank): Path<String>,
) -> Result<Json<Vec<Model>>, ApiError> {
    let models = app.call(move |service| service.list_models(&bank)).await?;
    Ok(Json(models))
}

/// `POST /v1/banks/{bank}/models`: `asphodel model create`. 201.
async fn create_model(
    State(app): State<Shared>,
    Path(bank): Path<String>,
    body: Result<Json<ModelSpec>, JsonRejection>,
) -> Result<(StatusCode, Json<Model>), ApiError> {
    let Json(spec) = body?;
    let model = app
        .call(move |service| service.create_model(&bank, &spec))
        .await?;
    Ok((StatusCode::CREATED, Json(model)))
}

/// `PATCH /v1/banks/{bank}/models/{model}`: `asphodel model edit`.
async fn edit_model(
    State(app): State<Shared>,
    Path((bank, name)): Path<(String, String)>,
    body: Result<Json<ModelEdit>, JsonRejection>,
) -> Result<Json<Model>, ApiError> {
    let Json(edit) = body?;
    let model = app
        .call(move |service| service.edit_model(&bank, &name, &edit))
        .await?;
    Ok(Json(model))
}

#[derive(Debug, Default, Deserialize)]
struct RefreshQuery {
    /// Skip the fingerprint check.
    #[serde(default)]
    force: bool,
}

/// `POST /v1/banks/{bank}/models/{model}/refresh[?force=true]`:
/// `asphodel model refresh`. 503 when no LLM is configured.
async fn refresh_model(
    State(app): State<Shared>,
    Path((bank, name)): Path<(String, String)>,
    query: Result<Query<RefreshQuery>, QueryRejection>,
) -> Result<Json<Refreshed>, ApiError> {
    let Query(query) = query?;
    let llm = app
        .ready()
        .ok_or_else(ApiError::not_ready)?
        .llm
        .clone()
        .ok_or_else(|| {
            ApiError::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "no LLM is configured, so models can't be refreshed",
            )
        })?;
    let outcome = app
        .call(move |service| service.refresh_model(&bank, &name, llm.as_ref(), query.force))
        .await?;
    Ok(Json(outcome))
}
