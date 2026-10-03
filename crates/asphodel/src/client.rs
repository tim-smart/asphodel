//! The CLI's HTTP client of the daemon: every subcommand but `serve`,
//! `models fetch`, `llm login` and `replay` goes through it, so nothing but
//! the daemon opens the database.
//!
//! `--url` takes `http://host:port` or `unix:/path`, the two forms `--listen`
//! takes. There's no TLS: the daemon serves plain HTTP, on loopback or behind
//! `kubectl port-forward`.

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, anyhow, bail};
use axum::http::{HeaderMap, Method, Request, Response, Uri, header};
use http_body_util::{BodyExt, Full};
use hyper::body::{Bytes, Incoming};
use hyper_util::rt::TokioIo;
use serde::Serialize;
use serde::de::DeserializeOwned;
use tokio::net::{TcpStream, UnixStream};

use crate::cli::ClientArgs;

/// The longest one request may take. Recall answers within the reranker
/// deadline and ingest never calls a model, so this only bounds a daemon
/// that's stuck.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

/// Where the daemon is.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Target {
    /// `host:port`, and the path prefix the URL carried, if any.
    Tcp {
        authority: String,
        prefix: String,
    },
    Unix(PathBuf),
}

pub(crate) struct Client {
    url: String,
    target: Target,
    token: Option<String>,
    runtime: tokio::runtime::Runtime,
}

impl Client {
    pub(crate) fn new(args: &ClientArgs) -> anyhow::Result<Self> {
        let target = parse_url(&args.url)?;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        Ok(Self {
            url: args.url.clone(),
            target,
            token: args.token.clone().filter(|token| !token.is_empty()),
            runtime,
        })
    }

    pub(crate) fn get<T: DeserializeOwned>(&self, path: &str) -> anyhow::Result<T> {
        self.request(Method::GET, path, None)
    }

    pub(crate) fn post<B: Serialize, T: DeserializeOwned>(
        &self,
        path: &str,
        body: &B,
    ) -> anyhow::Result<T> {
        self.request(Method::POST, path, Some(serde_json::to_vec(body)?))
    }

    pub(crate) fn patch<B: Serialize, T: DeserializeOwned>(
        &self,
        path: &str,
        body: &B,
    ) -> anyhow::Result<T> {
        self.request(Method::PATCH, path, Some(serde_json::to_vec(body)?))
    }

    pub(crate) fn delete<T: DeserializeOwned>(&self, path: &str) -> anyhow::Result<T> {
        self.request(Method::DELETE, path, None)
    }

    pub(crate) fn put<B: Serialize, T: DeserializeOwned>(
        &self,
        path: &str,
        body: &B,
    ) -> anyhow::Result<T> {
        self.request(Method::PUT, path, Some(serde_json::to_vec(body)?))
    }

    /// `POST path` with no body, handing the reply's body to `sink` frame
    /// by frame as it arrives, for the backup stream, which can be larger
    /// than is sensible to hold. Returns the reply's headers once the body
    /// has ended. A stream the daemon or the network cuts short is an
    /// error. Each frame gets [`REQUEST_TIMEOUT`], not the whole stream.
    pub(crate) fn download(
        &self,
        path: &str,
        sink: &mut dyn FnMut(&[u8]) -> anyhow::Result<()>,
    ) -> anyhow::Result<HeaderMap> {
        let timed_out = || anyhow!("the daemon at {} didn't answer in time", self.url);
        self.runtime.block_on(async {
            let response =
                tokio::time::timeout(REQUEST_TIMEOUT, self.open(Method::POST, path, None))
                    .await
                    .map_err(|_| timed_out())??;
            let status = response.status();
            let headers = response.headers().clone();
            let mut body = response.into_body();
            if !status.is_success() {
                let bytes = tokio::time::timeout(REQUEST_TIMEOUT, body.collect())
                    .await
                    .map_err(|_| timed_out())??
                    .to_bytes();
                bail!("the daemon answered {status}: {}", error_message(&bytes));
            }
            while let Some(frame) = tokio::time::timeout(REQUEST_TIMEOUT, body.frame())
                .await
                .map_err(|_| timed_out())?
            {
                let frame = frame.context("the stream from the daemon was cut short")?;
                if let Some(data) = frame.data_ref() {
                    sink(data)?;
                }
            }
            Ok(headers)
        })
    }

    fn request<T: DeserializeOwned>(
        &self,
        method: Method,
        path: &str,
        body: Option<Vec<u8>>,
    ) -> anyhow::Result<T> {
        let (status, bytes) = self
            .runtime
            .block_on(async {
                tokio::time::timeout(REQUEST_TIMEOUT, self.send(method, path, body)).await
            })
            .map_err(|_| anyhow!("the daemon at {} didn't answer in time", self.url))??;
        if !status.is_success() {
            bail!("the daemon answered {status}: {}", error_message(&bytes));
        }
        let bytes = if bytes.is_empty() {
            Bytes::from_static(b"null")
        } else {
            bytes
        };
        serde_json::from_slice(&bytes).context("the daemon's reply isn't the JSON expected")
    }

    async fn send(
        &self,
        method: Method,
        path: &str,
        body: Option<Vec<u8>>,
    ) -> anyhow::Result<(hyper::StatusCode, Bytes)> {
        let response = self.open(method, path, body).await?;
        let status = response.status();
        let bytes = response.into_body().collect().await?.to_bytes();
        Ok((status, bytes))
    }

    /// Sends one request on its own connection and returns the reply with
    /// its body still to read.
    async fn open(
        &self,
        method: Method,
        path: &str,
        body: Option<Vec<u8>>,
    ) -> anyhow::Result<Response<Incoming>> {
        let unreachable = || {
            format!(
                "can't reach the daemon at {}: is `asphodel serve` running?",
                self.url
            )
        };
        let (host, uri) = match &self.target {
            Target::Tcp { authority, prefix } => (authority.clone(), format!("{prefix}{path}")),
            Target::Unix(_) => ("localhost".to_string(), path.to_string()),
        };
        let mut request = Request::builder()
            .method(method)
            .uri(uri)
            .header(header::HOST, host)
            .header(header::ACCEPT, "application/json");
        if let Some(token) = &self.token {
            request = request.header(header::AUTHORIZATION, format!("Bearer {token}"));
        }
        if body.is_some() {
            request = request.header(header::CONTENT_TYPE, "application/json");
        }
        let request = request.body(Full::new(Bytes::from(body.unwrap_or_default())))?;

        let response = match &self.target {
            Target::Tcp { authority, .. } => {
                let stream = TcpStream::connect(authority)
                    .await
                    .with_context(unreachable)?;
                exchange(stream, request).await
            }
            Target::Unix(path) => {
                let stream = UnixStream::connect(path).await.with_context(unreachable)?;
                exchange(stream, request).await
            }
        }
        .with_context(unreachable)?;
        Ok(response)
    }
}

/// One request on its own connection. The connection runs until the
/// reply's body has been read or dropped.
async fn exchange<S>(stream: S, request: Request<Full<Bytes>>) -> anyhow::Result<Response<Incoming>>
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin + Send + 'static,
{
    let (mut sender, connection) =
        hyper::client::conn::http1::handshake(TokioIo::new(stream)).await?;
    tokio::spawn(connection);
    Ok(sender.send_request(request).await?)
}

/// The `error` of a JSON error body, or the body itself.
fn error_message(bytes: &[u8]) -> String {
    serde_json::from_slice::<serde_json::Value>(bytes)
        .ok()
        .and_then(|body| body.get("error")?.as_str().map(str::to_string))
        .unwrap_or_else(|| String::from_utf8_lossy(bytes).trim().to_string())
}

/// Parses `--url`: `unix:/path`, or `http://host[:port][/prefix]`.
fn parse_url(url: &str) -> anyhow::Result<Target> {
    if let Some(path) = url.strip_prefix("unix:") {
        if path.is_empty() {
            bail!("--url {url}: a unix socket needs a path after `unix:`");
        }
        return Ok(Target::Unix(PathBuf::from(path)));
    }
    let uri: Uri = url
        .parse()
        .with_context(|| format!("--url {url}: expected http://host:port or unix:/path"))?;
    match uri.scheme_str() {
        Some("http") => {}
        Some(scheme) => bail!("--url {url}: the daemon serves plain http, not {scheme}"),
        None => bail!("--url {url}: expected http://host:port or unix:/path"),
    }
    let authority = uri
        .authority()
        .ok_or_else(|| anyhow!("--url {url}: no host"))?;
    let authority = match authority.port_u16() {
        Some(_) => authority.to_string(),
        None => format!("{}:80", authority.host()),
    };
    let prefix = uri.path().trim_end_matches('/').to_string();
    Ok(Target::Tcp { authority, prefix })
}

/// Percent-encodes one path segment, such as a bank name, so a space or a
/// slash in it can't change the route.
pub(crate) fn segment(value: &str) -> String {
    let mut encoded = String::with_capacity(value.len());
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~') {
            encoded.push(char::from(byte));
        } else {
            encoded.push_str(&format!("%{byte:02X}"));
        }
    }
    encoded
}
