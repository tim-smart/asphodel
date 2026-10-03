//! The dashboard's page, served at `/dashboard` without the bearer token:
//! it's static and holds nothing. The page asks for the token and sends it
//! on each `/v1` call, which need it like any other client. Its files are
//! embedded in the binary, so the image needs nothing besides it.

use axum::http::header;
use axum::response::IntoResponse;

const INDEX: &str = include_str!("../../assets/dashboard/index.html");

/// `GET /dashboard`.
pub(crate) async fn page() -> impl IntoResponse {
    (
        [
            (header::CONTENT_TYPE, "text/html; charset=utf-8"),
            (header::CACHE_CONTROL, "no-cache"),
            (header::X_CONTENT_TYPE_OPTIONS, "nosniff"),
            (header::REFERRER_POLICY, "no-referrer"),
        ],
        INDEX,
    )
}
