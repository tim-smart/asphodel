//! The dashboard's page, served at `/dashboard` without the bearer token:
//! it's static and holds nothing. The page asks for the token and sends it
//! on each `/v1` call, which need it like any other client. Its files are
//! embedded in the binary as they are in `assets/dashboard/`, plain ES
//! modules and CSS with no build step, so the image needs nothing besides
//! it.

use axum::extract::Path;
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};

const INDEX: &str = include_str!("../../assets/dashboard/index.html");

const JAVASCRIPT: &str = "text/javascript; charset=utf-8";
const CSS: &str = "text/css; charset=utf-8";

/// Every file under `/dashboard/`, with its content type.
const ASSETS: &[(&str, &str, &str)] = &[
    (
        "main.js",
        JAVASCRIPT,
        include_str!("../../assets/dashboard/main.js"),
    ),
    (
        "app.js",
        JAVASCRIPT,
        include_str!("../../assets/dashboard/app.js"),
    ),
    (
        "api.js",
        JAVASCRIPT,
        include_str!("../../assets/dashboard/api.js"),
    ),
    (
        "dom.js",
        JAVASCRIPT,
        include_str!("../../assets/dashboard/dom.js"),
    ),
    (
        "pages.js",
        JAVASCRIPT,
        include_str!("../../assets/dashboard/pages.js"),
    ),
    (
        "style.css",
        CSS,
        include_str!("../../assets/dashboard/style.css"),
    ),
];

/// Everything the page loads comes from the daemon. The scripts build the
/// page with DOM calls, never markup, so no inline script or style is
/// needed; the token in session storage stays out of reach of anything
/// injected.
const CSP: &str = "default-src 'none'; script-src 'self'; style-src 'self'; \
    img-src 'self' data:; connect-src 'self'; base-uri 'none'; form-action 'none'; \
    frame-ancestors 'none'";

fn headers(response: &mut Response, content_type: &'static str) {
    let headers = response.headers_mut();
    headers.insert(header::CONTENT_TYPE, HeaderValue::from_static(content_type));
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    headers.insert(
        header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );
    headers.insert(
        header::REFERRER_POLICY,
        HeaderValue::from_static("no-referrer"),
    );
    headers.insert(
        header::CONTENT_SECURITY_POLICY,
        HeaderValue::from_static(CSP),
    );
}

/// `GET /dashboard`.
pub(crate) async fn page() -> Response {
    let mut response = INDEX.into_response();
    headers(&mut response, "text/html; charset=utf-8");
    response
}

/// `GET /dashboard/{file}`: one of [`ASSETS`], 404 for anything else.
pub(crate) async fn asset(Path(file): Path<String>) -> Response {
    let Some((_, content_type, body)) = ASSETS.iter().find(|(name, _, _)| *name == file) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let mut response = (*body).into_response();
    headers(&mut response, content_type);
    response
}
