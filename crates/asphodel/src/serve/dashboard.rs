//! The dashboard's page, served at `/dashboard` without the bearer token:
//! it's static and holds nothing. The page asks for the token and sends it
//! on each `/v1` call, which need it like any other client. Its files are
//! embedded in the binary as they are in `assets/dashboard/`, plain ES
//! modules, CSS and the serif's woff2 files with no build step, so the image
//! needs nothing besides it.

use axum::extract::Path;
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};

const INDEX: &str = include_str!("../../assets/dashboard/index.html");

const JAVASCRIPT: &str = "text/javascript; charset=utf-8";
const CSS: &str = "text/css; charset=utf-8";
const WOFF2: &str = "font/woff2";
const TEXT: &str = "text/plain; charset=utf-8";

/// Scripts and styles change with the binary, so they're revalidated.
const REVALIDATE: &str = "no-cache";
/// The fonts are Adobe's release files and don't change between builds.
const LASTING: &str = "public, max-age=604800";

/// Every file under `/dashboard/`, with its content type and caching.
const ASSETS: &[(&str, &str, &str, &[u8])] = &[
    (
        "main.js",
        JAVASCRIPT,
        REVALIDATE,
        include_bytes!("../../assets/dashboard/main.js"),
    ),
    (
        "app.js",
        JAVASCRIPT,
        REVALIDATE,
        include_bytes!("../../assets/dashboard/app.js"),
    ),
    (
        "api.js",
        JAVASCRIPT,
        REVALIDATE,
        include_bytes!("../../assets/dashboard/api.js"),
    ),
    (
        "dom.js",
        JAVASCRIPT,
        REVALIDATE,
        include_bytes!("../../assets/dashboard/dom.js"),
    ),
    (
        "pages.js",
        JAVASCRIPT,
        REVALIDATE,
        include_bytes!("../../assets/dashboard/pages.js"),
    ),
    (
        "style.css",
        CSS,
        REVALIDATE,
        include_bytes!("../../assets/dashboard/style.css"),
    ),
    // Source Serif 4, unmodified from Adobe's 4.005 release, under the SIL
    // Open Font License in OFL.txt beside it.
    (
        "SourceSerif4-Regular.ttf.woff2",
        WOFF2,
        LASTING,
        include_bytes!("../../assets/dashboard/SourceSerif4-Regular.ttf.woff2"),
    ),
    (
        "SourceSerif4-It.ttf.woff2",
        WOFF2,
        LASTING,
        include_bytes!("../../assets/dashboard/SourceSerif4-It.ttf.woff2"),
    ),
    (
        "SourceSerif4-Semibold.ttf.woff2",
        WOFF2,
        LASTING,
        include_bytes!("../../assets/dashboard/SourceSerif4-Semibold.ttf.woff2"),
    ),
    (
        "OFL.txt",
        TEXT,
        LASTING,
        include_bytes!("../../assets/dashboard/OFL.txt"),
    ),
];

/// Everything the page loads comes from the daemon. The scripts build the
/// page with DOM calls, never markup, so no inline script or style is
/// needed; the token in session storage stays out of reach of anything
/// injected.
const CSP: &str = "default-src 'none'; script-src 'self'; style-src 'self'; \
    font-src 'self'; img-src 'self' data:; connect-src 'self'; base-uri 'none'; \
    form-action 'none'; frame-ancestors 'none'";

fn headers(response: &mut Response, content_type: &'static str, cache: &'static str) {
    let headers = response.headers_mut();
    headers.insert(header::CONTENT_TYPE, HeaderValue::from_static(content_type));
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static(cache));
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
    headers(&mut response, "text/html; charset=utf-8", REVALIDATE);
    response
}

/// `GET /dashboard/{file}`: one of [`ASSETS`], 404 for anything else.
pub(crate) async fn asset(Path(file): Path<String>) -> Response {
    let Some((_, content_type, cache, body)) = ASSETS.iter().find(|(name, _, _, _)| *name == file)
    else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let mut response = (*body).into_response();
    headers(&mut response, content_type, cache);
    response
}
