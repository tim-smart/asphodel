//! The secret scan that runs before anything is stored (ADR 0002).
//!
//! "Extraction: significance, validity windows and supersession" (TIM-92,
//! other decision 1): when a pattern matches, the matched span is redacted in
//! place with a marker that names the pattern kind, the redacted text is
//! stored and extracted, and the kinds that fired are recorded on the source.
//!
//! The patterns are specific on purpose. Over-redaction loses memories, so
//! there's no generic high-entropy rule: a commit hash or a UUID is left
//! alone. Patterns run in a fixed order and each runs on the previous one's
//! output, so the more specific kind wins where two overlap (`sk-ant-` before
//! `sk-`). A marker never matches any pattern, so rescanning stored text
//! finds nothing.

use std::collections::BTreeSet;
use std::sync::LazyLock;

use regex::{Captures, Regex};
use serde::Serialize;

/// The kinds of secret the scan recognises. The name of each kind is what's
/// recorded on the source and shown by `memory show` (ADR 0010), so renaming
/// one is a migration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SecretKind {
    /// A PEM or OpenSSH private key block, BEGIN to END.
    PrivateKey,
    /// `AKIA` or `ASIA` and 16 upper-case alphanumerics.
    AwsAccessKey,
    /// `ghp_`, `gho_`, `ghu_`, `ghs_`, `ghr_` or `github_pat_` tokens.
    GithubToken,
    /// `sk-` keys, including `sk-proj-`, but not `sk-ant-`.
    OpenAiKey,
    /// `sk-ant-` keys.
    AnthropicKey,
    /// `xoxa-`, `xoxb-`, `xoxp-`, `xoxr-` and `xoxs-` tokens.
    SlackToken,
    /// `sk_live_`, `sk_test_`, `rk_live_` and `rk_test_` keys.
    StripeKey,
    /// `AIza` and 35 URL-safe characters.
    GoogleApiKey,
    /// Three dot-separated base64url segments, the first two starting `eyJ`.
    Jwt,
    /// The password in a URL's userinfo (`scheme://user:password@host`).
    /// Only the password is redacted; the user and host stay.
    UrlPassword,
}

impl SecretKind {
    pub const ALL: [SecretKind; 10] = [
        SecretKind::PrivateKey,
        SecretKind::AwsAccessKey,
        SecretKind::GithubToken,
        SecretKind::OpenAiKey,
        SecretKind::AnthropicKey,
        SecretKind::SlackToken,
        SecretKind::StripeKey,
        SecretKind::GoogleApiKey,
        SecretKind::Jwt,
        SecretKind::UrlPassword,
    ];

    /// The snake_case name.
    pub fn as_str(self) -> &'static str {
        match self {
            SecretKind::PrivateKey => "private_key",
            SecretKind::AwsAccessKey => "aws_access_key",
            SecretKind::GithubToken => "github_token",
            SecretKind::OpenAiKey => "openai_key",
            SecretKind::AnthropicKey => "anthropic_key",
            SecretKind::SlackToken => "slack_token",
            SecretKind::StripeKey => "stripe_key",
            SecretKind::GoogleApiKey => "google_api_key",
            SecretKind::Jwt => "jwt",
            SecretKind::UrlPassword => "url_password",
        }
    }

    /// The text a match is replaced with. It names the kind. The space in it
    /// keeps it from matching the URL password pattern when rescanned.
    pub fn marker(self) -> String {
        format!("[redacted {}]", self.as_str())
    }
}

impl std::fmt::Display for SecretKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// What one scan found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Scan {
    /// The input with every match replaced in place by its kind's marker.
    /// Equal to the input when nothing matched.
    pub text: String,
    /// The kinds that fired, each once however often it matched.
    pub kinds: BTreeSet<SecretKind>,
}

/// One pattern. `group` is the capture group that holds the secret; the rest
/// of the match stays.
struct Pattern {
    kind: SecretKind,
    regex: Regex,
    group: usize,
}

/// The patterns in the order they run. Specific kinds go before the general
/// ones they overlap with.
static PATTERNS: LazyLock<Vec<Pattern>> = LazyLock::new(|| {
    let pattern = |kind, source: &str, group| Pattern {
        kind,
        regex: Regex::new(source).expect("a secret pattern compiles"),
        group,
    };
    vec![
        pattern(
            SecretKind::PrivateKey,
            r"(?s)-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
            0,
        ),
        pattern(
            SecretKind::UrlPassword,
            r"[A-Za-z][A-Za-z0-9+.\-]*://[^\s/:@]+:([^\s/@]+)@",
            1,
        ),
        pattern(
            SecretKind::Jwt,
            r"\beyJ[A-Za-z0-9_\-]{5,}\.eyJ[A-Za-z0-9_\-]{5,}\.[A-Za-z0-9_\-]{10,}",
            0,
        ),
        pattern(SecretKind::AnthropicKey, r"\bsk-ant-[A-Za-z0-9_\-]{20,}", 0),
        pattern(SecretKind::OpenAiKey, r"\bsk-[A-Za-z0-9_\-]{20,}", 0),
        pattern(
            SecretKind::GithubToken,
            r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,})",
            0,
        ),
        pattern(
            SecretKind::SlackToken,
            r"\bxox[abprs]-[A-Za-z0-9\-]{10,}",
            0,
        ),
        pattern(
            SecretKind::StripeKey,
            r"\b[rs]k_(?:live|test)_[A-Za-z0-9]{16,}",
            0,
        ),
        pattern(
            SecretKind::AwsAccessKey,
            r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b",
            0,
        ),
        pattern(SecretKind::GoogleApiKey, r"\bAIza[0-9A-Za-z_\-]{35}", 0),
    ]
});

/// Scans `text` and redacts every match. Pure: no clock, no store.
pub fn scan(text: &str) -> Scan {
    let mut text = text.to_owned();
    let mut kinds = BTreeSet::new();
    for pattern in PATTERNS.iter() {
        if !pattern.regex.is_match(&text) {
            continue;
        }
        kinds.insert(pattern.kind);
        let marker = pattern.kind.marker();
        text = pattern
            .regex
            .replace_all(&text, |captures: &Captures<'_>| {
                let whole = captures.get(0).expect("group 0 always matches");
                let secret = captures
                    .get(pattern.group)
                    .expect("the secret's group matches whenever the pattern does");
                let mut replaced = String::with_capacity(whole.len());
                replaced.push_str(&whole.as_str()[..secret.start() - whole.start()]);
                replaced.push_str(&marker);
                replaced.push_str(&whole.as_str()[secret.end() - whole.start()..]);
                replaced
            })
            .into_owned();
    }
    Scan { text, kinds }
}
