//! `asphodel report html R`: one static page from a JSON report
//! (`docs/replay.md`, "The HTML page").
//!
//! The page inlines its one style sheet and has no script, image or link,
//! so a page full of Tim's history never fetches anything. It shows the
//! report's identity, its probes, and every number, section by section in
//! the report's own order; a section this module doesn't know is still
//! shown, as a table of its fields. The report and the page are history, so
//! both stay inside the private dir, and no error quotes the report.

use std::fmt::Write as _;
use std::path::PathBuf;

use anyhow::{Context as _, bail};
use serde_json::{Map, Value};

use crate::cli::HtmlArgs;

/// The style sheet, inline.
const STYLE: &str = "\
body { font: 14px/1.4 system-ui, sans-serif; margin: 2em auto; max-width: 70em; padding: 0 1em; color: #222; }
h1 { font-size: 1.5em; }
h2 { font-size: 1.15em; margin-top: 2em; border-bottom: 1px solid #ccc; }
table { border-collapse: collapse; margin: 0.5em 0; }
th, td { border: 1px solid #ddd; padding: 0.2em 0.6em; text-align: left; vertical-align: top; }
td.number { text-align: right; font-variant-numeric: tabular-nums; }
th { background: #f4f4f4; }
code, pre { font-family: ui-monospace, monospace; font-size: 0.95em; }
.passed { color: #176f2c; }
.failed { color: #b3261e; font-weight: bold; }
";

/// The report's sections in the order the page shows them, with their
/// headings. Anything else in the report follows under its own key.
const SECTIONS: &[(&str, &str)] = &[
    ("injected_tokens", "Injected tokens"),
    ("profile_tokens", "Profile tokens"),
    ("extraction_lag", "Extraction lag"),
    ("call2_rate", "Call 2 rate"),
    ("refresh_calls_per_day", "Refresh calls per day"),
    ("refresh_retries", "Refresh writes sent again"),
    ("agenda_lines_per_day", "Agenda lines per day"),
    ("bands_per_week", "Memories per band per week"),
    ("fade_outs_per_week", "Fade-outs per week"),
    ("purges_per_day", "Purges per day"),
    ("purged_then_re_mentioned", "Purged, then mentioned again"),
    ("significance_histogram", "Significance"),
    ("kind_histogram", "Kinds"),
    ("llm", "Where LLM replies came from"),
    ("memories", "Memories"),
    ("flags", "Flags"),
    ("tuning", "Tuning"),
];

/// The identity fields at the top of the page.
const IDENTITY: &[&str] = &[
    "kind",
    "scenario",
    "group",
    "version",
    "git_sha",
    "corpus_hash",
    "cassette_hash",
    "refresh_queries_hash",
];

/// Runs the command: writes the page and exits 0, or exits 2 when it's
/// refused or the report doesn't parse.
pub fn run(args: HtmlArgs) -> anyhow::Result<()> {
    match execute(&args) {
        Ok(page) => {
            tracing::info!(page = %page.display(), "wrote the report page");
            Ok(())
        }
        Err(error) => {
            eprintln!("error: {error:#}");
            std::process::exit(2)
        }
    }
}

fn execute(args: &HtmlArgs) -> anyhow::Result<PathBuf> {
    let dir = super::private_dir(args.replay_dir.as_deref())?;
    let report_path = super::inside_private(&dir, &args.report, "the report")?;
    let page_path = match &args.out {
        Some(path) => super::inside_private(&dir, path, "the page")?,
        None => super::inside_private(&dir, &report_path.with_extension("html"), "the page")?,
    };
    if page_path == report_path {
        bail!(
            "the page {} would overwrite the report",
            page_path.display()
        );
    }
    let text = std::fs::read_to_string(&report_path)
        .with_context(|| format!("reading {}", report_path.display()))?;
    let report: Value = serde_json::from_str(&text).map_err(|error| {
        super::json_error(&report_path, error.line(), "a replay report", &error)
    })?;
    let Value::Object(report) = report else {
        bail!("{} isn't a replay report", report_path.display());
    };
    super::write_file(&page_path, render(&report).as_bytes())
        .with_context(|| format!("writing the page to {}", page_path.display()))?;
    Ok(page_path)
}

/// The page for `report`.
fn render(report: &Map<String, Value>) -> String {
    let mut out = String::new();
    let title = format!(
        "Asphodel replay: {}",
        report
            .get("scenario")
            .and_then(Value::as_str)
            .unwrap_or("report")
    );
    let _ = write!(
        out,
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n<title>{}</title>\n<style>\n{STYLE}</style>\n</head>\n<body>\n<h1>{}</h1>\n",
        escape(&title),
        escape(&title)
    );

    out.push_str("<table>\n");
    for key in IDENTITY {
        let value = report.get(*key).unwrap_or(&Value::Null);
        let _ = writeln!(
            out,
            "<tr><th>{}</th><td><code>{}</code></td></tr>",
            escape(key),
            scalar(value)
        );
    }
    out.push_str("</table>\n");

    probes(&mut out, report.get("probes"));

    let known: Vec<&str> = IDENTITY
        .iter()
        .copied()
        .chain(["probes"])
        .chain(SECTIONS.iter().map(|(key, _)| *key))
        .collect();
    for (key, heading) in SECTIONS {
        if let Some(value) = report.get(*key) {
            section(&mut out, heading, value);
        }
    }
    for (key, value) in report {
        if !known.contains(&key.as_str()) {
            section(&mut out, key, value);
        }
    }
    out.push_str("</body>\n</html>\n");
    out
}

/// The probes: id, time, kind, result, and what the probe saw.
fn probes(out: &mut String, probes: Option<&Value>) {
    let probes = probes.and_then(Value::as_array).map(Vec::as_slice);
    let probes = probes.unwrap_or_default();
    let passed = probes
        .iter()
        .filter(|probe| probe.get("passed") == Some(&Value::Bool(true)))
        .count();
    let _ = writeln!(
        out,
        "<h2>Probes</h2>\n<p>{passed} of {} passed.</p>",
        probes.len()
    );
    if probes.is_empty() {
        return;
    }
    out.push_str(
        "<table>\n<tr><th>id</th><th>at</th><th>kind</th><th>result</th><th>observed</th></tr>\n",
    );
    for probe in probes {
        let field = |key: &str| scalar(probe.get(key).unwrap_or(&Value::Null));
        let result = if probe.get("passed") == Some(&Value::Bool(true)) {
            "<span class=\"passed\">passed</span>"
        } else {
            "<span class=\"failed\">failed</span>"
        };
        let observed = probe
            .get("observed")
            .map(|observed| escape(&observed.to_string()))
            .unwrap_or_default();
        let _ = writeln!(
            out,
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{result}</td><td><code>{observed}</code></td></tr>",
            field("id"),
            field("at"),
            field("kind")
        );
    }
    out.push_str("</table>\n");
}

/// A section: an array of objects as one table with a column per field,
/// an object as a table of its fields (nested ones by dotted path), and
/// anything else as itself.
fn section(out: &mut String, heading: &str, value: &Value) {
    let _ = writeln!(out, "<h2>{}</h2>", escape(heading));
    match value {
        Value::Array(rows) if rows.iter().all(Value::is_object) => {
            if rows.is_empty() {
                out.push_str("<p>None.</p>\n");
                return;
            }
            let mut columns: Vec<String> = Vec::new();
            let flattened: Vec<Vec<(String, &Value)>> = rows
                .iter()
                .map(|row| {
                    let mut fields = Vec::new();
                    flatten("", row, &mut fields);
                    for (column, _) in &fields {
                        if !columns.contains(column) {
                            columns.push(column.clone());
                        }
                    }
                    fields
                })
                .collect();
            out.push_str("<table>\n<tr>");
            for column in &columns {
                let _ = write!(out, "<th>{}</th>", escape(column));
            }
            out.push_str("</tr>\n");
            for fields in &flattened {
                out.push_str("<tr>");
                for column in &columns {
                    match fields.iter().find(|(name, _)| name == column) {
                        Some((_, value)) => cell(out, value),
                        None => out.push_str("<td></td>"),
                    }
                }
                out.push_str("</tr>\n");
            }
            out.push_str("</table>\n");
        }
        Value::Object(_) => {
            let mut fields = Vec::new();
            flatten("", value, &mut fields);
            if fields.is_empty() {
                out.push_str("<p>None.</p>\n");
                return;
            }
            out.push_str("<table>\n");
            for (name, value) in fields {
                let _ = write!(out, "<tr><th>{}</th>", escape(&name));
                cell(out, value);
                out.push_str("</tr>\n");
            }
            out.push_str("</table>\n");
        }
        other => {
            let _ = writeln!(out, "<p><code>{}</code></p>", scalar(other));
        }
    }
}

/// Every leaf of `value` by dotted path; an array that isn't all objects
/// is one leaf.
fn flatten<'a>(prefix: &str, value: &'a Value, fields: &mut Vec<(String, &'a Value)>) {
    match value {
        Value::Object(map) => {
            for (key, value) in map {
                let name = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}.{key}")
                };
                flatten(&name, value, fields);
            }
        }
        other => fields.push((prefix.to_owned(), other)),
    }
}

fn cell(out: &mut String, value: &Value) {
    if value.is_number() {
        let _ = write!(out, "<td class=\"number\">{}</td>", scalar(value));
    } else {
        let _ = write!(out, "<td>{}</td>", scalar(value));
    }
}

/// A value as escaped text: strings bare, numbers as JSON writes them, and
/// arrays and objects as compact JSON.
fn scalar(value: &Value) -> String {
    match value {
        Value::Null => "–".into(),
        Value::String(text) => escape(text),
        other => escape(&other.to_string()),
    }
}

fn escape(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '"' => escaped.push_str("&quot;"),
            '\'' => escaped.push_str("&#39;"),
            c => escaped.push(c),
        }
    }
    escaped
}
