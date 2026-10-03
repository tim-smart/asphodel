//! Splitting sources into chunks, the unit extraction runs on.
//!
//! A turn is one chunk. A document is one chunk per section, a section
//! running from an ATX heading to the next heading of any level, and a
//! section is split further only when it's longer than [`CHUNK_CHARS`]:
//! at paragraph breaks first, then line breaks, then sentence ends, then
//! whitespace, and at a hard cut only as a last resort. Small sections are
//! never merged, so editing one section never changes another's hash.
//!
//! Offsets are in characters, not bytes.
//! A chunk drops the blank lines before and after it, so a section's hash
//! doesn't depend on the blank lines that follow it or on whether it's last.

use sha2::{Digest, Sha256};

use crate::constants::CHUNK_CHARS;

/// One chunk of a document, before it's stored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocumentChunk {
    /// The headings above this chunk, outermost first, without their `#`
    /// marks. Empty for text before the first heading and for a document
    /// with no headings.
    pub heading_path: Vec<String>,
    /// The chunk's range in the document, in characters, end exclusive.
    pub start: usize,
    pub end: usize,
    /// The document's characters `start..end`.
    pub text: String,
}

/// Splits a plain-text or markdown document. Lines inside fenced code blocks
/// are never headings, and `#tag` with no space isn't one either.
pub fn split_document(text: &str) -> Vec<DocumentChunk> {
    let chars: Vec<char> = text.chars().collect();
    let mut chunks = Vec::new();
    for section in sections(&chars) {
        for (start, end) in pieces(&chars, section.start, section.end, Level::Paragraph) {
            chunks.push(DocumentChunk {
                heading_path: section.heading_path.clone(),
                start,
                end,
                text: chars[start..end].iter().collect(),
            });
        }
    }
    chunks
}

/// A chunk's identity: a hash of its text and its heading path, fixed at
/// ingest and never recomputed. It's the forget tombstone, not an integrity
/// check. Every part is length-prefixed, so different paths never
/// collide however their headings are spelled.
pub fn chunk_hash(heading_path: &[String], text: &str) -> String {
    let mut hash = Sha256::new();
    hash.update(b"asphodel chunk v1\0");
    hash.update((heading_path.len() as u64).to_le_bytes());
    for heading in heading_path {
        hash.update((heading.len() as u64).to_le_bytes());
        hash.update(heading.as_bytes());
    }
    hash.update((text.len() as u64).to_le_bytes());
    hash.update(text.as_bytes());
    hex(&hash.finalize())
}

/// Lower-case hex of `bytes`.
pub(crate) fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes.iter().fold(String::new(), |mut out, byte| {
        let _ = write!(out, "{byte:02x}");
        out
    })
}

struct Section {
    heading_path: Vec<String>,
    start: usize,
    end: usize,
}

/// The document's sections in order, each with its heading path. A section
/// starts at its heading line; text before the first heading is a section
/// with an empty path.
fn sections(chars: &[char]) -> Vec<Section> {
    let mut sections = Vec::new();
    let mut stack: Vec<(usize, String)> = Vec::new();
    let mut current = Section {
        heading_path: Vec::new(),
        start: 0,
        end: 0,
    };
    let mut fence: Option<(char, usize)> = None;
    for (start, end) in lines(chars) {
        let line: String = chars[start..end].iter().collect();
        if let Some(marker) = fence_marker(&line) {
            match fence {
                None => fence = Some(marker),
                Some((open, length)) if marker.0 == open && marker.1 >= length => fence = None,
                Some(_) => {}
            }
            continue;
        }
        if fence.is_some() {
            continue;
        }
        if let Some((level, heading)) = atx_heading(&line) {
            current.end = start;
            sections.push(current);
            while stack.last().is_some_and(|(open, _)| *open >= level) {
                stack.pop();
            }
            stack.push((level, heading));
            current = Section {
                heading_path: stack.iter().map(|(_, heading)| heading.clone()).collect(),
                start,
                end: 0,
            };
        }
    }
    current.end = chars.len();
    sections.push(current);
    sections
        .into_iter()
        .filter(|section| {
            chars[section.start..section.end]
                .iter()
                .any(|c| !c.is_whitespace())
        })
        .collect()
}

/// The lines of `chars` as ranges, each without its line break.
fn lines(chars: &[char]) -> Vec<(usize, usize)> {
    let mut lines = Vec::new();
    let mut start = 0;
    for (index, c) in chars.iter().enumerate() {
        if *c == '\n' {
            lines.push((start, index));
            start = index + 1;
        }
    }
    if start < chars.len() {
        lines.push((start, chars.len()));
    }
    lines
}

/// The fence character and length when `line` opens or closes a fenced code
/// block: up to three spaces, then three or more backticks or tildes.
fn fence_marker(line: &str) -> Option<(char, usize)> {
    let rest = strip_indent(line)?;
    let first = rest.chars().next()?;
    if first != '`' && first != '~' {
        return None;
    }
    let length = rest.chars().take_while(|c| *c == first).count();
    (length >= 3).then_some((first, length))
}

/// The level and text of an ATX heading: up to three spaces, one to six
/// `#`, then a space, a tab or the end of the line. A closing run of `#`
/// is dropped.
fn atx_heading(line: &str) -> Option<(usize, String)> {
    let rest = strip_indent(line)?;
    let level = rest.chars().take_while(|c| *c == '#').count();
    if !(1..=6).contains(&level) {
        return None;
    }
    let rest = &rest[level..];
    if !(rest.is_empty() || rest.starts_with(' ') || rest.starts_with('\t')) {
        return None;
    }
    let mut heading = rest.trim();
    let without_closing = heading.trim_end_matches('#');
    if without_closing.is_empty() {
        heading = "";
    } else if without_closing.ends_with([' ', '\t']) {
        heading = without_closing.trim_end();
    }
    Some((level, heading.to_owned()))
}

/// `line` without up to three leading spaces, or `None` when it's indented
/// further (an indented code block).
fn strip_indent(line: &str) -> Option<&str> {
    let indent = line.chars().take_while(|c| *c == ' ').count();
    (indent <= 3).then(|| &line[indent..])
}

/// Where a range that's too long is split, coarsest first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Level {
    Paragraph,
    Line,
    Sentence,
    Whitespace,
    Anywhere,
}

impl Level {
    fn finer(self) -> Level {
        match self {
            Level::Paragraph => Level::Line,
            Level::Line => Level::Sentence,
            Level::Sentence => Level::Whitespace,
            Level::Whitespace | Level::Anywhere => Level::Anywhere,
        }
    }
}

/// `start..end` without the blank lines at either end; `None` if nothing's
/// left.
fn trim(chars: &[char], start: usize, end: usize) -> Option<(usize, usize)> {
    let first = (start..end).find(|i| !chars[*i].is_whitespace())?;
    let last = (first..end).rev().find(|i| !chars[*i].is_whitespace())?;
    // Whole lines: keep the indent before the first character and the
    // spaces after the last, within the range.
    let start = (start..first)
        .rev()
        .find(|i| chars[*i] == '\n')
        .map_or(start, |i| i + 1);
    let end = (last + 1..end).find(|i| chars[*i] == '\n').unwrap_or(end);
    Some((start, end))
}

/// Splits `start..end` into trimmed pieces of at most [`CHUNK_CHARS`],
/// breaking at `level` and packing neighbours back together while they fit.
fn pieces(chars: &[char], start: usize, end: usize, level: Level) -> Vec<(usize, usize)> {
    let Some((start, end)) = trim(chars, start, end) else {
        return Vec::new();
    };
    if end - start <= CHUNK_CHARS {
        return vec![(start, end)];
    }
    let mut units = Vec::new();
    for (unit_start, unit_end) in units_at(chars, start, end, level) {
        let Some((unit_start, unit_end)) = trim(chars, unit_start, unit_end) else {
            continue;
        };
        if unit_end - unit_start > CHUNK_CHARS {
            units.extend(pieces(chars, unit_start, unit_end, level.finer()));
        } else {
            units.push((unit_start, unit_end));
        }
    }
    let mut packed: Vec<(usize, usize)> = Vec::new();
    for (unit_start, unit_end) in units {
        match packed.last_mut() {
            Some(last) if unit_end - last.0 <= CHUNK_CHARS => last.1 = unit_end,
            _ => packed.push((unit_start, unit_end)),
        }
    }
    packed
}

/// `start..end` cut at every break of `level`. The units cover the range.
fn units_at(chars: &[char], start: usize, end: usize, level: Level) -> Vec<(usize, usize)> {
    let mut cuts = Vec::new();
    match level {
        Level::Paragraph => {
            // After a line break followed by a whitespace-only line.
            let mut index = start;
            while index < end {
                if chars[index] == '\n' {
                    let mut next = index + 1;
                    while next < end && chars[next] != '\n' && chars[next].is_whitespace() {
                        next += 1;
                    }
                    if next < end && chars[next] == '\n' {
                        cuts.push(next + 1);
                        index = next;
                        continue;
                    }
                }
                index += 1;
            }
        }
        Level::Line => cuts.extend((start..end).filter(|i| chars[*i] == '\n').map(|i| i + 1)),
        Level::Sentence => cuts.extend(
            (start + 1..end)
                .filter(|i| chars[*i].is_whitespace() && matches!(chars[i - 1], '.' | '!' | '?'))
                .map(|i| i + 1),
        ),
        Level::Whitespace => cuts.extend(
            (start..end)
                .filter(|i| chars[*i].is_whitespace())
                .map(|i| i + 1),
        ),
        Level::Anywhere => cuts.extend((start + CHUNK_CHARS..end).step_by(CHUNK_CHARS)),
    }
    let mut units = Vec::new();
    let mut from = start;
    for cut in cuts {
        if cut > from && cut < end {
            units.push((from, cut));
            from = cut;
        }
    }
    units.push((from, end));
    units
}
