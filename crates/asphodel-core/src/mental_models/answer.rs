//! A model's answer: sections, each an optional heading and one paragraph.
//!
//! Stored as Markdown, each section a `### heading` line followed by its
//! paragraph, sections separated by a blank line. A section without a
//! heading is a paragraph alone; only answers converted from entries
//! written before sections have one, and it comes first.
//!
//! The refresh trims an answer over `max_tokens`, and the block cuts one
//! that doesn't fit what's left of its budget, the same way: the last
//! sentence of the last section goes, and a section left empty takes its
//! heading with it, until it fits. A sentence ends at `.`, `!`, `?`, `。`,
//! `！` or `？` followed by whitespace or the end of the text.

use rusqlite::Connection;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Answer {
    sections: Vec<Section>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Section {
    heading: Option<String>,
    text: String,
}

/// Runs of whitespace, newlines included, as one space, so a paragraph is
/// one line and a heading can't break the layout.
fn collapse(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

impl Answer {
    /// The sections in order. A heading or text that's blank once its
    /// whitespace is collapsed counts as none, and a section without text
    /// is left out, heading and all.
    pub(crate) fn new(sections: impl IntoIterator<Item = (Option<String>, String)>) -> Self {
        let sections = sections
            .into_iter()
            .map(|(heading, text)| Section {
                heading: heading.map(|h| collapse(&h)).filter(|h| !h.is_empty()),
                text: collapse(&text),
            })
            .filter(|section| !section.text.is_empty())
            .collect();
        Self { sections }
    }

    /// A stored answer.
    pub(crate) fn parse(text: &str) -> Self {
        let sections = text.split("\n\n").map(|block| {
            let block = block.trim();
            match block.strip_prefix("### ") {
                Some(rest) => {
                    let (heading, text) = rest.split_once('\n').unwrap_or((rest, ""));
                    (Some(heading.to_owned()), text.to_owned())
                }
                None => (None, block.to_owned()),
            }
        });
        Self::new(sections)
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.sections.is_empty()
    }

    /// The stored text.
    pub(crate) fn text(&self) -> String {
        self.render("###")
    }

    /// The text with each heading marked as `marks`.
    pub(crate) fn render(&self, marks: &str) -> String {
        self.sections
            .iter()
            .map(|section| match &section.heading {
                Some(heading) => format!("{marks} {heading}\n{}", section.text),
                None => section.text.clone(),
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Takes the last sentence off the end until `fits` holds or nothing is
    /// left. Returns how many sentences went.
    pub(crate) fn trim_to(&mut self, fits: impl Fn(&Answer) -> bool) -> usize {
        let mut trimmed = 0;
        while !self.is_empty() && !fits(self) {
            self.pop_sentence();
            trimmed += 1;
        }
        trimmed
    }

    /// Takes the last sentence of the last section off; a section left
    /// empty goes with its heading.
    fn pop_sentence(&mut self) {
        let Some(last) = self.sections.last_mut() else {
            return;
        };
        match last_sentence_start(&last.text) {
            Some(start) => last.text.truncate(start),
            None => {
                self.sections.pop();
            }
        }
    }
}

/// Where the text before the last sentence ends, or `None` when the text is
/// one sentence.
fn last_sentence_start(text: &str) -> Option<usize> {
    let mut end = None;
    let mut chars = text.char_indices().peekable();
    while let Some((at, c)) = chars.next() {
        if !matches!(c, '.' | '!' | '?' | '。' | '！' | '？') {
            continue;
        }
        let follows = chars.peek().map(|(_, next)| *next);
        let boundary = at + c.len_utf8();
        if follows.is_some_and(char::is_whitespace) && !text[boundary..].trim().is_empty() {
            end = Some(boundary);
        }
    }
    end
}

/// Schema 15's conversion: each model's answer built from its entries,
/// grouped by section in position order with entries from before sections
/// first, and the union of their citations; then the entry tables go. Does
/// nothing once they're gone.
pub(crate) fn entries_to_answers(conn: &Connection) -> Result<(), rusqlite::Error> {
    let exists: bool = conn.query_row(
        "SELECT EXISTS (SELECT 1 FROM sqlite_master
                        WHERE type = 'table' AND name = 'mental_model_entries')",
        [],
        |row| row.get(0),
    )?;
    if !exists {
        return Ok(());
    }
    let rows: Vec<(i64, Option<String>, String)> = conn
        .prepare(
            "SELECT model_id, section, text FROM mental_model_entries
             ORDER BY model_id, position, id",
        )?
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?
        .collect::<Result<_, _>>()?;
    // Each model's entries from before sections, then each section's
    // entries under its heading, in the order the headings first come.
    type Grouped = (i64, Vec<String>, Vec<(String, Vec<String>)>);
    let mut models: Vec<Grouped> = Vec::new();
    for (model, section, text) in rows {
        if models.last().is_none_or(|(last, ..)| *last != model) {
            models.push((model, Vec::new(), Vec::new()));
        }
        let (_, unsectioned, sections) = models.last_mut().expect("pushed above");
        match section.filter(|section| !section.trim().is_empty()) {
            None => unsectioned.push(text),
            Some(heading) => match sections.iter_mut().find(|(h, _)| *h == heading) {
                Some((_, texts)) => texts.push(text),
                None => sections.push((heading, vec![text])),
            },
        }
    }
    for (model, unsectioned, sections) in models {
        let answer = Answer::new(
            std::iter::once((None, unsectioned.join(" "))).chain(
                sections
                    .into_iter()
                    .map(|(heading, texts)| (Some(heading), texts.join(" "))),
            ),
        );
        let text = (!answer.is_empty()).then(|| answer.text());
        conn.execute(
            "UPDATE mental_models SET answer = ?2 WHERE id = ?1",
            (model, text),
        )?;
        conn.execute(
            "DELETE FROM mental_model_cites WHERE model_id = ?1",
            [model],
        )?;
    }
    conn.execute_batch(
        "INSERT OR IGNORE INTO mental_model_cites (model_id, memory_id)
           SELECT e.model_id, c.memory_id FROM mental_model_citations c
           JOIN mental_model_entries e ON e.id = c.entry_id
           ORDER BY e.model_id, e.position, e.id, c.rowid;
         DROP TABLE IF EXISTS mental_model_citations;
         DROP TABLE mental_model_entries;",
    )
}
