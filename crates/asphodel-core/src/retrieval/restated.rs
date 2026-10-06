//! What explicit recall shows and reranks of a memory's restatements
//! (TIM-206 measurement branch).
//!
//! A recalled memory is its head's sentence and the newest
//! [`RESTATED_SHOWN`] distinct sentences restated into any live version of
//! its chain. Restatements are told apart by their exact sentence, trimmed
//! of surrounding whitespace and nothing else, with the head's sentence
//! counted as already seen, so a repeat of the head or of another
//! restatement takes no place. A sentence said more than once stands at the
//! last time it was said, which decides both whether it's among the newest
//! and where it's shown. Equal times go to the higher rowid. The reranker
//! scores exactly these sentences and the head, and the text shows them, so
//! the two can't disagree.
//!
//! Restatements on a retracted or hidden version are left out, as its own
//! sentence is.

use std::collections::BTreeSet;

use rusqlite::Connection;

use crate::constants::RESTATED_SHOWN;

/// The restatements `head` is shown and reranked with, oldest first:
/// see the module docs. `members` is the head's chain.
pub(super) fn select(
    conn: &Connection,
    members: &BTreeSet<i64>,
    head: &str,
) -> Result<Vec<String>, rusqlite::Error> {
    let list = members
        .iter()
        .map(i64::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let mut statement = conn.prepare(&format!(
        "SELECT json_extract(r.claim, '$.sentence')
         FROM restatements r JOIN memories m ON m.id = r.memory_id
         WHERE r.memory_id IN ({list})
           AND m.invalidated_at IS NULL AND m.hidden_at IS NULL
         ORDER BY r.observed_at DESC, r.id DESC"
    ))?;
    let sentences = statement.query_map([], |row| row.get::<_, Option<String>>(0))?;
    let mut seen = BTreeSet::from([head.trim().to_owned()]);
    let mut newest = Vec::new();
    for sentence in sentences {
        let Some(sentence) = sentence? else {
            continue;
        };
        let sentence = sentence.trim();
        if sentence.is_empty() || !seen.insert(sentence.to_owned()) {
            continue;
        }
        newest.push(sentence.to_owned());
        if newest.len() == RESTATED_SHOWN {
            break;
        }
    }
    newest.reverse();
    Ok(newest)
}

/// A recalled memory's text: `head`, then each of `restated` marked as
/// restated, on one line, since the recall tool gives each memory a line.
pub(super) fn compose(head: &str, restated: &[String]) -> String {
    let mut text = head.to_owned();
    for sentence in restated {
        text.push_str(" (restated: ");
        text.push_str(&sentence.replace(['\n', '\r'], " "));
        text.push(')');
    }
    text
}

/// Schema 20's conversion: indexes every memory and restatement sentence in
/// `recall_fts`, and adds the triggers that keep it in step with
/// `memories` and `restatements`. A
/// memory's row is twice its rowid, a restatement's twice its rowid plus
/// one. Safe to run again: rows already indexed are skipped, and the
/// triggers are created only once.
pub(crate) fn index_for_recall(conn: &Connection) -> Result<(), rusqlite::Error> {
    conn.execute_batch(
        "INSERT INTO recall_fts (rowid, sentence)
           SELECT id * 2, content FROM memories
           WHERE id * 2 NOT IN (SELECT rowid FROM recall_fts);
         INSERT INTO recall_fts (rowid, sentence)
           SELECT id * 2 + 1, json_extract(claim, '$.sentence') FROM restatements
           WHERE id * 2 + 1 NOT IN (SELECT rowid FROM recall_fts);

         CREATE TRIGGER IF NOT EXISTS recall_fts_memory_insert AFTER INSERT ON memories BEGIN
           INSERT INTO recall_fts (rowid, sentence) VALUES (new.id * 2, new.content);
         END;
         CREATE TRIGGER IF NOT EXISTS recall_fts_memory_delete AFTER DELETE ON memories BEGIN
           DELETE FROM recall_fts WHERE rowid = old.id * 2;
         END;
         CREATE TRIGGER IF NOT EXISTS recall_restatement_insert AFTER INSERT ON restatements BEGIN
           INSERT INTO recall_fts (rowid, sentence)
             VALUES (new.id * 2 + 1, json_extract(new.claim, '$.sentence'));
         END;
         CREATE TRIGGER IF NOT EXISTS recall_restatement_delete AFTER DELETE ON restatements BEGIN
           DELETE FROM recall_fts WHERE rowid = old.id * 2 + 1;
         END;",
    )
}
