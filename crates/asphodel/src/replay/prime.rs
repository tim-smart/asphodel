//! `--prime-concurrency`: call 1 for every chunk, recorded before a `fast`
//! simulation (`docs/replay.md`, "Priming").
//!
//! `fast` reuses call 1's claims by chunk rather than by request, so they
//! can be recorded ahead of the simulation, many at once. The prime
//! ingests the timeline's turns and documents into the run's store in the
//! order the simulation syncs them, takes each chunk off the queue in the
//! order a serial run claims it, and builds its call 1 with nothing in
//! context: no in-context memories, no mental model entries, and no
//! entities but those ingestion makes. Chunks the cassette already has
//! claims for are skipped. The simulation's `open_store` resets the store
//! afterwards.

use std::path::Path;
use std::sync::Arc;

use asphodel_core::extraction::call1_request;
use asphodel_core::ingest::{Document, Turn, TurnAuthor};
use asphodel_core::store::bank::BankIdentity;
use asphodel_core::{Clock, Models, Service, SimulatedClock, Tuning};
use jiff::Timestamp;

use super::cassette::{ChunkContext, ChunkKey, Priming, Recorder};
use super::timeline::{self, Timeline};

/// What the prime builds the chunks' call 1s from.
pub(super) struct Input<'a> {
    pub dir: &'a Path,
    pub timeline: &'a Timeline,
    pub tuning: &'a Tuning,
    pub models: &'a Models,
    pub bank: &'a str,
    pub identity: &'a BankIdentity,
    pub start: Timestamp,
}

/// A source as the simulation syncs it.
enum Source<'a> {
    Turn(&'a timeline::Turn),
    Document(&'a super::scenario::Document),
}

/// The call 1s to prime: one for each chunk `recorder` has no claims for,
/// in the order a serial run claims them, each at its source's sync.
pub(super) fn chunks(input: &Input, recorder: &Recorder) -> anyhow::Result<Vec<Priming>> {
    let clock = Arc::new(SimulatedClock::new(input.start));
    let store = super::open_store(input.dir, Arc::clone(&clock) as Arc<dyn Clock>)?;
    store.check_fingerprint(&input.tuning.deletion_fingerprint())?;
    let service = Service::with_models(
        Arc::clone(&clock) as Arc<dyn Clock>,
        store,
        input.tuning.clone(),
        super::clone_models(input.models),
    )?;
    service.ensure_bank_with_models(input.bank, input.identity)?;

    // The engine's order: turns by message time, each synced at its reply,
    // then documents, stable by sync time.
    let mut turns: Vec<&timeline::Turn> = input
        .timeline
        .turns
        .iter()
        .filter(|turn| !turn.prefetch_only)
        .collect();
    turns.sort_by_key(|turn| turn.at);
    let mut syncs: Vec<(Timestamp, Source)> = turns
        .into_iter()
        .map(|turn| (turn.reply_at.max(turn.at), Source::Turn(turn)))
        .chain(
            input
                .timeline
                .documents
                .iter()
                .map(|document| (document.at, Source::Document(document))),
        )
        .collect();
    syncs.sort_by_key(|(at, _)| *at);

    let mut chunks = Vec::new();
    for (at, source) in syncs {
        clock.set(at);
        match source {
            Source::Turn(turn) => {
                service.ingest_turn(
                    input.bank,
                    &Turn {
                        session_id: turn.session.clone(),
                        message_at: turn.at,
                        timezone: None,
                        user_text: turn.user.clone(),
                        assistant_text: turn.assistant.clone(),
                        author: turn.author.as_ref().map(|author| TurnAuthor {
                            id: author.id.clone(),
                            name: author.name.clone(),
                            is_bot: false,
                        }),
                        platform: turn.platform.clone(),
                        recall_id: None,
                        forget_requested: false,
                    },
                )?;
            }
            Source::Document(document) => {
                service.ingest_document(
                    input.bank,
                    &Document {
                        document_id: document.id.clone(),
                        text: document.text.clone(),
                        reference_date: document.reference_date,
                        reference_date_exact: true,
                        timezone: document.timezone.clone(),
                    },
                )?;
            }
        }
        while let Some(lease) = service.claim_chunk(input.bank)? {
            let call1 = service.call1_input(&lease, &[])?;
            let key = ChunkKey {
                source: lease.source,
                position: lease.position,
            };
            service.complete_chunk(lease)?;
            if !recorder.has_claims(&key) {
                chunks.push(Priming {
                    request: call1_request(&call1),
                    context: ChunkContext::new(key, &call1),
                    at,
                });
            }
        }
    }
    Ok(chunks)
}
