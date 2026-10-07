//! Models that give their memory back when idle and load again on use.
//!
//! Each model has its own slot, so extraction, which only embeds, never
//! keeps the reranker loaded. A slot releases its model only when no call
//! is running on it and the last call ended at least the idle time ago.
//! The time is real latency, measured with [`Instant`] like the reranker
//! deadline, not the service's clock: replay runs years of bank time in
//! minutes, and memory is a property of the process.
//!
//! Every call holds its own `Arc` of the model while it runs, so even a
//! release that raced it can't free memory under a running inference. The
//! running count is what stops the idle check from dropping a model a call
//! is using, which would load a second copy on the next call. That includes
//! a reranker call its caller gave up on at the deadline, which runs on.

use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use super::{Embedder, ModelDir, ModelError, ModelOptions, Models, Reranker};

/// Loads a model again after a release.
type Load<T> = Box<dyn Fn() -> Result<Arc<T>, ModelError> + Send + Sync>;

/// The embedder and the reranker, each released when idle and loaded again
/// on its next call. [`ResidentModels::models`] hands out [`Models`] that
/// the service uses as it would loaded ones.
pub struct ResidentModels {
    embedder: Arc<Slot<dyn Embedder>>,
    /// The embedder's width, which never needs it loaded.
    dimensions: usize,
    reranker: Arc<Slot<dyn Reranker>>,
}

impl ResidentModels {
    /// Loads each model once now, so a model that can't load fails here,
    /// and keeps the loaders for after a release.
    pub fn new(
        load_embedder: impl Fn() -> Result<Arc<dyn Embedder>, ModelError> + Send + Sync + 'static,
        load_reranker: impl Fn() -> Result<Arc<dyn Reranker>, ModelError> + Send + Sync + 'static,
    ) -> Result<Self, ModelError> {
        let models = Models {
            embedder: load_embedder()?,
            reranker: load_reranker()?,
        };
        Ok(Self::loaded(
            models,
            Box::new(load_embedder),
            Box::new(load_reranker),
        ))
    }

    /// The ONNX models from `dir`, loaded now as [`Models::load`] does, so
    /// a missing or corrupt file stops startup. A reload checks the files
    /// again.
    pub fn load(dir: &ModelDir, options: &ModelOptions) -> Result<Self, ModelError> {
        let models = Models::load(dir, options)?;
        let load_embedder = {
            let (dir, options) = (dir.clone(), *options);
            move || -> Result<Arc<dyn Embedder>, ModelError> {
                Ok(Arc::new(super::load_embedder(&dir, &options)?))
            }
        };
        let load_reranker = {
            let (dir, options) = (dir.clone(), *options);
            move || -> Result<Arc<dyn Reranker>, ModelError> {
                Ok(Arc::new(super::load_reranker(&dir, &options)?))
            }
        };
        Ok(Self::loaded(
            models,
            Box::new(load_embedder),
            Box::new(load_reranker),
        ))
    }

    fn loaded(
        models: Models,
        load_embedder: Load<dyn Embedder>,
        load_reranker: Load<dyn Reranker>,
    ) -> Self {
        let dimensions = models.embedder.dimensions();
        let embedder = Slot {
            model_id: models.embedder.model_id().to_string(),
            load: load_embedder,
            state: Mutex::new(State::loaded(models.embedder)),
            loading: Mutex::new(()),
        };
        let reranker = Slot {
            model_id: models.reranker.model_id().to_string(),
            load: load_reranker,
            state: Mutex::new(State::loaded(models.reranker)),
            loading: Mutex::new(()),
        };
        Self {
            embedder: Arc::new(embedder),
            dimensions,
            reranker: Arc::new(reranker),
        }
    }

    /// The models to serve with. Their ids and dimensions answer without
    /// loading anything; a call loads its model if it was released.
    pub fn models(&self) -> Models {
        Models {
            embedder: Arc::new(ResidentEmbedder {
                slot: Arc::clone(&self.embedder),
                dimensions: self.dimensions,
            }),
            reranker: Arc::new(ResidentReranker(Arc::clone(&self.reranker))),
        }
    }

    /// Releases each model no call is running on and none has used for
    /// `idle`, then hands the freed memory back to the system. Whether
    /// either was released.
    pub fn release_idle(&self, idle: Duration) -> bool {
        let embedder = self.embedder.release_idle(idle);
        let reranker = self.reranker.release_idle(idle);
        if embedder || reranker {
            trim();
            return true;
        }
        false
    }
}

/// One model's place: loaded or released, and how it's being used.
struct Slot<T: ?Sized> {
    model_id: String,
    load: Load<T>,
    state: Mutex<State<T>>,
    /// Held while a call finds or loads the model, so callers that find it
    /// released wait for one load rather than each starting their own.
    loading: Mutex<()>,
}

struct State<T: ?Sized> {
    /// `None` once released.
    model: Option<Arc<T>>,
    /// The calls running on the model now.
    running: usize,
    /// When the model was loaded or a call on it last ended.
    used: Instant,
}

impl<T: ?Sized> State<T> {
    fn loaded(model: Arc<T>) -> Self {
        Self {
            model: Some(model),
            running: 0,
            used: Instant::now(),
        }
    }
}

impl<T: ?Sized> Slot<T> {
    fn state(&self) -> MutexGuard<'_, State<T>> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Runs `call` on the model, loading it first if it was released.
    fn with<R>(&self, call: impl FnOnce(&T) -> Result<R, ModelError>) -> Result<R, ModelError> {
        let running = self.enter()?;
        call(&running.model)
    }

    /// The model, counted as running until the guard drops, even if the
    /// call panics.
    fn enter(&self) -> Result<Running<'_, T>, ModelError> {
        let _loading = self
            .loading
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut state = self.state();
        let model = match &state.model {
            Some(model) => Arc::clone(model),
            None => {
                drop(state);
                let started = Instant::now();
                let model = (self.load)()?;
                tracing::info!(
                    model = %self.model_id,
                    elapsed_ms = started.elapsed().as_millis() as u64,
                    "loaded a released model again"
                );
                state = self.state();
                state.model = Some(Arc::clone(&model));
                model
            }
        };
        state.running += 1;
        Ok(Running { slot: self, model })
    }

    /// Drops the model if nothing is running on it and it's been idle for
    /// `idle`. Whether it did.
    fn release_idle(&self, idle: Duration) -> bool {
        let model = {
            let mut state = self.state();
            if state.model.is_none() || state.running > 0 || state.used.elapsed() < idle {
                return false;
            }
            state.model.take()
        };
        // Dropped outside the lock: freeing a session takes a moment, and a
        // call arriving meanwhile only waits to load it again.
        drop(model);
        tracing::info!(model = %self.model_id, "released an idle model");
        true
    }
}

/// A call running on a slot's model.
struct Running<'a, T: ?Sized> {
    slot: &'a Slot<T>,
    model: Arc<T>,
}

impl<T: ?Sized> Drop for Running<'_, T> {
    fn drop(&mut self) {
        let mut state = self.slot.state();
        state.running -= 1;
        state.used = Instant::now();
    }
}

struct ResidentEmbedder {
    slot: Arc<Slot<dyn Embedder>>,
    dimensions: usize,
}

impl Embedder for ResidentEmbedder {
    fn model_id(&self) -> &str {
        &self.slot.model_id
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, ModelError> {
        self.slot.with(|model| model.embed(texts))
    }
}

struct ResidentReranker(Arc<Slot<dyn Reranker>>);

impl Reranker for ResidentReranker {
    fn model_id(&self) -> &str {
        &self.0.model_id
    }

    fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<f32>, ModelError> {
        self.0.with(|model| model.rerank(query, documents))
    }
}

/// Hands freed heap back to the system. glibc keeps what a dropped model
/// freed in its arenas, so without this the process's resident memory
/// doesn't fall at all.
fn trim() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    // SAFETY: malloc_trim only returns free memory to the system; it
    // touches no allocation still in use.
    unsafe {
        libc::malloc_trim(0);
    }
}
