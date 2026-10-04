//! The store contracts cover opening, locking, migrations, the pre-migration
//! copy and bank create-or-merge.
//!
//! Every store here runs on a `SimulatedClock` stopped at one instant, so a
//! stored time that equals that instant can only have come from the Clock.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use asphodel_core::Service;
use asphodel_core::clock::SimulatedClock;
use asphodel_core::config::{PurgePause, Tuning};
use asphodel_core::constants::Volatility;
use asphodel_core::mental_models::{Model, ModelEdit};
use asphodel_core::store::bank::{Bank, BankError, BankIdentity, ModelIds, PROFILE_NAME};
use asphodel_core::store::fs::FilesystemKind;
use asphodel_core::store::migrations::PRE_MIGRATION_COPY_TTL;
use asphodel_core::store::{DB_FILE, OpenOptions, SCHEMA_VERSION, Store, StoreError};
use asphodel_core::strength::Kind;
use jiff::{SignedDuration, Timestamp};

const START: &str = "2026-10-01T07:00:00Z";

fn start() -> Timestamp {
    START.parse().unwrap()
}

fn clock() -> Arc<SimulatedClock> {
    Arc::new(SimulatedClock::new(start()))
}

fn days(n: i64) -> SignedDuration {
    SignedDuration::from_hours(24 * n)
}

/// A temporary directory removed even when an assertion unwinds.
struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("asphodel-store-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn data(&self) -> PathBuf {
        self.0.join("data")
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn try_open(dir: &TestDir) -> Result<Store, StoreError> {
    Store::open(&dir.data(), OpenOptions::default(), clock())
}

fn open(dir: &Path, clock: Arc<SimulatedClock>) -> Store {
    Store::open(dir, OpenOptions::default(), clock).unwrap()
}

/// A service on `dir`'s store, opened at `clock`.
fn service_at(dir: &TestDir, clock: &Arc<SimulatedClock>, tuning: Tuning) -> Service {
    Service::open(clock.clone(), open(&dir.data(), clock.clone()), tuning)
}

/// Creates or merges `name` with the usual identity and models.
fn bank(service: &Service, name: &str) -> Bank {
    service.ensure_bank(name, &identity(), &models()).unwrap()
}

fn models() -> ModelIds {
    ModelIds {
        embedding: "bge-small-en-v1.5:int8".into(),
        reranker: "jina-reranker-v1-turbo-en:int8".into(),
    }
}

fn identity() -> BankIdentity {
    BankIdentity {
        owner_name: Some("Tim".into()),
        owner_platform_ids: vec!["discord:1234".into()],
        assistant_name: Some("Hermes".into()),
        timezone: Some("Pacific/Auckland".into()),
    }
}

/// Puts an open store back to schema `version`, as a binary that stopped
/// there left it: the next open migrates it again. Every migration since is
/// safe to run over a store that already has what it adds.
fn downgrade(store: &Store, version: u32) {
    let sql = format!(
        "DELETE FROM migrations WHERE to_version = {SCHEMA_VERSION};
         PRAGMA user_version = {version};"
    );
    store.connection().execute_batch(&sql).unwrap();
}

/// Downgrades `dir`'s store to `version` and reopens it, which migrates it
/// again. Returns the reopened store and the copy the migration kept.
fn migrated_from(dir: &TestDir, clock: &Arc<SimulatedClock>, version: u32) -> (Store, PathBuf) {
    downgrade(&open(&dir.data(), clock.clone()), version);
    let store = open(&dir.data(), clock.clone());
    let copy = store.applied().unwrap().copy.clone().unwrap();
    (store, copy)
}

/// The file names in `dir`.
fn listing(dir: &Path) -> BTreeSet<String> {
    let entries = std::fs::read_dir(dir).unwrap();
    let names = entries.map(|entry| entry.unwrap().file_name().into_string());
    names.map(Result::unwrap).collect()
}

// The data-dir lock

#[test]
fn a_data_dir_takes_one_store_until_it_is_dropped() {
    let dir = TestDir::new();
    let first = open(&dir.data(), clock());
    // Repeated refusals never release the holder's lock.
    for _ in 0..3 {
        let refused = try_open(&dir);
        let holder = std::process::id().to_string();
        assert!(
            matches!(&refused, Err(StoreError::Locked { dir: d, holder: h })
                if *d == dir.data() && *h == holder),
            "{refused:?}"
        );
    }
    assert_eq!(first.schema_version().unwrap(), SCHEMA_VERSION);
    drop(first);
    let second = open(&dir.data(), clock());
    assert_eq!(second.schema_version().unwrap(), SCHEMA_VERSION);
}

// Opening and migrating from empty

#[test]
fn a_missing_data_dir_is_created_and_a_file_is_refused() {
    let dir = TestDir::new();
    let missing = dir.0.join("new").join("deeper");
    let options = OpenOptions {
        allow_network_fs: true,
        ..OpenOptions::default()
    };
    let store = Store::open(&missing, options, clock()).unwrap();
    assert!(missing.is_dir());
    assert_eq!(store.filesystem(), FilesystemKind::Local);
    drop(store);

    let file = dir.0.join("not-a-dir");
    std::fs::write(&file, b"irreplaceable contents").unwrap();
    match Store::open(&file, OpenOptions::default(), clock()) {
        Err(StoreError::NotADirectory { dir }) => assert_eq!(dir, file),
        other => panic!("{other:?}"),
    }
    assert_eq!(std::fs::read(&file).unwrap(), b"irreplaceable contents");
}

#[test]
fn an_empty_data_dir_migrates_once_and_reopening_applies_nothing() {
    let dir = TestDir::new();
    let first = service(&dir);
    let applied = first.store().unwrap().applied().unwrap().clone();
    // Nothing to copy before the first schema.
    assert_eq!((applied.from, applied.to), (0, SCHEMA_VERSION));
    assert_eq!(applied.copy, None);
    assert_eq!(first.status().unwrap().pre_migration_copy, None);
    drop(first);

    let again = service(&dir);
    let store = again.store().unwrap();
    assert_eq!(store.applied(), None);
    assert_eq!(store.schema_version().unwrap(), SCHEMA_VERSION);
    assert_eq!(again.status().unwrap().pre_migration_copy, None);
}

#[test]
fn a_store_newer_than_the_binary_is_refused() {
    // Restore checks the schema version isn't newer than the
    // binary, and opening does the same.
    let dir = TestDir::new();
    let store = open(&dir.data(), clock());
    store
        .connection()
        .pragma_update(None, "user_version", SCHEMA_VERSION + 1)
        .unwrap();
    drop(store);
    match try_open(&dir) {
        Err(StoreError::NewerSchema {
            found, supported, ..
        }) => assert_eq!((found, supported), (SCHEMA_VERSION + 1, SCHEMA_VERSION)),
        other => panic!("{other:?}"),
    }
}

// The pre-migration copy

#[test]
fn a_migration_keeps_one_clean_whole_copy_of_the_old_version() {
    let dir = TestDir::new();
    let clock = clock();
    let service = service_at(&dir, &clock, Tuning::default());
    bank(&service, "main");
    downgrade(service.store().unwrap(), 11);
    drop(service);
    let before = listing(&dir.data());

    let store = open(&dir.data(), clock.clone());
    let applied = store.applied().unwrap().clone();
    assert_eq!((applied.from, applied.to), (11, SCHEMA_VERSION));
    let copy = applied.copy.expect("a store with data is copied first");
    let service = Service::open(clock.clone(), store, Tuning::default());
    let status = service.status().unwrap().pre_migration_copy.unwrap();
    assert_eq!(status.from_version, 11);
    assert_eq!(status.path, copy);
    assert_eq!(status.expires_at, start() + days(7));
    let clean = std::fs::read(&copy).unwrap();

    // The live database changes, as a crash-looping migration would change
    // it, and the next migration from the same version keeps the clean copy.
    bank(&service, "other");
    downgrade(service.store().unwrap(), 11);
    drop(service);
    let store = open(&dir.data(), clock.clone());
    assert_eq!(store.applied().unwrap().copy.as_ref(), Some(&copy));
    drop(store);
    assert_eq!(std::fs::read(&copy).unwrap(), clean);

    // Taking it, checking it read-only and reusing it left nothing beside
    // it: no partial, no journal.
    let mut expected = before;
    expected.insert(copy.file_name().unwrap().to_str().unwrap().to_owned());
    assert_eq!(listing(&dir.data()), expected);

    // Restored into a data dir of its own, it's the whole database at the
    // old version, from before the second bank.
    let restored = TestDir::new();
    std::fs::create_dir_all(restored.data()).unwrap();
    std::fs::copy(&copy, restored.data().join(DB_FILE)).unwrap();
    let store = open(&restored.data(), clock.clone());
    assert_eq!(store.applied().unwrap().from, 11);
    let service = Service::open(clock, store, Tuning::default());
    let banks = service.banks().unwrap();
    assert_eq!(banks.len(), 1);
    assert_eq!(banks[0].name, "main");
}

#[test]
fn a_corrupt_existing_copy_refuses_the_migration_and_is_kept() {
    // A copy for the same from-version is never overwritten. A damaged file
    // under the copy's name can't be trusted and can't be replaced, so the
    // migration must stop and name it.
    let dir = TestDir::new();
    let (store, copy) = migrated_from(&dir, &clock(), 11);
    downgrade(&store, 11);
    drop(store);
    std::fs::write(&copy, b"not a database").unwrap();

    match try_open(&dir) {
        Err(StoreError::CorruptCopy { path, .. }) => assert_eq!(path, copy),
        other => panic!("a corrupt copy was accepted: {other:?}"),
    }
    // The corrupt copy is kept.
    assert_eq!(std::fs::read(&copy).unwrap(), b"not a database");
}

#[test]
fn reopening_seven_days_after_the_migration_deletes_the_copy() {
    let dir = TestDir::new();
    let clock = clock();
    let (store, copy) = migrated_from(&dir, &clock, 11);
    drop(store);

    clock.advance(days(7) - SignedDuration::from_micros(1));
    drop(open(&dir.data(), clock.clone()));
    assert!(copy.exists(), "deleted before the week was up");
    clock.advance(SignedDuration::from_micros(1));
    drop(open(&dir.data(), clock.clone()));
    assert!(!copy.exists(), "kept after the week was up");
}

#[test]
fn housekeeping_expires_each_copy_at_its_deadline_even_while_purge_is_paused() {
    let dir = TestDir::new();
    let clock = clock();
    let hours = SignedDuration::from_hours;
    // History no binary writes today: a migration from 9 that crashed and
    // left its copy with no row, then one from 10 that started now but
    // completed three hours later, then one from 11 that completed two hours
    // from now. Rows go in version order; deadlines in the opposite one.
    let (store, incomplete) = migrated_from(&dir, &clock, 9);
    store
        .connection()
        .execute_batch("DELETE FROM migrations; PRAGMA user_version = 10;")
        .unwrap();
    drop(store);
    let store = open(&dir.data(), clock.clone());
    let late = store.applied().unwrap().copy.clone().unwrap();
    let sql = format!(
        "UPDATE migrations SET to_version = 11, completed_at = {} WHERE from_version = 10;
         PRAGMA user_version = 11;",
        (start() + hours(3)).as_microsecond()
    );
    store.connection().execute_batch(&sql).unwrap();
    drop(store);
    clock.advance(hours(2));
    let store = open(&dir.data(), clock.clone());
    let early = store.applied().unwrap().copy.clone().unwrap();

    let stored = Tuning::default().deletion_fingerprint();
    let running = store.check_fingerprint(&stored).unwrap();
    assert_eq!(running, PurgePause::Running);
    let mut tuning = Tuning::default();
    tuning.clock.quiet_rate = 0.2;
    let paused = PurgePause::Paused { stored };
    let pause = store.check_fingerprint(&tuning.deletion_fingerprint());
    let pause = pause.unwrap();
    assert_eq!(pause, paused);
    let service = Service::open(clock.clone(), store, tuning).with_purge_pause(pause);

    // The earliest deadline wins, not the first row's.
    let early_due = start() + hours(2) + PRE_MIGRATION_COPY_TTL;
    let upkeep = service.housekeeping().unwrap();
    assert!(upkeep.copies_removed.is_empty());
    assert_eq!(upkeep.next_due, Some(early_due));
    clock.advance(PRE_MIGRATION_COPY_TTL - SignedDuration::from_micros(1));
    assert!(service.housekeeping().unwrap().copies_removed.is_empty());
    clock.advance(SignedDuration::from_micros(1));
    let upkeep = service.housekeeping().unwrap();
    assert_eq!(upkeep.copies_removed, std::slice::from_ref(&early));
    assert!(!early.exists());
    assert!(late.exists());
    // The TTL counts from completion, not from the start.
    let late_due = start() + hours(3) + PRE_MIGRATION_COPY_TTL;
    assert_eq!(upkeep.next_due, Some(late_due));

    // A copy already gone has no deadline, and a copy whose migration never
    // completed has none either and is never deleted.
    std::fs::remove_file(&late).unwrap();
    clock.advance(hours(1));
    let upkeep = service.housekeeping().unwrap();
    assert!(upkeep.copies_removed.is_empty());
    assert_eq!(upkeep.next_due, None);
    assert!(incomplete.exists());

    // Only an acknowledgement moves the stored fingerprint.
    assert_eq!(service.status().unwrap().purge, paused);
    assert!(service.health().ready);
}

// Bank create-or-merge

fn service(dir: &TestDir) -> Service {
    service_at(dir, &clock(), Tuning::default())
}

fn profile(service: &Service) -> Model {
    let mut models = service.list_models("main").unwrap();
    assert_eq!(models.len(), 1, "only the profile");
    models.remove(0)
}

#[test]
fn creating_a_bank_mints_a_clock_ordered_id_and_seeds_user_assistant_and_profile() {
    let dir = TestDir::new();
    let clock = clock();
    let tuning = Tuning::from_toml("[mental_models]\nprofile_max_tokens = 777\n").unwrap();
    let service = service_at(&dir, &clock, tuning);
    let main = bank(&service, "main");
    assert!(main.created);
    assert_eq!(main.name, "main");
    assert_eq!(main.owner_name.as_deref(), Some("Tim"));
    assert_eq!(main.assistant_name.as_deref(), Some("Hermes"));
    assert_eq!(main.timezone, "Pacific/Auckland");
    assert_eq!(main.embedding_model, models().embedding);
    assert_eq!(main.reranker_model, models().reranker);
    // Public ids are UUIDv7, timed by the clock.
    assert_eq!(main.id.get_version_num(), 7);
    let (seconds, _) = main.id.get_timestamp().unwrap().to_unix();
    assert_eq!(i64::try_from(seconds).unwrap(), start().as_second());
    assert_eq!(service.banks().unwrap()[0].created_at, start());

    // Ids minted in one instant are distinct and sort in order; later ones
    // carry the clock's time and sort after.
    let ids: Vec<_> = (0..20)
        .map(|n| bank(&service, &format!("bank-{n:02}")).id)
        .collect();
    let mut sorted = ids.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted, ids);
    clock.advance(SignedDuration::from_secs(3600));
    let later = bank(&service, "later").id;
    let (seconds, _) = later.get_timestamp().unwrap().to_unix();
    assert_eq!(i64::try_from(seconds).unwrap(), start().as_second() + 3600);
    assert!(later > ids[19], "ids sort by the clock");

    // The names are the first aliases. The owner is a person; the seeded
    // assistant has type `thing`.
    assert_eq!(service.entities("main").unwrap().len(), 2);
    let user = service.show_entity("main", "user").unwrap();
    assert_eq!((user.name.as_str(), user.kind.as_str()), ("Tim", "person"));
    let mut aliases = user.aliases.clone();
    aliases.sort();
    assert_eq!(aliases, ["Tim", "discord:1234"]);
    assert_eq!(user.created_at, start());
    let assistant = service.show_entity("main", "assistant").unwrap();
    assert_eq!(assistant.name, "Hermes");
    assert_eq!(assistant.kind, "thing");
    assert_eq!(assistant.aliases, ["Hermes"]);

    // Only "User profile" is seeded, enabled, at the tuned budget. It takes
    // facts, plus states with volatility of weeks or slower, about the
    // whole bank.
    let profile = profile(&service);
    assert_eq!(profile.name, PROFILE_NAME);
    assert_eq!(profile.max_tokens, 777);
    assert!(profile.enabled);
    let mut kinds = profile.kinds.clone();
    kinds.sort_by_key(|kind| format!("{kind:?}"));
    assert_eq!(kinds, [Kind::Fact, Kind::State]);
    assert_eq!(profile.min_volatility, Some(Volatility::Weeks));
    assert_eq!(profile.entity, None);
}

#[test]
fn ensure_bank_is_idempotent_and_merges_only_what_is_present() {
    let dir = TestDir::new();
    let service = service(&dir);
    let first = bank(&service, "main");
    let mut second = bank(&service, "main");
    assert!(first.created && !second.created);
    second.created = true;
    assert_eq!(second, first);

    // Present fields are set, absent ones left alone. A rename adds an
    // alias and removes none. The models are recorded at creation; changing
    // them is `asphodel reembed`, not a config merge.
    let other_models = ModelIds {
        embedding: "granite-embedding-small-english-r2:int8".into(),
        reranker: models().reranker,
    };
    let renamed = BankIdentity {
        owner_name: Some("Timothy".into()),
        assistant_name: Some("Jarvis".into()),
        timezone: Some("Europe/Lisbon".into()),
        owner_platform_ids: vec!["telegram:9".into()],
    };
    let merged = service
        .ensure_bank("main", &renamed, &other_models)
        .unwrap();
    assert_eq!(merged.id, first.id);
    assert_eq!(merged.timezone, "Europe/Lisbon");
    assert_eq!(merged.owner_name.as_deref(), Some("Timothy"));
    assert_eq!(merged.assistant_name.as_deref(), Some("Jarvis"));
    assert_eq!(merged.embedding_model, models().embedding);
    let unchanged = service.ensure_bank("main", &BankIdentity::default(), &models());
    assert_eq!(unchanged.unwrap(), merged);

    let aliases = |entity: &str| {
        let mut aliases = service.show_entity("main", entity).unwrap().aliases;
        aliases.sort();
        aliases
    };
    let user = aliases("user");
    assert_eq!(user, ["Tim", "Timothy", "discord:1234", "telegram:9"]);
    assert_eq!(aliases("assistant"), ["Hermes", "Jarvis"]);
    assert_eq!(service.entities("main").unwrap().len(), 2);
    assert_eq!(service.banks().unwrap().len(), 1);
    assert_eq!(service.list_models("main").unwrap().len(), 1);
}

#[test]
fn the_profile_is_seeded_on_create_only() {
    // an owner who deletes it doesn't get it back on
    // the next Hermes start.
    let dir = TestDir::new();
    let service = service(&dir);
    service.ensure_bank("main", &identity(), &models()).unwrap();
    service
        .store()
        .unwrap()
        .connection()
        .execute("DELETE FROM mental_models", [])
        .unwrap();
    service.ensure_bank("main", &identity(), &models()).unwrap();
    assert!(service.list_models("main").unwrap().is_empty());
}

#[test]
fn an_unknown_timezone_or_empty_name_creates_nothing() {
    let dir = TestDir::new();
    let service = service(&dir);
    let mars = BankIdentity {
        timezone: Some("Mars/Olympus_Mons".into()),
        ..BankIdentity::default()
    };
    let refused = service.ensure_bank("main", &mars, &models());
    assert!(matches!(refused, Err(BankError::InvalidTimezone)));
    let refused = service.ensure_bank("  ", &identity(), &models());
    assert!(matches!(refused, Err(BankError::EmptyName)));
    assert!(service.banks().unwrap().is_empty());
}

#[test]
fn merge_leaves_the_profiles_filters_alone() {
    // An owner edit to a model's filters survives every later
    // `PUT /v1/banks/{bank}`.
    let dir = TestDir::new();
    let service = service(&dir);
    bank(&service, "main");
    let edit = ModelEdit {
        question: Some("edited".into()),
        kinds: Some(vec![Kind::Fact]),
        min_volatility: Some(Some(Volatility::Months)),
        max_tokens: Some(123),
        enabled: None,
    };
    let edited = service.edit_model("main", PROFILE_NAME, &edit).unwrap();
    bank(&service, "main");
    let profile = profile(&service);
    assert_eq!(profile.question, "edited");
    assert_eq!(profile.kinds, [Kind::Fact]);
    assert_eq!(profile.min_volatility, Some(Volatility::Months));
    assert_eq!(profile.max_tokens, 123);
    assert_eq!(profile, edited);
}
