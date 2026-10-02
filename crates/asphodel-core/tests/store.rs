//! The store, checked against "Store: SQLite schema, migrations and the
//! data-dir lock" (TIM-103) and the decisions it rests on: "What is a memory
//! record?" (TIM-90), "Rust storage and search stack" (TIM-89), "API surface
//! and Hermes transport" (TIM-94, decisions 4 and 7), "Mental models"
//! (TIM-95, decisions 2 and 6), ADR 0002, ADR 0008, ADR 0009 and ADR 0010.
//!
//! Every store here runs on a `SimulatedClock` stopped at one instant, so a
//! stored time that equals that instant can only have come from the Clock.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use asphodel_core::Service;
use asphodel_core::clock::{Clock, SimulatedClock};
use asphodel_core::config::{PurgePause, Tuning};
use asphodel_core::store::bank::{BankError, BankIdentity, ModelIds, PROFILE_NAME};
use asphodel_core::store::fs::{self, FilesystemKind, classify, classify_name};
use asphodel_core::store::{
    DB_FILE, DataDirLock, EMBEDDING_DIMENSIONS, LOCK_FILE, OpenOptions, SCHEMA_VERSION, Store,
    StoreError, VectorError, VectorIndex, micros, migrations,
};
use jiff::{SignedDuration, Timestamp};
use rusqlite::Connection;

const START: &str = "2026-10-01T07:00:00Z";

fn start() -> Timestamp {
    START.parse().unwrap()
}

fn clock() -> Arc<SimulatedClock> {
    Arc::new(SimulatedClock::new(start()))
}

/// A temporary directory removed even when an assertion unwinds.
struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "asphodel-store-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
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

fn open(dir: &Path, clock: Arc<SimulatedClock>) -> Store {
    Store::open(dir, OpenOptions::default(), clock).unwrap()
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

fn count(conn: &Connection, sql: &str) -> i64 {
    conn.query_row(sql, [], |row| row.get(0)).unwrap()
}

fn bank_rowid(conn: &Connection, name: &str) -> i64 {
    conn.query_row("SELECT id FROM banks WHERE name = ?1", [name], |row| {
        row.get(0)
    })
    .unwrap()
}

/// A source and one chunk in `bank`, so memories have something to rest on.
fn seed_chunk(conn: &Connection, bank: i64, key: &str, now: i64) -> i64 {
    conn.execute(
        "INSERT INTO sources (uuid, bank_id, kind, session_id, message_at, content_hash,
                              observed_at, timezone, text, reply, ingested_at)
         VALUES (?1, ?2, 'turn', 'session', ?3, ?1, ?3, 'UTC', 'hi', 'hello', ?3)",
        (format!("source-{key}"), bank, now),
    )
    .unwrap();
    let source = conn.last_insert_rowid();
    conn.execute(
        "INSERT INTO chunks (uuid, bank_id, source_id, position, content_hash, start_offset,
                             end_offset, text)
         VALUES (?1, ?2, ?3, 0, ?1, 0, 2, 'hi')",
        (format!("chunk-{key}"), bank, source),
    )
    .unwrap();
    conn.last_insert_rowid()
}

fn insert_memory(conn: &Connection, bank: i64, chunk: i64, content: &str, now: i64) -> i64 {
    conn.execute(
        "INSERT INTO memories (uuid, bank_id, content, kind, significance, chunk_id, source_start,
                               source_end, observed_at, window_confidence, created_at, updated_at)
         VALUES (?1, ?2, ?3, 'fact', 'minor', ?4, 0, 2, ?5, 'high', ?5, ?5)",
        (
            format!("memory-{bank}-{content}"),
            bank,
            content,
            chunk,
            now,
        ),
    )
    .unwrap();
    conn.last_insert_rowid()
}

fn unit(axis: usize) -> Vec<f32> {
    let mut vector = vec![0.0; EMBEDDING_DIMENSIONS];
    vector[axis] = 1.0;
    vector
}

// Network filesystems (TIM-94, decision 4)

#[test]
fn nfs_smb_cifs_ceph_and_fuse_are_network_filesystems() {
    // Linux statfs magics from include/uapi/linux/magic.h and the NFS, CIFS
    // and Ceph headers.
    assert_eq!(classify(0x6969), FilesystemKind::Nfs);
    assert_eq!(classify(0x517B), FilesystemKind::Smb);
    assert_eq!(classify(0xFF53_4D42), FilesystemKind::Smb);
    assert_eq!(classify(0xFE53_4D42), FilesystemKind::Smb);
    assert_eq!(classify(0x00C3_6400), FilesystemKind::Ceph);
    assert_eq!(classify(0x6573_5546), FilesystemKind::Fuse);
    for kind in [
        FilesystemKind::Nfs,
        FilesystemKind::Smb,
        FilesystemKind::Ceph,
        FilesystemKind::Fuse,
    ] {
        assert!(kind.is_network(), "{kind:?}");
    }
}

#[test]
fn local_filesystems_are_not_refused() {
    // ext4, btrfs, xfs, tmpfs, overlayfs and zfs: the block and local
    // volumes a sidecar runs on.
    for magic in [
        0xEF53,
        0x9123_683E,
        0x5846_5342,
        0x0102_1994,
        0x794C_7630,
        0x2FC1_2FC1,
    ] {
        assert_eq!(classify(magic), FilesystemKind::Local, "{magic:#x}");
    }
    assert!(!FilesystemKind::Local.is_network());
}

#[test]
fn bsd_type_names_classify_the_same_way() {
    assert_eq!(classify_name("nfs"), FilesystemKind::Nfs);
    assert_eq!(classify_name("smbfs"), FilesystemKind::Smb);
    assert_eq!(classify_name("cifs"), FilesystemKind::Smb);
    assert_eq!(classify_name("ceph"), FilesystemKind::Ceph);
    assert_eq!(classify_name("macfuse"), FilesystemKind::Fuse);
    assert_eq!(classify_name("osxfuse"), FilesystemKind::Fuse);
    for local in ["apfs", "hfs", "ufs", "zfs", "ext4"] {
        assert_eq!(classify_name(local), FilesystemKind::Local, "{local}");
    }
}

#[test]
fn a_local_data_dir_never_needs_the_flag() {
    let dir = TestDir::new();
    assert_eq!(
        fs::check_data_dir(&dir.0, false).unwrap(),
        FilesystemKind::Local
    );
    assert_eq!(
        fs::check_data_dir(&dir.0, true).unwrap(),
        FilesystemKind::Local
    );
    assert_eq!(fs::filesystem_kind(&dir.0).unwrap(), FilesystemKind::Local);
}

#[test]
fn filesystem_policy_refuses_each_network_kind_unless_overridden() {
    let dir = Path::new("/var/lib/asphodel");
    for kind in [
        FilesystemKind::Nfs,
        FilesystemKind::Smb,
        FilesystemKind::Ceph,
        FilesystemKind::Fuse,
    ] {
        match fs::apply_policy(dir, kind, false) {
            Err(StoreError::NetworkFilesystem {
                dir: refused,
                kind: found,
            }) => {
                assert_eq!(refused, dir);
                assert_eq!(found, kind);
            }
            other => panic!("wrong policy result for {kind:?}: {other:?}"),
        }
        let message = fs::apply_policy(dir, kind, false).unwrap_err().to_string();
        assert!(message.contains(dir.to_str().unwrap()), "{message}");
        assert!(message.contains(&kind.to_string()), "{message}");
        assert!(message.contains("--allow-network-fs"), "{message}");
        assert_eq!(fs::apply_policy(dir, kind, true).unwrap(), kind);
    }
    for allow in [false, true] {
        assert_eq!(
            fs::apply_policy(dir, FilesystemKind::Local, allow).unwrap(),
            FilesystemKind::Local
        );
    }
}

// The data-dir lock (TIM-94, decision 4)

#[test]
fn two_stores_cannot_share_a_data_dir() {
    let dir = TestDir::new();
    let first = open(&dir.data(), clock());
    // Repeated refusals never release the holder's lock.
    for _ in 0..3 {
        let error = Store::open(&dir.data(), OpenOptions::default(), clock()).unwrap_err();
        let message = error.to_string();
        assert!(message.contains(dir.data().to_str().unwrap()), "{message}");
        assert!(message.to_lowercase().contains("lock"), "{message}");
        match error {
            StoreError::Locked {
                dir: locked,
                holder,
            } => {
                assert_eq!(locked, dir.data());
                assert_eq!(holder, std::process::id().to_string());
            }
            other => panic!("a second store opened a locked data dir: {other:?}"),
        }
    }
    // The refusals left the first store untouched.
    assert_eq!(first.schema_version().unwrap(), SCHEMA_VERSION);
}

#[test]
fn dropping_a_store_releases_its_data_dir() {
    let dir = TestDir::new();
    let first = open(&dir.data(), clock());
    drop(first);
    let second = open(&dir.data(), clock());
    assert_eq!(second.schema_version().unwrap(), SCHEMA_VERSION);
}

#[test]
fn the_lock_alone_can_be_taken_and_released() {
    let dir = TestDir::new();
    let lock = DataDirLock::acquire(&dir.0).unwrap();
    assert_eq!(lock.path(), dir.0.join(LOCK_FILE));
    assert!(matches!(
        DataDirLock::acquire(&dir.0),
        Err(StoreError::Locked { .. })
    ));
    drop(lock);
    DataDirLock::acquire(&dir.0).unwrap();
}

// Opening and migrating from empty (ADR 0010)

#[test]
fn a_missing_data_dir_is_created_and_a_file_is_refused() {
    let dir = TestDir::new();
    let missing = dir.0.join("new").join("deeper");
    let store = open(&missing, clock());
    assert!(missing.is_dir());
    assert!(missing.join(DB_FILE).is_file());
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
fn an_empty_data_dir_migrates_to_the_current_version() {
    let dir = TestDir::new();
    let store = open(&dir.data(), clock());
    let applied = store.applied().expect("a fresh store applies migrations");
    assert_eq!(applied.from, 0);
    assert_eq!(applied.to, SCHEMA_VERSION);
    assert_eq!(
        applied.copy, None,
        "nothing to copy before the first schema"
    );
    assert_eq!(store.schema_version().unwrap(), SCHEMA_VERSION);

    let conn = store.connection();
    assert_eq!(migrations::version(&conn).unwrap(), SCHEMA_VERSION);
    let (from, to, started_at, completed_at): (i64, i64, i64, i64) = conn
        .query_row(
            "SELECT from_version, to_version, started_at, completed_at FROM migrations",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .unwrap();
    assert_eq!((from, to), (0, i64::from(SCHEMA_VERSION)));
    // The clock never moved, so both times are its one instant.
    assert_eq!(started_at, micros(start()));
    assert_eq!(completed_at, micros(start()));
}

#[test]
fn reopening_a_current_store_applies_nothing() {
    let dir = TestDir::new();
    drop(open(&dir.data(), clock()));
    let store = open(&dir.data(), clock());
    assert_eq!(store.applied(), None);
    assert_eq!(
        count(&store.connection(), "SELECT count(*) FROM migrations"),
        1
    );
    assert!(!migrations::copy_path(&dir.data(), 0).exists());
}

#[test]
fn a_store_newer_than_the_binary_is_refused() {
    // ADR 0010: restore checks the schema version isn't newer than the
    // binary, and opening does the same.
    let dir = TestDir::new();
    let store = open(&dir.data(), clock());
    store
        .connection()
        .pragma_update(None, "user_version", SCHEMA_VERSION + 1)
        .unwrap();
    drop(store);
    match Store::open(&dir.data(), OpenOptions::default(), clock()) {
        Err(StoreError::NewerSchema {
            found, supported, ..
        }) => {
            assert_eq!(found, SCHEMA_VERSION + 1);
            assert_eq!(supported, SCHEMA_VERSION);
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn the_database_runs_in_wal_mode_with_foreign_keys_on() {
    let dir = TestDir::new();
    let store = open(&dir.data(), clock());
    let conn = store.connection();
    let journal: String = conn
        .query_row("PRAGMA journal_mode", [], |row| row.get(0))
        .unwrap();
    assert_eq!(journal, "wal");
    let foreign_keys: i64 = conn
        .query_row("PRAGMA foreign_keys", [], |row| row.get(0))
        .unwrap();
    assert_eq!(foreign_keys, 1);
    assert!(dir.data().join(DB_FILE).is_file());
}

#[test]
fn every_id_column_is_an_autoincrement_rowid() {
    // TIM-90: rowids are never reused, because sqlite-vec keys vectors by
    // them.
    let dir = TestDir::new();
    let store = open(&dir.data(), clock());
    let conn = store.connection();
    let mut statement = conn
        .prepare(
            "SELECT name, sql FROM sqlite_master
             WHERE type = 'table' AND sql IS NOT NULL AND name NOT LIKE 'sqlite_%'",
        )
        .unwrap();
    let tables: Vec<(String, String)> = statement
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
        .unwrap()
        .map(Result::unwrap)
        .collect();
    // FTS5 and vec0 own their shadow tables; only the tables the schema
    // declares are checked.
    let virtual_tables: Vec<String> = tables
        .iter()
        .filter(|(_, sql)| sql.to_uppercase().starts_with("CREATE VIRTUAL TABLE"))
        .map(|(name, _)| name.clone())
        .collect();
    let mut checked = 0;
    for (name, sql) in tables {
        let sql = sql.to_uppercase();
        if sql.starts_with("CREATE VIRTUAL TABLE")
            || virtual_tables
                .iter()
                .any(|v| name.starts_with(&format!("{v}_")))
        {
            continue;
        }
        if sql.contains("INTEGER PRIMARY KEY") {
            assert!(
                sql.contains("INTEGER PRIMARY KEY AUTOINCREMENT"),
                "{name} has a reusable rowid"
            );
            checked += 1;
        }
    }
    assert!(checked >= 10, "only {checked} rowid tables found");
    assert!(
        conn.query_row(
            "SELECT count(*) FROM sqlite_master WHERE name = 'sqlite_sequence'",
            [],
            |row| row.get::<_, i64>(0)
        )
        .unwrap()
            == 1
    );
}

#[test]
fn no_time_column_defaults_to_sqlites_clock() {
    // TIM-90: the schema never takes a timestamp from the database clock.
    let dir = TestDir::new();
    let store = open(&dir.data(), clock());
    let conn = store.connection();
    let mut names = conn
        .prepare("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'")
        .unwrap();
    let mut time_columns = 0;
    for table in names
        .query_map([], |row| row.get::<_, String>(0))
        .unwrap()
        .map(Result::unwrap)
    {
        let mut columns = conn
            .prepare(&format!("PRAGMA table_info(\"{table}\")"))
            .unwrap();
        for column in columns
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, Option<String>>(4)?,
                ))
            })
            .unwrap()
            .map(Result::unwrap)
        {
            let (name, declared, default) = column;
            if name == "at" || name.ends_with("_at") {
                time_columns += 1;
                assert_eq!(declared, "INTEGER", "{table}.{name} is {declared}");
                assert_eq!(default, None, "{table}.{name} has a default");
            }
        }
    }
    assert!(time_columns >= 30, "only {time_columns} time columns found");
}

#[test]
fn rows_the_store_writes_carry_the_clocks_time() {
    let dir = TestDir::new();
    let clock = clock();
    let store = open(&dir.data(), clock.clone());
    let service = Service::open(clock.clone(), store, Tuning::default());
    service.ensure_bank("main", &identity(), &models()).unwrap();
    let conn = service.store().unwrap().connection();
    let expected = micros(start());
    for sql in [
        "SELECT created_at FROM banks",
        "SELECT updated_at FROM banks",
        "SELECT min(created_at) FROM entities",
        "SELECT max(updated_at) FROM entities",
        "SELECT min(created_at) FROM entity_aliases",
        "SELECT min(at) FROM edits",
        "SELECT max(at) FROM edits",
        "SELECT created_at FROM mental_models",
        "SELECT started_at FROM migrations",
    ] {
        let at: i64 = conn.query_row(sql, [], |row| row.get(0)).unwrap();
        assert_eq!(at, expected, "{sql}");
    }
}

// The schema's own rules

#[test]
fn a_memorys_content_and_kind_never_change() {
    // TIM-90: a different claim is a new memory; everything else is
    // metadata that can be edited in place.
    let dir = TestDir::new();
    let clock = clock();
    let store = open(&dir.data(), clock.clone());
    let service = Service::open(clock.clone(), store, Tuning::default());
    service.ensure_bank("main", &identity(), &models()).unwrap();
    let conn = service.store().unwrap().connection();
    let now = micros(clock.now());
    let bank = bank_rowid(&conn, "main");
    let chunk = seed_chunk(&conn, bank, "a", now);
    let memory = insert_memory(&conn, bank, chunk, "User drinks coffee.", now);

    assert!(
        conn.execute(
            "UPDATE memories SET content = 'User drinks tea.' WHERE id = ?1",
            [memory]
        )
        .is_err()
    );
    assert!(
        conn.execute("UPDATE memories SET kind = 'state' WHERE id = ?1", [memory])
            .is_err()
    );
    conn.execute(
        "UPDATE memories SET owner_significance = 'kept', updated_at = ?2 WHERE id = ?1",
        (memory, now),
    )
    .unwrap();
    conn.execute(
        "UPDATE memories SET valid_until = ?2, valid_until_precision = 'day' WHERE id = ?1",
        (memory, now),
    )
    .unwrap();
}

#[test]
fn memory_content_is_searchable_with_fts5() {
    let dir = TestDir::new();
    let clock = clock();
    let store = open(&dir.data(), clock.clone());
    let service = Service::open(clock.clone(), store, Tuning::default());
    service.ensure_bank("main", &identity(), &models()).unwrap();
    let conn = service.store().unwrap().connection();
    let now = micros(clock.now());
    let bank = bank_rowid(&conn, "main");
    let chunk = seed_chunk(&conn, bank, "a", now);
    let coffee = insert_memory(&conn, bank, chunk, "User drinks coffee every morning.", now);
    let lisbon = insert_memory(&conn, bank, chunk, "User lives in Lisbon.", now);

    let hits = |query: &str| -> Vec<i64> {
        let mut statement = conn
            .prepare("SELECT rowid FROM memories_fts WHERE memories_fts MATCH ?1 ORDER BY rank")
            .unwrap();
        statement
            .query_map([query], |row| row.get(0))
            .unwrap()
            .map(Result::unwrap)
            .collect()
    };
    assert_eq!(hits("coffee"), [coffee]);
    assert_eq!(hits("lisbon"), [lisbon]);
    assert!(hits("berlin").is_empty());

    conn.execute("DELETE FROM memories WHERE id = ?1", [coffee])
        .unwrap();
    assert!(hits("coffee").is_empty(), "a deleted memory still matches");
}

#[test]
fn entity_aliases_are_searchable_with_fts5() {
    // TIM-90: aliases get their own table so full-text search can match
    // them; TIM-94: the owner's platform ids are aliases of `user`.
    let dir = TestDir::new();
    let clock = clock();
    let store = open(&dir.data(), clock.clone());
    let service = Service::open(clock.clone(), store, Tuning::default());
    service.ensure_bank("main", &identity(), &models()).unwrap();
    let conn = service.store().unwrap().connection();
    let user: i64 = conn
        .query_row("SELECT id FROM entities WHERE seeded = 'user'", [], |row| {
            row.get(0)
        })
        .unwrap();
    let entity_for = |query: &str| -> Vec<i64> {
        let mut statement = conn
            .prepare(
                "SELECT entity_id FROM entity_aliases
                 WHERE id IN (SELECT rowid FROM entity_aliases_fts WHERE entity_aliases_fts MATCH ?1)",
            )
            .unwrap();
        statement
            .query_map([query], |row| row.get(0))
            .unwrap()
            .map(Result::unwrap)
            .collect()
    };
    assert_eq!(entity_for("tim"), [user]);
    assert_eq!(entity_for("\"discord:1234\""), [user]);
    assert!(entity_for("sam").is_empty());
}

#[test]
fn source_keys_make_ingest_idempotent() {
    // ADR 0002 and TIM-90: a conflicting ingest does nothing. The key is
    // bank, session, message time and content hash for a turn, and bank,
    // document id and content hash for a document.
    let dir = TestDir::new();
    let clock = clock();
    let store = open(&dir.data(), clock.clone());
    let service = Service::open(clock.clone(), store, Tuning::default());
    service.ensure_bank("main", &identity(), &models()).unwrap();
    service
        .ensure_bank("other", &identity(), &models())
        .unwrap();
    let conn = service.store().unwrap().connection();
    let now = micros(clock.now());
    let main = bank_rowid(&conn, "main");
    let other = bank_rowid(&conn, "other");

    let turn = |uuid: &str, bank: i64, hash: &str| {
        conn.execute(
            "INSERT INTO sources (uuid, bank_id, kind, session_id, message_at, content_hash,
                                  observed_at, timezone, text, ingested_at)
             VALUES (?1, ?2, 'turn', 'session', ?3, ?4, ?3, 'UTC', 'hi', ?3)",
            (uuid, bank, now, hash),
        )
    };
    turn("t1", main, "hash-a").unwrap();
    assert!(turn("t2", main, "hash-a").is_err(), "the same turn twice");
    turn("t3", main, "hash-b").unwrap();
    turn("t4", other, "hash-a").unwrap();

    let document = |uuid: &str, bank: i64, id: &str, hash: &str| {
        conn.execute(
            "INSERT INTO sources (uuid, bank_id, kind, document_id, content_hash, observed_at,
                                  reference_date, timezone, text, ingested_at)
             VALUES (?1, ?2, 'document', ?3, ?4, ?5, '2026-10-01', 'UTC', 'doc', ?5)",
            (uuid, bank, id, hash, now),
        )
    };
    document("d1", main, "notes.md", "hash-a").unwrap();
    assert!(document("d2", main, "notes.md", "hash-a").is_err());
    // Re-ingesting a document id with new content is allowed (TIM-94).
    document("d3", main, "notes.md", "hash-c").unwrap();
    document("d4", other, "notes.md", "hash-a").unwrap();
}

#[test]
fn at_most_one_access_per_memory_per_turn() {
    // TIM-90: `used` and `mentioned_again` in the same turn don't count
    // twice.
    let dir = TestDir::new();
    let clock = clock();
    let store = open(&dir.data(), clock.clone());
    let service = Service::open(clock.clone(), store, Tuning::default());
    service.ensure_bank("main", &identity(), &models()).unwrap();
    let conn = service.store().unwrap().connection();
    let now = micros(clock.now());
    let bank = bank_rowid(&conn, "main");
    let chunk = seed_chunk(&conn, bank, "a", now);
    let memory = insert_memory(&conn, bank, chunk, "User drinks coffee.", now);
    let access = |kind: &str, turn: i64| {
        conn.execute(
            "INSERT INTO accesses (bank_id, memory_id, kind, at, turn) VALUES (?1, ?2, ?3, ?4, ?5)",
            (bank, memory, kind, now, turn),
        )
    };
    access("created", 1).unwrap();
    assert!(access("used", 1).is_err());
    access("used", 2).unwrap();
    assert!(
        access("retrieved", 3).is_err(),
        "retrieval is not an access"
    );
}

// The pre-migration copy (ADR 0010)

#[test]
fn the_copy_is_keyed_by_the_from_version_and_never_overwritten() {
    let dir = TestDir::new();
    let clock = clock();
    let store = open(&dir.data(), clock.clone());
    let conn = store.connection();
    let path = migrations::take_copy(&conn, &dir.data(), 1).unwrap();
    assert_eq!(path, migrations::copy_path(&dir.data(), 1));
    assert!(path.file_name().unwrap().to_str().unwrap().contains("1"));
    let clean = std::fs::read(&path).unwrap();

    // The live database changes, as a crash-looping migration would change
    // it, and the copy stays the clean one.
    conn.execute(
        "INSERT INTO store_meta (key, value, updated_at) VALUES ('damage', 'x', ?1)",
        [micros(clock.now())],
    )
    .unwrap();
    let again = migrations::take_copy(&conn, &dir.data(), 1).unwrap();
    assert_eq!(again, path);
    assert_eq!(std::fs::read(&path).unwrap(), clean);
    assert!(!dir.data().join(format!("{DB_FILE}.v1.partial")).exists());
}

#[test]
fn the_copy_is_a_whole_database_at_the_old_version() {
    let dir = TestDir::new();
    let clock = clock();
    let store = open(&dir.data(), clock.clone());
    let service = Service::open(clock.clone(), store, Tuning::default());
    service.ensure_bank("main", &identity(), &models()).unwrap();
    let path = {
        let conn = service.store().unwrap().connection();
        migrations::take_copy(&conn, &dir.data(), SCHEMA_VERSION).unwrap()
    };
    let copy =
        Connection::open_with_flags(&path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
    let integrity: String = copy
        .query_row("PRAGMA integrity_check", [], |row| row.get(0))
        .unwrap();
    assert_eq!(integrity, "ok");
    assert_eq!(migrations::version(&copy).unwrap(), SCHEMA_VERSION);
    assert_eq!(count(&copy, "SELECT count(*) FROM banks"), 1);
}

#[test]
fn reopening_after_seven_days_deletes_the_copy() {
    let dir = TestDir::new();
    let clock = clock();
    let copy = {
        let store = open(&dir.data(), clock.clone());
        let conn = store.connection();
        let copy = migrations::take_copy(&conn, &dir.data(), 1).unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO migrations (from_version, to_version, binary_version, started_at, completed_at)
             VALUES (1, 2, 'test', ?1, ?1)",
            [micros(clock.now())],
        )
        .unwrap();
        copy
    };
    drop(open(&dir.data(), clock.clone()));
    assert!(copy.exists(), "deleted before the week was up");

    clock.advance(SignedDuration::from_hours(7 * 24));
    drop(open(&dir.data(), clock.clone()));
    assert!(!copy.exists(), "kept after the week was up");
}

// The deletion fingerprint (ADR 0009)

fn check_housekeeping_expiry(paused: bool) {
    let dir = TestDir::new();
    let clock = clock();
    let store = open(&dir.data(), clock.clone());
    let original = Tuning::default().deletion_fingerprint();
    assert_eq!(
        store.check_fingerprint(&original).unwrap(),
        PurgePause::Running
    );
    let mut tuning = Tuning::default();
    if paused {
        tuning.clock.quiet_rate = 0.2;
    }
    let expected_pause = if paused {
        PurgePause::Paused { stored: original }
    } else {
        PurgePause::Running
    };
    assert_eq!(
        store
            .check_fingerprint(&tuning.deletion_fingerprint())
            .unwrap(),
        expected_pause
    );

    // A later completion, not the start of the migration, sets the deadline.
    clock.advance(SignedDuration::from_hours(1));
    let (copy, incomplete) = {
        let conn = store.connection();
        let copy = migrations::take_copy(&conn, &dir.data(), 1).unwrap();
        let incomplete = migrations::take_copy(&conn, &dir.data(), 3).unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO migrations (from_version, to_version, binary_version, started_at, completed_at)
             VALUES (1, 2, 'test', ?1, ?2)",
            (micros(start()), micros(clock.now())),
        ).unwrap();
        (copy, incomplete)
    };
    let service = Service::open(clock.clone(), store, tuning);
    let deadline = clock
        .now()
        .checked_add(migrations::PRE_MIGRATION_COPY_TTL)
        .unwrap();
    let upkeep = service.housekeeping().unwrap();
    assert!(upkeep.copies_removed.is_empty());
    assert_eq!(upkeep.next_due, Some(deadline));
    clock.advance(SignedDuration::from_hours(7 * 24) - SignedDuration::from_micros(1));
    let upkeep = service.housekeeping().unwrap();
    assert!(upkeep.copies_removed.is_empty());
    assert_eq!(upkeep.next_due, Some(deadline));
    assert!(copy.exists(), "expired before completion plus seven days");
    clock.advance(SignedDuration::from_micros(1));
    let upkeep = service.housekeeping().unwrap();
    assert_eq!(upkeep.copies_removed, std::slice::from_ref(&copy));
    assert_eq!(upkeep.next_due, None);
    assert!(
        !copy.exists(),
        "housekeeping needed a restart to expire the copy"
    );
    assert!(
        incomplete.exists(),
        "removed a copy without a completed migration"
    );
    let upkeep = service.housekeeping().unwrap();
    assert!(upkeep.copies_removed.is_empty());
    assert_eq!(
        upkeep.next_due, None,
        "an incomplete migration has no deadline"
    );
    assert_eq!(
        service
            .store()
            .unwrap()
            .check_fingerprint(&service.tuning().deletion_fingerprint())
            .unwrap(),
        expected_pause
    );
    assert!(service.health().ready);
}

#[test]
fn housekeeping_expires_copies_on_the_simulated_clock_without_restarting() {
    check_housekeeping_expiry(false);
}

#[test]
fn housekeeping_expires_copies_even_while_purge_is_paused() {
    check_housekeeping_expiry(true);
}

#[test]
fn housekeeping_has_no_deadline_without_a_copy() {
    let dir = TestDir::new();
    let service = service(&dir);
    let upkeep = service.housekeeping().unwrap();
    assert!(upkeep.copies_removed.is_empty());
    assert_eq!(upkeep.next_due, None);
    assert_eq!(service.store().unwrap().next_copy_expiry().unwrap(), None);
}

// Only the initial migration exists today. These rows model later completed
// migrations without changing the live schema, just as the expiry tests do.
fn completed_copy(store: &Store, from: u32, completed: Timestamp) -> PathBuf {
    let conn = store.connection();
    let copy = migrations::take_copy(&conn, store.dir(), from).unwrap();
    conn.execute(
        "INSERT OR REPLACE INTO migrations (from_version, to_version, binary_version, started_at, completed_at)
         VALUES (?1, ?2, 'test', ?3, ?3)",
        (from, from + 1, micros(completed)),
    )
    .unwrap();
    copy
}

#[test]
fn housekeeping_ignores_a_missing_copy_even_after_its_deadline() {
    let dir = TestDir::new();
    let clock = clock();
    let store = open(&dir.data(), clock.clone());
    let copy = completed_copy(&store, 1, clock.now());
    let deadline = clock
        .now()
        .checked_add(migrations::PRE_MIGRATION_COPY_TTL)
        .unwrap();
    let service = Service::open(clock.clone(), store, Tuning::default());
    assert_eq!(service.housekeeping().unwrap().next_due, Some(deadline));
    std::fs::remove_file(copy).unwrap();
    for advance in [SignedDuration::ZERO, migrations::PRE_MIGRATION_COPY_TTL] {
        clock.advance(advance);
        let upkeep = service.housekeeping().unwrap();
        assert!(upkeep.copies_removed.is_empty());
        assert_eq!(upkeep.next_due, None);
        assert_eq!(service.store().unwrap().next_copy_expiry().unwrap(), None);
    }
}

#[test]
fn housekeeping_selects_the_earliest_existing_copy_deadline() {
    let dir = TestDir::new();
    let clock = clock();
    let store = open(&dir.data(), clock.clone());
    // Reverse completion order relative to version order to catch selecting
    // the first row rather than the minimum deadline.
    let later_completion = start().checked_add(SignedDuration::from_hours(2)).unwrap();
    let later = completed_copy(&store, 1, later_completion);
    let earlier = completed_copy(&store, 2, start());
    let missing = completed_copy(
        &store,
        3,
        start().checked_sub(SignedDuration::from_hours(1)).unwrap(),
    );
    std::fs::remove_file(missing).unwrap();
    let earlier_deadline = start()
        .checked_add(migrations::PRE_MIGRATION_COPY_TTL)
        .unwrap();
    let later_deadline = later_completion
        .checked_add(migrations::PRE_MIGRATION_COPY_TTL)
        .unwrap();
    let service = Service::open(clock.clone(), store, Tuning::default());

    let upkeep = service.housekeeping().unwrap();
    assert!(upkeep.copies_removed.is_empty());
    assert_eq!(upkeep.next_due, Some(earlier_deadline));
    assert_eq!(
        service.store().unwrap().next_copy_expiry().unwrap(),
        Some(earlier_deadline)
    );
    clock.advance(migrations::PRE_MIGRATION_COPY_TTL);
    let upkeep = service.housekeeping().unwrap();
    assert_eq!(upkeep.copies_removed, std::slice::from_ref(&earlier));
    assert!(!earlier.exists());
    assert!(later.exists());
    assert_eq!(upkeep.next_due, Some(later_deadline));
    clock.advance(SignedDuration::from_hours(2));
    let upkeep = service.housekeeping().unwrap();
    assert_eq!(upkeep.copies_removed, std::slice::from_ref(&later));
    assert!(!later.exists());
    assert_eq!(upkeep.next_due, None);
}

#[test]
fn a_matching_fingerprint_keeps_purge_running_across_reopens() {
    let dir = TestDir::new();
    let current = Tuning::default().deletion_fingerprint();
    let store = open(&dir.data(), clock());
    store.check_fingerprint(&current).unwrap();
    drop(store);
    let store = open(&dir.data(), clock());
    assert_eq!(
        store.check_fingerprint(&current).unwrap(),
        PurgePause::Running
    );
    assert_eq!(
        store.check_fingerprint(&current).unwrap(),
        PurgePause::Running
    );
}

#[test]
fn a_changed_fingerprint_pauses_purge_and_keeps_the_stored_one() {
    let dir = TestDir::new();
    let original = Tuning::default().deletion_fingerprint();
    let mut changed = Tuning::default();
    changed.clock.quiet_rate = 0.2;
    let changed = changed.deletion_fingerprint();
    assert_ne!(original, changed);

    let store = open(&dir.data(), clock());
    store.check_fingerprint(&original).unwrap();
    drop(store);
    let store = open(&dir.data(), clock());
    assert_eq!(
        store.check_fingerprint(&changed).unwrap(),
        PurgePause::Paused {
            stored: original.clone()
        }
    );
    // Only an acknowledgement moves the stored value (ADR 0010).
    assert_eq!(
        store.check_fingerprint(&changed).unwrap(),
        PurgePause::Paused { stored: original }
    );
}

#[test]
fn an_unrelated_tuning_change_does_not_pause_purge() {
    let dir = TestDir::new();
    let store = open(&dir.data(), clock());
    store
        .check_fingerprint(&Tuning::default().deletion_fingerprint())
        .unwrap();
    let mut tuning = Tuning::default();
    tuning.injection.cap = 3;
    tuning.recall.strong_cutoff = 0.5;
    assert_eq!(
        store
            .check_fingerprint(&tuning.deletion_fingerprint())
            .unwrap(),
        PurgePause::Running
    );
}

// Public ids (TIM-90)

#[test]
fn public_ids_are_uuidv7_timed_by_the_clock() {
    let dir = TestDir::new();
    let clock = clock();
    let store = open(&dir.data(), clock.clone());
    let id = store.new_id();
    assert_eq!(id.get_version_num(), 7);
    let (seconds, _) = id.get_timestamp().unwrap().to_unix();
    assert_eq!(i64::try_from(seconds).unwrap(), start().as_second());

    clock.advance(SignedDuration::from_secs(3600));
    let later = store.new_id();
    let (seconds, _) = later.get_timestamp().unwrap().to_unix();
    assert_eq!(i64::try_from(seconds).unwrap(), start().as_second() + 3600);
    assert!(later > id, "ids sort by the clock");
}

#[test]
fn ids_minted_in_the_same_instant_are_distinct_and_ordered() {
    let dir = TestDir::new();
    let store = open(&dir.data(), clock());
    let ids: Vec<_> = (0..1000).map(|_| store.new_id()).collect();
    let mut sorted = ids.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), ids.len(), "duplicate ids");
    assert_eq!(sorted, ids, "ids minted in order don't sort in order");
}

#[test]
fn every_row_the_store_creates_has_a_uuidv7() {
    let dir = TestDir::new();
    let clock = clock();
    let store = open(&dir.data(), clock.clone());
    let service = Service::open(clock.clone(), store, Tuning::default());
    let bank = service.ensure_bank("main", &identity(), &models()).unwrap();
    assert_eq!(bank.id.get_version_num(), 7);
    let conn = service.store().unwrap().connection();
    for table in ["banks", "entities", "edits", "mental_models"] {
        let mut statement = conn.prepare(&format!("SELECT uuid FROM {table}")).unwrap();
        let uuids: Vec<String> = statement
            .query_map([], |row| row.get(0))
            .unwrap()
            .map(Result::unwrap)
            .collect();
        assert!(!uuids.is_empty(), "{table} is empty");
        for uuid in uuids {
            let parsed: uuid::Uuid = uuid.parse().unwrap_or_else(|_| panic!("{table}: {uuid}"));
            assert_eq!(parsed.get_version_num(), 7, "{table}: {uuid}");
        }
    }
}

// Vector search (TIM-89)

#[test]
fn the_index_takes_384_dimensions_and_rejects_others() {
    let dir = TestDir::new();
    let store = open(&dir.data(), clock());
    let vectors = store.vectors();
    assert_eq!(vectors.dimensions(), 384);
    let conn = store.connection();
    for wrong in [0, 3, 383, 385, 768] {
        let vector = vec![0.5; wrong];
        assert!(
            matches!(
                vectors.upsert(&conn, 1, 1, &vector),
                Err(VectorError::Dimensions { expected: 384, got }) if got == wrong
            ),
            "{wrong} dimensions accepted"
        );
        assert!(vectors.nearest(&conn, 1, &vector, 5).is_err());
    }
}

#[test]
fn nearest_returns_the_closest_memory_first_with_cosine_distance() {
    let dir = TestDir::new();
    let store = open(&dir.data(), clock());
    let vectors = store.vectors();
    let conn = store.connection();
    let mut leaning = unit(0);
    leaning[1] = 0.5;
    vectors.upsert(&conn, 1, 10, &unit(0)).unwrap();
    vectors.upsert(&conn, 1, 11, &leaning).unwrap();
    vectors.upsert(&conn, 1, 12, &unit(1)).unwrap();

    let near = vectors.nearest(&conn, 1, &unit(0), 3).unwrap();
    let ids: Vec<i64> = near.iter().map(|n| n.memory_id).collect();
    assert_eq!(ids, [10, 11, 12]);
    assert!(near[0].distance.abs() < 1e-6, "identical vectors: {near:?}");
    assert!(
        (near[2].distance - 1.0).abs() < 1e-6,
        "orthogonal vectors: {near:?}"
    );
    assert!(near[1].distance > near[0].distance && near[1].distance < near[2].distance);

    assert_eq!(vectors.nearest(&conn, 1, &unit(0), 1).unwrap().len(), 1);
    assert!(vectors.nearest(&conn, 1, &unit(0), 0).unwrap().is_empty());
    assert_eq!(vectors.nearest(&conn, 1, &unit(0), 50).unwrap().len(), 3);
}

#[test]
fn nearest_past_the_knn_limit_scans_exactly() {
    // sqlite-vec refuses a KNN query for more than KNN_K_MAX neighbours, so a
    // larger k is an exact scan (the TIM-108 re-review). It must give the
    // same neighbours in the same order and metric, within the bank.
    use asphodel_core::store::Neighbour;
    use asphodel_core::store::vector::KNN_K_MAX;
    let dir = TestDir::new();
    let store = open(&dir.data(), clock());
    let vectors = store.vectors();
    let conn = store.connection();
    for (step, id) in (0..6).zip(10..) {
        let mut leaning = unit(0);
        leaning[1] = step as f32 * 0.4;
        vectors.upsert(&conn, 1, id, &leaning).unwrap();
    }
    vectors.upsert(&conn, 1, 20, &unit(1)).unwrap();
    vectors.upsert(&conn, 2, 30, &unit(0)).unwrap();

    let knn = vectors.nearest(&conn, 1, &unit(0), 7).unwrap();
    let exact = vectors.nearest(&conn, 1, &unit(0), KNN_K_MAX + 1).unwrap();
    let ids = |found: &[Neighbour]| -> Vec<i64> { found.iter().map(|n| n.memory_id).collect() };
    assert_eq!(ids(&knn), [10, 11, 12, 13, 14, 15, 20]);
    assert_eq!(ids(&exact), ids(&knn), "another bank's vector stays out");
    for (k, e) in knn.iter().zip(&exact) {
        assert!(
            (k.distance - e.distance).abs() < 1e-5,
            "{k:?} against {e:?}"
        );
    }
    assert!(
        vectors
            .nearest(&conn, 2, &unit(0), KNN_K_MAX + 1)
            .unwrap()
            .len()
            == 1
    );
    assert!(
        vectors
            .nearest(&conn, 3, &unit(0), KNN_K_MAX + 1)
            .unwrap()
            .is_empty()
    );
    assert!(matches!(
        vectors.nearest(&conn, 1, &[0.5; 3], KNN_K_MAX + 1),
        Err(VectorError::Dimensions {
            expected: 384,
            got: 3
        })
    ));
}

#[test]
fn a_bank_never_sees_another_banks_vectors() {
    let dir = TestDir::new();
    let store = open(&dir.data(), clock());
    let vectors = store.vectors();
    let conn = store.connection();
    vectors.upsert(&conn, 1, 10, &unit(0)).unwrap();
    vectors.upsert(&conn, 2, 20, &unit(0)).unwrap();
    let ids = |bank: i64| -> Vec<i64> {
        vectors
            .nearest(&conn, bank, &unit(0), 10)
            .unwrap()
            .into_iter()
            .map(|n| n.memory_id)
            .collect()
    };
    assert_eq!(ids(1), [10]);
    assert_eq!(ids(2), [20]);
    assert!(ids(3).is_empty());
}

#[test]
fn upsert_replaces_and_remove_forgets() {
    let dir = TestDir::new();
    let store = open(&dir.data(), clock());
    let vectors = store.vectors();
    let conn = store.connection();
    vectors.upsert(&conn, 1, 10, &unit(0)).unwrap();
    vectors.upsert(&conn, 1, 10, &unit(1)).unwrap();
    let near = vectors.nearest(&conn, 1, &unit(1), 5).unwrap();
    assert_eq!(near.len(), 1);
    assert_eq!(near[0].memory_id, 10);
    assert!(near[0].distance.abs() < 1e-6, "the old vector survived");

    vectors.remove(&conn, 10).unwrap();
    assert!(vectors.nearest(&conn, 1, &unit(1), 5).unwrap().is_empty());
    vectors.remove(&conn, 10).unwrap();
    vectors.remove(&conn, 999).unwrap();
}

#[test]
fn a_vector_commits_or_rolls_back_with_its_memory() {
    // TIM-89: vec0 stores into shadow tables in the same file, so vectors
    // commit or roll back atomically with the enclosing transaction.
    let dir = TestDir::new();
    let clock = clock();
    let store = open(&dir.data(), clock.clone());
    let service = Service::open(clock.clone(), store, Tuning::default());
    service.ensure_bank("main", &identity(), &models()).unwrap();
    let store = service.store().unwrap();
    let vectors = store.vectors();
    let mut conn = store.connection();
    let now = micros(clock.now());
    let bank = bank_rowid(&conn, "main");
    let chunk = seed_chunk(&conn, bank, "a", now);

    let tx = conn.transaction().unwrap();
    let memory = insert_memory(&tx, bank, chunk, "User drinks coffee.", now);
    vectors.upsert(&tx, bank, memory, &unit(0)).unwrap();
    tx.rollback().unwrap();
    assert_eq!(count(&conn, "SELECT count(*) FROM memories"), 0);
    assert!(
        vectors
            .nearest(&conn, bank, &unit(0), 5)
            .unwrap()
            .is_empty()
    );

    let tx = conn.transaction().unwrap();
    let memory = insert_memory(&tx, bank, chunk, "User drinks coffee.", now);
    vectors.upsert(&tx, bank, memory, &unit(0)).unwrap();
    tx.commit().unwrap();
    let near = vectors.nearest(&conn, bank, &unit(0), 5).unwrap();
    assert_eq!(near.len(), 1);
    assert_eq!(near[0].memory_id, memory);
}

// Bank create-or-merge (TIM-94, decision 7)

fn service(dir: &TestDir) -> Service {
    let clock = clock();
    let store = open(&dir.data(), clock.clone());
    Service::open(clock, store, Tuning::default())
}

#[test]
fn creating_a_bank_seeds_user_assistant_and_the_profile() {
    let dir = TestDir::new();
    let service = service(&dir);
    let bank = service.ensure_bank("main", &identity(), &models()).unwrap();
    assert!(bank.created);
    assert_eq!(bank.name, "main");
    assert_eq!(bank.owner_name.as_deref(), Some("Tim"));
    assert_eq!(bank.assistant_name.as_deref(), Some("Hermes"));
    assert_eq!(bank.timezone, "Pacific/Auckland");
    assert_eq!(bank.embedding_model, models().embedding);
    assert_eq!(bank.reranker_model, models().reranker);

    let conn = service.store().unwrap().connection();
    let mut statement = conn
        .prepare("SELECT seeded, name, kind FROM entities ORDER BY seeded DESC")
        .unwrap();
    let entities: Vec<(String, String, String)> = statement
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
        .unwrap()
        .map(Result::unwrap)
        .collect();
    // TIM-92: the names are the first aliases; the seeded assistant has
    // type `thing`. TIM-94: the owner is a person.
    assert_eq!(
        entities,
        [
            ("user".to_owned(), "Tim".to_owned(), "person".to_owned()),
            (
                "assistant".to_owned(),
                "Hermes".to_owned(),
                "thing".to_owned()
            ),
        ]
    );
    let mut statement = conn
        .prepare(
            "SELECT e.seeded, a.alias FROM entity_aliases a JOIN entities e ON e.id = a.entity_id
             ORDER BY e.seeded DESC, a.alias",
        )
        .unwrap();
    let aliases: Vec<(String, String)> = statement
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
        .unwrap()
        .map(Result::unwrap)
        .collect();
    assert_eq!(
        aliases,
        [
            ("user".to_owned(), "Tim".to_owned()),
            ("user".to_owned(), "discord:1234".to_owned()),
            ("assistant".to_owned(), "Hermes".to_owned()),
        ]
    );

    // TIM-95, decision 2: only "User profile" is seeded, enabled, at the
    // tuned budget.
    let (name, max_tokens, enabled): (String, i64, i64) = conn
        .query_row(
            "SELECT name, max_tokens, enabled FROM mental_models",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .unwrap();
    assert_eq!(name, PROFILE_NAME);
    assert_eq!(
        max_tokens,
        i64::from(Tuning::default().mental_models.profile_max_tokens)
    );
    assert_eq!(enabled, 1);
}

#[test]
fn ensure_bank_is_idempotent() {
    let dir = TestDir::new();
    let service = service(&dir);
    let first = service.ensure_bank("main", &identity(), &models()).unwrap();
    let second = service.ensure_bank("main", &identity(), &models()).unwrap();
    let third = service
        .ensure_bank("main", &BankIdentity::default(), &models())
        .unwrap();
    assert!(first.created);
    assert!(!second.created);
    assert!(!third.created);
    assert_eq!(second.id, first.id);
    assert_eq!(third.id, first.id);
    assert_eq!(third.owner_name, first.owner_name);
    assert_eq!(third.assistant_name, first.assistant_name);
    assert_eq!(third.timezone, first.timezone);

    let conn = service.store().unwrap().connection();
    assert_eq!(count(&conn, "SELECT count(*) FROM banks"), 1);
    assert_eq!(count(&conn, "SELECT count(*) FROM entities"), 2);
    assert_eq!(count(&conn, "SELECT count(*) FROM entity_aliases"), 3);
    assert_eq!(count(&conn, "SELECT count(*) FROM mental_models"), 1);
}

#[test]
fn merge_sets_present_fields_and_leaves_absent_ones_alone() {
    let dir = TestDir::new();
    let service = service(&dir);
    service.ensure_bank("main", &identity(), &models()).unwrap();
    let bank = service
        .ensure_bank(
            "main",
            &BankIdentity {
                timezone: Some("Europe/Lisbon".into()),
                owner_platform_ids: vec!["telegram:9".into()],
                ..BankIdentity::default()
            },
            &models(),
        )
        .unwrap();
    assert_eq!(bank.timezone, "Europe/Lisbon");
    assert_eq!(bank.owner_name.as_deref(), Some("Tim"));
    assert_eq!(bank.assistant_name.as_deref(), Some("Hermes"));

    let conn = service.store().unwrap().connection();
    let user_aliases = count(
        &conn,
        "SELECT count(*) FROM entity_aliases a JOIN entities e ON e.id = a.entity_id
         WHERE e.seeded = 'user'",
    );
    assert_eq!(user_aliases, 3, "Tim, discord:1234 and telegram:9");
}

#[test]
fn renaming_adds_an_alias_and_removes_none() {
    let dir = TestDir::new();
    let service = service(&dir);
    service.ensure_bank("main", &identity(), &models()).unwrap();
    let bank = service
        .ensure_bank(
            "main",
            &BankIdentity {
                owner_name: Some("Timothy".into()),
                assistant_name: Some("Jarvis".into()),
                ..BankIdentity::default()
            },
            &models(),
        )
        .unwrap();
    assert_eq!(bank.owner_name.as_deref(), Some("Timothy"));
    assert_eq!(bank.assistant_name.as_deref(), Some("Jarvis"));

    let conn = service.store().unwrap().connection();
    let aliases = |seeded: &str| -> Vec<String> {
        let mut statement = conn
            .prepare(
                "SELECT a.alias FROM entity_aliases a JOIN entities e ON e.id = a.entity_id
                 WHERE e.seeded = ?1 ORDER BY a.alias",
            )
            .unwrap();
        statement
            .query_map([seeded], |row| row.get(0))
            .unwrap()
            .map(Result::unwrap)
            .collect()
    };
    assert_eq!(aliases("user"), ["Tim", "Timothy", "discord:1234"]);
    assert_eq!(aliases("assistant"), ["Hermes", "Jarvis"]);
    assert_eq!(count(&conn, "SELECT count(*) FROM entities"), 2);
}

#[test]
fn the_profile_is_seeded_on_create_only() {
    // TIM-95, decision 2: an owner who deletes it doesn't get it back on
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
    assert_eq!(
        count(
            &service.store().unwrap().connection(),
            "SELECT count(*) FROM mental_models"
        ),
        0
    );
}

#[test]
fn models_are_recorded_at_creation_and_merge_does_not_change_them() {
    // TIM-94, decision 4: a bank records its model ids. ADR 0010: changing
    // them is `asphodel reembed`, not a config merge.
    let dir = TestDir::new();
    let service = service(&dir);
    service.ensure_bank("main", &identity(), &models()).unwrap();
    let other = ModelIds {
        embedding: "granite-embedding-small-english-r2:int8".into(),
        reranker: models().reranker,
    };
    let bank = service
        .ensure_bank("main", &BankIdentity::default(), &other)
        .unwrap();
    assert_eq!(bank.embedding_model, models().embedding);
}

#[test]
fn banks_are_isolated_from_each_other() {
    let dir = TestDir::new();
    let service = service(&dir);
    let main = service.ensure_bank("main", &identity(), &models()).unwrap();
    let work = service
        .ensure_bank(
            "work",
            &BankIdentity {
                owner_name: Some("Tim".into()),
                ..BankIdentity::default()
            },
            &models(),
        )
        .unwrap();
    assert_ne!(main.id, work.id);
    let conn = service.store().unwrap().connection();
    assert_eq!(count(&conn, "SELECT count(*) FROM entities"), 4);
    assert_eq!(
        count(
            &conn,
            "SELECT count(*) FROM entities WHERE seeded = 'user' AND name = 'Tim'"
        ),
        2,
        "\"Tim\" in two banks is two entities"
    );
}

#[test]
fn an_unknown_timezone_or_empty_name_creates_nothing() {
    let dir = TestDir::new();
    let service = service(&dir);
    assert!(matches!(
        service.ensure_bank(
            "main",
            &BankIdentity {
                timezone: Some("Mars/Olympus_Mons".into()),
                ..BankIdentity::default()
            },
            &models()
        ),
        Err(BankError::InvalidTimezone)
    ));
    assert!(matches!(
        service.ensure_bank("  ", &identity(), &models()),
        Err(BankError::EmptyName)
    ));
    let conn = service.store().unwrap().connection();
    assert_eq!(count(&conn, "SELECT count(*) FROM banks"), 0);
    assert_eq!(count(&conn, "SELECT count(*) FROM entities"), 0);
    assert_eq!(count(&conn, "SELECT count(*) FROM edits"), 0);
}

#[test]
fn a_bank_survives_a_reopen() {
    let dir = TestDir::new();
    let created = {
        let service = service(&dir);
        service.ensure_bank("main", &identity(), &models()).unwrap()
    };
    let service = service(&dir);
    let reopened = service
        .ensure_bank("main", &BankIdentity::default(), &models())
        .unwrap();
    assert!(!reopened.created);
    assert_eq!(reopened.id, created.id);
    assert_eq!(reopened.owner_name, created.owner_name);
}

// Review regressions (TIM-103 review through c157b84)

#[test]
fn the_seeded_profile_takes_facts_and_slow_states() {
    // TIM-95, decision 3: the profile takes facts, plus states with
    // volatility of weeks or slower, and null volatility passes. Decision 2
    // makes those the model's own filters, not words in its question.
    let dir = TestDir::new();
    let service = service(&dir);
    service.ensure_bank("main", &identity(), &models()).unwrap();
    let conn = service.store().unwrap().connection();
    let (kinds, min_volatility, entity): (Option<String>, Option<String>, Option<i64>) = conn
        .query_row(
            "SELECT filter_kinds, filter_min_volatility, filter_entity_id FROM mental_models
             WHERE name = ?1",
            [PROFILE_NAME],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .unwrap();
    let kinds: Vec<String> =
        serde_json::from_str(kinds.as_deref().expect("the profile filters its kinds")).unwrap();
    let mut kinds = kinds;
    kinds.sort();
    assert_eq!(kinds, ["fact", "state"]);
    assert_eq!(min_volatility.as_deref(), Some("weeks"));
    assert_eq!(entity, None, "the profile is about the whole bank");
}

#[test]
fn merge_leaves_the_profiles_filters_alone() {
    // An owner edit to a model's filters survives every later
    // `PUT /v1/banks/{bank}` (TIM-94, decision 7; TIM-95, decision 2).
    let dir = TestDir::new();
    let service = service(&dir);
    service.ensure_bank("main", &identity(), &models()).unwrap();
    service
        .store()
        .unwrap()
        .connection()
        .execute(
            "UPDATE mental_models SET filter_kinds = '[\"fact\"]', filter_min_volatility = 'months',
                                      max_tokens = 123, question = 'edited'
             WHERE name = ?1",
            [PROFILE_NAME],
        )
        .unwrap();
    service.ensure_bank("main", &identity(), &models()).unwrap();
    let conn = service.store().unwrap().connection();
    let (kinds, min_volatility, max_tokens, question): (String, String, i64, String) = conn
        .query_row(
            "SELECT filter_kinds, filter_min_volatility, max_tokens, question FROM mental_models
             WHERE name = ?1",
            [PROFILE_NAME],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .unwrap();
    assert_eq!(kinds, "[\"fact\"]");
    assert_eq!(min_volatility, "months");
    assert_eq!(max_tokens, 123);
    assert_eq!(question, "edited");
}

#[test]
fn a_deleted_memory_takes_its_citations_and_leaves_the_entry() {
    // TIM-95, decision 6: forget and purge delete the memory row, the
    // cascade removes its citations, and code drops the entry. The entry
    // has to survive the cascade for code to see what it cited.
    let dir = TestDir::new();
    let service = service(&dir);
    service.ensure_bank("main", &identity(), &models()).unwrap();
    let conn = service.store().unwrap().connection();
    let bank = bank_rowid(&conn, "main");
    let chunk = seed_chunk(&conn, bank, "a", 0);
    let tea = insert_memory(&conn, bank, chunk, "Tim likes green tea.", 0);
    let cat = insert_memory(&conn, bank, chunk, "Tim's cat is called Miso.", 0);
    let model: i64 = conn
        .query_row(
            "SELECT id FROM mental_models WHERE bank_id = ?1 AND name = ?2",
            (bank, PROFILE_NAME),
            |row| row.get(0),
        )
        .unwrap();
    conn.execute(
        "INSERT INTO mental_model_entries (uuid, model_id, position, text, created_at, updated_at)
         VALUES ('entry-1', ?1, 0, 'Tim drinks green tea with his cat Miso.', 0, 0)",
        [model],
    )
    .unwrap();
    let entry = conn.last_insert_rowid();
    for memory in [tea, cat] {
        conn.execute(
            "INSERT INTO mental_model_citations (entry_id, memory_id) VALUES (?1, ?2)",
            (entry, memory),
        )
        .unwrap();
    }

    conn.execute("DELETE FROM memories WHERE id = ?1", [tea])
        .unwrap();
    assert_eq!(count(&conn, "SELECT count(*) FROM mental_model_entries"), 1);
    let cited: Vec<i64> = conn
        .prepare("SELECT memory_id FROM mental_model_citations WHERE entry_id = ?1")
        .unwrap()
        .query_map([entry], |row| row.get(0))
        .unwrap()
        .map(Result::unwrap)
        .collect();
    assert_eq!(cited, [cat]);
}

#[test]
fn a_deleted_model_takes_its_entries_and_their_citations() {
    let dir = TestDir::new();
    let service = service(&dir);
    service.ensure_bank("main", &identity(), &models()).unwrap();
    let conn = service.store().unwrap().connection();
    let bank = bank_rowid(&conn, "main");
    let chunk = seed_chunk(&conn, bank, "a", 0);
    let tea = insert_memory(&conn, bank, chunk, "Tim likes green tea.", 0);
    conn.execute(
        "INSERT INTO mental_model_entries (uuid, model_id, position, text, created_at, updated_at)
         SELECT 'entry-1', id, 0, 'Tim likes green tea.', 0, 0 FROM mental_models",
        [],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO mental_model_citations (entry_id, memory_id) VALUES (?1, ?2)",
        (conn.last_insert_rowid(), tea),
    )
    .unwrap();

    conn.execute("DELETE FROM mental_models", []).unwrap();
    assert_eq!(count(&conn, "SELECT count(*) FROM mental_model_entries"), 0);
    assert_eq!(
        count(&conn, "SELECT count(*) FROM mental_model_citations"),
        0
    );
    assert_eq!(
        count(&conn, "SELECT count(*) FROM memories"),
        1,
        "the memory stays"
    );
}

#[test]
fn a_corrupt_existing_copy_refuses_the_migration_and_is_kept() {
    // ADR 0010: the pre-migration copy is the same integrity-checked copy
    // backup makes, and a copy for the same from-version is never
    // overwritten. A damaged file under the copy's name can't be trusted and
    // can't be replaced, so the migration must stop and name it.
    let dir = TestDir::new();
    let clock = clock();
    let store = open(&dir.data(), clock.clone());
    let conn = store.connection();
    let path = migrations::copy_path(&dir.data(), 1);
    std::fs::write(&path, b"not a database").unwrap();

    let error = match migrations::take_copy(&conn, &dir.data(), 1) {
        Err(error) => error,
        Ok(path) => panic!("a corrupt copy was accepted: {}", path.display()),
    };
    assert!(
        error.to_string().contains(path.to_str().unwrap()),
        "the refusal doesn't name the copy: {error}"
    );
    assert_eq!(
        std::fs::read(&path).unwrap(),
        b"not a database",
        "the corrupt copy was replaced"
    );
}

#[test]
fn a_copy_stays_a_single_file_after_read_only_checks_and_reuse() {
    let dir = TestDir::new();
    let store = open(&dir.data(), clock());
    let conn = store.connection();
    let listing = || {
        let mut names: Vec<_> = std::fs::read_dir(dir.data())
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        names.sort();
        names
    };
    let mut expected = listing();
    let path = migrations::take_copy(&conn, &dir.data(), 1).unwrap();
    expected.push(path.file_name().unwrap().to_owned());
    expected.sort();
    assert_eq!(
        listing(),
        expected,
        "copy creation left sidecars or partials"
    );
    for _ in 0..2 {
        let copy =
            Connection::open_with_flags(&path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
        let journal: String = copy
            .query_row("PRAGMA journal_mode", [], |row| row.get(0))
            .unwrap();
        assert_eq!(journal, "delete");
        let integrity: String = copy
            .query_row("PRAGMA integrity_check", [], |row| row.get(0))
            .unwrap();
        assert_eq!(integrity, "ok");
        assert_eq!(listing(), expected, "a read-only check created sidecars");
        drop(copy);
        assert_eq!(migrations::take_copy(&conn, &dir.data(), 1).unwrap(), path);
        assert_eq!(listing(), expected, "reusing the copy created sidecars");
    }
}
