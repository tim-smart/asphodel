//! `asphodel serve`'s data dir, run as a process: the store it opens there,
//! the exclusive lock on it, and what a second daemon sees.
//!
//! "Store: SQLite schema, migrations and the data-dir lock" (TIM-103), from
//! "API surface and Hermes transport" (TIM-94, decision 4): SQLite in WAL
//! mode lives under `--data-dir`, and the daemon takes an exclusive lock on
//! the data dir. These tests see only what an operator sees: flags, exit
//! codes, stderr, `/v1/health` and the files in the data dir. The daemon
//! listens on a Unix socket so the tests never have to guess a port.

use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// How long a daemon gets to start listening, or to refuse and exit.
const STARTUP: Duration = Duration::from_secs(10);

/// `asphodel serve` with a clean environment, so the caller's `ASPHODEL_*`
/// variables can't leak in. It runs on the fake models with a floor for
/// each, because the daemon loads its models before it is ready and the
/// real ones aren't on a CI machine (TIM-105).
fn serve(dir: &TestDir) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_asphodel"));
    command
        .env_clear()
        .env("ASPHODEL_MODELS", "fake")
        .arg("serve")
        .arg("--config")
        .arg(dir.floors_for_fakes());
    command
}

/// A temporary directory removed even when an assertion unwinds.
struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "asphodel-serve-store-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    /// An empty data dir, created so the daemon doesn't have to.
    fn data_dir(&self) -> PathBuf {
        let path = self.0.join("data");
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn socket(&self, name: &str) -> PathBuf {
        self.0.join(format!("{name}.sock"))
    }

    /// A tuning file with a floor for each fake model, outside the data
    /// dir so the store's files are all the data dir holds.
    fn floors_for_fakes(&self) -> PathBuf {
        let path = self.0.join("tuning.toml");
        fs::write(
            &path,
            "[injection.reranker_floors]\n\"fake-reranker:v1\" = 0.0\n\
             [reconcile.embedding_floors]\n\"fake-embedder:v1\" = 0.5\n",
        )
        .unwrap();
        path
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// A running daemon, killed on drop.
struct Daemon {
    child: Child,
    socket: PathBuf,
    /// Its stderr so far; `drain` appends what arrived since.
    log: String,
    lines: mpsc::Receiver<String>,
}

impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Starts a daemon on `socket` and `data_dir` and collects its log up to the
/// "listening" line. Panics if it exits or stalls first.
fn start(dir: &TestDir, data_dir: &Path, socket: &Path) -> Daemon {
    let mut child = serve(dir)
        .arg("--listen")
        .arg(format!("unix:{}", socket.display()))
        .arg("--data-dir")
        .arg(data_dir)
        .env("ASPHODEL_LOG", "info")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let stderr = child.stderr.take().unwrap();
    let (lines, received) = mpsc::channel();
    std::thread::spawn(move || {
        for line in BufReader::new(stderr).lines() {
            let Ok(line) = line else { break };
            if lines.send(line).is_err() {
                break;
            }
        }
    });

    let mut log = String::new();
    let deadline = Instant::now() + STARTUP;
    loop {
        let left = deadline.saturating_duration_since(Instant::now());
        match received.recv_timeout(left) {
            Ok(line) => {
                log.push_str(&line);
                log.push('\n');
                if line.contains("asphodel listening") {
                    return Daemon {
                        child,
                        socket: socket.to_owned(),
                        log,
                        lines: received,
                    };
                }
            }
            Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                panic!("the daemon never became ready:\n{log}");
            }
        }
    }
}

impl Daemon {
    /// `GET /v1/health` over the Unix socket: the status code and the body.
    /// Fails only when the socket can't be connected to.
    fn try_health(&self) -> std::io::Result<(u16, String)> {
        let mut stream = UnixStream::connect(&self.socket)?;
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        stream
            .write_all(b"GET /v1/health HTTP/1.1\r\nHost: asphodel\r\nConnection: close\r\n\r\n")
            .unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        let status = response
            .lines()
            .next()
            .and_then(|line| line.split_whitespace().nth(1))
            .and_then(|code| code.parse().ok())
            .unwrap_or_else(|| panic!("no status line in:\n{response}"));
        let body = response.split("\r\n\r\n").nth(1).unwrap_or("").to_owned();
        Ok((status, body))
    }

    /// `GET /v1/health` on a daemon that is already known to be ready.
    fn health(&self) -> (u16, String) {
        self.try_health()
            .unwrap_or_else(|error| panic!("connecting to {}: {error}", self.socket.display()))
    }

    /// Waits for `/v1/health` to answer 200. The socket is bound before the
    /// store opens, and health answers 503 while migrations run and the
    /// models load (TIM-94, decision 3); "asphodel listening" is logged
    /// once it answers 200.
    fn wait_ready(&self) {
        let deadline = Instant::now() + STARTUP;
        loop {
            let last = match self.try_health() {
                Ok((200, _)) => return,
                Ok((status, body)) => format!("{status} {body}"),
                Err(error) => format!("connect: {error}"),
            };
            assert!(
                Instant::now() < deadline,
                "the daemon never became ready; last health was {last}\n{}",
                self.log
            );
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    /// Waits for the daemon to exit, or kills it at the deadline.
    fn wait_exit(&mut self) -> ExitStatus {
        let deadline = Instant::now() + STARTUP;
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                return status;
            }
            if Instant::now() >= deadline {
                let _ = self.child.kill();
                let status = self.child.wait().unwrap();
                self.drain();
                panic!(
                    "the daemon did not stop in time, killed it: {status}\n{}",
                    self.log
                );
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// Appends every log line that has arrived, up to the end of stderr if
    /// the daemon has exited.
    fn drain(&mut self) {
        while let Ok(line) = self.lines.recv_timeout(Duration::from_millis(500)) {
            self.log.push_str(&line);
            self.log.push('\n');
        }
    }

    /// SIGTERM, as a supervisor sends it, then the exit status and the
    /// whole log.
    fn terminate(mut self) -> (ExitStatus, String) {
        let signalled = Command::new("kill")
            .args(["-TERM", &self.child.id().to_string()])
            .status()
            .unwrap();
        assert!(signalled.success(), "kill -TERM failed");
        let status = self.wait_exit();
        self.drain();
        (status, self.log.clone())
    }

    /// SIGKILL, as a crash or an OOM kill: no cleanup code runs.
    fn crash(mut self) {
        self.child.kill().unwrap();
        self.wait_exit();
    }
}

/// Runs a daemon that is expected to refuse and exit on its own. Its stderr
/// goes to a file so a pipe can't fill while waiting, and the child is killed
/// at the deadline so a daemon that wrongly starts cannot hang the suite.
fn run_bounded(dir: &TestDir, name: &str, data_dir: &Path) -> (Option<ExitStatus>, String) {
    let log_path = dir.0.join(format!("{name}.stderr.log"));
    let mut child = serve(dir)
        .arg("--listen")
        .arg(format!("unix:{}", dir.socket(name).display()))
        .arg("--data-dir")
        .arg(data_dir)
        .env("ASPHODEL_LOG", "info")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(fs::File::create(&log_path).unwrap())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + STARTUP;
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break Some(status);
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            break None;
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    (status, fs::read_to_string(log_path).unwrap())
}

/// The first 100 bytes of a file: SQLite's database header.
type Header = [u8; 100];

/// The SQLite databases directly under `dir`, found by their header rather
/// than by name, so the tests don't fix the file name.
fn sqlite_files(dir: &Path) -> Vec<(PathBuf, Header)> {
    let mut found = Vec::new();
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if !path.is_file() {
            continue;
        }
        let mut header = [0; 100];
        let mut file = fs::File::open(&path).unwrap();
        let mut read = 0;
        while read < header.len() {
            match file.read(&mut header[read..]).unwrap() {
                0 => break,
                n => read += n,
            }
        }
        if header.starts_with(b"SQLite format 3\0") {
            found.push((path, header));
        }
    }
    found.sort();
    found
}

fn entries(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

/// Checks a header says WAL mode. Bytes 18 and 19 are the file format write
/// and read versions, and 2 means WAL (sqlite.org/fileformat2.html). SQLite
/// writes them when `journal_mode=WAL` is set, so they hold while the
/// database is open as well as after it's closed.
fn assert_wal(path: &Path, header: &Header) {
    assert_eq!(
        (header[18], header[19]),
        (2, 2),
        "{} is not in WAL mode",
        path.display()
    );
}

#[test]
fn the_store_is_one_sqlite_database_in_wal_mode_under_the_data_dir() {
    let dir = TestDir::new();
    let data = dir.data_dir();
    let daemon = start(&dir, &data, &dir.socket("daemon"));
    daemon.wait_ready();

    let files = sqlite_files(&data);
    assert_eq!(
        files.len(),
        1,
        "expected one SQLite database under {}, found {:?} among {:?}",
        data.display(),
        files.iter().map(|(path, _)| path).collect::<Vec<_>>(),
        entries(&data)
    );
    let (path, header) = &files[0];
    assert_wal(path, header);

    // A clean stop checkpoints the WAL (TIM-94, decision 3) and leaves the
    // same database behind, still in WAL mode.
    let (status, log) = daemon.terminate();
    assert!(status.success(), "clean stop exited with {status}:\n{log}");
    let after = sqlite_files(&data);
    assert_eq!(after.len(), 1, "{:?}", entries(&data));
    assert_eq!(&after[0].0, path);
    assert_wal(&after[0].0, &after[0].1);
}

#[test]
fn a_restart_reopens_the_existing_database() {
    let dir = TestDir::new();
    let data = dir.data_dir();
    let first = start(&dir, &data, &dir.socket("first"));
    first.wait_ready();
    let created = sqlite_files(&data);
    assert_eq!(created.len(), 1, "{:?}", entries(&data));
    let (status, log) = first.terminate();
    assert!(status.success(), "clean stop exited with {status}:\n{log}");

    // Opening a store that is already at the current schema version is not
    // a migration: no second database and no pre-migration copy appears.
    let second = start(&dir, &data, &dir.socket("second"));
    second.wait_ready();
    assert_eq!(second.health().0, 200);
    let reopened = sqlite_files(&data);
    assert_eq!(
        reopened.iter().map(|(path, _)| path).collect::<Vec<_>>(),
        created.iter().map(|(path, _)| path).collect::<Vec<_>>(),
        "{:?}",
        entries(&data)
    );
    assert_wal(&reopened[0].0, &reopened[0].1);
}

#[test]
fn a_second_daemon_on_the_same_data_dir_is_refused() {
    let dir = TestDir::new();
    let data = dir.data_dir();
    let first = start(&dir, &data, &dir.socket("first"));
    first.wait_ready();
    let before = entries(&data);

    let (status, log) = run_bounded(&dir, "second", &data);
    let status = status
        .unwrap_or_else(|| panic!("the second daemon kept running on a locked data dir:\n{log}"));
    assert!(
        !status.success(),
        "a second daemon started on a locked data dir:\n{log}"
    );
    assert!(
        log.contains(data.to_str().unwrap()),
        "the refusal doesn't name the data dir:\n{log}"
    );
    assert!(
        log.to_lowercase().contains("lock"),
        "the refusal doesn't say the data dir is locked:\n{log}"
    );

    // Refusing must not touch the first daemon's lock or files: a third
    // daemon is refused the same way, and the first still serves.
    let (status, log) = run_bounded(&dir, "third", &data);
    assert!(
        status.is_some_and(|status| !status.success()),
        "a third daemon got in after the second was refused:\n{log}"
    );
    assert_eq!(
        entries(&data),
        before,
        "a refused daemon changed the data dir"
    );
    assert_eq!(first.health().0, 200);
}

#[test]
fn the_lock_does_not_outlive_a_crashed_daemon() {
    // A supervisor restarts a crashed daemon at once (ADR 0006), so a lock
    // that needs cleanup code to release it would turn every crash into a
    // crash loop. SIGKILL runs no cleanup.
    let dir = TestDir::new();
    let data = dir.data_dir();
    let first = start(&dir, &data, &dir.socket("first"));
    first.wait_ready();
    first.crash();

    let second = start(&dir, &data, &dir.socket("second"));
    second.wait_ready();
    assert_eq!(second.health().0, 200);
}

#[test]
fn a_data_dir_that_is_a_regular_file_is_refused_and_preserved() {
    let dir = TestDir::new();
    let file = dir.0.join("not-a-dir");
    fs::write(&file, b"irreplaceable contents").unwrap();

    let (status, log) = run_bounded(&dir, "daemon", &file);
    let status =
        status.unwrap_or_else(|| panic!("the daemon ran with a file as its data dir:\n{log}"));
    assert!(
        !status.success(),
        "the daemon started with a file as its data dir:\n{log}"
    );
    assert!(
        log.contains(file.to_str().unwrap()),
        "the refusal doesn't name the path:\n{log}"
    );
    assert_eq!(fs::read(&file).unwrap(), b"irreplaceable contents");
}

#[test]
fn a_pre_migration_copy_is_deleted_at_its_deadline_while_the_daemon_runs() {
    // ADR 0010: the copy is deleted 7 days after its migration completes,
    // and forget reaches it within 7 days. A daemon started just before
    // that deadline keeps the copy at open, so the deletion has to come from
    // a wake at the deadline itself, not from the next hourly poll.
    use asphodel_core::store::migrations::{self, PRE_MIGRATION_COPY_TTL};
    use asphodel_core::store::{OpenOptions, Store, micros};
    use asphodel_core::{Clock, SystemClock};
    use std::sync::Arc;

    /// How long before its deadline the daemon starts, and how late past it
    /// the deletion may land: scheduler latency, not a polling interval.
    const LEAD: Duration = Duration::from_secs(5);
    const GRACE: Duration = Duration::from_secs(5);

    let dir = TestDir::new();
    let data = dir.data_dir();
    let copy = {
        // A store with a copy whose migration completed 7 days ago, less
        // LEAD. The row is a fixture, and replaces the fresh store's own row
        // where their `to_version`s meet: `expire_copies` reads
        // `from_version` and `completed_at`, not the live schema version.
        let clock: Arc<dyn Clock> = Arc::new(SystemClock);
        let store = Store::open(&data, OpenOptions::default(), clock.clone()).unwrap();
        let conn = store.connection();
        let copy = migrations::take_copy(&conn, &data, 1).unwrap();
        let completed_at = clock
            .now()
            .checked_sub(PRE_MIGRATION_COPY_TTL)
            .unwrap()
            .checked_add(LEAD)
            .unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO migrations (from_version, to_version, binary_version, started_at,
                                     completed_at)
             VALUES (1, 2, 'fixture', ?1, ?1)",
            [micros(completed_at)],
        )
        .unwrap();
        copy
    };
    let deadline = Instant::now() + LEAD;

    let daemon = start(&dir, &data, &dir.socket("daemon"));
    daemon.wait_ready();
    assert!(
        Instant::now() < deadline,
        "the daemon took longer than {LEAD:?} to become ready, so this run can't tell \
         an open-time deletion from a timed one:\n{}",
        daemon.log
    );
    assert!(
        copy.exists(),
        "the copy was deleted before its deadline:\n{}",
        daemon.log
    );

    while copy.exists() {
        assert!(
            Instant::now() < deadline + GRACE,
            "the copy was still there {GRACE:?} past its deadline:\n{}",
            daemon.log
        );
        std::thread::sleep(Duration::from_millis(50));
    }
    assert_eq!(daemon.health().0, 200);
}
