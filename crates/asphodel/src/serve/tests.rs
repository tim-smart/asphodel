use super::*;
use std::{
    io::Read,
    os::unix::{fs::symlink, net::UnixListener as StdUnixListener},
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};

// No new dependencies: each test owns an exclusively created, short socket
// directory, removed even when an assertion unwinds.
struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        loop {
            let path = std::env::temp_dir().join(format!(
                "asphodel-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(error) if error.kind() == ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("creating test directory: {error}"),
            }
        }
    }

    fn socket(&self) -> PathBuf {
        self.0.join("listen.sock")
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn identity(path: &Path) -> (u64, u64) {
    let metadata = fs::symlink_metadata(path).unwrap();
    (metadata.dev(), metadata.ino())
}

async fn refusal(path: &Path, reason: &str) {
    let error = match tokio::time::timeout(Duration::from_secs(2), bind_unix(path))
        .await
        .expect("bind probe timed out")
    {
        Err(error) => error,
        Ok(_) => panic!("occupied path was accepted"),
    };
    let message = error.to_string();
    assert!(message.contains(reason), "{message}");
    assert!(message.contains(path.to_str().unwrap()), "{message}");
}

async fn assert_connectivity(path: &Path, listener: &UnixListener) {
    let mut client = std::os::unix::net::UnixStream::connect(path).unwrap();
    client
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    let (server, _) = tokio::time::timeout(Duration::from_secs(2), listener.accept())
        .await
        .expect("listener did not accept a fresh connection")
        .unwrap();
    tokio::time::timeout(Duration::from_secs(2), server.writable())
        .await
        .expect("accepted stream did not become writable")
        .unwrap();
    assert_eq!(server.try_write(b"ok").unwrap(), 2);
    let mut reply = [0; 2];
    client.read_exact(&mut reply).unwrap();
    assert_eq!(&reply, b"ok");
}

#[tokio::test]
async fn regular_file_is_refused_and_preserved() {
    let dir = TestDir::new();
    let path = dir.socket();
    fs::write(&path, b"irreplaceable contents").unwrap();
    let original = identity(&path);

    refusal(&path, "regular file").await;

    assert_eq!(identity(&path), original);
    assert!(fs::symlink_metadata(&path).unwrap().file_type().is_file());
    assert_eq!(fs::read(&path).unwrap(), b"irreplaceable contents");
}

#[tokio::test]
async fn symlink_to_live_socket_is_refused_and_preserved() {
    let dir = TestDir::new();
    let path = dir.socket();
    let target = dir.0.join("target.sock");
    let listener = UnixListener::bind(&target).unwrap();
    symlink(&target, &path).unwrap();
    let original = identity(&path);
    let target_identity = identity(&target);

    refusal(&path, "symlink").await;

    assert_eq!(identity(&path), original);
    assert_eq!(fs::read_link(&path).unwrap(), target);
    assert_eq!(identity(&target), target_identity);
    assert_connectivity(&path, &listener).await;
}

#[tokio::test]
async fn dangling_symlink_is_refused_and_preserved() {
    let dir = TestDir::new();
    let path = dir.socket();
    let target = dir.0.join("missing");
    symlink(&target, &path).unwrap();
    let original = identity(&path);

    refusal(&path, "symlink").await;

    assert_eq!(identity(&path), original);
    assert_eq!(fs::read_link(&path).unwrap(), target);
    assert!(!target.exists());
}

#[tokio::test]
async fn live_listener_is_refused_without_disrupting_connectivity() {
    let dir = TestDir::new();
    let path = dir.socket();
    let listener = UnixListener::bind(&path).unwrap();
    let original = identity(&path);

    refusal(&path, "already in use").await;

    assert_eq!(identity(&path), original);
    // Drain the liveness probe before checking a fresh client's data transfer.
    let (probe, _) = tokio::time::timeout(Duration::from_secs(2), listener.accept())
        .await
        .expect("missing liveness probe")
        .unwrap();
    drop(probe);
    assert_connectivity(&path, &listener).await;
}

#[tokio::test]
async fn stale_socket_is_recovered_and_accepts_connections() {
    let dir = TestDir::new();
    let path = dir.socket();
    drop(StdUnixListener::bind(&path).unwrap());
    assert!(fs::symlink_metadata(&path).unwrap().file_type().is_socket());
    assert_eq!(
        std::os::unix::net::UnixStream::connect(&path)
            .unwrap_err()
            .kind(),
        ErrorKind::ConnectionRefused
    );

    let (listener, cleanup) = bind_unix(&path).await.unwrap();

    assert!(fs::symlink_metadata(&path).unwrap().file_type().is_socket());
    assert_connectivity(&path, &listener).await;
    drop(listener);
    cleanup.remove().unwrap();
    assert!(!path.exists());
}

#[tokio::test]
async fn cleanup_removes_own_socket_after_listener_is_dropped() {
    let dir = TestDir::new();
    let path = dir.socket();
    let (listener, cleanup) = bind_unix(&path).await.unwrap();
    assert_connectivity(&path, &listener).await;
    drop(listener);
    assert!(fs::symlink_metadata(&path).unwrap().file_type().is_socket());

    cleanup.remove().unwrap();

    assert_eq!(
        fs::symlink_metadata(&path).unwrap_err().kind(),
        ErrorKind::NotFound
    );
}

#[tokio::test]
async fn cleanup_preserves_replacement_socket_and_its_connectivity() {
    let dir = TestDir::new();
    let path = dir.socket();
    let (listener, cleanup) = bind_unix(&path).await.unwrap();
    let original = identity(&path);
    drop(listener);
    fs::remove_file(&path).unwrap();
    let replacement = UnixListener::bind(&path).unwrap();
    let replacement_identity = identity(&path);
    assert_ne!(replacement_identity, original);

    cleanup.remove().unwrap();

    assert_eq!(identity(&path), replacement_identity);
    assert_connectivity(&path, &replacement).await;
}

#[tokio::test]
async fn cleanup_preserves_replacement_regular_file() {
    let dir = TestDir::new();
    let path = dir.socket();
    let (listener, cleanup) = bind_unix(&path).await.unwrap();
    drop(listener);
    fs::remove_file(&path).unwrap();
    fs::write(&path, b"replacement contents").unwrap();
    let replacement = identity(&path);

    cleanup.remove().unwrap();

    assert_eq!(identity(&path), replacement);
    assert_eq!(fs::read(&path).unwrap(), b"replacement contents");
}

#[tokio::test]
async fn cleanup_preserves_replacement_symlink_and_target() {
    let dir = TestDir::new();
    let path = dir.socket();
    let (listener, cleanup) = bind_unix(&path).await.unwrap();
    drop(listener);
    fs::remove_file(&path).unwrap();
    let target = dir.0.join("target");
    fs::write(&target, b"target contents").unwrap();
    symlink(&target, &path).unwrap();
    let replacement = identity(&path);

    cleanup.remove().unwrap();

    assert_eq!(identity(&path), replacement);
    assert_eq!(fs::read_link(&path).unwrap(), target);
    assert_eq!(fs::read(&target).unwrap(), b"target contents");
}

#[tokio::test]
async fn cleanup_succeeds_if_socket_was_already_removed() {
    let dir = TestDir::new();
    let path = dir.socket();
    let (listener, cleanup) = bind_unix(&path).await.unwrap();
    drop(listener);
    fs::remove_file(&path).unwrap();

    cleanup.remove().unwrap();

    assert!(!path.exists());
}
