//! Current temp-name collisions must be skipped, not opened or removed.
//! This must be the only test in its integration-test binary: the
//! models::write process-wide counter must start at 0.

use std::os::unix::fs::{PermissionsExt, symlink};
use std::path::{Path, PathBuf};

use asphodel_core::config::Secret;
use asphodel_core::models::*;
use sha2::{Digest, Sha256};

struct TestDir(PathBuf);

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn candidate(path: &Path, counter: u64) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(format!(".part-{}-{counter}", std::process::id()));
    PathBuf::from(name)
}

struct CannedFetcher;

const MODEL_BYTES: &[u8] = b"known model bytes";

impl Fetcher for CannedFetcher {
    fn fetch(&self, url: &str) -> Result<Vec<u8>, FetchError> {
        assert_eq!(url, "https://models.example/model.onnx");
        Ok(MODEL_BYTES.to_vec())
    }
}

#[test]
fn current_temp_name_collisions_leave_planted_files_untouched() {
    let dir = TestDir(
        std::env::temp_dir().join(format!("asphodel-temp-file-names-{}", std::process::id())),
    );
    std::fs::create_dir(&dir.0).unwrap();
    let data = dir.0.join("data");
    std::fs::create_dir(&data).unwrap();
    let store = TokenStore::open(&data);
    let victim = dir.0.join("victim.txt");
    std::fs::write(&victim, "irreplaceable contents").unwrap();
    let planted_link = candidate(store.path(), 0);
    let planted_file = candidate(store.path(), 1);
    symlink(&victim, &planted_link).unwrap();
    std::fs::write(&planted_file, "someone else's file").unwrap();
    let tokens = ChatgptTokens {
        access_token: Secret::new("access"),
        refresh_token: Secret::new("refresh"),
        id_token: Secret::new("id"),
        account_id: "account".into(),
        last_refresh: "2026-03-02T09:00:00Z".parse().unwrap(),
    };

    // Candidates 0 and 1 are occupied; the save must use candidate 2.
    store.save(&tokens).unwrap();
    assert_eq!(
        std::fs::read_to_string(&victim).unwrap(),
        "irreplaceable contents",
        "wrote through the symlink"
    );
    assert_eq!(
        std::fs::read_to_string(&planted_file).unwrap(),
        "someone else's file"
    );
    assert!(
        std::fs::symlink_metadata(&planted_link)
            .unwrap()
            .file_type()
            .is_symlink()
    );
    let installed = std::fs::symlink_metadata(store.path()).unwrap();
    assert!(installed.file_type().is_file());
    assert_eq!(installed.permissions().mode() & 0o777, 0o600);
    assert_eq!(store.load().unwrap(), Some(tokens));

    let models = ModelDir::at(dir.0.join("models"));
    let checksum = format!("{:x}", Sha256::digest(MODEL_BYTES));
    let spec = ModelSpec {
        id: "test-model".into(),
        dir: "test-model".into(),
        files: vec![ModelFile {
            name: "model.onnx".into(),
            url: "https://models.example/model.onnx".into(),
            sha256: checksum.clone(),
        }],
    };
    let model_path = models.file(&spec, "model.onnx");
    std::fs::create_dir_all(model_path.parent().unwrap()).unwrap();
    let model_link = candidate(&model_path, 3);
    symlink(&victim, &model_link).unwrap();

    let report = fetch_models(&models, &[spec], &CannedFetcher).unwrap();
    assert_eq!(report.fetched, std::slice::from_ref(&model_path));
    assert!(report.skipped.is_empty());
    assert_eq!(
        std::fs::read_to_string(&victim).unwrap(),
        "irreplaceable contents",
        "fetch wrote through the symlink"
    );
    assert!(
        std::fs::symlink_metadata(&model_link)
            .unwrap()
            .file_type()
            .is_symlink()
    );
    assert!(
        std::fs::symlink_metadata(&model_path)
            .unwrap()
            .file_type()
            .is_file()
    );
    let fetched = std::fs::read(&model_path).unwrap();
    assert_eq!(fetched, MODEL_BYTES);
    assert_eq!(format!("{:x}", Sha256::digest(&fetched)), checksum);
}
