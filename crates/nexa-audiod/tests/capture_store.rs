#![allow(clippy::unwrap_used)]

use std::fs;
use std::os::unix::fs::{MetadataExt, symlink};
use std::sync::atomic::{AtomicBool, Ordering};

use nexa_audiod::capture_seal::{CaptureError, CapturePhase, CaptureState};
use nexa_audiod::capture_store::CaptureStore;

#[test]
fn restart_carries_unretired_owners_and_interrupted_resume_stays_muted() {
    let root = tempfile::tempdir().unwrap();
    let data = root.path();
    let path = data.join("audiod/capture_seal.json");
    let mut capture = CaptureStore::open(data).unwrap().restore().unwrap();
    assert!(CaptureStore::open(data).is_err());
    capture.acknowledge_resumed(0).unwrap();
    let first = capture.begin_capture().unwrap();
    capture.acknowledge_started(first).unwrap();
    capture.begin_capture().unwrap();
    capture.set_muted(true, 0).unwrap();
    assert_eq!(capture.status().phase(), CapturePhase::Muting);
    assert_eq!(path.metadata().unwrap().mode() & 0o777, 0o600);
    let written: CaptureState = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    assert_eq!(&written, capture.durable_state());
    drop(capture);

    let mut recovery = CaptureStore::open(data).unwrap().restore().unwrap();
    let status = recovery.status();
    assert_eq!(status.phase(), CapturePhase::Muting);
    assert_eq!(status.revision, 1);
    assert_eq!(status.carried_capture_owners, 2);
    assert!(!status.total_mute_confirmed);
    recovery.acknowledge_seal(1).unwrap();
    assert_eq!(recovery.set_muted(false, 1), Err(CaptureError::Busy));
    assert_eq!(recovery.forget_carried_owners(2, true), Ok(2));
    assert!(recovery.status().total_mute_confirmed);
    recovery.set_muted(false, 1).unwrap();
    assert_eq!(recovery.status().phase(), CapturePhase::Resuming);
    drop(recovery);

    let recovered = CaptureStore::open(data).unwrap().restore().unwrap();
    let status = recovered.status();
    assert_eq!(status.phase(), CapturePhase::Muting);
    assert_eq!(status.revision, 2);
    assert!(status.ingress_sealed && status.persisted_muted);
    assert_eq!(status.carried_capture_owners, 0);
}

#[test]
fn atomic_replacements_are_readable_and_unclean_running_restart_is_sealed() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("audiod/capture_seal.json");
    let mut capture = CaptureStore::open(root.path()).unwrap().restore().unwrap();
    capture.acknowledge_resumed(0).unwrap();
    let before = fs::File::open(&path).unwrap();
    let done = AtomicBool::new(false);
    std::thread::scope(|scope| {
        let reader = scope.spawn(|| {
            let mut reads = 0;
            while !done.load(Ordering::Acquire) || reads == 0 {
                let _: CaptureState = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
                reads += 1;
            }
        });
        for _ in 0..20 {
            let owner = capture.begin_capture().unwrap();
            capture.acknowledge_stopped(owner).unwrap();
        }
        capture.begin_capture().unwrap();
        done.store(true, Ordering::Release);
        reader.join().unwrap();
    });
    assert_ne!(
        before.metadata().unwrap().ino(),
        path.metadata().unwrap().ino()
    );
    assert_eq!(
        serde_json::from_reader::<_, CaptureState>(before).unwrap(),
        CaptureState::default()
    );
    assert_eq!(fs::read_dir(path.parent().unwrap()).unwrap().count(), 2);
    drop(capture);
    let recovered = CaptureStore::open(root.path()).unwrap().restore().unwrap();
    assert_eq!(recovered.status().phase(), CapturePhase::Muting);
    assert_eq!(recovered.status().carried_capture_owners, 1);
    assert_eq!(recovered.running_epoch(), None);
}

#[test]
fn invalid_snapshots_and_failed_writes_are_fail_closed() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("audiod/capture_seal.json");
    let mut capture = CaptureStore::open(root.path()).unwrap().restore().unwrap();
    capture.acknowledge_resumed(0).unwrap();
    let owner = capture.begin_capture().unwrap();
    fs::remove_file(&path).unwrap();
    fs::create_dir(&path).unwrap();

    assert_eq!(capture.set_muted(true, 0), Err(CaptureError::Persistence));
    assert_eq!(
        capture.acknowledge_stopped(owner),
        Err(CaptureError::Persistence)
    );
    assert_eq!(capture.running_epoch(), None);
    assert_eq!(capture.status().live_capture_owners, 1);
    assert!(!capture.status().total_mute_confirmed);
    drop(capture);
    assert!(CaptureStore::open(root.path()).unwrap().restore().is_err());

    fs::remove_dir(&path).unwrap();
    symlink(root.path().join("missing"), &path).unwrap();
    assert!(CaptureStore::open(root.path()).unwrap().restore().is_err());
}
