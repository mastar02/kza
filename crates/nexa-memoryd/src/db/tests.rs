use crate::voice::Observation;

use super::schema::{AMBIENT, MAIN};
use super::{Connection, Error, FACT_DIMENSIONS, Store, VOICE_DIMENSIONS, VOICE_MODEL};

fn sample() -> Vec<f32> {
    let mut vector = vec![0.0; VOICE_DIMENSIONS];
    vector[0] = 1.0;
    vector
}

fn count(conn: &Connection, table: &str) -> i64 {
    conn.query_row(&format!("SELECT count(*) FROM {table}"), [], |row| {
        row.get(0)
    })
    .expect("count rows")
}

fn insert_source(store: &mut Store) {
    store
        .configure_voice_attribution(true)
        .expect("enable voice");
    store
        .record_observation(Observation {
            utterance: "source",
            room: "room",
            model: VOICE_MODEL,
            preview: "synthetic",
            captured_at: 123,
            embedding: &sample(),
        })
        .expect("seed capture");
    let mut embedding = vec![0.0; FACT_DIMENSIONS];
    embedding[0] = 1.0;
    store
        .store_fact("source", "", "synthetic fact", "", &embedding)
        .expect("seed fact");
}

#[test]
fn on_disk_migrations_set_pragmas_and_reopen() {
    let root = tempfile::tempdir().expect("tempdir");
    let path = root.path().join("data");
    for _ in 0..2 {
        let store = Store::open(&path).expect("open databases");
        let conn = store.read().expect("read");
        let timeout: u32 = conn
            .pragma_query_value(None, "busy_timeout", |row| row.get(0))
            .expect("timeout");
        assert_eq!(timeout, 5000);
        for schema in [MAIN, AMBIENT] {
            for (pragma, expected) in [
                ("user_version", 1),
                ("foreign_keys", 1),
                ("secure_delete", 1),
                ("synchronous", 2),
            ] {
                let actual: u32 = conn
                    .pragma_query_value(Some(schema), pragma, |row| row.get(0))
                    .expect("pragma");
                assert_eq!(expected, actual, "{schema}.{pragma}");
            }
            let mode: String = conn
                .pragma_query_value(Some(schema), "journal_mode", |row| row.get(0))
                .expect("journal mode");
            assert_eq!(mode, "wal");
        }
    }
}

#[test]
fn rejects_future_and_partial_schemas_without_repairing_them() {
    let root = tempfile::tempdir().expect("tempdir");
    let path = root.path().join("data");
    {
        let store = Store::open(&path).expect("initialize");
        store
            .read()
            .expect("read")
            .execute_batch("PRAGMA ambient.user_version=42;")
            .expect("future version");
    }
    assert!(matches!(
        Store::open(&path),
        Err(Error::Schema("unsupported user_version"))
    ));
    {
        let conn = Connection::open(path.join("ambient.db")).expect("open");
        conn.execute_batch("PRAGMA user_version=1; DROP TABLE pending_audio_deletions;")
            .expect("remove table");
    }
    assert!(matches!(
        Store::open(&path),
        Err(Error::Schema("incomplete or incompatible schema"))
    ));
}

#[test]
fn startup_rejects_changed_dimensions_and_corrupted_ledger() {
    let root = tempfile::tempdir().expect("tempdir");
    let path = root.path().join("data");
    {
        let store = Store::open(&path).expect("initialize");
        store
            .read()
            .expect("read")
            .execute(
                "UPDATE schema_meta SET dimensions=100 WHERE vector_kind='fact'",
                [],
            )
            .expect("changed dimensions");
    }
    assert!(matches!(
        Store::open(&path),
        Err(Error::Schema("unrecognized vector model or dimensions"))
    ));
    {
        let conn = Connection::open(path.join("memory.db")).expect("open");
        conn.execute(
            "UPDATE schema_meta SET dimensions=?1 WHERE vector_kind='fact'",
            [FACT_DIMENSIONS],
        )
        .expect("restore dimensions");
        drop(conn);
        let mut store = Store::open(&path).expect("open again");
        store.erase_conversation("test").expect("append to ledger");
        store
            .read()
            .expect("read")
            .execute("UPDATE deletion_ledger_v1 SET row_sha256=zeroblob(32)", [])
            .expect("damage ledger");
    }
    assert!(matches!(
        Store::open(&path),
        Err(Error::Schema("deletion ledger chain invalid"))
    ));
}

#[test]
fn erasure_failure_rolls_back_vectors_tombstones_and_both_databases() {
    let root = tempfile::tempdir().expect("tempdir");
    let mut store = Store::open(&root.path().join("data")).expect("open");
    insert_source(&mut store);
    let conn = store.read().expect("read");
    conn.execute_batch(
        "UPDATE ambient.utterances SET audio_path='test-clip';
         INSERT INTO ambient.audio_write_intents VALUES ('test-intent', 'source', 'source', 100);
         CREATE TEMP TRIGGER fail_erase BEFORE DELETE ON main.voice_fingerprints
         BEGIN SELECT RAISE(ABORT, 'injected after both databases have changed'); END;",
    )
    .expect("inject constraint failure");

    assert!(store.erase_conversation("source").is_err());
    let conn = store.read().expect("read after rollback");
    for table in [
        "facts",
        "fact_vec",
        "voice_fingerprints",
        "voice_vec",
        "ambient.utterances",
        "ambient.audio_write_intents",
        "ambient.capture_conversations",
    ] {
        assert_eq!(count(conn, table), 1, "{table}");
    }
    for table in [
        "conversation_tombstones",
        "deletion_ledger_v1",
        "erasure_ops",
        "ambient.pending_audio_deletions",
    ] {
        assert_eq!(count(conn, table), 0, "{table}");
    }
    conn.execute_batch("DROP TRIGGER fail_erase")
        .expect("allow retry");
    let receipt = store.erase_conversation("source").expect("retry");
    assert_eq!(
        (
            receipt.facts_deleted,
            receipt.vectors_deleted,
            receipt.captures_deleted
        ),
        (1, 2, 1)
    );
    let conn = store.read().expect("read after erasure");
    assert_eq!(count(conn, "ambient.pending_audio_deletions"), 2);
    for schema in [MAIN, AMBIENT] {
        let mode: String = conn
            .pragma_query_value(Some(schema), "journal_mode", |row| row.get(0))
            .expect("WAL restored");
        assert_eq!(mode, "wal");
    }
    for table in ["fact_vec_vector_chunks00", "voice_vec_vector_chunks00"] {
        let remaining: i64 = conn
            .query_row(
                &format!("SELECT count(*) FROM {table} WHERE vectors<>zeroblob(length(vectors))"),
                [],
                |row| row.get(0),
            )
            .expect("check residual vector bytes");
        assert_eq!(remaining, 0, "erased bytes remain in {table}");
    }
}

#[test]
fn voice_observation_keeps_wal_available_to_active_readers() {
    let root = tempfile::tempdir().expect("tempdir");
    let path = root.path().join("data");
    let mut store = Store::open(&path).expect("open");
    let mut memory = Connection::open(path.join("memory.db")).expect("open memory reader");
    let mut ambient = Connection::open(path.join("ambient.db")).expect("open ambient reader");
    let memory_snapshot = memory.transaction().expect("begin memory reader");
    let ambient_snapshot = ambient.transaction().expect("begin ambient reader");
    assert_eq!(count(&memory_snapshot, "voice_fingerprints"), 0);
    assert_eq!(count(&ambient_snapshot, "utterances"), 0);

    insert_source(&mut store);

    assert_eq!(count(&memory_snapshot, "voice_fingerprints"), 0);
    assert_eq!(count(&ambient_snapshot, "utterances"), 0);
    let conn = store.read().expect("read after observation");
    assert_eq!(count(conn, "voice_fingerprints"), 1);
    assert_eq!(count(conn, "ambient.utterances"), 1);
}

#[test]
fn voice_observation_retries_after_only_provenance_is_committed() {
    let root = tempfile::tempdir().expect("tempdir");
    let path = root.path().join("data");
    let mut store = Store::open(&path).expect("open");
    store
        .read()
        .expect("read")
        .execute_batch(
            "CREATE TEMP TRIGGER fail_observation BEFORE INSERT ON main.voice_fingerprints
             BEGIN SELECT RAISE(ABORT, 'injected fingerprint failure'); END;",
        )
        .expect("inject fingerprint failure");
    let embedding = sample();
    let observe = |store: &mut Store| {
        store.record_observation(Observation {
            utterance: "source",
            room: "room",
            model: VOICE_MODEL,
            preview: "synthetic",
            captured_at: 123,
            embedding: &embedding,
        })
    };
    assert!(matches!(observe(&mut store), Err(Error::Sql(_))));
    drop(store);

    let mut store = Store::open(&path).expect("reopen after failed observation");
    let conn = store.read().expect("read after reopen");
    assert_eq!(count(conn, "ambient.utterances"), 1);
    for table in ["voice_profiles", "voice_vec", "voice_fingerprints"] {
        assert_eq!(count(conn, table), 0, "{table}");
    }

    let capture = observe(&mut store)
        .expect("retry using stored source")
        .capture_id;
    assert_eq!(observe(&mut store).expect("replay").capture_id, capture);
    let conn = store.read().expect("read after retry");
    assert_eq!(count(conn, "ambient.utterances"), 1);
    assert_eq!(count(conn, "voice_fingerprints"), 1);
    store.erase_conversation("source").expect("erase capture");
    assert!(matches!(observe(&mut store), Err(Error::Precondition(_))));
}
