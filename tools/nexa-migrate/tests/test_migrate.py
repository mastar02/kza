"""End-to-end synthetic KZA migration fixtures, including the real daemon."""

import ast
import hashlib
import hmac
import json
import os
import secrets
import socket
import sqlite3
import stat
import struct
import subprocess
import sys
import time
from contextlib import closing
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1]
ROOT = TOOL.parents[1]
SHARED_TARGET = Path("/Users/gabriel/Documents/Proyectos/kza/target")
MODEL = "speechbrain/spkrec-ecapa-voxceleb"
ENCODER = "enc_" + hashlib.sha256(json.dumps([MODEL, "speechbrain", 16000]).encode()).hexdigest()
SALT = bytes(range(32))
AT = 1_700_000_000.125
FUTURE = 4_000_000_000.0
ALIVE = "synthetic-person-never-in-report"
RETIRED = "retired-person-never-in-report"
FACT = "synthetic fact content never in the report"
TRANSCRIPT = "synthetic transcript never in the report"
TOKEN = "synthetic-token-never-in-report"
EMBEDDING = struct.pack("<192f", 1, *([0] * 191))
VERSIONS = {"memory/memory.db": 1, "memory/ambient.db": 1,
            "household/household.db": 2, "auth/auth.db": 0}


def ddl(file: str, marker: str) -> str:
    """Select the literal DDL used by the legacy writer, without importing it."""
    tree = ast.parse((ROOT / file).read_text())
    return next(node.value for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and type(node.value) is str
                and marker in node.value)


def add_ledger(db: sqlite3.Connection, entries: list[tuple]) -> list[tuple]:
    db.executescript(ddl("src/memory/deletion_ledger.py", "CREATE TABLE deletion_ledger_v1"))
    db.execute("INSERT INTO deletion_ledger_salt_v1 VALUES (1,?)", (SALT.hex(),))
    previous = "0" * 64
    rows = []
    for seq, (kind, identity) in enumerate(entries, 1):
        subject = hmac.new(SALT, identity.encode(), hashlib.sha256).hexdigest()
        at = AT + seq
        content = json.dumps([seq, kind, subject, at], separators=(",", ":"))
        digest = hashlib.sha256((previous + content).encode()).hexdigest()
        rows.append((seq, kind, subject, at, previous, digest))
        previous = digest
    db.executemany("INSERT INTO deletion_ledger_v1 VALUES (?,?,?,?,?,?)", rows)
    return rows


def sources(capture: str, conversation: str) -> str:
    return json.dumps([{"capture_id": capture, "conversation_id": conversation, "captured_at": AT,
                        "expires_at": FUTURE, "origin": "ambient", "depends_on_fact_ids": []}])


def privacy_db(path: Path) -> list[tuple]:
    with closing(sqlite3.connect(path)) as db, db:
        for file, marker in (
            ("src/memory/fact_ledger.py", "CREATE TABLE IF NOT EXISTS facts"),
            ("src/memory/fact_ledger.py", "CREATE TABLE fact_owners"),
            ("src/memory/voice_schema.py", "CREATE TABLE IF NOT EXISTS voice_fingerprints"),
            ("src/memory/voice_schema.py", "CREATE TABLE voice_profile_people"),
            ("src/memory/voice_schema.py", "CREATE TABLE voice_attribution_consents"),
            ("src/ambient/control.py", "CREATE TABLE IF NOT EXISTS ambient_runtime_control"),
        ):
            db.executescript(ddl(file, marker))
        db.execute("INSERT INTO ambient_runtime_control VALUES (1,1,7)")
        db.execute("INSERT INTO conversation_tombstones VALUES ('tombstone',?,?)", (AT, ALIVE))
        entries = add_ledger(db, [
            ("conversation_erased", "tombstone"), ("user_retired", RETIRED),
            ("fact_deleted", "erased-fact"), ("voice_profile_deleted", "forgotten-profile"),
            ("credential_revoked", "revoked-credential"),
        ])
        cases = [
            ("good-fact", "good-conversation", ALIVE), ("bad-fact", "tombstone", ALIVE),
            ("retired-fact", "different-conversation", RETIRED),
            ("dependent-fact", "good-conversation", ALIVE), ("erased-fact", "good-conversation", ALIVE),
        ]
        for identity, conversation, owner in cases:
            db.execute("INSERT INTO facts VALUES (?,?,'personal',1,'confirmed',?,NULL,2,?,?)",
                       (identity, FACT + " " + identity, AT, AT, owner))
            db.execute("INSERT INTO fact_sources VALUES (?,?,?,?,?,'ambient','[]')",
                       (identity, "capture-" + identity, conversation, AT, FUTURE))
            db.execute("INSERT INTO fact_owners VALUES (?,?,'person_session',2,?,?,'{}')",
                       (identity, owner, AT, owner))
            db.execute("INSERT INTO fact_owner_revisions VALUES (?,2,'assigned',1,?)", (identity, AT))
        db.execute("INSERT INTO fact_dependencies VALUES ('dependent-fact','bad-fact')")
        voice = [
            ("profile-good", "capture-good", "good-conversation", ALIVE, "active"),
            ("profile-tomb", "capture-tomb", "tombstone", "synthetic-tomb-user", "active"),
            ("profile-retired", "capture-retired-profile", "profile-conversation", RETIRED, "active"),
            ("profile-unlinked", "capture-unlinked", "unlinked-conversation", None, "active"),
            ("profile-pending", "capture-pending", "pending-conversation", "synthetic-pending", "pending_verification"),
        ]
        for profile, capture, conversation, person, consent_state in voice:
            fingerprint = "fingerprint-" + capture
            db.execute("INSERT INTO voice_profiles VALUES (?,NULL,?,192,?,?,?,?,3,NULL)",
                       (profile, ENCODER, EMBEDDING, sources(capture, conversation), AT, ALIVE))
            db.execute("INSERT INTO voice_fingerprints VALUES (?,?,192,?,1,?,?,?,NULL,NULL,NULL)",
                       (fingerprint, ENCODER, EMBEDDING, AT, FUTURE, sources(capture, conversation)))
            db.execute("INSERT INTO voice_profile_sources VALUES (?,?)", (profile, conversation))
            db.execute("INSERT INTO voice_fingerprint_sources VALUES (?,?)", (fingerprint, conversation))
            db.execute("INSERT INTO voice_profile_people VALUES (?,?,2,?,?)", (profile, person, AT, ALIVE))
            consent_person = person or "synthetic-former-link"
            verified = (profile, 3, 2) if consent_state == "active" else (None, None, None)
            db.execute("INSERT INTO voice_attribution_consents VALUES (?,?,1,?,?,?,?,?)",
                       (consent_person, consent_state, *verified, AT, ALIVE))
        db.execute("INSERT INTO voice_attribution_control VALUES (1,1,4,?,?)", (AT, ALIVE))
        return entries


def ambient_db(path: Path) -> None:
    with closing(sqlite3.connect(path)) as db, db:
        db.executescript(ddl("src/ambient/store.py", "CREATE TABLE IF NOT EXISTS utterances"))
        # Optional ownership exported by the source, absent in older stores.
        db.execute("ALTER TABLE capture_conversations ADD COLUMN owner_user_id TEXT")
        for capture, conversation, owner in [
            ("capture-good", "good-conversation", ALIVE), ("capture-tomb", "tombstone", None),
            ("capture-retired", "retired-ambient", RETIRED),
            ("capture-retired-profile", "profile-conversation", None),
        ]:
            db.execute("INSERT INTO capture_conversations VALUES (?,'kitchen',?,?,?)",
                       (conversation, AT, AT, owner))
            db.execute("""INSERT INTO utterances(room_id,t0,t1,text,audio_path,capture_id,
                              expires_at,conversation_id,created_at)
                          VALUES ('kitchen',?,?,?,?,?,?,?,?)""",
                       (AT, AT + 1, TRANSCRIPT, "synthetic-file.wav", capture, FUTURE, conversation, AT))
        db.execute("INSERT INTO pending_audio_deletions VALUES ('pending-synthetic.wav',?,4,'private diagnostic')",
                   (AT,))
        db.execute("INSERT INTO audio_write_intents VALUES ('intent-synthetic.wav',1,'good-conversation',?)",
                   (FUTURE,))


def household_dbs(root: Path) -> None:
    with closing(sqlite3.connect(root / "reminders.db")) as db, db:
        for marker in ("CREATE TABLE IF NOT EXISTS reminders", "CREATE TABLE IF NOT EXISTS timers",
                       "CREATE TABLE IF NOT EXISTS reminder_home_actions", "CREATE TABLE IF NOT EXISTS alert_preferences"):
            db.executescript(ddl("src/reminders/reminder_store.py", marker))
        db.execute("ALTER TABLE reminders ADD COLUMN conversation_id TEXT")
        action = {"domain": "light", "service": "turn_on", "entity_id": "light.synthetic", "data": {}}
        for key, owner, scope, conversation in [
            ("good-reminder", ALIVE, "private", "good-conversation"),
            ("tomb-reminder", ALIVE, "private", "tombstone"),
            ("retired-reminder", RETIRED, "private", "retired-conversation"),
            ("shared-reminder", RETIRED, "shared", None),
        ]:
            db.execute("""INSERT INTO reminders(id,user_id,text,trigger_at,created_at,scope,delivery_room,
                              conversation_id,ha_actions) VALUES (?,?,?,?,?,?,'kitchen',?,?)""",
                       (key, owner, key + " message", FUTURE, AT, scope, conversation,
                        json.dumps([action]) if scope == "shared" else None))
        db.execute("INSERT INTO timers VALUES ('timer-legacy','Tea','household','kitchen','running',30,?,NULL,?,?,1)",
                   (FUTURE, AT, AT))
        db.execute("INSERT INTO timers VALUES ('old-timer','Old','household','kitchen','completed',60,NULL,0,?,?,3)",
                   (AT, AT))
        db.execute("INSERT INTO alert_preferences VALUES ('alert_zones','[\"kitchen\"]',?,3)", (AT,))
        request = "reminder:" + hashlib.sha256(json.dumps(
            ["shared-reminder", FUTURE, 0], separators=(",", ":")).encode()).hexdigest()
        db.execute("INSERT INTO reminder_home_actions VALUES (?,?,'sending',NULL)",
                   (request, json.dumps(action)))
    with closing(sqlite3.connect(root / "routines.db")) as db, db:
        db.executescript(ddl("src/routines/routine_store.py", "CREATE TABLE IF NOT EXISTS routines"))
        definition = {"name": "Synthetic scene", "description": "", "enabled": True,
                      "timezone": "America/Montevideo", "schedule": None, "conditions": [],
                      "actions": [{"operation": "turn_on", "entity_id": "light.synthetic"}]}
        date = "2025-08-01T10:00:00+00:00"
        db.execute("INSERT INTO routines VALUES ('routine-legacy',3,?,?,?)",
                   (ALIVE, date, json.dumps(definition)))
        snapshot = dict(definition, routine_id="routine-legacy", created_at=date, created_by=ALIVE,
                        revision=3)
        db.execute("INSERT INTO routine_runs VALUES ('run-legacy','routine-legacy','occurrence',?,"
                   "'manual','running',NULL,?,NULL)", (ALIVE, json.dumps(snapshot)))
        db.execute("INSERT INTO routine_run_actions VALUES ('run-legacy',0,'light.synthetic','sending',NULL)")
    with closing(sqlite3.connect(root / "lists.db")) as db, db:
        for marker in ("CREATE TABLE IF NOT EXISTS lists", "CREATE TABLE IF NOT EXISTS list_items"):
            db.executescript(ddl("src/lists/list_store.py", marker))
        db.execute("ALTER TABLE lists ADD COLUMN conversation_id TEXT")
        for key, owner, owner_type, conversation in [
            ("good-list", ALIVE, "user", "good-conversation"),
            ("tomb-list", ALIVE, "user", "tombstone"),
            ("retired-list", RETIRED, "user", "retired-conversation"),
            ("shared-list", None, "shared", None),
        ]:
            db.execute("INSERT INTO lists VALUES (?,?,?,?,?,?,?)",
                       (key, key + " label", owner_type, owner, AT, AT, conversation))
            db.execute("INSERT INTO list_items VALUES (?,?,?,0,?,?,NULL)",
                       ("item-" + key, key, key + " item", owner, AT))
    (root / "alerts.json").write_text(json.dumps({"alerts": [
        {"alert_id": "synthetic-alert", "alert_type": "health", "condition": "health",
         "source": "sensor.synthetic", "priority": "HIGH", "message": "private alert example",
         "timestamp": "2025-08-01T10:00:00+00:00"},
    ]}))


def auth_db(root: Path) -> None:
    with closing(sqlite3.connect(root / "dashboard_auth.db")) as db, db:
        db.executescript(ddl("src/dashboard/authority.py", "CREATE TABLE IF NOT EXISTS credentials"))
        db.execute("INSERT INTO retired_users VALUES (?,?)", (RETIRED, AT))
        for key, owner in [("good-credential", ALIVE), ("retired-credential", RETIRED),
                           ("revoked-credential", ALIVE)]:
            digest = hashlib.sha256((TOKEN + key).encode()).hexdigest()
            db.execute("INSERT INTO credentials VALUES (?,?,?,'home',?,?,?,1,'[]')",
                       (key, digest, "Synthetic user", owner, AT, FUTURE))
        db.execute("INSERT INTO sessions VALUES (?,?,NULL,?,?,1)", (TOKEN, "good-credential", AT, FUTURE))
        db.execute("INSERT INTO auth_meta VALUES ('clock_floor',?)", (str(AT),))
    (root / "users_accounts_v1.json").write_text(json.dumps({
        "schema": "kza-users-accounts-v1",
        "users": [
            {"user_id": ALIVE, "name": "Synthetic user", "permission_level": 4, "is_active": True},
            {"user_id": RETIRED, "name": "Synthetic former user", "permission_level": 3, "is_active": False},
        ],
    }))


@pytest.fixture
def legacy(tmp_path):
    root = tmp_path / "legacy"
    root.mkdir()
    entries = privacy_db(root / "memory_privacy.db")
    ambient_db(root / "ambient.db")
    household_dbs(root)
    auth_db(root)
    (root / "private-name-not-for-report").write_text(TOKEN)
    return root, entries


@pytest.fixture(scope="session")
def vec_extension(tmp_path_factory):
    if extension := os.environ.get("NEXA_SQLITE_VEC"):
        return Path(extension)
    cache = Path(os.environ.get("CARGO_HOME", Path.home() / ".cargo"))
    source = next(cache.glob("registry/src/*/sqlite-vec-0.1.7/sqlite-vec.c"), None)
    if source is None:
        pytest.fail("set NEXA_SQLITE_VEC or make the pinned sqlite-vec source available in the Cargo cache")
    library = tmp_path_factory.mktemp("sqlite-vec") / "vec0.so"
    subprocess.run(["cc", "-O2", "-fPIC", "-shared", str(source), "-o", str(library), "-lm"], check=True)
    return library


@pytest.fixture
def run_migration(vec_extension):
    def run(source, dest, *args):
        return subprocess.run([sys.executable, str(TOOL / "nexa_migrate.py"), "--source", str(source),
                               "--dest", str(dest), *map(str, args)], capture_output=True, text=True,
                              env=dict(os.environ, NEXA_SQLITE_VEC=str(vec_extension)), timeout=30)
    return run


def rows(dest, file, sql, vec_extension=None):
    with closing(sqlite3.connect(dest / file)) as db:
        if file == "memory/memory.db":
            assert vec_extension is not None
            db.enable_load_extension(True)
            db.load_extension(str(vec_extension), entrypoint="sqlite3_vec_init")
        return db.execute(sql).fetchall()


def checksums(source):
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source.iterdir() if path.is_file()}


def test_full_migration_and_privacy(legacy, tmp_path, run_migration, vec_extension):
    source, original_ledger = legacy
    dest = tmp_path / "nexa"
    result = run_migration(source, dest)
    assert result.returncode == 0, result.stderr
    mem = lambda sql: rows(dest, "memory/memory.db", sql, vec_extension)
    amb = lambda sql: rows(dest, "memory/ambient.db", sql)
    home = lambda sql: rows(dest, "household/household.db", sql)
    auth = lambda sql: rows(dest, "auth/auth.db", sql)
    assert mem("SELECT fact_id, content, owner_user_id, confirmed FROM facts") == [
        ("good-fact", FACT + " good-fact", ALIVE, 1)]
    assert mem("SELECT fact_id,conversation_id FROM fact_sources") == [("good-fact", "good-conversation")]
    assert mem("SELECT fact_id,person_id,method,revision FROM fact_owners") == [
        ("good-fact", ALIVE, "authenticated", 2)]
    assert mem("SELECT revision,change_kind FROM fact_owner_revisions") == [(2, "assign")]
    assert mem("SELECT conversation_id,erased_at_unix_ms FROM conversation_tombstones") == [
        ("tombstone", round(AT * 1000))]
    assert mem("SELECT salt FROM deletion_ledger_salt_v1") == [(SALT,)]
    expected_kinds = {"conversation": "tombstone", "retired_user": RETIRED, "fact": "erased-fact"}
    assert mem("SELECT kind,subject_sha256 FROM deletion_ledger_v1 ORDER BY seq") == [
        (kind, hashlib.sha256(SALT + kind.encode() + b"\0" + identity.encode()).digest())
        for kind, identity in expected_kinds.items()]
    saved = json.loads((dest / "memory/legacy-deletion-ledger.json").read_text())
    assert saved["salt"] == SALT.hex()
    assert saved["rows"] == [dict(zip(("seq", "kind", "subject_sha256", "at", "prev_sha256", "row_sha256"), row))
                             for row in original_ledger]
    assert mem("SELECT profile_id,person_id FROM voice_profile_people ORDER BY profile_id") == [
        ("profile-good", ALIVE), ("profile-pending", "synthetic-pending"), ("profile-unlinked", None)]
    assert mem("SELECT person_id FROM voice_attribution_consents WHERE granted=1") == [(ALIVE,)]
    assert mem("SELECT count(*) FROM voice_attribution_consents WHERE granted=0 AND profile_id IS NULL") == [(4,)]
    assert mem("SELECT enabled,revision FROM voice_attribution_control") == [(1, 4)]
    assert mem("SELECT count(*) FROM voice_vec") == [(3,)]
    assert mem("SELECT count(*) FROM voice_fingerprints WHERE capture_id='fingerprint-capture-tomb'") == [(0,)]
    assert amb("SELECT utterance_id,transcript_preview FROM utterances") == [("capture-good", TRANSCRIPT)]
    assert amb("SELECT paused,revision FROM ambient_runtime_control") == [(1, 7)]
    assert amb("SELECT path,requested_at_unix_ms FROM pending_audio_deletions") == [
        ("pending-synthetic.wav", round(AT * 1000))]
    assert amb("SELECT path,utterance_id FROM audio_write_intents") == [("intent-synthetic.wav", "capture-good")]
    assert home("SELECT name,state FROM timers ORDER BY name") == [("Old", "completed"), ("Tea", "running")]
    assert home("SELECT message,owner FROM reminders ORDER BY message") == [
        ("good-reminder message", ALIVE), ("shared-reminder message", None)]
    assert home("SELECT status FROM reminder_home_actions") == [("uncertain",)]
    assert home("SELECT name,owner FROM lists ORDER BY name") == [("good-list label", ALIVE), ("shared-list label", None)]
    assert home("SELECT text FROM list_items ORDER BY text") == [("good-list item",), ("shared-list item",)]
    assert home("SELECT name,revision FROM routines") == [("Synthetic scene", 3)]
    assert home("SELECT status FROM routine_runs") == [("interrupted",)]
    assert home("SELECT priority,message FROM alerts") == [(1, "private alert example")]
    assert home("SELECT zones,revision FROM alert_preferences") == [('["kitchen"]', 3)]
    assert auth("SELECT id,permission_level FROM nexa_accounts") == [(ALIVE, "admin")]
    expected_digest = hashlib.sha256((TOKEN + "good-credential").encode()).hexdigest()
    assert auth("SELECT id,digest,role FROM credentials") == [("good-credential", expected_digest, "home")]
    assert auth("SELECT * FROM sessions") == []
    assert auth("SELECT * FROM retired_users") == [(RETIRED, AT)]

    raw = (dest / "migration-report.json").read_text()
    report = json.loads(raw)
    for secret in (FACT, TRANSCRIPT, ALIVE, RETIRED, TOKEN, expected_digest, SALT.hex(),
                   original_ledger[0][-1], "private-name-not-for-report", str(source), "synthetic-file.wav"):
        assert secret not in raw
    for table in ("facts", "utterances", "lists", "reminders", "voice_profile_people"):
        counts = report["tables"][table]
        assert counts["filtered_tombstoned"] and counts["filtered_retired"]
    assert report["tables"]["deletion_ledger_v1"] == {
        "source": 5, "migrated": 3, "archived": 2, "discarded": 0,
        "filtered_tombstoned": 0, "filtered_retired": 0,
    }
    assert report["tables"]["sessions"]["discarded"] == 1
    assert report["destination_schema_versions"] == {"memory": 1, "ambient": 1, "household": 2, "auth": 0}
    for file, version in VERSIONS.items():
        path = dest / file
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
        assert rows(dest, file, "PRAGMA integrity_check", vec_extension) == [("ok",)]
        assert rows(dest, file, "PRAGMA foreign_key_check", vec_extension) == []
        assert rows(dest, file, "PRAGMA user_version", vec_extension) == [(version,)]
        assert rows(dest, file, "PRAGMA journal_mode", vec_extension) == [("wal",)]
    assert stat.S_IMODE(dest.stat().st_mode) == 0o700
    assert stat.S_IMODE((dest / "memory/legacy-deletion-ledger.json").stat().st_mode) == 0o600


def test_read_only_live_wal_copy_and_timestamps(legacy, tmp_path, run_migration):
    source, _ = legacy
    with closing(sqlite3.connect(source / "lists.db")) as writer:
        assert writer.execute("PRAGMA journal_mode=WAL").fetchone() == ("wal",)
        writer.execute("INSERT INTO list_items VALUES ('wal-item','good-list','wal only',0,NULL,?,NULL)", (AT,))
        writer.commit()
        assert (source / "lists.db-wal").stat().st_size > 0
        # The snapshot must reconstruct the committed WAL from WAL bytes alone.
        before_hash = checksums(source)
        for path in source.iterdir():
            path.chmod(0o444)
        source.chmod(0o555)
        before_stats = {p.name: (p.stat().st_mtime_ns, p.stat().st_atime_ns, p.stat().st_ctime_ns)
                        for p in source.iterdir()}
        before_root = source.stat()
        try:
            dest = tmp_path / "nexa"
            result = run_migration(source, dest)
            assert result.returncode == 0, result.stderr
            after_root = source.stat()
            assert (before_root.st_mtime_ns, before_root.st_atime_ns, before_root.st_ctime_ns) == (
                after_root.st_mtime_ns, after_root.st_atime_ns, after_root.st_ctime_ns)
            assert before_stats == {p.name: (p.stat().st_mtime_ns, p.stat().st_atime_ns, p.stat().st_ctime_ns)
                                    for p in source.iterdir()}
            assert checksums(source) == before_hash
            assert ("wal only",) in rows(dest, "household/household.db", "SELECT text FROM list_items")
            report = json.loads((dest / "migration-report.json").read_text())
            assert {item["sha256_before"] for item in report["source_checksums"]} == set(before_hash.values())
            assert all(item["sha256_before"] == item["sha256_after"] for item in report["source_checksums"])
        finally:
            source.chmod(0o700)
            for path in source.iterdir():
                path.chmod(0o600)


def test_dry_run_and_explicit_report(legacy, tmp_path, run_migration):
    source, _ = legacy
    dest, output = tmp_path / "nexa", tmp_path / "dry-run.json"
    result = run_migration(source, dest, "--dry-run", "--report", output)
    assert result.returncode == 0, result.stderr
    assert not dest.exists()
    assert json.loads(output.read_text())["dry_run"] is True
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    result = run_migration(source, dest, "--dry-run")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["dry_run"] is True
    assert not dest.exists()


@pytest.mark.parametrize("damage", ["bad-db", "bad-ledger", "missing-salt", "plaintext-digest"])
def test_corrupt_source_fails_closed(legacy, tmp_path, run_migration, damage):
    source, _ = legacy
    if damage == "bad-db":
        (source / "ambient.db").write_bytes(b"not SQLite")
    elif damage == "plaintext-digest":
        with closing(sqlite3.connect(source / "dashboard_auth.db")) as db, db:
            db.execute("UPDATE credentials SET digest=? WHERE id='good-credential'", (TOKEN,))
    else:
        with closing(sqlite3.connect(source / "memory_privacy.db")) as db, db:
            if damage == "bad-ledger":
                db.execute("UPDATE deletion_ledger_v1 SET row_sha256=? WHERE seq=2", ("f" * 64,))
            else:
                db.execute("DELETE FROM deletion_ledger_salt_v1")
    dest = tmp_path / "nexa"
    result = run_migration(source, dest)
    assert result.returncode == 1
    assert TOKEN not in result.stderr
    assert not dest.exists()
    assert not list(tmp_path.glob(".nexa-migrate-*"))


def test_dest_overlap_and_existing_data_are_refused(legacy, tmp_path, run_migration):
    source, _ = legacy
    dest = tmp_path / "nexa"
    dest.mkdir()
    keep = dest / "existing"
    keep.write_text("keep me")
    assert run_migration(source, dest).returncode == 1
    assert keep.read_text() == "keep me"
    assert run_migration(source, source / "nexa").returncode == 1
    assert run_migration(source, tmp_path / "another", "--report", source / "report").returncode == 1
    assert not (source / "report").exists()
    keep.unlink()
    result = run_migration(source, dest, "--report", dest / "custom.json")
    assert result.returncode == 0, result.stderr
    assert (dest / "custom.json").read_bytes() == (dest / "migration-report.json").read_bytes()


def test_real_memoryd_opens_migrated_pair(legacy, tmp_path, run_migration, vec_extension):
    binary = Path(os.environ.get("CARGO_TARGET_DIR", SHARED_TARGET)) / "debug/nexa-memoryd"
    assert binary.is_file(), f"build nexa-memoryd in {binary.parent.parent}"
    source, _ = legacy
    dest = tmp_path / "nexa"
    result = run_migration(source, dest)
    assert result.returncode == 0, result.stderr
    cfg = tmp_path / "config"
    cfg.mkdir()
    config = (ROOT / "config/nexa/nexa.toml").read_text()
    config = config.replace('data_dir = "/var/lib/nexa"', f"data_dir = {json.dumps(str(dest))}")
    # All Unix socket paths, including the macOS pytest temp prefix, must fit sockaddr_un.
    runtime = Path("/tmp") / ("nexa-migrate-" + secrets.token_hex(8))
    runtime.mkdir(mode=0o700)
    sock = runtime / "memory.sock"
    config = config.replace("voice_proposals_enabled = false", "voice_proposals_enabled = true")
    (cfg / "nexa.toml").write_text(config)
    env = {k: v for k, v in os.environ.items() if k not in
           {"NOTIFY_SOCKET", "WATCHDOG_USEC", "WATCHDOG_PID"}}
    env["RUNTIME_DIRECTORY"] = str(runtime)
    process = subprocess.Popen([str(binary), "--config", str(cfg)], env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    connected = False
    try:
        until = time.monotonic() + 10
        while time.monotonic() < until and process.poll() is None:
            with socket.socket(socket.AF_UNIX) as client:
                try:
                    client.connect(str(sock))
                    connected = True
                    break
                except (FileNotFoundError, ConnectionRefusedError):
                    time.sleep(0.02)
    finally:
        if process.poll() is None:
            process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate(timeout=5)
        sock.unlink(missing_ok=True)
        (runtime / "memory-socket.lock").unlink(missing_ok=True)
        runtime.rmdir()
    assert connected and process.returncode == 0, stdout + stderr
    assert rows(dest, "memory/memory.db", "SELECT count(*) FROM deletion_ledger_v1", vec_extension) == [(3,)]
    assert rows(dest, "memory/memory.db", "SELECT enabled,revision FROM voice_attribution_control",
                vec_extension) == [(1, 4)]


def test_source_change_after_snapshot_refuses_promotion(legacy, tmp_path, vec_extension, monkeypatch):
    # Inject a concurrent source edit between actual migration and re-hashing.
    monkeypatch.syspath_prepend(str(TOOL))
    monkeypatch.setenv("NEXA_SQLITE_VEC", str(vec_extension))
    import nexa_migrate

    source, _ = legacy
    original = nexa_migrate.populate

    def mutate_after_populating(snapshot, output):
        report = original(snapshot, output)
        (source / "private-name-not-for-report").write_text("synthetic changed bytes")
        return report

    monkeypatch.setattr(nexa_migrate, "populate", mutate_after_populating)
    dest = tmp_path / "nexa"
    with pytest.raises(nexa_migrate.MigrationError, match="source changed during migration"):
        nexa_migrate.migrate(source, dest, False, None)
    assert not dest.exists()
    assert not list(tmp_path.glob(".nexa-migrate-*"))


def test_native_ledger_is_preserved_byte_for_byte(legacy, tmp_path, run_migration, vec_extension):
    source, _ = legacy
    subject = hashlib.sha256(SALT + b"conversation\0tombstone").digest()
    retired_subject = hashlib.sha256(SALT + b"retired_user\0" + RETIRED.encode()).digest()
    chain = []
    prev = hashlib.sha256(SALT).digest()
    for seq, (kind, digest) in enumerate((("conversation", subject), ("retired_user", retired_subject)), 1):
        at = round(AT * 1000)
        head = hashlib.sha256(prev + struct.pack(">QQ", seq, at) + digest + kind.encode()).digest()
        chain.append((seq, kind, digest, at, prev, head))
        prev = head
    with closing(sqlite3.connect(source / "memory_privacy.db")) as db, db:
        db.executescript("DROP TABLE deletion_ledger_v1; DROP TABLE deletion_ledger_salt_v1;")
        sql = (ROOT / "crates/nexa-memoryd/src/memory.sql").read_text()
        db.executescript(sql[sql.index("CREATE TABLE deletion_ledger_salt_v1"):sql.index("CREATE TABLE erasure_ops")])
        db.execute("INSERT INTO deletion_ledger_salt_v1 VALUES (1,?)", (SALT,))
        db.executemany("INSERT INTO deletion_ledger_v1 VALUES (?,?,?,?,?,?)", chain)
    dest = tmp_path / "nexa"
    result = run_migration(source, dest)
    assert result.returncode == 0, result.stderr
    assert rows(dest, "memory/memory.db", "SELECT * FROM deletion_ledger_v1 ORDER BY seq", vec_extension) == chain
    assert not (dest / "memory/legacy-deletion-ledger.json").exists()


def test_legacy_profile_keeps_erasure_lineage_after_raw_capture_expiry(
    legacy, tmp_path, run_migration, vec_extension,
):
    source, _ = legacy
    with closing(sqlite3.connect(source / "memory_privacy.db")) as db, db:
        db.execute("UPDATE voice_fingerprints SET expires_at=? WHERE id='fingerprint-capture-good'", (AT,))
    with closing(sqlite3.connect(source / "ambient.db")) as db, db:
        db.execute("UPDATE utterances SET expires_at=? WHERE capture_id='capture-good'", (AT,))
    dest = tmp_path / "nexa"
    result = run_migration(source, dest)
    assert result.returncode == 0, result.stderr
    assert rows(dest, "memory/ambient.db", "SELECT * FROM utterances") == []
    sql = """SELECT c.capture_id, f.conversation_id FROM voice_profile_sources c
             JOIN voice_fingerprint_sources f USING(capture_id) WHERE c.profile_id='profile-good'"""
    [(capture, conversation)] = rows(dest, "memory/memory.db", sql, vec_extension)
    assert capture.startswith("legacy-profile-source:")
    assert conversation == "good-conversation"
    assert rows(dest, "memory/memory.db", "SELECT count(*) FROM voice_fingerprints WHERE capture_id="
                "'fingerprint-capture-good'", vec_extension) == [(0,)]


def test_withdrawn_consent_does_not_reenable_a_restored_opt_in(legacy, tmp_path, run_migration, vec_extension):
    source, _ = legacy
    with closing(sqlite3.connect(source / "memory_privacy.db")) as db, db:
        seq, prev = db.execute("SELECT seq + 1,row_sha256 FROM deletion_ledger_v1 ORDER BY seq DESC "
                               "LIMIT 1").fetchone()
        kind = "voice_consent_withdrawn"
        subject = hmac.new(SALT, (ALIVE + "#2").encode(), hashlib.sha256).hexdigest()
        at = AT + 8
        content = json.dumps([seq, kind, subject, at], separators=(",", ":"))
        digest = hashlib.sha256((prev + content).encode()).hexdigest()
        db.execute("INSERT INTO deletion_ledger_v1 VALUES (?,?,?,?,?,?)",
                   (seq, kind, subject, at, prev, digest))
    dest = tmp_path / "nexa"
    result = run_migration(source, dest)
    assert result.returncode == 0, result.stderr
    assert rows(dest, "memory/memory.db", "SELECT count(*) FROM voice_attribution_consents WHERE granted=1",
                vec_extension) == [(0,)]
