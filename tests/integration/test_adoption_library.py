"""U3b-0e2: the adoption library on a production-shaped state written by b0866e8's own code.

Adoption design v3 sub-unit U3b-0e2 acceptance (one test group per item):
copy discipline (A1), hot journal (A2), P2=A drop-and-vacuum (A3),
database-scoped contracts (A4), S3 against the generated diff (A5), the
relocator and the first-start target set (A6), legacy_training (A7), the
ledger seed (A8) and residue (A9). The fixture (tests/mocks/adoption_legacy)
runs b0866e8's builders, ledger and journal; the R1 steps run in the real
schema-only child.
"""
from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack, closing
import hashlib
import hmac
import json
from pathlib import Path
import shutil
import sqlite3
import sys

import pytest

from scripts import adoption
from scripts import managed_launcher as launcher
from scripts import stage_runtime as stage
from tests.mocks import adoption_legacy as fixture

pytestmark = pytest.mark.skipif(not fixture.available(), reason="b0866e8 is not available in this clone")

REPOSITORY = Path(__file__).resolve().parents[2]
PLAN = "c" * 64
AT = 1_800_000_500.0
ERASED = "11111111-1111-4111-8111-111111111111"
NO_OPERATION = "77777777-7777-4777-8777-777777777777"


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    return fixture.b0866e8_tree(tmp_path_factory.mktemp("b0866e8") / "tree")


@pytest.fixture(scope="module")
def adopted(tmp_path_factory, tree):
    """One legacy layout, adopted once; the tests below only read it."""
    root = tmp_path_factory.mktemp("adopted").resolve()
    layout = fixture.build_legacy(root / "legacy", tree)
    source_before = fixture.tree_bytes(root / "legacy")
    snapshot = root / "legacy-before"
    shutil.copytree(layout.legacy_state, snapshot)
    report = _adopt(layout, root)
    return {"root": root, "layout": layout, "report": report, "work": root / "work",
            "state_root": root / "slot" / "state", "source_before": source_before, "snapshot": snapshot}


def _adopt(layout, root, **options):
    return adoption.adopt_copy(layout, work=root / "work", scratch=root / "scratch",
                               state_root=root / "slot" / "state", plan_sha256=PLAN, adoption_time=AT,
                               python=sys.executable, runtime_root=REPOSITORY, **options)


def _variant(tmp_path, tree, mutate=None, **build):
    root = tmp_path.resolve()
    layout = fixture.build_legacy(root / "legacy", tree, **build)
    if mutate is not None:
        mutate(layout.legacy_state / "data")
    return layout, root


def _rows(path, query, parameters=()):
    with closing(sqlite3.connect(Path(path).as_uri() + "?mode=ro", uri=True)) as db:
        return db.execute(query, parameters).fetchall()


# --- A1: copy discipline -----------------------------------------------------------------------------

def test_every_source_file_is_hashed_and_the_source_is_untouched(adopted):
    layout, report = adopted["layout"], adopted["report"]
    assert fixture.tree_bytes(adopted["root"] / "legacy") == adopted["source_before"]
    entries = adoption.file_manifest(layout.legacy_state)
    members = set(adoption.metrics_store_set(layout, adoption.PRODUCTION_METRICS)["members"])
    metrics = [["metrics/" + item[0], *item[1:]] for item in adoption.file_manifest(layout.metrics_root)
               if item[0] in members]
    assert len(metrics) == 3  # the jsonl, its '.1' partition and the catalog
    assert report["source_manifest_sha256"] == adoption.manifest_digest(entries + metrics)
    assert report["source_files"] == len(entries) + len(metrics)
    assert {"data/audit.db", "logs/voice_assistant.log", ".env", "data/lists.db-journal"} <= {
        item[0] for item in entries}
    assert report["copy"]["not_carried_files"] == 1 and report["copy"]["sqlite-journal"] == 1
    copied = {item[0]: item[4] for item in adoption.file_manifest(adopted["work"])}
    for relative, *_rest, digest in entries:
        if relative.startswith(("data/chroma_db/", "data/memory_db/", "data/wakeword_training/",
                                "data/ambient_audio/")) and not relative.endswith(".sqlite3"):
            assert copied[relative] == digest, relative  # payload files travel byte-identical
    for item in metrics:
        assert copied["data/obs/" + item[0][len("metrics/"):]] == item[4] or item[0].endswith(".sqlite3")


def test_the_audit_hook_saw_the_copy_and_refused_nothing(adopted):
    guard = adopted["report"]["guard"]
    assert guard["refused"] == 0 and guard["connects_observed"] > 0


def test_a_source_access_inside_the_pipeline_is_refused(tmp_path, tree, monkeypatch):
    layout, root = _variant(tmp_path, tree)
    original = adoption.recover_copies

    def touch_source(state):
        sqlite3.connect(layout.container / "data" / "lists.db").close()  # through the container link
        return original(state)
    monkeypatch.setattr(adoption, "recover_copies", touch_source)
    before = fixture.tree_bytes(root / "legacy")
    with pytest.raises(adoption.AdoptionRefusal, match="runtime_admission_adoption_source_access:sqlite3.connect"):
        _adopt(layout, root)
    assert fixture.tree_bytes(root / "legacy") == before


# --- A2: the hot journal --------------------------------------------------------------------------

def test_the_hot_journal_is_recovered_on_the_copy_and_counted(adopted):
    assert adopted["report"]["recovery"]["hot_journals_recovered"] == 1
    lists = adopted["work"] / "data" / "lists.db"
    assert not Path(str(lists) + "-journal").exists()
    assert _rows(lists, "SELECT COUNT(*) FROM lists WHERE name LIKE ?", (fixture.UNCOMMITTED + "%",)) == [(0,)]
    assert _rows(lists, "PRAGMA quick_check") == [("ok",)]
    # The source keeps its hot journal: recovery never happened there.
    assert Path(str(adopted["layout"].legacy_state / "data" / "lists.db") + "-journal").exists()


# --- A3: P2=A ----------------------------------------------------------------------------------------

def test_audit_db_and_the_legacy_events_table_reach_no_byte_of_the_copy(adopted):
    work = adopted["work"]
    assert not (work / "data" / "audit.db").exists()
    assert adopted["report"]["drops"] == {"data/events.db": {"tables_dropped": 1}}
    events = work / "data" / "events.db"
    assert _rows(events, "PRAGMA freelist_count") == [(0,)]
    assert "events" not in {row[0] for row in _rows(events, "SELECT name FROM sqlite_master WHERE type='table'")}
    for marker in (b"KZA-LEGACY-EVENTS-MARKER", b"KZA-LEGACY-AUDIT-MARKER"):
        for path in work.rglob("*"):
            if path.is_file():
                assert marker not in path.read_bytes(), path
    # The legacy image keeps both (they stay there, documented).
    legacy = adopted["layout"].legacy_state / "data"
    assert b"KZA-LEGACY-EVENTS-MARKER" in (legacy / "events.db").read_bytes()
    assert (legacy / "audit.db").exists()
    assert not any((adopted["root"] / "scratch").rglob("*.db"))


# --- A4: database-scoped contracts ------------------------------------------------------------------

def test_latency_records_is_admitted_only_in_data_latency_db(adopted, tmp_path):
    work = adopted["work"]
    assert _rows(work / "data" / "latency.db", "SELECT COUNT(*) FROM sqlite_master WHERE name='latency_records'") == [(1,)]
    authority = stage.capture_state_authority(work)
    assert "data/latency.db" in authority["projection"]["tables"]
    elsewhere = tmp_path / "state"
    (elsewhere / "data").mkdir(parents=True)
    shutil.copyfile(work / "data" / "latency.db", elsewhere / "data" / "other.db")
    with pytest.raises(ValueError, match="runtime_admission_unknown_schema"):
        stage.capture_state_authority(elsewhere)
    with pytest.raises(adoption.AdoptionRefusal, match="unknown_table:data/other.db:latency_records"):
        adoption.contract_census(elsewhere)


# --- A5: S3 ---------------------------------------------------------------------------------------

def test_s3_holds_with_the_column_table_the_seed_and_the_leases(adopted):
    s3 = adopted["report"]["s3"]
    assert s3["column_additions"] == len(adoption.COLUMN_ADDITIONS) and s3["leases_emptied"] == 2
    assert _rows(adopted["work"] / "data" / "reminders.db", "SELECT scope FROM reminders") == [("private",)]
    assert _rows(adopted["work"] / "data" / "memory_privacy.db",
                 "SELECT owner_namespace FROM live_deliveries") == [("legacy-unbound",)]
    assert adopted["report"]["contracts"] == len(adoption.sqlite_files(adopted["work"]))


async def _r1_fresh_schema(root: Path, monkeypatch) -> dict:
    """R1's own schema for the production store paths: the effective main factory plus the stores it leaves closed."""
    import src.vectordb.chroma_sync as index_module
    from src.ambient.store import AmbientStore
    from src.analytics.event_logger import EventLogger
    from src.analytics.suggestion_store import SuggestionStore
    from src.application import ApplicationFactories
    from src.main import build_application
    from tests.mocks.application_graph import OfflineEncoder, OfflineSTT, OfflineTTS, offline_config, offline_home

    monkeypatch.chdir(root)
    monkeypatch.setattr(index_module, "SentenceTransformer", lambda *args, **kwargs: OfflineEncoder())
    stt, tts = OfflineSTT(), OfflineTTS()
    leaves = ApplicationFactories(stt=lambda _: stt, tts=lambda _: tts, preflight=lambda: None)
    async with offline_home() as (url, _observations):
        config = offline_config(root, url)
        config["memory"].update(privacy_db_path="./data/memory_privacy.db", chroma_path="./data/memory_db")
        for name in ("lists", "reminders", "routines"):
            config[name]["db_path"] = "./data/" + name + ".db"
        config["analytics"] = {"enabled": False, "events_db": "./data/events.db",
                               "suggestions_path": "./data/suggestions.json"}
        config["ambient"] = {"enabled": False, "db_path": "./data/ambient.db"}
        config["monitoring"] = {"enabled": True, "db_path": "./data/latency.db"}
        config["observability"] = {"metrics_path": "./data/obs/kza-metrics.jsonl"}
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=leaves)
            events = EventLogger("./data/events.db", source_policy=graph.privacy.ledger)
            await asyncio.to_thread(events.initialize)
            SuggestionStore(events, "./data/suggestions.json")
            ambient = AmbientStore("./data/ambient.db")
            await ambient.init()
            await ambient.close()
            await events.aclose()
    return _schema(root)


def _schema(root: Path) -> dict:
    result = {}
    for path in adoption.sqlite_files(root):
        tables = {}
        with closing(sqlite3.connect(path)) as db:  # test-owned copies only (a WAL needs a writer)
            for (table,) in db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
                tables[table] = [(row[1], row[4]) for row in db.execute('PRAGMA table_info("' + table + '")')]
        result[path.relative_to(root).as_posix()] = tables
    return result


def _generated_diff(legacy: dict, fresh: dict) -> tuple[set, set, dict]:
    """Column additions (with their stored default), tables rebuilt without a default, tables added."""
    additions, rebuilt, added = set(), set(), {}
    for relative, tables in legacy.items():
        if relative not in fresh:
            continue
        for table, columns in fresh[relative].items():
            if table in ("sqlite_sequence",):
                continue
            if table not in tables:
                added.setdefault(relative, set()).add(table)
                continue
            old = [name for name, _default in tables[table]]
            for name, default in columns:
                if name not in old:
                    if default is None and table in {item[1] for item in adoption.REBUILT_LEASES}:
                        rebuilt.add((relative, table, name))
                    else:
                        additions.add((relative, table, name,
                                       None if default is None else default.strip("'")))
    return additions, rebuilt, added


async def test_the_hand_written_table_equals_the_generated_b0866e8_vs_r1_diff(adopted, tmp_path, monkeypatch):
    """N11: the build generates the diff and fails if it differs from the hand-written table."""
    fresh = await _r1_fresh_schema(tmp_path.resolve(), monkeypatch)
    legacy = _schema(adopted["snapshot"])
    legacy.pop("data/audit.db")
    for relative in list(legacy):
        if "chroma" in relative:
            legacy.pop(relative)
    additions, rebuilt, added = _generated_diff(legacy, fresh)
    assert additions == set(adoption.COLUMN_ADDITIONS)
    assert rebuilt == {(db, table, column) for db, table, columns in adoption.REBUILT_LEASES for column in columns}
    # Tables R1 adds lazily at its first start (not by the schema-only open) are not in ADDITIVE_TABLES.
    lazy = {"capture_browser_recorders", "capture_runtime_control"}
    assert {db: tables - lazy for db, tables in added.items()} == {
        db: set(tables) for db, tables in adoption.ADDITIVE_TABLES.items()}


def test_an_unlisted_column_drift_refuses(tmp_path, tree, monkeypatch):
    """Mutant guard: a schema change missing from the hand-written table must not pass S3."""
    from src.core import adoption_schema

    layout, root = _variant(tmp_path, tree)
    monkeypatch.setattr(adoption, "COLUMN_ADDITIONS", tuple(item for item in adoption.COLUMN_ADDITIONS
                                                             if item[1] != "reminders"))
    with pytest.raises(adoption.AdoptionRefusal, match="runtime_admission_adoption_s3:data/reminders.db:reminders"):
        _adopt(layout, root)
    assert adoption_schema.initialize_reminder_schema in adoption_schema.SCHEMA_INITIALIZERS


def test_a_changed_protected_row_refuses(tmp_path, tree, monkeypatch):
    layout, root = _variant(tmp_path, tree)
    original = adoption.run_child

    def tamper(python, runtime_root, state, phase, **options):
        result = original(python, runtime_root, state, phase, **options)
        if phase == "seed":
            with closing(sqlite3.connect(state / "data" / "lists.db")) as db, db:
                db.execute("INSERT INTO lists(id,name,owner_type,owner_id,created_at,updated_at)"
                           " VALUES ('injected','x','user','user_a',1,1)")
        return result
    monkeypatch.setattr(adoption, "run_child", tamper)
    with pytest.raises(adoption.AdoptionRefusal, match="runtime_admission_adoption_s3:data/lists.db:lists"):
        _adopt(layout, root)


# --- A6: the relocator ----------------------------------------------------------------------------

def test_no_legacy_string_remains_and_rows_are_kept(adopted):
    work, layout = adopted["work"], adopted["layout"]
    assert adoption.legacy_strings(work, layout) == []
    after = adopted["report"]["census_after"]
    assert after["legacy_state"] == after["container"] == 0
    assert adopted["report"]["census_before"]["legacy_state"] > 0
    state_root = str(adopted["state_root"])
    for target in adoption.registered_targets(work):
        for value in stage._descriptor_paths(target["descriptor"]):
            assert value.startswith(state_root + "/"), target["participant"]
    [(audio,)] = _rows(work / "data" / "ambient.db", "SELECT audio_path FROM utterances")
    assert audio == state_root + "/data/ambient_audio/policy_v1/live.flac"
    clip = _rows(work / "data" / "memory_privacy.db", "SELECT target_id FROM wake_clips_v1")[0][0]
    assert clip in {target["id"] for target in adoption.registered_targets(work)}


def test_the_relocated_targets_equal_r1s_first_start_registration(adopted):
    assert adopted["report"]["targets"] == {"registered": 25, "added_at_first_start": 1}


async def test_first_start_targets_mirror_the_main_factory(tmp_path, monkeypatch):
    """The R1 builder mirror equals what build_application actually registers for the same config."""
    import src.vectordb.chroma_sync as index_module
    from src.application import ApplicationFactories
    from src.core.adoption_schema import first_start_targets
    from src.main import build_application
    from tests.mocks.application_graph import OfflineEncoder, OfflineSTT, OfflineTTS, offline_config, offline_home

    root = tmp_path.resolve()
    monkeypatch.chdir(root)
    monkeypatch.setattr(index_module, "SentenceTransformer", lambda *args, **kwargs: OfflineEncoder())
    stt, tts = OfflineSTT(), OfflineTTS()
    leaves = ApplicationFactories(stt=lambda _: stt, tts=lambda _: tts, preflight=lambda: None)
    async with offline_home() as (url, _observations):
        config = offline_config(root, url)
        config["memory"].update(privacy_db_path="./data/memory_privacy.db", chroma_path="./data/memory_db")
        config["analytics"] = {"enabled": False, "events_db": "./data/events.db",
                               "suggestions_path": "./data/suggestions.json"}
        config["ambient"] = {"enabled": False, "db_path": "./data/ambient.db",
                             "keep_audio": {"dir": "./data/ambient_audio"}}
        config["observability"] = {"metrics_path": "./data/obs/kza-metrics.jsonl"}
        config["training"] = {"nightly": {"enabled": False, "output_dir": "./data/training_outputs/nightly",
                                          "data_dir": "./data/nightly_training",
                                          "whisper_finetune": {"output_dir": "./data/training_outputs/whisper"}},
                              "lora": {"output_dir": "./data/training_outputs/lora"}}
        async with AsyncExitStack() as stack:
            await build_application(config, shutdown=stack, factories=leaves)
        registered = [{"participant": row[0], "version": row[1], "descriptor": json.loads(row[2])}
                      for row in _rows(root / "data" / "memory_privacy.db",
                                       "SELECT participant, version, descriptor FROM erasure_targets")]
    key = lambda item: json.dumps(item, sort_keys=True)  # noqa: E731
    assert sorted(map(key, registered)) == sorted(map(key, first_start_targets(config, root)))


def test_an_extra_or_missing_target_refuses(adopted):
    work, state_root = adopted["work"], adopted["state_root"]
    expected = [{key: item[key] for key in ("participant", "version", "descriptor")}
                for item in adoption.registered_targets(work)]
    assert adoption.check_target_set(work, expected, state_root)["added_at_first_start"] == 0
    with pytest.raises(adoption.AdoptionRefusal, match="target_set:extra:events:1"):
        adoption.check_target_set(work, [item for item in expected if item["participant"] != "events"], state_root)
    missing = expected + [{"participant": "streams", "version": 5, "descriptor": {"kind": "new"}}]
    with pytest.raises(adoption.AdoptionRefusal, match="target_set:missing:streams:5"):
        adoption.check_target_set(work, missing, state_root)


def test_a_dropped_row_during_relocation_refuses(tmp_path, tree, monkeypatch):
    """Rows are never cut: a relocation that loses one row refuses."""
    layout, root = _variant(tmp_path, tree)
    original = adoption._canonical_like

    def lose_a_row(text, value):
        with closing(sqlite3.connect(root / "work" / "data" / "lists.db")) as db, db:
            db.execute("DELETE FROM lists WHERE rowid IN (SELECT rowid FROM lists LIMIT 1)")
        return original(text, value)
    monkeypatch.setattr(adoption, "_canonical_like", lose_a_row)
    with closing(sqlite3.connect(layout.legacy_state / "data" / "lists.db")) as db:
        db.execute("INSERT INTO lists(id,name,owner_type,owner_id,created_at,updated_at) VALUES"
                   " ('kept','compras','user','user_a',1,1)")
        db.commit()
    with pytest.raises(adoption.AdoptionRefusal, match="^runtime_admission_adoption_relocation_rows$"):
        _adopt(layout, root)


def test_a_legacy_path_outside_the_remap_list_refuses_as_incomplete(tmp_path, tree):
    """Completeness: a legacy locator in a column the REMAP list does not name is never left behind."""
    def note(data):
        with closing(sqlite3.connect(data / "lists.db")) as db, db:
            db.execute("INSERT INTO lists(id,name,owner_type,owner_id,created_at,updated_at) VALUES"
                       " ('stray',?,'user','user_a',1,1)", (str(data / "ambient_audio" / "x.flac"),))
    layout, root = _variant(tmp_path, tree, note)
    with pytest.raises(adoption.AdoptionRefusal, match="^runtime_admission_adoption_relocation_incomplete:"):
        _adopt(layout, root)


def test_a_legacy_path_in_env_refuses_by_key(tmp_path, tree):
    layout, root = _variant(tmp_path, tree)
    (layout.legacy_state / ".env").write_text("CHROMA_PATH=" + str(layout.legacy_state / "data" / "x") + "\n")
    with pytest.raises(adoption.AdoptionRefusal, match="^runtime_admission_adoption_env_path:CHROMA_PATH$"):
        _adopt(layout, root)


def test_a_context_locator_is_relocated(tmp_path, tree):
    def locator(data):
        (data / "contexts" / "policy_v3" / "user_b.json").write_text(
            json.dumps({"user_id": "user_b", "audio": str(data / "ambient_audio" / "x.flac")}))
    layout, root = _variant(tmp_path, tree, locator)
    report = _adopt(layout, root)
    assert report["relocation"]["changed"]["contexts:*.json"] == 1
    value = json.loads((root / "work" / "data" / "contexts" / "policy_v3" / "user_b.json").read_text())
    assert value["audio"] == str(root / "slot" / "state" / "data" / "ambient_audio" / "x.flac")


# --- A7: legacy_training (P1=D) --------------------------------------------------------------------

def test_adapters_under_a_lora_root_refuse_with_or_without_a_record(tmp_path, tree):
    layout, root = _variant(tmp_path, tree, adapters=True)
    with pytest.raises(adoption.AdoptionRefusal, match="^runtime_admission_adoption_legacy_training_present$"):
        _adopt(layout, root)
    record = {"schema_version": 1, "decision_id": "P1-D",
              "roots": [{"root": str(layout.models_root.resolve() / "lora_adapters"), "absence": "enoent"}]}
    with pytest.raises(adoption.AdoptionRefusal, match="^runtime_admission_adoption_legacy_training_present$"):
        adoption.legacy_training_disposition(layout, [str(layout.models_root / "lora_adapters")], record)


def test_retired_roots_are_relocated_and_cite_the_record(tmp_path, tree):
    layout, root = _variant(tmp_path, tree, adapters=True)
    shutil.rmtree(layout.models_root / "lora_adapters")  # what retire-legacy-training (0e3) leaves
    record = {"schema_version": 1, "decision_id": "user-decisions-20261110/P1=D",
              "roots": [{"root": str(layout.models_root.resolve() / "lora_adapters"), "absence": "enoent"}]}
    report = _adopt(layout, root, retirement=record)
    assert report["legacy_training"] == {"retirement": "user-decisions-20261110/P1=D", "roots": 3}
    roots = {target["descriptor"]["root"] for target in adoption.registered_targets(root / "work")
             if target["descriptor"].get("kind") == "legacy_training"}
    state = str(root / "slot" / "state")
    assert {state + "/data/training_outputs/" + name for name in ("lora", "nightly", "whisper")} <= roots


def test_absent_roots_are_relocated_without_a_record(adopted):
    assert adopted["report"]["legacy_training"] == {"retirement": "not_needed", "roots": 3}


# --- A8: the ledger seed ---------------------------------------------------------------------------

def test_the_seed_per_kind_equals_the_plan_and_facts_carry_the_adoption_time(adopted):
    work, snapshot = adopted["work"], adopted["snapshot"] / "data" / "memory_privacy.db"
    plan = {"conversation_erased": _rows(snapshot, "SELECT COUNT(*) FROM conversation_tombstones")[0][0],
            "fact_deleted": _rows(snapshot, "SELECT COUNT(*) FROM facts WHERE state='deleted'")[0][0],
            "voice_profile_deleted": _rows(snapshot, "SELECT COUNT(*) FROM voice_profiles WHERE deleted_at IS NOT NULL")[0][0]}
    assert adopted["report"]["seed"] == {"data/memory_privacy.db": plan}
    privacy = work / "data" / "memory_privacy.db"
    [(salt,)] = _rows(privacy, "SELECT salt FROM deletion_ledger_salt_v1")
    facts = {row[0] for row in _rows(snapshot, "SELECT id FROM facts WHERE state='deleted'")}
    rows = _rows(privacy, "SELECT kind, subject_sha256, at FROM deletion_ledger_v1 ORDER BY seq")
    digest = {hmac.new(bytes.fromhex(salt), fact.encode(), hashlib.sha256).hexdigest() for fact in facts}
    assert {row[1] for row in rows if row[0] == "fact_deleted"} == digest
    assert {row[2] for row in rows if row[0] == "fact_deleted"} == {AT}
    assert [row[0] for row in rows] == sorted((row[0] for row in rows), key=[
        "conversation_erased", "fact_deleted", "voice_profile_deleted", "user_retired"].index)
    assert launcher.resurrection_counts(work, descriptor_root=adopted["state_root"]) == {
        "conversation": 0, "fact": 0, "voice_profile": 0, "voice_consent": 0}


# --- A9: residue (P3=A) ----------------------------------------------------------------------------

def _restore_event(data):
    with closing(sqlite3.connect(data / "events.db")) as db, db:
        db.execute("INSERT INTO events_policy_v1(timestamp,event_type,entity_id,action,sources_json,metrics_json,"
                   "expires_at) VALUES (1,'a','b','c',?,'{}',9e9)", (json.dumps([{"conversation_id": ERASED}]),))


def test_a_restored_conversation_payload_reopens_its_governing_step(tmp_path, tree):
    layout, root = _variant(tmp_path, tree, _restore_event)
    report = _adopt(layout, root)
    assert report["residue"] == {"conversation_operations_created": 0, "conversation_steps_reopened": 1,
                                 "facts_requeued": 0}
    assert report["s3"]["steps_reopened"] == 1
    events = [target["id"] for target in adoption.registered_targets(root / "work") if target["participant"] == "events"]
    assert _rows(root / "work" / "data" / "memory_privacy.db",
                 "SELECT state, last_error FROM erasure_steps WHERE target_id=?", (events[0],)) == [
        ("pending", "interrupted_by_restart")]


def test_a_tombstone_without_an_operation_gets_one_through_the_journal(tmp_path, tree):
    def restore(data):
        with closing(sqlite3.connect(data / "memory_privacy.db")) as db, db:
            db.execute("INSERT INTO conversation_tombstones VALUES (?,?,?)", (NO_OPERATION, 1_799_999_000.0, "legacy"))
        with closing(sqlite3.connect(data / "ambient.db")) as db, db:
            db.execute("INSERT INTO utterances(room_id,t0,t1,text,source,conversation_id,created_at)"
                       " VALUES ('hall',1,2,'t','ambient',?,1)", (NO_OPERATION,))
    layout, root = _variant(tmp_path, tree, restore)
    report = _adopt(layout, root)
    assert report["residue"]["conversation_operations_created"] == 1
    assert _rows(root / "work" / "data" / "memory_privacy.db",
                 "SELECT requested_by, requested_at FROM conversation_erasures WHERE conversation_id=?",
                 (NO_OPERATION,)) == [("adoption:" + PLAN, AT)]


@pytest.mark.parametrize("queue", ["acknowledged", "never_queued"])
def test_a_restored_deleted_fact_is_requeued(tmp_path, tree, queue):
    def restore(data):
        with closing(sqlite3.connect(data / "memory_privacy.db")) as db, db:
            db.execute("DELETE FROM index_jobs")  # the index deletion had been acknowledged
            if queue == "never_queued":  # no destination row: only the journal's enqueue creates it
                db.execute("DELETE FROM policy_index_deletions")
    layout, root = _variant(tmp_path, tree, restore)
    report = _adopt(layout, root)
    deleted = fixture.report_of(root / "legacy")["facts"]["deleted"]
    assert report["residue"]["facts_requeued"] == 1
    assert _rows(root / "work" / "data" / "memory_privacy.db",
                 "SELECT state, last_error FROM policy_index_deletions WHERE fact_id=?", (deleted,)) == [
        ("pending", "index_startup_recheck" if queue == "acknowledged" else None)]  # enqueue writes a fresh row


@pytest.mark.parametrize("variant, code", [("voice", "voice_profile"), ("orphan_turn", "conversation")])
def test_an_unmarkable_residue_aborts(tmp_path, tree, variant, code):
    def restore(data):
        with closing(sqlite3.connect(data / "memory_privacy.db")) as db, db:
            if variant == "voice":
                db.execute("INSERT INTO voice_fingerprints(id,model_key,dimensions,embedding,duration_s,created_at,"
                           "expires_at,sources_json,suggested_profile,suggested_revision,similarity)"
                           " VALUES ('fp','ecapa',2,x'00',1,1,9e9,'[]','voice_deleted_profile',1,.9)")
            else:
                db.execute("INSERT INTO conversation_turns_v1 VALUES ('orphan','g',?,'unmarked','d')",
                           (json.dumps([{"conversation_id": ERASED}]),))
    layout, root = _variant(tmp_path, tree, restore)
    with pytest.raises(adoption.AdoptionRefusal, match="^runtime_admission_adoption_residue:" + code + "$"):
        _adopt(layout, root)
