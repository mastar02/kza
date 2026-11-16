"""SC-4c: Chroma's closed collection list in owner compaction (sc-release-design v4 implementation_units SC-4c).

mechanism.start.chroma_allowlist (PRIV-3, PA-8, V4R2-C1): the policy memory
collection is rebuilt from the allowlist of facts get() would return at
t_clone whose owner is not retired (ledger and tombstones as a second
gate); the command-index collections are copied whole; home_assistant_routines
is not copied (a named drop, K's boot recreates it empty); any other
collection, the legacy 'user_memories' included, refuses at plan and in the
worker; D5 for rebuilt stores. Crash row S (a worker refusal) exits by a
plain abort; a crash after the rebuild's evidence resumes through the pin.

The end-to-end worlds are the first post-R1 transition (test_sc1b_release_plan):
an epoch-4 head (R1's stand-in) whose first run writes real Chroma stores
(tests/mocks/sc4c_chroma.py), T0 to X5 (this working tree), then plan and
start in release mode, V and retire.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import time

import pytest

from scripts import compaction
from scripts import stage_runtime as stage
from src.memory import owner_compaction as oc
from src.memory.deletion_ledger import subject_digest
from src.memory.state_snapshot import ClosedStateSnapshot
from tests.mocks import compaction_world as cw
from tests.mocks import launcher_rehearsal as rehearsal
from tests.mocks import sc4c_chroma as sc
from tests.mocks.epoch4 import epoch4_in_process
from tests.unit.tools.test_sc1b_release_plan import _install, _refusal, clone, x4, x5  # noqa: F401

UNLISTED = "runtime_admission_compaction_unlisted_collection"
NOW = 1_800_000_000.0
SALT = "5a" * 32


@pytest.fixture(autouse=True)
def checkout_controller(monkeypatch):
    """P7's installed-controller location cannot hold for an in-process controller."""
    monkeypatch.setattr(compaction, "controller_location", lambda root, members: None)


# --- the closed list ---------------------------------------------------------------------------------

NAMES = ["home_assistant_commands", "home_assistant_commands__pointer", sc.STAGING, "home_assistant_routines",
         "user_memories", "user_memories_policy_v1", "home_assistant_command", "home_assistant_routines_v2",
         "code_chunks"]


def _worker_kind(name: str, policy: bool) -> str | None:
    try:
        return oc.collection_kind(name, policy=policy)
    except oc.CompactionCloneError as error:
        assert str(error) == UNLISTED
        return None


def test_the_closed_list_is_the_same_at_plan_and_in_the_worker():
    assert compaction.CHROMA_COMMAND_INDEX == oc.CHROMA_COMMAND_INDEX == "home_assistant_commands"
    assert compaction.CHROMA_DROPPED == oc.CHROMA_DROPPED == frozenset({"home_assistant_routines"})
    expected = {"home_assistant_commands": "copied", "home_assistant_commands__pointer": "copied",
                sc.STAGING: "copied", "home_assistant_routines": "dropped"}
    for name in NAMES:
        assert compaction.collection_kind(name, policy=False) == _worker_kind(name, False) == expected.get(name)
        assert compaction.collection_kind(name, policy=True) == _worker_kind(name, True) == "rebuilt"
    assert oc._snapshot_class("data/chroma_db", "home_assistant_routines", False) == "omit"
    assert oc._snapshot_class("data/chroma_db", sc.STAGING, False) == "export"
    _refusal(UNLISTED, lambda: oc._snapshot_class("data/memory_db", "user_memories", False))


# --- the policy allowlist on snapshot images ---------------------------------------------------------

def _record(identity: str, document: str, revision: int = 1, metadata: dict | None = None) -> dict:
    return {"id": identity, "document": document, "embedding": [0.5, 0.25],
            "metadata": metadata if metadata is not None else {"fact_policy": 1, "revision": revision}}


def _fact(identity: str, content: str, *, state: str = "automatic", expires: float | None = NOW + 60,
          revision: int = 1) -> dict:
    return {"id": identity, "content": content, "category": "fact", "confidence": .9, "state": state,
            "extracted_at": NOW - 600, "expires_at": expires, "revision": revision,
            "confirmed_at": None, "confirmed_by": None}


def _source(fact_id: str, conversation: str) -> dict:
    return {"fact_id": fact_id, "capture_id": "cap-" + fact_id, "conversation_id": conversation,
            "captured_at": NOW - 600, "expires_at": NOW + 600, "origin": "voice", "dependencies": "[]"}


def _snapshot(exports: list[dict], *, stores=("data/chroma_db", "data/memory_db"), listed: bool = True):
    facts = [_fact("f-live", "live"), _fact("f-confirmed", "confirmed", state="confirmed", expires=None),
             _fact("f-overdue", "overdue", expires=NOW - 1), _fact("f-retired", "retired owner", state="confirmed",
             expires=None), _fact("f-tombstoned", "tombstoned source"), _fact("f-ledger", "ledger recorded"),
             _fact("f-proposed", "proposed", state="proposed"), _fact("f-deleted", None, state="deleted"),
             _fact("f-stale", "stale", revision=3), _fact("f-extra", "extra metadata")]
    privacy = {"tables": {
        "facts": {"rows": facts},
        "fact_sources": {"rows": [_source(fact["id"], "conv-" + fact["id"]) for fact in facts]},
        "conversation_tombstones": {"rows": [{"conversation_id": "conv-f-tombstoned", "deleted_at": NOW - 9,
                                              "deleted_by": "x"}]},
        "fact_owners": {"rows": [{"fact_id": "f-retired", "person_id": sc.PERSON},
                                 {"fact_id": "f-confirmed", "person_id": "user_" + "b" * 32}]},
        "deletion_ledger_salt_v1": {"rows": [{"id": 1, "salt": SALT}]},
        "deletion_ledger_v1": {"rows": [{"seq": 1, "kind": "fact_deleted",
                                         "subject_sha256": subject_digest(SALT, "f-ledger")}]}}}
    auth = {"tables": {"retired_users": {"rows": [{"user_id": sc.PERSON, "retired_at": NOW - 9}]}}}
    manifest = {"databases": {"data/privacy.db": privacy, "data/auth.db": auth}, "index_exports": exports}
    if listed:
        manifest["sdk_collections"] = {"schema_version": 1, "stores": list(stores)}
    return ClosedStateSnapshot(Path("/nonexistent-sc4c"), "0" * 64, {}, manifest)


def _export(store: str, name: str, records: list[dict], *, registered: bool = False, omitted: bool = False):
    return {"relative_path": store, "collection": name, "registered": registered, "omitted": omitted,
            "metadata": None if omitted else {"hnsw:space": "cosine"}, "configuration": None,
            "records": [] if omitted else records}


COMMAND = {"id": "light.x_0", "document": "cmd", "embedding": [0.5, 0.25], "metadata": {"entity_id": "light.x"}}
POLICY_RECORDS = [_record("f-live", "live"), _record("f-confirmed", "confirmed"), _record("f-overdue", "overdue"),
                  _record("f-retired", "retired owner"), _record("f-tombstoned", "tombstoned source"),
                  _record("f-ledger", "ledger recorded"), _record("f-proposed", "proposed"),
                  _record("f-deleted", "deleted residue"), _record("f-stale", "stale", revision=2),
                  _record("f-orphan", "orphan"),
                  _record("f-extra", "extra metadata", metadata={"fact_policy": 1, "revision": 1, "x": 1})]


def test_the_policy_collection_keeps_only_facts_get_returns_with_a_live_owner():
    """Canaries: overdue fact, retired-owner fact not pruned, orphan id; and every other exclusion."""
    snapshot = _snapshot([_export("data/memory_db", sc.POLICY, POLICY_RECORDS, registered=True),
                          _export("data/chroma_db", sc.STAGING, [COMMAND])])
    assert set(oc.policy_allowlist(snapshot, NOW)) == {"f-live", "f-confirmed", "f-stale", "f-extra"}
    stores, plans = oc.chroma_plan(snapshot, NOW)
    assert stores == ("data/chroma_db", "data/memory_db")
    by_name = {item.collection: item for item in plans}
    policy = by_name[sc.POLICY]
    assert policy.kind == "rebuilt" and [row["id"] for row in policy.records] == ["f-live", "f-confirmed"]
    assert policy.excluded == len(POLICY_RECORDS) - 2
    staging = by_name[sc.STAGING]
    assert staging.kind == "copied" and list(staging.records) == [COMMAND] and staging.excluded == 0
    summary = json.dumps([item.summary() for item in plans])
    assert "live" not in summary and "cmd" not in summary  # content-free: counts and digests only


def test_the_plan_refuses_an_unlisted_or_unclassified_export():
    routines = _export("data/chroma_db", "home_assistant_routines", [], omitted=True)
    stores, plans = oc.chroma_plan(_snapshot([routines]), NOW)
    assert [(item.kind, item.records) for item in plans] == [("dropped", ())]
    for exports, listed in (
            ([_export("data/memory_db", "user_memories", [_record("x", "legacy")])], True),
            ([_export("data/memory_db", sc.POLICY, POLICY_RECORDS)], True),  # a policy name H never registered
            ([_export("data/chroma_db", "home_assistant_routines", [_record("r", "routine")])], True),  # read
            ([dict(_export("data/chroma_db", sc.STAGING, []), omitted=True)], True),
            ([_export("data/other_db", sc.STAGING, [])], True),  # not one of the listed stores
            ([], False)):  # a snapshot taken without the closed list
        _refusal(UNLISTED, lambda: oc.chroma_plan(_snapshot(exports, listed=listed), NOW))


# --- end to end: the first post-R1 transition ------------------------------------------------------

def _post_r1(x4: dict, x5: dict, tmp_path: Path, *first: str) -> dict:
    """An epoch-4 head whose first run wrote ``first`` (R1's stand-in), then T0 to X5."""
    with epoch4_in_process():
        world = cw.world(x4["archive"], tmp_path, hops=(), first=first)
    _install(world["root"], x5)
    world["h_head"] = stage.read_admission_head(world["root"])
    return world


def _stores(state: Path, scratch: Path) -> dict:
    """(store, collection) -> metadata, configuration and records, read from a copy (never the original)."""
    import chromadb
    from chromadb.config import Settings
    from src.memory.state_snapshot import _sdk_json, collection_records
    found = {}
    for store in ("data/chroma_db", "data/memory_db"):
        copy = scratch / store
        shutil.copytree(state / store, copy)
        client = chromadb.PersistentClient(path=str(copy), settings=Settings(anonymized_telemetry=False))
        try:
            for item in client.list_collections():
                collection = client.get_collection(name=item.name, embedding_function=None)
                found[(store, item.name)] = {"metadata": _sdk_json(collection.metadata),
                                             "configuration": _sdk_json(collection.configuration_json),
                                             "records": collection_records(collection)}
        finally:
            client.close()
    return found


def _raw_hits(root: Path, needle: str) -> set:
    return {path.relative_to(root).as_posix() for path in root.rglob("*")
            if path.is_file() and not path.is_symlink() and needle.encode() in path.read_bytes()}


EXCLUDED = (sc.OVERDUE, sc.RETIRED, sc.TOMBSTONED, sc.ORPHAN, sc.ROUTINE, sc.ROUTINE_GONE)


def test_the_first_post_r1_transition_rebuilds_chroma_from_the_closed_list(x4, x5, tmp_path, monkeypatch):
    world = _post_r1(x4, x5, tmp_path, sc.chroma_state())
    root, h_state = world["root"], world["h"]["state"]
    before = _stores(h_state, tmp_path / "h-copy")
    assert {key: len(value["records"]) for key, value in before.items()} == {
        ("data/chroma_db", "home_assistant_commands"): 1, ("data/chroma_db", sc.STAGING): 2,
        ("data/chroma_db", "home_assistant_commands__pointer"): 0, ("data/chroma_db", "home_assistant_routines"): 2,
        ("data/memory_db", sc.POLICY): 5}
    planned = cw.plan(root, x5["archive"])
    assert planned["chroma"] == {"stores": {"data/chroma_db": {"rebuilt": 0, "copied": 3, "dropped": 1},
                                            "data/memory_db": {"rebuilt": 1, "copied": 0, "dropped": 0}}}
    started = cw.start(root, x5["archive"], planned=planned)
    compaction_id = started["compaction_id"]
    log = stage.read_compaction_log(root, compaction_id)
    [materialized] = [event for event in log["events"] if event["kind"] == "materialized"]
    assert materialized["data"]["chroma"] == {"stores": 2, "rebuilt": 1, "copied": 3, "dropped": 1,
                                              "records": 4, "excluded": 4}
    k_state = root / "slots" / "k" / "state"
    after = _stores(k_state, tmp_path / "k-copy")
    # Copied whole: the base, the pointer (its 'active' metadata) and the staging collection.
    for name in ("home_assistant_commands", "home_assistant_commands__pointer", sc.STAGING):
        assert after[("data/chroma_db", name)] == before[("data/chroma_db", name)]
    assert after[("data/chroma_db", "home_assistant_commands__pointer")]["metadata"] == {"active": sc.STAGING}
    # Rebuilt from the allowlist: only the live fact, same collection metadata and configuration.
    policy, source = after[("data/memory_db", sc.POLICY)], before[("data/memory_db", sc.POLICY)]
    assert [row["document"] for row in policy["records"]] == [sc.LIVE]
    assert policy["records"] == [row for row in source["records"] if row["document"] == sc.LIVE]
    assert (policy["metadata"], policy["configuration"]) == (source["metadata"], source["configuration"])
    # Not copied: home_assistant_routines; nothing excluded survives in K's bytes.
    assert set(after) == set(before) - {("data/chroma_db", "home_assistant_routines")}
    for needle in EXCLUDED:
        assert _raw_hits(k_state / "data" / "chroma_db", needle) == set(), needle
        assert _raw_hits(k_state / "data" / "memory_db", needle) == set(), needle
    assert _raw_hits(h_state / "data" / "chroma_db", sc.ROUTINE)  # H keeps it until H is retired
    # Content-free records: the log, the authority record and K's preparation evidence.
    for path in [*stage.compaction_log_path(root, compaction_id).rglob("*"), k_state / ".kza-payload-preparation.json"]:
        if path.is_file():
            assert sc.CANARY not in path.read_text(), path
    # K's boot (ChromaSync._construct, with a stub embedder) recreates the routines collection empty and
    # resolves the command index through the copied pointer.
    from src.vectordb import chroma_sync
    boot = tmp_path / "k-boot"
    shutil.copytree(k_state / "data" / "chroma_db", boot)
    monkeypatch.setattr(chroma_sync, "SentenceTransformer", lambda *args, **kwargs: object())
    sync = chroma_sync.ChromaSync(str(boot), "stub", "cpu")
    sync._construct()
    try:
        assert sync._routines_collection.count() == 0
        assert sync._commands_collection.name == sc.STAGING and sync._index_degraded is None
    finally:
        sync._client.close()
    # V, then retire (strict W equality and the resurrection scan over K''s rebuilt index), then K' serves.
    assert compaction.compact_verify(root, compaction_id, smoke_seconds=10)["armed"] is True
    phase_file = tmp_path / "phases-v"
    origin = time.monotonic()
    result = rehearsal.run_serve(root, rehearsal.child_command(phase_file, "verify"), phase_file=phase_file,
                                 stop_grace=200.0, timeout=60.0,
                                 clock=lambda: origin + (time.monotonic() - origin) * 20.0)
    assert result.code == 0
    assert compaction.compact_retire(root, compaction_id, scan=cw.no_units)["phase"] == "complete"
    assert not h_state.exists()
    assert cw.serve(root, tmp_path, "k-prime").code == 0


def test_a_legacy_user_memories_collection_refuses_the_plan_and_writes_nothing(x4, x5, tmp_path):
    """PA-8, crash row P: the refused plan writes nothing and the unit serves H."""
    world = _post_r1(x4, x5, tmp_path, sc.chroma_state(legacy=True))
    root = world["root"]
    witness = cw.witness(root)
    error = _refusal(UNLISTED, lambda: cw.plan(root, x5["archive"]))
    assert error.report == {"collections": [["data/memory_db", "user_memories"]]}
    _refusal(UNLISTED, lambda: cw.start(root, x5["archive"], planned={"plan_sha256": "0" * 64,
                                                                      "untouched_sha256": "0" * 64}))
    assert cw.witness(root) == witness
    assert cw.serve(root, tmp_path, "h").code == 0


def test_the_worker_refuses_an_unlisted_collection_the_plan_missed(x4, x5, tmp_path, monkeypatch):
    """The worker's own gate (snapshot and plan): crash row S, then the lossless plain abort serves H."""
    world = _post_r1(x4, x5, tmp_path, sc.chroma_state(legacy=True))
    root, h_state = world["root"], world["h"]["state"]
    h_tree = cw.tree(h_state)
    monkeypatch.setattr(compaction, "chroma_collections", lambda state, privacy: {"stores": {}})
    _refusal(UNLISTED, lambda: cw.start(root, x5["archive"]))
    [log] = stage.compaction_logs(root)
    kinds = [event["kind"] for event in log["events"]]
    assert "snapshot_reserved" in kinds and "materializing" not in kinds
    k_state = root / "slots" / "k" / "state"
    assert not k_state.exists() or not any(k_state.rglob("chroma.sqlite3"))
    monkeypatch.undo()
    monkeypatch.setattr(compaction, "controller_location", lambda root, members: None)
    assert compaction.compact_abort(root, log["compaction_id"])["phase"] == "aborted"
    head = stage.read_admission_head(root)
    assert head["status"] == "closed" and head["binding_id"] == world["h_head"]["binding_id"]
    assert cw.tree(h_state) == h_tree and not (root / "slots" / "k").exists()
    assert cw.serve(root, tmp_path, "h").code == 0


def _in_process_worker(**options):
    def worker(root_, compaction_id):
        oc.run_materialization(root_, compaction_id, wall=time.time, **options)
    return worker


def test_d5_refuses_a_rebuilt_store_changed_after_its_check(x4, x5, tmp_path):
    """D5 for rebuilt stores: a file added to K's rebuilt store before VACUUM fails the second D5."""
    world = _post_r1(x4, x5, tmp_path, sc.chroma_state())
    root = world["root"]

    def hook():
        (root / "slots" / "k" / "state" / "data" / "chroma_db" / "extra.bin").write_bytes(b"x")
    _refusal("runtime_admission_compaction_payload_changed:chroma",
             lambda: cw.start(root, x5["archive"], worker=_in_process_worker(before_vacuum=hook)))
    [log] = stage.compaction_logs(root)
    assert "materialized" not in [event["kind"] for event in log["events"]]
    assert stage.read_admission_head(root) == world["h_head"]
    assert compaction.compact_abort(root, log["compaction_id"])["phase"] == "aborted"
    assert sorted(os.listdir(root / "slots")) == ["a"]


def test_a_crash_after_the_rebuilt_evidence_resumes_through_the_pin(x4, x5, tmp_path, monkeypatch):
    """Crash row S: K materialized with its evidence but no 'materialized' event; resume re-checks the pin."""
    world = _post_r1(x4, x5, tmp_path, sc.chroma_state())
    root = world["root"]
    original = oc.materialize_compaction_clone

    def crash(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("synthetic-crash")
    monkeypatch.setattr(oc, "materialize_compaction_clone", crash)
    with pytest.raises(RuntimeError, match="synthetic-crash"):
        cw.start(root, x5["archive"], worker=_in_process_worker())
    monkeypatch.setattr(oc, "materialize_compaction_clone", original)
    [log] = stage.compaction_logs(root)
    evidence = json.loads((root / "slots" / "k" / "state" / ".kza-payload-preparation.json").read_text())
    assert set(evidence["chroma"]["pins"]) == {"data/chroma_db", "data/memory_db"}
    resumed = compaction.compact_resume(root, log["compaction_id"])
    assert resumed["phase"] == "activated"
    [materialized] = [event for event in stage.read_compaction_log(root, log["compaction_id"])["events"]
                      if event["kind"] == "materialized"]
    assert materialized["data"]["chroma"]["dropped"] == 1 and materialized["data"]["chroma"]["excluded"] == 4
