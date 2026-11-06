"""Fresh compiled-process observations for a private two-root slot rehearsal.

Domain counterexamples are data: the parent asserts them only after every phase
has closed its real graph.  This entry never changes an authority row directly.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import closing
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import sys
import time


UNIT = "T024-F1/privacy-safe-runtime-state-rollback"
SEED_KEYS = {"old_root", "clock_now", "subject", "foreign", "manual", "foreign_manifest",
    "manual_manifest", "index_collection", "index_metadata", "original_u_source",
    "old_reader_namespace", "consent", "closed_state"}
CANDIDATE_KEYS = {"seed_sha256", "candidate_root", "clock_now", "operation", "steps",
    "latest_profile_id", "latest_profile_rows", "credential_id", "latest_credential_rows",
    "latest_user", "latest_manual_manifest", "consent", "closed_state", "before_graph"}
RUNTIME_TABLES = {"live_deliveries", "live_delivery_sources", "live_delivery_lineage"}


def _require(value, code):
    if not value:
        raise AssertionError(code)


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _e_module():
    """Load only E's stdlib definitions before its own preimport bootstrap."""
    path = Path(__file__).with_name("compiled_restore_child.py")
    spec = importlib.util.spec_from_file_location("_slot_rehearsal_e_bootstrap", path)
    _require(spec is not None and spec.loader is not None, "E_bootstrap_spec_missing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _checkpoint_path(args, control):
    path = Path(args.checkpoint)
    excluded = [Path(control[key]) for key in
        ("runtime_root", "state_root", "island_root", "checkout_root")]
    _require(path.is_absolute() and path.parent.resolve() == path.parent
        and not path.is_symlink() and all(not path.is_relative_to(root) for root in excluded),
        "slot_checkpoint_location")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


def _write_bag(e, path, control, kind, payload):
    expected = SEED_KEYS if kind == "seed" else CANDIDATE_KEYS
    _require(set(payload) == expected and not path.exists(), "slot_bag_schema_or_exists")
    codec = e.DataCodec(path.parent / "slot-blobs")
    encoded = codec.encode(payload)
    _require(sum(codec.manifest.values()) <= 64 * 1024 * 1024, "slot_bag_too_large")
    return e._atomic_json(path, {"schema_version": 1, "unit": UNIT, "kind": kind,
        "source_commit": control["source_commit"], "runtime_root": control["runtime_root"],
        "state_root": control["state_root"], "release_sha256": control["release_sha256"],
        "helper_manifest_sha256": _sha(json.dumps(control["island_files"], sort_keys=True).encode()),
        "payload": encoded, "blob_manifest": codec.manifest})


def _read_bag(e, path, control, kind):
    _require(path.is_file() and not path.is_symlink()
        and path.stat().st_size <= 64 * 1024 * 1024, "slot_bag_location_or_size")
    raw = path.read_bytes()
    document = json.loads(raw)
    keys = {"schema_version", "unit", "kind", "source_commit", "runtime_root", "state_root",
        "release_sha256", "helper_manifest_sha256", "payload", "blob_manifest"}
    _require(type(document) is dict and set(document) == keys
        and document["schema_version"] == 1 and document["unit"] == UNIT
        and document["kind"] == kind and document["source_commit"] == control["source_commit"]
        and document["release_sha256"] == control["release_sha256"]
        and document["helper_manifest_sha256"] == _sha(json.dumps(
            control["island_files"], sort_keys=True).encode()), "slot_bag_identity")
    # These are the original physical identities, not targets to rewrite for a new root.
    for key in ("runtime_root", "state_root"):
        value = e._canonical(document[key])
        _require(not value.is_relative_to(Path(control["checkout_root"])), "slot_bag_checkout_path")
    manifest = document["blob_manifest"]
    _require(type(manifest) is dict and len(manifest) <= 10000
        and all(type(size) is int and 0 <= size <= 16 * 1024 * 1024 for size in manifest.values())
        and sum(manifest.values()) <= 64 * 1024 * 1024, "slot_blob_manifest")
    codec = e.DataCodec(path.parent / "slot-blobs", manifest)
    payload = codec.decode(document["payload"])
    expected = SEED_KEYS if kind == "seed" else CANDIDATE_KEYS
    _require(type(payload) is dict and set(payload) == expected
        and codec.used == set(manifest), "slot_payload_schema")
    return payload, _sha(raw)


def _quote(value):
    return '"' + value.replace('"', '""') + '"'


def _rows(path, table):
    """No connection here may create a database or acquire its writer lease."""
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), "slot_read_database_missing")
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        return [dict(row) for row in db.execute("SELECT * FROM " + _quote(table))]


def _table_subset(path, table, expected):
    with closing(sqlite3.connect(Path(path).as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        info = list(db.execute("PRAGMA table_info(" + _quote(table) + ")"))
        primary = [row["name"] for row in sorted(info, key=lambda row: row["pk"]) if row["pk"]]
        _require(primary or not expected, "slot_comparison_requires_primary_key")
        current = [dict(row) for row in db.execute("SELECT * FROM " + _quote(table))]
    selected = []
    for row in expected:
        keys = tuple(row[key] for key in primary)
        matches = [item for item in current if tuple(item[key] for key in primary) == keys]
        selected.extend(matches)
    return selected


def _same_rows(left, right):
    def token(row):
        return json.dumps(row, sort_keys=True, default=lambda item:
            {"blob_sha256": _sha(item)} if type(item) is bytes else str(item))
    return sorted(map(token, left)) == sorted(map(token, right))


def _existing_index(root, name, *, ids=None):
    import chromadb
    from chromadb.config import Settings
    from tests.mocks.application_restore import index_values
    client = chromadb.PersistentClient(path=str(Path(root) / "chroma"),
        settings=Settings(anonymized_telemetry=False))
    closed = False
    try:
        collection = client.get_collection(name=name, embedding_function=None)
        value = collection.get(ids=ids, include=["embeddings", "documents", "metadatas"])
        result = index_values(value)
    finally:
        client.close()
        closed = True
    _require(closed, "slot_read_SDK_close_not_returned")
    return result


def _project_manifest(root, original_root, manifest, collection):
    """Translate only comparison locations; stored DTOs/descriptors stay unchanged."""
    root, original_root = Path(root), Path(original_root)
    tables, files = {}, {}
    for original, selections in manifest["tables"].items():
        path = root / Path(original).relative_to(original_root)
        tables[original] = {table: _table_subset(path, table, expected)
            for table, expected in selections.items()}
    for original in manifest["files"]:
        path = root / Path(original).relative_to(original_root)
        files[original] = path.read_bytes() if path.is_file() else None
    index = _existing_index(root, collection, ids=manifest["index"]["ids"])
    return {"tables": tables, "files": files, "index": index}


def _manifest_equal(left, right):
    return (set(left["tables"]) == set(right["tables"])
        and all(set(tables) == set(right["tables"][path]) and all(
            _same_rows(rows, right["tables"][path][table]) for table, rows in tables.items())
            for path, tables in left["tables"].items())
        and left["files"] == right["files"] and left["index"] == right["index"])


def _manual_manifest(root, manual, suggestions_path):
    root = Path(root)
    tables = {}
    for table, expected in manual.rows.items():
        database = Path(suggestions_path) if table == "suggestions_policy_v1" else root / "privacy.db"
        tables.setdefault(str(database), {})[table] = _table_subset(database, table, expected)
    return tables


def _manual_equal(root, original_root, expected):
    for original, tables in expected.items():
        path = Path(root) / Path(original).relative_to(original_root)
        for table, rows in tables.items():
            if not _same_rows(rows, _table_subset(path, table, rows)):
                return False
    return True


def _consent(config):
    reasoner = config.get("reasoner", {}) or {}
    compaction = reasoner.get("compaction", {}) or {}
    return {"destination": reasoner.get("destination"), "cloud": reasoner.get("cloud", {}),
        "compaction_destination": compaction.get("destination"),
        "compaction_cloud_consent": compaction.get("cloud_consent")}


def _closed_snapshot(root, collection):
    from tests.mocks.restore_preservation import capture_preservation
    return asdict(capture_preservation(root, index_state=_existing_index(root, collection)))


def _sdk_closed(bundle):
    return bundle.client.closed and all(client.closed for client in bundle.sdk_clients)


def _resources(bundle=None, *, native=None):
    graph = bundle is not None
    if graph:
        _require(_sdk_closed(bundle), "slot_graph_SDK_close_incomplete")
        _require(not bundle.privacy.resource_status()["writer_lease"], "slot_writer_lease_retained")
        _require(bundle.home["service_calls"] == 0, "slot_unexpected_HA_call")
    return {"graph_constructed": graph, "graph_closed": True if graph else None,
        "writer_lease_released": True if graph else None, "sdk_clients_closed": True,
        "native_gates_returned": native, "zero_HA_service_calls": True if graph else None}


async def _seed(args, control, guard, e):
    import pytest
    from tests.mocks import application_restore as replay
    from tests.mocks.memory_graph import Clock, MemoryClient
    from tests.mocks.restore_application_graph import application_graph
    from tests.mocks.restore_preservation import assert_manual_sentinels, seed_manual_sentinels
    clock, patch = Clock(time.time()), pytest.MonkeyPatch()
    marker = len(MemoryClient._created)
    try:
        async with application_graph(Path(args.state_root), patch, clock) as graph:
            e._graph_contract(graph, replay)
            subject, foreign = await replay.seed_all(graph)
            sources = await replay.manual_voice_inputs(graph, foreign)
            manual = await seed_manual_sentinels(graph, sources=sources,
                voice_ids=tuple(foreign.voice_ids), suggestion_ids=tuple(foreign.suggestion_ids),
                root_lease=replay.root_lease(graph), delete_profile=True)
            await assert_manual_sentinels(graph, manual)
            manifest = await replay.payload(graph, foreign)
            # Closing/recovery disposes runtime stream leases; they are not persistent authority.
            manifest["tables"][str(graph.root / "privacy.db")] = {
                table: rows for table, rows in manifest["tables"][str(graph.root / "privacy.db")].items()
                if table not in RUNTIME_TABLES}
            collection = graph.memory.manager.long_term._collection.name
            metadata = graph.memory.manager.long_term._collection.metadata
            namespace, consent = graph.reader.namespace, _consent(graph.graph.config)
            observations = {"ten_drivers": len(graph.owner._drivers) == 10,
                "sources_live": graph.privacy.sources_available((subject.source, foreign.source)),
                "foreign_source_exact": foreign.source.to_dict() == sources[0].to_dict(),
                "manual_current": True, "consent_current": consent == manual.config_consent,
                "reader_namespace": namespace}
            guard.audit()
        MemoryClient.close_since(marker)
        resources = _resources(graph)
        # Capture the state after genuine close, not an earlier live-owner approximation.
        closed_manifest = _project_manifest(graph.root, graph.root, manifest, collection)
        _require(all(value is not None for value in closed_manifest["files"].values()),
            "seed_U_file_disappeared_during_close")
        saved = {"old_root": str(graph.root), "clock_now": clock.now, "subject": subject,
            "foreign": foreign, "manual": manual, "foreign_manifest": closed_manifest,
            "manual_manifest": _manual_manifest(graph.root, manual, graph.suggestions.store.path),
            "index_collection": collection,
            "index_metadata": metadata, "original_u_source": foreign.source.to_dict(),
            "old_reader_namespace": namespace, "consent": consent,
            "closed_state": _closed_snapshot(graph.root, collection)}
        checkpoint = _checkpoint_path(args, control)
        digest = _write_bag(e, checkpoint, control, "seed", saved)
        return resources, observations, digest, None
    finally:
        MemoryClient.close_since(marker)
        patch.undo()


async def _current_manual_decisions(graph, saved):
    """New branch decisions use only the actual graph's public authority APIs."""
    from src.users.user_manager import PermissionLevel
    from tests.mocks import application_restore as replay
    manual, authority = saved["manual"], graph.graph.dashboard_authority
    actor = replay.root_lease(graph)
    assert await graph.voices.acall(graph.voices.ledger.delete_profile,
        manual.profile_id, manual.profile_revision, graph.voice_actor)
    credentials = _rows(authority.path, "credentials")
    matches = [row for row in credentials if row["user_id"] == manual.user_id]
    _require(len(matches) == 1, "slot_original_HOME_grant_ambiguous")
    identity = matches[0]["id"]
    assert await authority.revoke_credential(actor, identity)
    users = graph.graph.pipeline.request_router.user_manager
    def current():
        if not actor.current(frozenset({"admin:manage"})):
            raise PermissionError("slot_account_authority_revoked")
    changed, _ = await asyncio.to_thread(users.update_user, manual.user_id,
        permission_level=PermissionLevel.GUEST, before_save=current)
    _require(changed, "slot_current_account_not_committed")
    user = await asyncio.to_thread(users.get_user, manual.user_id, before_read=current)
    _require(user is not None, "slot_current_account_missing")
    profile_rows = [row for row in _rows(graph.root / "privacy.db", "voice_profiles")
        if row["id"] == manual.profile_id]
    _require(len(profile_rows) == 1 and profile_rows[0]["deleted_at"] is not None
        and profile_rows[0]["revision"] == manual.profile_revision + 1,
        "slot_profile_deletion_not_committed")
    credential_rows = [row for row in _rows(authority.path, "credentials") if row["id"] == identity]
    _require(not credential_rows and authority.resolve(manual.home_token) is None,
        "slot_credential_revocation_not_committed")
    return {"latest_profile_id": manual.profile_id, "latest_profile_rows": profile_rows,
        "credential_id": identity, "latest_credential_rows": credential_rows,
        "latest_user": user.to_dict()}


async def _candidate(args, control, guard, e):
    import pytest
    from tests.mocks import application_restore as replay
    from tests.mocks import integrated_restore as data_helpers
    from tests.mocks.memory_graph import Clock, MemoryClient
    from tests.mocks.producer_erasure_graph import NativeGate
    from tests.mocks.restore_application_graph import application_graph
    checkpoint = _checkpoint_path(args, control)
    saved, seed_sha = _read_bag(e, checkpoint, control, "seed")
    old_root, root = Path(saved["old_root"]), Path(args.state_root)
    _require(root != old_root, "slot_candidate_not_a_distinct_state_root")
    inventory_path = Path(args.evidence_root) / "copied-target-inventory-before-Graph.json"
    _require(inventory_path.is_file() and not inventory_path.is_symlink(),
        "slot_parent_preconstruction_inventory_missing")
    inventory = json.loads(inventory_path.read_text())
    _require(type(inventory) is list and bool(inventory), "slot_parent_target_inventory_empty")
    # This precedes build_application: recovery and maintenance already write during construction.
    before_graph = _closed_snapshot(root, saved["index_collection"])
    before_graph["parent_target_inventory_sha256"] = _sha(inventory_path.read_bytes())
    clock, patch = Clock(saved["clock_now"]), pytest.MonkeyPatch()
    subject_gate, unrelated_gate = NativeGate(), NativeGate()
    marker = len(MemoryClient._created)
    try:
        async with application_graph(root, patch, clock,
                subject_gate=subject_gate, unrelated_gate=unrelated_gate) as graph:
            e._graph_contract(graph, replay)
            subject, foreign = saved["subject"], saved["foreign"]
            _require(graph.privacy.sources_available((subject.source, foreign.source)),
                "slot_copied_sources_not_initially_live")
            decisions = await _current_manual_decisions(graph, saved)
            operation = await replay.erase_with_real_producer(
                graph, subject, foreign, subject_gate, unrelated_gate)
            # Historical descriptors may leave honest pending work/orphan copies.  Do not
            # invent completion or rewrite them to make D's same-root settle oracle pass.
            for _ in range(12):
                await graph.owner.reconcile()
                clock.now += .1
                await asyncio.sleep(.01)
            steps = await data_helpers.steps(graph, operation)
            guards = {("diagnostics", 1), ("streams", 2), ("voices", 2)}
            current_targets = _rows(root / "privacy.db", "erasure_targets")
            original_targets = saved["closed_state"]["tables"][str(old_root / "privacy.db")]["erasure_targets"]
            observations = {"offline_inventory_before_graph": True,
                "producer_completed_while_U_held": True,
                "source_revoked": graph.privacy.ledger.conversation_revoked(subject.source.conversation_id),
                "three_guards_pending": all(any(row["participant"] == part and row["version"] == version
                    and row["state"] == "pending" for row in steps) for part, version in guards),
                "foreign_source_exact": foreign.source.to_dict() == saved["original_u_source"]
                    and graph.privacy.sources_available((foreign.source,)),
                "foreign_payload_preserved": None,
                "manual_current": True, "consent_current": _consent(graph.graph.config) == saved["consent"],
                "reader_namespace_changed": graph.reader.namespace != saved["old_reader_namespace"],
                "historical_targets_retained": all(row in current_targets for row in original_targets)}
            consent = _consent(graph.graph.config)
            guard.audit()
        _require(subject_gate.returned.is_set() and unrelated_gate.returned.is_set()
            and subject_gate.completed.is_set() and unrelated_gate.completed.is_set(),
            "slot_actual_native_return_missing")
        MemoryClient.close_since(marker)
        resources = _resources(graph, native=True)
        current_u = _project_manifest(root, old_root, saved["foreign_manifest"], saved["index_collection"])
        observations["foreign_payload_preserved"] = _manifest_equal(saved["foreign_manifest"], current_u)
        candidate = {"seed_sha256": seed_sha, "candidate_root": str(root), "clock_now": clock.now,
            "operation": operation, "steps": steps, **decisions,
            "latest_manual_manifest": _manual_manifest(root, saved["manual"], graph.suggestions.store.path),
            "consent": consent,
            "closed_state": _closed_snapshot(root, saved["index_collection"]), "before_graph": before_graph}
        candidate_sha = _write_bag(e, checkpoint.with_name("candidate-state.json"), control, "candidate", candidate)
        _require(_sha(checkpoint.read_bytes()) == seed_sha, "slot_seed_changed")
        return resources, observations, seed_sha, candidate_sha
    finally:
        subject_gate.release.set()
        unrelated_gate.release.set()
        MemoryClient.close_since(marker)
        patch.undo()


def _find_user(root, identity):
    found = []
    for path in Path(root).rglob("*_accounts_v1.json"):
        value = json.loads(path.read_text())
        def visit(item):
            if isinstance(item, dict):
                if item.get("user_id") == identity:
                    found.append(item)
                for child in item.values():
                    visit(child)
            elif isinstance(item, list):
                for child in item:
                    visit(child)
        visit(value)
    _require(len(found) == 1, "slot_offline_account_ambiguous")
    return found[0]


async def _inspect_old(args, control, guard, e):
    checkpoint = _checkpoint_path(args, control)
    saved, seed_sha = _read_bag(e, checkpoint, control, "seed")
    candidate_path = checkpoint.with_name("candidate-state.json")
    candidate, candidate_sha = _read_bag(e, candidate_path, control, "candidate")
    _require(candidate["seed_sha256"] == seed_sha and args.state_root == saved["old_root"],
        "slot_inspect_not_original_root")
    root, manual = Path(args.state_root), saved["manual"]
    # There is no Graph, DashboardAuthority, reader lease or writer in this phase.
    current_u = _project_manifest(root, root, saved["foreign_manifest"], saved["index_collection"])
    tombstones = _rows(root / "privacy.db", "conversation_tombstones")
    profile_rows = [row for row in _rows(root / "privacy.db", "voice_profiles")
        if row["id"] == candidate["latest_profile_id"]]
    # Discover the existing auth DB by its table, never fabricate or create it.
    auth = []
    for original, tables in saved["closed_state"]["tables"].items():
        if "credentials" in tables and "auth_meta" in tables:
            auth.append(Path(original))
    _require(len(auth) == 1, "slot_offline_auth_database_ambiguous")
    credentials = [row for row in _rows(auth[0], "credentials") if row["id"] == candidate["credential_id"]]
    observations = {"old_U_preserved": _manifest_equal(saved["foreign_manifest"], current_u),
        "old_manual_preserved": _manual_equal(root, root, saved["manual_manifest"]),
        "latest_tombstone_present": any(row["conversation_id"] == saved["subject"].source.conversation_id
            for row in tombstones),
        "latest_profile_deletion_present": _same_rows(profile_rows, candidate["latest_profile_rows"]),
        "latest_credential_revocation_present": _same_rows(credentials, candidate["latest_credential_rows"]),
        "latest_account_permission_present": _find_user(root, manual.user_id) == candidate["latest_user"],
        # Input consent is unchanged deny; no unimplemented consent mutation is claimed.
        "consent_current": saved["consent"] == candidate["consent"]}
    guard.audit()
    return _resources(), observations, seed_sha, candidate_sha


def _main():
    parser = argparse.ArgumentParser(description="Compiled two-root slot rollback observations")
    parser.add_argument("--mode", required=True, choices=("seed", "candidate", "inspect-old"))
    for flag in ("control", "runtime", "state-root", "evidence-root", "checkpoint"):
        parser.add_argument("--" + flag, required=True)
    args = parser.parse_args()
    e = _e_module()
    control, release, guard, hits = e._bootstrap(args)
    own = Path(__file__).resolve().relative_to(Path(control["island_root"])).as_posix()
    _require(own in control["island_files"], "slot_entry_not_hashlisted")
    _require(release["entrypoint"] == ["python", "-m", "src.main"]
        and release["external_paths"] == ["config", "data", "models", "logs"], "slot_runtime_contract")
    runner = {"seed": _seed, "candidate": _candidate, "inspect-old": _inspect_old}[args.mode]
    resources, observations, seed_sha, candidate_sha = asyncio.run(runner(args, control, guard, e))
    origins = guard.audit()
    _require(not hits, "slot_forbidden_SDK_or_hardware_call")
    for relative, digest in release["files"].items():
        _require(_sha(e._canonical(Path(args.runtime) / relative).read_bytes()) == digest,
            "slot_artifact_changed")
    _require(_sha((Path(args.runtime) / "release.json").read_bytes()) == control["release_sha256"],
        "slot_release_changed")
    result = {"schema_version": 1, "unit": UNIT, "mode": args.mode,
        "source_commit": control["source_commit"], "runtime_root": args.runtime,
        "state_root": args.state_root, "pid": os.getpid(),
        "checkpoint_sha256": seed_sha, "candidate_checkpoint_sha256": candidate_sha,
        "module_origins": origins, "sys_flags": {
            "isolated": int(sys.flags.isolated), "ignore_environment": int(sys.flags.ignore_environment),
            "no_user_site": int(sys.flags.no_user_site), "no_site": int(sys.flags.no_site),
            "dont_write_bytecode": int(sys.dont_write_bytecode)},
        "resources": resources, "observations": observations}
    e._atomic_json(Path(args.evidence_root) / (args.mode + "-result.json"), result)
    print(json.dumps({"mode": args.mode, "checkpoint_sha256": seed_sha,
        "candidate_checkpoint_sha256": candidate_sha, "observations": observations}, sort_keys=True))


if __name__ == "__main__":
    _main()
