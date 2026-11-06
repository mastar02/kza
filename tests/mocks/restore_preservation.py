"""Offline restore preflight and exact preservation oracles for real graphs.

This helper never writes a persistent store. SQLite validation uses disposable
backups with the real constraints enabled; the caller owns the eventual replay.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from copy import deepcopy
from contextlib import closing
from dataclasses import dataclass, field
import hashlib
from importlib.metadata import version as installed_version
import json
import math
from pathlib import Path
import sqlite3
from uuid import UUID, uuid4


class ReplayRejected(ValueError):
    """A fixed compatibility failure, without private payload in its message."""


ALLOWED_TABLES = frozenset({
    "wake_clips_v1", "stt_corrections_v1", "conversation_turns_v1",
    "conversation_export_parents_v1", "voice_fingerprints", "voice_fingerprint_sources",
    "voice_capture_reviews", "voice_groups", "voice_group_members", "voice_group_lineage",
    "live_deliveries", "live_delivery_sources", "live_delivery_lineage",
    "utterances", "capture_conversations", "events_policy_v1", "suggestions_policy_v1",
    "metrics", "lineage",
})


@dataclass(frozen=True)
class PreservationState:
    root: str
    schemas: dict
    tables: dict
    files: dict
    index: dict | None


@dataclass(frozen=True)
class PreparedRestore:
    data: dict
    diagnostic_files: dict
    preservation: PreservationState
    root_identities: dict
    index_path: str
    index_collection: str
    sdk_version: str


@dataclass(frozen=True)
class ManualSentinels:
    profile_id: str
    profile_revision: int
    link_revision: int
    user_id: str
    user: dict
    fact_id: str
    group_request: str
    actor_subject: str
    voice_ids: tuple
    suggestion_ids: tuple
    rows: dict
    config_consent: object
    deleted_profile_id: str | None = None
    home_token: str = field(default="", repr=False)


def _manual_rows(g, *, profile_ids, voice_ids, fact_id, group_request, actor_subject, suggestion_ids):
    def read():
        with g.voices.ledger.ledger._transaction() as db:
            placeholders = ",".join("?" for _ in profile_ids)
            selected = {}
            for table, column in (("voice_profiles", "id"), ("voice_profile_sources", "profile_id"),
                    ("voice_review_requests", "profile_id"), ("voice_profile_people", "profile_id")):
                selected[table] = [dict(row) for row in db.execute(
                    "SELECT * FROM " + _quote(table) + " WHERE " + _quote(column)
                    + " IN (" + placeholders + ")", profile_ids)]
            selected["voice_group_requests"] = [dict(row) for row in db.execute(
                "SELECT * FROM voice_group_requests WHERE actor=? AND request_id=?", (actor_subject, group_request))]
            selected["voice_group_request_lineage"] = [dict(row) for row in db.execute(
                "SELECT * FROM voice_group_request_lineage WHERE actor=? AND request_id=?", (actor_subject, group_request))]
            marks = ",".join("?" for _ in voice_ids)
            for table, column in (("voice_fingerprints", "id"),
                    ("voice_fingerprint_sources", "fingerprint_id"), ("voice_capture_reviews", "fingerprint_id")):
                selected[table] = [dict(row) for row in db.execute("SELECT * FROM " + _quote(table)
                    + " WHERE " + _quote(column) + " IN (" + marks + ")", voice_ids)]
            members = [dict(row) for row in db.execute(
                "SELECT * FROM voice_group_members WHERE fingerprint_id IN (" + marks + ")", voice_ids)]
            selected["voice_group_members"] = members
            group_ids = tuple(sorted({row["group_id"] for row in members}))
            if group_ids:
                marks = ",".join("?" for _ in group_ids)
                for table, column in (("voice_groups", "id"), ("voice_group_lineage", "group_id")):
                    selected[table] = [dict(row) for row in db.execute("SELECT * FROM " + _quote(table)
                        + " WHERE " + _quote(column) + " IN (" + marks + ")", group_ids)]
            for table, column in (("facts", "id"), ("fact_sources", "fact_id"), ("fact_dependencies", "fact_id")):
                selected[table] = [dict(row) for row in db.execute(
                    "SELECT * FROM " + _quote(table) + " WHERE " + _quote(column) + "=?", (fact_id,))]
            return selected
    return read


def _consent_state(graph):
    reasoner = graph.config.get("reasoner", {}) or {}
    compaction = reasoner.get("compaction", {}) or {}
    return deepcopy({"destination": reasoner.get("destination"), "cloud": reasoner.get("cloud", {}),
        "compaction_destination": compaction.get("destination"),
        "compaction_cloud_consent": compaction.get("cloud_consent")})


async def seed_manual_sentinels(g, *, sources, voice_ids, suggestion_ids,
                                root_lease, delete_profile=False) -> ManualSentinels:
    """Make current decisions through graph-owned APIs after the old data snapshot."""
    from src.dashboard.authority import DashboardRole
    from src.memory.fact_ledger import MemoryReviewer
    from src.users.user_manager import PermissionLevel
    from src.users.voice_ledger import VoiceReviewer
    graph, voices = g.graph, g.voices
    authority = graph.dashboard_authority
    required = frozenset({"voices:review", "admin:manage", "auth:manage", "memory:review"})
    if root_lease._authority is not authority or not root_lease.current(required):
        raise PermissionError("manual_sentinel_authority_unavailable")
    if len(voice_ids) < 3 or not sources:
        raise ValueError("manual_sentinel_three_captures_required")
    actor = VoiceReviewer.from_dashboard(root_lease.principal,
        lambda: root_lease.current(frozenset({"voices:review", "admin:manage"})), person_authority=authority)
    profile = await voices.acall(voices.ledger.confirm, tuple(voice_ids[:3]),
        "Synthetic current reviewed voice", str(uuid4()), actor)
    users = graph.pipeline.request_router.user_manager
    def current():
        if not root_lease.current(frozenset({"admin:manage"})):
            raise PermissionError("manual_sentinel_account_denied")
    user, _ = await asyncio.to_thread(users.add_user, "Synthetic restore person " + uuid4().hex,
                                    PermissionLevel.ADULT, before_save=current)
    if user is None:
        raise AssertionError("manual account was not committed")
    links = voices.ledger.people
    review_only = VoiceReviewer.from_dashboard(root_lease.principal,
        lambda: root_lease.current(frozenset({"voices:review"})))
    try:
        await voices.acall(links.set_person, profile["id"], user.user_id, profile["revision"], 0, review_only)
    except PermissionError:
        pass
    else:
        raise AssertionError("voice review alone acquired person-link authority")
    receipt = await voices.acall(links.set_person, profile["id"], user.user_id, profile["revision"], 0, actor)
    receipt = await voices.acall(links.set_person, profile["id"], None, profile["revision"], receipt.link_revision, actor)
    receipt = await voices.acall(links.set_person, profile["id"], user.user_id, profile["revision"], receipt.link_revision, actor)
    snapshot = await voices.acall(voices.ledger.snapshot)
    first = next(row for row in snapshot["fingerprints"] if row["id"] == voice_ids[0])
    request = str(uuid4())
    assert await voices.acall(voices.ledger.groups.correct, (voice_ids[0],),
        {first["group_id"]: first["group_revision"]}, "unknown", request, actor)
    memory_actor = MemoryReviewer.from_dashboard(root_lease.principal,
        lambda: root_lease.current(frozenset({"memory:review"})))
    proposed = await g.privacy.acall(g.privacy.ledger.store, "Synthetic reviewed preference " + uuid4().hex,
                                    "preference", .9, (sources[0],), proposed=True)
    confirmed = await g.privacy.acall(g.privacy.ledger.confirm, proposed.id, memory_actor)
    await g.memory.manager.long_term.areconcile()
    for number, identity in enumerate(suggestion_ids):
        result = await g.events.acall(g.suggestions.respond_to_suggestion, identity,
                                      number % 3 == 2, 1 if number % 3 == 1 else 0)
        assert result["success"]
    deleted = None
    if delete_profile:
        retired = await voices.acall(voices.ledger.confirm, tuple(voice_ids[:3]),
            "Synthetic deleted reviewed voice", str(uuid4()), actor)
        await voices.acall(links.set_person, retired["id"], user.user_id, retired["revision"], 0, actor)
        assert await voices.acall(voices.ledger.delete_profile, retired["id"], retired["revision"], actor)
        deleted = retired["id"]
    issued = await authority.issue_credential(root_lease, subject="synthetic-restore-home-" + uuid4().hex,
        role=DashboardRole.HOME, expires_at=authority._now() + 3600, user_id=user.user_id)
    home = authority.resolve(issued.token)
    assert home is not None and home.current() and "voices:review" not in home.principal.permissions
    profile_ids = (profile["id"],) if deleted is None else (profile["id"], deleted)
    rows = await voices.acall(_manual_rows(g, profile_ids=profile_ids, voice_ids=tuple(voice_ids),
        fact_id=confirmed.id, group_request=request, actor_subject=actor.subject, suggestion_ids=tuple(suggestion_ids)))
    with closing(_image(Path(g.suggestions.store.path))) as db:
        rows["suggestions_policy_v1"] = [dict(row) for row in db.execute("SELECT * FROM suggestions_policy_v1")
                                         if row["id"] in suggestion_ids]
    return ManualSentinels(profile["id"], profile["revision"], receipt.link_revision, user.user_id,
        user.to_dict(), confirmed.id, request, actor.subject, tuple(voice_ids), tuple(suggestion_ids), rows,
        _consent_state(graph), deleted, issued.token)


async def assert_manual_sentinels(g, saved: ManualSentinels):
    """Use Graph2's real policy/serving APIs and compare every selected SQL field."""
    from src.users.voice_ledger import VoiceReviewer
    graph = g.graph
    ids = (saved.profile_id,) if saved.deleted_profile_id is None else (saved.profile_id, saved.deleted_profile_id)
    current = await g.voices.acall(_manual_rows(g, profile_ids=ids, voice_ids=saved.voice_ids,
        fact_id=saved.fact_id, group_request=saved.group_request, actor_subject=saved.actor_subject,
        suggestion_ids=saved.suggestion_ids))
    with closing(_image(Path(g.suggestions.store.path))) as db:
        current["suggestions_policy_v1"] = [dict(row) for row in db.execute("SELECT * FROM suggestions_policy_v1")
                                            if row["id"] in saved.suggestion_ids]
    assert _freeze({key: sorted(value, key=repr) for key, value in current.items()}) == _freeze(
        {key: sorted(value, key=repr) for key, value in saved.rows.items()})
    user = graph.pipeline.request_router.user_manager.get_user(saved.user_id)
    assert user is not None and user.to_dict() == saved.user
    assert _consent_state(graph) == saved.config_consent
    snapshot = await g.voices.acall(g.voices.ledger.snapshot)
    profile = next(row for row in snapshot["profiles"] if row["id"] == saved.profile_id)
    assert profile["revision"] == saved.profile_revision and profile["expires_at"] is None
    assert saved.deleted_profile_id not in {row["id"] for row in snapshot["profiles"]}
    home = graph.dashboard_authority.resolve(saved.home_token)
    assert home is not None and home.current()
    assert home.principal.user_id == saved.user_id
    assert not ({"voices:review", "admin:manage", "memory:review"} & home.principal.permissions)
    try:
        VoiceReviewer.from_dashboard(home.principal, home.current)
    except PermissionError:
        pass
    else:
        raise AssertionError("a domestic credential acquired voice-review authority")
    fact = await g.privacy.acall(g.privacy.ledger.get, saved.fact_id)
    assert fact is not None and fact.expires_at is None and fact.confirmed_by == saved.actor_subject


def _reject(code):
    raise ReplayRejected(code)


def _path(root, value, *, exists=True):
    path = Path(value)
    if not path.is_absolute() or path.is_symlink():
        _reject("restore_path_unverified")
    try:
        resolved = path.resolve(strict=exists)
    except OSError:
        _reject("restore_path_missing")
    if resolved != path or not path.is_relative_to(root):
        _reject("restore_path_outside_root")
    return path


def _quote(value):
    return '"' + value.replace('"', '""') + '"'


def _image(path):
    if not path.is_file():
        _reject("restore_database_missing")
    image = sqlite3.connect(":memory:")
    try:
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as source:
            source.backup(image)
        image.row_factory = sqlite3.Row
        image.execute("PRAGMA foreign_keys=ON")
        return image
    except BaseException:
        image.close()
        raise


def _schema(db, path):
    info = path.stat()
    return {
        "identity": (info.st_dev, info.st_ino),
        "application_id": db.execute("PRAGMA application_id").fetchone()[0],
        "user_version": db.execute("PRAGMA user_version").fetchone()[0],
        "objects": tuple(tuple(row) for row in db.execute(
            "SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name")),
    }


def _freeze(value):
    if isinstance(value, dict):
        return tuple((key, _freeze(item)) for key, item in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if hasattr(value, "tolist"):
        return _freeze(value.tolist())
    return value


def _index_records(state):
    if state is None:
        return None
    ids = list(state.get("ids", ()))
    if any(type(identity) is not str for identity in ids) or len(ids) != len(set(ids)):
        _reject("restore_duplicate_vector_id")
    if any(state.get(key) is not None and len(state[key]) != len(ids)
           for key in ("embeddings", "documents", "metadatas")):
        _reject("restore_vector_shape")
    records = {}
    for position, identity in enumerate(ids):
        records[identity] = {key: _freeze(state[key][position])
            for key in ("embeddings", "documents", "metadatas")
            if state.get(key) is not None}
    return records


def _record_digest(record):
    return hashlib.sha256(json.dumps(_freeze(record), ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


async def capture_index_lineage(g, index_payload) -> dict:
    """Witness real G1 SDK records and live ledger provenance without replaying authority."""
    records = deepcopy(_index_records(index_payload))
    if not records:
        _reject("restore_index_witness_empty")
    ledger = g.privacy.ledger
    expected = {"path": str((Path(g.root) / "chroma").resolve()),
                "collection": g.memory.manager.long_term.index_collection_name}
    def read():
        with ledger._transaction() as db:
            targets = [dict(row) for row in db.execute(
                "SELECT id,version,descriptor FROM erasure_targets WHERE participant='semantic_index' AND version=1")
                if json.loads(row["descriptor"]) == expected]
            if len(targets) != 1:
                _reject("restore_index_witness_target")
            target = targets[0]
            lineage = {}
            for identity, record in records.items():
                fact = ledger._fact(db, identity)
                metadata = dict(record.get("metadatas") or ())
                if (set(record) != {"embeddings", "documents", "metadatas"}
                        or fact is None or not ledger._eligible(db, fact)
                        or record["documents"] != fact.content
                        or type(metadata.get("fact_policy")) is not int or metadata["fact_policy"] != 1
                        or type(metadata.get("revision")) is not int or metadata["revision"] != fact.revision):
                    _reject("restore_index_witness_record")
                sources = [source.to_dict() for source in fact.sources]
                _sources(json.dumps(sources))
                if not sources or not ledger._sources_live(db, fact.sources):
                    _reject("restore_index_witness_sources")
                lineage[identity] = {"fact_policy": 1, "revision": fact.revision,
                    "sources": deepcopy(sources), "record_digest": _record_digest(record),
                    "target_id": target["id"], "target_version": target["version"],
                    "target_descriptor": deepcopy(expected)}
            return {"index_records": records, "index_lineage": lineage}
    return await g.privacy.acall(read)


def _index_lineage_positions(privacy, payload, current, incoming_sources, tombstones, witness):
    """Validate the old data witness against current fences, never missing fact_sources."""
    bundle = _witness_value(witness, "index_lineage")
    if type(bundle) is not dict or set(bundle) != {"index_records", "index_lineage"}:
        _reject("restore_index_witness_shape")
    records, proofs = bundle["index_records"], bundle["index_lineage"]
    incoming = _index_records(payload)
    if type(records) is not dict or type(proofs) is not dict or set(incoming) != set(records) or set(incoming) != set(proofs):
        _reject("restore_index_witness_ids")
    expected_descriptor = {"path": _witness_value(witness, "index_path"),
                           "collection": _witness_value(witness, "index_collection")}
    keep = []
    for position, identity in enumerate(payload["ids"]):
        record, proof = incoming[identity], proofs[identity]
        if (type(proof) is not dict or set(proof) != {"fact_policy", "revision", "sources",
                "record_digest", "target_id", "target_version", "target_descriptor"}
                or _freeze(record) != _freeze(records[identity])
                or proof["record_digest"] != _record_digest(record)):
            _reject("restore_index_witness_changed")
        metadata = dict(record.get("metadatas") or ())
        if (set(record) != {"embeddings", "documents", "metadatas"}
                or type(proof["fact_policy"]) is not int or proof["fact_policy"] != 1
                or type(proof["revision"]) is not int or proof["revision"] < 1
                or type(metadata.get("fact_policy")) is not int or metadata["fact_policy"] != proof["fact_policy"]
                or type(metadata.get("revision")) is not int or metadata["revision"] != proof["revision"]
                or type(proof["target_version"]) is not int or proof["target_version"] != 1
                or proof["target_descriptor"] != expected_descriptor):
            _reject("restore_index_witness_metadata")
        sources = _sources(json.dumps(proof["sources"]))
        if len({source["capture_id"] for source in sources}) != len(sources):
            _reject("restore_index_witness_sources")
        target = privacy.execute("SELECT participant,version,descriptor FROM erasure_targets WHERE id=?",
                                 (proof["target_id"],)).fetchone()
        if (target is None or target["participant"] != "semantic_index" or target["version"] != 1
                or json.loads(target["descriptor"]) != expected_descriptor):
            _reject("restore_index_target_changed")
        if identity in current:
            if current[identity] != record:
                _reject("restore_vector_collision")
            continue
        if not {source["conversation_id"] for source in sources} <= tombstones:
            _reject("restore_index_source_not_revoked")
        for source in sources:
            copies = [value for value in incoming_sources if value["capture_id"] == source["capture_id"]]
            if not copies or any(_freeze(value) != _freeze(source) for value in copies):
                _reject("restore_index_source_witness_changed")
        fact = privacy.execute("SELECT state,revision,content,confirmed_at,confirmed_by FROM facts WHERE id=?",
                               (identity,)).fetchone()
        if (fact is None or fact["state"] != "deleted" or fact["content"] is not None
                or fact["confirmed_at"] is not None or fact["confirmed_by"] is not None
                or type(fact["revision"]) is not int or fact["revision"] <= proof["revision"]):
            _reject("restore_index_deletion_fence_missing")
        receipt = privacy.execute("SELECT revision,state FROM policy_index_deletions WHERE target_id=? AND fact_id=?",
                                  (proof["target_id"], identity)).fetchone()
        if receipt is None or receipt["state"] != "complete" or receipt["revision"] != fact["revision"]:
            _reject("restore_index_deletion_receipt_missing")
        keep.append(position)
    return keep


def capture_preservation(root, *, extra_files=(), index_state=None) -> PreservationState:
    """Capture every logical SQL field and file; SDK contents use its real get()."""
    root = Path(root).resolve(strict=True)
    tables, schemas, files = {}, {}, {}
    sdk = root / "chroma"
    databases = set()
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in {".db", ".sqlite", ".sqlite3"}:
            path = _path(root, path)
            databases.add(path)
            with closing(_image(path)) as db:
                schemas[str(path)] = _schema(db, path)
                if path.is_relative_to(sdk):
                    continue  # SDK data is compared through its actual get(), not private SQL rows.
                names = [row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
                tables[str(path)] = {name: tuple(dict(row) for row in db.execute(
                    "SELECT * FROM " + _quote(name))) for name in names}
    for path in root.rglob("*"):
        if not path.is_file() or path.is_relative_to(sdk) or path in databases:
            continue
        if path.name.endswith(("-wal", "-shm", ".lock")):
            continue
        path = _path(root, path)
        files[str(path)] = path.read_bytes()
    for value in extra_files:
        path = _path(root, Path(value))
        files[str(path)] = path.read_bytes()
    return PreservationState(str(root), schemas, tables, files, _index_records(index_state))


def assert_preserved(before, after, *, inserted_rows=()):
    """Compare all existing mixed rows and authority columns without exclusions."""
    plan = inserted_rows if isinstance(inserted_rows, PreparedRestore) else None
    inserted = plan.data["tables"] if plan is not None else dict(inserted_rows)
    assert before.root == after.root and before.schemas == after.schemas
    assert set(before.tables) == set(after.tables)
    for path, tables in before.tables.items():
        assert set(tables) == set(after.tables[path])
        for table, records in tables.items():
            expected = list(records) + list(inserted.get(path, {}).get(table, ()))
            assert sorted(map(_freeze, expected), key=repr) == sorted(
                map(_freeze, after.tables[path][table]), key=repr), (Path(path).name, table)
    expected_files = dict(before.files)
    if plan is not None:
        expected_files.update(plan.data["files"])
        expected_files.update(plan.diagnostic_files)
    assert expected_files == after.files
    expected_index = deepcopy(before.index)
    if plan is not None and expected_index is not None:
        expected_index.update(_index_records(plan.data["index"]))
    assert expected_index == after.index


def _sources(encoded):
    try:
        values = json.loads(encoded)
        if type(values) is not list or not values:
            _reject("restore_sources_missing")
        for value in values:
            if type(value) is not dict:
                _reject("restore_sources_invalid")
            for name in ("capture_id", "conversation_id"):
                if type(value[name]) is not str or str(UUID(value[name])) != value[name]:
                    _reject("restore_sources_invalid")
            born, expiry = value["captured_at"], value["expires_at"]
            if (type(born) not in (int, float) or type(expiry) not in (int, float)
                    or not math.isfinite(born) or not math.isfinite(expiry)
                    or not born < expiry <= born + 48 * 3600):
                _reject("restore_sources_expiry")
        return values
    except (ValueError, TypeError, KeyError, OverflowError):
        _reject("restore_sources_invalid")


def _source_digest(sources):
    canonical = sorted({json.dumps(value, sort_keys=True, separators=(",", ":")) for value in sources})
    return hashlib.sha256(json.dumps(canonical, separators=(",", ":")).encode()).hexdigest()


def _lineage(tables):
    clips = {row["id"] for row in tables.get("wake_clips_v1", ())}
    for table in ("stt_corrections_v1", "conversation_turns_v1"):
        if not {row["id"] for row in tables.get(table, ())} <= clips:
            _reject("restore_foreign_clip_child")
    for row in tables.get("conversation_export_parents_v1", ()):
        if row["export_id"] not in clips or row["parent_id"] not in clips:
            _reject("restore_export_parent_missing")
    fingerprints = {row["id"]: row for row in tables.get("voice_fingerprints", ())}
    groups = {row["id"]: row for row in tables.get("voice_groups", ())}
    members = tables.get("voice_group_members", ())
    proofs = {row["group_id"]: row for row in tables.get("voice_group_lineage", ())}
    for table in ("voice_fingerprint_sources", "voice_capture_reviews"):
        if not {row["fingerprint_id"] for row in tables.get(table, ())} <= set(fingerprints):
            _reject("restore_foreign_voice_child")
    if fingerprints:
        if any(row["suggested_profile"] is not None or row["suggested_revision"] is not None
               or row["similarity"] is not None for row in fingerprints.values()):
            _reject("restore_voice_manual_suggestion")
        if {row["fingerprint_id"] for row in members} != set(fingerprints):
            _reject("restore_voice_members_incomplete")
        if set(proofs) != set(groups) or {row["group_id"] for row in members} != set(groups):
            _reject("restore_voice_proof_missing")
        for identity, group in groups.items():
            if group["state"] not in {"suggested", "unknown"} or group["reason"].startswith("manual"):
                _reject("restore_manual_group")
            ids = sorted(row["fingerprint_id"] for row in members if row["group_id"] == identity)
            if not ids or any(value not in fingerprints for value in ids):
                _reject("restore_foreign_voice_member")
            proof = proofs[identity]
            if (proof["source_digest"] != _source_digest(_sources(group["sources_json"]))
                    or proof["member_digest"] != hashlib.sha256(
                        json.dumps(ids, separators=(",", ":")).encode()).hexdigest()):
                _reject("restore_voice_lineage_mismatch")
        sources = tables.get("voice_fingerprint_sources", ())
        for identity, row in fingerprints.items():
            expected = {source["conversation_id"] for source in _sources(row["sources_json"])}
            if {source["conversation_id"] for source in sources if source["fingerprint_id"] == identity} != expected:
                _reject("restore_voice_sources_incomplete")
    tickets = {row["ticket"] for row in tables.get("live_deliveries", ())}
    delivery_sources = tables.get("live_delivery_sources", ())
    delivery_proofs = {row["ticket"]: row for row in tables.get("live_delivery_lineage", ())}
    if tickets != set(delivery_proofs) or {row["ticket"] for row in delivery_sources} != tickets:
        _reject("restore_stream_lineage_missing")
    for ticket in tickets:
        roots = sorted(row["conversation_id"] for row in delivery_sources if row["ticket"] == ticket)
        proof = delivery_proofs[ticket]
        if type(proof["version"]) is not int or proof["version"] != 1 or proof["source_digest"] != hashlib.sha256(json.dumps(roots).encode()).hexdigest():
            _reject("restore_stream_lineage_mismatch")
    if "metrics" in tables or "lineage" in tables:
        ids = {row["id"] for row in tables.get("metrics", ())}
        if {row["record_id"] for row in tables.get("lineage", ())} != ids:
            _reject("restore_metric_lineage_missing")


def _jsonl(raw):
    try:
        if type(raw) is not bytes or raw and not raw.endswith(b"\n"):
            _reject("restore_jsonl_encoding")
        result = []
        for line in raw.splitlines(keepends=True):
            document = json.loads(line)
            if type(document) is not dict or type(document.get("record_id")) is not str:
                _reject("restore_jsonl_record")
            result.append((document, line))
        return result
    except (ValueError, TypeError, UnicodeError):
        _reject("restore_jsonl_invalid")


def _diagnostic_plan(root, snapshots, metrics):
    selected = {row["id"]: row for row in metrics}
    seen, current_ids, result = set(), set(), {}
    for value in snapshots:
        path = _path(root, value, exists=False)
        if path.exists():
            for doc, _ in _jsonl(path.read_bytes()):
                if doc["record_id"] in current_ids:
                    _reject("restore_current_jsonl_duplicate")
                current_ids.add(doc["record_id"])
    for value, raw in snapshots.items():
        path = _path(root, value, exists=False)
        current = path.read_bytes() if path.exists() else b""
        incoming = []
        for document, line in _jsonl(raw):
            identity = document["record_id"]
            if identity not in selected:
                continue  # Old U cache rows never overwrite current U.
            if identity in seen or identity in current_ids:
                _reject("restore_jsonl_collision")
            row = selected[identity]
            payload = json.loads(row["payload"])
            metadata = {"record_id", "recorded_at", "expires_at", "@timestamp", "service"}
            stamp = datetime.fromtimestamp(row["born"], timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
            if (document.get("recorded_at") != row["born"] or document.get("expires_at") != row["expires"]
                    or document.get("@timestamp") != stamp or document.get("service") != "kza-voice"
                    or set(document) != set(payload) | metadata
                    or any(document.get(key) != value for key, value in payload.items())):
                _reject("restore_metric_cache_mismatch")
            seen.add(identity)
            incoming.append(line)
        result[str(path)] = current + b"".join(incoming)
    if seen != set(selected):
        _reject("restore_metric_cache_missing")
    return result


def _witness_value(witness, name):
    return witness[name] if isinstance(witness, dict) else getattr(witness, name)


def validate_restore_locator(root, snapshot, *, witness):
    """Validate trusted original locators before any SDK client is constructed."""
    try:
        root = Path(root).resolve(strict=True)
        if witness is None or Path(_witness_value(witness, "root")).resolve(strict=True) != root:
            _reject("restore_original_locator_missing")
        for name in ("index_path", "index_collection", "sdk_version", "root_identities"):
            if getattr(snapshot, name) != _witness_value(witness, name):
                _reject("restore_original_locator_changed")
        if installed_version("chromadb") != _witness_value(witness, "sdk_version"):
            _reject("restore_sdk_version")
        if _path(root, snapshot.index_path) != root / "chroma":
            _reject("restore_index_root")
        if type(snapshot.index_collection) is not str or not snapshot.index_collection:
            _reject("restore_index_collection")
        for value, identity in _witness_value(witness, "root_identities").items():
            path = _path(root, value)
            info = path.stat()
            if not path.is_dir() or (info.st_dev, info.st_ino) != tuple(identity):
                _reject("restore_root_replaced")
        if not set(snapshot.data["tables"]) <= set(_witness_value(witness, "schemas")):
            _reject("restore_database_locator_changed")
        stores = [(Path(path), schema) for path, schema in _witness_value(witness, "schemas").items()
                  if Path(path).is_relative_to(root / "chroma")]
        if not stores:
            _reject("restore_sdk_store_witness_missing")
        for path, schema in stores:
            path = _path(root, path)
            with closing(_image(path)) as db:
                if _schema(db, path) != schema:
                    _reject("restore_sdk_store_replaced")
    except ReplayRejected:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        _reject("restore_locator_incompatible")


def preflight_restore(root, snapshot, *, index_state, index_version,
                      schema_witness=None, witness=None) -> PreparedRestore:
    """Reject deterministic incompatibilities before persistent writes.

    This is not crash atomicity across databases, files and the SDK.
    Exact existing rows are no-ops; changed mixed rows are never updated.
    """
    root = Path(root).resolve(strict=True)
    images = {}
    try:
        validate_restore_locator(root, snapshot, witness=witness)
        if schema_witness is None:
            schema_witness = _witness_value(witness, "schemas")
        if type(index_version) is not str or index_version != snapshot.sdk_version:
            _reject("restore_sdk_version")
        if _path(root, snapshot.index_path) != root / "chroma":
            _reject("restore_index_root")
        if type(snapshot.index_collection) is not str or not snapshot.index_collection:
            _reject("restore_index_collection")
        if index_state is None or schema_witness is None:
            _reject("restore_original_witness_missing")
        for value, identity in snapshot.root_identities.items():
            path = _path(root, value)
            info = path.stat()
            if not path.is_dir() or (info.st_dev, info.st_ino) != tuple(identity):
                _reject("restore_root_replaced")
        data = deepcopy(snapshot.data)
        if set(data) != {"tables", "files", "index"}:
            _reject("restore_manifest_shape")
        preservation = capture_preservation(root, index_state=index_state)
        original = deepcopy(data["tables"])
        for value, tables in original.items():
            path = _path(root, value)
            if path.is_relative_to(root / "chroma"):
                _reject("restore_sdk_sql_unsupported")
            db = _image(path)
            images[value] = db
            if value not in schema_witness or _schema(db, path) != schema_witness[value]:
                _reject("restore_schema_changed")
            if not set(tables) <= ALLOWED_TABLES:
                _reject("restore_authority_table_forbidden")
            _lineage(tables)
            for table, records in tables.items():
                info = list(db.execute("PRAGMA table_info(" + _quote(table) + ")"))
                columns = [row[1] for row in info]
                primary = [row[1] for row in sorted(info, key=lambda row: row[5]) if row[5]]
                if not columns or not primary:
                    _reject("restore_schema_unverified")
                inserted = []
                for row in records:
                    if set(row) != set(columns):
                        _reject("restore_column_mismatch")
                    clause = " AND ".join(_quote(column) + " IS ?" for column in primary)
                    current = db.execute("SELECT * FROM " + _quote(table) + " WHERE " + clause,
                                         tuple(row[column] for column in primary)).fetchone()
                    if current is not None:
                        if _freeze(dict(current)) != _freeze(row):
                            _reject("restore_mixed_row_conflict")
                    else:
                        inserted.append(row)
                data["tables"][value][table] = inserted
        privacy = images.get(str(root / "privacy.db"))
        if privacy is None:
            _reject("restore_authority_witness_missing")
        return _finish_preflight(root, snapshot, data, original, images, privacy, preservation, witness)
    except ReplayRejected:
        raise
    except (sqlite3.Error, OSError, ValueError, TypeError, LookupError, AttributeError, OverflowError):
        _reject("restore_manifest_incompatible")
    finally:
        for image in images.values():
            image.close()


def _finish_preflight(root, snapshot, data, original, images, privacy, preservation, witness):
    tombstones = {row[0] for row in privacy.execute("SELECT conversation_id FROM conversation_tombstones")}
    incoming_sources, metrics, metric_paths = [], [], set()
    order = {name: i for i, name in enumerate((
        "wake_clips_v1", "stt_corrections_v1", "conversation_turns_v1", "conversation_export_parents_v1",
        "voice_fingerprints", "voice_fingerprint_sources", "voice_capture_reviews", "voice_groups",
        "voice_group_members", "voice_group_lineage", "live_deliveries", "live_delivery_sources",
        "live_delivery_lineage", "capture_conversations", "utterances", "events_policy_v1",
        "suggestions_policy_v1", "metrics", "lineage"))}
    for path, tables in data["tables"].items():
        db = images[path]
        data["tables"][path] = {table: tables[table] for table in sorted(tables, key=order.__getitem__)}
        for table, records in tables.items():
            for row in records:
                for column in ("sources_json", "input_sources_json"):
                    if column in row:
                        incoming_sources.extend(_sources(row[column]))
                if "conversation_id" in row and row["conversation_id"] not in tombstones:
                    _reject("restore_foreign_conversation")
                if table == "capture_conversations" and row["id"] not in tombstones:
                    _reject("restore_foreign_conversation")
                if table == "suggestions_policy_v1" and (row["status"] != "pending" or row["snooze_until"] is not None):
                    _reject("restore_manual_suggestion")
            if table == "metrics":
                metrics.extend(records)
                metric_paths.add(path)
        has_sequence = db.execute("SELECT 1 FROM sqlite_master WHERE name='sqlite_sequence'").fetchone()
        before = tuple(tuple(row) for row in db.execute("SELECT name,seq FROM sqlite_sequence ORDER BY name")) if has_sequence else ()
        for table in sorted(tables, key=order.__getitem__):
            for row in tables[table]:
                columns = list(row)
                db.execute("INSERT INTO " + _quote(table) + " (" + ",".join(map(_quote, columns))
                    + ") VALUES (" + ",".join("?" for _ in columns) + ")", tuple(row.values()))
        if db.execute("PRAGMA foreign_key_check").fetchall():
            _reject("restore_foreign_key_mismatch")
        after = tuple(tuple(row) for row in db.execute("SELECT name,seq FROM sqlite_sequence ORDER BY name")) if has_sequence else ()
        if after != before:
            _reject("restore_sequence_changed")
    for source in incoming_sources:
        if source["conversation_id"] not in tombstones:
            _reject("restore_foreign_source")
        rows = privacy.execute("SELECT conversation_id,captured_at,expires_at,origin,dependencies FROM fact_sources WHERE capture_id=?",
                               (source["capture_id"],)).fetchall()
        for row in rows:
            if (tuple(row[:4]) != (source["conversation_id"], source["captured_at"], source["expires_at"], source["origin"])
                    or json.loads(row[4]) != source.get("fact_dependencies", [])):
                _reject("restore_source_deadline_changed")
    conversations = {source["conversation_id"] for source in incoming_sources}
    keep = _index_lineage_positions(privacy, data["index"], preservation.index,
        incoming_sources, tombstones, witness)
    data["index"] = {key: ([value[i] for i in keep] if value is not None else None)
                     for key, value in data["index"].items()}
    for request in privacy.execute("SELECT sources_json FROM voice_group_requests"):
        if {source["conversation_id"] for source in _sources(request[0])} & conversations:
            _reject("restore_manual_group_overlap")
    _preflight_files(root, snapshot, data, privacy, tombstones)
    allowed_caches = {candidate for path in metric_paths for candidate in
                      (path.removesuffix(".sqlite3"), path.removesuffix(".sqlite3") + ".1")}
    if not set(snapshot.diagnostic_files) <= allowed_caches:
        _reject("restore_metric_cache_path")
    diagnostic = _diagnostic_plan(root, snapshot.diagnostic_files, metrics)
    if set(diagnostic) & set(data["files"]):
        _reject("restore_file_role_collision")
    return PreparedRestore(data, diagnostic, preservation, deepcopy(snapshot.root_identities),
                           snapshot.index_path, snapshot.index_collection, snapshot.sdk_version)


def _preflight_files(root, snapshot, data, privacy, tombstones):
    required = set()
    shared = data["tables"][str(root / "privacy.db")]
    for row in shared.get("wake_clips_v1", ()):
        target = privacy.execute("SELECT descriptor FROM erasure_targets WHERE id=? AND participant='collectors' AND version=1",
                                 (row["target_id"],)).fetchone()
        if target is None:
            _reject("restore_clip_target_missing")
        descriptor = json.loads(target[0])
        extension = {"accepted": ".wav", "stt_correction": ".wav", "conversation_turn": ".json",
                     "conversation_export": ".jsonl"}.get(row["bucket"])
        if extension is None:
            _reject("restore_clip_bucket_unsupported")
        required.add(str(Path(descriptor["root"]) / "policy_v1" / (row["id"] + extension)))
    for tables in data["tables"].values():
        for row in tables.get("utterances", ()):
            if row.get("audio_path") is not None:
                required.add(row["audio_path"])
    if not required <= set(data["files"]):
        _reject("restore_file_manifest_incomplete")
    writes = {}
    for value, raw in data["files"].items():
        path = _path(root, value, exists=False)
        if type(raw) is not bytes or not path.parent.is_dir() or path.is_relative_to(root / "chroma"):
            _reject("restore_file_unverified")
        if path.exists():
            if not path.is_file() or path.read_bytes() != raw:
                _reject("restore_file_collision")
            continue
        if not any(path.is_relative_to(Path(directory)) for directory in snapshot.root_identities
                   if Path(directory) != root / "chroma"):
            _reject("restore_file_root_unregistered")
        if value not in required:
            if path.parent.name != "policy_v3" or path.suffix != ".json":
                _reject("restore_file_unverified")
            document = json.loads(raw)
            if path.name != document["user_id"] + ".json" or not document.get("summary_sources"):
                _reject("restore_context_unverified")
            sources = _sources(json.dumps(document["summary_sources"]))
            if not {source["conversation_id"] for source in sources} <= tombstones:
                _reject("restore_context_foreign_source")
        writes[value] = raw
    data["files"] = writes
