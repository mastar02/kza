"""Data adapters for replay between public ApplicationGraph generations.

Only saved, automatic payload is replayed. Authority/manual decisions stay
current. Preflight is not a crash-atomic multi-store restore protocol or a
certificate for opaque SDK/browser copies.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
import json
from pathlib import Path
import sqlite3
from uuid import uuid4

import chromadb
from chromadb.config import Settings
import numpy as np

from src.ambient.types import AmbientUtterance
from src.dashboard.authority import DashboardPermission
from src.dashboard.live_event_bus import LiveEvent, LiveEventType
from src.memory.erasure_journal import ErasureParticipant
from src.memory.fact_ledger import MemoryReviewer
from src.memory.producer_context import producer_context
from src.users.voice_ledger import VoiceReviewer
from tests.mocks import integrated_restore as data_helpers
from tests.mocks.memory_graph import SyntheticEmbedding
from tests.mocks.memory_sources import memory_source
from tests.mocks.producer_erasure_graph import launch, producer_step, until
from tests.mocks.restore_application_graph import ApplicationPeer, TOKEN
from tests.mocks.restore_preservation import (
    ReplayRejected, assert_preserved, capture_index_lineage, capture_preservation, preflight_restore,
    validate_restore_locator,
)


REMAINING_GUARDS = {("diagnostics", 1), ("streams", 2), ("voices", 2)}
PARTICIPANTS = {part.value for part in ErasureParticipant}
VOICE_DATA = ("voice_fingerprints", "voice_fingerprint_sources", "voice_capture_reviews",
    "voice_groups", "voice_group_members", "voice_group_lineage")
COLLECTOR_DATA = ("wake_clips_v1", "stt_corrections_v1", "conversation_turns_v1",
    "conversation_export_parents_v1")
STREAM_DATA = ("live_deliveries", "live_delivery_sources", "live_delivery_lineage")
SHARED_DATA = COLLECTOR_DATA + VOICE_DATA + STREAM_DATA


@dataclass(frozen=True)
class ReplayWitness:
    root: str
    schemas: dict
    root_identities: dict
    index_path: str
    index_collection: str
    sdk_version: str
    index_metadata: object
    index_lineage: dict


def root_lease(g):
    lease = g.graph.dashboard_authority.resolve("Bearer " + TOKEN)
    assert lease is not None and lease.current(frozenset({"admin:manage"}))
    return lease


def memory_actor(g):
    lease = root_lease(g)
    return MemoryReviewer.from_dashboard(lease.principal,
        lambda: lease.current(frozenset({"memory:review"})))


def voice_actor(g):
    lease = root_lease(g)
    return VoiceReviewer.from_dashboard(lease.principal,
        lambda: lease.current(frozenset({"voices:review", "admin:manage"})),
        person_authority=g.graph.dashboard_authority)


def reader(g):
    return root_lease(g).reader_binding(frozenset({DashboardPermission.CONVERSATIONS_READ}))


async def peer(g, *, client=None):
    return await ApplicationPeer(g.bus, reader(g), client=client).start()


async def seed_cohort(g, cohort, value):
    """Use the owners already returned by the application factory."""
    source = cohort.source
    audio = np.full(16000, value, np.float32)
    cohort.fact = await g.memory.manager.long_term.astore_fact(
        f"Synthetic fact {source.conversation_id}", "fact", sources=(source,))
    utterance = AmbientUtterance("synthetic-room", source.captured_at, source.captured_at + 1,
        "Synthetic capture text", capture_id=source.capture_id,
        conversation_id=source.conversation_id, expires_at=source.expires_at)
    cohort.ambient_id = await g.ambient.add(utterance)
    cohort.archive_path = await g.archive.write("synthetic-room", cohort.ambient_id, audio, source=source)
    assert cohort.archive_path and await g.ambient.set_audio_path(cohort.ambient_id, cohort.archive_path)
    assert await g.ambient.read_audio(cohort.ambient_id) is not None
    existing = data_helpers.rows(g.root / "privacy.db", "wake_clips_v1")
    before = {row["id"] for row in existing}
    after_sequence = max((row["sequence"] for row in existing), default=0)
    assert g.wake.submit("synthetic-room", .99, audio, sources=(source,))
    await until(lambda: g.wake._busy is None and not g.wake._queue)
    correction = await g.stt.add_correction(audio=audio, original_text="Synthetic original",
        corrected_text="Synthetic corrected", sources=(source,), correction_sources=(source,))
    assert correction is not None
    with producer_context("synthetic-room", (source,)):
        turn = await g.conversations.add_turn("Synthetic input", "Synthetic response",
            input_sources=(source,), response_sources=(source,))
    assert turn is not None
    view, = await g.conversations.get_turns(conversation_id=source.conversation_id)
    assert await g.conversations.mark_response(view, "good", sources=(source,))
    assert await g.conversations.export_for_training(g.targets["exports"].root,
        after_sequence=after_sequence) is not None
    live = [row for row in data_helpers.rows(g.root / "privacy.db", "wake_clips_v1")
        if row["id"] not in before and row["published"] == 1 and row["expires_at"] > g.clock.now]
    assert len(live) == 4
    assert {row["bucket"] for row in live} == {
        "accepted", "stt_correction", "conversation_turn", "conversation_export"}
    assert all(row["captured_at"] == source.captured_at and row["expires_at"] <= source.expires_at
        for row in live)
    cohort.clip_ids = sorted(row["id"] for row in live)
    cohort.user = f"context-{source.conversation_id}"
    context = await g.manager.aget_or_create(cohort.user)
    g.manager.add_turn(cohort.user, "user", "Synthetic sourced turn", sources=(source,))
    context.compacted_summary, context.summary_sources = "Synthetic summary", (source,)
    await g.manager._offload(g.persister.save, context)
    g.memory.manager.short_term.add_turn("Synthetic input", "Synthetic answer", sources=(source,))
    g.diagnostics.emitter.emit_request(user_id="fixture", zone_id=None, text="Synthetic metric",
        intent=None, path="fast", success=True, timings={"stt": 12., "total": 20.}, sources=(source,))
    await g.diagnostics.emitter.flush()
    assert g.voices.observe(g.encoder, audio, (source,), transcript="Synthetic reviewed clip",
        room_id="synthetic-room")
    await until(lambda: g.voices.resource_status()["pending"] == 0)
    cohort.voice_ids = [row["id"] for row in data_helpers.rows(g.root / "privacy.db", "voice_fingerprints")
        if source.conversation_id in row["sources_json"]]
    assert cohort.voice_ids
    for identity in cohort.voice_ids:
        review = await g.voices.acall(g.voices.ledger.review_capture, identity, voice_actor(g))
        assert review["transcript"] == "Synthetic reviewed clip" and review["audio_base64"]
        # No external STT/audio provenance document is invented by this leaf.
        assert review["audio_metadata"] is review["transcript_metadata"] is None
    connection = await peer(g)
    try:
        cohort.browser_client = connection.client
        await g.bus.publish(LiveEvent(LiveEventType.TURN,
            payload={"stt": "Synthetic private turn"}, sources=(source,)))
        assert (await connection.frame("turn"))["receipt"]
    finally:
        await connection.close()


async def seed_all(g):
    assert g.graph.producer_erasure.inventory.verify().valid
    assert {part.value for part in g.owner._drivers} == PARTICIPANTS
    subject = data_helpers.Cohort(memory_source(now=g.clock.now))
    await data_helpers.seed_patterns(g, subject, "erased")
    g.clock.now += 11 * 60
    foreign = data_helpers.Cohort(memory_source(now=g.clock.now))
    await data_helpers.seed_patterns(g, foreign, "survivor")
    proposals = await g.events.acall(g.suggestions.generate_suggestions, .7, 50)
    for cohort in (subject, foreign):
        cohort.suggestion_ids = [item.id for item in proposals
            if any(source.conversation_id == cohort.source.conversation_id for source in item.sources)]
        assert cohort.suggestion_ids, "real pattern evidence must produce a sourced proposal"
    await seed_cohort(g, subject, .1)
    await seed_cohort(g, foreign, .2)
    return subject, foreign


async def manual_voice_inputs(g, foreign):
    """Three U captures with distinct identities and the original deadline."""
    sources = [foreign.source]
    for _ in range(2):
        source = replace(foreign.source, capture_id=str(uuid4()))
        sources.append(source)
        assert g.voices.observe(g.encoder, np.full(16000, .2, np.float32), (source,),
            transcript="Synthetic reviewed clip", room_id="synthetic-room")
    await until(lambda: g.voices.resource_status()["pending"] == 0)
    foreign.voice_ids = [row["id"] for row in data_helpers.rows(g.root / "privacy.db", "voice_fingerprints")
        if foreign.source.conversation_id in row["sources_json"]]
    assert len(foreign.voice_ids) == 3
    assert all(source.expires_at == foreign.source.expires_at for source in sources)
    return tuple(sources)


def selected_shared(g, cohort, *, automatic=False):
    tables = data_helpers.copied_tables(g.root / "privacy.db", SHARED_DATA)
    clips, voices = set(cohort.clip_ids), set(cohort.voice_ids)
    members = [row for row in tables["voice_group_members"] if row["fingerprint_id"] in voices]
    groups = {row["group_id"] for row in members}
    assert all(row["fingerprint_id"] in voices for row in tables["voice_group_members"]
        if row["group_id"] in groups), "snapshot must not copy another conversation's membership"
    fingerprints = [row for row in tables["voice_fingerprints"] if row["id"] in voices]
    if automatic:
        assert all(row["suggested_profile"] is row["suggested_revision"] is row["similarity"] is None
            for row in fingerprints)
        requests = data_helpers.rows(g.root / "privacy.db", "voice_group_requests")
        assert not any(group in json.dumps(row, default=str) for group in groups for row in requests)
    tickets = {row["ticket"] for row in tables["live_delivery_sources"]
        if row["conversation_id"] == cohort.source.conversation_id}
    result = {
        "wake_clips_v1": [row for row in tables["wake_clips_v1"] if row["id"] in clips],
        "stt_corrections_v1": [row for row in tables["stt_corrections_v1"] if row["id"] in clips],
        "conversation_turns_v1": [row for row in tables["conversation_turns_v1"] if row["id"] in clips],
        "conversation_export_parents_v1": [row for row in tables["conversation_export_parents_v1"]
            if row["export_id"] in clips],
        "voice_fingerprints": fingerprints,
        "voice_fingerprint_sources": [row for row in tables["voice_fingerprint_sources"]
            if row["fingerprint_id"] in voices],
        "voice_capture_reviews": [row for row in tables["voice_capture_reviews"]
            if row["fingerprint_id"] in voices],
        "voice_groups": [row for row in tables["voice_groups"] if row["id"] in groups],
        "voice_group_members": members,
        "voice_group_lineage": [row for row in tables["voice_group_lineage"] if row["group_id"] in groups],
        "live_deliveries": [row for row in tables["live_deliveries"] if row["ticket"] in tickets],
        "live_delivery_sources": [row for row in tables["live_delivery_sources"] if row["ticket"] in tickets],
        "live_delivery_lineage": [row for row in tables["live_delivery_lineage"] if row["ticket"] in tickets],
    }
    assert len(result["voice_group_lineage"]) == len(groups)
    assert len(result["live_delivery_lineage"]) == len(tickets)
    return result


def selected_payload(g, cohort, *, automatic=False):
    conversation = cohort.source.conversation_id
    shared = selected_shared(g, cohort, automatic=automatic)
    ambient = data_helpers.copied_tables(g.root / "ambient.db", ("utterances", "capture_conversations"))
    ambient = {name: [row for row in records if row.get("conversation_id") == conversation
        or row.get("id") == conversation] for name, records in ambient.items()}
    events = data_helpers.copied_tables(g.events.db_path, ("events_policy_v1",))
    events["events_policy_v1"] = [row for row in events["events_policy_v1"]
        if conversation in row["sources_json"]]
    suggestions = data_helpers.copied_tables(g.suggestions.store.path, ("suggestions_policy_v1",))
    suggestions["suggestions_policy_v1"] = [row for row in suggestions["suggestions_policy_v1"]
        if row["id"] in cohort.suggestion_ids]
    if automatic:
        assert all(row["status"] == "pending" and row["snooze_until"] is None
            for row in suggestions["suggestions_policy_v1"])
    catalog = g.diagnostics.emitter._store.catalog
    diagnostic = data_helpers.copied_tables(catalog, ("metrics", "lineage"))
    lineage = [row for row in diagnostic["lineage"] if row["conversation_id"] == conversation]
    ids = {row["record_id"] for row in lineage}
    diagnostic = {"metrics": [row for row in diagnostic["metrics"] if row["id"] in ids], "lineage": lineage}
    assert ids and len(diagnostic["metrics"]) == len(ids)
    disk_ids = set()
    metric_path = g.diagnostics.emitter.path
    for path in (metric_path, Path(str(metric_path) + ".1")):
        if path.is_file():
            disk_ids.update(json.loads(line)["record_id"] for line in path.read_text().splitlines())
    assert ids <= disk_ids, "flush must persist the actual sourced metrics"
    context_path = g.persister.base_path / f"{cohort.user}.json"
    files = {str(cohort.archive_path): Path(cohort.archive_path).read_bytes(),
        str(context_path): context_path.read_bytes()}
    targets = {target.id: target for target in g.targets.values()}
    for row in shared["wake_clips_v1"]:
        target = targets[row["target_id"]]
        path = Path(target.root) / "policy_v1" / (row["id"] + target.extension)
        files[str(path)] = path.read_bytes()
    index = index_values(g.memory.manager.long_term._collection.get(ids=[cohort.fact],
        include=["embeddings", "documents", "metadatas"]))
    assert index["ids"] == [cohort.fact]
    return {"tables": {str(g.root / "privacy.db"): shared, str(g.root / "ambient.db"): ambient,
        str(g.events.db_path): events, str(g.suggestions.store.path): suggestions,
        str(catalog): diagnostic}, "files": files, "index": index}


def index_values(result):
    return {key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in result.items() if key in {"ids", "embeddings", "documents", "metadatas"}}


async def payload(g, cohort):
    return await asyncio.to_thread(selected_payload, g, cohort)


async def snapshot_payload(g, cohort):
    data = await asyncio.to_thread(selected_payload, g, cohort, automatic=True)
    paths = {Path(target.root) for target in g.targets.values()}
    paths.update((Path(g.ambient.audio_dir), g.persister.base_path, g.root / "chroma"))
    roots = {str(path): (path.stat().st_dev, path.stat().st_ino) for path in paths}
    catalog = str(g.diagnostics.emitter._store.catalog)
    metrics = {row["id"] for row in data["tables"][catalog]["metrics"]}
    metric_path = g.diagnostics.emitter.path
    diagnostic_files = {}
    for path in (metric_path, Path(str(metric_path) + ".1")):
        if path.is_file():
            lines = [line for line in path.read_text().splitlines()
                if json.loads(line)["record_id"] in metrics]
            if lines:
                diagnostic_files[str(path)] = ("\n".join(lines) + "\n").encode()
    saved = data_helpers.DataSnapshot(data, roots, diagnostic_files, str(g.root / "chroma"),
        g.memory.manager.long_term.index_collection_name, chromadb.__version__)
    witness = await asyncio.to_thread(capture_preservation, g.root)
    metadata = json.loads(json.dumps(g.memory.manager.long_term._collection.metadata))
    lineage = await capture_index_lineage(g, data["index"])
    trusted = ReplayWitness(str(g.root.resolve()), witness.schemas, dict(roots),
        saved.index_path, saved.index_collection, saved.sdk_version, metadata, lineage)
    return saved, trusted


def validate_locator(root, snapshot, witness):
    """Reject destination drift without opening an SDK client or making dirs."""
    validate_restore_locator(root, snapshot, witness=witness)
    if type(witness) is not ReplayWitness or str(Path(root).resolve()) != witness.root:
        raise ReplayRejected("restore_root_mismatch")
    if (type(snapshot.index_path) is not str or snapshot.index_path != witness.index_path
        or Path(witness.index_path).resolve() != Path(witness.root) / "chroma"
        or type(snapshot.index_collection) is not str
        or snapshot.index_collection != witness.index_collection
        or snapshot.sdk_version != witness.sdk_version
        or chromadb.__version__ != witness.sdk_version
        or snapshot.root_identities != witness.root_identities):
        raise ReplayRejected("restore_index_locator_mismatch")
    for path, identity in witness.root_identities.items():
        target = Path(path)
        if not target.is_dir() or target.is_symlink():
            raise ReplayRejected("restore_root_mismatch")
        info = target.stat()
        if (info.st_dev, info.st_ino) != identity:
            raise ReplayRejected("restore_root_mismatch")
    if not (Path(witness.index_path) / "chroma.sqlite3").is_file():
        raise ReplayRejected("restore_index_missing")


def read_index(root, snapshot, witness):
    validate_locator(root, snapshot, witness)
    client = chromadb.PersistentClient(path=witness.index_path,
        settings=Settings(anonymized_telemetry=False))
    try:
        collection = client.get_collection(name=witness.index_collection,
            embedding_function=SyntheticEmbedding())
        if collection.metadata != witness.index_metadata:
            raise ReplayRejected("restore_index_metadata_mismatch")
        return index_values(collection.get(include=["embeddings", "documents", "metadatas"]))
    finally:
        client.close()


def restore_payload(root, snapshot, witness):
    """Preflight all stores before the first data write; no authority rollback."""
    validate_locator(root, snapshot, witness)
    client = chromadb.PersistentClient(path=witness.index_path,
        settings=Settings(anonymized_telemetry=False))
    try:
        collection = client.get_collection(name=witness.index_collection,
            embedding_function=SyntheticEmbedding())
        if collection.metadata != witness.index_metadata:
            raise ReplayRejected("restore_index_metadata_mismatch")
        current_index = index_values(collection.get(include=["embeddings", "documents", "metadatas"]))
        prepared = preflight_restore(root, snapshot, index_state=current_index,
            index_version=chromadb.__version__, schema_witness=witness.schemas, witness=witness)
        for path, tables in prepared.data["tables"].items():
            with sqlite3.connect(path) as db:
                db.execute("PRAGMA foreign_keys=ON")
                for table, records in tables.items():
                    data_helpers.insert_rows(db, table, records)
                assert not db.execute("PRAGMA foreign_key_check").fetchall()
        for path, raw in prepared.data["files"].items():
            Path(path).write_bytes(raw)
        for path, raw in prepared.diagnostic_files.items():
            Path(path).write_bytes(raw)
        values = {key: value for key, value in prepared.data["index"].items()
            if value is not None and not (key == "metadatas" and all(item is None for item in value))}
        if values.get("ids"):
            collection.upsert(**values)
        after_index = index_values(collection.get(include=["embeddings", "documents", "metadatas"]))
        after = capture_preservation(root, index_state=after_index)
        assert_preserved(prepared.preservation, after, inserted_rows=prepared)
        return prepared
    finally:
        client.close()


async def clear_browser(g, cohort):
    connection = await peer(g, client=cohort.browser_client)
    try:
        await until(lambda: g.bus._retention.pending_conversation(cohort.source.conversation_id) == 0)
    finally:
        await connection.close()


async def observed_step(g, operation, predicate):
    """Drive journal retry time without changing any capture's fixed deadline."""
    async with asyncio.timeout(4):
        while True:
            await g.owner.reconcile()
            step = await producer_step(g.graph, operation)
            if step is not None and predicate(step):
                return step
            g.clock.now += .1
            await asyncio.sleep(.01)


async def erase_with_real_producer(g, subject, foreign, subject_gate, unrelated_gate):
    """P may finish for S after S exits while the same model still owns U."""
    processor = g.graph.pipeline.command_processor
    original_u = foreign.source.to_dict()
    subject_caller = launch(g.graph, subject.source, .125)
    unrelated_caller = None
    try:
        await until(subject_gate.entered.is_set)
        assert subject_gate.scope == ((subject.source,), 0)
        unrelated_caller = launch(g.graph, foreign.source, .25)
        await until(lambda: processor.resource_status()["native"]["operations"] == 2)
        assert not unrelated_gate.entered.is_set()
        operation = await g.owner.request(subject.source.conversation_id, memory_actor(g))
        subject_caller.cancel()
        await asyncio.gather(subject_caller, return_exceptions=True)
        pending = await observed_step(g, operation,
            lambda item: item["state"] == "pending" and bool(item["pending_count"]))
        assert pending["pending_count"] and processor.conversation_busy(subject.source.conversation_id)
        assert not subject_gate.returned.is_set()
        subject_gate.release.set()
        await until(unrelated_gate.entered.is_set)
        await until(lambda: not processor.conversation_busy(subject.source.conversation_id))
        complete = await observed_step(g, operation, lambda item: item["state"] == "complete")
        assert complete["pending_count"] == 0
        assert unrelated_gate.scope == ((foreign.source,), 0)
        assert not unrelated_gate.returned.is_set() and not unrelated_caller.done()
        assert processor.conversation_busy(foreign.source.conversation_id)
        assert g.privacy.sources_available((foreign.source,)) and foreign.source.to_dict() == original_u
        receipt = await g.privacy.acall(g.privacy.ledger.erasures.snapshot, operation["id"])
        assert receipt["state"] != "complete"
        assert all(any(step["participant"] == name and step["state"] != "complete"
            for step in receipt["steps"]) for name in ("diagnostics", "streams", "voices"))
        assert g.graph.capture_control.snapshot()["target_muted"] is False
        unrelated_gate.release.set()
        result = await unrelated_caller
        assert result.success and result.text == "Synthetic U command"
        await until(lambda: not processor.snapshot_operations())
        assert subject_gate.completed.is_set() and unrelated_gate.completed.is_set()
        assert foreign.source.to_dict() == original_u
        return operation
    finally:
        subject_gate.release.set()
        unrelated_gate.release.set()
        await asyncio.gather(*(task for task in (subject_caller, unrelated_caller)
            if task is not None), return_exceptions=True)


async def settle(g, operation):
    async with asyncio.timeout(6):
        while True:
            await g.owner.reconcile()
            selected = await data_helpers.steps(g, operation)
            if all(step["state"] == "complete" for step in selected
                if (step["participant"], step["version"]) not in REMAINING_GUARDS):
                break
            g.clock.now += .1
            await asyncio.sleep(.005)
    assert {step["participant"] for step in selected} == PARTICIPANTS
    assert {(step["participant"], step["version"]) for step in selected
        if step["state"] != "complete"} == REMAINING_GUARDS
    state = await g.privacy.acall(g.privacy.ledger.erasures.snapshot, operation["id"])
    assert state["state"] != "complete" and state["revoked"]
    assert g.graph.producer_erasure.inventory.verify().valid


async def assert_foreign_unchanged(g, cohort, saved, *, rehydrated=False):
    assert g.privacy.sources_available((cohort.source,))
    assert await g.ambient.read_audio(cohort.ambient_id) is not None
    turns = await g.conversations.get_turns(conversation_id=cohort.source.conversation_id)
    assert len(turns) == 1
    context = await g.manager.aget_or_create(cohort.user)
    assert context.compacted_summary == "Synthetic summary" and context.summary_sources == (cohort.source,)
    assert bool(context.conversation_history) is not rehydrated
    for identity in cohort.voice_ids:
        review = await g.voices.acall(g.voices.ledger.review_capture, identity, voice_actor(g))
        assert review["audio_base64"] and review["transcript"] == "Synthetic reviewed clip"
    assert data_helpers.normalize(await payload(g, cohort)) == data_helpers.normalize(saved)


async def assert_subject_erased(g, cohort):
    await data_helpers.assert_subject_erased(g, cohort)
    assert not g.privacy.sources_available((cohort.source,))
