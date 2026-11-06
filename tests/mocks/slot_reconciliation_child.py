"""Public-store data points for the compiled O/N/R/F reconciliation rehearsal.

The process entry/admission protocol is deliberately added only after the
production derivation contract is fixed.  All imports of project/SDK code stay
inside functions, after the compiled child's preimport bootstrap.
"""
from __future__ import annotations

import asyncio
from contextlib import closing
import json
from pathlib import Path
import sqlite3


POINTS = {
    "O": {"label": "slot ordinary O", "tag": "slot_o", "pcm": .20, "total_ms": 20.},
    "N": {"label": "slot ordinary N", "tag": "slot_n", "pcm": .30, "total_ms": 30.},
    "N-only": {"label": "slot ordinary N-only", "tag": "slot_n_only", "pcm": .40, "total_ms": 40.},
}
FAMILIES = frozenset({"semantic", "captures", "collectors", "contexts", "events",
    "suggestions", "diagnostics", "voices", "streams"})


def rows(path, table):
    """Read a synthetic persisted table without creating it or keeping a reader."""
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise AssertionError("reconciliation_read_requires_existing_regular_database")
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        return [dict(row) for row in db.execute('SELECT * FROM "' + table.replace('"', '""') + '"')]


def original_scope(sources):
    """The same already-declared acquisition scope is used in every data point."""
    from src.memory.fact_ledger import MemorySource
    if (type(sources) is not tuple or len(sources) != 3
            or any(type(source) is not MemorySource for source in sources)
            or len({source.capture_id for source in sources}) != 3
            or len({source.conversation_id for source in sources}) != 1
            or len({source.origin for source in sources}) != 1
            or any(source.expires_at != sources[0].expires_at for source in sources)
            or any(source.captured_at != sources[0].captured_at for source in sources)
            or sources[0].expires_at - sources[0].captured_at != 48 * 3600):
        raise AssertionError("reconciliation_requires_original_U_scope")
    return tuple(source.to_dict() for source in sources)


def voice_scope(point, sources):
    """VoiceLedger is write-once by model and acquisition source set."""
    original_scope(sources)
    if point == "O":
        return sources
    if point == "N":
        return sources[:2]
    if point == "N-only":
        return sources[0], sources[2]
    raise ValueError("unknown_reconciliation_data_point")


async def publish_current(g, reference):
    """A new current peer, never a restored namespace, connection or ACK."""
    from src.dashboard.live_event_bus import LiveEvent, LiveEventType
    from tests.mocks.application_restore import peer
    connection = await peer(g)
    try:
        await g.bus.publish(LiveEvent(LiveEventType.TURN,
            payload={"stt": reference["label"]}, sources=reference["sources"]))
        frame = await connection.frame("turn")
        assert frame["receipt"]
        return {"frame": frame, "client": connection.client,
            "reader_namespace": g.reader.namespace}
    finally:
        await connection.close()


async def append_ordinary(g, point, sources):
    """Create one real append-only point in all nine managed families.

    The STT correction is protected manual feedback: the public collector
    requires a real correction. Wake, unmarked turns and exports are the
    ordinary subset. No used flag, profile or account/consent changes occur.
    Source identities and deadlines are checked again on return.
    """
    import numpy as np
    from src.ambient.types import AmbientUtterance
    from src.memory.producer_context import producer_context
    from tests.mocks.restore_application_graph import until
    scope_before = original_scope(sources)
    if point not in POINTS:
        raise ValueError("unknown_reconciliation_data_point")
    spec = POINTS[point]
    primary = sources[0]
    scope = (primary,)
    assert g.privacy.sources_available(sources)
    label, tag = spec["label"], spec["tag"]
    audio = np.full(16000, spec["pcm"], np.float32)
    reference = {"schema_version": 1, "point": point, "label": label,
        "sources": scope, "original_scope": sources, "families": {}}
    families = reference["families"]
    families["semantic"] = {"id": await g.memory.manager.long_term.astore_fact(
        label + " sourced automatic fact", "fact", sources=scope)}

    utterance = AmbientUtterance("synthetic-room", primary.captured_at,
        primary.captured_at + 1, label + " capture transcript", capture_id=primary.capture_id,
        conversation_id=primary.conversation_id, expires_at=primary.expires_at)
    utterance_id = await g.ambient.add(utterance)
    audio_path = await g.archive.write("synthetic-room", utterance_id, audio, source=primary)
    assert audio_path and await g.ambient.set_audio_path(utterance_id, audio_path)
    assert await g.ambient.read_audio(utterance_id) is not None
    families["captures"] = {"id": utterance_id}

    existing = await asyncio.to_thread(rows, g.root / "privacy.db", "wake_clips_v1")
    old_ids = {row["id"] for row in existing}
    cursor = max((row["sequence"] for row in existing), default=0)
    assert g.wake.submit("synthetic-room", .99, audio, sources=scope)
    await until(lambda: g.wake._busy is None and not g.wake._queue)
    correction_id = await g.stt.add_correction(audio=audio, original_text=label + " original",
        corrected_text=label + " corrected", sources=scope, correction_sources=scope)
    assert correction_id is not None
    with producer_context("synthetic-room", scope):
        turn_id = await g.conversations.add_turn(label + " input", label + " response",
            input_sources=scope, response_sources=scope)
    assert turn_id is not None
    export = await g.conversations.export_for_training(g.targets["exports"].root,
        only_marked=False, after_sequence=cursor)
    assert export is not None
    try:
        assert export.available and export.expires_at <= primary.expires_at
        export_id = export.id
    finally:
        export.clear()
    current = await asyncio.to_thread(rows, g.root / "privacy.db", "wake_clips_v1")
    created = [row for row in current if row["id"] not in old_ids and row["published"] == 1
        and row["expires_at"] > g.clock.now]
    assert len(created) == 4
    assert {row["bucket"] for row in created} == {
        "accepted", "stt_correction", "conversation_turn", "conversation_export"}
    assert all(row["captured_at"] == primary.captured_at
        and row["expires_at"] <= primary.expires_at for row in created)
    families["collectors"] = {"ids": tuple(sorted(row["id"] for row in created)),
        "ordinary_ids": tuple(sorted(row["id"] for row in created
            if row["bucket"] != "stt_correction")),
        "protected_correction_ids": (correction_id,),
        "correction_id": correction_id, "turn_id": turn_id, "export_id": export_id}

    user = tag + "-" + primary.conversation_id
    context = await g.manager.aget_or_create(user)
    g.manager.add_turn(user, "user", label + " context turn", sources=scope)
    # UserContext is the public data object accepted by the public persister.
    # This does not claim that conversation_history itself is serialized.
    context.compacted_summary = label + " compacted summary"
    context.summary_sources = scope
    await g.manager._offload(g.persister.save, context)
    assert g.persister.load(user)["compacted_summary"] == label + " compacted summary"
    families["contexts"] = {"user": user}

    before_suggestions = {row["id"] for row in await asyncio.to_thread(
        rows, g.suggestions.store.path, "suggestions_policy_v1")}
    event_ids = []
    for _ in range(5):
        g.clock.now += .01
        trigger = await g.events.acall(g.events.log, "light." + tag + "_trigger", "turn_on", sources=scope)
        g.clock.now += .01
        target = await g.events.acall(g.events.log, "light." + tag + "_target", "turn_on", sources=scope)
        event_ids.extend((trigger.record_id, target.record_id))
    assert len(set(event_ids)) == 10
    families["events"] = {"ids": tuple(event_ids), "entity_prefix": "light." + tag}
    await g.events.acall(g.suggestions.generate_suggestions, .7, 50)
    candidates = [row for row in await asyncio.to_thread(
        rows, g.suggestions.store.path, "suggestions_policy_v1")
        if row["id"] not in before_suggestions and tag in row["pattern_json"]]
    assert candidates and all(row["status"] == "pending" and row["snooze_until"] is None
        for row in candidates)
    families["suggestions"] = {"ids": tuple(sorted(row["id"] for row in candidates))}

    catalog = g.diagnostics.emitter._store.catalog
    old_metric_ids = {row["id"] for row in await asyncio.to_thread(rows, catalog, "metrics")}
    g.diagnostics.emitter.emit_request(user_id="synthetic", zone_id=None, text=label,
        intent=None, path="fast", success=True,
        timings={"stt": 12., "total": spec["total_ms"]}, sources=scope)
    await g.diagnostics.emitter.flush()
    lineage = await asyncio.to_thread(rows, catalog, "lineage")
    admitted = {row["record_id"] for row in lineage
        if row["conversation_id"] == primary.conversation_id}
    metrics = [row for row in await asyncio.to_thread(rows, catalog, "metrics")
        if row["id"] not in old_metric_ids and row["id"] in admitted]
    assert len(metrics) == 1, "flush alone does not prove persisted sourced metrics"
    assert json.loads(metrics[0]["payload"])["total_ms"] == spec["total_ms"]
    disk_ids = set()
    path = g.diagnostics.emitter.path
    for candidate in (path, Path(str(path) + ".1")):
        if candidate.is_file():
            disk_ids.update(json.loads(line)["record_id"] for line in candidate.read_text().splitlines())
    assert metrics[0]["id"] in disk_ids
    families["diagnostics"] = {"ids": (metrics[0]["id"],), "total_ms": spec["total_ms"]}

    before_voice = {row["id"] for row in await asyncio.to_thread(
        rows, g.root / "privacy.db", "voice_fingerprints")}
    selected = voice_scope(point, sources)
    assert g.voices.observe(g.encoder, audio, selected, transcript=label + " raw voice",
        room_id="synthetic-room")
    await until(lambda: g.voices.resource_status()["pending"] == 0)
    voices = [row for row in await asyncio.to_thread(
        rows, g.root / "privacy.db", "voice_fingerprints") if row["id"] not in before_voice]
    assert len(voices) == 1, "distinct original source sets must create an actual raw observation"
    review = await g.voices.acall(g.voices.ledger.review_capture, voices[0]["id"], g.voice_actor)
    assert review["transcript"] == label + " raw voice" and review["audio_base64"]
    families["voices"] = {"id": voices[0]["id"], "sources": selected}
    families["streams"] = await publish_current(g, reference)
    assert set(families) == FAMILIES
    assert original_scope(sources) == scope_before
    audio = context = review = utterance = None
    return reference


async def read_ordinary(g, reference, *, publish=False):
    """Read actual current stores; preserve full metadata for edge validation."""
    from tests.mocks.application_restore import index_values
    assert set(reference["families"]) == FAMILIES
    original_scope(reference["original_scope"])
    assert g.privacy.sources_available(reference["original_scope"])
    selected, output = reference["families"], {}
    fact = await g.privacy.acall(g.privacy.ledger.get, selected["semantic"]["id"])
    assert fact is not None
    index = g.memory.manager.long_term._collection.get(ids=[fact.id],
        include=["embeddings", "documents", "metadatas"])
    output["semantic"] = {"fact": fact.to_dict(), "index": index_values(index)}
    capture_rows = [row for row in await asyncio.to_thread(rows, g.root / "ambient.db", "utterances")
        if row["id"] == selected["captures"]["id"]]
    assert len(capture_rows) == 1
    capture_audio = await g.ambient.read_audio(selected["captures"]["id"])
    assert capture_audio is not None
    output["captures"] = {"rows": capture_rows, "audio": capture_audio}

    clip_ids = set(selected["collectors"]["ids"])
    clip_rows = [row for row in await asyncio.to_thread(rows, g.root / "privacy.db", "wake_clips_v1")
        if row["id"] in clip_ids]
    assert len(clip_rows) == len(clip_ids)
    targets = {target.id: target for target in g.targets.values()}
    files = {}
    for row in clip_rows:
        target = targets[row["target_id"]]
        with g.clips.reading(target, reference["sources"]) as admitted:
            assert admitted
            files[row["id"]] = (Path(target.root) / "policy_v1" /
                (row["id"] + target.extension)).read_bytes()
    corrections = await g.stt.get_training_pairs()
    try:
        matches = [item for item in corrections if item.id == selected["collectors"]["correction_id"]]
        if matches:
            correction = matches[0]
            assert correction.available
            correction_data = {"id": correction.id, "original_text": correction.original_text,
                "corrected_text": correction.corrected_text, "used": correction.used,
                "expires_at": correction.expires_at,
                "sources": tuple(source.to_dict() for source in correction.sources)}
        else:
            # A real mark_used receipt removes the record from the public
            # training reader. Its protected row must still retain used=1.
            identity = selected["collectors"]["correction_id"]
            records = [row for row in await asyncio.to_thread(
                rows, g.root / "privacy.db", "stt_corrections_v1") if row["id"] == identity]
            parents = [row for row in clip_rows if row["id"] == identity]
            assert len(records) == len(parents) == 1 and records[0]["used"] == 1
            correction_data = {"id": identity, "original_text": records[0]["original_text"],
                "corrected_text": records[0]["corrected_text"], "used": True,
                "expires_at": parents[0]["expires_at"],
                "sources": tuple(json.loads(parents[0]["sources_json"]))}
    finally:
        for item in corrections:
            item.clear()
    turns = await g.conversations.get_turns(conversation_id=reference["sources"][0].conversation_id)
    try:
        turn = next(item for item in turns if item.id == selected["collectors"]["turn_id"])
        assert turn.available
        turn_data = {"id": turn.id, "user_input": turn.user_input,
            "assistant_response": turn.assistant_response, "quality": turn.quality.value,
            "correction": turn.correction, "expires_at": turn.expires_at,
            "sources": tuple(source.to_dict() for source in turn.sources)}
    finally:
        for item in turns:
            item.clear()
    output["collectors"] = {"rows": clip_rows, "files": files,
        "correction": correction_data, "turn": turn_data}
    context = await g.manager.aget_or_create(selected["contexts"]["user"])
    output["contexts"] = {"summary": context.compacted_summary,
        "sources": tuple(source.to_dict() for source in context.summary_sources),
        "persisted": g.persister.load(selected["contexts"]["user"])}
    event_ids = set(selected["events"]["ids"])
    output["events"] = [row for row in await asyncio.to_thread(rows, g.events.db_path, "events_policy_v1")
        if row["id"] in event_ids]
    proposal_ids = set(selected["suggestions"]["ids"])
    proposals = await g.events.acall(g.suggestions.store.records)
    output["suggestions"] = [row for row in await asyncio.to_thread(
        rows, g.suggestions.store.path, "suggestions_policy_v1") if row["id"] in proposal_ids]
    assert proposal_ids <= {item.id for item in proposals}
    catalog = g.diagnostics.emitter._store.catalog
    metric_ids = set(selected["diagnostics"]["ids"])
    output["diagnostics"] = {"metrics": [row for row in await asyncio.to_thread(rows, catalog, "metrics")
        if row["id"] in metric_ids], "lineage": [row for row in await asyncio.to_thread(rows, catalog, "lineage")
        if row["record_id"] in metric_ids]}
    identity = selected["voices"]["id"]
    output["voices"] = {"review": await g.voices.acall(g.voices.ledger.review_capture, identity, g.voice_actor),
        "rows": [row for row in await asyncio.to_thread(rows, g.root / "privacy.db", "voice_fingerprints")
            if row["id"] == identity]}
    output["streams"] = await publish_current(g, reference) if publish else {
        "historical_obligation": selected["streams"], "current_reader_namespace": g.reader.namespace}
    assert set(output) == FAMILIES
    return output


def write_phase_bag(e, path, control, mode, payload):
    """Preserve each branch's original code/state identity in a typed checkpoint."""
    import hashlib
    path = Path(path)
    if path.exists() or type(payload) is not dict:
        raise ValueError("reconciliation_checkpoint_exists_or_invalid")
    codec = e.DataCodec(path.parent / "reconciliation-blobs")
    encoded = codec.encode(payload)
    if sum(codec.manifest.values()) > 64 * 1024 * 1024:
        raise ValueError("reconciliation_checkpoint_too_large")
    return e._atomic_json(path, {"schema_version": 1,
        "unit": "T024-F1/material-code-data-reconciliation", "mode": mode,
        "source_commit": control["source_commit"], "runtime_root": control["runtime_root"],
        "state_root": control["state_root"], "release_sha256": control["release_sha256"],
        "helper_manifest_sha256": hashlib.sha256(json.dumps(
            control["island_files"], sort_keys=True).encode()).hexdigest(),
        "payload": encoded, "blob_manifest": codec.manifest})


def read_phase_bag(e, path, source_control, mode):
    """Validate the original branch's control rather than relabel its witness."""
    import hashlib
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("reconciliation_checkpoint_location")
    raw = path.read_bytes()
    document = json.loads(raw)
    fields = {"schema_version", "unit", "mode", "source_commit", "runtime_root", "state_root",
        "release_sha256", "helper_manifest_sha256", "payload", "blob_manifest"}
    if (type(document) is not dict or set(document) != fields or document["schema_version"] != 1
            or document["unit"] != "T024-F1/material-code-data-reconciliation" or document["mode"] != mode
            or any(document[key] != source_control[key] for key in
                ("source_commit", "runtime_root", "state_root", "release_sha256"))
            or document["helper_manifest_sha256"] != hashlib.sha256(json.dumps(
                source_control["island_files"], sort_keys=True).encode()).hexdigest()):
        raise ValueError("reconciliation_original_branch_identity")
    manifest = document["blob_manifest"]
    if (type(manifest) is not dict or len(manifest) > 10000
            or any(type(size) is not int or not 0 <= size <= 16 * 1024 * 1024
                for size in manifest.values()) or sum(manifest.values()) > 64 * 1024 * 1024):
        raise ValueError("reconciliation_blob_manifest")
    codec = e.DataCodec(path.parent / "reconciliation-blobs", manifest)
    payload = codec.decode(document["payload"])
    if type(payload) is not dict or codec.used != set(manifest):
        raise ValueError("reconciliation_checkpoint_payload")
    return payload, hashlib.sha256(raw).hexdigest()


async def seed_ordinary_O(args, control, guard, e):
    """Create real O data and separate manual sentinels before any upgrade."""
    from dataclasses import replace
    import time
    from uuid import uuid4
    import pytest
    from src.dashboard.authority import DashboardRole
    from src.llm.cloud_consent import cloud_reasoner_allowed
    from tests.mocks import application_restore as replay
    from tests.mocks.managed_slot_graph import managed_application_graph
    from tests.mocks.memory_graph import Clock, MemoryClient
    from tests.mocks.memory_sources import memory_source
    from tests.mocks.restore_preservation import assert_manual_sentinels, seed_manual_sentinels
    from tests.mocks.slot_rollback_child import _resources

    clock, patch = Clock(time.time()), pytest.MonkeyPatch()
    marker = len(MemoryClient._created)
    try:
        async with managed_application_graph(Path(args.state_root), patch, clock, control=control) as g:
            e._graph_contract(g, replay)
            assert g.graph.config["reasoner"]["cloud"]["consent"] is True
            assert cloud_reasoner_allowed(g.graph.config["reasoner"]) is True
            subject, foreign = await replay.seed_all(g)
            manual_sources = await replay.manual_voice_inputs(g, foreign)
            manual = await seed_manual_sentinels(g, sources=manual_sources,
                voice_ids=tuple(foreign.voice_ids), suggestion_ids=tuple(foreign.suggestion_ids),
                root_lease=replay.root_lease(g), delete_profile=True)
            await assert_manual_sentinels(g, manual)
            primary = memory_source(now=clock.now)
            sources = (primary, replace(primary, capture_id=str(uuid4())),
                replace(primary, capture_id=str(uuid4())))
            reference = await append_ordinary(g, "O", sources)
            observed = await read_ordinary(g, reference, publish=True)
            authority = g.graph.dashboard_authority
            forward_guard = await authority.issue_credential(replay.root_lease(g),
                subject="synthetic-forward-withdrawal-" + uuid4().hex,
                role=DashboardRole.OPERATOR, expires_at=authority._now() + 3600)
            payload = {"clock_now": clock.now, "subject": subject, "foreign": foreign,
                "manual": manual, "ordinary_sources": sources, "O": reference,
                "O_observed": observed,
                "forward_guard": {"identity": forward_guard.identity, "token": forward_guard.token}}
            guard.audit()
        MemoryClient.close_since(marker)
        resources = _resources(g)
        digest = write_phase_bag(e, Path(args.checkpoint), control, "seed-O", payload)
        return resources, digest
    finally:
        MemoryClient.close_since(marker)
        patch.undo()


async def advance_ordinary_N(args, control, guard, e, *, source_control, source_path):
    """Append real N data, withdraw authority, drain, and persist next consent."""
    from dataclasses import asdict
    import pytest
    import yaml
    from tests.mocks import application_restore as replay
    from tests.mocks import integrated_restore as data_helpers
    from tests.mocks.managed_slot_graph import managed_application_graph
    from tests.mocks.memory_graph import Clock, MemoryClient
    from tests.mocks.producer_erasure_graph import NativeGate
    from tests.mocks.slot_rollback_child import _current_manual_decisions, _resources

    saved, original_sha = read_phase_bag(e, source_path, source_control, "seed-O")
    clock, patch = Clock(saved["clock_now"]), pytest.MonkeyPatch()
    subject_gate, unrelated_gate = NativeGate(), NativeGate()
    marker = len(MemoryClient._created)
    try:
        async with managed_application_graph(Path(args.state_root), patch, clock, control=control,
                subject_gate=subject_gate, unrelated_gate=unrelated_gate) as g:
            e._graph_contract(g, replay)
            sources = saved["ordinary_sources"]
            assert original_scope(sources) == original_scope(saved["O"]["original_scope"])
            ordinary_N = await append_ordinary(g, "N", sources)
            ordinary_new = await append_ordinary(g, "N-only", sources)
            mixed = await protect_mixed_feedback(g, ordinary_new, sources)
            decisions = await _current_manual_decisions(g, saved)
            operation = await replay.erase_with_real_producer(g, saved["subject"], saved["foreign"],
                subject_gate, unrelated_gate)
            for _ in range(12):
                await g.owner.reconcile()
                clock.now += .1
                await asyncio.sleep(.01)
            steps = await data_helpers.steps(g, operation)
            guards = {("diagnostics", 1), ("streams", 2), ("voices", 2)}
            assert all(any(row["participant"] == part and row["version"] == version
                and row["state"] == "pending" for row in steps) for part, version in guards)
            assert not g.privacy.sources_available((saved["subject"].source,))
            assert g.privacy.sources_available(sources)
            observed = await read_ordinary(g, ordinary_N, publish=True)
            observed_new = await read_ordinary(g, ordinary_new, publish=True)
            assert g.graph.dashboard_authority.resolve(saved["forward_guard"]["token"]) is not None
            changes = g.graph.configuration_changes
            assert changes is not None
            pending = changes.queue_consent_withdrawal(replay.root_lease(g))
            assert pending.previous_consent is True and pending.next_generation_only is True
            assert changes.receipt is None
            guard.audit()
        assert subject_gate.returned.is_set() and unrelated_gate.returned.is_set()
        assert subject_gate.completed.is_set() and unrelated_gate.completed.is_set()
        MemoryClient.close_since(marker)
        resources = _resources(g, native=True)
        settings = Path(args.state_root).parents[1] / "config/settings.yaml"
        persisted = yaml.safe_load(settings.read_text())
        assert persisted["reasoner"]["cloud"]["consent"] is False
        receipt = changes.receipt
        assert receipt is not None and receipt.consent is False and receipt.next_generation_only is True
        payload = {"clock_now": clock.now, "O_source_sha256": original_sha,
            "N": ordinary_N, "N_only": ordinary_new, "N_observed": observed,
            "N_only_observed": observed_new, "decisions": decisions, "operation": operation,
            "steps": steps, "consent_transition": asdict(receipt), "mixed_sentinels": mixed}
        digest = write_phase_bag(e, Path(args.checkpoint), control, "advance-N", payload)
        return resources, digest
    finally:
        subject_gate.release.set()
        unrelated_gate.release.set()
        MemoryClient.close_since(marker)
        patch.undo()


async def protect_mixed_feedback(g, reference, sources):
    """Create real used/feedback transitions separate from ordinary turn markers."""
    from src.memory.producer_context import producer_context

    identity = reference["families"]["collectors"]["correction_id"]
    pairs = await g.stt.get_training_pairs()
    try:
        selected = next(item for item in pairs if item.id == identity)
        assert selected.available and not selected.used
        await g.stt.mark_used([selected])
        assert selected.used is True
    finally:
        for item in pairs:
            item.clear()
    scope = (sources[0],)
    clips = await asyncio.to_thread(rows, g.root / "privacy.db", "wake_clips_v1")
    cursor = max(row["sequence"] for row in clips)
    with producer_context("synthetic-room", scope):
        old = await g.conversations.add_turn("Latest protected feedback input",
            "Latest protected feedback response", input_sources=scope, response_sources=scope)
    assert old
    views = await g.conversations.get_turns(conversation_id=sources[0].conversation_id)
    try:
        selected = next(item for item in views if item.id == old)
        assert await g.conversations.mark_response(selected, "corrected",
            "Latest protected manual correction", sources=scope)
    finally:
        for item in views:
            item.clear()
    turns = await g.conversations.get_turns(conversation_id=sources[0].conversation_id)
    try:
        assert old not in {item.id for item in turns}
        successors = [item for item in turns if item.quality.value == "corrected"
            and item.correction == "Latest protected manual correction"]
        assert len(successors) == 1 and successors[0].id != old
        current = successors[0].id
    finally:
        for item in turns:
            item.clear()
    export = await g.conversations.export_for_training(g.targets["exports"].root,
        only_marked=True, after_sequence=cursor)
    assert export is not None
    try:
        assert export.available and export.expires_at <= sources[0].expires_at
        export_id = export.id
    finally:
        export.clear()
    return {"used_correction_id": identity, "superseded_turn_id": old,
        "current_turn_id": current, "protected_export_id": export_id}


async def assert_withdrawals_current(g, seed, latest):
    """Check current authority through real readers without relabeling provenance."""
    from src.llm.cloud_consent import cloud_reasoner_allowed
    from tests.mocks.slot_rollback_child import _find_user
    decisions = latest["decisions"]
    assert not g.privacy.sources_available((seed["subject"].source,))
    assert g.privacy.sources_available(seed["ordinary_sources"])
    authority = g.graph.dashboard_authority
    assert authority.resolve(seed["manual"].home_token) is None
    assert not [r for r in await asyncio.to_thread(rows, authority.path, "credentials")
        if r["id"] == decisions["credential_id"]]
    profiles = [r for r in await asyncio.to_thread(rows, g.root / "privacy.db", "voice_profiles")
        if r["id"] == decisions["latest_profile_id"]]
    assert profiles == decisions["latest_profile_rows"]
    user = await asyncio.to_thread(_find_user, g.root, seed["manual"].user_id)
    assert user["permission_level"] == decisions["latest_user"]["permission_level"]
    assert g.graph.config["reasoner"]["cloud"]["consent"] is False
    assert cloud_reasoner_allowed(g.graph.config["reasoner"]) is False
    mixed = latest["mixed_sentinels"]
    corrections = await asyncio.to_thread(rows, g.root / "privacy.db", "stt_corrections_v1")
    current = [row for row in corrections if row["id"] == mixed["used_correction_id"]]
    assert len(current) == 1 and current[0]["used"] == 1
    pairs = await g.stt.get_training_pairs()
    try:
        assert mixed["used_correction_id"] not in {item.id for item in pairs}
    finally:
        for item in pairs:
            item.clear()
    turns = await g.conversations.get_turns(conversation_id=seed["ordinary_sources"][0].conversation_id)
    try:
        assert mixed["superseded_turn_id"] not in {item.id for item in turns}
        current = [item for item in turns if item.id == mixed["current_turn_id"]]
        assert len(current) == 1 and current[0].quality.value == "corrected"
        assert current[0].correction == "Latest protected manual correction"
    finally:
        for item in turns:
            item.clear()
    parent_rows = await asyncio.to_thread(rows, g.root / "privacy.db", "conversation_export_parents_v1")
    assert any(row["export_id"] == mixed["protected_export_id"]
        and row["parent_id"] == mixed["current_turn_id"] for row in parent_rows)


async def assert_N_only_ordinary_absent(g, reference):
    """Inspect ordinary records while retaining current manual STT corrections."""
    selected = reference["families"]
    assert await g.privacy.acall(g.privacy.ledger.get, selected["semantic"]["id"]) is None
    assert g.memory.manager.long_term._collection.get(ids=[selected["semantic"]["id"]])["ids"] == []
    assert not [r for r in await asyncio.to_thread(rows, g.root / "ambient.db", "utterances")
        if r["id"] == selected["captures"]["id"]]
    clips = await asyncio.to_thread(rows, g.root / "privacy.db", "wake_clips_v1")
    clip_ids = {r["id"] for r in clips}
    assert not set(selected["collectors"]["ordinary_ids"]) & clip_ids
    assert set(selected["collectors"]["protected_correction_ids"]) <= clip_ids
    assert g.persister.load(selected["contexts"]["user"]) is None
    assert not set(selected["events"]["ids"]) & {
        r["id"] for r in await asyncio.to_thread(rows, g.events.db_path, "events_policy_v1")}
    assert not set(selected["suggestions"]["ids"]) & {
        r["id"] for r in await asyncio.to_thread(rows, g.suggestions.store.path, "suggestions_policy_v1")}
    assert not set(selected["diagnostics"]["ids"]) & {
        r["id"] for r in await asyncio.to_thread(rows, g.diagnostics.emitter._store.catalog, "metrics")}
    assert selected["voices"]["id"] not in {
        r["id"] for r in await asyncio.to_thread(rows, g.root / "privacy.db", "voice_fingerprints")}


async def observe_rollback_R(args, control, guard, e, *, seed_control, seed_path,
                             latest_control, latest_path):
    """Read O on R, reject S before model work, then withdraw another credential."""
    import time
    import pytest
    from src.memory.prompt_context import SourcesInvalidated
    from tests.mocks import application_restore as replay
    from tests.mocks.managed_slot_graph import managed_application_graph
    from tests.mocks.memory_graph import Clock, MemoryClient
    from tests.mocks.producer_erasure_graph import launch
    from tests.mocks.slot_rollback_child import _resources

    seed, seed_sha = read_phase_bag(e, seed_path, seed_control, "seed-O")
    latest, latest_sha = read_phase_bag(e, latest_path, latest_control, "advance-N")
    clock, patch = Clock(max(time.time(), latest["clock_now"])), pytest.MonkeyPatch()
    marker = len(MemoryClient._created)
    try:
        async with managed_application_graph(Path(args.state_root), patch, clock, control=control) as g:
            e._graph_contract(g, replay)
            await assert_withdrawals_current(g, seed, latest)
            calls, requests = list(g.models.leaf.calls), list(g.model_requests)
            with pytest.raises(SourcesInvalidated):
                await launch(g.graph, seed["subject"].source, .125)
            assert g.models.leaf.calls == calls and g.model_requests == requests
            await assert_N_only_ordinary_absent(g, latest["N_only"])
            await assert_N_only_ordinary_absent(g, latest["N"])
            observed = await read_ordinary(g, seed["O"], publish=True)
            authority, extra = g.graph.dashboard_authority, seed["forward_guard"]
            assert authority.resolve(extra["token"]) is not None
            assert await authority.revoke_credential(replay.root_lease(g), extra["identity"])
            assert authority.resolve(extra["token"]) is None
            assert not [r for r in await asyncio.to_thread(rows, authority.path, "credentials")
                if r["id"] == extra["identity"]]
            guard.audit()
        MemoryClient.close_since(marker)
        resources = _resources(g)
        digest = write_phase_bag(e, Path(args.checkpoint), control, "rollback-R",
            {"clock_now": clock.now, "O_source_sha256": seed_sha, "N_source_sha256": latest_sha,
             "R_observed": observed, "post_R_revoked_credential": extra["identity"]})
        return resources, digest
    finally:
        MemoryClient.close_since(marker)
        patch.undo()


async def observe_forward_F(args, control, guard, e, *, seed_control, seed_path,
                            latest_control, latest_path, rollback_control, rollback_path):
    """Read N on newer code while preserving the protected revision made on R."""
    import time
    import pytest
    from src.memory.prompt_context import SourcesInvalidated
    from tests.mocks import application_restore as replay
    from tests.mocks.managed_slot_graph import managed_application_graph
    from tests.mocks.memory_graph import Clock, MemoryClient
    from tests.mocks.producer_erasure_graph import launch
    from tests.mocks.slot_rollback_child import _resources

    seed, seed_sha = read_phase_bag(e, seed_path, seed_control, "seed-O")
    latest, latest_sha = read_phase_bag(e, latest_path, latest_control, "advance-N")
    rollback, rollback_sha = read_phase_bag(e, rollback_path, rollback_control, "rollback-R")
    clock, patch = Clock(max(time.time(), rollback["clock_now"])), pytest.MonkeyPatch()
    marker = len(MemoryClient._created)
    try:
        async with managed_application_graph(Path(args.state_root), patch, clock, control=control) as g:
            e._graph_contract(g, replay)
            await assert_withdrawals_current(g, seed, latest)
            calls, requests = list(g.models.leaf.calls), list(g.model_requests)
            with pytest.raises(SourcesInvalidated):
                await launch(g.graph, seed["subject"].source, .125)
            assert g.models.leaf.calls == calls and g.model_requests == requests
            authority = g.graph.dashboard_authority
            assert rollback["post_R_revoked_credential"] == seed["forward_guard"]["identity"]
            assert authority.resolve(seed["forward_guard"]["token"]) is None
            assert not [r for r in await asyncio.to_thread(rows, authority.path, "credentials")
                if r["id"] == rollback["post_R_revoked_credential"]]
            observed = await read_ordinary(g, latest["N"], publish=True)
            observed_new = await read_ordinary(g, latest["N_only"], publish=True)
            guard.audit()
        MemoryClient.close_since(marker)
        resources = _resources(g)
        digest = write_phase_bag(e, Path(args.checkpoint), control, "forward-F",
            {"clock_now": clock.now, "O_source_sha256": seed_sha, "N_source_sha256": latest_sha,
             "R_source_sha256": rollback_sha, "F_observed": observed, "F_N_only_observed": observed_new,
             "post_R_revoked_credential": rollback["post_R_revoked_credential"]})
        return resources, digest
    finally:
        MemoryClient.close_since(marker)
        patch.undo()


def phase_inputs(e, args, control):
    """Read a fixed plan that pins each original control and checkpoint path."""
    import hashlib
    path = Path(args.evidence_root) / "phase-inputs.json"
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise ValueError("reconciliation_phase_inputs_location")
    raw = path.read_bytes()
    plan = json.loads(raw)
    if (type(plan) is not dict or set(plan) != {"schema_version", "unit", "controls", "outputs"}
            or plan["schema_version"] != 1
            or plan["unit"] != "T024-F1/material-code-data-reconciliation"
            or set(plan["controls"]) != {"O", "N", "R", "F"}
            or set(plan["outputs"]) != {"O", "N", "R", "F"}):
        raise ValueError("reconciliation_phase_inputs_schema")
    controls = {}
    for branch, record in plan["controls"].items():
        if type(record) is not dict or set(record) != {"path", "sha256"}:
            raise ValueError("reconciliation_source_control_record")
        path = e._canonical(record["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("reconciliation_source_control_changed")
        original = json.loads(path.read_text())
        if (set(original) != set(control)
                or original["island_files"] != control["island_files"]
                or original["island_root"] != control["island_root"]
                or original["evidence_root"] != control["evidence_root"]):
            raise ValueError("reconciliation_source_control_scope")
        controls[branch] = original
    if (len({item["state_root"] for item in controls.values()}) != 4
            or len({item["runtime_root"] for item in controls.values()}) != 4
            or controls["O"]["source_commit"] != controls["R"]["source_commit"]
            or controls["N"]["source_commit"] != controls["F"]["source_commit"]
            or controls["O"]["source_commit"] == controls["N"]["source_commit"]):
        raise ValueError("reconciliation_code_pair_or_physical_branches")
    modes = {"seed-O": "O", "advance-N": "N", "rollback-R": "R", "forward-F": "F"}
    if controls[modes[args.mode]] != control or plan["outputs"][modes[args.mode]] != args.checkpoint:
        raise ValueError("reconciliation_current_branch_identity")
    for path in plan["outputs"].values():
        location = Path(path)
        if (not location.is_absolute() or location.is_symlink()
                or location.parent.resolve(strict=True) != location.parent
                or location.is_relative_to(Path(control["checkout_root"]))):
            raise ValueError("reconciliation_checkpoint_scope")
    return plan, controls, hashlib.sha256(raw).hexdigest()


def reconciliation_main():
    """Run the four-phase protocol under the real compiled-source guard."""
    import argparse
    import hashlib
    import importlib.util
    import os
    import sys
    parser = argparse.ArgumentParser(description="Compiled material O/N/R/F reconciliation")
    parser.add_argument("--mode", choices=("seed-O", "advance-N", "rollback-R", "forward-F"), required=True)
    for flag in ("control", "runtime", "state-root", "evidence-root", "checkpoint"):
        parser.add_argument("--" + flag, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("_kza_full_compiled_bootstrap",
        Path(__file__).with_name("compiled_restore_child.py"))
    e = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = e
    spec.loader.exec_module(e)
    control, release, guard, hits = e._bootstrap(args)
    own = Path(__file__).resolve().relative_to(Path(control["island_root"])).as_posix()
    assert own in control["island_files"]
    plan, controls, inputs_sha = phase_inputs(e, args, control)

    async def run():
        if args.mode == "seed-O":
            return await seed_ordinary_O(args, control, guard, e)
        common = {"seed_control": controls["O"], "seed_path": plan["outputs"]["O"],
            "latest_control": controls["N"], "latest_path": plan["outputs"]["N"]}
        if args.mode == "advance-N":
            return await advance_ordinary_N(args, control, guard, e,
                source_control=controls["O"], source_path=plan["outputs"]["O"])
        if args.mode == "rollback-R":
            return await observe_rollback_R(args, control, guard, e, **common)
        return await observe_forward_F(args, control, guard, e, **common,
            rollback_control=controls["R"], rollback_path=plan["outputs"]["R"])

    resources, digest = asyncio.run(run())
    origins = guard.audit()
    assert not hits
    for relative, expected in release["files"].items():
        assert hashlib.sha256((Path(args.runtime) / relative).read_bytes()).hexdigest() == expected
    result = {"schema_version": 1, "unit": "T024-F1/material-code-data-reconciliation",
        "mode": args.mode, "pid": os.getpid(), "source_commit": control["source_commit"],
        "runtime_root": args.runtime, "state_root": args.state_root,
        "phase_inputs_sha256": inputs_sha, "checkpoint_sha256": digest,
        "module_origins": origins, "resources": resources,
        "sys_flags": {"isolated": int(sys.flags.isolated), "no_site": int(sys.flags.no_site),
            "no_user_site": int(sys.flags.no_user_site), "ignore_environment": int(sys.flags.ignore_environment),
            "dont_write_bytecode": int(sys.dont_write_bytecode)}}
    e._atomic_json(Path(args.evidence_root) / (args.mode + "-result.json"), result)
    print(json.dumps({"mode": args.mode, "checkpoint_sha256": digest}), flush=True)


if __name__ == "__main__":
    reconciliation_main()
