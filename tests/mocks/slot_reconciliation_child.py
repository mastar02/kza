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

    No user feedback, used flag, profile confirmation or role/consent mutation
    occurs here.  Those are separate latest-authority sentinels, never payload
    to rewind.  Source identities and deadlines are checked again on return.
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
        correction = next(item for item in corrections if item.id == selected["collectors"]["correction_id"])
        assert correction.available
        correction_data = {"id": correction.id, "original_text": correction.original_text,
            "corrected_text": correction.corrected_text, "used": correction.used,
            "expires_at": correction.expires_at,
            "sources": tuple(source.to_dict() for source in correction.sources)}
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
