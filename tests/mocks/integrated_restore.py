"""One real erasure graph; data restores never rewind its authority.

These helpers deliberately preserve the unresolved inventories.  A local
receipt is not a claim about vendor allocations or a physical browser.
"""
from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
from types import SimpleNamespace
from uuid import uuid4

import numpy as np

from src.ambient.audio_archive import AudioArchiver
from src.ambient.erasure import build_capture_erasure
from src.ambient.store import AmbientStore
from src.ambient.types import AmbientUtterance
from src.analytics.erasure import build_event_erasure
from src.analytics.services import build_event_logger
from src.analytics.suggestion_engine import SuggestionEngine
from src.analytics.suggestion_erasure import build_suggestion_erasure
from src.dashboard.live_erasure import build_live_services
from src.dashboard.live_event_bus import LiveEvent, LiveEventType
from src.memory.fact_ledger import MemoryReviewer
from src.memory.index_erasure import build_erasure_services
from src.memory.privacy_services import build_privacy_services
from src.memory.producer_context import producer_context
from src.memory.services import build_memory_services
from src.obs.diagnostic_erasure import build_diagnostic_services
from src.orchestrator.context_erasure import build_context_erasure
from src.orchestrator.context_manager import ContextManager
from src.orchestrator.context_persister import ContextPersister
from src.pipeline.command_processor import CommandProcessor
from src.stt.whisper_fast import STTResult
from src.training.conversation_collector import ConversationCollector
from src.training.stt_correction_collector import STTCorrectionCollector
from src.users.voice_ledger import VoiceReviewer
from src.users.voice_registry import build_voice_registry
from src.wakeword.clip_store import CollectorKind, build_wake_clip_store
from src.wakeword.wake_clip_writer import WakeClipWriter
from tests.mocks.browser_authority import bind_browser_authority, make_browser_authority
from tests.mocks.collector_erasure import configuration as collector_configuration
from tests.mocks.memory_graph import Clock, MemoryClient
from tests.mocks.memory_sources import memory_source
from tests.mocks.voice_registry import Encoder


ACTOR = MemoryReviewer("synthetic-integrated-admin")
LOCAL = frozenset({"captures", "contexts", "semantic_index", "events",
    "diagnostics", "collectors", "streams", "suggestions", "voices"})
BLOCKERS = {("producers", 0), ("diagnostics", 1), ("streams", 2), ("voices", 2)}
SHARED_TABLES = (
    "wake_clips_v1", "stt_corrections_v1", "conversation_turns_v1",
    "conversation_export_parents_v1", "voice_fingerprints",
    "voice_fingerprint_sources", "voice_capture_reviews", "live_deliveries",
    "live_delivery_sources", "live_delivery_lineage", "voice_groups",
    "voice_group_members", "voice_group_lineage",
)
VOICE_COLUMNS = ("id", "model_key", "dimensions", "embedding", "duration_s",
    "created_at", "expires_at", "sources_json")


async def until(predicate, *, timeout=4):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(.001)


class NativeGate:
    """A gate in a delegated native leaf, with a real return receipt."""
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.returned = threading.Event()
        self.threads = set()

    def call(self, function, *args, **kwargs):
        self.threads.add(threading.current_thread())
        self.entered.set()
        try:
            assert self.release.wait(8), "fixture native gate was not released"
            return function(*args, **kwargs)
        finally:
            self.returned.set()


class CPUSTT:
    def __init__(self, gate=None):
        self.gate = gate

    def transcribe_with_confidence(self, audio, sample_rate=16000):
        def result():
            return STTResult("Synthetic command", 1., no_speech_prob=.01, avg_logprob=-.1)
        return self.gate.call(result) if self.gate is not None else result()


class DistinctEncoder(Encoder):
    def get_embedding(self, audio):
        self.calls += 1
        result = np.zeros(192, dtype=np.float32)
        result[0 if float(audio[0]) < .15 else 1] = 1
        return result


def configuration(root):
    config = collector_configuration(root)
    config.update(
        memory={"enabled": True, "privacy_db_path": str(root / "privacy.db"),
            "chroma_path": str(root / "chroma"), "preferences_path": str(root / "preferences.json")},
        ambient={"enabled": False, "db_path": str(root / "ambient.db"),
            "keep_audio": {"enabled": False, "dir": str(root / "audio")}},
        orchestrator={"enabled": False, "context": {"persistence": {
            "enabled": False, "base_path": str(root / "contexts")}}},
        analytics={"enabled": True, "events_db": str(root / "events.db"),
            "suggestions_path": str(root / "suggestions.json")},
        observability={"enabled": True, "metrics_path": str(root / "metrics.jsonl")},
    )
    return config


@asynccontextmanager
async def retirement_stack(gates):
    stack = AsyncExitStack()
    try:
        yield stack
    finally:
        # This runs even if a builder fails before integrated_graph yields.
        for gate in gates:
            gate.release.set()
        await stack.aclose()


@asynccontextmanager
async def integrated_graph(root, clock=None, *, stt_gate=None, gates=(), before_voices=None):
    """Public builders share one lease, ledger, journal and coordinator."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    clock = clock or Clock()
    config = configuration(root)
    async with AsyncExitStack() as authority_stack:
        privacy = await build_privacy_services(config, shutdown=authority_stack, clock=clock)
        async with AsyncExitStack() as resources:
            client = MemoryClient(root / "chroma")
            resources.callback(client.close)
            memory = await build_memory_services(config, shutdown=resources,
                privacy=privacy, clock=clock, chroma_client=client)
            async with retirement_stack((*gates, *((stt_gate,) if stt_gate else ()))) as coordinator_stack:
                owner = await build_erasure_services(config, privacy=privacy,
                    shutdown=coordinator_stack, memory=memory.manager.long_term)
                capture = await build_capture_erasure(config, privacy=privacy, owner=owner)
                ambient = AmbientStore(str(root / "ambient.db"), audio_dir=str(root / "audio"),
                    source_policy=privacy.ledger, clock=clock)
                resources.push_async_callback(ambient.close)
                archive = AudioArchiver(str(root / "audio"), enabled=True, min_free_bytes=0,
                    privacy=privacy, store=ambient, discard_audio=ambient.discard_audio)
                await capture.attach(SimpleNamespace(store=ambient,
                    transcriber=SimpleNamespace(_archiver=archive)))
                await ambient.init()
                clips = await build_wake_clip_store(config, privacy=privacy,
                    shutdown=resources, owner=owner)
                targets = {key: await clips._work.run(clips.register, root / key, kind)
                    for key, kind in (("wake", CollectorKind.WAKE_CLIPS),
                        ("stt", CollectorKind.STT_CORRECTIONS),
                        ("conversations", CollectorKind.CONVERSATIONS),
                        ("exports", CollectorKind.CONVERSATION_EXPORTS))}
                wake = WakeClipWriter(targets["wake"].root, store=clips, target=targets["wake"])
                stt = STTCorrectionCollector(targets["stt"].root, store=clips, target=targets["stt"])
                conversations = ConversationCollector(targets["conversations"].root,
                    store=clips, target=targets["conversations"])
                for producer in (wake, stt, conversations):
                    resources.push_async_callback(producer.aclose)
                context = await build_context_erasure(config, privacy=privacy,
                    owner=owner, memory=memory.manager)
                persister = ContextPersister(str(root / "contexts"), source_policy=privacy.ledger, clock=clock)
                manager = ContextManager(persister=persister, source_policy=privacy.ledger,
                    clock=clock, compaction_threshold=100)
                context.attach_manager(manager)
                resources.push_async_callback(manager.aclose)
                events_driver = await build_event_erasure(config, privacy=privacy, owner=owner)
                events = await build_event_logger(config, shutdown=resources, privacy=privacy,
                    retention=events_driver.retention)
                suggestions_driver = await build_suggestion_erasure(config, privacy=privacy, owner=owner)
                suggestions = await events.acall(SuggestionEngine, events,
                    suggestions_path=config["analytics"]["suggestions_path"])
                suggestions_driver.attach_engine(suggestions)
                diagnostics = await build_diagnostic_services(config, privacy=privacy, owner=owner)
                browser_authority = await make_browser_authority(resources, root / "browser-auth.db")
                bus = await build_live_services(config, privacy=privacy, owner=owner, shutdown=resources)
                bind_browser_authority(bus, browser_authority)
                if before_voices is not None:
                    await before_voices(privacy)
                voices = await build_voice_registry(privacy=privacy, owner=owner, shutdown=resources)
                encoder = DistinctEncoder()
                voices.register_encoder(encoder)
                processor = CommandProcessor(CPUSTT(stt_gate), privacy=privacy, close_timeout_s=.03)
                resources.push_async_callback(processor.aclose)
                assert {part.value for part in owner._drivers} == LOCAL
                assert clips.privacy is capture.privacy is privacy
                yield SimpleNamespace(root=root, clock=clock, config=config, privacy=privacy,
                    owner=owner, memory=memory, client=client, capture=capture, ambient=ambient,
                    archive=archive, clips=clips, targets=targets, wake=wake, stt=stt,
                    conversations=conversations, manager=manager, persister=persister,
                    events=events, suggestions=suggestions, diagnostics=diagnostics,
                    bus=bus, voices=voices, encoder=encoder, processor=processor)


class BrowserPeer:
    """Actual receipt protocol over bounded synthetic websocket leaves."""
    def __init__(self, bus, client=None):
        self.bus, self.client = bus, client or str(uuid4())
        self.incoming, self.outgoing = asyncio.Queue(), asyncio.Queue()
        self.task = None

    async def message(self, document):
        await self.incoming.put({"type": "websocket.receive", "text": json.dumps(document)})

    async def start(self):
        sub, _ = await self.bus.subscribe()
        self.task = asyncio.create_task(self.bus.browser_stream(sub, self.outgoing.put,
            receiver=self.incoming.get, reader=self.bus.test_reader))
        await self.message({"type": "live_hello", "version": 1, "client_id": self.client})
        reset = await self.frame("live_reset")
        await self.message({"type": "live_reset_ack", **reset["payload"]})
        await until(lambda: self.bus._retention._states[sub].browser.ready)
        return self

    async def frame(self, kind):
        async with asyncio.timeout(4):
            while True:
                frame = await self.outgoing.get()
                if frame["type"] == kind:
                    return frame

    async def close(self):
        if self.task is not None:
            await self.incoming.put({"type": "websocket.disconnect", "code": 1000})
            async with asyncio.timeout(4):
                await self.task
            self.task = None


@dataclass
class Cohort:
    source: object
    fact: str = ""
    ambient_id: int = 0
    archive_path: str = ""
    clip_ids: list = field(default_factory=list)
    event_ids: list = field(default_factory=list)
    suggestion_ids: list = field(default_factory=list)
    voice_ids: list = field(default_factory=list)
    user: str = ""
    browser_client: str = ""


def rows(path, table):
    with sqlite3.connect(path) as db:
        db.row_factory = sqlite3.Row
        return [dict(row) for row in db.execute(f'SELECT * FROM "{table}"')]


async def seed_patterns(g, cohort, prefix):
    start = g.clock.now
    for ordinal in range(5):
        g.clock.now = start + ordinal * 60
        event = await g.events.acall(g.events.log,
            f"light.{prefix}_trigger", "turn_on", sources=(cohort.source,))
        cohort.event_ids.append(event.record_id)
        g.clock.now += 1
        event = await g.events.acall(g.events.log,
            f"light.{prefix}_target", "turn_on", sources=(cohort.source,))
        cohort.event_ids.append(event.record_id)


async def seed_cohort(g, cohort, value):
    source = cohort.source
    audio = np.full(16000, value, np.float32)
    cohort.fact = await g.memory.manager.long_term.astore_fact(
        f"Synthetic fact {source.conversation_id}", "fact", sources=(source,))
    utterance = AmbientUtterance("synthetic-room", source.captured_at, source.captured_at + 1,
        "Synthetic capture text", capture_id=source.capture_id,
        conversation_id=source.conversation_id, expires_at=source.expires_at)
    cohort.ambient_id = await g.ambient.add(utterance)
    cohort.archive_path = await g.archive.write("synthetic-room", cohort.ambient_id,
        audio, source=source)
    assert cohort.archive_path and await g.ambient.set_audio_path(cohort.ambient_id, cohort.archive_path)
    assert await g.ambient.read_audio(cohort.ambient_id) is not None
    existing = rows(g.root / "privacy.db", "wake_clips_v1")
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
    exported = await g.conversations.export_for_training(g.targets["exports"].root,
        after_sequence=after_sequence)
    assert exported is not None
    # Feedback appends a new immutable turn and expires its old revision.
    # Snapshot the live descendants; retired revisions retain their real
    # journal/retention obligations rather than becoming foreign live data.
    live = [row for row in rows(g.root / "privacy.db", "wake_clips_v1")
        if row["id"] not in before and row["published"] == 1 and row["expires_at"] > g.clock.now]
    assert len(live) == 4
    assert {row["bucket"] for row in live} == {
        "accepted", "stt_correction", "conversation_turn", "conversation_export"}
    assert all(row["captured_at"] == source.captured_at and row["expires_at"] <= source.expires_at
        for row in live)
    cohort.clip_ids = sorted(row["id"] for row in live)
    assert all(source.conversation_id in row["sources_json"] for row in
        rows(g.root / "privacy.db", "wake_clips_v1") if row["id"] in cohort.clip_ids)
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
    cohort.voice_ids = [row["id"] for row in rows(g.root / "privacy.db", "voice_fingerprints")
        if source.conversation_id in row["sources_json"]]
    assert cohort.voice_ids
    for identity in cohort.voice_ids:
        review = await g.voices.acall(g.voices.ledger.review_capture, identity,
            VoiceReviewer("synthetic-reviewer"))
        assert review["transcript"] == "Synthetic reviewed clip" and review["audio_base64"]
        # No STT/audio provenance document was invented by this leaf fixture.
        assert review["audio_metadata"] is review["transcript_metadata"] is None
    peer = await BrowserPeer(g.bus).start()
    try:
        cohort.browser_client = peer.client
        await g.bus.publish(LiveEvent(LiveEventType.TURN,
            payload={"stt": "Synthetic private turn"}, sources=(source,)))
        frame = await peer.frame("turn")
        assert frame["receipt"]
    finally:
        await peer.close()


async def seed_all(g):
    subject = Cohort(memory_source(now=g.clock.now))
    await seed_patterns(g, subject, "erased")
    g.clock.now += 11 * 60
    foreign = Cohort(memory_source(now=g.clock.now))
    await seed_patterns(g, foreign, "survivor")
    proposals = await g.events.acall(g.suggestions.generate_suggestions, .7, 50)
    for cohort in (subject, foreign):
        cohort.suggestion_ids = [proposal.id for proposal in proposals
            if any(source.conversation_id == cohort.source.conversation_id for source in proposal.sources)]
        assert cohort.suggestion_ids, "actual pattern evidence did not produce a proposal"
    await seed_cohort(g, subject, .1)
    await seed_cohort(g, foreign, .2)
    return subject, foreign


async def steps(g, operation):
    def read():
        with g.privacy.ledger._transaction() as db:
            return [dict(row) for row in db.execute("""SELECT s.*,t.descriptor
                FROM erasure_steps s LEFT JOIN erasure_targets t ON t.id=s.target_id
                WHERE s.operation_id=? ORDER BY t.participant,t.id""", (operation["id"],))]
    return await g.privacy.acall(read)


def supported(step):
    return (step["participant"], step["version"]) not in BLOCKERS


async def clear_browser(g, cohort):
    peer = await BrowserPeer(g.bus, cohort.browser_client).start()
    await peer.close()


async def settle(g, operation, *, timeout=6):
    async with asyncio.timeout(timeout):
        while True:
            await g.owner.reconcile()
            if all(step["state"] == "complete" for step in await steps(g, operation) if supported(step)):
                return
            g.clock.now += .1
            await asyncio.sleep(.005)


async def assert_honest_blockers(g, operation):
    selected = await steps(g, operation)
    assert {step["participant"] for step in selected} == LOCAL | {"producers"}
    assert {(step["participant"], step["version"]) for step in selected
        if step["state"] != "complete"} == BLOCKERS
    state = await g.privacy.acall(g.privacy.ledger.erasures.snapshot, operation["id"])
    assert state["state"] != "complete" and state["revoked"]


def copied_tables(path, names):
    """Coherent SQLite image for selecting data, never restoring authority."""
    with sqlite3.connect(path) as source, sqlite3.connect(":memory:") as copy:
        source.backup(copy)
        copy.row_factory = sqlite3.Row
        return {name: [dict(row) for row in copy.execute(f'SELECT * FROM "{name}"')]
            for name in names}


def selected_shared(g, cohort):
    privacy = copied_tables(g.root / "privacy.db", SHARED_TABLES)
    clips, voices = set(cohort.clip_ids), set(cohort.voice_ids)
    members = [row for row in privacy["voice_group_members"] if row["fingerprint_id"] in voices]
    groups = {row["group_id"] for row in members}
    assert all(row["fingerprint_id"] in voices for row in privacy["voice_group_members"]
        if row["group_id"] in groups)
    assert not rows(g.root / "privacy.db", "voice_group_requests")
    fingerprints = [row for row in privacy["voice_fingerprints"] if row["id"] in voices]
    assert all(row["suggested_profile"] is row["suggested_revision"] is row["similarity"] is None
        for row in fingerprints)
    tickets = {row["ticket"] for row in privacy["live_delivery_sources"]
        if row["conversation_id"] == cohort.source.conversation_id}
    selected = {
        "wake_clips_v1": [row for row in privacy["wake_clips_v1"] if row["id"] in clips],
        "stt_corrections_v1": [row for row in privacy["stt_corrections_v1"] if row["id"] in clips],
        "conversation_turns_v1": [row for row in privacy["conversation_turns_v1"] if row["id"] in clips],
        "conversation_export_parents_v1": [row for row in privacy["conversation_export_parents_v1"] if row["export_id"] in clips],
        "voice_fingerprints": fingerprints,
        "voice_fingerprint_sources": [row for row in privacy["voice_fingerprint_sources"] if row["fingerprint_id"] in voices],
        "voice_capture_reviews": [row for row in privacy["voice_capture_reviews"] if row["fingerprint_id"] in voices],
        "voice_groups": [row for row in privacy["voice_groups"] if row["id"] in groups],
        "voice_group_members": members,
        "voice_group_lineage": [row for row in privacy["voice_group_lineage"] if row["group_id"] in groups],
        "live_deliveries": [row for row in privacy["live_deliveries"] if row["ticket"] in tickets],
        "live_delivery_sources": [row for row in privacy["live_delivery_sources"] if row["ticket"] in tickets],
        "live_delivery_lineage": [row for row in privacy["live_delivery_lineage"] if row["ticket"] in tickets],
    }
    assert len(selected["voice_group_lineage"]) == len(groups)
    assert len(selected["live_delivery_lineage"]) == len(tickets)
    return selected


def selected_payload(g, cohort):
    conversation = cohort.source.conversation_id
    shared = selected_shared(g, cohort)
    ambient = copied_tables(g.root / "ambient.db", ("utterances", "capture_conversations"))
    ambient = {name: [row for row in records if
        row.get("conversation_id") == conversation or row.get("id") == conversation]
        for name, records in ambient.items()}
    events = copied_tables(g.events.db_path, ("events_policy_v1",))
    events["events_policy_v1"] = [row for row in events["events_policy_v1"] if conversation in row["sources_json"]]
    suggestions = copied_tables(g.suggestions.store.path, ("suggestions_policy_v1",))
    suggestions["suggestions_policy_v1"] = [row for row in suggestions["suggestions_policy_v1"] if row["id"] in cohort.suggestion_ids]
    assert all(row["status"] == "pending" and row["snooze_until"] is None
        for row in suggestions["suggestions_policy_v1"])
    catalog = g.diagnostics.emitter._store.catalog
    diagnostic = copied_tables(catalog, ("metrics", "lineage"))
    lineage = [row for row in diagnostic["lineage"] if row["conversation_id"] == conversation]
    metrics = {row["record_id"] for row in lineage}
    diagnostic = {"metrics": [row for row in diagnostic["metrics"] if row["id"] in metrics], "lineage": lineage}
    assert metrics and len(diagnostic["metrics"]) == len(metrics)
    disk_ids = set()
    metric_path = g.diagnostics.emitter.path
    for path in (metric_path, Path(str(metric_path) + ".1")):
        if path.is_file():
            disk_ids.update(json.loads(line)["record_id"] for line in path.read_text().splitlines())
    assert metrics <= disk_ids, "flush must actually persist the sourced metrics"
    context_path = g.persister.base_path / f"{cohort.user}.json"
    files = {str(cohort.archive_path): Path(cohort.archive_path).read_bytes(),
        str(context_path): context_path.read_bytes()}
    targets = {target.id: target for target in g.targets.values()}
    for row in shared["wake_clips_v1"]:
        target = targets[row["target_id"]]
        path = Path(target.root) / "policy_v1" / (row["id"] + target.extension)
        files[str(path)] = path.read_bytes()
    result = g.memory.manager.long_term._collection.get(ids=[cohort.fact],
        include=["embeddings", "documents", "metadatas"])
    assert result["ids"] == [cohort.fact]
    index = {key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in result.items() if key in {"ids", "embeddings", "documents", "metadatas"}}
    return {"tables": {str(g.root / "privacy.db"): shared, str(g.root / "ambient.db"): ambient,
        str(g.events.db_path): events, str(g.suggestions.store.path): suggestions,
        str(catalog): diagnostic}, "files": files, "index": index}


async def payload(g, cohort):
    return await asyncio.to_thread(selected_payload, g, cohort)


async def assert_foreign_unchanged(g, cohort, saved, *, rehydrated=False):
    assert g.privacy.sources_available((cohort.source,))
    assert await g.ambient.read_audio(cohort.ambient_id) is not None
    turns = await g.conversations.get_turns(conversation_id=cohort.source.conversation_id)
    assert len(turns) == 1
    context = await g.manager.aget_or_create(cohort.user)
    assert context.compacted_summary == "Synthetic summary"
    assert context.summary_sources == (cohort.source,)
    # ContextPersister deliberately persists summaries rather than turn history.
    assert bool(context.conversation_history) is not rehydrated
    for identity in cohort.voice_ids:
        review = await g.voices.acall(g.voices.ledger.review_capture, identity,
            VoiceReviewer("synthetic-reviewer"))
        assert review["audio_base64"] and review["transcript"] == "Synthetic reviewed clip"
    assert normalize(await payload(g, cohort)) == normalize(saved)


def normalize(value):
    if isinstance(value, dict):
        return tuple((key, normalize(item)) for key, item in sorted(value.items()))
    if isinstance(value, list):
        return tuple(normalize(item) for item in value)
    return value


def authority_digest(path):
    with sqlite3.connect(path) as db:
        names = [name for name, in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
            if name not in SHARED_TABLES]
        records = {name: sorted(list(db.execute(f'SELECT * FROM "{name}"')), key=repr)
            for name in sorted(names)}
    return hashlib.sha256(repr(normalize(records)).encode()).hexdigest()


@dataclass
class DataSnapshot:
    data: dict
    root_identities: dict
    diagnostic_files: dict
    index_path: str
    index_collection: str
    sdk_version: str


async def snapshot_payload(g, cohort):
    import chromadb
    data = await payload(g, cohort)
    paths = {Path(target.root) for target in g.targets.values()}
    paths.update((Path(g.ambient.audio_dir), g.persister.base_path, g.root / "chroma"))
    roots = {str(path): (path.stat().st_dev, path.stat().st_ino) for path in paths}
    path = g.diagnostics.emitter.path
    diagnostic_files = {str(candidate): candidate.read_bytes()
        for candidate in (path, Path(str(path) + ".1")) if candidate.is_file()}
    return DataSnapshot(data, roots, diagnostic_files, str(g.root / "chroma"),
        g.memory.manager.long_term.index_collection_name, chromadb.__version__)


def insert_rows(db, table, records):
    for row in records:
        columns = list(row)
        clause = " AND ".join(f'"{column}" IS ?' for column in columns)
        if db.execute(f'SELECT 1 FROM "{table}" WHERE {clause}', tuple(row.values())).fetchone():
            continue
        names = ",".join(f'"{column}"' for column in columns)
        placeholders = ",".join("?" for _ in columns)
        db.execute(f'INSERT INTO "{table}" ({names}) VALUES ({placeholders})', tuple(row.values()))


def restore_payload(root, snapshot):
    """Data only, with no writer open and no authority rows rolled back."""
    import chromadb
    assert chromadb.__version__ == snapshot.sdk_version
    authority = Path(root) / "privacy.db"
    before = authority_digest(authority)
    for path, identity in snapshot.root_identities.items():
        info = Path(path).stat()
        assert (info.st_dev, info.st_ino) == identity
    for path, tables in snapshot.data["tables"].items():
        with sqlite3.connect(path) as db:
            db.execute("PRAGMA foreign_keys=ON")
            for table, records in tables.items():
                insert_rows(db, table, records)
            assert not db.execute("PRAGMA foreign_key_check").fetchall()
    for path, raw in snapshot.data["files"].items():
        target = Path(path)
        assert not target.exists(), f"erased data remained at {target.name}"
        target.write_bytes(raw)
    for path, raw in snapshot.diagnostic_files.items():
        Path(path).write_bytes(raw)
    client = MemoryClient(snapshot.index_path)
    try:
        values = {key: value for key, value in snapshot.data["index"].items()
            if value is not None and not (key == "metadatas" and all(item is None for item in value))}
        client.get_or_create_collection(snapshot.index_collection).upsert(**values)
    finally:
        client.close()
    assert authority_digest(authority) == before
    return before


def stored_steps(root, operation):
    with sqlite3.connect(Path(root) / "privacy.db") as db:
        db.row_factory = sqlite3.Row
        return [dict(row) for row in db.execute("SELECT * FROM erasure_steps WHERE operation_id=?",
            (operation["id"],))]


def install_restore_gates(monkeypatch, subject, delete_gate, unlink_gate):
    from chromadb.api.models.Collection import Collection
    delete, unlink = Collection.delete, Path.unlink

    def deleting(collection, *args, **kwargs):
        if subject.fact in (kwargs.get("ids") or ()):
            return delete_gate.call(delete, collection, *args, **kwargs)
        return delete(collection, *args, **kwargs)

    def unlinking(path, *args, **kwargs):
        if str(path) == str(subject.archive_path):
            return unlink_gate.call(unlink, path, *args, **kwargs)
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Collection, "delete", deleting)
    monkeypatch.setattr(Path, "unlink", unlinking)


async def assert_subject_erased(g, cohort):
    source = cohort.source.conversation_id
    assert await g.ambient.read_audio(cohort.ambient_id) is None
    assert not Path(cohort.archive_path).exists()
    assert not (g.persister.base_path / f"{cohort.user}.json").exists()
    assert not set(cohort.clip_ids) & {row["id"] for row in rows(g.root / "privacy.db", "wake_clips_v1")}
    assert not set(cohort.voice_ids) & {row["id"] for row in rows(g.root / "privacy.db", "voice_fingerprints")}
    assert not [row for row in rows(g.events.db_path, "events_policy_v1") if source in row["sources_json"]]
    assert not set(cohort.suggestion_ids) & {row["id"] for row in rows(g.suggestions.store.path, "suggestions_policy_v1")}
    assert not [row for row in rows(g.diagnostics.emitter._store.catalog, "lineage") if row["conversation_id"] == source]
    assert g.memory.manager.long_term._collection.get(ids=[cohort.fact])["ids"] == []
