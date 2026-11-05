"""Data-only compatible restore behind two real erasure participants."""
from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
import hashlib
from pathlib import Path
import sqlite3
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from src.ambient.audio_archive import AudioArchiver
from src.ambient.erasure import build_capture_erasure
from src.ambient.retention import CaptureRetention
from src.ambient.store import AmbientStore
from src.ambient.types import AmbientUtterance
from src.memory.erasure_coordinator import ErasureCoordinator
from src.memory.erasure_journal import ErasureError, ErasureParticipant
from src.memory.privacy_services import build_privacy_services
from src.memory.producer_context import producer_context
from src.training.collector_erasure import build_collector_erasure
from src.training.conversation_collector import ConversationCollector
from src.wakeword.clip_store import CollectorKind, WakeClipStore
from tests.integration.test_transient_privacy import until
from tests.mocks.collector_erasure import ACTOR, configuration as collector_configuration
from tests.mocks.memory_graph import Clock
from tests.mocks.memory_sources import memory_source


pytestmark = pytest.mark.asyncio
SELECTED = {"captures", "collectors"}
DATA_TABLES = ("wake_clips_v1", "conversation_turns_v1")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def authority_digest(path):
    """All authority tables are excluded from the data restoration allowlist."""
    with sqlite3.connect(path) as db:
        names = [row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            if row[0] not in DATA_TABLES]
        values = []
        for name in names:
            escaped = name.replace('"', '""')
            rows = [list(row) for row in db.execute(f'SELECT * FROM "{escaped}"')]
            rows.sort(key=repr)
            values.append((name, rows))
    return hashlib.sha256(repr(values).encode()).hexdigest()


def data_rows(path, table, identifier):
    assert table in DATA_TABLES
    with sqlite3.connect(path) as db:
        db.row_factory = sqlite3.Row
        return [dict(row) for row in db.execute(f'SELECT * FROM "{table}" WHERE id=?', (identifier,))]


def ambient_rows(path, row_id, conversation_id):
    with sqlite3.connect(path) as db:
        db.row_factory = sqlite3.Row
        return {
            "utterances": [dict(row) for row in db.execute("SELECT * FROM utterances WHERE id=?", (row_id,))],
            "capture_conversations": [dict(row) for row in db.execute(
                "SELECT * FROM capture_conversations WHERE id=?", (conversation_id,))],
        }


@dataclass(frozen=True)
class Descendants:
    source: object
    ambient_row: int
    archive_path: str
    collector_id: str
    collector_path: str


@dataclass(frozen=True)
class DataSnapshot:
    ambient_image: Path
    files: dict
    collector_rows: dict
    root_identities: dict
    foreign_rows: dict
    foreign_ambient_rows: dict


@asynccontextmanager
async def graph(root, clock):
    config = collector_configuration(root)
    config.update(memory={"enabled": False, "privacy_db_path": str(root / "privacy.db")},
        ambient={"enabled": False, "db_path": str(root / "ambient" / "ambient.db"),
            "keep_audio": {"enabled": False, "dir": str(root / "ambient" / "audio")}})
    async with AsyncExitStack() as shutdown:
        privacy = await build_privacy_services(config, shutdown=shutdown, clock=clock)
        owner = ErasureCoordinator(privacy, close_timeout_s=.03)
        captures = await build_capture_erasure(config, privacy=privacy, owner=owner)
        ambient = AmbientStore(config["ambient"]["db_path"],
            audio_dir=config["ambient"]["keep_audio"]["dir"], source_policy=privacy.ledger, clock=clock)
        shutdown.push_async_callback(ambient.close)
        archive = AudioArchiver(str(ambient.audio_dir), enabled=True, min_free_bytes=0,
            privacy=privacy, store=ambient, discard_audio=ambient.discard_audio)
        await captures.attach(SimpleNamespace(store=ambient, transcriber=SimpleNamespace(_archiver=archive)))
        await ambient.init()
        clips = WakeClipStore(privacy, close_timeout_s=.03)
        shutdown.push_async_callback(clips.aclose)
        await clips._work.run(clips.initialize)
        target = await clips._work.run(clips.register, root / "conversations", CollectorKind.CONVERSATIONS)
        conversations = ConversationCollector(target.root, store=clips, target=target, close_timeout_s=.03)
        shutdown.push_async_callback(conversations.aclose)
        collectors = await build_collector_erasure(config, store=clips, owner=owner, shutdown=shutdown)
        # The coordinator drains the real registered drivers before their
        # stores and the shared SQLite authority leave the ExitStack.
        shutdown.push_async_callback(owner.aclose)
        assert captures.privacy is clips.privacy is privacy
        assert ambient.source_policy is privacy.ledger and archive._store is ambient
        state = SimpleNamespace(root=root, clock=clock, config=config, privacy=privacy,
            owner=owner, captures=captures, ambient=ambient, archive=archive,
            clips=clips, collectors=collectors, conversations=conversations, target=target)
        try:
            yield state
        finally:
            await asyncio.shield(owner.begin_close())


async def add_descendants(g, source):
    utterance = AmbientUtterance("synthetic-room", source.captured_at, source.captured_at + 1,
        "synthetic capture text", capture_id=source.capture_id,
        conversation_id=source.conversation_id, expires_at=source.expires_at)
    row = await g.ambient.add(utterance)
    audio_path = await g.archive.write("synthetic-room", row, np.full(1600, .1, np.float32), source=source)
    assert audio_path and await g.ambient.set_audio_path(row, audio_path)
    with producer_context("synthetic-room", (source,)):
        identifier = await g.conversations.add_turn("synthetic input", "synthetic response",
            input_sources=(source,), response_sources=(source,))
    assert identifier is not None
    collector_path = Path(g.target.root) / "policy_v1" / f"{identifier}.json"
    assert collector_path.is_file()
    return Descendants(source, row, audio_path, identifier, str(collector_path))


async def snapshot(g, subject, foreign):
    image = g.root / "fixture-snapshot-ambient.sqlite"

    def save():
        with sqlite3.connect(g.ambient.db_path) as source, sqlite3.connect(image) as destination:
            source.backup(destination)
        files = {path: Path(path).read_bytes() for path in (
            subject.archive_path, subject.collector_path, foreign.archive_path, foreign.collector_path)}
        rows = {table: data_rows(g.privacy.ledger.path, table, subject.collector_id) for table in DATA_TABLES}
        foreign_rows = {table: data_rows(g.privacy.ledger.path, table, foreign.collector_id) for table in DATA_TABLES}
        roots = {str(path): (path.stat().st_dev, path.stat().st_ino) for path in (
            Path(subject.archive_path).parent, Path(subject.collector_path).parent)}
        foreign_ambient = ambient_rows(g.ambient.db_path, foreign.ambient_row, foreign.source.conversation_id)
        assert foreign_ambient["utterances"] and foreign_ambient["capture_conversations"]
        return DataSnapshot(image, files, rows, roots, foreign_rows, foreign_ambient)

    return await asyncio.to_thread(save)


def restore_data(root, saved, subject):
    """Restore data only, with no writer open and no authority table edits."""
    privacy_path = root / "privacy.db"
    before = authority_digest(privacy_path)
    for name, identity in saved.root_identities.items():
        info = Path(name).stat()
        assert (info.st_dev, info.st_ino) == identity
    with sqlite3.connect(saved.ambient_image) as source, sqlite3.connect(root / "ambient" / "ambient.db") as dest:
        source.backup(dest)
    for path in (subject.archive_path, subject.collector_path):
        assert not Path(path).exists()
        Path(path).write_bytes(saved.files[path])
    with sqlite3.connect(privacy_path) as db:
        for table in DATA_TABLES:
            for row in saved.collector_rows[table]:
                columns = list(row)
                quoted = ",".join('"' + column.replace('"', '""') + '"' for column in columns)
                placeholders = ",".join("?" for _ in columns)
                db.execute(f'INSERT INTO "{table}" ({quoted}) VALUES ({placeholders})', tuple(row.values()))
    assert authority_digest(privacy_path) == before
    return before


async def receipt(g, identifier):
    return await g.privacy.acall(g.privacy.ledger.erasures.snapshot, identifier)


async def target_step(g, operation_id, target_id):
    def read():
        with g.privacy.ledger._transaction() as db:
            row = db.execute("SELECT state,attempts,last_error FROM erasure_steps WHERE operation_id=? AND target_id=?",
                (operation_id, target_id)).fetchone()
            return dict(row) if row is not None else None

    return await g.privacy.acall(read)


def selected_steps(value, participant=None):
    return [row for row in value["steps"] if row["participant"] in SELECTED
        and (participant is None or row["participant"] == participant)]


async def settle(g, identifier, participants=SELECTED):
    async with asyncio.timeout(3):
        while True:
            await g.owner.reconcile()
            value = await receipt(g, identifier)
            relevant = [row for row in value["steps"] if row["participant"] in participants]
            if relevant and all(row["state"] == "complete" for row in relevant):
                return value
            g.clock.now += .02
            await asyncio.sleep(.001)


async def foreign_unchanged(g, foreign, saved):
    assert await g.ambient.read_audio(foreign.ambient_row)
    turns = await g.conversations.get_turns(conversation_id=foreign.source.conversation_id)
    assert [turn.id for turn in turns] == [foreign.collector_id]
    for turn in turns:
        assert turn.expires_at == foreign.source.expires_at
    await foreign_storage_unchanged(g, foreign, saved)


async def foreign_storage_unchanged(g, foreign, saved):
    for path in (foreign.archive_path, foreign.collector_path):
        assert Path(path).read_bytes() == saved.files[path]
    for table in DATA_TABLES:
        assert await asyncio.to_thread(data_rows, g.privacy.ledger.path, table, foreign.collector_id) == saved.foreign_rows[table]
    assert await asyncio.to_thread(ambient_rows, g.ambient.db_path,
        foreign.ambient_row, foreign.source.conversation_id) == saved.foreign_ambient_rows


class HeldUnlink:
    def __init__(self, monkeypatch, path):
        self.path = str(path)
        self.entered, self.release, self.returned = (threading.Event() for _ in range(3))
        self.calls = self.successes = 0
        self.workers = set()
        self.lock = threading.Lock()
        unlink = Path.unlink

        def forwarding(candidate, *args, **kwargs):
            if str(candidate) != self.path:
                return unlink(candidate, *args, **kwargs)
            with self.lock:
                self.calls += 1
                self.workers.add(threading.current_thread())
            self.entered.set()
            assert self.release.wait(5), "fixture unlink gate was not released"
            try:
                with self.lock:
                    existed = candidate.exists()
                    result = unlink(candidate, *args, **kwargs)
                    if existed and not candidate.exists():
                        self.successes += 1
                return result
            finally:
                self.returned.set()

        monkeypatch.setattr(Path, "unlink", forwarding)


async def erased_snapshot(root, clock):
    async with graph(root, clock) as g:
        subject = await add_descendants(g, memory_source(now=clock.now))
        foreign = await add_descendants(g, memory_source(now=clock.now))
        saved = await snapshot(g, subject, foreign)
        original = await g.owner.request(subject.source.conversation_id, ACTOR)
        value = await settle(g, original["id"])
        assert selected_steps(value)
        assert not Path(subject.archive_path).exists() and not Path(subject.collector_path).exists()
        assert value["state"] != "complete", "missing drivers must not become fake global completion"
        await foreign_unchanged(g, foreign, saved)
    assert not g.privacy.resource_status()["writer_lease"]
    return subject, foreign, saved, original


async def test_original_multistore_erasure_cleans_one_source_and_preserves_foreign_source(tmp_path):
    subject, foreign, saved, original = await erased_snapshot(tmp_path, Clock())
    assert not Path(subject.archive_path).exists() and not Path(subject.collector_path).exists()
    assert Path(foreign.archive_path).read_bytes() == saved.files[foreign.archive_path]
    assert original["conversation_id"] == subject.source.conversation_id


async def test_restored_capture_receipt_cannot_stay_complete_while_native_unlink_is_held(tmp_path, monkeypatch):
    clock = Clock()
    subject, foreign, saved, original = await erased_snapshot(tmp_path, clock)
    await asyncio.to_thread(restore_data, tmp_path, saved, subject)
    held = HeldUnlink(monkeypatch, subject.archive_path)
    clock.now += 1
    try:
        async with graph(tmp_path, clock) as g:
            try:
                await until(held.entered.is_set)
                assert Path(subject.archive_path).exists() and held.workers
                value = await settle(g, original["id"], {"collectors"})
                captures = selected_steps(value, "captures")
                assert captures and all(row["state"] != "complete" for row in captures), (
                    "restored capture receipt cannot acknowledge physical cleanup while actual unlink is held")
                assert value["state"] != "complete"
                assert not g.privacy.sources_available((subject.source,))
                assert g.privacy.resource_status()["writer_lease"]
            finally:
                # Must run before graph.__aexit__ waits on the real disk worker.
                held.release.set()
            await until(held.returned.is_set)
            await settle(g, original["id"])
            assert not Path(subject.archive_path).exists() and not Path(subject.collector_path).exists()
            await foreign_unchanged(g, foreign, saved)
    finally:
        held.release.set()
    assert held.successes == 1
    assert all(worker._handle.is_done() for worker in held.workers)


async def test_restored_payload_is_not_served_and_never_renews_original_deadline(tmp_path, monkeypatch):
    clock = Clock()
    subject, foreign, saved, original = await erased_snapshot(tmp_path, clock)
    await asyncio.to_thread(restore_data, tmp_path, saved, subject)
    clock.now += 1
    with monkeypatch.context() as patch:
        # Only this read-policy case keeps the autonomous scheduler off, so a
        # denied read cannot be explained by earlier physical deletion.
        patch.setattr(CaptureRetention, "start", lambda self: None)
        async with graph(tmp_path, clock) as g:
            assert await asyncio.to_thread(ambient_rows, g.ambient.db_path,
                subject.ambient_row, subject.source.conversation_id) == await asyncio.to_thread(
                    ambient_rows, saved.ambient_image, subject.ambient_row, subject.source.conversation_id)
            assert Path(subject.archive_path).read_bytes() == saved.files[subject.archive_path]
            assert Path(subject.collector_path).read_bytes() == saved.files[subject.collector_path]
            assert not g.privacy.sources_available((subject.source,))
            assert await g.ambient.read_audio(subject.ambient_row) is None
            assert await g.conversations.get_turns(conversation_id=subject.source.conversation_id) == []
            assert await asyncio.to_thread(ambient_rows, g.ambient.db_path,
                subject.ambient_row, subject.source.conversation_id) == await asyncio.to_thread(
                    ambient_rows, saved.ambient_image, subject.ambient_row, subject.source.conversation_id)
            assert Path(subject.archive_path).read_bytes() == saved.files[subject.archive_path]
            assert Path(subject.collector_path).read_bytes() == saved.files[subject.collector_path]
            value = await g.owner.request(subject.source.conversation_id, ACTOR)
            assert value["id"] == original["id"]
            assert value["requested_at"] == original["requested_at"]
            assert value["requested_by"] == original["requested_by"]
            await foreign_unchanged(g, foreign, saved)


async def test_cancelled_observer_cannot_release_one_held_participant_or_duplicate_cleanup(tmp_path, monkeypatch):
    clock = Clock()
    async with graph(tmp_path, clock) as g:
        subject = await add_descendants(g, memory_source(now=clock.now))
        foreign = await add_descendants(g, memory_source(now=clock.now))
        saved = await snapshot(g, subject, foreign)
        await foreign_unchanged(g, foreign, saved)
        held = HeldUnlink(monkeypatch, subject.archive_path)
        observer = None
        try:
            original = await g.owner.request(subject.source.conversation_id, ACTOR)
            value = await settle(g, original["id"], {"collectors"})
            await until(held.entered.is_set)
            assert selected_steps(value, "collectors")
            assert all(row["state"] == "complete" for row in selected_steps(value, "collectors"))
            value = await receipt(g, original["id"])
            assert all(row["state"] != "complete" for row in selected_steps(value, "captures"))
            observer = asyncio.create_task(g.owner.aclose())
            await asyncio.sleep(0)
            closing = g.owner.begin_close()
            observer.cancel()
            await asyncio.gather(observer, return_exceptions=True)
            assert not closing.done() and not held.returned.is_set()
            assert Path(subject.archive_path).exists()
            assert g.privacy.resource_status()["writer_lease"]
            pending = await receipt(g, original["id"])
            assert pending["id"] == original["id"] and pending["state"] != "complete"
        finally:
            held.release.set()
            if observer is not None:
                await asyncio.gather(observer, return_exceptions=True)
            await asyncio.shield(g.owner.begin_close())
        assert held.returned.is_set() and held.successes == 1
        assert all(worker._handle.is_done() for worker in held.workers)
        value = await receipt(g, original["id"])
        assert all(row["state"] == "complete" for row in selected_steps(value))
        assert value["state"] != "complete"
        assert not Path(subject.archive_path).exists() and not Path(subject.collector_path).exists()
        await foreign_storage_unchanged(g, foreign, saved)


@pytest.mark.parametrize("incompatibility", ["opaque_legacy_root", "unsupported_target_version"])
async def test_incompatible_restore_stays_pending_without_deleting_foreign_payload(tmp_path, incompatibility):
    clock = Clock()
    subject, foreign, saved, original = await erased_snapshot(tmp_path, clock)
    await asyncio.to_thread(restore_data, tmp_path, saved, subject)
    directory = tmp_path / "models" if incompatibility == "opaque_legacy_root" else tmp_path / "unsupported"
    directory.mkdir(parents=True, exist_ok=True)
    payload = directory / "unattributed.bin"
    payload.write_bytes(b"synthetic unrelated opaque payload")
    fingerprint = digest(payload)
    clock.now += 1
    async with graph(tmp_path, clock) as g:
        if incompatibility == "unsupported_target_version":
            target_id = await g.privacy.acall(g.privacy.ledger.erasures.register_target,
                ErasureParticipant.COLLECTORS, 2,
                {"kind": g.target.kind.value, "root": str(directory.resolve())})
            expected_error = ErasureError.UNAVAILABLE.value
        else:
            targets = await g.privacy.acall(g.privacy.ledger.erasures.targets, ErasureParticipant.COLLECTORS)
            matching = [row for row in targets if row["version"] == 1
                and row["descriptor"] == {"kind": "legacy_training", "root": str(directory.resolve())}]
            assert len(matching) == 1
            target_id = matching[0]["id"]
            expected_error = ErasureError.PENDING.value
        # Require an actual coordinator/driver result, not merely an untouched
        # initial pending row. Neither unknown case grants erase consent.
        async with asyncio.timeout(3):
            while True:
                await g.owner.reconcile()
                value = await receipt(g, original["id"])
                collector_steps = selected_steps(value, "collectors")
                current = await target_step(g, original["id"], target_id)
                if current is not None and current["state"] == "pending" and current["last_error"] == expected_error:
                    break
                clock.now += .02
                await asyncio.sleep(.001)
        assert value["state"] != "complete"
        assert payload.exists() and digest(payload) == fingerprint
        assert any(row["state"] != "complete" for row in collector_steps)
        await until(lambda: not Path(subject.archive_path).exists())
        assert not g.privacy.sources_available((subject.source,))
        await foreign_unchanged(g, foreign, saved)
