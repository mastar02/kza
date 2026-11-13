"""A production-shaped erased conversation for the U3b-0d4a resurrection checks.

The state is written by the real code (tests/mocks/integrated_restore): one
ledger and erasure journal in data/privacy.db with every target registered,
including the three cleaner-less literals (diagnostics v1, streams v2,
voices v2), and two conversations seeded into every store. The subject is
erased through the coordinator and settled, so each store's own step is
complete and only the honest blockers stay pending.

Restores are data-only: whole older files for events.db and ambient.db (as
an operator would restore them), rows for the clip store (it shares
privacy.db with the journal, so restoring the file would also rewind the
ledger), the saved snapshot file for contexts and an upsert for Chroma.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import sqlite3

from src.memory.privacy_services import PrivacyServices
from tests.mocks.integrated_restore import (
    ACTOR, assert_honest_blockers, assert_subject_erased, clear_browser, insert_rows,
    integrated_graph, seed_all, settle, snapshot_payload,
)
from tests.mocks.memory_graph import Clock, MemoryClient

DATA = "data"
TOKEN = "synthetic-resurrection-reader-token"


@dataclass
class ErasedState:
    """Where the erased fixture lives and what was saved before the erasure."""

    state: Path
    before: Path
    subject: str
    foreign: str
    operation: str
    saved: object
    subject_cohort: object
    clock: Clock


@contextmanager
def _graph_token():
    """The live bus needs a reader token; the process environment is restored afterwards."""
    previous = os.environ.get("KZA_DASHBOARD_TOKEN")
    os.environ["KZA_DASHBOARD_TOKEN"] = TOKEN
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("KZA_DASHBOARD_TOKEN", None)
        else:
            os.environ["KZA_DASHBOARD_TOKEN"] = previous


async def build_erased_state(state: Path, before: Path) -> ErasedState:
    """Seed two conversations under ``state``/data, copy the closed state to ``before``, erase and settle."""
    state, before = Path(state), Path(before)
    data = state / DATA
    clock = Clock()
    with _graph_token():
        async with integrated_graph(data, clock) as g:
            subject, foreign = await seed_all(g)
            saved = await snapshot_payload(g, subject)
        shutil.copytree(state, before, symlinks=True)  # every writer closed: a coherent older image
        async with integrated_graph(data, clock) as g:
            operation = await g.owner.request(subject.source.conversation_id, ACTOR)
            await clear_browser(g, subject)
            await settle(g, operation)
            await assert_honest_blockers(g, operation)
            await assert_subject_erased(g, subject)
    return ErasedState(state, before, subject.source.conversation_id, foreign.source.conversation_id,
                       operation["id"], saved, subject, clock)


def reset(erased: ErasedState, pristine: Path) -> ErasedState:
    """Put the erased state back at its own path (descriptors name absolute paths under it)."""
    shutil.rmtree(erased.state)
    shutil.copytree(pristine, erased.state, symlinks=True)
    return erased


def restore_file(erased: ErasedState, relative: str) -> None:
    """Replace one store with its pre-erasure bytes (with its -wal and -journal)."""
    for suffix in ("", "-wal", "-journal", "-shm"):
        target = erased.state / (relative + suffix)
        target.unlink(missing_ok=True)
        source = erased.before / (relative + suffix)
        if source.exists() and suffix != "-shm":
            shutil.copy2(source, target)


def restore_clip_rows(erased: ErasedState, tables=("wake_clips_v1", "conversation_turns_v1")) -> None:
    """Re-insert the subject's clip-store rows into privacy.db (data only, no authority row)."""
    privacy = str(erased.state / DATA / "privacy.db")
    records = erased.saved.data["tables"][privacy]
    with sqlite3.connect(erased.state / DATA / "privacy.db") as db:
        db.execute("PRAGMA foreign_keys=ON")
        for table in tables:
            insert_rows(db, table, records[table])
        assert not db.execute("PRAGMA foreign_key_check").fetchall()


def restore_context_files(erased: ErasedState) -> list[Path]:
    """Write the subject's saved context snapshot back."""
    written = []
    for path, raw in erased.saved.data["files"].items():
        if "contexts" in Path(path).parts:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            written.append(target)
    assert written
    return written


def restore_collector_files(erased: ErasedState, suffixes=(".json", ".jsonl")) -> list[Path]:
    """Write the subject's saved collector files (under a collectors root's policy_v1) back."""
    written = []
    for path, raw in erased.saved.data["files"].items():
        target = Path(path)
        if target.parent.name == "policy_v1" and "audio" not in target.parts and target.suffix in suffixes:
            target.write_bytes(raw)
            written.append(target)
    assert written
    return written


def restore_index_with_conversation_metadata(erased: ErasedState) -> None:
    """Upsert the subject's saved embedding, carrying conversation_id in its metadata.

    No current writer puts conversation_id into Chroma metadata; this is the
    defensive map entry's shape, written by the real Chroma SDK.
    """
    index = dict(erased.saved.data["index"])
    index["metadatas"] = [dict(item or {}, conversation_id=erased.subject) for item in index["metadatas"]]
    values = {key: value for key, value in index.items() if value is not None}
    client = MemoryClient(erased.state / DATA / "chroma")
    try:
        client.client.get_collection(erased.saved.index_collection).upsert(**values)
    finally:
        client.close()


def restart_journal(erased: ErasedState) -> None:
    """A leased restart's journal recovery (real PrivacyServices), without any cleaner.

    ErasureJournal.recover reopens the local participants' steps to pending,
    exactly what a crash-restart does before the cleaners run again.
    """
    async def run():
        services = PrivacyServices(str(erased.state / DATA / "privacy.db"), clock=erased.clock)
        try:
            await services.acall(services.initialize)
        finally:
            await services.aclose()
    asyncio.run(run())


async def run_cleaners(erased: ErasedState) -> None:
    """Reopen the real graph on the restored state: leased recovery, then every cleaner, settled."""
    with _graph_token():
        async with integrated_graph(erased.state / DATA, erased.clock) as g:
            operation = {"id": erased.operation}
            await clear_browser(g, erased.subject_cohort)
            await settle(g, operation)
            await assert_honest_blockers(g, operation)


def steps(erased: ErasedState) -> list[dict]:
    """The subject operation's steps with their target participant/version (content-free)."""
    with sqlite3.connect(erased.state / DATA / "privacy.db") as db:
        db.row_factory = sqlite3.Row
        return [dict(row) for row in db.execute(
            "SELECT s.participant, s.version, s.target_id, s.state FROM erasure_steps s"
            " WHERE s.operation_id=? ORDER BY s.participant, s.target_id", (erased.operation,))]


def set_step(erased: ErasedState, participant: str, state: str) -> int:
    with sqlite3.connect(erased.state / DATA / "privacy.db") as db:
        return db.execute("UPDATE erasure_steps SET state=? WHERE operation_id=? AND participant=?",
                          (state, erased.operation, participant)).rowcount
