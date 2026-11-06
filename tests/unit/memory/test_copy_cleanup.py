"""Selective IO/protocol tests; no fabricated closed Graph or cleanup grant.

Public successful claims are exercised by the compiled controller integration.
These fixtures use real collector APIs and test bounded IO and denied authority.
"""
import asyncio
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import pickle
import sqlite3
import threading
from types import SimpleNamespace
from uuid import uuid4

import pytest

from scripts import stage_runtime as stage
from src.core.owned_work import OwnedBlockingWork
from src.memory import copy_cleanup as cleanup
from src.memory.erasure_journal import ErasureParticipant
from tests.mocks.conversations import graph


@pytest.fixture
async def corpus(tmp_path):
    root = (tmp_path / "copy").resolve()
    async with graph(root) as g:
        subject = g.sources()
        foreign = g.sources()
        subject_id = await g.add(sources=subject, user_input="Subject input", assistant_response="Subject output")
        cursor = max(row["sequence"] for row in await g.base.rows())
        await g.add(sources=foreign, user_input="Foreign manual input", assistant_response="Foreign output")
        views = await g.collector.get_turns()
        try:
            view = next(item for item in views if item.id != subject_id)
            assert await g.collector.mark_response(view, "corrected", "Foreign latest correction",
                                                   sources=(foreign[0],))
        finally:
            for view in views:
                view.clear()
        views = await g.collector.get_turns()
        try:
            foreign_id, = (item.id for item in views if item.quality.value == "corrected")
        finally:
            for view in views:
                view.clear()
        exported = await g.collector.export_for_training(str(root / "exports"), only_marked=False)
        assert exported is not None
        mixed = exported.id
        exported.clear()
        exported = await g.collector.export_for_training(str(root / "exports"), only_marked=False,
                                                         after_sequence=cursor)
        assert exported is not None
        foreign_export = exported.id
        exported.clear()
        targets = await g.privacy.acall(g.privacy.ledger.erasures.targets, ErasureParticipant.COLLECTORS)
    assert g.privacy.resource_status()["close_phase"] == "complete"
    layout = cleanup.CollectorCopyLayout("privacy.db", tuple(cleanup.CollectorRoot(item["id"],
        item["descriptor"]["kind"], Path(item["descriptor"]["root"]).relative_to(root).as_posix())
        for item in targets))
    return root, layout, subject[0].conversation_id, subject_id, foreign_id, mixed, foreign_export


def capture(corpus):
    root, layout, conversation, *_ = corpus
    return cleanup._capture(stage, root, layout, conversation)


async def test_actual_mixed_export_withdrawal_does_not_select_foreign_manual_coparent(corpus):
    plan = capture(corpus)
    _, _, _, subject, foreign, mixed, foreign_export = corpus
    assert {subject, mixed} <= plan["ids"]
    assert foreign not in plan["ids"] and foreign_export not in plan["ids"]
    assert {foreign, foreign_export} <= plan["manual"]
    assert any(item["id"] == mixed for item in plan["files"].values())
    assert any(item["id"] == foreign for item in plan["files"].values())
    assert all("Subject input" not in value for value in (plan["preserved_rows"], plan["preserved_files"]))


@pytest.mark.parametrize("change", ["target", "root", "missing_root"])
async def test_layout_cannot_relabel_actual_registered_collector_ownership(corpus, change):
    root, layout, conversation, *_ = corpus
    first, *others = layout.roots
    if change == "target":
        changed = replace(first, target_id="f" * 64)
    elif change == "root":
        changed = replace(first, relative_root="another-root")
    else:
        changed = None
    bad = replace(layout, roots=tuple(others) if changed is None else (changed, *others))
    with pytest.raises(cleanup.CopyCleanupError, match="layout_target_coverage"):
        cleanup._capture(stage, root, bad, conversation)


@pytest.mark.parametrize("kind", ["unknown", "symlink", "hardlink"])
async def test_unowned_or_aliased_collector_files_block_before_mutation(corpus, kind):
    root, _, _, _, _, mixed, _ = corpus
    plan = capture(corpus)
    relative, = (key for key, item in plan["files"].items() if item["id"] == mixed)
    original = root / relative
    before = original.read_bytes()
    path = original.with_name("unowned.bin") if kind == "unknown" else original.with_name(str(uuid4()) + ".jsonl")
    if kind == "unknown":
        path.write_bytes(b"unreviewed")
    elif kind == "symlink":
        path.symlink_to(original)
    else:
        path.hardlink_to(original)
    try:
        with pytest.raises(cleanup.CopyCleanupError):
            capture(corpus)
        assert original.read_bytes() == before
    finally:
        path.unlink()


async def test_real_unlink_changes_only_selected_file_and_preserves_manual_bytes(corpus):
    root, _, _, subject, foreign, _, _ = corpus
    plan = capture(corpus)
    removed, = (item for item in plan["files"].values() if item["id"] == subject)
    foreign_files = {key: (root / key).read_bytes() for key, item in plan["files"].items() if item["id"] == foreign}
    target = plan["roots"][removed["target_id"]]
    assert cleanup._unlink_file(root / target.relative_root, subject, removed["suffix"]) is True
    assert cleanup._unlink_file(root / target.relative_root, subject, removed["suffix"]) is False
    assert {key: (root / key).read_bytes() for key in foreign_files} == foreign_files


def test_checkpoint_busy_and_retained_wal_never_yield_a_retirement_witness(tmp_path):
    path = tmp_path / "payload.db"
    writer = sqlite3.connect(path, timeout=.01)
    reader = sqlite3.connect(path, timeout=.01)
    try:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("CREATE TABLE records(value TEXT)")
        writer.execute("INSERT INTO records VALUES ('foreign exact')")
        writer.commit()
        reader.execute("BEGIN")
        assert reader.execute("SELECT value FROM records").fetchall() == [("foreign exact",)]
        writer.execute("INSERT INTO records VALUES ('subject')")
        writer.commit()
        with pytest.raises(cleanup.CopyCleanupError, match="checkpoint_unconfirmed"):
            cleanup._checkpoint(writer)
        with pytest.raises(cleanup.CopyCleanupError, match="sidecar_retention_unknown"):
            cleanup._check_sidecars(path)
        reader.rollback()
        cleanup._checkpoint(writer)
    finally:
        reader.close()
        writer.close()
    cleanup._check_sidecars(path)


@pytest.mark.parametrize("suffix", ["-wal", "-journal"])
def test_nonempty_postclose_sidecar_is_not_silently_deleted(tmp_path, suffix):
    path = tmp_path / "payload.db"
    sidecar = Path(str(path) + suffix)
    sidecar.write_bytes(b"retained old payload")
    with pytest.raises(cleanup.CopyCleanupError, match="sidecar_retention_unknown"):
        cleanup._check_sidecars(path)
    assert sidecar.read_bytes() == b"retained old payload"


@pytest.fixture
def protocol():
    """A journal codec input, not a closed controller or executable claim."""
    return {"schema_version": 1, "scope": cleanup.SCOPE, "copy_id": uuid4().hex,
        "claim_id": uuid4().hex, "operation_id": str(uuid4()), "conversation_id": str(uuid4()),
        "authority_head": {"schema_version": 1, "generation": 3, "binding_id": "a" * 64,
                           "status": "closed", "run_id": None},
        "catalog_revision": 10, "catalog_record_sha256": "b" * 64, "layout_sha256": "c" * 64,
        "fence_sha256": "d" * 64, "phase": "claimed", "native_started": False,
        "native_returned": False, "receipt": None, "error_code": None}


def journal(tmp_path, protocol):
    path = (tmp_path / "claims" / protocol["copy_id"]).resolve()
    path.mkdir(parents=True)
    return path


def test_cleanup_protocol_cas_is_separate_and_history_cannot_skip_native_return(tmp_path, protocol):
    path = journal(tmp_path, protocol)
    head = cleanup._append(path, protocol, None)
    running = dict(protocol, phase="running", native_started=True)
    new_head = cleanup._append(path, running, head)
    with pytest.raises(cleanup.CopyCleanupError, match="claim_cas"):
        cleanup._append(path, dict(running, phase="failed", native_returned=True, error_code="OSError"), head)
    with pytest.raises(cleanup.CopyCleanupError, match="journal_receipt"):
        cleanup._append(path, dict(running, phase="complete"), new_head)
    assert cleanup._state(path) == (running, new_head)


@pytest.mark.parametrize("field", ["schema_version", "catalog_revision", "authority_head.schema_version",
                                    "authority_head.generation", "native_started", "native_returned"])
def test_malformed_typed_protocol_does_not_publish_an_event(tmp_path, protocol, field):
    path = journal(tmp_path, protocol)
    changed = deepcopy(protocol)
    holder, name = (changed["authority_head"], field.split(".")[1]) if "." in field else (changed, field)
    holder[name] = 1 if name.startswith("native_") else True
    with pytest.raises(cleanup.CopyCleanupError):
        cleanup._append(path, changed, None)
    assert not list(path.iterdir())


def test_orphaned_cleanup_event_is_unknown_without_repair_or_permission(tmp_path, protocol, monkeypatch):
    path = journal(tmp_path, protocol)
    head = cleanup._append(path, protocol, None)
    before = (path / "head.json").read_bytes()
    from src.core.runtime_admission import _stage
    original = _stage()._atomic_record

    def failed(name, value, **kwargs):
        if name == path / "head.json":
            raise OSError("synthetic event before head crash")
        return original(name, value, **kwargs)

    monkeypatch.setattr(_stage(), "_atomic_record", failed)
    with pytest.raises(OSError):
        cleanup._append(path, dict(protocol, phase="running", native_started=True), head)
    with pytest.raises(cleanup.CopyCleanupError, match="journal_orphan"):
        cleanup._state(path)
    assert (path / "head.json").read_bytes() == before and len(list(path.iterdir())) == 3


@pytest.mark.parametrize("purported", ["history", "namespace"])
def test_history_and_nominal_strings_are_not_current_controller_authority(tmp_path, purported):
    layout = cleanup.CollectorCopyLayout("privacy.db", (cleanup.CollectorRoot("a" * 64, "wake_clips", "clips"),))
    value = {"copy_id": uuid4().hex, "phase": "complete"} if purported == "history" else SimpleNamespace(root=tmp_path)
    with pytest.raises(cleanup.catalog.CopyCatalogError, match="controller$"):
        cleanup.CollectorCopyCleanup(value, uuid4().hex, layout=layout)
    assert not (tmp_path / "admission").exists()


def test_real_controller_without_current_closed_head_cannot_mutate_copied_payload(tmp_path):
    root = (tmp_path / "controller").resolve()
    layout = cleanup.CollectorCopyLayout("privacy.db", (cleanup.CollectorRoot("a" * 64, "wake_clips", "clips"),))
    with stage.AdmissionController(root) as controller:
        with pytest.raises(cleanup.catalog.CopyCatalogError, match="closed_head$"):
            cleanup.CollectorCopyCleanup(controller, uuid4().hex, layout=layout)
        with pytest.raises(cleanup.catalog.CopyCatalogError, match="closed_head$"):
            cleanup.read_copy_cleanup(controller, uuid4().hex)
    assert not (root / "admission/copy-cleanup").exists()


def test_claim_serialization_cannot_transfer_the_local_capability():
    claim = cleanup.CopyCleanupClaim(uuid4().hex, uuid4().hex, str(uuid4()), str(uuid4()), 1, "a" * 64,
        1, "b" * 64, {}, "c" * 64, "d" * 64)
    with pytest.raises(TypeError, match="not_serializable"):
        pickle.dumps(claim)


async def test_cancelled_unlink_observer_does_not_replace_actual_worker_return(corpus):
    root, _, _, subject, *_ = corpus
    plan = capture(corpus)
    item, = (row for row in plan["files"].values() if row["id"] == subject)
    target = plan["roots"][item["target_id"]]
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    work = OwnedBlockingWork(capacity=1, close_timeout_s=.1, isolate_context=True)
    workers = []

    def native():
        workers.append(threading.current_thread())
        try:
            result = cleanup._unlink_file(root / target.relative_root, subject, item["suffix"])
            entered.set()
            assert release.wait(3)
            return result
        finally:
            returned.set()

    observer = asyncio.create_task(work.run(native))
    try:
        async with asyncio.timeout(2):
            while not entered.is_set():
                await asyncio.sleep(.001)
        observer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await observer
        drain = work.begin_close()
        assert not drain.done() and not returned.is_set() and work.status()["operations"] == 1
        assert workers and workers[0] is not threading.current_thread()
    finally:
        release.set()
        await work.begin_close()
        await asyncio.gather(observer, return_exceptions=True)
    assert returned.is_set() and work.status()["operations"] == 0
