"""CaptureRecorders drain/aclose must make progress over already-finished work.

Regression for the full-suite hang in
tests/integration/test_capture_native_clients.py (T047 round-01 faulthandler at
src/memory/capture_recorders.py ``drain``). An operation task can be done while
its ``finished`` done-callback (which discards it from ``_operations``) is still
queued with ``call_soon``. ``asyncio.gather`` over only-done futures completes
synchronously and ``shield`` returns a done future as-is, so ``await`` never
yields: the queued callback never runs and the loop spins without yielding.

``asyncio.timeout`` cannot pre-empt a coroutine that never yields, so the
operation set below also bounds how many snapshots the drain loop may take.
A correct drain needs one or two; the spin takes unbounded snapshots.
"""
from __future__ import annotations

import asyncio

import pytest

from src.memory.capture_recorders import CaptureRecorders
from src.memory.privacy_services import PrivacyServices


pytestmark = pytest.mark.asyncio
SNAPSHOT_BOUND = 1000


class _BoundedSnapshots(set):
    """Operation set that fails after more drain snapshots than progress allows."""

    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.snapshots = 0

    def __iter__(self):
        self.snapshots += 1
        if self.snapshots > SNAPSHOT_BOUND:
            raise AssertionError("capture_recorders_drain_spun_without_yielding")
        return super().__iter__()


async def _finished_with_pending_discard(recorders: CaptureRecorders, monkeypatch,
                                         outcome: BaseException | None):
    """Admit one recorder operation and release it so it completes next step.

    Returns the operation task and the observer awaiting it. The caller then
    schedules drain/aclose *after* the release, so on the next loop iteration
    the operation finishes first and its ``finished`` discard is queued behind
    the drain step: drain observes a done member still in ``_operations``.
    """
    gate = asyncio.Event()

    async def gated(function, *args):
        await gate.wait()
        if outcome is not None:
            raise outcome
        return None

    with monkeypatch.context() as patch:
        patch.setattr(recorders._work, "run", gated)
        observer = asyncio.create_task(recorders._call(lambda: None))
        await asyncio.sleep(0)  # observer admits the operation and awaits it
    (operation,) = tuple(set.__iter__(recorders._operations))
    assert not operation.done()
    gate.set()  # schedules the operation's final step before what follows
    return operation, observer


async def _recorders(tmp_path) -> tuple[PrivacyServices, CaptureRecorders]:
    privacy = PrivacyServices(str(tmp_path / "recorders.sqlite"))
    await privacy.acall(privacy.initialize)
    recorders = CaptureRecorders(privacy, capture_allowed=lambda: True)
    await recorders.initialize()
    recorders._operations = _BoundedSnapshots(recorders._operations)
    return privacy, recorders


async def test_drain_progresses_when_finished_operation_discard_is_still_queued(tmp_path, monkeypatch):
    privacy, recorders = await _recorders(tmp_path)
    try:
        operation, observer = await _finished_with_pending_discard(recorders, monkeypatch, None)
        drainer = asyncio.create_task(recorders.drain())
        async with asyncio.timeout(5):
            await drainer
        assert operation.done() and observer.done()
        await observer
        assert recorders._operations.snapshots <= 3
        assert recorders.pending() == 0
    finally:
        await recorders.aclose()
        await privacy.aclose()


async def test_aclose_progresses_and_still_surfaces_a_finished_operation_failure(tmp_path, monkeypatch):
    privacy, recorders = await _recorders(tmp_path)
    try:
        operation, observer = await _finished_with_pending_discard(
            recorders, monkeypatch, RuntimeError("native_failure"))
        closer = asyncio.create_task(recorders.aclose())
        async with asyncio.timeout(5):
            with pytest.raises(RuntimeError, match="capture_recorder_state_unconfirmed"):
                await closer
        assert operation.done()
        with pytest.raises(RuntimeError, match="native_failure"):
            await observer
        assert recorders._operations.snapshots <= 3
        assert not recorders._operations
    finally:
        await privacy.aclose()
