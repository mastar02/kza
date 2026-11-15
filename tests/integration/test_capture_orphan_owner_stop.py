"""R5V-1: a browser owner orphaned by a restart and the real managed stop path.

The first boot issues a real browser permission and closes normally. The
recorder token lives only in the tab's memory (``capture-recorder.js``), so a
tab closed while the server was down leaves no one able to retire it. The
second boot turns that row ``unknown`` and carries it: the server grants it
nothing, and no stop or mute waits for an ACK that can no longer arrive. It is
never counted as an ACK either: ``total_mute_confirmed`` and the closure's
``capture_permissions_retired`` stay False while it exists. These tests hold
the token only to show that; using it is exactly what the lost tab can no
longer do.
"""
import asyncio
from contextlib import AsyncExitStack
import time
from types import SimpleNamespace
from uuid import uuid4

import pytest

from src.core import runtime_admission
from src.memory.capture_control import CaptureController
from tests.mocks.application_restore import root_lease
from tests.mocks.memory_graph import Clock
from tests.mocks.restore_application_graph import application_graph, until


pytestmark = pytest.mark.asyncio


def _actor(bundle):
    lease = root_lease(bundle)
    return CaptureController.from_dashboard(lease.principal, lease.current)


async def _orphaned_browser_owner(root, monkeypatch) -> str:
    """Boot once, issue a real browser permission, close normally."""
    async with application_graph(root, monkeypatch, Clock(time.time())) as bundle:
        recorders = bundle.graph.capture_control.recorders
        actor = _actor(bundle)
        issued = await recorders.register(actor)
        active = await recorders.activate(issued["token"], actor)
        assert active["state"] == "active" and active["capture_allowed"] is True
    return issued["token"]


async def _refused(recorders, token, actor) -> None:
    # Boot turned the issued permission into an unknown owner. The server
    # grants it nothing: no capture, no reactivation (its actor died with the
    # first process).
    assert recorders.pending(kind="browser") == recorders.pending() == 1
    status = await recorders.poll(token, actor)
    assert status["state"] == "unknown" and status["capture_allowed"] is False
    with pytest.raises(ValueError, match="^capture_recorder_actor_retired$"):
        await recorders.activate(token, actor)


def _carried(recorders) -> None:
    # Carried, never acknowledged: counted for confirmation, not awaited.
    assert recorders.pending(kind="browser") == recorders.pending() == 1
    assert recorders.carried() == 1 and recorders.pending_live() == 0


async def test_owner_orphaned_by_restart_does_not_block_the_managed_stop(tmp_path, monkeypatch):
    root = tmp_path / "graph"
    token = await _orphaned_browser_owner(root, monkeypatch)
    receipts, revalidated = [], []
    monkeypatch.setattr(runtime_admission, "_stage", lambda: SimpleNamespace(
        write_admission_receipt=lambda ticket, resources: receipts.append((ticket, resources))))
    monkeypatch.setattr(runtime_admission, "_open_state_writers", lambda state: ("checked", []))
    run_id = uuid4().hex
    observer = AsyncExitStack()
    async with application_graph(root, monkeypatch, Clock(time.time())) as bundle:
        graph = bundle.graph
        recorders = graph.capture_control.recorders
        await _refused(recorders, token, _actor(bundle))
        admission = runtime_admission.RuntimeAdmission({"run_id": run_id},
            {"state_root": str(tmp_path / "state")}, graph.privacy.ledger.path,
            runtime_admission._config_digest(graph.config))
        monkeypatch.setattr(admission, "revalidate", lambda: revalidated.append(True))
        admission.install_on(observer)
        admission.attach_graph(graph)
        # Application.run's finally calls exactly this observed stop. Nothing
        # retires the carried owner; the stop completes on its own.
        stopping = asyncio.create_task(graph._stop_services(None, None))
        try:
            await asyncio.wait_for(asyncio.shield(stopping), 30)
        finally:
            if not stopping.done():
                # Failure path only: release the fixture with the token.
                await recorders.retire(token)
                await asyncio.gather(stopping, return_exceptions=True)
        assert admission._closure_prepared and not admission._closure_failed
        # The row is kept durable and unacknowledged, never invented retired.
        _carried(recorders)
        assert (await recorders.poll(token, _actor(bundle)))["state"] == "unknown"
    # The owned stack closed; the receipt path runs with the carried owner.
    await observer.__aexit__(None, None, None)
    assert admission.receipt_written and revalidated == [True]
    assert receipts == [({"run_id": run_id}, dict.fromkeys(
        ("graph_closed", "privacy_closed", "sdk_closed", "native_closed"), True))]
    proof = admission._closure_witness.verify_closed()
    assert proof["capture_permissions_retired"] is False
    assert proof["capture_owners_carried"] == 1


async def test_owner_orphaned_by_restart_does_not_lock_global_mute_and_unmute(tmp_path, monkeypatch):
    root = tmp_path / "graph"
    token = await _orphaned_browser_owner(root, monkeypatch)
    async with application_graph(root, monkeypatch, Clock(time.time())) as bundle:
        control = bundle.graph.capture_control
        recorders = control.recorders
        actor = _actor(bundle)
        assert control.snapshot()["phase"] == "running"
        await _refused(recorders, token, actor)
        try:
            control.request(True, control.snapshot()["revision"], actor)
            snapshot = await _phase(control, "muted")
            _carried(recorders)
            # Muted, but never confirmed while the carried owner exists.
            assert snapshot["persisted_muted"] is True and snapshot["ingress_sealed"] is True
            assert snapshot["total_mute_confirmed"] is False
            assert snapshot["pending_browser_recorders"] == 1
            # The authenticated control can always reactivate a total mute.
            control.request(False, snapshot["revision"], actor)
            snapshot = await _phase(control, "running")
            assert snapshot["persisted_muted"] is False
            assert snapshot["pending_browser_recorders"] == 1
            # Reactivation never revives the carried owner.
            status = await recorders.poll(token, actor)
            assert status["state"] == "unknown" and status["capture_allowed"] is False
            with pytest.raises(ValueError, match="^capture_recorder_actor_retired$"):
                await recorders.activate(token, actor)
            # A second mute is equally free and equally unconfirmed.
            control.request(True, snapshot["revision"], actor)
            assert (await _phase(control, "muted"))["total_mute_confirmed"] is False
            # Only an explicit retire is an ACK; then the mute is confirmed.
            assert await recorders.retire(token) == {"retired": True}
            assert recorders.carried() == recorders.pending() == 0
            assert control.snapshot()["total_mute_confirmed"] is True
        finally:
            await recorders.retire(token)


async def _phase(control, phase):
    await until(lambda: control.snapshot()["phase"] == phase)
    return control.snapshot()
