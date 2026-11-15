"""R5V-1: a browser owner orphaned by a restart and the real managed stop path.

The first boot issues a real browser permission and closes normally. The
recorder token lives only in the tab's memory (``capture-recorder.js``), so a
tab closed while the server was down leaves no one able to retire it. The
second boot turns that row ``unknown`` and carries it: the server grants it
nothing, and no stop or mute waits for an ACK that can no longer arrive. It is
never counted as an ACK either: ``total_mute_confirmed`` and the closure's
``capture_permissions_retired`` stay False while it exists. These tests hold
the token only to show that; using it is exactly what the lost tab can no
longer do. Presenting it proves a live holder, which is awaited again, so the
stop test uses it only after the owner set closed.
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
    # first process). Presenting the token reclaims it: the holder is alive.
    assert recorders.pending(kind="browser") == recorders.pending() == 1
    status = await recorders.poll(token, actor)
    assert status["state"] == "unknown" and status["capture_allowed"] is False
    assert status["retirement_required"] is True
    with pytest.raises(ValueError, match="^capture_recorder_actor_retired$"):
        await recorders.activate(token, actor)
    assert recorders.carried() == 0 and recorders.pending_live() == 1


def _carried(recorders) -> None:
    # Carried, never acknowledged: counted for confirmation, not awaited.
    assert recorders.pending(kind="browser") == recorders.pending() == 1
    assert recorders.carried() == 1 and recorders.pending_live() == 0


async def _managed_stop(tmp_path, monkeypatch, root, token, before_stop=None):
    """Boot, run Application.run's observed managed stop, return its receipt path.

    The lost tab never contacts this process. Nothing retires the carried
    owner; the stop completes on its own.
    """
    receipts, revalidated = [], []
    monkeypatch.setattr(runtime_admission, "_stage", lambda: SimpleNamespace(
        write_admission_receipt=lambda ticket, resources: receipts.append((ticket, resources))))
    monkeypatch.setattr(runtime_admission, "_open_state_writers", lambda state: ("checked", []))
    run_id = uuid4().hex
    observer = AsyncExitStack()
    async with application_graph(root, monkeypatch, Clock(time.time())) as bundle:
        graph = bundle.graph
        recorders = graph.capture_control.recorders
        _carried(recorders)
        if before_stop is not None:
            await before_stop(bundle)
        admission = runtime_admission.RuntimeAdmission({"run_id": run_id},
            {"state_root": str(tmp_path / "state")}, graph.privacy.ledger.path,
            runtime_admission._config_digest(graph.config))
        monkeypatch.setattr(admission, "revalidate", lambda: revalidated.append(True))
        admission.install_on(observer)
        admission.attach_graph(graph)
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
        late_actor = _actor(bundle)
    # The owned stack closed; the receipt path runs with the carried owner.
    await observer.__aexit__(None, None, None)
    assert admission.receipt_written and revalidated == [True]
    assert receipts == [({"run_id": run_id}, dict.fromkeys(
        ("graph_closed", "privacy_closed", "sdk_closed", "native_closed"), True))]
    proof = admission._closure_witness.verify_closed()
    assert proof["capture_permissions_retired"] is False
    assert proof["capture_owners_carried"] == 1
    return admission, recorders, late_actor, proof


async def test_owner_orphaned_by_restart_does_not_block_the_managed_stop(tmp_path, monkeypatch):
    root = tmp_path / "graph"
    token = await _orphaned_browser_owner(root, monkeypatch)
    admission, recorders, late_actor, proof = await _managed_stop(tmp_path, monkeypatch, root, token)
    # Closed: a late token is refused and no longer reclaims the owner.
    assert (await recorders.poll(token, late_actor))["state"] == "unknown"
    _carried(recorders)
    assert admission._closure_witness.verify_closed() == proof


async def _mute_and_unmute(bundle) -> None:
    control = bundle.graph.capture_control
    actor = _actor(bundle)
    control.request(True, control.snapshot()["revision"], actor)
    snapshot = await _phase(control, "muted")
    assert snapshot["total_mute_confirmed"] is False
    control.request(False, snapshot["revision"], actor)
    assert (await _phase(control, "running"))["pending_browser_recorders"] == 1
    _carried(control.recorders)


async def test_owner_stays_carried_and_unblocking_after_a_second_restart(tmp_path, monkeypatch):
    """R1B1-2: an owner carried by one boot is carried again by the next.

    The second boot found the row already ``unknown`` (no ``active`` row left
    to turn). It must still be carried there, or every stop from the second
    restart on would hang again (R5V-1).
    """
    root = tmp_path / "graph"
    token = await _orphaned_browser_owner(root, monkeypatch)
    await _managed_stop(tmp_path, monkeypatch, root, token)
    await _managed_stop(tmp_path, monkeypatch, root, token, before_stop=_mute_and_unmute)


async def test_owner_orphaned_by_restart_does_not_lock_global_mute_and_unmute(tmp_path, monkeypatch):
    root = tmp_path / "graph"
    token = await _orphaned_browser_owner(root, monkeypatch)
    async with application_graph(root, monkeypatch, Clock(time.time())) as bundle:
        control = bundle.graph.capture_control
        recorders = control.recorders
        actor = _actor(bundle)
        assert control.snapshot()["phase"] == "running"
        _carried(recorders)
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
            # A second mute is equally free and equally unconfirmed.
            control.request(True, snapshot["revision"], actor)
            snapshot = await _phase(control, "muted")
            assert snapshot["total_mute_confirmed"] is False
            control.request(False, snapshot["revision"], actor)
            snapshot = await _phase(control, "running")
            # Reactivation never revives the carried owner. Its holder shows
            # up, so it is reclaimed: the next mute waits for its retire.
            await _refused(recorders, token, actor)
            control.request(True, snapshot["revision"], actor)
            await _phase(control, "muting")
            with pytest.raises(TimeoutError):
                async with asyncio.timeout(.1):
                    await _phase(control, "muted")
            # Only an explicit retire is an ACK; then the mute is confirmed.
            assert await recorders.retire(token) == {"retired": True}
            snapshot = await _phase(control, "muted")
            assert recorders.carried() == recorders.pending() == 0
            assert snapshot["total_mute_confirmed"] is True
        finally:
            await recorders.retire(token)


async def _phase(control, phase):
    await until(lambda: control.snapshot()["phase"] == phase)
    return control.snapshot()
