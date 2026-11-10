"""OQ-8 B3: authority -> gate -> durable marker -> queued shutdown write.

The managed journal is the configuration publisher harness (synthetic
admission validation; real authority, owner, coordinator and owned stack).
"""
import asyncio

import pytest

from src.core.egress import ConsentWithdrawalSource
from src.core.state_configuration import ConfigurationChanges, MarkerRecordState
from src.dashboard.authority import DashboardRole
from src.memory import output_registration_v2 as registration
from tests.integration.test_configuration_publisher_registration import (
    RAW, config_error, configuration_graph, drained, graph_ready, messages, phases, shutdown,
)


async def reached_configuration_exit(g, closing: asyncio.Task) -> None:
    """Hold the withdrawal until the owned stack is inside ConfigurationChanges' exit.

    Earlier exits (privacy services) take several loop turns; releasing before
    ``_after_graph_close`` runs would let the withdrawal finish first and hide
    whether close itself awaits it.
    """
    async with asyncio.timeout(5):
        while not g.changes._closed:
            assert not closing.done()
            await asyncio.sleep(0.001)
    for _ in range(20):
        await asyncio.sleep(0)

async def test_unauthorized_actor_has_no_effect(tmp_path, monkeypatch):
    async with configuration_graph(tmp_path, monkeypatch) as g:
        issued = await g.authority.issue_credential(g.actor, subject="synthetic-home", role=DashboardRole.HOME,
                                                    expires_at=g.authority._now() + 600)
        home = g.authority.resolve(issued.token)
        for actor in (home, None, object()):
            with pytest.raises(PermissionError, match="^configuration_actor_denied$"):
                await g.changes.withdraw_cloud_consent(actor)
        assert g.gate.withdrawn is False and phases(g.owner) == ["intent", "attached"]
        assert g.changes.pending is None and not g.owner._failed


async def test_gate_closes_before_the_journal_append_and_revalidate_after(tmp_path, monkeypatch):
    async with configuration_graph(tmp_path, monkeypatch) as g:
        order = []
        real = registration.OutputRegistrationV2._append_configuration

        def append(self, phase, plan, aggregate=None):
            order.append((phase, g.gate.withdrawn))
            return real(self, phase, plan, aggregate)
        monkeypatch.setattr(registration.OutputRegistrationV2, "_append_configuration", append)
        revalidate = g.admission.revalidate

        def spy():
            order.append(("revalidate", g.gate.withdrawn, phases(g.owner)[-1]))
            return revalidate()
        monkeypatch.setattr(g.admission, "revalidate", spy)
        outcome = await g.changes.withdraw_cloud_consent(g.actor)
        assert outcome.marker is MarkerRecordState.DURABLE and outcome.pending is not None
        assert order[0] == ("withdrawn", True)
        assert all(item[1] is True and item[2] == "withdrawn" for item in order if item[0] == "revalidate")
        assert any(item[0] == "revalidate" for item in order)
        assert g.gate.source is ConsentWithdrawalSource.ACTOR


async def test_journal_failure_poisons_and_withholds_the_receipt(tmp_path, monkeypatch):
    async with configuration_graph(tmp_path, monkeypatch) as g:
        graph_ready(g)
        real = registration.OutputRegistrationV2._append_configuration

        def append(self, phase, plan, aggregate=None):
            if phase == "withdrawn":
                raise OSError("synthetic ENOSPC before the record exists")
            return real(self, phase, plan, aggregate)
        monkeypatch.setattr(registration.OutputRegistrationV2, "_append_configuration", append)
        outcome = await g.changes.withdraw_cloud_consent(g.actor)
        assert outcome.gate_closed is True and g.gate.withdrawn is True
        assert outcome.marker is MarkerRecordState.REGISTRATION_FAILED and outcome.pending is None
        assert outcome.failure == "output_registration_unconfirmed"
        assert g.owner._failed is True and g.changes.pending is None
        await shutdown(g)
        assert "configuration_withdrawal_unconfirmed" in messages(config_error(g))
        assert "drained" not in phases(g.owner) and g.settings.read_bytes() == RAW
        assert str(g.owner._closing.exception()) in {
            "output_registration_epoch_incomplete", "output_registration_registry_unavailable"}


async def test_queue_failure_after_durable_marker_keeps_marker_and_clean_epoch(tmp_path, monkeypatch):
    async with configuration_graph(tmp_path, monkeypatch) as g:
        graph_ready(g)
        g.settings.write_bytes(RAW.replace(b"consent: true", b'consent: "true"'))
        outcome = await g.changes.withdraw_cloud_consent(g.actor)
        assert outcome.marker is MarkerRecordState.DURABLE and outcome.gate_closed is True
        assert outcome.pending is None and outcome.failure == "configuration_consent_invalid"
        assert phases(g.owner) == ["intent", "attached", "withdrawn"] and not g.owner._failed
        g.settings.write_bytes(RAW)  # Restore the attached file so the terminal observation matches.
        assert await shutdown(g) is None
        assert phases(g.owner) == ["intent", "attached", "withdrawn", "drained"]
        state = drained(g.owner)
        assert state["aggregate"]["outcome"] == "no_pending" and state["marker_out"]["origin"] == "actor"


async def test_queue_pending_without_durable_marker_is_refused(tmp_path, monkeypatch):
    async with configuration_graph(tmp_path, monkeypatch) as g:
        with pytest.raises(RuntimeError, match="^configuration_marker_required$"):
            await asyncio.to_thread(g.changes._queue_pending, g.actor)
        assert g.changes.pending is None
        assert not hasattr(ConfigurationChanges, "queue_consent_withdrawal")


async def test_second_call_is_already_recorded_and_writes_one_marker(tmp_path, monkeypatch):
    async with configuration_graph(tmp_path, monkeypatch) as g:
        first = await g.changes.withdraw_cloud_consent(g.actor)
        second = await g.changes.withdraw_cloud_consent(g.actor)
        assert second.marker is MarkerRecordState.ALREADY_RECORDED and second.pending is first.pending
        assert phases(g.owner).count("withdrawn") == 1


async def test_close_awaits_an_in_flight_withdrawal(tmp_path, monkeypatch):
    async with configuration_graph(tmp_path, monkeypatch) as g:
        graph_ready(g)
        entered, release = asyncio.Event(), asyncio.Event()
        real = registration.configuration_withdrawn

        async def blocked(owner):
            entered.set()
            await release.wait()
            return await real(owner)
        monkeypatch.setattr(registration, "configuration_withdrawn", blocked)
        request = asyncio.create_task(g.changes.withdraw_cloud_consent(g.actor))
        await asyncio.wait_for(entered.wait(), 5)
        closing = asyncio.create_task(shutdown(g))
        await reached_configuration_exit(g, closing)
        assert not closing.done() and "drained" not in phases(g.owner)
        release.set()
        outcome = await request
        assert await closing is None
        # The marker is durable before the drain; the owner already closed, so no write is queued.
        assert outcome.marker is MarkerRecordState.DURABLE and outcome.gate_closed
        assert outcome.pending is None and outcome.failure == "configuration_owner_not_open"
        assert phases(g.owner) == ["intent", "attached", "withdrawn", "drained"]
        assert drained(g.owner)["aggregate"]["outcome"] == "no_pending"
        assert drained(g.owner)["withdrawal"] is not None


async def test_cancelled_request_after_gate_close_poisons_and_withholds_receipt(tmp_path, monkeypatch):
    async with configuration_graph(tmp_path, monkeypatch) as g:
        graph_ready(g)
        entered = asyncio.Event()

        async def never(owner):
            entered.set()
            await asyncio.Event().wait()
        monkeypatch.setattr(registration, "configuration_withdrawn", never)
        request = asyncio.create_task(g.changes.withdraw_cloud_consent(g.actor))
        await asyncio.wait_for(entered.wait(), 5)
        g.changes._withdrawing.cancel()
        outcome = await request
        assert outcome.marker is MarkerRecordState.REGISTRATION_FAILED and g.owner._failed
        await shutdown(g)
        assert "configuration_withdrawal_unconfirmed" in messages(config_error(g))
        assert "drained" not in phases(g.owner)


async def test_graph_exit_while_marker_write_suspended_then_failing_writes_no_receipt(tmp_path, monkeypatch):
    """N1: close settles the withdrawal before reading _pending; a late failure leaves no receipt."""
    from types import SimpleNamespace
    from src.core import runtime_admission
    receipts = []
    monkeypatch.setattr(runtime_admission, "_stage",
                        lambda: SimpleNamespace(write_admission_receipt=lambda *a: receipts.append(a)))
    async with configuration_graph(tmp_path, monkeypatch) as g:
        graph_ready(g)
        entered, release = asyncio.Event(), asyncio.Event()

        async def failing(owner):
            entered.set()
            await release.wait()
            raise registration.OutputRegistrationError("output_registration_capacity")
        monkeypatch.setattr(registration, "configuration_withdrawn", failing)
        request = asyncio.create_task(g.changes.withdraw_cloud_consent(g.actor))
        await asyncio.wait_for(entered.wait(), 5)
        closing = asyncio.create_task(shutdown(g))
        await reached_configuration_exit(g, closing)
        assert not closing.done() and "drained" not in phases(g.owner)
        release.set()
        outcome = await request
        await closing
        assert outcome.marker is MarkerRecordState.REGISTRATION_FAILED
        assert outcome.failure == "output_registration_capacity" and g.owner._failed
        assert "configuration_withdrawal_unconfirmed" in messages(config_error(g))
        assert "drained" not in phases(g.owner) and receipts == []
        assert g.settings.read_bytes() == RAW


async def test_failure_inside_gate_close_poisons(tmp_path, monkeypatch):
    """N1: the poisoning try starts right after the authority check."""
    async with configuration_graph(tmp_path, monkeypatch) as g:
        graph_ready(g)
        real = type(g.gate).withdraw

        def closing_then_failing(self, source):
            real(self, source)
            raise RuntimeError("synthetic failure after the latch")
        monkeypatch.setattr(type(g.gate), "withdraw", closing_then_failing)
        outcome = await g.changes.withdraw_cloud_consent(g.actor)
        assert g.gate.withdrawn and outcome.marker is MarkerRecordState.REGISTRATION_FAILED
        assert g.owner._failed and "withdrawn" not in phases(g.owner)
        await shutdown(g)
        assert "configuration_withdrawal_unconfirmed" in messages(config_error(g))


async def test_managed_run_without_coordinator_refuses_never_unmanaged(tmp_path, monkeypatch):
    """N2: unmanaged only when not _managed(); a missing coordinator never fails open."""
    async with configuration_graph(tmp_path, monkeypatch) as g:
        graph_ready(g)
        assert registration._managed()
        monkeypatch.setattr(registration, "_coordinator", lambda owner=None: None)
        with pytest.raises(registration.OutputRegistrationError, match="^output_registration_required$"):
            await registration.configuration_withdrawn(g.changes)
        outcome = await g.changes.withdraw_cloud_consent(g.actor)
        assert outcome.marker is MarkerRecordState.REGISTRATION_FAILED
        assert outcome.failure == "output_registration_required" and outcome.pending is None
        assert g.gate.withdrawn
