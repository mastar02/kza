"""R5V-1: owners unknown at boot are carried, counted, and never awaited.

A carried owner's token lived only in a page of an earlier process. It keeps
counting for confirmation (``pending``) until an explicit retire, but it does
not hold a drain (``pending_live``) or a slot of this process's capacity.
"""
from __future__ import annotations

import asyncio
import logging
import threading

import pytest

from src.memory import capture_recorders
from src.memory.capture_recorders import CaptureRecorders
from src.memory.privacy_services import PrivacyServices
from tests.integration.test_capture_producer_privacy import until
from tests.integration.test_global_capture_control import actor


pytestmark = pytest.mark.asyncio


async def _boot(path: str) -> tuple[PrivacyServices, CaptureRecorders]:
    privacy = PrivacyServices(path)
    await privacy.acall(privacy.initialize)
    recorders = CaptureRecorders(privacy, capture_allowed=lambda: True)
    await recorders.initialize()
    return privacy, recorders


async def _close(privacy: PrivacyServices, recorders: CaptureRecorders) -> None:
    try:
        await asyncio.wait_for(recorders.aclose(), 2)
    finally:
        await privacy.aclose()


async def _issue(recorders: CaptureRecorders) -> str:
    principal = actor()
    token = (await recorders.register(principal))["token"]
    await recorders.activate(token, principal)
    return token


async def test_only_owners_unknown_at_boot_are_carried(tmp_path, caplog):
    path = str(tmp_path / "carried.sqlite")
    privacy, recorders = await _boot(path)
    try:
        lost = await _issue(recorders)
        assert recorders.carried() == 0 and recorders.pending_live() == 1
    finally:
        await _close(privacy, recorders)
    with caplog.at_level(logging.WARNING, logger=capture_recorders.__name__):
        privacy, recorders = await _boot(path)
    try:
        assert recorders.pending() == recorders.carried() == 1
        assert recorders.pending_live() == 0
        # Content-free: a count, never a token or its hash.
        warned = [r.getMessage() for r in caplog.records if r.name == capture_recorders.__name__]
        assert warned == ["[CaptureRecorders] 1 unacknowledged owner(s) carried from an "
                          "earlier process; never counted as retired"]
        assert recorders._key(lost) not in caplog.text and lost not in caplog.text
        # An owner issued by this process is live, never carried.
        live = await _issue(recorders)
        assert recorders.carried() == 1 and recorders.pending_live() == 1
        assert recorders.pending() == 2
        await recorders.retire(live)
        await asyncio.wait_for(recorders.drain(), 2)
        assert recorders.pending() == recorders.carried() == 1
    finally:
        await _close(privacy, recorders)


async def test_capacity_counts_live_owners_only(tmp_path, monkeypatch):
    assert capture_recorders.LIVE_OWNER_CAPACITY == 256
    path = str(tmp_path / "capacity.sqlite")
    privacy, recorders = await _boot(path)
    try:
        await _issue(recorders)
        await _issue(recorders)
    finally:
        await _close(privacy, recorders)
    monkeypatch.setattr(capture_recorders, "LIVE_OWNER_CAPACITY", 2)
    privacy, recorders = await _boot(path)
    try:
        assert recorders.carried() == 2
        # Carried rows take no live slot.
        first = await _issue(recorders)
        await _issue(recorders)
        assert recorders.pending() == 4 and recorders.pending_live() == 2
        with pytest.raises(ValueError, match="^capture_recorders_full$"):
            await recorders.register(actor())
        await recorders.retire(first)
        await _issue(recorders)
        assert recorders.pending_live() == 2 and recorders.carried() == 2
    finally:
        await _close(privacy, recorders)


async def test_owner_stays_carried_after_a_second_restart(tmp_path):
    """R1B1-2: the carried set is every row unknown at boot, not only rows
    this boot turned from active. A second restart carries it again."""
    path = str(tmp_path / "twice.sqlite")
    privacy, recorders = await _boot(path)
    try:
        await _issue(recorders)
    finally:
        await _close(privacy, recorders)
    for _ in range(2):
        privacy, recorders = await _boot(path)
        try:
            assert recorders.pending() == recorders.carried() == 1
            assert recorders.pending_live() == 0
            await asyncio.wait_for(recorders.drain(), 2)
            assert recorders.pending() == recorders.carried() == 1
        finally:
            await _close(privacy, recorders)


async def test_pending_live_counts_work_and_fault_exactly_like_pending(tmp_path, monkeypatch):
    """R1B1-5: pending_live is pending minus carried owners, term by term.

    An in-flight operation counts through both the operation set and the
    retained worker; a fault counts once. Only carried rows differ.
    """
    path = str(tmp_path / "terms.sqlite")
    privacy, recorders = await _boot(path)
    try:
        await _issue(recorders)
    finally:
        await _close(privacy, recorders)
    privacy, recorders = await _boot(path)
    try:
        assert recorders.carried() == recorders.pending() == 1 and recorders.pending_live() == 0
        entered, release = threading.Event(), threading.Event()

        def blocked():
            entered.set()
            assert release.wait(5)

        operation = asyncio.create_task(recorders._call(blocked))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            assert len(recorders._operations) == recorders._work.status()["operations"] == 1
            assert recorders.pending_live() == recorders.pending() - recorders.carried() == 2
        finally:
            release.set()
            await operation
        await until(lambda: not recorders._operations)
        assert recorders.pending_live() == 0
        live = await _issue(recorders)

        def failing(*args, **kwargs):
            raise OSError("synthetic ledger failure")

        with monkeypatch.context() as patch:
            patch.setattr(privacy.ledger, "_transaction", failing)
            with pytest.raises(OSError):
                await recorders.retire(live)
        assert recorders.pending_live() == recorders.pending() - recorders.carried() == 2
        with pytest.raises(RuntimeError, match="^capture_recorder_state_unconfirmed$"):
            await asyncio.wait_for(recorders.drain(), 2)
    finally:
        with pytest.raises(RuntimeError, match="^capture_recorder_state_unconfirmed$"):
            await asyncio.wait_for(recorders.aclose(), 2)
        await privacy.aclose()
