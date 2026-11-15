"""R1-B2: an administrator forgets lost recordings (carried owners).

User decision 2026-11-15 (``r1b1_lost_recordings``): an administrator forgets
them from Privacidad, affirming no tab is still recording, with the carried
count the page showed as confirmation. Only owners still carried are
forgotten (never live or reclaimed ones); a repeat is a no-op; the only
record is a content-free log line with the count.
"""
from __future__ import annotations

import asyncio
import logging
import threading

import pytest

from src.memory import capture_recorders
from src.memory.capture_recorders import CaptureRecorders
from src.memory.privacy_services import PrivacyServices
from tests.integration.test_global_capture_control import actor
from tests.unit.memory.test_capture_recorders_carried import _boot, _close, _issue


pytestmark = pytest.mark.asyncio


def _allowed() -> None:
    return None


def _rows(privacy: PrivacyServices) -> dict[str, str]:
    with privacy.ledger._lock:
        return {str(row[0]): str(row[1]) for row in privacy.ledger._db.execute(
            "SELECT token_hash, state FROM capture_browser_recorders").fetchall()}


async def _carried_boot(tmp_path, name: str, count: int = 2):
    """Issue ``count`` owners in a first process; boot a second that carries them."""
    path = str(tmp_path / name)
    privacy, recorders = await _boot(path)
    try:
        tokens = [await _issue(recorders) for _ in range(count)]
    finally:
        await _close(privacy, recorders)
    privacy, recorders = await _boot(path)
    assert recorders.carried() == recorders.pending() == count
    return path, privacy, recorders, tokens


def _forget_lines(caplog) -> list[str]:
    return [record.getMessage() for record in caplog.records
            if record.name == capture_recorders.__name__ and "forgot" in record.getMessage()]


async def test_forget_deletes_only_carried_rows_and_logs_only_the_count(tmp_path, caplog):
    path, privacy, recorders, lost = await _carried_boot(tmp_path, "forget.sqlite")
    try:
        principal = actor()
        live = await _issue(recorders)
        assert recorders.pending() == 3 and recorders.pending_live() == 1
        with caplog.at_level(logging.WARNING, logger=capture_recorders.__name__):
            assert await recorders.forget_carried(2, authorize=_allowed) == 2
        assert recorders.carried() == 0
        assert recorders.pending() == recorders.pending_live() == recorders.live_owners() == 1
        # The live owner is untouched: its row is still active and granted.
        assert _rows(privacy) == {recorders._key(live): "active"}
        assert (await recorders.poll(live, principal))["capture_allowed"] is True
        # The forgotten holders are told to stop, as any retired owner.
        status = await recorders.poll(lost[0], principal)
        assert status["state"] == "retired" and status["retirement_required"] is True
        assert _forget_lines(caplog) == ["[CaptureRecorders] an administrator forgot 2 carried "
                                         "owner(s), affirming none is still recording"]
        for token in lost:
            assert token not in caplog.text and recorders._key(token) not in caplog.text
        await recorders.retire(live)
    finally:
        await _close(privacy, recorders)
    # Durable: the next process carries nothing.
    privacy, recorders = await _boot(path)
    try:
        assert recorders.pending() == recorders.carried() == 0
    finally:
        await _close(privacy, recorders)


@pytest.mark.parametrize("expected", [0, 1, 3])
async def test_a_count_other_than_the_carried_count_forgets_nothing(tmp_path, caplog, expected):
    _, privacy, recorders, _tokens = await _carried_boot(tmp_path, "stale.sqlite")
    try:
        before = _rows(privacy)
        with caplog.at_level(logging.WARNING, logger=capture_recorders.__name__):
            with pytest.raises(ValueError, match="^capture_carried_count_conflict$"):
                await recorders.forget_carried(expected, authorize=_allowed)
        assert _rows(privacy) == before and recorders.carried() == 2
        assert _forget_lines(caplog) == [] and not recorders._fault
    finally:
        await _close(privacy, recorders)


@pytest.mark.parametrize("expected", [True, -1, None, 2.0, "2"])
async def test_the_confirmation_must_be_a_non_negative_int(tmp_path, expected):
    _, privacy, recorders, _tokens = await _carried_boot(tmp_path, "types.sqlite")
    try:
        with pytest.raises(ValueError, match="^capture_carried_count_conflict$"):
            await recorders.forget_carried(expected, authorize=_allowed)
        assert recorders.carried() == 2
    finally:
        await _close(privacy, recorders)


async def test_forget_is_idempotent_and_silent_when_nothing_is_carried(tmp_path, caplog):
    _, privacy, recorders, _tokens = await _carried_boot(tmp_path, "again.sqlite")
    try:
        assert await recorders.forget_carried(2, authorize=_allowed) == 2
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger=capture_recorders.__name__):
            # The same request repeated (a lost response) and a stale page.
            assert await recorders.forget_carried(2, authorize=_allowed) == 0
            assert await recorders.forget_carried(0, authorize=_allowed) == 0
        assert _forget_lines(caplog) == []
        assert recorders.pending() == recorders.carried() == 0
    finally:
        await _close(privacy, recorders)


async def test_owners_issued_here_are_never_forgotten(tmp_path):
    privacy, recorders = await _boot(str(tmp_path / "fresh.sqlite"))
    try:
        live = await _issue(recorders)
        reserved = (await recorders.register(actor()))["token"]
        assert await recorders.forget_carried(0, authorize=_allowed) == 0
        assert await recorders.forget_carried(2, authorize=_allowed) == 0
        assert set(_rows(privacy)) == {recorders._key(live), recorders._key(reserved)}
        await recorders.retire(live)
        await recorders.retire(reserved)
    finally:
        await _close(privacy, recorders)


async def test_a_reclaimed_owner_is_live_and_never_forgotten(tmp_path):
    _, privacy, recorders, (reclaimed, lost) = await _carried_boot(tmp_path, "reclaimed.sqlite")
    try:
        await recorders.poll(reclaimed, actor())
        assert recorders.carried() == 1 and recorders.live_owners() == 1
        # The page still showed 2: refused, nothing forgotten.
        with pytest.raises(ValueError, match="^capture_carried_count_conflict$"):
            await recorders.forget_carried(2, authorize=_allowed)
        assert len(_rows(privacy)) == 2
        assert await recorders.forget_carried(1, authorize=_allowed) == 1
        assert _rows(privacy) == {recorders._key(reclaimed): "unknown"}
        assert recorders.pending_live() == recorders.pending() == 1
        await recorders.retire(reclaimed)
        await asyncio.wait_for(recorders.drain(), 2)
        assert recorders.pending() == 0
    finally:
        await _close(privacy, recorders)


async def _held_forget(recorders, expected, during):
    """Run forget_carried, calling ``during`` after its rows are deleted, before commit."""
    entered, release = threading.Event(), threading.Event()
    calls = []

    def authorize() -> None:
        calls.append(None)
        if len(calls) == 3:  # The first guard after the deletes.
            entered.set()
            assert release.wait(5)

    forgetting = asyncio.create_task(recorders.forget_carried(expected, authorize=authorize))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        await during()
    finally:
        release.set()
    return await asyncio.gather(forgetting, return_exceptions=True)


async def test_a_holder_presenting_its_token_before_commit_aborts_the_forget(tmp_path):
    _, privacy, recorders, (holder, lost) = await _carried_boot(tmp_path, "race.sqlite")
    try:
        before = _rows(privacy)

        async def contact():
            status = await recorders.poll(holder, actor())
            assert status["retirement_required"] is True

        (result,) = await _held_forget(recorders, 2, contact)
        assert isinstance(result, ValueError) and str(result) == "capture_carried_count_conflict"
        # Rolled back: both rows remain; the holder is live and awaited.
        assert _rows(privacy) == before and not recorders._fault
        assert recorders.carried() == 1 and recorders.live_owners() == 1
        assert recorders._forgetting == frozenset() and recorders._forget_contested is False
        assert await recorders.forget_carried(1, authorize=_allowed) == 1
        assert set(_rows(privacy)) == {recorders._key(holder)}
        await recorders.retire(holder)
    finally:
        await _close(privacy, recorders)


async def test_a_contact_by_an_owner_not_being_forgotten_does_not_abort(tmp_path):
    path = str(tmp_path / "other.sqlite")
    privacy, recorders = await _boot(path)
    try:
        lost = await _issue(recorders)
    finally:
        await _close(privacy, recorders)
    privacy, recorders = await _boot(path)
    try:
        live = await _issue(recorders)

        async def contact():
            await recorders.poll(live, actor())

        (result,) = await _held_forget(recorders, 1, contact)
        assert result == 1
        assert set(_rows(privacy)) == {recorders._key(live)}
        await recorders.retire(live)
        assert recorders._key(lost) not in _rows(privacy)
    finally:
        await _close(privacy, recorders)


async def test_authority_lapsing_inside_the_transaction_rolls_back(tmp_path):
    _, privacy, recorders, _tokens = await _carried_boot(tmp_path, "lapse.sqlite")
    try:
        before = _rows(privacy)
        calls = []

        def authorize():
            calls.append(None)
            if len(calls) >= 3:
                raise PermissionError("capture_control_denied")

        with pytest.raises(PermissionError):
            await recorders.forget_carried(2, authorize=authorize)
        assert _rows(privacy) == before and recorders.carried() == 2
        assert not recorders._fault
        await asyncio.wait_for(recorders.drain(), 2)
    finally:
        await _close(privacy, recorders)


async def test_a_storage_failure_faults_the_recorders(tmp_path, monkeypatch):
    _, privacy, recorders, _tokens = await _carried_boot(tmp_path, "fault.sqlite")
    try:
        def failing(*args, **kwargs):
            raise OSError("synthetic ledger failure")

        with monkeypatch.context() as patch:
            patch.setattr(privacy.ledger, "_transaction", failing)
            with pytest.raises(OSError):
                await recorders.forget_carried(2, authorize=_allowed)
        assert recorders._fault
        with pytest.raises(RuntimeError, match="^capture_recorders_unavailable$"):
            await recorders.forget_carried(2, authorize=_allowed)
    finally:
        with pytest.raises(RuntimeError, match="^capture_recorder_state_unconfirmed$"):
            await asyncio.wait_for(recorders.aclose(), 2)
        await privacy.aclose()


async def test_forget_is_refused_after_close(tmp_path):
    _, privacy, recorders, _tokens = await _carried_boot(tmp_path, "closed.sqlite")
    await _close(privacy, recorders)
    with pytest.raises(RuntimeError, match="^capture_recorders_unavailable$"):
        await recorders.forget_carried(2, authorize=_allowed)


async def test_forget_is_refused_before_initialization(tmp_path):
    privacy = PrivacyServices(str(tmp_path / "early.sqlite"))
    await privacy.acall(privacy.initialize)
    recorders = CaptureRecorders(privacy, capture_allowed=lambda: True)
    try:
        with pytest.raises(RuntimeError, match="^capture_recorders_unavailable$"):
            await recorders.forget_carried(0, authorize=_allowed)
    finally:
        await privacy.aclose()
