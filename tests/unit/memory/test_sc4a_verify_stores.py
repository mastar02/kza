"""SC-4a skip list on the stores (sc-release-design v4 v_verification_run.skip_list).

Each verify-mode pin is tested against the same store in a normal boot, so a
test that passes in verify mode would fail if the repair still ran:
d-fact-owner-schema-prune, c-index-target-registration, b-erasure-recover,
a-privacy-maintenance, d-recorders-initialize, capture-close (R5V-1) and
d-reminder-sending. Outside verify mode nothing changes: drain still deletes
reserved rows and waits for this process's acknowledgements.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from uuid import uuid4

import pytest

from src.memory.capture_recorders import CaptureRecorders
from src.memory.fact_ledger import FactLedger, MemoryAssigner, MemoryOrigin, MemorySource
from src.memory.privacy_services import PrivacyServices
from src.reminders.reminder_store import ReminderStore
from tests.integration.test_global_capture_control import actor

pytestmark = pytest.mark.asyncio


def _source() -> MemorySource:
    now = time.time()
    return MemorySource(str(uuid4()), str(uuid4()), now - 1, now + 86400, MemoryOrigin.VOICE, ())


def _owned_deleted_fact(path: str) -> str:
    ledger = FactLedger(path)
    ledger.initialize()
    try:
        fact = ledger.store("Synthetic owned fact later deleted", "fact", .9, (_source(),))
        ledger.assign_owner(fact.id, "user_" + "b" * 32, fact.revision, 0, MemoryAssigner("synthetic-operator"),
                            retired=lambda ids: False, active=lambda person_id: True)
    finally:
        ledger.close()
    with sqlite3.connect(path) as db:
        db.execute("UPDATE facts SET state='deleted', content=NULL WHERE id=?", (fact.id,))
    return fact.id


def _owner_rows(path: str, fact_id: str) -> int:
    with sqlite3.connect(path) as db:
        return sum(db.execute(f"SELECT count(*) FROM {table} WHERE fact_id=?", (fact_id,)).fetchone()[0]
                   for table in ("fact_owners", "fact_owner_revisions"))


async def test_fact_owner_prune_at_open_is_skipped_only_in_verify_mode(tmp_path):
    path = str(tmp_path / "privacy.db")
    fact_id = _owned_deleted_fact(path)
    assert _owner_rows(path, fact_id) == 2
    verify = FactLedger(path, verify_only=True)
    verify.initialize()
    verify.close()
    assert _owner_rows(path, fact_id) == 2  # validated, not pruned
    normal = FactLedger(path)
    normal.initialize()
    normal.close()
    assert _owner_rows(path, fact_id) == 0


def _index_rows(path: str) -> int:
    with sqlite3.connect(path) as db:
        return db.execute("SELECT count(*) FROM policy_index_deletions").fetchone()[0]


async def test_register_target_inserts_nothing_in_verify_mode(tmp_path):
    path = str(tmp_path / "privacy.db")
    ledger = FactLedger(path)
    ledger.initialize()
    try:
        fact = ledger.store("Synthetic fact deleted early", "fact", .9, (_source(),))
    finally:
        ledger.close()
    with sqlite3.connect(path) as db:
        db.execute("UPDATE facts SET state='deleted', content=NULL, revision=revision+1 WHERE id=?", (fact.id,))
        db.execute("DELETE FROM policy_index_deletions")
    for verify_only, expected in ((True, 0), (False, 1)):
        ledger = FactLedger(path, verify_only=verify_only)
        ledger.initialize()
        try:
            with ledger._transaction() as db:
                ledger.index_deletions.register_target(db, "synthetic-target")
        finally:
            ledger.close()
        assert _index_rows(path) == expected


async def test_privacy_services_verify_mode_runs_no_recover_and_no_maintenance(tmp_path, monkeypatch):
    calls = []
    for verify_only in (True, False):
        services = PrivacyServices(str(tmp_path / f"privacy-{verify_only}.db"), verify_only=verify_only)
        monkeypatch.setattr(services.ledger.erasures, "recover", lambda: calls.append(("erasures", verify_only)))
        monkeypatch.setattr(services.ledger.index_deletions, "recover",
                            lambda: calls.append(("index", verify_only)))
        try:
            await services.acall(services.initialize)
            services.start()
            assert (services.maintenance is None) is verify_only
            assert services.ledger.verify_only is verify_only
        finally:
            await services.aclose()
    assert calls == [("erasures", False), ("index", False)]


# --- capture_close (R5V-1) ---------------------------------------------------------------------------

def _initialized(path: str) -> None:
    ledger = FactLedger(path)
    ledger.initialize()
    ledger.close()


def _plant_recorders(path: str, states: list[str]) -> set:
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE IF NOT EXISTS capture_browser_recorders (token_hash TEXT PRIMARY KEY, "
                   "state TEXT NOT NULL, kind TEXT NOT NULL DEFAULT 'browser')")
        for state in states:
            db.execute("INSERT INTO capture_browser_recorders VALUES (?,?,'browser')", (uuid4().hex * 2, state))
    return _rows(path)


def _rows(path: str) -> set:
    with sqlite3.connect(path) as db:
        return set(db.execute("SELECT token_hash, state, kind FROM capture_browser_recorders"))


async def _boot(path: str, verify_only: bool) -> tuple[PrivacyServices, CaptureRecorders]:
    privacy = PrivacyServices(path, verify_only=verify_only)
    await privacy.acall(privacy.initialize)
    recorders = CaptureRecorders(privacy, capture_allowed=lambda: True, verify_only=verify_only)
    await recorders.initialize()
    return privacy, recorders


async def test_verify_capture_close_carries_every_row_unchanged_without_waiting(tmp_path):
    path = str(tmp_path / "privacy.db")
    _initialized(path)
    rows = _plant_recorders(path, ["reserved", "active", "unknown"])
    privacy, recorders = await _boot(path, True)
    try:
        assert _rows(path) == rows  # d-recorders-initialize: 'active' stays 'active'
        assert recorders.pending_live() == 2  # the active and the reserved row would hold a normal drain
        await asyncio.wait_for(recorders.drain(), 2)  # verify drain: no _retire_reserved, no wait
        assert _rows(path) == rows
        assert await recorders.confirm_unchanged() is True
        await recorders.aclose()
        assert recorders.unchanged_confirmed() is True and recorders.carried_rows() == 3
    finally:
        await privacy.aclose()
    assert _rows(path) == rows


async def test_verify_capture_close_refuses_a_changed_recorder_row(tmp_path):
    path = str(tmp_path / "privacy.db")
    _initialized(path)
    _plant_recorders(path, ["active", "reserved"])
    privacy, recorders = await _boot(path, True)
    try:
        await asyncio.wait_for(recorders.drain(), 2)
        with privacy.ledger._transaction() as db:  # a writer the skip list missed
            db.execute("UPDATE capture_browser_recorders SET state='unknown' WHERE state='active'")
        assert await recorders.confirm_unchanged() is False
        await recorders.aclose()
        assert recorders.unchanged_confirmed() is False
    finally:
        await privacy.aclose()


async def test_normal_drain_still_turns_active_unknown_deletes_reserved_and_waits(tmp_path):
    path = str(tmp_path / "privacy.db")
    _initialized(path)
    _plant_recorders(path, ["active", "reserved"])
    privacy, recorders = await _boot(path, False)
    try:
        assert {state for _key, state, _kind in _rows(path)} == {"unknown", "reserved"}
        principal = actor()
        token = (await recorders.register(principal))["token"]
        await recorders.activate(token, principal)
        drain = asyncio.create_task(recorders.drain())
        await asyncio.sleep(0.2)
        assert not drain.done()  # waits for this process's live owner's acknowledgement
        assert {state for _key, state, _kind in _rows(path)} == {"unknown", "active"}  # reserved deleted
        with pytest.raises(RuntimeError, match="capture_recorders_not_verify_mode"):
            await recorders.confirm_unchanged()
        await recorders.retire(token)
        await asyncio.wait_for(drain, 2)
        await recorders.aclose()
    finally:
        await privacy.aclose()


# --- d-reminder-sending ------------------------------------------------------------------------------

async def test_reminder_sending_action_is_left_as_is_only_in_verify_mode(tmp_path):
    path = str(tmp_path / "reminders.db")
    store = ReminderStore(path)
    await store.initialize()
    await store.close()
    body = json.dumps({"domain": "light", "entity_id": "light.living", "service": "turn_on"},
                      sort_keys=True, separators=(",", ":"))
    with sqlite3.connect(path) as db:
        db.execute("INSERT INTO reminder_home_actions (request_id, body, status) VALUES (?,?,'sending')",
                   (str(uuid4()), body))

    def statuses():
        with sqlite3.connect(path) as db:
            return [row[0] for row in db.execute("SELECT status FROM reminder_home_actions")]

    for verify_only, expected in ((True, ["sending"]), (False, ["uncertain"])):
        store = ReminderStore(path, verify_only=verify_only)
        await store.initialize()
        await store.close()
        assert statuses() == expected
