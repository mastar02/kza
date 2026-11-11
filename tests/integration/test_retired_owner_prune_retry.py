"""U3b-1b F2 / U3b-2 review R1, R6: the retired-owner prune never reports a silent success.

Real DashboardAuthority, privacy services and FactLedger (the U3b-1b graph).
An authority that cannot answer is 'unknown': the boot prune defers, the
maintenance tick keeps retrying while pending and the operator health signal
stays on. The tick also runs the prune hourly without any pending flag.
"""

import ast
import logging
from pathlib import Path
import sqlite3

import pytest

from src.main import _prune_retired_memory_owners
from src.memory import privacy_services as privacy_module
from tests.integration.test_fact_owner_retirement import assign, confirmed, owner_of, owners as _owners
from tests.integration.test_voice_person_api import people_web as _people_web
from tests.integration.test_voice_person_links import people_graph as _people_graph

people_graph = _people_graph
people_web = _people_web
owners = _owners
ROOT = Path(__file__).resolve().parents[2]


async def _retired_owner(g, content="Synthetic fact of A"):
    fact = await confirmed(g, content)
    await assign(g, fact, g.a)
    await g.authority.retire_user(g.root, g.a)
    return fact


async def test_reviewer_probe_unusable_authority_defers_instead_of_silent_success(owners):
    """The review R1 probe: before the fix both assertions on 'pending' failed."""
    g = owners
    fact = await _retired_owner(g)
    g.authority._failed = True  # unusable: users_retired answers False, memory_owner_retired None
    assert await _prune_retired_memory_owners(g.privacy, g.authority) == 0
    assert g.ledger.retired_owner_prune_pending is True and owner_of(g, fact.id) == g.a
    # A retry while still unusable keeps the signal (it used to clear it).
    await g.privacy._retry_retired_owner_prune()
    assert g.ledger.retired_owner_prune_pending is True and owner_of(g, fact.id) == g.a
    # Recovery: the next tick deletes the retired person's memories and clears the signal.
    g.authority._failed = False
    await g.privacy._retry_retired_owner_prune()
    assert g.ledger.retired_owner_prune_pending is False and owner_of(g, fact.id) is None
    assert g.ledger.owner_refs_cleared(frozenset({g.a}))


async def test_proven_retirees_are_deleted_even_when_another_answer_is_unknown(owners, monkeypatch):
    g = owners
    gone = await _retired_owner(g, "Synthetic fact of retired A")
    kept = await confirmed(g, "Synthetic fact of B")
    await assign(g, kept, g.b)
    real = g.authority.memory_owner_retired
    monkeypatch.setattr(g.authority, "memory_owner_retired",
                        lambda ids: None if ids == frozenset({g.b}) else real(ids))
    with pytest.raises(PermissionError, match="memory_owner_authority_unavailable"):
        await g.privacy.acall(g.ledger.prune_retired_owners, g.authority.users_retired,
                              probe=g.authority.memory_owner_retired)
    assert owner_of(g, gone.id) is None and owner_of(g, kept.id) == g.b


async def test_a_raising_probe_counts_as_unknown(owners, monkeypatch):
    g = owners
    await _retired_owner(g)
    other = await confirmed(g, "Synthetic fact of B")
    await assign(g, other, g.b)

    def broken(ids):
        raise RuntimeError("synthetic")
    with pytest.raises(PermissionError, match="memory_owner_authority_unavailable"):
        await g.privacy.acall(g.ledger.prune_retired_owners, g.authority.users_retired, probe=broken)
    assert owner_of(g, other.id) == g.b


async def test_failed_boot_prune_is_flagged_and_completed_by_the_maintenance_tick(owners, monkeypatch, caplog):
    g = owners
    fact = await _retired_owner(g)
    assert g.ledger.retired_owner_prune_pending is False
    real = g.ledger.prune_retired_owners

    def broken(retired, *, probe=None):
        raise sqlite3.OperationalError("synthetic private detail")

    monkeypatch.setattr(g.ledger, "prune_retired_owners", broken)
    assert await _prune_retired_memory_owners(g.privacy, g.authority) == 0
    assert g.ledger.retired_owner_prune_pending is True and owner_of(g, fact.id) == g.a
    await g.privacy._retry_retired_owner_prune()
    assert g.ledger.retired_owner_prune_pending is True and owner_of(g, fact.id) == g.a
    assert "synthetic private detail" not in caplog.text
    monkeypatch.setattr(g.ledger, "prune_retired_owners", real)
    await g.privacy._retry_retired_owner_prune()
    assert g.ledger.retired_owner_prune_pending is False and owner_of(g, fact.id) is None
    # Nothing pending and the interval not elapsed: the tick does no owner work.
    monkeypatch.setattr(g.ledger, "prune_retired_owners", broken)
    await g.privacy._retry_retired_owner_prune()
    assert g.ledger.retired_owner_prune_pending is False


async def test_successful_boot_prune_leaves_no_pending_signal(owners):
    g = owners
    fact = await _retired_owner(g)
    assert await _prune_retired_memory_owners(g.privacy, g.authority) == 1
    assert g.ledger.retired_owner_prune_pending is False and owner_of(g, fact.id) is None


async def test_hourly_prune_runs_without_a_pending_flag(owners, monkeypatch):
    """Rows that reappear later (e.g. a restore) self-heal within one interval."""
    g = owners
    clock = [1000.0]
    monkeypatch.setattr(privacy_module.time, "monotonic", lambda: clock[0])
    assert await _prune_retired_memory_owners(g.privacy, g.authority) == 0
    fact = await _retired_owner(g, "Synthetic fact restored after boot")
    clock[0] += privacy_module.RETIRED_OWNER_PRUNE_INTERVAL_S - 1
    await g.privacy._retry_retired_owner_prune()
    assert owner_of(g, fact.id) == g.a  # not due yet
    clock[0] += 1
    await g.privacy._retry_retired_owner_prune()
    assert owner_of(g, fact.id) is None and g.ledger.retired_owner_prune_pending is False


async def test_hourly_prune_failure_raises_the_signal(owners, monkeypatch):
    g = owners
    clock = [1000.0]
    monkeypatch.setattr(privacy_module.time, "monotonic", lambda: clock[0])
    assert await _prune_retired_memory_owners(g.privacy, g.authority) == 0
    await _retired_owner(g)
    g.authority._failed = True
    clock[0] += privacy_module.RETIRED_OWNER_PRUNE_INTERVAL_S
    await g.privacy._retry_retired_owner_prune()
    assert g.ledger.retired_owner_prune_pending is True


async def test_failure_warnings_are_rate_limited(owners, monkeypatch, caplog):
    g = owners
    clock = [1000.0]
    monkeypatch.setattr(privacy_module.time, "monotonic", lambda: clock[0])
    await _retired_owner(g)
    g.authority._failed = True
    assert await _prune_retired_memory_owners(g.privacy, g.authority) == 0
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=privacy_module.logger.name):
        for _ in range(20):  # 20 ticks of 30 s = 10 minutes - 30 s of failure
            await g.privacy._retry_retired_owner_prune()
            clock[0] += 30
        first = [r for r in caplog.records if "Retired-owner prune pending" in r.getMessage()]
        await g.privacy._retry_retired_owner_prune()  # 10 minutes after the first warning
        second = [r for r in caplog.records if "Retired-owner prune pending" in r.getMessage()]
    assert (len(first), len(second)) == (1, 2)


def test_the_maintenance_tick_runs_the_prune_before_expiry_and_boot_wires_the_probe():
    tree = ast.parse((ROOT / "src/memory/privacy_services.py").read_text(encoding="utf-8"))
    start = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "start")
    maintain = next(n for n in ast.walk(start) if isinstance(n, ast.AsyncFunctionDef) and n.name == "maintain")
    awaited = [ast.unparse(n.value) for n in sorted((n for n in ast.walk(maintain) if isinstance(n, ast.Await)),
                                                     key=lambda n: (n.lineno, n.col_offset))]
    assert awaited[:2] == ["self._retry_retired_owner_prune()", "self.acall(self.ledger.expire)"]
    main = ast.parse((ROOT / "src/main.py").read_text(encoding="utf-8"))
    prune = next(n for n in ast.walk(main) if isinstance(n, ast.AsyncFunctionDef)
                 and n.name == "_prune_retired_memory_owners")
    text = ast.unparse(prune)
    assert ("privacy_services.attach_retired_owner_prune(dashboard_authority.users_retired, "
            "dashboard_authority.memory_owner_retired)") in text
    assert "probe=dashboard_authority.memory_owner_retired" in text
    handler = next(n for n in ast.walk(prune) if isinstance(n, ast.ExceptHandler))
    assert "privacy_services.defer_retired_owner_prune()" in ast.unparse(handler)
