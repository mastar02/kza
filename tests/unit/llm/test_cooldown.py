"""Tests for CooldownManager (in-memory behavior, sin persistence)."""

import time
import pytest
from src.llm.cooldown import CooldownManager, BACKOFF_SCHEDULE_S
from src.llm.types import ErrorKind


class TestCooldownManagerInMemory:
    def test_initially_available(self, tmp_path):
        mgr = CooldownManager(persistence_path=tmp_path / "cd.json")
        assert mgr.is_available("primary") is True

    def test_record_success_clears_state(self, tmp_path):
        mgr = CooldownManager(persistence_path=tmp_path / "cd.json")
        mgr.record_failure("primary", ErrorKind.RATE_LIMIT)
        assert mgr.is_available("primary") is False
        mgr.record_success("primary")
        assert mgr.is_available("primary") is True
        assert mgr.get_state("primary").error_count == 0

    def test_backoff_schedule_progresses(self, tmp_path, monkeypatch):
        mgr = CooldownManager(persistence_path=tmp_path / "cd.json")
        now = [1000.0]
        monkeypatch.setattr("src.llm.cooldown.time.time", lambda: now[0])

        # 1ra falla → 60s
        mgr.record_failure("primary", ErrorKind.RATE_LIMIT)
        assert mgr.get_state("primary").cooldown_until == pytest.approx(1060.0)

        # avanzar past cooldown
        now[0] = 1100.0
        assert mgr.is_available("primary") is True

        # 2da falla → 300s (5min)
        mgr.record_failure("primary", ErrorKind.RATE_LIMIT)
        assert mgr.get_state("primary").cooldown_until == pytest.approx(1400.0)

        # 3ra falla → 1500s (25min)
        now[0] = 1500.0
        mgr.record_failure("primary", ErrorKind.RATE_LIMIT)
        assert mgr.get_state("primary").cooldown_until == pytest.approx(1500.0 + 1500.0)

        # 4ta falla → 3600s (1h, cap)
        now[0] = 5000.0
        mgr.record_failure("primary", ErrorKind.RATE_LIMIT)
        assert mgr.get_state("primary").cooldown_until == pytest.approx(5000.0 + 3600.0)

        # 5ta falla → sigue 3600s (cap)
        now[0] = 10000.0
        mgr.record_failure("primary", ErrorKind.RATE_LIMIT)
        assert mgr.get_state("primary").cooldown_until == pytest.approx(10000.0 + 3600.0)

    def test_billing_uses_long_backoff(self, tmp_path, monkeypatch):
        """Billing → cooldown directo de 1h (no progresión gradual)."""
        mgr = CooldownManager(persistence_path=tmp_path / "cd.json")
        now = [1000.0]
        monkeypatch.setattr("src.llm.cooldown.time.time", lambda: now[0])

        mgr.record_failure("primary", ErrorKind.BILLING)
        assert mgr.get_state("primary").cooldown_until == pytest.approx(1000.0 + 3600.0)

    def test_is_available_respects_now(self, tmp_path, monkeypatch):
        mgr = CooldownManager(persistence_path=tmp_path / "cd.json")
        now = [1000.0]
        monkeypatch.setattr("src.llm.cooldown.time.time", lambda: now[0])

        mgr.record_failure("primary", ErrorKind.RATE_LIMIT)
        assert mgr.is_available("primary") is False

        # apenas antes del expiry
        now[0] = 1059.9
        assert mgr.is_available("primary") is False

        # justo después
        now[0] = 1060.1
        assert mgr.is_available("primary") is True

    def test_next_attempt_at(self, tmp_path, monkeypatch):
        mgr = CooldownManager(persistence_path=tmp_path / "cd.json")
        now = [1000.0]
        monkeypatch.setattr("src.llm.cooldown.time.time", lambda: now[0])

        assert mgr.next_attempt_at("primary") == 0.0  # available now
        mgr.record_failure("primary", ErrorKind.RATE_LIMIT)
        assert mgr.next_attempt_at("primary") == pytest.approx(1060.0)

    def test_backoff_schedule_constants(self):
        # Verifica que los valores documentados no se alteren accidentalmente
        assert BACKOFF_SCHEDULE_S == (60, 300, 1500, 3600)
