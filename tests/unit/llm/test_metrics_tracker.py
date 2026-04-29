"""Tests para LLMMetricsTracker (rolling window de tok/s)."""

import time

import pytest

from src.llm.metrics import LLMMetricsTracker


def test_empty_returns_none():
    t = LLMMetricsTracker()
    assert t.snapshot("nonexistent") is None


def test_records_and_computes_tps():
    t = LLMMetricsTracker()
    # 100 tokens en 1000ms = 100 tps
    t.record("ep1", tokens=100, elapsed_ms=1000)
    snap = t.snapshot("ep1")
    assert snap["tps"] == 100.0
    assert snap["calls"] == 1
    assert snap["ttft_ms"] == 1000


def test_aggregates_multiple_calls():
    t = LLMMetricsTracker()
    t.record("ep1", 50, 500)
    t.record("ep1", 100, 1000)
    t.record("ep1", 30, 300)
    snap = t.snapshot("ep1")
    # 180 tokens / 1.8s = 100 tps
    assert snap["tps"] == 100.0
    assert snap["calls"] == 3


def test_ignores_zero_elapsed():
    t = LLMMetricsTracker()
    t.record("ep1", tokens=100, elapsed_ms=0)
    assert t.snapshot("ep1") is None


def test_window_evicts_old_entries(monkeypatch):
    t = LLMMetricsTracker(window_s=0.5)
    base = 1_000_000.0  # constante, evita drift del wall clock real
    cur = [base]
    monkeypatch.setattr(time, "time", lambda: cur[0])
    t.record("ep1", 10, 100)
    cur[0] = base + 0.1
    t.record("ep1", 20, 200)
    # Avanzamos 1.0s — todas las entries quedaron fuera de la ventana de 0.5s
    cur[0] = base + 1.0
    assert t.snapshot("ep1") is None


def test_per_endpoint_isolation():
    t = LLMMetricsTracker()
    t.record("a", 100, 1000)
    t.record("b", 50, 500)
    sa = t.snapshot("a")
    sb = t.snapshot("b")
    assert sa["calls"] == 1 and sa["tps"] == 100.0
    assert sb["calls"] == 1 and sb["tps"] == 100.0


def test_reset_clears_endpoint():
    t = LLMMetricsTracker()
    t.record("ep1", 100, 1000)
    t.reset("ep1")
    assert t.snapshot("ep1") is None


def test_reset_all():
    t = LLMMetricsTracker()
    t.record("a", 1, 100)
    t.record("b", 1, 100)
    t.reset()
    assert t.snapshot("a") is None
    assert t.snapshot("b") is None


def test_max_per_endpoint_caps_memory():
    t = LLMMetricsTracker(max_per_endpoint=3)
    for i in range(10):
        t.record("ep", 10, 100)
    snap = t.snapshot("ep")
    assert snap["calls"] == 3
