"""Tests: preflight de VRAM al arranque (aviso pre-OOM, 2026-05-29)."""

import sys
import logging
from unittest.mock import MagicMock

from src.main import _preflight_vram


def _fake_torch(available, devices_free_mib):
    t = MagicMock()
    t.cuda.is_available.return_value = available
    t.cuda.device_count.return_value = len(devices_free_mib)

    def mem_get_info(i):
        return (int(devices_free_mib[i] * 1024 * 1024), 8 * 1024 * 1024 * 1024)

    t.cuda.mem_get_info.side_effect = mem_get_info
    return t


def test_warns_when_vram_tight(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(True, [900, 4000]))
    with caplog.at_level(logging.WARNING):
        _preflight_vram(threshold_mib=1500)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("cuda:0" in r.message and "900" in r.message for r in warnings), (
        "debe avisar por cuda:0 (900 MiB < 1500)"
    )
    # cuda:1 con 4000 MiB no debe disparar warning
    assert not any("cuda:1" in r.message and r.levelno == logging.WARNING for r in caplog.records)


def test_noop_when_cuda_unavailable(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(False, [100]))
    with caplog.at_level(logging.WARNING):
        _preflight_vram(threshold_mib=1500)
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test_noop_when_torch_missing(monkeypatch):
    # Simula ausencia de torch (dev macOS): no debe lanzar.
    monkeypatch.setitem(sys.modules, "torch", None)
    _preflight_vram(threshold_mib=1500)  # import torch → TypeError/ImportError → swallowed
