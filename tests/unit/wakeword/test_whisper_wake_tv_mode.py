"""Tests for WhisperWakeDetector TV-mode rate limiter (Fix C).

Verifica que:
1. Un solo reject NO activa TV-mode.
2. ≥ TV_MODE_ENTRY_REJECTS rejects en la ventana → TV-mode activo.
3. Rejects fuera de la ventana se purgan (no acumulan).
4. TV-mode expira tras TV_MODE_DURATION_S sin nuevos rejects.
5. Cada reject dentro de la ventana extiende TV-mode.
"""
from __future__ import annotations

import sys
import time
from unittest.mock import MagicMock

import pytest

sys.modules.setdefault("torch", MagicMock())
sys.modules.setdefault("torch.cuda", MagicMock())

from src.wakeword.whisper_wake import (
    WhisperWakeDetector,
    TV_MODE_DURATION_S,
    TV_MODE_ENTRY_REJECTS,
    TV_MODE_ENTRY_WINDOW_S,
)


def _make_detector() -> WhisperWakeDetector:
    det = WhisperWakeDetector(
        whisper_stt=MagicMock(),
        wake_words=["nexa"],
    )
    det._loaded = True
    det._vad = None
    return det


class TestRejectAccumulation:
    def test_single_reject_no_tv_mode(self):
        det = _make_detector()
        det._record_reject("multi_wake_hallucination")
        assert det._is_tv_mode_active() is False

    def test_threshold_minus_one_no_tv_mode(self):
        det = _make_detector()
        for _ in range(TV_MODE_ENTRY_REJECTS - 1):
            det._record_reject("tv_stop_phrase")
        assert det._is_tv_mode_active() is False

    def test_threshold_reached_activates_tv_mode(self):
        det = _make_detector()
        for _ in range(TV_MODE_ENTRY_REJECTS):
            det._record_reject("tv_stop_phrase")
        assert det._is_tv_mode_active() is True


class TestWindowPurging:
    def test_old_rejects_outside_window_dont_count(self, monkeypatch):
        det = _make_detector()
        # Simular rejects viejos manualmente.
        old_ts = time.time() - TV_MODE_ENTRY_WINDOW_S - 10.0
        for _ in range(TV_MODE_ENTRY_REJECTS):
            det._reject_timestamps.append(old_ts)
        # Un reject nuevo purga los viejos.
        det._record_reject("multi_wake_hallucination")
        # Solo queda 1 reject vivo → no activa TV-mode.
        assert det._is_tv_mode_active() is False
        assert len(det._reject_timestamps) == 1


class TestTvModeExpiration:
    def test_tv_mode_expires_after_duration(self, monkeypatch):
        det = _make_detector()
        for _ in range(TV_MODE_ENTRY_REJECTS):
            det._record_reject("tv_stop_phrase")
        assert det._is_tv_mode_active() is True

        # Simular que pasó TV_MODE_DURATION_S + 1.
        future = time.time() + TV_MODE_DURATION_S + 1.0
        monkeypatch.setattr(time, "time", lambda: future)
        assert det._is_tv_mode_active() is False

    def test_new_reject_extends_tv_mode(self, monkeypatch):
        """Un reject dentro de la ventana renueva el `_tv_mode_until`."""
        det = _make_detector()
        for _ in range(TV_MODE_ENTRY_REJECTS):
            det._record_reject("tv_stop_phrase")
        first_until = det._tv_mode_until

        # Avanzar 100s (dentro del rolling window) y agregar otro reject.
        future = time.time() + 100.0
        monkeypatch.setattr(time, "time", lambda: future)
        det._record_reject("multi_wake_hallucination")
        # Nuevo until > anterior.
        assert det._tv_mode_until > first_until


class TestEmitWakeIntegration:
    def test_emit_reject_records_rejection(self):
        """Llamar `_emit_wake(matched=False)` con `rejection_reason` cuenta como reject."""
        det = _make_detector()
        for _ in range(TV_MODE_ENTRY_REJECTS):
            det._emit_wake(
                matched=False, wake_word=None, matched_via="rejected",
                text="ruido", dur_ms=500, stt_ms=100,
                rejection_reason="tv_stop_phrase",
            )
        assert det._is_tv_mode_active() is True

    def test_emit_accept_does_not_record_reject(self):
        det = _make_detector()
        for _ in range(TV_MODE_ENTRY_REJECTS + 2):
            det._emit_wake(
                matched=True, wake_word="nexa", matched_via="exact",
                text="nexa prende la luz", dur_ms=1000, stt_ms=200,
            )
        assert det._is_tv_mode_active() is False
        assert len(det._reject_timestamps) == 0

    def test_emit_reject_without_reason_does_not_count(self):
        """Reject sin `rejection_reason` no cuenta — defensivo."""
        det = _make_detector()
        for _ in range(TV_MODE_ENTRY_REJECTS + 2):
            det._emit_wake(
                matched=False, wake_word=None, matched_via="rejected",
                text="x", dur_ms=200, stt_ms=50,
                rejection_reason=None,
            )
        assert det._is_tv_mode_active() is False


class TestMetricsEmitterTvModeFlag:
    def test_emitter_receives_tv_mode_kwarg(self):
        """El metrics_emitter debe recibir `tv_mode` en `emit_wake`."""
        emitter = MagicMock()
        det = _make_detector()
        det.metrics_emitter = emitter

        # Activar TV-mode.
        for _ in range(TV_MODE_ENTRY_REJECTS):
            det._record_reject("tv_stop_phrase")

        # Aceptar wake en TV-mode.
        det._emit_wake(
            matched=True, wake_word="nexa", matched_via="exact",
            text="nexa prende", dur_ms=900, stt_ms=180,
        )

        # Última call debe llevar tv_mode=True.
        kwargs = emitter.emit_wake.call_args.kwargs
        assert kwargs.get("tv_mode") is True

    def test_emitter_without_tv_mode_support_falls_back(self):
        """Si el emitter no acepta `tv_mode`, hace fallback sin el flag."""
        # Emitter que falla con tv_mode kwarg pero acepta sin él.
        def emit_old_signature(*, room_id, matched, wake_word, matched_via, text,
                                audio_duration_ms, wake_stt_ms, fuzzy_ratio,
                                rejection_reason):
            return None

        emitter = MagicMock()
        # Configurar side_effect: primer call (con tv_mode) raise TypeError,
        # segundo call (sin tv_mode) ok.
        emitter.emit_wake.side_effect = [TypeError("unexpected kwarg 'tv_mode'"), None]

        det = _make_detector()
        det.metrics_emitter = emitter
        det._emit_wake(
            matched=True, wake_word="nexa", matched_via="exact",
            text="nexa prende", dur_ms=900, stt_ms=180,
        )
        assert emitter.emit_wake.call_count == 2
        # Segunda llamada (fallback) NO debe tener tv_mode.
        second_kwargs = emitter.emit_wake.call_args_list[1].kwargs
        assert "tv_mode" not in second_kwargs
