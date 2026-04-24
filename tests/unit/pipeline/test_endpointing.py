"""
Tests para endpointing adaptativo (S5).

Cubre:
1. `_adaptive_endpoint_threshold` devuelve short_ms cuando el parser ya tiene
   intent+entity (ready_to_dispatch=True).
2. Devuelve medium_ms cuando no hay early_command (parser sin señal).
3. Devuelve medium_ms cuando el early_command existe pero no está ready.
4. Thresholds custom son respetados.
5. `WhisperWakeDetector._voice_prob` devuelve float en [0, 1].
6. `_is_speech` es wrapper de `_voice_prob >= vad_threshold`.

Los tests 1-4 requieren importar `multi_room_audio_loop`, que solo se puede
cargar con Python 3.10+ (usa `np.ndarray | None` en method signatures). En
Python 3.9 local esos tests fallan en collection; corren OK en el server
(Python 3.13). Los tests 5-6 (wakeword) funcionan en cualquier Python porque
`whisper_wake.py` usa `from __future__ import annotations`.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

# Mock system-level modules BEFORE any imports
sys.modules.setdefault('sounddevice', MagicMock())
sys.modules.setdefault('soundfile', MagicMock())
sys.modules.setdefault('pyaudio', MagicMock())
sys.modules.setdefault('torch', MagicMock())
sys.modules.setdefault('torch.cuda', MagicMock())

from src.nlu.command_grammar import PartialCommand
from src.wakeword.whisper_wake import WhisperWakeDetector, SAMPLE_RATE


# ============================================================
# Helpers — adaptive endpoint (requiere importar loop module)
# ============================================================

def _import_loop_module():
    """
    Import lazy para no romper collection en Python 3.9 (los tests del loop
    requieren 3.10+ por `np.ndarray | None` sin future annotations).
    """
    from src.pipeline.multi_room_audio_loop import (  # noqa: WPS433
        MultiRoomAudioLoop,
        RoomStream,
    )
    return MultiRoomAudioLoop, RoomStream


def _make_wake_detector():
    m = MagicMock()
    m.load = MagicMock()
    m.detect = MagicMock(return_value=None)
    m.get_active_models = MagicMock(return_value=["nexa"])
    return m


def _make_echo_suppressor():
    m = MagicMock()
    m.is_safe_to_listen = True
    m.should_process_audio = MagicMock(return_value=(True, "ok"))
    m.is_human_voice = MagicMock(return_value=True)
    m.config = MagicMock()
    m.config.post_speech_buffer_ms = 400
    return m


def _make_follow_up():
    m = MagicMock()
    m.is_active = False
    m.follow_up_window = 8.0
    m.start_conversation = MagicMock()
    return m


def _make_loop(**kwargs):
    MultiRoomAudioLoop, RoomStream = _import_loop_module()
    rooms = {
        "cocina": RoomStream(
            room_id="cocina",
            device_index=0,
            wake_detector=_make_wake_detector(),
            echo_suppressor=_make_echo_suppressor(),
        ),
    }
    defaults = dict(
        room_streams=rooms,
        follow_up=_make_follow_up(),
    )
    defaults.update(kwargs)
    return MultiRoomAudioLoop(**defaults), rooms["cocina"]


# ============================================================
# Tests — _adaptive_endpoint_threshold
# ============================================================


class TestAdaptiveEndpointThreshold:
    """Verifica el mapping early_command → threshold."""

    def test_adaptive_threshold_with_ready_command(self):
        """parser ready_to_dispatch (intent+entity) → short_ms."""
        pytest.importorskip(
            "src.pipeline.multi_room_audio_loop",
            reason="Requiere Python 3.10+ (np.ndarray | None)",
            exc_type=ImportError,
        )
        loop, rs = _make_loop()
        # intent + entity presentes → ready_to_dispatch=True
        rs.early_command = PartialCommand(intent="turn_off", entity="light")
        assert rs.early_command.ready_to_dispatch() is True
        assert loop._adaptive_endpoint_threshold(rs) == 150  # default short_ms

    def test_adaptive_threshold_without_ready(self):
        """Sin early_command → medium_ms (comportamiento clásico)."""
        pytest.importorskip(
            "src.pipeline.multi_room_audio_loop",
            reason="Requiere Python 3.10+ (np.ndarray | None)",
            exc_type=ImportError,
        )
        loop, rs = _make_loop()
        rs.early_command = None
        assert loop._adaptive_endpoint_threshold(rs) == 300  # default medium_ms

    def test_adaptive_threshold_command_not_ready(self):
        """early_command existe pero incompleto (solo intent, sin entity) → medium_ms."""
        pytest.importorskip(
            "src.pipeline.multi_room_audio_loop",
            reason="Requiere Python 3.10+ (np.ndarray | None)",
            exc_type=ImportError,
        )
        loop, rs = _make_loop()
        # solo intent, sin entity → ready_to_dispatch=False
        rs.early_command = PartialCommand(intent="turn_off", entity=None)
        assert rs.early_command.ready_to_dispatch() is False
        assert loop._adaptive_endpoint_threshold(rs) == 300

    def test_adaptive_threshold_respects_custom_values(self):
        """Los thresholds custom del constructor se usan, no los defaults."""
        pytest.importorskip(
            "src.pipeline.multi_room_audio_loop",
            reason="Requiere Python 3.10+ (np.ndarray | None)",
            exc_type=ImportError,
        )
        loop, rs = _make_loop(
            endpointing_short_ms=100,
            endpointing_medium_ms=250,
            endpointing_long_ms=800,
        )
        rs.early_command = PartialCommand(intent="turn_on", entity="light")
        assert loop._adaptive_endpoint_threshold(rs) == 100

        rs.early_command = None
        assert loop._adaptive_endpoint_threshold(rs) == 250

    def test_endpointing_enabled_flag_stored(self):
        """El flag endpointing_enabled se guarda en el constructor."""
        pytest.importorskip(
            "src.pipeline.multi_room_audio_loop",
            reason="Requiere Python 3.10+ (np.ndarray | None)",
            exc_type=ImportError,
        )
        loop, _ = _make_loop(endpointing_enabled=False)
        assert loop.endpointing_enabled is False

        loop2, _ = _make_loop(endpointing_enabled=True)
        assert loop2.endpointing_enabled is True


class TestAdaptiveEndpointLogic:
    """
    Verifica la LÓGICA del helper sin importar el módulo del loop. Usa un
    fake objeto que replica la interface mínima (atributos short/medium + el
    método). El objetivo es tener cobertura de la semántica en Python 3.9.
    """

    class _FakeLoop:
        """Duck-typed fake con el mismo método."""
        def __init__(self, short_ms=150, medium_ms=300):
            self.endpointing_short_ms = short_ms
            self.endpointing_medium_ms = medium_ms

        def _adaptive_endpoint_threshold(self, rs):
            if rs.early_command is not None and rs.early_command.ready_to_dispatch():
                return self.endpointing_short_ms
            return self.endpointing_medium_ms

    class _FakeRoomStream:
        def __init__(self, early_command=None):
            self.early_command = early_command

    def test_logic_ready_command_returns_short(self):
        """ready_to_dispatch=True → short_ms."""
        loop = self._FakeLoop(short_ms=150, medium_ms=300)
        rs = self._FakeRoomStream(
            early_command=PartialCommand(intent="turn_off", entity="light"),
        )
        assert loop._adaptive_endpoint_threshold(rs) == 150

    def test_logic_none_command_returns_medium(self):
        """early_command=None → medium_ms."""
        loop = self._FakeLoop(short_ms=150, medium_ms=300)
        rs = self._FakeRoomStream(early_command=None)
        assert loop._adaptive_endpoint_threshold(rs) == 300

    def test_logic_incomplete_command_returns_medium(self):
        """early_command sin entity → ready_to_dispatch=False → medium_ms."""
        loop = self._FakeLoop(short_ms=150, medium_ms=300)
        rs = self._FakeRoomStream(
            early_command=PartialCommand(intent="turn_off", entity=None),
        )
        assert rs.early_command.ready_to_dispatch() is False
        assert loop._adaptive_endpoint_threshold(rs) == 300

    def test_logic_custom_thresholds_honored(self):
        """Cambiar short/medium cambia el output."""
        loop = self._FakeLoop(short_ms=80, medium_ms=220)
        rs_ready = self._FakeRoomStream(
            early_command=PartialCommand(intent="turn_on", entity="light"),
        )
        rs_none = self._FakeRoomStream(early_command=None)
        assert loop._adaptive_endpoint_threshold(rs_ready) == 80
        assert loop._adaptive_endpoint_threshold(rs_none) == 220


# ============================================================
# Tests — _voice_prob / _is_speech (whisper_wake)
# ============================================================


def _make_detector(vad_threshold: float = 0.7) -> WhisperWakeDetector:
    det = WhisperWakeDetector(
        whisper_stt=MagicMock(),
        wake_words=["nexa"],
        vad_threshold=vad_threshold,
    )
    det._loaded = True
    det._vad = None  # fallback RMS (no torch needed)
    det._torch = None
    return det


def _silent_chunk(duration_s: float = 0.08) -> np.ndarray:
    """Chunk de silencio absoluto."""
    return np.zeros(int(duration_s * SAMPLE_RATE), dtype=np.float32)


def _loud_chunk(duration_s: float = 0.08, amp: float = 0.2) -> np.ndarray:
    """Chunk con energía > min_rms default (0.025)."""
    return (np.random.randn(int(duration_s * SAMPLE_RATE)) * amp).astype(np.float32)


class TestVoiceProb:
    """Verifica el refactor _voice_prob → float [0,1] + _is_speech wrapper."""

    def test_voice_prob_returns_float(self):
        """_voice_prob siempre devuelve float en rango [0, 1]."""
        det = _make_detector()
        for chunk in (_silent_chunk(), _loud_chunk(), _loud_chunk(amp=0.5)):
            prob = det._voice_prob(chunk)
            assert isinstance(prob, float)
            assert 0.0 <= prob <= 1.0

    def test_voice_prob_silence_returns_zero(self):
        """Chunk por debajo de min_rms → 0.0."""
        det = _make_detector()
        assert det._voice_prob(_silent_chunk()) == 0.0

    def test_voice_prob_loud_returns_above_zero(self):
        """Chunk con energía clara → prob > 0 (fallback RMS da 0.5)."""
        det = _make_detector()
        prob = det._voice_prob(_loud_chunk(amp=0.3))
        assert prob > 0.0

    def test_is_speech_uses_voice_prob(self):
        """_is_speech es wrapper de _voice_prob >= vad_threshold."""
        det = _make_detector(vad_threshold=0.6)
        # Silent → prob=0 < 0.6 → False
        assert det._is_speech(_silent_chunk()) is False
        # Loud (fallback RMS prob=0.5) < 0.6 → False
        assert det._is_speech(_loud_chunk(amp=0.3)) is False

        # Threshold permisivo → loud pasa
        det_low = _make_detector(vad_threshold=0.4)
        assert det_low._is_speech(_loud_chunk(amp=0.3)) is True

    def test_voice_prob_uses_vad_when_available(self):
        """Si _vad está disponible, usa su output directo (no fallback)."""
        det = _make_detector()
        # Mock torch + vad
        fake_torch = MagicMock()
        fake_tensor = MagicMock()
        fake_torch.from_numpy = MagicMock(return_value=fake_tensor)
        fake_torch.no_grad = MagicMock()
        fake_torch.no_grad.return_value.__enter__ = MagicMock()
        fake_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)

        fake_vad = MagicMock()
        fake_result = MagicMock()
        fake_result.item = MagicMock(return_value=0.92)
        fake_vad.return_value = fake_result

        det._torch = fake_torch
        det._vad = fake_vad

        prob = det._voice_prob(_loud_chunk(amp=0.3))
        assert prob == pytest.approx(0.92)
        fake_vad.assert_called_once()
