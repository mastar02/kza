"""Tests de integración: TextualWakeDetector wired en AmbientTranscriber +
MultiRoomAudioLoop (Task 2 del plan wake textual, 2026-07-05).

Cubre el contrato de `_handle_segment` (persist SIEMPRE antes del dispatch,
fail-open ante excepción del detector, TV nunca dispara) y el timestamp de
dispatch acústico expuesto por `MultiRoomAudioLoop` para el dedup cruzado.
"""
import asyncio
import sys
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

# Mocks de módulos de sistema (igual que test_multi_room_audio_loop.py) —
# necesarios porque multi_room_audio_loop importa sounddevice a nivel módulo.
sys.modules.setdefault("sounddevice", MagicMock())
sys.modules.setdefault("soundfile", MagicMock())
sys.modules.setdefault("pyaudio", MagicMock())

from src.ambient.tap import MultiChannelTap
from src.ambient.segmenter import UtteranceSegmenter
from src.ambient.source_classifier import SourceClassifier, SourceClassifierConfig
from src.ambient.transcriber import AmbientTranscriber
from src.ambient.textual_wake import TextualWakeDetector
from src.pipeline.command_event import CommandEvent
from src.pipeline.multi_room_audio_loop import MultiRoomAudioLoop, RoomStream
from src.stt.whisper_fast import STTResult

SR = 16000
CHUNK = 1280


# ============================================================
# Helpers — AmbientTranscriber (mismo patrón que test_transcriber.py)
# ============================================================

class FakeAmbientSTT:
    def __init__(self, text: str = "hola nexa", asr_col: int = 1):
        self._text = text
        self.asr_col = asr_col

    async def transcribe(self, audio):
        return STTResult(
            text=self._text, elapsed_ms=10.0,
            no_speech_prob=0.05, avg_logprob=-0.2, compression_ratio=1.0,
        )

    def asr_mono(self, audio):
        """Mismo contrato que AmbientSTT.asr_mono real (fixture Fix 1)."""
        if audio.ndim == 1:
            return audio
        col = self.asr_col if audio.shape[1] > self.asr_col else 0
        return np.ascontiguousarray(audio[:, col])


class FakeTagger:
    async def tag(self, mono):
        return ("unknown", 0.0)


class FakeDoA:
    def __init__(self, azimuth: float = 1.0):
        self._az = azimuth

    def estimate(self, audio):
        from src.ambient.doa import DoAResult
        return DoAResult(azimuth=self._az, stability=0.95)


class RecordingStore:
    """Store fake que registra cada `add()` en una lista compartible con el
    detector fake — permite verificar el ORDEN persist→dispatch."""

    def __init__(self):
        self.calls: list = []

    async def add(self, utt):
        self.calls.append(("persist", utt))
        return len(self.calls)

    async def purge_expired(self):
        return 0


class RecordingDetector:
    """Detector fake: registra la llamada en la misma lista que el store."""

    def __init__(self, calls: list, raise_exc: bool = False):
        self._calls = calls
        self._raise = raise_exc

    async def maybe_dispatch(self, room_id, text, source, speaker, audio):
        self._calls.append(("dispatch", room_id, text, source))
        if self._raise:
            raise RuntimeError("boom del detector (fail-open esperado)")
        return True


def _segmenter_factory():
    def vad(mono):
        return 1.0 if float(np.abs(mono).max()) > 0.05 else 0.0
    return UtteranceSegmenter(
        vad_predict=vad, sample_rate=SR, vad_col=2, speech_threshold=0.5,
        close_silence_ms=160, preroll_ms=0, max_segment_s=30.0, min_speech_ms=80,
    )


def _make_transcriber(store, ambient_stt, tv_azimuth=2.5):
    tap = MultiChannelTap(maxlen_chunks=100)
    clf = SourceClassifier(SourceClassifierConfig(tv_azimuth=tv_azimuth))
    tr = AmbientTranscriber(
        tap=tap, segmenter_factory=_segmenter_factory,
        ambient_stt=ambient_stt, tagger=FakeTagger(),
        doa_estimator=FakeDoA(azimuth=1.0), classifier=clf, store=store,
        rooms=["escritorio"], poll_interval_s=0.01,
    )
    return tap, tr


async def _feed_and_wait(tap, store):
    now = time.time()
    voz = np.full((CHUNK, 6), 0.2, dtype=np.float32)
    sil = np.zeros((CHUNK, 6), dtype=np.float32)
    for i, ch in enumerate([voz, voz, sil, sil, sil]):
        tap.push("escritorio", ch, ts=now + i * 0.08)
    for _ in range(50):
        await asyncio.sleep(0.02)
        if store.calls:
            break


# ============================================================
# AmbientTranscriber.attach_textual_wake / hook en _handle_segment
# ============================================================

def test_utterance_persists_before_dispatch():
    """El orden es SIEMPRE persist→dispatch, nunca al revés."""
    store = RecordingStore()
    tap, tr = _make_transcriber(store, FakeAmbientSTT("hola nexa"), tv_azimuth=2.5)
    tr.attach_textual_wake(RecordingDetector(store.calls))

    async def inner():
        await tr.start()
        await _feed_and_wait(tap, store)
        await tr.stop()
    asyncio.run(inner())

    kinds = [c[0] for c in store.calls]
    assert kinds == ["persist", "dispatch"]


def test_detector_exception_does_not_prevent_persist_nor_kill_worker():
    """Fail-open: una excepción del detector no impide persistir ni mata el worker."""
    store = RecordingStore()
    tap, tr = _make_transcriber(store, FakeAmbientSTT("hola nexa"), tv_azimuth=2.5)
    tr.attach_textual_wake(RecordingDetector(store.calls, raise_exc=True))

    async def inner():
        await tr.start()
        await _feed_and_wait(tap, store)
        assert any(not t.done() for t in tr._tasks)
        await tr.stop()
    asyncio.run(inner())

    kinds = [c[0] for c in store.calls]
    assert kinds == ["persist", "dispatch"]


def test_tv_source_persists_but_never_reaches_dispatch_fn():
    """source==tv: se persiste igual, pero el TextualWakeDetector real no
    invoca dispatch_fn (regla de negocio de Task 1, verificada acá end-to-end)."""
    store = RecordingStore()
    # tv_azimuth == azimuth del DoA fake (1.0) → SourceClassifier clasifica 'tv'
    tap, tr = _make_transcriber(store, FakeAmbientSTT("hola nexa"), tv_azimuth=1.0)

    dispatched = []

    async def dispatch_fn(event):
        dispatched.append(event)
        return {}

    detector = TextualWakeDetector(
        dispatch_fn=dispatch_fn,
        last_acoustic_command_ts_fn=lambda room_id: 0.0,
    )
    tr.attach_textual_wake(detector)

    async def inner():
        await tr.start()
        await _feed_and_wait(tap, store)
        await tr.stop()
    asyncio.run(inner())

    assert len(store.calls) == 1
    assert store.calls[0][1].source == "tv"
    assert dispatched == []


def test_textual_wake_receives_mono_audio_not_raw_multichannel_segment():
    """CRITICAL: el hook debe pasar la vista ASR mono (1D), no `seg.audio` crudo.

    `seg.audio` es 2D `(frames, channels)` en producción (mic 6ch); el
    CommandEvent que arma el detector se re-inyecta al router, que corre
    SpeakerID/ECAPA sobre `event.audio` esperando 1D. Pasar 2D rompería el
    speaker-ID en TODO dispatch textual (sin try/except en
    `command_processor._identify_speaker`).
    """
    store = RecordingStore()
    captured_audio: list[np.ndarray] = []

    class CapturingDetector:
        async def maybe_dispatch(self, room_id, text, source, speaker, audio):
            captured_audio.append(audio)
            return True

    tap, tr = _make_transcriber(store, FakeAmbientSTT("hola nexa"), tv_azimuth=2.5)
    tr.attach_textual_wake(CapturingDetector())

    async def inner():
        await tr.start()
        await _feed_and_wait(tap, store)
        await tr.stop()
    asyncio.run(inner())

    assert len(captured_audio) == 1
    assert captured_audio[0].ndim == 1


def test_no_detector_attached_is_pure_noop():
    """Sin attach_textual_wake() (default None) el pipeline persiste igual."""
    store = RecordingStore()
    tap, tr = _make_transcriber(store, FakeAmbientSTT("hola nexa"), tv_azimuth=2.5)

    async def inner():
        await tr.start()
        await _feed_and_wait(tap, store)
        await tr.stop()
    asyncio.run(inner())

    assert len(store.calls) == 1


# ============================================================
# MultiRoomAudioLoop — timestamp de dispatch acústico
# ============================================================

def _make_wake_detector():
    m = MagicMock()
    m.load = MagicMock()
    m.detect = MagicMock(return_value=None)
    m.get_active_models = MagicMock(return_value=["hey_jarvis"])
    return m


def _make_echo_suppressor():
    m = MagicMock()
    m.is_safe_to_listen = True
    m.should_process_audio = MagicMock(return_value=(True, "ok"))
    m.is_human_voice = MagicMock(return_value=True)
    return m


def _make_follow_up():
    m = MagicMock()
    m.is_active = False
    m.start_conversation = MagicMock()
    return m


def _make_loop() -> MultiRoomAudioLoop:
    rooms = {
        "cocina": RoomStream(
            room_id="cocina", device_index=2,
            wake_detector=_make_wake_detector(),
            echo_suppressor=_make_echo_suppressor(),
        ),
    }
    return MultiRoomAudioLoop(room_streams=rooms, follow_up=_make_follow_up())


@pytest.mark.asyncio
async def test_dispatch_command_registers_acoustic_timestamp():
    """_dispatch_command registra time.monotonic() por room, expuesto por el getter."""
    loop = _make_loop()

    async def cb(event):
        return {"success": True}
    loop.on_command(cb)

    audio = np.zeros(8000, dtype=np.float32)
    event = CommandEvent(audio=audio, room_id="cocina")

    before = time.monotonic()
    await loop._dispatch_command(event)
    after = time.monotonic()

    ts = loop.last_command_dispatch_ts("cocina")
    assert before <= ts <= after


def test_last_command_dispatch_ts_unknown_room_returns_zero():
    """Getter devuelve 0.0 (nunca) para una room sin dispatch previo."""
    loop = _make_loop()
    assert loop.last_command_dispatch_ts("sala_jamas_vista") == 0.0


@pytest.mark.asyncio
async def test_acoustic_ts_is_visible_during_slow_callback():
    """CRITICAL: el ts debe existir ANTES de awaitear el callback, no después.

    Un comando slow-path (segundos, reasoner cloud) deja una ventana donde,
    si el ts se registrara solo al final, el canal textual evaluaría la
    MISMA utterance sin ver el dedup acústico → doble ejecución. Este test
    verifica que el getter ya ve un ts > 0.0 DESDE ADENTRO del callback,
    antes de que este retorne.
    """
    loop = _make_loop()
    seen_ts_during_callback = []

    async def slow_cb(event):
        # Mientras el callback está "corriendo" (comando slow-path), el ts
        # acústico ya debe estar registrado para esta room.
        seen_ts_during_callback.append(loop.last_command_dispatch_ts(event.room_id))
        await asyncio.sleep(0)  # cede el control, simula I/O del slow path
        return {"success": True}

    loop.on_command(slow_cb)

    audio = np.zeros(8000, dtype=np.float32)
    event = CommandEvent(audio=audio, room_id="cocina")
    await loop._dispatch_command(event)

    assert len(seen_ts_during_callback) == 1
    assert seen_ts_during_callback[0] > 0.0
