"""Tests: AmbientTranscriber — integración tap→segmenter→STT→store con fakes."""
import asyncio
import time

import numpy as np

from src.ambient.tap import MultiChannelTap
from src.ambient.segmenter import UtteranceSegmenter
from src.ambient.source_classifier import SourceClassifier, SourceClassifierConfig
from src.ambient.transcriber import AmbientTranscriber
from src.stt.whisper_fast import STTResult

SR = 16000
CHUNK = 1280


class FakeAmbientSTT:
    async def transcribe(self, audio):
        return STTResult(text="hola che", elapsed_ms=10.0,
                         no_speech_prob=0.05, avg_logprob=-0.2,
                         compression_ratio=1.0)


class FakeTagger:
    async def tag(self, mono):
        return ("unknown", 0.0)


class FakeDoA:
    def estimate(self, audio):
        from src.ambient.doa import DoAResult
        return DoAResult(azimuth=1.0, stability=0.95)


class FakeStore:
    def __init__(self):
        self.added = []

    async def add(self, utt):
        self.added.append(utt)
        return len(self.added)

    async def purge_expired(self):
        return 0


def _segmenter_factory():
    # VAD fake: voz si el chunk tiene energía
    def vad(mono):
        return 1.0 if float(np.abs(mono).max()) > 0.05 else 0.0
    return UtteranceSegmenter(
        vad_predict=vad, sample_rate=SR, vad_col=2, speech_threshold=0.5,
        close_silence_ms=160, preroll_ms=0, max_segment_s=30.0, min_speech_ms=80,
    )


def _make(store, tv_azimuth=2.5):
    tap = MultiChannelTap(maxlen_chunks=100)
    clf = SourceClassifier(SourceClassifierConfig(tv_azimuth=tv_azimuth))
    tr = AmbientTranscriber(
        tap=tap, segmenter_factory=_segmenter_factory,
        ambient_stt=FakeAmbientSTT(), tagger=FakeTagger(),
        doa_estimator=FakeDoA(), classifier=clf, store=store,
        rooms=["escritorio"], poll_interval_s=0.01,
    )
    return tap, tr


def test_voice_segment_lands_in_store_labeled():
    store = FakeStore()
    tap, tr = _make(store, tv_azimuth=2.5)  # DoA fake da 1.0 → no-TV → live

    async def inner():
        await tr.start()
        now = time.time()
        voz = np.full((CHUNK, 6), 0.2, dtype=np.float32)
        sil = np.zeros((CHUNK, 6), dtype=np.float32)
        for i, ch in enumerate([voz, voz, sil, sil, sil]):
            tap.push("escritorio", ch, ts=now + i * 0.08)
        # darle ciclos al worker
        for _ in range(50):
            await asyncio.sleep(0.02)
            if store.added:
                break
        await tr.stop()
    asyncio.run(inner())

    assert len(store.added) == 1
    u = store.added[0]
    assert u.text == "hola che"
    assert u.room_id == "escritorio"
    assert u.source == "live"
    assert u.azimuth == 1.0


def test_tv_direction_labels_tv_and_signal_fires():
    store = FakeStore()
    tap, tr = _make(store, tv_azimuth=1.0)  # DoA fake da 1.0 → TV

    async def inner():
        await tr.start()
        now = time.time()
        voz = np.full((CHUNK, 6), 0.2, dtype=np.float32)
        sil = np.zeros((CHUNK, 6), dtype=np.float32)
        for i, ch in enumerate([voz, voz, sil, sil, sil]):
            tap.push("escritorio", ch, ts=now + i * 0.08)
        for _ in range(50):
            await asyncio.sleep(0.02)
            if store.added:
                break
        # señal en caliente para el shadow del wake
        assert tr.tv_active_recent("escritorio", window_s=10.0) is True
        assert tr.tv_active_recent("living", window_s=10.0) is False
        await tr.stop()
    asyncio.run(inner())

    assert store.added[0].source == "tv"


def test_store_error_does_not_kill_worker():
    class BrokenStore(FakeStore):
        async def add(self, utt):
            raise RuntimeError("disco lleno")

    store = BrokenStore()
    tap, tr = _make(store)

    async def inner():
        await tr.start()
        now = time.time()
        voz = np.full((CHUNK, 6), 0.2, dtype=np.float32)
        sil = np.zeros((CHUNK, 6), dtype=np.float32)
        for i, ch in enumerate([voz, voz, sil, sil, sil]):
            tap.push("escritorio", ch, ts=now + i * 0.08)
        await asyncio.sleep(0.3)
        # el worker sigue vivo a pesar del error
        assert any(not t.done() for t in tr._tasks)
        await tr.stop()
    asyncio.run(inner())


def test_empty_text_is_not_stored():
    class EmptySTT:
        async def transcribe(self, audio):
            return STTResult(text="", elapsed_ms=5.0)

    store = FakeStore()
    tap = MultiChannelTap(maxlen_chunks=100)
    clf = SourceClassifier(SourceClassifierConfig())
    tr = AmbientTranscriber(
        tap=tap, segmenter_factory=_segmenter_factory,
        ambient_stt=EmptySTT(), tagger=FakeTagger(), doa_estimator=FakeDoA(),
        classifier=clf, store=store, rooms=["escritorio"], poll_interval_s=0.01,
    )

    async def inner():
        await tr.start()
        now = time.time()
        voz = np.full((CHUNK, 6), 0.2, dtype=np.float32)
        sil = np.zeros((CHUNK, 6), dtype=np.float32)
        for i, ch in enumerate([voz, voz, sil, sil, sil]):
            tap.push("escritorio", ch, ts=now + i * 0.08)
        await asyncio.sleep(0.3)
        await tr.stop()
    asyncio.run(inner())
    assert store.added == []
