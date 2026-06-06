"""Tests: UtteranceSegmenter — VAD chunked → RawSegment con pre-roll."""
import numpy as np

from src.ambient.segmenter import UtteranceSegmenter

SR = 16000
CHUNK = 1280  # 80ms


def _chunk(val: float) -> np.ndarray:
    return np.full((CHUNK, 6), val, dtype=np.float32)


def _make_segmenter(probs: list[float], **kw) -> tuple[UtteranceSegmenter, list]:
    """Segmenter con VAD fake que devuelve probs en orden (luego 0.0)."""
    seq = list(probs)
    calls: list = []

    def fake_vad(mono: np.ndarray) -> float:
        calls.append(mono.copy())
        return seq.pop(0) if seq else 0.0

    seg = UtteranceSegmenter(
        vad_predict=fake_vad, sample_rate=SR, vad_col=2,
        speech_threshold=0.5, close_silence_ms=160,  # 2 chunks de silencio
        preroll_ms=80, max_segment_s=30.0, min_speech_ms=80,
        **kw,
    )
    return seg, calls


def test_opens_and_closes_segment_with_preroll():
    # silencio, voz, voz, silencio, silencio → cierra con 2 chunks de cola
    seg, calls = _make_segmenter([0.0, 0.9, 0.9, 0.0, 0.0])
    out = []
    for i, p in enumerate([0.0, 0.1, 0.2, 0.0, 0.0]):
        out.extend(seg.feed(ts=100.0 + i * 0.08, chunk=_chunk(p)))
    assert len(out) == 1
    s = out[0]
    # pre-roll de 1 chunk (80ms): el segmento incluye el chunk de silencio previo
    assert s.t0 == 100.0
    # 5 chunks en total (preroll + 2 voz + 2 silencio de cola)
    assert s.audio.shape == (CHUNK * 5, 6)
    assert s.during_tts is False
    # el VAD recibió la columna cruda (col 2)
    assert all(c.shape == (CHUNK,) for c in calls)


def test_short_blip_below_min_speech_is_discarded():
    # 1 solo chunk de voz (80ms) con min_speech_ms=160 → descartado
    seg2 = UtteranceSegmenter(
        vad_predict=lambda m: 0.0, sample_rate=SR, vad_col=2,
        speech_threshold=0.5, close_silence_ms=160, preroll_ms=0,
        max_segment_s=30.0, min_speech_ms=160,
    )
    probs = iter([0.9, 0.0, 0.0, 0.0])
    seg2._vad_predict = lambda m: next(probs)
    out = []
    for i in range(4):
        out.extend(seg2.feed(ts=1.0 + i * 0.08, chunk=_chunk(0.1)))
    assert out == []


def test_max_segment_force_closes():
    # voz infinita → corta a max_segment_s
    seg = UtteranceSegmenter(
        vad_predict=lambda m: 0.9, sample_rate=SR, vad_col=2,
        speech_threshold=0.5, close_silence_ms=160, preroll_ms=0,
        max_segment_s=0.24, min_speech_ms=80,  # 3 chunks máx
    )
    out = []
    for i in range(6):
        out.extend(seg.feed(ts=1.0 + i * 0.08, chunk=_chunk(0.1)))
    assert len(out) >= 1
    assert out[0].audio.shape[0] <= CHUNK * 3


def test_during_tts_propagates():
    seg = UtteranceSegmenter(
        vad_predict=lambda m: 0.9, sample_rate=SR, vad_col=2,
        speech_threshold=0.5, close_silence_ms=80, preroll_ms=0,
        max_segment_s=30.0, min_speech_ms=80,
    )
    probs = iter([0.9, 0.9, 0.0])
    seg._vad_predict = lambda m: next(probs)
    out = []
    out.extend(seg.feed(ts=1.0, chunk=_chunk(0.1), tts_active=False))
    out.extend(seg.feed(ts=1.08, chunk=_chunk(0.1), tts_active=True))
    out.extend(seg.feed(ts=1.16, chunk=_chunk(0.0), tts_active=False))
    assert len(out) == 1
    assert out[0].during_tts is True


def test_vad_col_fallback_when_missing():
    # device 2ch (sin firmware 6ch): col 2 no existe → usa col 0 y avisa una vez
    seg = UtteranceSegmenter(
        vad_predict=lambda m: 0.0, sample_rate=SR, vad_col=2,
        speech_threshold=0.5, close_silence_ms=160, preroll_ms=0,
        max_segment_s=30.0, min_speech_ms=80,
    )
    two_ch = np.zeros((CHUNK, 2), dtype=np.float32)
    seg.feed(ts=1.0, chunk=two_ch)  # no debe lanzar
