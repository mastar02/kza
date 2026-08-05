"""Tests: AmbientTranscriber — integración tap→segmenter→STT→store con fakes."""
import asyncio
import time

import numpy as np
import pytest

from src.ambient.audio_archive import AudioArchiver
from src.ambient.tap import MultiChannelTap
from src.ambient.segmenter import UtteranceSegmenter
from src.ambient.source_classifier import SourceClassifier, SourceClassifierConfig
from src.ambient.transcriber import AmbientTranscriber
from src.ambient.types import RawSegment
from src.stt.whisper_fast import STTResult

SR = 16000
CHUNK = 1280


class FakeAmbientSTT:
    async def transcribe(self, audio):
        return STTResult(text="hola che", elapsed_ms=10.0,
                         no_speech_prob=0.05, avg_logprob=-0.2,
                         compression_ratio=1.0)


class FakeTagger:
    def __init__(self):
        # Contador de invocaciones: el review 2026-08-05 pidió verificar con
        # un assert, no solo por estructura del código, que la rama de texto
        # vacío NUNCA llama al tagger (es GPU, cara, y no aporta nada a una
        # fila que solo existe para medir).
        self.calls = 0

    async def tag(self, mono):
        self.calls += 1
        return ("unknown", 0.0)


class FakeDoA:
    def __init__(self):
        # Mismo motivo que FakeTagger.calls: la rama de texto vacío no debe
        # llamar GCC-PHAT tampoco.
        self.calls = 0

    def estimate(self, audio):
        self.calls += 1
        from src.ambient.doa import DoAResult
        return DoAResult(azimuth=1.0, stability=0.95)


class FakeStore:
    def __init__(self):
        self.added = []
        self.audio_paths = {}     # utt_id → path registrado

    async def add(self, utt):
        self.added.append(utt)
        return len(self.added)

    async def set_audio_path(self, utt_id, path):
        self.audio_paths[utt_id] = path

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


def test_vad_prob_propagates_to_utterance():
    # El RawSegment lleva el mean de Silero; el transcriber lo persiste
    # (señal anti-alucinación — no_speech_prob del turbo es inservible).
    store = FakeStore()
    tap, tr = _make(store, tv_azimuth=2.5)

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
        await tr.stop()
    asyncio.run(inner())

    assert len(store.added) == 1
    # vad fake: voz=1.0 ×2 + cola de silencio 0.0 ×2 → mean 0.5
    assert store.added[0].vad_prob == pytest.approx(0.5)


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


def test_quality_fn_flags_idioma_en_utterance():
    # B (flag-no-drop): el transcriber computa lang/lang_prob/lang_ok del texto
    # vía quality_fn inyectado y los persiste. NO descarta nada (es flag).
    store = FakeStore()
    seen = []

    def quality_fn(text, vad_prob):
        seen.append((text, vad_prob))
        return ("es", 0.97, True)

    tap = MultiChannelTap(maxlen_chunks=100)
    clf = SourceClassifier(SourceClassifierConfig(tv_azimuth=2.5))
    tr = AmbientTranscriber(
        tap=tap, segmenter_factory=_segmenter_factory,
        ambient_stt=FakeAmbientSTT(), tagger=FakeTagger(),
        doa_estimator=FakeDoA(), classifier=clf, store=store,
        rooms=["escritorio"], poll_interval_s=0.01, quality_fn=quality_fn,
    )

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
        await tr.stop()
    asyncio.run(inner())

    assert len(store.added) == 1
    u = store.added[0]
    assert (u.lang, u.lang_prob, u.lang_ok) == ("es", 0.97, True)
    # quality_fn recibe el texto transcripto (stripped) y el vad del segmento
    assert seen and seen[0][0] == "hola che"
    assert seen[0][1] == pytest.approx(0.5)  # vad fake: 1.0×2 + 0.0×2 → mean 0.5


def test_sin_quality_fn_idioma_queda_none():
    # default: sin quality_fn el flag es opcional → campos de idioma None
    store = FakeStore()
    tap, tr = _make(store, tv_azimuth=2.5)

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
        await tr.stop()
    asyncio.run(inner())

    assert len(store.added) == 1
    u = store.added[0]
    assert (u.lang, u.lang_prob, u.lang_ok) == (None, None, None)


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


def _seg(vad: float = 0.8) -> RawSegment:
    return RawSegment(t0=100.0, t1=102.0,
                      audio=np.full((1600, 6), 0.2, dtype=np.float32),
                      vad_prob=vad)


class EmptySTT:
    async def transcribe(self, audio):
        return STTResult(text="", elapsed_ms=5.0)


def test_segmento_sin_texto_se_persiste_si_hay_archiver(tmp_path):
    """Sin esto la tasa de deleción es invisible: el modo de falla más
    importante del ambient es transcribir habla real como vacío."""
    store = FakeStore()
    tap, tr = _make(store)
    tr._stt = EmptySTT()
    tr._archiver = AudioArchiver(base_dir=str(tmp_path), enabled=True)

    asyncio.run(tr._handle_segment("escritorio", _seg()))

    assert len(store.added) == 1
    assert store.added[0].text == ""
    assert store.added[0].text_empty is True
    assert store.audio_paths[1].endswith("escritorio/1.flac")
    # Review 2026-08-05: la rama de texto vacío NO debe llamar tagger ni DoA
    # (GPU + GCC-PHAT) — son operaciones caras que no aportan nada a una fila
    # que solo existe para medir la tasa de deleción.
    assert tr._tagger.calls == 0
    assert tr._doa.calls == 0


def test_segmento_sin_texto_NO_se_persiste_sin_archiver():
    """Regresión: con keep_audio apagado el comportamiento es el de hoy."""
    store = FakeStore()
    tap, tr = _make(store)
    tr._stt = EmptySTT()

    asyncio.run(tr._handle_segment("escritorio", _seg()))

    assert store.added == []


def test_segmento_con_texto_archiva_el_audio(tmp_path):
    store = FakeStore()
    tap, tr = _make(store)
    tr._archiver = AudioArchiver(base_dir=str(tmp_path), enabled=True)

    asyncio.run(tr._handle_segment("escritorio", _seg()))

    assert store.added[0].text == "hola che"
    assert store.added[0].text_empty is False
    assert store.audio_paths[1].endswith("escritorio/1.flac")


def test_archiver_wired_via_constructor_kwarg(tmp_path):
    """El wiring real de producción (build_ambient_path) pasa archiver= por
    keyword al constructor — nunca setea el atributo después de crear el
    objeto, que es lo único que probaban los tests de arriba. Si `archiver`
    se hubiera omitido de la firma de __init__, esta es la única prueba que
    lo hubiera detectado (con un TypeError), en vez de recién en producción."""
    store = FakeStore()
    tap = MultiChannelTap(maxlen_chunks=100)
    clf = SourceClassifier(SourceClassifierConfig(tv_azimuth=2.5))
    tr = AmbientTranscriber(
        tap=tap, segmenter_factory=_segmenter_factory,
        ambient_stt=FakeAmbientSTT(), tagger=FakeTagger(),
        doa_estimator=FakeDoA(), classifier=clf, store=store,
        rooms=["escritorio"], poll_interval_s=0.01,
        archiver=AudioArchiver(base_dir=str(tmp_path), enabled=True),
    )

    asyncio.run(tr._handle_segment("escritorio", _seg()))

    assert store.added[0].text == "hola che"
    assert store.audio_paths[1].endswith("escritorio/1.flac")


def test_fallo_del_archiver_no_rompe_la_utterance(tmp_path):
    store = FakeStore()
    tap, tr = _make(store)
    tr._archiver = AudioArchiver(base_dir=str(tmp_path), enabled=True,
                                 min_free_bytes=10**18)   # nunca hay lugar

    asyncio.run(tr._handle_segment("escritorio", _seg()))

    assert store.added[0].text == "hola che"
    assert store.audio_paths == {}      # no se registró ninguna ruta


def test_update_fallido_borra_el_archivo(tmp_path):
    """Una fila sin audio_path deja el archivo fuera del alcance de la purga
    por TTL — huérfano permanente. Se borra en el momento."""
    class BrokenStore(FakeStore):
        async def set_audio_path(self, utt_id, path):
            raise RuntimeError("db caída")

    store = BrokenStore()
    tap, tr = _make(store)
    tr._archiver = AudioArchiver(base_dir=str(tmp_path), enabled=True)

    asyncio.run(tr._handle_segment("escritorio", _seg()))   # no propaga

    assert not (tmp_path / "escritorio" / "1.flac").exists()
    # La otra mitad del requisito (review 2026-08-05): el fallo es del
    # puntero al audio, no de la utterance — sin esto el test pasaría
    # vacuamente si write() hubiera devuelto None por cualquier otro motivo.
    assert store.added[0].text == "hola che"
