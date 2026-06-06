"""AmbientTranscriber — orquestador del ambient path (spec §3-§5).

Un worker async por room: drena el tap, alimenta el segmenter y por cada
RawSegment corre STT (cuda:0) + speaker + DoA, clasifica la fuente y persiste.
Contrato best-effort: cualquier excepción → log + backoff exponencial; el
command path jamás se entera. build_ambient_path() arma el grafo completo
desde la config (DI por constructor en todos los niveles).
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Callable

from src.ambient.types import AmbientUtterance, RawSegment

logger = logging.getLogger(__name__)

_PURGE_INTERVAL_S = 3600.0


class AmbientTranscriber:
    """Workers por room + purga + señal tv_active_recent()."""

    def __init__(
        self,
        tap,
        segmenter_factory: Callable[[], object],
        ambient_stt,
        tagger,
        doa_estimator,
        classifier,
        store,
        rooms: list[str],
        poll_interval_s: float = 0.25,
    ):
        self._tap = tap
        self._segmenter_factory = segmenter_factory
        self._stt = ambient_stt
        self._tagger = tagger
        self._doa = doa_estimator
        self._classifier = classifier
        self._store = store
        self._rooms = rooms
        self.poll_interval_s = poll_interval_s

        self._running = False
        self._tasks: list[asyncio.Task] = []
        # Última utterance 'tv' vista por room: timestamp del registro — señal shadow
        self._last_tv: dict[str, float] = {}

    async def start(self) -> None:
        """Lanzar un worker por room + el worker de purga."""
        self._running = True
        for room_id in self._rooms:
            self._tap.register_room(room_id)
            self._tasks.append(asyncio.create_task(self._room_worker(room_id)))
        self._tasks.append(asyncio.create_task(self._purge_worker()))
        logger.info(f"AmbientTranscriber activo ({len(self._rooms)} rooms)")

    async def stop(self) -> None:
        """Cancelar todos los workers y esperar su cierre (idempotente)."""
        self._running = False
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []

    def tv_active_recent(self, room_id: str, window_s: float = 5.0) -> bool:
        """True si hubo una utterance 'tv' que terminó hace < window_s.

        Señal para el shadow anti-TV del wake (Fase 2: solo log).
        """
        last = self._last_tv.get(room_id)
        return last is not None and (time.time() - last) < window_s

    async def _room_worker(self, room_id: str) -> None:
        """Loop best-effort de una room: drain → segmentar → procesar.

        Errores → log + backoff exponencial (1s→60s); jamás propaga.
        """
        segmenter = self._segmenter_factory()
        backoff = 1.0
        while self._running:
            try:
                items = self._tap.drain(room_id)
                segments = []
                for ts, chunk, tts_active in items:
                    segments.extend(
                        segmenter.feed(ts=ts, chunk=chunk, tts_active=tts_active)
                    )
                for seg in segments:
                    await self._handle_segment(room_id, seg)
                backoff = 1.0
                await asyncio.sleep(self.poll_interval_s)
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception(
                    f"AmbientTranscriber[{room_id}]: error (best-effort, "
                    f"backoff {backoff:.0f}s)"
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

    async def _handle_segment(self, room_id: str, seg: RawSegment) -> None:
        """Procesar un segmento: STT + speaker + DoA → clasificar → persistir.

        Best-effort: un segmento malo se loguea y se descarta sin matar el
        worker de la room.
        """
        try:
            stt_result = await self._stt.transcribe(seg.audio)
            if not stt_result.text.strip():
                return
            mono = seg.audio[:, 0] if seg.audio.ndim == 2 else seg.audio
            speaker, sp_conf = await self._tagger.tag(mono)
            doa = await asyncio.to_thread(self._doa.estimate, seg.audio)
            azimuth = doa.azimuth if doa else None
            stability = doa.stability if doa else 0.0
            source = self._classifier.classify(
                speaker=speaker, azimuth=azimuth, stability=stability,
                during_tts=seg.during_tts,
            )
            utt = AmbientUtterance(
                room_id=room_id, t0=seg.t0, t1=seg.t1,
                text=stt_result.text.strip(),
                speaker=speaker, speaker_confidence=sp_conf,
                azimuth=azimuth, azimuth_stability=stability,
                source=source,
                confidence=stt_result.avg_logprob,
                no_speech_prob=stt_result.no_speech_prob,
                during_tts=seg.during_tts,
            )
            if source == "tv":
                self._last_tv[room_id] = time.time()
            await self._store.add(utt)
            logger.debug(
                f"[Ambient] {room_id} {source} {speaker}: "
                f"{utt.text[:60]!r} (az={azimuth}, stab={stability:.2f})"
            )
        except Exception:
            # un segmento malo no tira el worker — se pierde ese segmento
            logger.exception(f"AmbientTranscriber[{room_id}]: segmento descartado")

    async def _purge_worker(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(_PURGE_INTERVAL_S)
                await self._store.purge_expired()
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("AmbientTranscriber: purga falló (sigo)")


@dataclass
class AmbientPath:
    """Grafo armado del ambient path — lo que main.py necesita."""

    tap: object
    transcriber: AmbientTranscriber
    store: object
    distiller: object | None

    async def start(self) -> None:
        """Arrancar: store → workers → job del distiller (si hay memoria)."""
        await self.store.init()
        await self.transcriber.start()
        if self.distiller is not None:
            # En _tasks del transcriber: su stop() cancela TODO de una.
            self.transcriber._tasks.append(
                asyncio.create_task(self.distiller.run_forever())
            )

    async def stop(self) -> None:
        """Apagar en orden inverso; seguro de llamar sin start() previo."""
        if self.distiller is not None:
            self.distiller.stop()
        await self.transcriber.stop()
        await self.store.close()


def build_ambient_path(
    ambient_cfg: dict,
    stt_base_cfg: dict,
    room_ids: list[str],
    store_fact_fn: Callable[..., str] | None,
) -> AmbientPath:
    """Construir el ambient path completo desde config (wiring de main.py).

    Args:
        ambient_cfg: bloque ``ambient:`` de settings.yaml.
        stt_base_cfg: bloque ``stt:`` top-level (hereda model/compute_type/language).
        room_ids: rooms con mic real (las que tienen RoomStream).
        store_fact_fn: LongTermMemory.store_fact del MemoryManager de main, o
            None (sin memoria → sin distiller, solo buffer TTL).
    """
    from src.ambient.ambient_stt import AmbientSTT
    from src.ambient.distiller import Distiller, make_local_chat_fn
    from src.ambient.doa import DoAEstimator
    from src.ambient.segmenter import UtteranceSegmenter, make_silero_predictor
    from src.ambient.source_classifier import SourceClassifier, SourceClassifierConfig
    from src.ambient.speaker_tagger import SpeakerTagger
    from src.ambient.store import AmbientStore
    from src.ambient.tap import MultiChannelTap
    from src.stt.whisper_fast import FastWhisperSTT
    from src.users.speaker_identifier import SpeakerIdentifier

    stt_cfg = ambient_cfg.get("stt", {}) or {}
    # Hereda del stt top-level; overridea device/beam/prompt (desviación 3 del plan)
    ambient_whisper = FastWhisperSTT(
        model=stt_cfg.get("model", stt_base_cfg.get("model", "./models/whisper-v3-turbo")),
        device=stt_cfg.get("device", "cuda:0"),
        compute_type=stt_cfg.get("compute_type", stt_base_cfg.get("compute_type", "int8_float16")),
        language=stt_base_cfg.get("language", "es"),
        beam_size=stt_cfg.get("beam_size", 1),
        initial_prompt=None,
        vad_filter=False,
    )
    ambient_stt = AmbientSTT(stt=ambient_whisper, asr_col=ambient_cfg.get("asr_col", 1))

    seg_cfg = ambient_cfg.get("segmenter", {}) or {}
    vad_predict = make_silero_predictor()

    def segmenter_factory() -> UtteranceSegmenter:
        return UtteranceSegmenter(
            vad_predict=vad_predict,
            vad_col=ambient_cfg.get("vad_col", 2),
            speech_threshold=seg_cfg.get("speech_threshold", 0.5),
            close_silence_ms=seg_cfg.get("close_silence_ms", 700),
            preroll_ms=seg_cfg.get("preroll_ms", 500),
            max_segment_s=seg_cfg.get("max_segment_s", 30.0),
            min_speech_ms=seg_cfg.get("min_speech_ms", 300),
        )

    sp_cfg = ambient_cfg.get("speaker", {}) or {}
    identifier = SpeakerIdentifier(
        model_name="speechbrain/spkrec-ecapa-voxceleb",
        device=stt_cfg.get("device", "cuda:0"),
        similarity_threshold=0.75,
    )

    def embeddings_loader() -> dict:
        # Enrolamiento pendiente en el proyecto (data/users.json no existe);
        # loader vacío = speaker 'unknown' siempre, sin gastar GPU.
        from pathlib import Path

        import numpy as np

        emb_dir = Path(sp_cfg.get("embeddings_dir", "./data/users"))
        out = {}
        if emb_dir.is_dir():
            for f in emb_dir.glob("*_voice.npy"):
                out[f.stem.replace("_voice", "")] = np.load(f)
        return out

    tagger = SpeakerTagger(
        identifier=identifier,
        embeddings_loader=embeddings_loader,
        min_audio_s=sp_cfg.get("min_audio_s", 0.8),
    )

    clf_cfg = ambient_cfg.get("classifier", {}) or {}
    rooms_cfg = ambient_cfg.get("rooms", {}) or {}
    # tv_azimuth: por ahora una sola room con mic — usar la primera que lo tenga
    tv_azimuth = None
    for rid in room_ids:
        v = (rooms_cfg.get(rid, {}) or {}).get("tv_azimuth")
        if v is not None:
            tv_azimuth = float(v)
            break
    classifier = SourceClassifier(SourceClassifierConfig(
        tv_azimuth=tv_azimuth,
        tv_tolerance_rad=clf_cfg.get("tv_tolerance_rad", 0.35),
        min_stability=clf_cfg.get("min_stability", 0.6),
        require_known_speaker_for_live=clf_cfg.get("require_known_speaker_for_live", False),
    ))

    store = AmbientStore(
        db_path=ambient_cfg.get("db_path", "./data/ambient.db"),
        retention_hours=ambient_cfg.get("retention_hours", 12.0),
    )

    tap = MultiChannelTap()
    transcriber = AmbientTranscriber(
        tap=tap,
        segmenter_factory=segmenter_factory,
        ambient_stt=ambient_stt,
        tagger=tagger,
        doa_estimator=DoAEstimator(raw_first_col=ambient_cfg.get("raw_first_col", 2)),
        classifier=classifier,
        store=store,
        rooms=room_ids,
        poll_interval_s=ambient_cfg.get("poll_interval_s", 0.25),
    )

    distiller = None
    dis_cfg = ambient_cfg.get("distill", {}) or {}
    if store_fact_fn is not None:
        distiller = Distiller(
            store=store,
            chat_fn=make_local_chat_fn(
                llm_url=dis_cfg.get("llm_url", "http://127.0.0.1:8101/v1"),
                model=dis_cfg.get("model", "local"),
            ),
            store_fact_fn=store_fact_fn,
            interval_hours=dis_cfg.get("interval_hours", 6.0),
            min_batch=dis_cfg.get("min_batch", 5),
            max_batch_chars=dis_cfg.get("max_batch_chars", 12000),
        )

    return AmbientPath(tap=tap, transcriber=transcriber, store=store, distiller=distiller)
