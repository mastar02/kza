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
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Callable

from src.ambient.types import AmbientUtterance, RawSegment
from src.ambient.room_producer import RoomProducer
from src.core.owned_work import OwnedBlockingWork, WorkCloseTimeout
from src.memory.producer_context import producer_context
from src.memory.producer_scopes import ProducerScopes
from src.memory.prompt_context import SourcesInvalidated, run_private

if TYPE_CHECKING:
    import numpy as np

    from src.ambient.audio_archive import AudioArchiver
    from src.ambient.textual_wake import TextualWakeDetector

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
        quality_fn: Callable[[str, float | None], tuple[str | None, float | None, bool]]
        | None = None,
        archiver: AudioArchiver | None = None,
        privacy=None,
        close_timeout_s: float = 1,
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
        # Regla "español conservable" (language_quality.make_quality_fn): marca
        # cada utterance con (lang, lang_prob, lang_ok) al persistir — FLAG, no
        # drop. Opcional (None = sin clasificar, campos quedan None).
        self._quality_fn = quality_fn
        # AudioArchiver opcional (spec 2026-08-05). None = comportamiento
        # histórico exacto: no se guarda audio y los segmentos sin texto se
        # descartan sin persistir.
        self._archiver = archiver
        self._privacy = privacy
        if privacy is not None:
            privacy.require_available()
            if store.source_policy is not privacy.ledger:
                raise ValueError("ambient_authority_mismatch")
            if tap is not None:
                tap.bind_ingress(privacy.scopes, rooms)
        self._native = OwnedBlockingWork(close_timeout_s=close_timeout_s)
        self._close_timeout_s = close_timeout_s
        self._closed = False
        self._closing = None
        self._segments = {}
        self._batches: set[asyncio.Task] = set()
        self._room_states: dict[str, RoomProducer] = {}
        self._prunes = {}
        self._scopes = ProducerScopes(self.snapshot_operations, capacity=min(256, max(8, len(rooms)*4)))
        if privacy is not None:
            privacy.add_borrower(self.begin_close)

        self._running = False
        self._tasks: list[asyncio.Task] = []
        # Última utterance 'tv' vista por room: timestamp del registro — señal shadow
        self._last_tv: dict[str, float] = {}
        # Wake textual (spec 2026-07-05): None = feature apagada (comportamiento
        # previo exacto). Se inyecta post-init vía attach_textual_wake() — mismo
        # patrón que MultiRoomAudioLoop.attach_ambient (orden de DI en main.py:
        # el detector necesita el request_router, que se construye después).
        self._textual_wake = None

    async def start(self) -> None:
        """Lanzar un worker por room + el worker de purga."""
        if self._closed:
            raise RuntimeError("ambient_transcriber_closed")
        if self._running:
            return
        self._running = True
        privacy = self._privacy
        def buffer_source(room_id, timestamp):
            from uuid import uuid4
            from src.memory.fact_ledger import MemoryOrigin
            return privacy.source_for_capture(str(uuid4()), timestamp, room_id, origin=MemoryOrigin.AMBIENT)
        def epoch_available(epoch, room_id):
            from src.memory.fact_ledger import MemoryOrigin
            return privacy.scopes.epoch_available(epoch, room_id, origin=MemoryOrigin.AMBIENT)
        for room_id in self._rooms:
            self._room_states[room_id] = RoomProducer(room_id, self._segmenter_factory,
                require_cleanup=self._privacy is not None,
                boundary_fn=self._privacy.scopes.buffer_boundary if self._privacy is not None else None,
                source_fn=buffer_source if privacy is not None else None,
                epoch_available=epoch_available if privacy is not None else None)
            self._tap.register_room(room_id)
            with producer_context(room_id):
                self._tasks.append(asyncio.create_task(self._room_worker(room_id)))
        with producer_context(None):
            self._tasks.append(asyncio.create_task(self._purge_worker()))
            self._tasks.append(asyncio.create_task(self._privacy_worker()))
        logger.info(f"AmbientTranscriber activo ({len(self._rooms)} rooms)")

    async def stop(self) -> None:
        """Bound close observation while retaining actual native work and buffers."""
        closing = self.begin_close()
        cancelled = False
        while not closing.done():
            try:
                async with asyncio.timeout(self._close_timeout_s):
                    await asyncio.shield(closing)
            except asyncio.CancelledError:
                cancelled = True
            except TimeoutError as error:
                raise WorkCloseTimeout("ambient_resources_still_running") from error
        closing.result()
        if cancelled:
            raise asyncio.CancelledError

    def begin_close(self) -> asyncio.Task:
        """Close admission and retain actual VAD, ASR, speaker and delivery cleanup."""
        self._closed = True
        self._running = False
        if self._closing is None:
            if self._tap is not None:
                self._tap.close()
            for task in (*self._tasks, *self._segments, *self._batches):
                task.cancel()
            drains = [room.begin_close() for room in self._room_states.values()]
            if callable(getattr(type(self._tap), 'begin_close', None)):
                drains.append(self._tap.begin_close())
            drains.append(self._native.begin_close())
            for adapter in (self._stt, self._tagger, self._archiver):
                if callable(getattr(type(adapter), "begin_close", None)):
                    drains.append(adapter.begin_close())
            async def finish():
                await asyncio.gather(*self._tasks, *self._segments, *self._batches, *self._prunes.values(), return_exceptions=True)
                await asyncio.gather(*drains)
                self._tasks.clear()
                self._last_tv.clear()
                self._prunes.clear()
            self._closing = asyncio.create_task(finish())
        return self._closing

    def snapshot_operations(self) -> tuple[asyncio.Task, ...]:
        """Snapshot finite actual operations, excluding idle room/maintenance loops."""
        tasks = set(self._segments) | self._batches | set(self._native._tasks) | set(self._prunes.values())
        for room in self._room_states.values():
            tasks.update(room.snapshot_operations())
        for adapter in (self._stt, self._tagger):
            if callable(getattr(type(adapter), "snapshot_operations", None)):
                tasks.update(adapter.snapshot_operations())
        if self._closing is not None and not self._closing.done():
            tasks.add(self._closing)
        return tuple(tasks)

    def conversation_busy(self, conversation_id: str) -> bool:
        """Include actual finite work, retained VAD inputs and failed native reset."""
        if self._privacy is None:
            raise RuntimeError('privacy_authority_unavailable')
        ingress_busy = self._tap is not None and (self._tap.ingress_pending() or self._tap.buffered_count() > 0)
        return ingress_busy or self._scopes.busy(conversation_id, self._privacy.ledger) or any(
            room.conversation_busy(conversation_id, self._privacy.ledger) for room in self._room_states.values())

    def resource_status(self) -> dict:
        """Expose counts without projecting capture IDs, room names, text or audio."""
        return {"closed": self._closed, "segments": len(self._segments),
                "batches": len(self._batches), "source_scopes": self._scopes.resource_status(),
                "operations": sum(not task.done() for task in self.snapshot_operations()),
                "tap_chunks": self._tap.buffered_count() if self._tap is not None else 0,
                "rooms": [room.status() for room in self._room_states.values()],
                "closing": self._closing is not None and not self._closing.done()}

    def prune_private(self) -> None:
        """Publish capture fences and schedule native reset without blocking capture."""
        if self._closed or self._privacy is None:
            return
        for room_id in self._rooms:
            boundary = self._privacy.scopes.buffer_boundary(room_id)
            if self._tap is not None:
                self._tap.publish_boundary(room_id, boundary)
            state = self._room_states.get(room_id)
            if state is None:
                continue
            state.publish_boundary(boundary)
            previous = self._prunes.get(room_id)
            if previous is not None and previous.done():
                previous.result()
                del self._prunes[room_id]
            if state.needs_prune() and room_id not in self._prunes:
                with producer_context(room_id):
                    self._prunes[room_id] = asyncio.create_task(state.prune())

    async def _privacy_worker(self) -> None:
        while self._running:
            try:
                self.prune_private()
            except Exception as error:
                logger.warning("Ambient buffer retirement deferred: %s", type(error).__name__)
            await asyncio.sleep(.05)

    def attach_textual_wake(self, detector: "TextualWakeDetector") -> None:
        """Inyectar el TextualWakeDetector post-init (orden de DI en main.py:
        el detector se construye después del request_router, que a su vez se
        construye después del ambient path — mismo patrón que
        MultiRoomAudioLoop.attach_ambient/attach_response_handler).

        Args:
            detector: `TextualWakeDetector` (o compatible) con
                `async maybe_dispatch(room_id, text, source, speaker, audio,
                vad_prob=None)`.
        """
        self._textual_wake = detector

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
        state = self._room_states[room_id]
        backoff = 1.0
        task = None
        while self._running:
            try:
                self.prune_private()
                with producer_context(room_id):
                    task = asyncio.create_task(self._consume_batch(room_id, state))
                self._batches.add(task)
                def finished(done):
                    self._batches.discard(done)
                    if not done.cancelled():
                        done.exception()
                task.add_done_callback(finished)
                await asyncio.shield(task)
                backoff = 1.0
                await asyncio.sleep(self.poll_interval_s)
            except asyncio.CancelledError:
                if task is not None:
                    task.cancel()
                return
            except Exception as error:
                logger.warning("Ambient room processing deferred: %s", type(error).__name__)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
            finally:
                task = None

    async def _consume_batch(self, room_id, state):
        # The idle worker never owns arrays across an await. This finite task
        # and its actual native children retain the metadata scope instead.
        items = seg = None
        with self._scopes.operation():
            try:
                from src.ambient.tap import MultiChannelTap
                items = self._tap.drain_owned(room_id) if isinstance(self._tap, MultiChannelTap) else self._tap.drain(room_id)
                if items:
                    await state.feed(items)
                items = None
                while state.ready:
                    self._scopes.replace_sources(None)
                    seg = state.ready.pop(0)
                    await self._handle_segment(room_id, seg)
                    seg = None
            finally:
                items = seg = None

    async def _handle_segment(self, room_id: str, seg: RawSegment) -> None:
        """Own one sourced pipeline even after its observer is cancelled."""
        task = capture_source = input_source = None
        try:
            if self._closed:
                return
            if self._privacy is not None:
                from uuid import uuid4
                from src.memory.fact_ledger import MemoryOrigin, MemorySource
                input_source = seg.input_source
                if input_source is not None and (not isinstance(input_source, MemorySource)
                    or input_source.origin is not MemoryOrigin.AMBIENT or seg.t0 < input_source.captured_at
                    or seg.t0 >= input_source.expires_at or not self._privacy.sources_available((input_source,))):
                    raise PermissionError('ambient_segment_source_conflict')
                options = {} if input_source is None else {
                    'conversation_id': input_source.conversation_id,
                    'expires_at': min(input_source.expires_at, seg.t0+48*3600)}
                capture_source = self._privacy.source_for_capture(str(uuid4()), seg.t0, room_id, origin=MemoryOrigin.AMBIENT, **options)
                if input_source is not None:
                    capture_source = replace(capture_source, fact_dependencies=input_source.fact_dependencies)
            self._require_live(capture_source)
            if len(self._segments) >= max(1, len(self._rooms)):
                logger.debug("Ambient segment admission deferred: capacity")
                return
            sources = (capture_source,) if capture_source is not None else ()
            self._scopes.replace_sources(sources)
            with self._scopes.operation(sources):
                with producer_context(room_id, sources):
                    task = asyncio.create_task(self._run_segment(room_id, seg, capture_source))
                self._segments[task] = capture_source
                def finished(done):
                    self._segments.pop(done, None)
                    if not done.cancelled():
                        done.exception()
                task.add_done_callback(finished)
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    if asyncio.current_task().cancelling():
                        task.cancel()
                        raise
        except (SourcesInvalidated, PermissionError):
            pass
        finally:
            task = seg = capture_source = input_source = None

    def _live(self, source) -> bool:
        return not self._closed and (self._privacy is None or self._privacy.sources_available((source,)))

    def _require_live(self, source) -> None:
        if not self._live(source):
            raise SourcesInvalidated("ambient_sources_invalidated")

    async def _run_segment(self, room_id, seg, capture_source):
        try:
            await run_private(self._process_segment(room_id, seg, capture_source), lambda: self._live(capture_source))
        except SourcesInvalidated:
            pass
        except Exception as error:
            logger.warning("Ambient segment processing deferred: %s", type(error).__name__)
        finally:
            seg = capture_source = None

    async def _process_segment(self, room_id, seg, capture_source):
        stt_result = text = mono = doa = utt = empty = None
        try:
            provenance = {}
            if capture_source is not None:
                provenance = {"capture_id": capture_source.capture_id, "conversation_id": capture_source.conversation_id,
                              "expires_at": capture_source.expires_at}
            self._require_live(capture_source)
            stt_result = await self._stt.transcribe(seg.audio)
            self._require_live(capture_source)
            text = stt_result.text.strip()
            archiving = self._archiver is not None and self._archiver.enabled
            if not text:
                # Sin archiver: comportamiento histórico (descartar y salir).
                # Con archiver: se persiste una fila mínima + el audio, porque
                # la tasa de deleción —habla real transcripta como vacío— es
                # el modo de falla que ninguna otra señal deja ver.
                if not archiving:
                    return
                empty = AmbientUtterance(
                    room_id=room_id, t0=seg.t0, t1=seg.t1, text="",
                    source="self" if seg.during_tts else "unknown",
                    vad_prob=seg.vad_prob, during_tts=seg.during_tts,
                    text_empty=True, **provenance,
                )
                empty_id = await self._store.add(empty)
                try:
                    await self._archive_audio(room_id, empty_id, seg.audio, source=capture_source)
                except Exception:
                    # No re-lanzar acá: el fallo es del puntero al audio, no
                    # de la utterance (ya persistida). Dejar que escale al
                    # except externo pondría "segmento descartado" en el log
                    # cuando en realidad el segmento SÍ quedó guardado.
                    logger.exception(
                        f"AmbientTranscriber[{room_id}]: fallo al archivar "
                        f"audio (utterance {empty_id} ya persistida, sigue "
                        f"sin audio_path)"
                    )
                return
            # Flag de idioma sobre el TEXTO (no la energía): el discriminante
            # real de calidad del rioplatense. FLAG, no drop — la utterance se
            # persiste igual; lang_ok solo gobierna qué consume el distiller.
            lang = lang_prob = None
            lang_ok = None
            if self._quality_fn is not None:
                self._require_live(capture_source)
                lang, lang_prob, lang_ok = await self._native.run(self._quality, text, seg.vad_prob, capture_source)
            # Tagger sobre ch0 (Conference): full-band, presente en TODOS los
            # firmwares (6ch/2ch/mono). Asimétrico a propósito con el STT
            # (ch1 ASR): los embeddings de hablante prefieren la señal con
            # post-proceso completo; el ASR prefiere el tap sin NS.
            mono = seg.audio[:, 0] if seg.audio.ndim == 2 else seg.audio
            self._require_live(capture_source)
            speaker, sp_conf = await self._tagger.tag(mono)
            self._require_live(capture_source)
            doa = await self._native.run(self._direction, seg.audio, capture_source)
            self._require_live(capture_source)
            azimuth = doa.azimuth if doa else None
            stability = doa.stability if doa else 0.0
            source = self._classifier.classify(
                speaker=speaker, azimuth=azimuth, stability=stability,
                during_tts=seg.during_tts,
            )
            utt = AmbientUtterance(
                room_id=room_id, t0=seg.t0, t1=seg.t1,
                text=text,
                speaker=speaker, speaker_confidence=sp_conf,
                azimuth=azimuth, azimuth_stability=stability,
                source=source,
                confidence=stt_result.avg_logprob,
                no_speech_prob=stt_result.no_speech_prob,
                vad_prob=seg.vad_prob,
                lang=lang, lang_prob=lang_prob, lang_ok=lang_ok,
                during_tts=seg.during_tts, **provenance,
            )
            if source == "tv":
                self._last_tv[room_id] = time.time()
            utt_id = await self._store.add(utt)
            logger.debug(
                "[Ambient] room=%s source=%s azimuth=%s stability=%.2f",
                room_id, source, azimuth, stability
            )
            # Wake textual (spec 2026-07-05): SIEMPRE después de persistir —
            # la utterance ya quedó en el store pase lo que pase acá. Fail-open
            # explícito: una excepción del detector (o de dispatch_fn/router)
            # jamás debe matar el worker de la room. Va ANTES del archivado
            # (review 2026-08-05): el archivado es puramente instrumentación
            # (encode FLAC + shutil.disk_usage) y no puede quedar en serie
            # por delante de un dispatch de cara al usuario, ni un fallo del
            # UPDATE de audio_path puede tragarse un "nexa" real.
            if self._textual_wake is not None and seg.complete:
                self._require_live(capture_source)
                try:
                    # review PR #16: `seg.t0` es wall-clock (tap.py stampea
                    # time.time()); TextualWakeDetector compara en
                    # time.monotonic(). Lectura dual fresca en el punto de
                    # uso — sin estado persistido, sin drift acumulado — para
                    # derivar el equivalente monotónico de seg.t0. Clamp a
                    # 0.0: un salto de reloj hacia atrás no debe producir un
                    # ts "futuro" (eso sub-estimaría la brecha real y
                    # des-suprimiría el dedup acústico, la dirección
                    # peligrosa del bug original).
                    mono_now = time.monotonic()
                    wall_now = time.time()
                    utterance_started_ts = mono_now - max(0.0, wall_now - seg.t0)
                    provenance = {}
                    if utt.conversation_id is not None and utt.expires_at is not None:
                        from src.memory.fact_ledger import MemoryOrigin, MemorySource
                        provenance["memory_source"] = MemorySource(utt.capture_id, utt.conversation_id,
                                                                   utt.t0, utt.expires_at, MemoryOrigin.AMBIENT)
                    await self._textual_wake.maybe_dispatch(
                        room_id, text, source, speaker,
                        audio=self._stt.asr_mono(seg.audio),
                        vad_prob=seg.vad_prob,
                        utterance_started_ts=utterance_started_ts,
                        **provenance,
                    )
                except Exception as error:
                    logger.warning("Ambient textual wake deferred: %s", type(error).__name__)
            if archiving:
                try:
                    await self._archive_audio(room_id, utt_id, seg.audio, source=capture_source)
                except Exception:
                    # Igual que en la rama de texto vacío: no dejar que esto
                    # escale al except externo, que loguearía "segmento
                    # descartado" siendo que la utterance (y el wake) ya se
                    # procesaron con éxito.
                    logger.exception(
                        f"AmbientTranscriber[{room_id}]: fallo al archivar "
                        f"audio (utterance {utt_id} ya persistida, sigue "
                        f"sin audio_path)"
                    )
        finally:
            seg = capture_source = stt_result = text = mono = doa = utt = empty = None

    def _quality(self, text, probability, source):
        self._require_live(source)
        return self._quality_fn(text, probability)

    def _direction(self, audio, source):
        self._require_live(source)
        return self._doa.estimate(audio)

    async def _archive_audio(self, room_id: str, utt_id: int, audio: np.ndarray, *, source=None) -> None:
        """Guardar el audio del segmento y apuntar la fila al archivo.

        Si el UPDATE falla, borra el archivo antes de propagar: una fila con
        audio_path NULL deja el archivo fuera del alcance de la purga por TTL.
        Ya no es un huérfano permanente (M5, review PR #15): AmbientStore
        ahora barre huérfanos por mtime en cada purge_expired() (ver
        `_sweep_orphans`), así que el borrado inmediato acá acota la ventana
        de exposición a prácticamente cero en vez de dejarlo colgado hasta
        que alguien note el archivo suelto — el sweep de todos modos lo
        habría alcanzado, esto solo evita esperar el TTL entero.

        Args:
            room_id: Habitación de origen.
            utt_id: rowid devuelto por store.add().
            audio: Audio multicanal del segmento.
        """
        if source is not None and not self._privacy.sources_available((source,)):
            return
        options = {'source': source} if source is not None else {}
        path = await self._archiver.write(room_id, utt_id, audio, **options)
        if not path:
            return
        try:
            await self._store.set_audio_path(utt_id, path)
        except BaseException:
            # A cancelled/failed attach also retains the durable file locator.
            discard = asyncio.create_task(self._archiver._discard(path))
            while not discard.done():
                try:
                    await asyncio.shield(discard)
                except asyncio.CancelledError:
                    pass
            discard.result()
            raise

    async def _purge_worker(self) -> None:
        while self._running:
            try:
                self._store.expiry_changed.clear()
                await self._store.purge_expired()
                # I3 (review PR #15): AudioArchiver.stats se incrementaba pero
                # nadie lo leía ni lo logueaba — write-only, cero visibilidad
                # para la campaña de medición. Loguear en cada ciclo de purga
                # (1/hora) es suficiente cadencia sin agregar un endpoint.
                if self._archiver is not None and self._archiver.enabled:
                    logger.info("AudioArchiver stats: %s", self._archiver.stats)
                if not self._running:
                    return
                delay = await self._store.next_expiration_delay()
                try:
                    async with asyncio.timeout(delay):
                        await self._store.expiry_changed.wait()
                except TimeoutError:
                    pass
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("AmbientTranscriber: purga falló (sigo)")
                await asyncio.sleep(min(_PURGE_INTERVAL_S, 60.0))


@dataclass
class AmbientPath:
    """Grafo armado del ambient path — lo que main.py necesita."""

    tap: object
    transcriber: AmbientTranscriber
    store: object
    distiller: object | None
    _closing: asyncio.Task | None = field(default=None, init=False, repr=False)

    async def start(self) -> None:
        """Arrancar: store → workers → job del distiller (si hay memoria)."""
        if self._closing is not None:
            raise RuntimeError("ambient_path_closed")
        await self.store.init()
        await self.transcriber.start()
        if self.distiller is not None:
            # En _tasks del transcriber: su stop() cancela TODO de una.
            self.transcriber._tasks.append(
                asyncio.create_task(self.distiller.run_forever())
            )

    async def stop(self) -> None:
        """Apagar en orden inverso; seguro de llamar sin start() previo."""
        await asyncio.shield(self.begin_close())

    def begin_close(self) -> asyncio.Task:
        """Retain actual distillation and store cleanup after observer cancellation."""
        if self._closing is None:
            distilling = self.distiller.begin_close() if self.distiller is not None else None
            async def finish():
                if callable(getattr(type(self.transcriber), "begin_close", None)):
                    await self.transcriber.begin_close()
                else:
                    await self.transcriber.stop()
                if distilling is not None:
                    await distilling
                await self.store.close()
            self._closing = asyncio.create_task(finish())
        return self._closing


def _build_ambient_stt_engine(stt_cfg: dict, stt_base_cfg: dict):
    """Seleccionar el motor ASR del ambient path según ``ambient.stt.engine``.

    - ``parakeet`` (swap 2026-06-07): Parakeet-TDT-0.6B-v3 onnx en CPU —
      0/5 alucinaciones sobre no-voz vs 5/5 del turbo en el benchmark A/B
      con audio real (doc 2026-06-07_SOTA_ASR_ESPANOL_INVESTIGACION.md).
    - ``whisper`` (default, compat): FastWhisperSTT dedicado en cuda:0,
      hereda model/compute_type del bloque ``stt:`` top-level.
    """
    engine = stt_cfg.get("engine", "whisper")
    if engine == "parakeet":
        from src.stt.parakeet_stt import ParakeetSTT

        return ParakeetSTT(
            model_name=stt_cfg.get("parakeet_model", "nemo-parakeet-tdt-0.6b-v3"),
            language=stt_base_cfg.get("language", "es"),
        )
    from src.stt.whisper_fast import FastWhisperSTT

    return FastWhisperSTT(
        model=stt_cfg.get("model", stt_base_cfg.get("model", "./models/whisper-v3-turbo")),
        device=stt_cfg.get("device", "cuda:0"),
        compute_type=stt_cfg.get("compute_type", stt_base_cfg.get("compute_type", "int8_float16")),
        language=stt_base_cfg.get("language", "es"),
        beam_size=stt_cfg.get("beam_size", 1),
        initial_prompt=None,
        vad_filter=False,
    )


def build_ambient_path(
    ambient_cfg: dict,
    stt_base_cfg: dict,
    room_ids: list[str],
    store_fact_fn: Callable[..., str] | None,
    privacy=None,
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
    from src.ambient.audio_archive import AudioArchiver
    from src.ambient.distiller import Distiller, make_langid_fn, make_local_chat_fn
    from src.ambient.doa import DoAEstimator
    from src.ambient.language_quality import make_quality_fn
    from src.ambient.segmenter import UtteranceSegmenter, make_silero_predictor
    from src.ambient.source_classifier import SourceClassifier, SourceClassifierConfig
    from src.ambient.speaker_tagger import SpeakerTagger
    from src.ambient.store import AmbientStore
    from src.ambient.tap import MultiChannelTap
    from src.users.speaker_identifier import SpeakerIdentifier

    stt_cfg = ambient_cfg.get("stt", {}) or {}
    ambient_engine = _build_ambient_stt_engine(stt_cfg, stt_base_cfg)
    ambient_stt = AmbientSTT(stt=ambient_engine, asr_col=ambient_cfg.get("asr_col", 1),
                             source_policy=privacy.ledger if privacy is not None else None)

    seg_cfg = ambient_cfg.get("segmenter", {}) or {}

    def segmenter_factory() -> UtteranceSegmenter:
        # Predictor Silero POR ROOM (review final 2026-06-06): el modelo es
        # stateful (GRU) y el segmenter lo resetea en cada límite de utterance.
        # Compartirlo entre rooms haría que el cierre de una room corrompa el
        # estado de la otra a mitad de utterance. El modelo pesa ~2MB: N copias
        # es trivial frente al bug.
        return UtteranceSegmenter(
            vad_predict=make_silero_predictor(),
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
        source_policy=privacy.ledger if privacy is not None else None,
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

    ka_cfg = ambient_cfg.get("keep_audio", {}) or {}
    store = AmbientStore(
        db_path=ambient_cfg.get("db_path", "./data/ambient.db"),
        retention_hours=ambient_cfg.get("retention_hours", 48.0),
        source_policy=privacy.ledger if privacy is not None else None,
        clock=privacy.ledger.clock if privacy is not None else None,
        audio_dir=ka_cfg.get("dir", "./data/ambient_audio"),
    )

    # Regla "español conservable" (flag-no-drop): marca cada utterance con
    # (lang, lang_prob, lang_ok) al persistir. py3langid + marcadores rioplatenses
    # corren en el hot-path del worker ambient (CPU, determinista, liviano). El
    # discriminante de calidad del rioplatense es el IDIOMA DEL TEXTO, no la
    # energía (vad). min_vad se pasa por compat y language_quality lo ignora;
    # el distiller sigue usando distill.min_vad_prob.
    q_cfg = ambient_cfg.get("quality", {}) or {}
    quality_fn = None
    if q_cfg.get("enabled", False):
        quality_fn = make_quality_fn(
            make_langid_fn(),
            min_len=q_cfg.get("min_len", 8),
            min_vad=q_cfg.get("min_vad", 0.45),
        )

    archiver = AudioArchiver(
        base_dir=ka_cfg.get("dir", "./data/ambient_audio"),
        enabled=bool(ka_cfg.get("enabled", False)),
        min_free_bytes=int(ka_cfg.get("min_free_mb", 1000)) * 1_000_000,
        discard_audio=store.discard_audio,
        privacy=privacy,
        store=store,
    )
    store.audio_inflight = archiver.is_writing

    tap = MultiChannelTap(clock=privacy.ledger.clock if privacy is not None else None)
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
        quality_fn=quality_fn,
        archiver=archiver,
        privacy=privacy,
    )

    distiller = None
    dis_cfg = ambient_cfg.get("distill", {}) or {}
    # Spec §6 Fase 2: en shadow_mode se etiqueta y persiste pero NO se
    # destila a memoria — el distiller arranca recién en Fase 3 (flip).
    if store_fact_fn is not None and not ambient_cfg.get("shadow_mode", True):
        distiller = Distiller(
            store=store,
            chat_fn=make_local_chat_fn(
                llm_url=dis_cfg.get("llm_url", "http://127.0.0.1:8101/v1"),
                model=dis_cfg.get("model", "local"),
                destination=dis_cfg.get("destination", "cloud"),
            ),
            store_fact_fn=store_fact_fn,
            interval_hours=dis_cfg.get("interval_hours", 6.0),
            min_batch=dis_cfg.get("min_batch", 5),
            max_batch_chars=dis_cfg.get("max_batch_chars", 12000),
            min_vad_prob=dis_cfg.get("min_vad_prob", 0.0),
            spanish_only=dis_cfg.get("spanish_only", False),
            lang_detect_fn=(
                make_langid_fn()
                if (dis_cfg.get("log_language", False) or dis_cfg.get("drop_language"))
                else None
            ),
            drop_language=dis_cfg.get("drop_language"),
            drop_language_min_prob=dis_cfg.get("drop_language_min_prob", 0.9),
            privacy=privacy,
        )

    path = AmbientPath(tap=tap, transcriber=transcriber, store=store, distiller=distiller)
    if privacy is not None:
        privacy.add_borrower(path.begin_close)
    return path
