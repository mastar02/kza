"""ParakeetSTT — NVIDIA Parakeet-TDT-0.6B-v3 vía onnx-asr (CPU).

Motor ASR alternativo al faster-whisper turbo. Origen: swap del AMBIENT path
(2026-06-07, benchmark A/B con audio real XVF3800: 0/5 alucinaciones sobre
no-voz vs 5/5 del turbo, mejor calidad sobre voz, RTF ~0.03 en CPU). Movido de
``src/ambient/`` a ``src/stt/`` para que también lo pueda seleccionar el COMMAND
path (fast path) vía ``create_stt``, sin que el STT del fast path dependa del
módulo ``ambient``. Razones de arquitectura:
- Transducer frame-synchronous: no "free-runea" texto como el decoder
  autoregresivo de Whisper → sin "Gracias por ver el video." sobre silencio
  ni repetition loops (entrenado con 36kh de no-voz/target-vacío,
  arxiv 2509.14128).
- Corre en CPU (onnxruntime): 0 VRAM. GPU descartado para el fast path — el
  decoder TDT autoregresivo en onnxruntime-gpu (loop Python) rinde PEOR que
  CPU en clips cortos (RTF ~0.1 vs ~0.03); ver plan Parakeet fast path.

Duck-type de FastWhisperSTT: expone ``transcribe_with_confidence(audio) ->
STTResult`` y ``transcribe(audio) -> (text, ms)``. Parakeet no produce
no_speech_prob/avg_logprob/compression_ratio → None ('sin penalizar' en el
CommandGate, cuyas reglas duras son todas por texto); la señal de calidad del
ambient path es ``vad_prob`` (Silero), no el STT.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
import logging
import threading
from contextvars import Context
import time

import numpy as np

from src.stt.whisper_fast import STTResult

logger = logging.getLogger(__name__)


class _ParakeetLease:
    def __init__(self, owner):
        self.owner = owner
        self.thread = threading.current_thread()
        self.active = True

class _ParakeetLifecycle:
    """Retain this adapter's real operations without owning its callers' data."""

    def _parakeet_initialize(self) -> None:
        self._parakeet_condition = threading.Condition()
        self._parakeet_local = threading.local()
        self._parakeet_leases = []
        self._parakeet_sealed = False
        self._parakeet_close_started = False
        self._parakeet_close_owner = None
        self._parakeet_close_done = threading.Event()
        self._parakeet_failure = None
        self._parakeet_closing = None

    @property
    def _parakeet_active(self) -> int:
        return sum(lease.active for lease in self._parakeet_leases)

    def _parakeet_current(self):
        lease = getattr(self._parakeet_local, "lease", None)
        return lease if (lease is not None and lease.owner is self
                         and lease.thread is threading.current_thread()
                         and lease.active) else None

    @contextmanager
    def _parakeet_admit(self):
        lease = self._parakeet_current()
        owned = lease is None
        if owned:
            lease = _ParakeetLease(self)
        previous = getattr(self._parakeet_local, "lease", None)
        try:
            if owned:
                with self._parakeet_condition:
                    if self._parakeet_sealed:
                        raise RuntimeError("parakeet_closed")
                    self._parakeet_leases.append(lease)
            self._parakeet_local.lease = lease
            yield
        finally:
            # Logical exit precedes fallible TLS restoration and notification.
            if owned:
                lease.active = False
            try:
                self._parakeet_local.lease = previous
            finally:
                if owned:
                    with self._parakeet_condition:
                        self._parakeet_leases = [
                            active for active in self._parakeet_leases if active.active
                        ]
                        self._parakeet_condition.notify_all()

    def _parakeet_require_close_caller(self) -> None:
        current = threading.current_thread()
        with self._parakeet_condition:
            if any(lease.active and lease.thread is current
                   for lease in self._parakeet_leases):
                raise RuntimeError("parakeet_close_from_worker")
            if (self._parakeet_close_owner is current
                    and not self._parakeet_close_done.is_set()):
                raise RuntimeError("parakeet_close_from_owner")

    def request_close(self) -> None:
        """Seal admission using memory only, without waiting for model locks."""
        with self._parakeet_condition:
            self._parakeet_sealed = True
            self._parakeet_condition.notify_all()

    def wait_closed(self) -> None:
        """Drain actual operations and drop only this adapter's Python references."""
        self._parakeet_require_close_caller()
        self.request_close()
        with self._parakeet_condition:
            own_close = not self._parakeet_close_started
            if own_close:
                self._parakeet_close_started = True
                self._parakeet_close_owner = threading.current_thread()
        if own_close:
            failure = None
            try:
                with self._parakeet_condition:
                    while self._parakeet_active:
                        self._parakeet_condition.wait(timeout=0.05)
                self._parakeet_release_resources()
            except BaseException:
                failure = "parakeet_close_failed"
            with self._parakeet_condition:
                self._parakeet_failure = failure
                self._parakeet_close_owner = None
                self._parakeet_close_done.set()
        else:
            self._parakeet_close_done.wait()
        if self._parakeet_failure is not None:
            raise RuntimeError(self._parakeet_failure)

    def close(self) -> None:
        """Synchronous retirement; async callers use begin_close or aclose."""
        self.wait_closed()

    def begin_close(self) -> asyncio.Task:
        """Return the sole retained owner; never cancel this owner directly."""
        self._parakeet_require_close_caller()
        loop = asyncio.get_running_loop()
        self.request_close()
        with self._parakeet_condition:
            if self._parakeet_closing is None:
                async def finish():
                    # Publish the task outside the condition even with eager tasks.
                    await asyncio.sleep(0)
                    await asyncio.to_thread(self.wait_closed)

                self._parakeet_closing = loop.create_task(finish(), context=Context())
                self._parakeet_closing.add_done_callback(
                    lambda done: None if done.cancelled() else done.exception(),
                    context=Context(),
                )
            return self._parakeet_closing

    async def aclose(self) -> None:
        """Keep the real retirement owner through repeated observer cancellation."""
        closing = self.begin_close()
        cancelled = False
        while not closing.done():
            try:
                await asyncio.shield(closing)
            except asyncio.CancelledError:
                cancelled = True
        closing.result()
        if cancelled:
            raise asyncio.CancelledError


class ParakeetSTT(_ParakeetLifecycle):
    """ASR Parakeet-TDT vía onnx-asr (CPU), carga lazy del modelo."""

    def __init__(
        self,
        model_name: str = "nemo-parakeet-tdt-0.6b-v3",
        language: str = "es",
    ):
        """
        Args:
            model_name: Modelo del hub de onnx-asr (descarga/caché en
                ~/.cache/huggingface la primera vez).
            language: Hint de idioma para recognize(). ⚠️ NO-OP con Parakeet
                TDT: en onnx-asr solo la clase AED (Canary, p.ej.
                ``nemo-canary-1b-v2``) consume el kwarg; el TDT lo ignora y
                autodetecta por diseño — no hay forma de forzar idioma
                (NVIDIA remite a Canary para eso). Con SNR bajo la
                autodetección deriva a garble inglés; el gate real es por
                ``vad_prob`` aguas abajo (language_quality / textual_wake),
                no acá. Se conserva el parámetro porque sí aplicaría si
                ``model_name`` fuera un Canary.
        """
        self.model_name = model_name
        self.language = language
        self._model = None
        self._parakeet_initialize()
        self._model_lock = threading.RLock()

    def load(self) -> None:
        """Cargar el modelo onnx (idempotente)."""
        with self._parakeet_admit():
            with self._model_lock:
                if self._model is not None:
                    return
                import onnx_asr

                logger.info(f"Cargando Parakeet (onnx, CPU): {self.model_name}")
                start = time.time()
                self._model = onnx_asr.load_model(self.model_name)
                logger.info(f"Parakeet cargado en {time.time() - start:.1f}s")

    def transcribe_with_confidence(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> STTResult:
        """Transcribir audio mono → STTResult (señales Whisper en None).

        Args:
            audio: Array mono float32 [-1, 1] o int16/int32 (se normaliza).
            sample_rate: Solo 16000 (contrato del pipeline).

        Returns:
            STTResult con text y elapsed_ms; no_speech_prob/avg_logprob/
            compression_ratio en None (Parakeet no los expone).
        """
        with self._parakeet_admit():
            with self._model_lock:
                if self._model is None:
                    self.load()

                if audio.dtype == np.int16:
                    audio = audio.astype(np.float32) / 32768.0
                elif audio.dtype == np.int32:
                    audio = audio.astype(np.float32) / 2147483648.0
                elif audio.dtype != np.float32:
                    audio = audio.astype(np.float32)

                start = time.perf_counter()
                text = self._model.recognize(audio, language=self.language) or ""
                elapsed_ms = (time.perf_counter() - start) * 1000
                Context().run(logger.debug, "Parakeet STT completed in %.0fms", elapsed_ms)

                return STTResult(text=text.strip(), elapsed_ms=elapsed_ms)

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> tuple[str, float]:
        """Transcribir audio a texto. Firma compat (text, elapsed_ms).

        Espejo de ``FastWhisperSTT.transcribe``: el command path lo usa en el
        early-parse (``multi_room_audio_loop._early_parse_worker``), que solo
        necesita (texto, ms) sin las señales de confianza.
        """
        with self._parakeet_admit():
            r = self.transcribe_with_confidence(audio, sample_rate)
            return r.text, r.elapsed_ms

    def _parakeet_release_resources(self) -> None:
        # Release only this adapter reference, not peer resources or SDK pools.
        with self._model_lock:
            self._model = None
