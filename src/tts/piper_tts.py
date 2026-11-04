"""
Text-to-Speech Module
TTS rápido con Piper y opción de alta calidad con XTTS
Soporta streaming para menor latencia percibida.
"""

from __future__ import annotations

import logging
import math
import time
import threading
from enum import StrEnum
from pathlib import Path
from typing import Generator, Callable
import numpy as np
import sounddevice as sd

from src.core.adapter_lifecycle import _AdapterLifecycle

logger = logging.getLogger(__name__)


class PlaybackState(StrEnum):
    """Lifecycle of one PCM stream, independent of the text producer."""

    IDLE = "idle"
    PLAYING = "playing"
    DRAINING = "draining"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class AudioPlaybackError(RuntimeError):
    """The output stopped or failed to make progress before its deadline."""


class StreamingAudioPlayer:
    """Fixed-size PCM ring with complete writes and cancelable backpressure.

    The producer runs outside the asyncio loop. The device callback only copies
    available frames, pads silence and notifies space; it never waits for data.
    Normal completion waits for the device's finished callback, including tail
    buffers. Cancellation aborts and wakes blocked writers immediately.
    """

    def __init__(
        self, sample_rate: int = 22050, channels: int = 1, device: int = None,
        buffer_ms: int = 150, prebuffer_ms: int = 30, block_size: int = 512,
        *, stall_timeout_s: float = 1.0, drain_timeout_s: float | None = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        if any(type(value) is not int or value <= 0 for value in (sample_rate, channels, buffer_ms, block_size)):
            raise ValueError("Invalid PCM buffer configuration")
        if type(prebuffer_ms) is not int or not 0 <= prebuffer_ms <= buffer_ms:
            raise ValueError("Prebuffer must fit inside the PCM buffer")
        if drain_timeout_s is None:
            drain_timeout_s = max(2.0, buffer_ms / 1000 + 1.0)
        if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in (stall_timeout_s, drain_timeout_s)):
            raise ValueError("Playback deadlines must be finite and positive")
        self.sample_rate, self.channels, self.device = sample_rate, channels, device
        self.block_size = block_size
        self.buffer_samples = max(1, int(sample_rate * buffer_ms / 1000))
        self.prebuffer_samples = int(sample_rate * prebuffer_ms / 1000)
        self.stall_timeout_s, self.drain_timeout_s = stall_timeout_s, drain_timeout_s
        self._cancel_event = cancel_event
        self._buffer = np.zeros((self.buffer_samples, channels), dtype=np.float32)
        self._condition = threading.Condition()
        self._lock = self._condition
        self._device_lock = threading.Lock()
        self._stream = None
        self._state = PlaybackState.IDLE
        self._running = False
        self._started_playback = False
        self._finished_writing = False
        self._playback_complete = threading.Event()
        self._generation = 0
        self._producer_owner = None
        self._error = None
        self._write_pos = self._read_pos = self._samples_available = 0
        self._underruns = self._total_samples_written = self._played_samples = 0
        self._max_buffered = 0
        self._software_underruns = 0

    def _cancelled(self) -> bool:
        return not self._running or (self._cancel_event is not None and self._cancel_event.is_set())

    @property
    def is_playing(self) -> bool:
        """Whether the device has started and still owns audible output."""
        with self._condition:
            return self._started_playback and self._state in {PlaybackState.PLAYING, PlaybackState.DRAINING}

    def _fail(self, reason: str) -> None:
        with self._condition:
            self._error = reason
            self._state = PlaybackState.FAILED
            self._running = False
            self._playback_complete.set()
            self._condition.notify_all()

    def _audio_callback(self, outdata, frames, time_info, status) -> None:
        outdata.fill(0)
        with self._condition:
            if self._cancelled():
                raise sd.CallbackAbort()
            if status.output_underflow:
                self._underruns += 1
            count = min(frames, self._samples_available)
            if count < frames and not self._finished_writing:
                self._software_underruns += 1
            first = min(count, self.buffer_samples - self._read_pos)
            outdata[:first, :] = self._buffer[self._read_pos:self._read_pos + first]
            if first < count:
                outdata[first:count, :] = self._buffer[:count - first]
            self._read_pos = (self._read_pos + count) % self.buffer_samples
            self._samples_available -= count
            self._played_samples += count
            self._condition.notify_all()
            if self._finished_writing and self._samples_available == 0:
                # finished_callback, not this callback, proves device draining.
                raise sd.CallbackStop()

    def _driver_finished(self) -> None:
        with self._condition:
            if self._state in {PlaybackState.PLAYING, PlaybackState.DRAINING}:
                if self._cancel_event is not None and self._cancel_event.is_set():
                    self._state = PlaybackState.CANCELLED
                    self._running = False
                elif self._finished_writing and self._samples_available == 0:
                    self._state = PlaybackState.COMPLETED
                    self._running = False
                else:
                    self._state = PlaybackState.FAILED
                    self._error = "audio_device_stopped"
                    self._running = False
            self._playback_complete.set()
            self._condition.notify_all()

    def _start_stream(self) -> None:
        with self._device_lock:
            with self._condition:
                if self._started_playback or self._cancelled():
                    return
            stream = sd.OutputStream(
                samplerate=self.sample_rate, channels=self.channels, device=self.device,
                callback=self._audio_callback, finished_callback=self._driver_finished,
                blocksize=self.block_size, dtype="float32", latency="low",
            )
            with self._condition:
                self._stream = stream
                if self._cancelled():
                    stream.close()
                    self._stream = None
                    return
                self._started_playback = True
            try:
                stream.start()
            except Exception:
                self._fail("audio_device_start_failed")
                # The caller's finally closes this registered handle.
                raise

    def _close_stream(self, *, abort: bool) -> None:
        with self._device_lock:
            with self._condition:
                stream, self._stream = self._stream, None
            if stream is not None:
                try:
                    stream.abort() if abort else stream.stop()
                finally:
                    stream.close()

    def _write_to_buffer(self, audio: np.ndarray) -> bool:
        values = np.asarray(audio, dtype=np.float32)
        if values.ndim == 1 and self.channels == 1:
            values = values[:, None]
        if values.ndim != 2 or values.shape[1] != self.channels or not np.isfinite(values).all():
            raise ValueError("PCM must contain finite frames with the configured channels")
        offset = 0
        with self._condition:
            generation = self._generation
        last_progress = time.monotonic()
        while offset < len(values):
            start_stream = False
            with self._condition:
                if self._error is not None:
                    raise AudioPlaybackError(self._error)
                if self._cancelled() or generation != self._generation:
                    return False
                if self._finished_writing:
                    raise RuntimeError("Cannot feed a finished PCM stream")
                space = self.buffer_samples - self._samples_available
                if space:
                    count = min(space, len(values) - offset)
                    first = min(count, self.buffer_samples - self._write_pos)
                    self._buffer[self._write_pos:self._write_pos + first] = values[offset:offset + first]
                    if first < count:
                        self._buffer[:count - first] = values[offset + first:offset + count]
                    self._write_pos = (self._write_pos + count) % self.buffer_samples
                    self._samples_available += count
                    self._total_samples_written += count
                    self._max_buffered = max(self._max_buffered, self._samples_available)
                    offset += count
                    last_progress = time.monotonic()
                    start_stream = not self._started_playback and self._samples_available >= self.prebuffer_samples
                elif not self._started_playback:
                    start_stream = True
                else:
                    remaining = self.stall_timeout_s - (time.monotonic() - last_progress)
                    if remaining <= 0:
                        self._fail("audio_buffer_stalled")
                        raise AudioPlaybackError("audio_buffer_stalled")
                    self._condition.wait(timeout=min(remaining, 0.05))
            if start_stream:
                self._start_stream()
        with self._condition:
            if self._error is not None:
                raise AudioPlaybackError(self._error)
            return not self._cancelled()

    def feed(self, audio_chunk: np.ndarray) -> bool:
        """Write every frame, waiting for space until drained, failed or cancelled."""
        return self._write_to_buffer(audio_chunk)

    def start(self) -> bool:
        """Prepare a fresh session; an already-cancelled external token stays cancelled."""
        with self._condition:
            if self._producer_owner is not None and self._producer_owner != threading.get_ident():
                raise RuntimeError("Previous PCM producer is still stopping")
            if self._running or self._stream is not None:
                raise RuntimeError("PCM stream is already active")
            self._generation += 1
            self._state = PlaybackState.PLAYING
            self._running = True
            self._started_playback = self._finished_writing = False
            self._playback_complete.clear()
            self._error = None
            self._write_pos = self._read_pos = self._samples_available = 0
            self._underruns = self._total_samples_written = self._played_samples = 0
            self._max_buffered = 0
            self._software_underruns = 0
            if self._cancel_event is not None and self._cancel_event.is_set():
                self._state = PlaybackState.CANCELLED
                self._running = False
                self._playback_complete.set()
            return self._running

    def finish(self) -> bool:
        """Drain queued frames and device tail; never turn a timeout into success."""
        try:
            with self._condition:
                if self._error:
                    raise AudioPlaybackError(self._error)
                if self._cancelled():
                    return False
                self._finished_writing = True
                self._state = PlaybackState.DRAINING
                has_audio = self._samples_available > 0
                started = self._started_playback
                if not started and not has_audio:
                    self._state = PlaybackState.COMPLETED
                    self._running = False
                    self._playback_complete.set()
            if has_audio and not started:
                self._start_stream()
            deadline = time.monotonic() + self.drain_timeout_s
            while not self._playback_complete.is_set():
                if self._cancel_event is not None and self._cancel_event.is_set():
                    self.stop()
                    return False
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._fail("audio_drain_timeout")
                    break
                self._playback_complete.wait(timeout=min(remaining, 0.05))
            if self._error:
                raise AudioPlaybackError(self._error)
            return self._state is PlaybackState.COMPLETED
        finally:
            self._close_stream(abort=self._state is not PlaybackState.COMPLETED)

    def stop(self) -> None:
        """Abort playback and wake blocked writers without draining old audio."""
        with self._condition:
            self._running = False
            if self._state not in {PlaybackState.COMPLETED, PlaybackState.FAILED}:
                self._state = PlaybackState.CANCELLED
            self._generation += 1
            self._playback_complete.set()
            self._condition.notify_all()
        self._close_stream(abort=True)

    def play_stream(self, audio_generator: Generator[np.ndarray, None, None], on_first_chunk: Callable = None) -> bool:
        """Own a generator through bounded playback and close it on every exit."""
        iterator = iter(audio_generator)
        first = True
        completed = False
        owned = False
        exhausted = object()
        try:
            with self._condition:
                if self._producer_owner is not None:
                    raise RuntimeError("PCM producer is already active")
                self._producer_owner = threading.get_ident()
                owned = True
            if not self.start():
                return False
            while not self._cancelled():
                chunk = next(iterator, exhausted)
                if chunk is exhausted:
                    break
                if np.size(chunk) == 0:
                    if not self.feed(chunk):
                        return False
                    continue
                if first:
                    first = False
                    if on_first_chunk is not None:
                        on_first_chunk()
                if not self.feed(chunk):
                    return False
            completed = self.finish()
            return completed
        except Exception:
            if owned and self._state is not PlaybackState.CANCELLED and self._error is None:
                self._fail("audio_stream_failed")
            raise
        finally:
            try:
                if owned and not completed:
                    self.stop()
            finally:
                try:
                    close = getattr(iterator, "close", None)
                    if close is not None:
                        close()
                finally:
                    if owned:
                        with self._condition:
                            self._producer_owner = None

    def get_stats(self) -> dict:
        """Read frame accounting and terminal state without copying audio."""
        with self._condition:
            return {
                "total_samples": self._total_samples_written, "played_samples": self._played_samples,
                "queued_samples": self._samples_available, "max_buffered_samples": self._max_buffered,
                "software_underruns": self._software_underruns,
                "underruns": self._underruns, "state": self._state.value,
                "buffer_size_ms": self.buffer_samples * 1000 / self.sample_rate,
                "prebuffer_ms": self.prebuffer_samples * 1000 / self.sample_rate,
            }


class PiperTTS(_AdapterLifecycle):
    """
    Text-to-Speech ultra-rápido con Piper
    Latencia: ~50-80ms
    """

    def __init__(self, model: str='es_ES-davefx-medium.onnx', sample_rate: int=22050):
        self.model_path = model
        self.sample_rate = sample_rate
        self._adapter_initialize('tts')
        self._voice = None

    def load(self, warmup: bool=True):
        """
        Cargar modelo Piper.

        Args:
            warmup: Si True, ejecuta síntesis dummy para calentar el modelo (reduce latencia de primer uso)
        """
        with self._adapter_admit():
            with self._adapter_lock:
                try:
                    try:
                        from piper import PiperVoice
                        logger.info(f'Cargando Piper: {self.model_path}')
                        start = time.time()
                        self._voice = PiperVoice.load(self.model_path)
                        elapsed = time.time() - start
                        logger.info(f'Piper cargado en {elapsed:.1f}s')
                        if warmup:
                            self._warmup()
                    except ImportError:
                        logger.error('Piper no instalado: pip install piper-tts')
                        raise
                finally:
                    PiperVoice = None

    def _warmup(self):
        """
        Pre-calentar el modelo con síntesis dummy.
        Reduce latencia del primer uso real de ~80ms a ~50ms.
        """
        with self._adapter_admit():
            with self._adapter_lock:
                try:
                    if self._voice is None:
                        return
                    try:
                        t_warmup = time.perf_counter()
                        _ = self._voice.synthesize('Hola')
                        warmup_ms = (time.perf_counter() - t_warmup) * 1000
                        logger.debug(f'Piper warmup completado en {warmup_ms:.0f}ms')
                    except Exception as e:
                        logger.warning(f'Piper warmup falló (no crítico): {e}')
                finally:
                    _ = None

    def synthesize(self, text: str) -> tuple[np.ndarray, float]:
        """
        Sintetizar texto a audio
        
        Args:
            text: Texto a sintetizar
        
        Returns:
            (audio_array, tiempo_ms)
        """
        with self._adapter_admit():
            with self._adapter_lock:
                try:
                    if self._voice is None:
                        self.load()
                    start = time.perf_counter()
                    audio = self._voice.synthesize(text)
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    logger.debug(f'Piper TTS ({elapsed_ms:.0f}ms): {text[:30]}...')
                    return (np.array(audio), elapsed_ms)
                finally:
                    audio = text = None

    def synthesize_stream(self, text: str) -> Generator[np.ndarray, None, None]:
        """
        Sintetizar texto a audio en streaming (chunks).

        Args:
            text: Texto a sintetizar

        Yields:
            Chunks de audio como numpy arrays
        """
        source = iterator = audio_chunk = chunk_array = None
        exhausted = False
        with self._adapter_stream_admit() as lease:
            try:
                with self._adapter_entered(lease):
                    with self._adapter_lock:
                        if self._voice is None:
                            self.load()
                        source = self._voice.synthesize_stream_raw(text)
                        self._adapter_track_iterator(source)
                        iterator = iter(source)
                        self._adapter_track_iterator(iterator)
                while True:
                    with self._adapter_entered(lease):
                        with self._adapter_lock:
                            try:
                                audio_chunk = next(iterator)
                            except StopIteration:
                                exhausted = True
                                break
                            chunk_array = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
                            audio_chunk = None
                    yield chunk_array
            finally:
                try:
                    with self._adapter_entered(lease):
                        with self._adapter_lock:
                            try:
                                self._adapter_close_iterator(iterator, exhausted=exhausted)
                            finally:
                                if source is not iterator:
                                    self._adapter_close_iterator(source, exhausted=exhausted)
                finally:
                    source = iterator = audio_chunk = chunk_array = text = None

    def speak(self, text: str, blocking: bool=True):
        """Sintetizar y reproducir por parlantes"""
        with self._adapter_admit():
            try:
                audio, _ = self.synthesize(text)
                sd.play(audio, samplerate=self.sample_rate)
                if blocking:
                    sd.wait()
            finally:
                audio = _ = text = None

    def speak_stream(self, text: str, device: int=None, buffer_ms: int=150, prebuffer_ms: int=30, *, on_player: Callable | None=None, cancel_event: threading.Event | None=None, stall_timeout_s: float=1.0, drain_timeout_s: float | None=None) -> bool:
        """
        Sintetizar y reproducir en streaming (menor latencia).

        El audio comienza a sonar mientras se sigue generando.

        Args:
            text: Texto a sintetizar
            device: Dispositivo de salida (None = default)
            buffer_ms: Tamaño del buffer (mayor = más suave, menor = menos latencia)
            prebuffer_ms: Pre-buffer antes de iniciar (latencia inicial vs suavidad)
        """
        with self._adapter_admit():
            try:
                player = StreamingAudioPlayer(sample_rate=self.sample_rate, device=device, buffer_ms=buffer_ms, prebuffer_ms=prebuffer_ms, cancel_event=cancel_event, stall_timeout_s=stall_timeout_s, drain_timeout_s=drain_timeout_s)
                if on_player is not None:
                    try:
                        on_player(player)
                    except Exception:
                        player.stop()
                        raise
                start_time = time.perf_counter()
                first_audio_time = [None]

                def on_first_chunk():
                    first_audio_time[0] = time.perf_counter()
                completed = player.play_stream(self.synthesize_stream(text), on_first_chunk=on_first_chunk)
                if first_audio_time[0]:
                    latency_ms = (first_audio_time[0] - start_time) * 1000
                    stats = player.get_stats()
                    logger.info(f'TTS Streaming: latencia={latency_ms:.0f}ms, underruns={stats['underruns']}, prebuffer={prebuffer_ms}ms')
                return completed
            finally:
                text = player = on_player = cancel_event = None

    def get_stream_generator(self, text: str) -> Generator[np.ndarray, None, None]:
        """
        Obtener generador de audio para uso externo (ej: zone manager).

        Args:
            text: Texto a sintetizar

        Returns:
            Generador que yield chunks de audio
        """
        with self._adapter_admit():
            try:
                return self.synthesize_stream(text)
            finally:
                text = None

    def save(self, text: str, output_path: str | Path) -> float:
        """Sintetizar y guardar a archivo"""
        with self._adapter_admit():
            try:
                import soundfile as sf
                audio, elapsed_ms = self.synthesize(text)
                sf.write(str(output_path), audio, self.sample_rate)
                return elapsed_ms
            finally:
                audio = text = sf = None

    def _adapter_release_resources(self) -> None:
        """Drop this instance's voice after actual operations/iterators retire."""
        with self._adapter_lock:
            self._voice = None


class XTTS:
    """
    Text-to-Speech de alta calidad con XTTS-v2
    Soporta clonación de voz
    Latencia: ~1-2s
    """
    
    def __init__(
        self,
        model: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        device: str = "cuda:3",
        speaker_wav: str | None = None
    ):
        self.model_name = model
        self.device = device
        self.speaker_wav = speaker_wav
        self._tts = None
        self.sample_rate = 24000  # XTTS usa 24kHz
    
    def load(self):
        """Cargar modelo XTTS"""
        try:
            from TTS.api import TTS
            
            logger.info(f"Cargando XTTS: {self.model_name}")
            start = time.time()
            
            self._tts = TTS(self.model_name).to(self.device)
            
            elapsed = time.time() - start
            logger.info(f"XTTS cargado en {elapsed:.1f}s")
        except ImportError:
            logger.error("TTS no instalado: pip install TTS")
            raise
    
    def synthesize(
        self,
        text: str,
        language: str = "es",
        speaker_wav: str | None = None
    ) -> tuple[np.ndarray, float]:
        """
        Sintetizar texto a audio
        
        Args:
            text: Texto a sintetizar
            language: Código de idioma
            speaker_wav: Audio para clonar voz (opcional)
        
        Returns:
            (audio_array, tiempo_ms)
        """
        if self._tts is None:
            self.load()
        
        start = time.perf_counter()
        
        speaker = speaker_wav or self.speaker_wav
        
        if speaker:
            audio = self._tts.tts(
                text=text,
                language=language,
                speaker_wav=speaker
            )
        else:
            audio = self._tts.tts(
                text=text,
                language=language
            )
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug(f"XTTS ({elapsed_ms:.0f}ms): {text[:30]}...")
        
        return np.array(audio), elapsed_ms
    
    def speak(self, text: str, language: str = "es", blocking: bool = False):
        """Sintetizar y reproducir"""
        audio, _ = self.synthesize(text, language)
        sd.play(audio, samplerate=self.sample_rate)
        if blocking:
            sd.wait()
    
    def save(
        self,
        text: str,
        output_path: str | Path,
        language: str = "es"
    ) -> float:
        """Sintetizar y guardar"""
        import soundfile as sf
        
        audio, elapsed_ms = self.synthesize(text, language)
        sf.write(str(output_path), audio, self.sample_rate)
        
        return elapsed_ms
    
    def set_speaker(self, speaker_wav: str):
        """Configurar voz a clonar"""
        self.speaker_wav = speaker_wav


class _TTSRetirementError(RuntimeError):
    """A fixed lifecycle failure, without retaining a native exception."""


class _TTSLifecycle(_AdapterLifecycle):
    """Keep legacy TTS contracts on the shared identity/retirement owner."""

    def _tts_initialize(self):
        self._adapter_initialize('tts')

    @property
    def _tts_active(self) -> int:
        return self._adapter_active

    @property
    def _tts_sealed(self) -> bool:
        return self._adapter_sealed

    @property
    def _tts_retained_iterators(self):
        return tuple(self._adapter_retained_iterators.values())

    def _tts_current(self):
        return self._adapter_current()

    def _tts_admit(self):
        return self._adapter_admit()

    def _tts_stream_admit(self):
        return self._adapter_stream_admit()

    def _tts_entered(self, lease):
        return self._adapter_entered(lease)

    def _tts_require_close_caller(self):
        self._adapter_require_close_caller()

    def _tts_close_iterator(self, source, *, exhausted: bool=False):
        self._adapter_close_iterator(source, exhausted=exhausted)

    def _tts_require_retired_iterators(self):
        if self._adapter_retained_iterators:
            raise _TTSRetirementError('tts_iterator_close_failed')

    def _adapter_release_resources(self):
        self._tts_release_resources()

    def _adapter_close_failure_code(self, phase: str, error: BaseException) -> str:
        if phase == 'iterators' and self._adapter_retained_iterators:
            return 'tts_iterator_close_failed'
        if phase == 'resource' and type(error) is _TTSRetirementError:
            args = error.args
            if len(args) == 1 and type(args[0]) is str:
                if args[0] == 'tts_iterator_close_failed':
                    return 'tts_iterator_close_failed'
                if args[0] == 'tts_backend_close_unsupported':
                    return 'tts_backend_close_unsupported'
        return 'tts_close_failed'


class KokoroTTS(_TTSLifecycle):
    """
    TTS rapido con Kokoro-82M (~1.5GB VRAM).
    Para respuestas cortas de domotica: "Listo", "Luz encendida", etc.
    https://github.com/hexgrad/kokoro
    """

    def __init__(
        self,
        model: str = "hexgrad/Kokoro-82M",
        device: str = "cuda:3",
        default_voice: str = "ef_dora",
        lang_code: str = "e",
    ):
        self.model_name = model
        self.device = device
        self.default_voice = default_voice
        self.lang_code = lang_code
        self._model = None
        self.sample_rate = 24000
        self._tts_initialize()
        self._model_lock = threading.RLock()

    def load(self):
        """Load once under native ownership; never reload a sealed instance."""
        with self._tts_admit():
            with self._model_lock:
                if self._model is None:
                    self._load_model()

    def _load_model(self):
        try:
            import kokoro

            logger.info(
                f"Cargando Kokoro TTS: {self.model_name} "
                f"(lang_code={self.lang_code!r}, voice={self.default_voice!r})"
            )
            start = time.time()
            # Kokoro >= 0.7 exige KPipeline(lang_code=..., device=...).
            # lang_code 'e' = español.
            self._model = kokoro.KPipeline(lang_code=self.lang_code, device=self.device)
            elapsed = time.time() - start
            logger.info(f"Kokoro cargado en {elapsed:.1f}s")
        except ImportError:
            logger.error("Kokoro no instalado: pip install kokoro")
            raise
        except Exception as e:
            logger.error(
                f"KokoroTTS load failed (non-ImportError): {type(e).__name__}: {e}",
                exc_info=True,
            )
            raise

    def synthesize(self, text: str, voice: str=None) -> tuple[np.ndarray, float]:
        """Sintetizar con Kokoro.

        KPipeline.__call__ devuelve un generator de Result (un chunk por
        segmento según split_pattern). Concatenamos los chunks y pasamos
        el audio a np.ndarray.
        """
        raw_source = source = result = chunk = chunks = audio = None
        exhausted = False
        try:
            with self._tts_admit():
                try:
                    with self._model_lock:
                        if self._model is None:
                            self.load()
                        voice = voice or self.default_voice
                        start = time.perf_counter()
                        chunks = []
                        try:
                            raw_source = self._model(text, voice=voice)
                            if raw_source is not None:
                                self._adapter_track_iterator(raw_source)
                            source = iter(raw_source)
                            self._adapter_track_iterator(source)
                            while True:
                                try:
                                    result = next(source)
                                except StopIteration:
                                    exhausted = True
                                    break
                                chunk = result.audio
                                if chunk is None:
                                    continue
                                if hasattr(chunk, 'cpu'):
                                    chunk = chunk.cpu().numpy()
                                chunks.append(chunk)
                            if not chunks:
                                audio = np.zeros(0, dtype=np.float32)
                            elif len(chunks) == 1:
                                audio = chunks[0]
                            else:
                                audio = np.concatenate(chunks)
                            elapsed_ms = (time.perf_counter() - start) * 1000
                            logger.debug('Kokoro TTS completed in %.0fms', elapsed_ms)
                            return (audio, elapsed_ms)
                        finally:
                            try:
                                self._tts_close_iterator(source, exhausted=exhausted)
                            finally:
                                if raw_source is not source:
                                    self._tts_close_iterator(raw_source, exhausted=exhausted)
                finally:
                    raw_source = text = voice = source = result = chunk = chunks = audio = _ = None
        finally:
            raw_source = text = voice = source = result = chunk = chunks = audio = _ = None

    def synthesize_stream(self, text: str) -> Generator[np.ndarray, None, None]:
        """Admit on first next; retain each unchanged 4096-sample PCM view."""
        lease = audio = chunk = None
        try:
            with self._tts_stream_admit() as lease:
                try:
                    with self._tts_entered(lease):
                        audio, _ = self.synthesize(text)
                    text = None
                    for i in range(0, len(audio), 4096):
                        with self._tts_entered(lease):
                            chunk = audio[i:i + 4096]
                        yield chunk
                finally:
                    try:
                        with self._tts_entered(lease):
                            text = audio = chunk = _ = None
                    finally:
                        text = audio = chunk = _ = None
        finally:
            text = audio = chunk = _ = None

    def speak(self, text: str, blocking: bool=True):
        """Sintetizar y reproducir"""
        audio = None
        try:
            with self._tts_admit():
                try:
                    audio, _ = self.synthesize(text)
                    sd.play(audio, samplerate=self.sample_rate)
                    if blocking:
                        sd.wait()
                finally:
                    text = audio = _ = None
        finally:
            text = audio = _ = None

    def _tts_release_resources(self):
        self._tts_require_retired_iterators()
        with self._model_lock:
            self._model = None


class Qwen3TTS:
    """
    TTS conversacional con Qwen3-TTS y voice cloning (~4.5GB VRAM).
    Para respuestas largas con voz natural y clonada.
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen3-TTS-0.6B",
        device: str = "cuda:3",
        speaker_wav: str | None = None,
    ):
        self.model_name = model
        self.device = device
        self.speaker_wav = speaker_wav
        self._model = None
        self._processor = None
        self._speaker_embedding = None
        self.sample_rate = 24000

    def load(self):
        """Cargar Qwen3-TTS con transformers"""
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor

            logger.info(f"Cargando Qwen3-TTS: {self.model_name}")
            start = time.time()

            self._processor = AutoProcessor.from_pretrained(  # nosec B615 -- Qwen3TTS dormido, no muerto: create_tts pasa {} y DualTTS.__init__ deja self.qwen3=None; se re-arma descomentando settings.yaml:255-258 (requiere 3ra GPU)
                self.model_name, trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(  # nosec B615 -- ídem: dormido, se re-arma descomentando settings.yaml:255-258 (requiere 3ra GPU), no muerto
                self.model_name, trust_remote_code=True
            ).to(self.device)

            elapsed = time.time() - start
            logger.info(f"Qwen3-TTS cargado en {elapsed:.1f}s")

            # Pre-cargar speaker embedding si hay audio de referencia
            if self.speaker_wav and Path(self.speaker_wav).exists():
                self._load_speaker_embedding()

        except ImportError:
            logger.error("transformers no instalado: pip install transformers")
            raise

    def _load_speaker_embedding(self):
        """Pre-cargar embedding del speaker de referencia."""
        try:
            import soundfile as sf

            audio, sr = sf.read(self.speaker_wav)
            self._speaker_embedding = audio
            logger.info(f"Speaker reference loaded: {self.speaker_wav}")
        except Exception as e:
            logger.warning(f"Failed to load speaker reference: {e}")

    def synthesize(self, text: str) -> tuple[np.ndarray, float]:
        """Sintetizar texto a audio con Qwen3-TTS."""
        if self._model is None:
            self.load()

        start = time.perf_counter()

        inputs = self._processor(text=text, return_tensors="pt").to(self.device)

        with __import__("torch").no_grad():
            outputs = self._model.generate(**inputs, max_new_tokens=2048)

        audio = self._processor.decode(outputs[0])
        if isinstance(audio, np.ndarray):
            audio_array = audio
        else:
            audio_array = np.array(audio, dtype=np.float32)

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug(f"Qwen3 TTS ({elapsed_ms:.0f}ms): {text[:30]}...")
        return audio_array, elapsed_ms

    def synthesize_stream(self, text: str) -> Generator[np.ndarray, None, None]:
        """Sintetizar en streaming."""
        audio, _ = self.synthesize(text)
        chunk_size = 4096
        for i in range(0, len(audio), chunk_size):
            yield audio[i:i + chunk_size]

    def speak(self, text: str, blocking: bool = True):
        """Sintetizar y reproducir"""
        audio, _ = self.synthesize(text)
        sd.play(audio, samplerate=self.sample_rate)
        if blocking:
            sd.wait()


class HybridTTS:
    """
    TTS híbrido: Piper para respuestas rápidas, XTTS para calidad
    """
    
    def __init__(
        self,
        piper_model: str = "es_ES-davefx-medium.onnx",
        xtts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        xtts_device: str = "cuda:3",
        quality_threshold: int = 100  # Caracteres
    ):
        self.piper = PiperTTS(piper_model)
        self.xtts = XTTS(xtts_model, xtts_device)
        self.quality_threshold = quality_threshold
        self._piper_loaded = False
        self._xtts_loaded = False
    
    def load_fast(self):
        """Cargar solo Piper (rápido)"""
        if not self._piper_loaded:
            self.piper.load()
            self._piper_loaded = True
    
    def load_quality(self):
        """Cargar XTTS (calidad)"""
        if not self._xtts_loaded:
            self.xtts.load()
            self._xtts_loaded = True
    
    def synthesize(
        self,
        text: str,
        force_quality: bool = False
    ) -> tuple[np.ndarray, float, str]:
        """
        Sintetizar eligiendo automáticamente el motor
        
        Args:
            text: Texto a sintetizar
            force_quality: Forzar uso de XTTS
        
        Returns:
            (audio, tiempo_ms, motor_usado)
        """
        # Usar XTTS para textos largos o si se fuerza calidad
        if force_quality or len(text) > self.quality_threshold:
            self.load_quality()
            audio, elapsed = self.xtts.synthesize(text)
            return audio, elapsed, "xtts"
        else:
            self.load_fast()
            audio, elapsed = self.piper.synthesize(text)
            return audio, elapsed, "piper"
    
    def speak(self, text: str, force_quality: bool = False, blocking: bool = False):
        """Sintetizar y reproducir"""
        audio, _, engine = self.synthesize(text, force_quality)
        sample_rate = self.xtts.sample_rate if engine == "xtts" else self.piper.sample_rate
        sd.play(audio, samplerate=sample_rate)
        if blocking:
            sd.wait()


class DualTTS(_TTSLifecycle):
    """
    TTS dual: Kokoro para fast path, Qwen3 para conversacional.

    Routing automatico por longitud de texto:
    - len(text) <= threshold -> Kokoro (rapido, ~30ms)
    - len(text) > threshold  -> Qwen3 (conversacional, voice cloning)
    """

    def __init__(
        self,
        kokoro_config: dict = None,
        qwen3_config: dict = None,
        quality_threshold: int = 50,
    ):
        self._tts_initialize()
        self.kokoro = KokoroTTS(**(kokoro_config or {}))
        self.qwen3 = Qwen3TTS(**(qwen3_config or {})) if qwen3_config else None
        self._owned_tts_engines = (self.kokoro,) + ((self.qwen3,) if self.qwen3 is not None else ())
        self.quality_threshold = quality_threshold
        self.sample_rate = 24000  # Both engines use 24kHz

    def load(self):
        """Cargar motores TTS disponibles."""
        with self._tts_admit():
            self.kokoro.load()
            if self.qwen3:
                try:
                    self.qwen3.load()
                except Exception as e:
                    logger.warning(f"Qwen3-TTS no disponible, usando solo Kokoro: {e}")
                    self.qwen3 = None

    def _select_engine(self, text: str) -> str:
        """Seleccionar motor segun longitud del texto."""
        if not self.qwen3 or len(text) <= self.quality_threshold:
            return "kokoro"
        return "qwen3"

    def synthesize(self, text: str, force_quality: bool=False) -> tuple[np.ndarray, float, str]:
        """
        Sintetizar eligiendo motor automaticamente.

        Returns:
            (audio, tiempo_ms, motor_usado)
        """
        audio = None
        try:
            with self._tts_admit():
                try:
                    if self.qwen3 and (force_quality or self._select_engine(text) == 'qwen3'):
                        audio, elapsed = self.qwen3.synthesize(text)
                        return (audio, elapsed, 'qwen3')
                    audio, elapsed = self.kokoro.synthesize(text)
                    return (audio, elapsed, 'kokoro')
                finally:
                    text = audio = _ = None
        finally:
            text = audio = _ = None

    def synthesize_stream(self, text: str) -> Generator[np.ndarray, None, None]:
        """Admit on first next; retain both returned source and its iterator."""
        lease = raw_source = source = chunk = engine = None
        exhausted = False
        try:
            with self._tts_stream_admit() as lease:
                try:
                    with self._tts_entered(lease):
                        engine = self.qwen3 if self._select_engine(text) == 'qwen3' and self.qwen3 else self.kokoro
                        raw_source = engine.synthesize_stream(text)
                        if raw_source is not None:
                            self._adapter_track_iterator(raw_source)
                        source = iter(raw_source)
                        self._adapter_track_iterator(source)
                    text = None
                    while True:
                        try:
                            with self._tts_entered(lease):
                                chunk = next(source)
                        except StopIteration:
                            exhausted = True
                            break
                        yield chunk
                finally:
                    try:
                        with self._tts_entered(lease):
                            try:
                                self._tts_close_iterator(source, exhausted=exhausted)
                            finally:
                                if raw_source is not source:
                                    self._tts_close_iterator(raw_source, exhausted=exhausted)
                    finally:
                        text = raw_source = source = chunk = engine = None
        finally:
            text = raw_source = source = chunk = engine = None

    def speak(self, text: str, blocking: bool=True):
        """Sintetizar y reproducir."""
        audio = None
        try:
            with self._tts_admit():
                try:
                    audio, _, _ = self.synthesize(text)
                    sd.play(audio, samplerate=self.sample_rate)
                    if blocking:
                        sd.wait()
                finally:
                    text = audio = _ = None
        finally:
            text = audio = _ = None

    def _adapter_require_close_caller(self) -> None:
        super()._adapter_require_close_caller()
        # Validate reviewed owned children in the original caller, before an
        # async bridge can reuse a suspended stream's latest consumer Thread.
        # Children remain open until all admitted parent delegation has drained.
        for engine in self._owned_tts_engines:
            if isinstance(engine, _TTSLifecycle):
                engine._adapter_require_close_caller()

    def _tts_release_resources(self):
        self._tts_require_retired_iterators()
        unsupported = False
        # A parent lease may still be about to enter a child. Seal children only
        # after the parent drain, preserving already admitted delegation/streams.
        for engine in self._owned_tts_engines:
            if isinstance(engine, _TTSLifecycle):
                if (
                    type(engine).close is _AdapterLifecycle.close
                    and type(engine).wait_closed is _AdapterLifecycle.wait_closed
                    and "close" not in vars(engine)
                    and "wait_closed" not in vars(engine)
                ):
                    engine._adapter_wait_closed()
                else:
                    engine.close()
            else:
                unsupported = True
        if unsupported:
            # Optional Qwen3 has no reviewed retirement API in this unit.
            # Retain its references and report failure instead of claiming closure.
            raise _TTSRetirementError("tts_backend_close_unsupported")
        if not any(self.kokoro is engine for engine in self._owned_tts_engines):
            self.kokoro = None
        if not any(self.qwen3 is engine for engine in self._owned_tts_engines):
            self.qwen3 = None
        self._owned_tts_engines = ()


def create_tts(config: dict) -> PiperTTS | XTTS | HybridTTS | DualTTS:
    """Factory para crear TTS según configuración"""
    engine = config.get("engine", "piper")
    
    if engine == "piper":
        piper_config = config.get("piper", {})
        return PiperTTS(
            model=piper_config.get("model", "es_ES-davefx-medium.onnx"),
            sample_rate=piper_config.get("sample_rate", 22050)
        )
    elif engine == "xtts":
        xtts_config = config.get("xtts", {})
        return XTTS(
            model=xtts_config.get("model", "tts_models/multilingual/multi-dataset/xtts_v2"),
            device=xtts_config.get("device", "cuda:3"),
            speaker_wav=xtts_config.get("speaker_wav")
        )
    elif engine == "hybrid":
        return HybridTTS(
            piper_model=config.get("piper", {}).get("model"),
            xtts_model=config.get("xtts", {}).get("model"),
            xtts_device=config.get("xtts", {}).get("device", "cuda:3")
        )
    elif engine == "dual":
        return DualTTS(
            kokoro_config=config.get("kokoro", {}),
            qwen3_config=config.get("qwen3", {}),
            quality_threshold=config.get("quality_threshold", 50),
        )
    else:
        raise ValueError(f"Engine TTS desconocido: {engine}")
