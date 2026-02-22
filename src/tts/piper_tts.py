"""
Text-to-Speech Module
TTS rápido con Piper y opción de alta calidad con XTTS
Soporta streaming para menor latencia percibida.
"""

from __future__ import annotations

import logging
import time
import threading
import queue
from pathlib import Path
from typing import Optional, Generator, Callable
import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class StreamingAudioPlayer:
    """
    Reproductor de audio en streaming con buffering adaptativo.

    Garantiza reproducción suave sin cortes mediante:
    - Pre-buffering: acumula audio antes de empezar
    - Buffer circular: absorbe variaciones de velocidad
    - Latencia configurable vs suavidad
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        channels: int = 1,
        device: int = None,
        buffer_ms: int = 150,        # Buffer total en ms
        prebuffer_ms: int = 30,      # Pre-buffer antes de iniciar (OPTIMIZADO: 30ms)
        block_size: int = 512        # Tamaño de bloque de audio
    ):
        """
        Args:
            sample_rate: Sample rate del audio
            channels: Número de canales
            device: Dispositivo de salida (None = default)
            buffer_ms: Tamaño del buffer circular en ms
            prebuffer_ms: Audio a acumular antes de iniciar reproducción
            block_size: Tamaño de bloque para el stream
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
        self.block_size = block_size

        # Calcular tamaños en samples
        self.buffer_samples = int(sample_rate * buffer_ms / 1000)
        self.prebuffer_samples = int(sample_rate * prebuffer_ms / 1000)

        # Buffer circular thread-safe
        self._buffer = np.zeros(self.buffer_samples, dtype=np.float32)
        self._write_pos = 0
        self._read_pos = 0
        self._samples_available = 0
        self._lock = threading.Lock()

        # Control
        self._stream = None
        self._running = False
        self._started_playback = False
        self._finished_writing = False
        self._playback_complete = threading.Event()

        # Stats
        self._underruns = 0
        self._total_samples_written = 0

    def _audio_callback(self, outdata, frames, time_info, status):
        """Callback para sounddevice - lee del buffer circular"""
        if status.output_underflow:
            self._underruns += 1
            logger.warning(f"Audio underrun #{self._underruns}")

        with self._lock:
            if self._samples_available >= frames:
                # Leer del buffer circular
                end_pos = self._read_pos + frames
                if end_pos <= self.buffer_samples:
                    outdata[:, 0] = self._buffer[self._read_pos:end_pos]
                else:
                    # Wrap around
                    first_part = self.buffer_samples - self._read_pos
                    outdata[:first_part, 0] = self._buffer[self._read_pos:]
                    outdata[first_part:, 0] = self._buffer[:frames - first_part]

                self._read_pos = end_pos % self.buffer_samples
                self._samples_available -= frames

            elif self._finished_writing and self._samples_available > 0:
                # Últimos samples
                available = self._samples_available
                end_pos = self._read_pos + available
                if end_pos <= self.buffer_samples:
                    outdata[:available, 0] = self._buffer[self._read_pos:end_pos]
                else:
                    first_part = self.buffer_samples - self._read_pos
                    outdata[:first_part, 0] = self._buffer[self._read_pos:]
                    outdata[first_part:available, 0] = self._buffer[:available - first_part]

                outdata[available:, 0] = 0
                self._samples_available = 0
                self._playback_complete.set()
                raise sd.CallbackStop()

            elif self._finished_writing:
                # No más audio
                outdata.fill(0)
                self._playback_complete.set()
                raise sd.CallbackStop()
            else:
                # Buffer underrun - llenar con silencio
                outdata.fill(0)

    def _write_to_buffer(self, audio: np.ndarray):
        """Escribir audio al buffer circular"""
        samples = len(audio)

        with self._lock:
            # Verificar espacio disponible
            space_available = self.buffer_samples - self._samples_available
            if samples > space_available:
                # Buffer lleno - esperar o descartar
                samples = space_available
                if samples == 0:
                    return
                audio = audio[:samples]

            # Escribir al buffer circular
            end_pos = self._write_pos + samples
            if end_pos <= self.buffer_samples:
                self._buffer[self._write_pos:end_pos] = audio
            else:
                # Wrap around
                first_part = self.buffer_samples - self._write_pos
                self._buffer[self._write_pos:] = audio[:first_part]
                self._buffer[:samples - first_part] = audio[first_part:]

            self._write_pos = end_pos % self.buffer_samples
            self._samples_available += samples
            self._total_samples_written += samples

    def _start_stream(self):
        """Iniciar stream de audio"""
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            device=self.device,
            callback=self._audio_callback,
            blocksize=self.block_size,
            latency='low'
        )
        self._stream.start()
        self._started_playback = True
        logger.debug(f"Stream iniciado (buffer: {self._samples_available} samples)")

    def feed(self, audio_chunk: np.ndarray):
        """
        Alimentar chunk de audio al buffer.
        Inicia reproducción automáticamente cuando hay suficiente buffer.
        """
        if not self._running:
            return

        # Asegurar float32
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        # Escribir al buffer
        self._write_to_buffer(audio_chunk)

        # Iniciar reproducción cuando tengamos suficiente pre-buffer
        if not self._started_playback and self._samples_available >= self.prebuffer_samples:
            self._start_stream()

    def start(self):
        """Preparar para recibir audio"""
        self._running = True
        self._started_playback = False
        self._finished_writing = False
        self._playback_complete.clear()
        self._write_pos = 0
        self._read_pos = 0
        self._samples_available = 0
        self._underruns = 0
        self._total_samples_written = 0

    def finish(self):
        """Señalar fin del audio y esperar reproducción completa"""
        self._finished_writing = True

        # Si nunca empezamos a reproducir (audio muy corto), iniciar ahora
        if not self._started_playback and self._samples_available > 0:
            self._start_stream()

        # Esperar que termine la reproducción
        if self._started_playback:
            self._playback_complete.wait(timeout=30.0)

        self.stop()

        # Log stats
        if self._underruns > 0:
            logger.warning(f"Streaming completado con {self._underruns} underruns")
        else:
            logger.debug(f"Streaming completado sin underruns ({self._total_samples_written} samples)")

    def stop(self):
        """Detener stream inmediatamente"""
        self._running = False
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def play_stream(
        self,
        audio_generator: Generator[np.ndarray, None, None],
        on_first_chunk: Callable = None
    ):
        """
        Reproducir audio desde un generador en streaming.

        Args:
            audio_generator: Generador que yield chunks de audio
            on_first_chunk: Callback cuando llega el primer chunk (para medir latencia)
        """
        self.start()
        first_chunk = True

        try:
            for chunk in audio_generator:
                if first_chunk:
                    first_chunk = False
                    if on_first_chunk:
                        on_first_chunk()

                self.feed(chunk)

                # Pequeña pausa si el buffer está muy lleno para no bloquear
                with self._lock:
                    if self._samples_available > self.buffer_samples * 0.9:
                        time.sleep(0.01)

            self.finish()

        except Exception as e:
            logger.error(f"Error en streaming: {e}")
            self.stop()
            raise

    def get_stats(self) -> dict:
        """Obtener estadísticas del streaming"""
        return {
            "total_samples": self._total_samples_written,
            "underruns": self._underruns,
            "buffer_size_ms": self.buffer_samples * 1000 / self.sample_rate,
            "prebuffer_ms": self.prebuffer_samples * 1000 / self.sample_rate
        }


class PiperTTS:
    """
    Text-to-Speech ultra-rápido con Piper
    Latencia: ~50-80ms
    """
    
    def __init__(
        self,
        model: str = "es_ES-davefx-medium.onnx",
        sample_rate: int = 22050
    ):
        self.model_path = model
        self.sample_rate = sample_rate
        self._voice = None
    
    def load(self, warmup: bool = True):
        """
        Cargar modelo Piper.

        Args:
            warmup: Si True, ejecuta síntesis dummy para calentar el modelo (reduce latencia de primer uso)
        """
        try:
            from piper import PiperVoice

            logger.info(f"Cargando Piper: {self.model_path}")
            start = time.time()

            self._voice = PiperVoice.load(self.model_path)

            elapsed = time.time() - start
            logger.info(f"Piper cargado en {elapsed:.1f}s")

            # Warmup: síntesis dummy para pre-calentar el modelo ONNX
            if warmup:
                self._warmup()

        except ImportError:
            logger.error("Piper no instalado: pip install piper-tts")
            raise

    def _warmup(self):
        """
        Pre-calentar el modelo con síntesis dummy.
        Reduce latencia del primer uso real de ~80ms a ~50ms.
        """
        if self._voice is None:
            return

        try:
            t_warmup = time.perf_counter()
            # Texto corto para warmup rápido
            _ = self._voice.synthesize("Hola")
            warmup_ms = (time.perf_counter() - t_warmup) * 1000
            logger.debug(f"Piper warmup completado en {warmup_ms:.0f}ms")
        except Exception as e:
            logger.warning(f"Piper warmup falló (no crítico): {e}")
    
    def synthesize(self, text: str) -> tuple[np.ndarray, float]:
        """
        Sintetizar texto a audio
        
        Args:
            text: Texto a sintetizar
        
        Returns:
            (audio_array, tiempo_ms)
        """
        if self._voice is None:
            self.load()
        
        start = time.perf_counter()
        
        audio = self._voice.synthesize(text)
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug(f"Piper TTS ({elapsed_ms:.0f}ms): {text[:30]}...")
        
        return np.array(audio), elapsed_ms
    
    def synthesize_stream(self, text: str) -> Generator[np.ndarray, None, None]:
        """
        Sintetizar texto a audio en streaming (chunks).

        Args:
            text: Texto a sintetizar

        Yields:
            Chunks de audio como numpy arrays
        """
        if self._voice is None:
            self.load()

        # Piper soporta streaming nativo
        for audio_chunk in self._voice.synthesize_stream_raw(text):
            # Convertir bytes a numpy array (int16 -> float32)
            chunk_array = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
            yield chunk_array

    def speak(self, text: str, blocking: bool = True):
        """Sintetizar y reproducir por parlantes"""
        audio, _ = self.synthesize(text)
        sd.play(audio, samplerate=self.sample_rate)
        if blocking:
            sd.wait()

    def speak_stream(
        self,
        text: str,
        device: int = None,
        buffer_ms: int = 150,
        prebuffer_ms: int = 30
    ):
        """
        Sintetizar y reproducir en streaming (menor latencia).

        El audio comienza a sonar mientras se sigue generando.

        Args:
            text: Texto a sintetizar
            device: Dispositivo de salida (None = default)
            buffer_ms: Tamaño del buffer (mayor = más suave, menor = menos latencia)
            prebuffer_ms: Pre-buffer antes de iniciar (latencia inicial vs suavidad)
        """
        player = StreamingAudioPlayer(
            sample_rate=self.sample_rate,
            device=device,
            buffer_ms=buffer_ms,
            prebuffer_ms=prebuffer_ms
        )

        start_time = time.perf_counter()
        first_audio_time = [None]

        def on_first_chunk():
            first_audio_time[0] = time.perf_counter()

        player.play_stream(self.synthesize_stream(text), on_first_chunk=on_first_chunk)

        # Log latencia
        if first_audio_time[0]:
            latency_ms = (first_audio_time[0] - start_time) * 1000
            stats = player.get_stats()
            logger.info(
                f"TTS Streaming: latencia={latency_ms:.0f}ms, "
                f"underruns={stats['underruns']}, "
                f"prebuffer={prebuffer_ms}ms"
            )

    def get_stream_generator(self, text: str) -> Generator[np.ndarray, None, None]:
        """
        Obtener generador de audio para uso externo (ej: zone manager).

        Args:
            text: Texto a sintetizar

        Returns:
            Generador que yield chunks de audio
        """
        return self.synthesize_stream(text)

    def save(self, text: str, output_path: str | Path) -> float:
        """Sintetizar y guardar a archivo"""
        import soundfile as sf
        
        audio, elapsed_ms = self.synthesize(text)
        sf.write(str(output_path), audio, self.sample_rate)
        
        return elapsed_ms


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
        speaker_wav: Optional[str] = None
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
        speaker_wav: Optional[str] = None
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


class KokoroTTS:
    """
    TTS rapido con Kokoro-82M (~1.5GB VRAM).
    Para respuestas cortas de domotica: "Listo", "Luz encendida", etc.
    https://github.com/hexgrad/kokoro
    """

    def __init__(
        self,
        model: str = "hexgrad/Kokoro-82M",
        device: str = "cuda:3",
        default_voice: str = "af_heart",
    ):
        self.model_name = model
        self.device = device
        self.default_voice = default_voice
        self._model = None
        self.sample_rate = 24000

    def load(self):
        """Cargar Kokoro"""
        try:
            import kokoro

            logger.info(f"Cargando Kokoro TTS: {self.model_name}")
            start = time.time()
            self._model = kokoro.Pipeline(device=self.device)
            elapsed = time.time() - start
            logger.info(f"Kokoro cargado en {elapsed:.1f}s")
        except ImportError:
            logger.error("Kokoro no instalado: pip install kokoro")
            raise

    def synthesize(self, text: str, voice: str = None) -> tuple[np.ndarray, float]:
        """Sintetizar con Kokoro"""
        if self._model is None:
            self.load()

        voice = voice or self.default_voice
        start = time.perf_counter()
        audio, _ = self._model(text, voice=voice)
        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.debug(f"Kokoro TTS ({elapsed_ms:.0f}ms): {text[:30]}...")
        return audio, elapsed_ms

    def synthesize_stream(self, text: str) -> Generator[np.ndarray, None, None]:
        """Sintetizar en streaming (Kokoro genera rapido, yield en chunks)."""
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


class Qwen3TTS:
    """
    TTS conversacional con Qwen3-TTS y voice cloning (~4.5GB VRAM).
    Para respuestas largas con voz natural y clonada.
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen3-TTS-0.6B",
        device: str = "cuda:3",
        speaker_wav: Optional[str] = None,
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

            self._processor = AutoProcessor.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
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


class DualTTS:
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
        self.kokoro = KokoroTTS(**(kokoro_config or {}))
        self.qwen3 = Qwen3TTS(**(qwen3_config or {}))
        self.quality_threshold = quality_threshold
        self.sample_rate = 24000  # Both engines use 24kHz

    def load(self):
        """Cargar ambos motores (comparten GPU 3)."""
        self.kokoro.load()
        self.qwen3.load()

    def _select_engine(self, text: str) -> str:
        """Seleccionar motor segun longitud del texto."""
        if len(text) <= self.quality_threshold:
            return "kokoro"
        return "qwen3"

    def synthesize(
        self, text: str, force_quality: bool = False
    ) -> tuple[np.ndarray, float, str]:
        """
        Sintetizar eligiendo motor automaticamente.

        Returns:
            (audio, tiempo_ms, motor_usado)
        """
        if force_quality or self._select_engine(text) == "qwen3":
            audio, elapsed = self.qwen3.synthesize(text)
            return audio, elapsed, "qwen3"

        audio, elapsed = self.kokoro.synthesize(text)
        return audio, elapsed, "kokoro"

    def synthesize_stream(self, text: str) -> Generator[np.ndarray, None, None]:
        """Streaming delegado al motor seleccionado."""
        engine = self._select_engine(text)
        if engine == "qwen3":
            yield from self.qwen3.synthesize_stream(text)
        else:
            yield from self.kokoro.synthesize_stream(text)

    def speak(self, text: str, blocking: bool = True):
        """Sintetizar y reproducir."""
        audio, _, engine = self.synthesize(text)
        sd.play(audio, samplerate=self.sample_rate)
        if blocking:
            sd.wait()


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
