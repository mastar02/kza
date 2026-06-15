"""
Speech-to-Text Module
Transcripción de audio ultra-rápida con Whisper

Optimizado para latencia mínima:
- Procesamiento 100% en RAM (sin I/O de disco)
- VAD agresivo para reducir silencios
- Configuración de beam_size=1 para velocidad
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class STTResult:
    """Resultado de transcripción con confianza del STT.

    no_speech_prob / avg_logprob / compression_ratio son None cuando no hay
    segmentos (audio vacío) o el motor no los expone (Moonshine). El gate
    trata None como 'sin penalizar'.

    ⚠️ no_speech_prob con large-v3-turbo está DEGENERADO: devuelve ~1e-10
    siempre, incluso sobre silencio absoluto que alucina "Gracias por ver el
    video." (reproducido en aislamiento 2026-06-07, CPU int8, faster-whisper
    1.2.1). El head de no-speech se perdió en la destilación del turbo. NO
    usarlo como señal anti-alucinación: usar avg_logprob, compression_ratio
    y el VAD externo (Silero — `vad_prob` del ambient path).

    compression_ratio es el MÁXIMO entre segmentos (no la media): un solo
    segmento basura repetitiva debe poder disparar el guard anti-alucinación
    del CommandAcceptanceGate (openai/whisper #2378).
    """

    text: str
    elapsed_ms: float
    no_speech_prob: float | None = None
    avg_logprob: float | None = None
    compression_ratio: float | None = None


class FastWhisperSTT:
    """Speech-to-Text con Faster-Whisper optimizado para baja latencia"""

    def __init__(
        self,
        model: str = "distil-whisper/distil-small.en",
        device: str = "cuda:0",
        compute_type: str = "float16",
        language: str = "es",
        beam_size: int = 1,
        best_of: int = 1,
        initial_prompt: str | None = None,
        vad_filter: bool = True,
        temperature_fallback: bool = False,
        fallback_temperatures: list[float] | None = None,
        compression_ratio_threshold: float = 2.0,
        log_prob_threshold: float = -3.0,
    ):
        """
        Args:
            beam_size: 1 = greedy (rápido, ~200ms turbo). 5 = beam search
                (+30% latencia, +5-10% precisión en palabras raras).
            best_of: Muestras paralelas (con temperature>0). 1 = determinístico.
            initial_prompt: Texto que sesga la decodificación. Útil para
                enseñarle a Whisper palabras novel ("Nexa") y vocabulario
                del dominio (verbos y rooms). Máx ~224 tokens.
            vad_filter: Pre-filtro Silero interno de faster-whisper. Para
                audio del XVF3800 va en False: el chip YA hace VAD/NS/
                beamforming por hardware y Silero lee prob~0 sobre su salida
                — borraba capturas ENTERAS → Text='' con voz real
                (confirmado en prod 2026-06-04).
            temperature_fallback: Habilitar re-decodificación adaptativa de
                Whisper a temperatura creciente cuando el segmento resulta
                garble (compression_ratio alto). En turbo, no_speech_prob y
                avg_logprob están degenerados → el fallback se gatea SOLO por
                compression_ratio_threshold. False = escalar 0 (comportamiento
                anterior, sin fallback).
            fallback_temperatures: Lista de temperaturas a probar en orden.
                Solo se usa cuando temperature_fallback=True.
            compression_ratio_threshold: Umbral de compression_ratio para
                disparar fallback. Por defecto 2.0 (valor openai/whisper).
            log_prob_threshold: Umbral de avg_logprob. Se deja muy bajo (-3.0)
                para que NO dispare fallback espurio por el logprob invertido
                del turbo; el gating real lo hace compression_ratio.
        """
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self.best_of = best_of
        self.initial_prompt = initial_prompt
        self.vad_filter = vad_filter
        self.temperature_fallback = temperature_fallback
        self.fallback_temperatures = fallback_temperatures or [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        self.compression_ratio_threshold = compression_ratio_threshold
        self.log_prob_threshold = log_prob_threshold
        self._model = None
        # Thread dedicado del modelo (fix 2026-06-07): CUDA/CTranslate2
        # mantienen estado per-thread; con DOS WhisperModel en GPUs distintas
        # (command path cuda:1 + ambient cuda:0) ejecutados desde el pool
        # default compartido (asyncio.to_thread / run_in_executor(None)), el
        # reuso de threads alternando devices producía `RuntimeError: CUDA
        # failed with error invalid argument` intermitente (12×/1.5h en prod,
        # 2026-06-06, SOLO con ambos modelos activos). Pin: cada instancia
        # carga y ejecuta su modelo SIEMPRE en su propio thread.
        self._executor: ThreadPoolExecutor | None = None
        self._worker_ident: int | None = None

    def _run_pinned(self, fn, *args):
        """Ejecutar fn en el thread dedicado del modelo (bloqueante)."""
        if threading.get_ident() == self._worker_ident:
            return fn(*args)  # re-entrada desde el propio worker
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix=f"whisper-{self.device}"
            )
            self._worker_ident = self._executor.submit(
                threading.get_ident
            ).result()
        return self._executor.submit(fn, *args).result()

    def load(self):
        """Cargar modelo en GPU (idempotente, en el thread dedicado)."""
        self._run_pinned(self._load_impl)

    def _load_impl(self):
        """Cargar modelo en GPU (idempotente).

        Idempotencia crítica: `load()` se invoca desde más de un sitio sobre
        la MISMA instancia compartida (warmup vía transcribe() lazy, y
        CommandProcessor.__init__ explícito). Sin este guard, la 2da llamada
        construía un segundo WhisperModel mientras el primero seguía
        referenciado en self._model → pico 2× VRAM → `CUDA out of memory` en
        cuda:1 al reiniciar el service (2026-05-29). Con el guard, la 2da
        llamada es no-op.
        """
        if self._model is not None:
            return

        from faster_whisper import WhisperModel

        logger.info(f"Cargando Whisper: {self.model_name} en {self.device}")
        start = time.time()
        
        self._model = WhisperModel(
            self.model_name,
            device="cuda" if "cuda" in self.device else self.device,
            device_index=int(self.device.split(":")[-1]) if ":" in self.device else 0,
            compute_type=self.compute_type
        )
        
        elapsed = time.time() - start
        logger.info(f"Whisper cargado en {elapsed:.1f}s")
    
    def _transcribe_impl(self, audio_input: "np.ndarray | str") -> STTResult:
        """Transcribe y agrega la confianza de los segmentos.

        Args:
            audio_input: Array numpy float32 normalizado [-1, 1] o path de
                archivo de audio como string. Preparado por
                ``transcribe_with_confidence`` antes de llegar aquí.

        Returns:
            STTResult con text, elapsed_ms, no_speech_prob y avg_logprob
            agregados a partir de todos los segmentos detectados. no_speech_prob
            y avg_logprob son None cuando no hay segmentos (audio vacío).
        """
        start = time.perf_counter()

        # sample_rate no se reenvía: faster-whisper infiere la tasa desde el
        # array numpy (asume 16 kHz, que es el formato que siempre recibe).

        # Transcribir con configuración (beam/prompt configurables — defaults
        # priorizan velocidad, subir beam_size=5 + initial_prompt mejora precisión
        # para palabras novel como "nexa" a costa de ~30% latencia).

        # Temperature: lista (fallback adaptativo de Whisper) o escalar 0.
        # En turbo, no_speech/avg_logprob están muertos → el fallback se gatea
        # por compression_ratio (única señal viva). log_prob_threshold se deja
        # muy bajo para NO disparar fallback espurio por el logprob invertido.
        temperature = (
            self.fallback_temperatures if self.temperature_fallback else 0
        )
        segments, info = self._model.transcribe(
            audio_input,
            language=self.language,
            beam_size=self.beam_size,
            best_of=self.best_of,
            temperature=temperature,
            compression_ratio_threshold=self.compression_ratio_threshold,
            log_prob_threshold=self.log_prob_threshold,
            initial_prompt=self.initial_prompt,
            condition_on_previous_text=False,
            vad_filter=self.vad_filter,
            vad_parameters={
                "min_silence_duration_ms": 300,
                "speech_pad_ms": 100,
                "threshold": 0.5,
            },
        )

        seg_list = list(segments)
        text = " ".join(s.text.strip() for s in seg_list)
        if seg_list:
            no_speech = sum(s.no_speech_prob for s in seg_list) / len(seg_list)
            avg_lp = sum(s.avg_logprob for s in seg_list) / len(seg_list)
            comp_ratio = max(s.compression_ratio for s in seg_list)
        else:
            no_speech = None
            avg_lp = None
            comp_ratio = None

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug(f"STT ({elapsed_ms:.0f}ms): {text[:50]}...")

        return STTResult(
            text=text,
            elapsed_ms=elapsed_ms,
            no_speech_prob=no_speech,
            avg_logprob=avg_lp,
            compression_ratio=comp_ratio,
        )

    def transcribe(
        self,
        audio: np.ndarray | str | Path,
        sample_rate: int = 16000,
    ) -> tuple[str, float]:
        """Transcribir audio a texto. Firma compat (text, elapsed_ms).

        Args:
            audio: Array de audio (float32/int16) o path al archivo
            sample_rate: Sample rate del audio (16000 recomendado)

        Returns:
            (texto, tiempo_ms)
        """
        r = self.transcribe_with_confidence(audio, sample_rate)
        return r.text, r.elapsed_ms

    def transcribe_with_confidence(
        self,
        audio: np.ndarray | str | Path,
        sample_rate: int = 16000,
    ) -> STTResult:
        """Transcribir devolviendo también la confianza del STT.

        Args:
            audio: Array de audio (float32/int16) o path al archivo
            sample_rate: Sample rate del audio (16000 recomendado)

        Returns:
            STTResult con text, elapsed_ms, no_speech_prob y avg_logprob.
        """
        # Preparar audio en el caller (solo numpy, sin CUDA)
        if isinstance(audio, np.ndarray):
            # Faster-whisper acepta numpy arrays directamente
            # Debe ser float32 normalizado entre -1.0 y 1.0
            audio_input = self._prepare_audio(audio)
        else:
            # Path a archivo - faster-whisper lo maneja internamente
            audio_input = str(audio)

        # load lazy + generate SIEMPRE en el thread dedicado del modelo
        return self._run_pinned(self._transcribe_pinned, audio_input)

    def _transcribe_pinned(self, audio_input: "np.ndarray | str") -> STTResult:
        if self._model is None:
            self._load_impl()
        return self._transcribe_impl(audio_input)

    def _prepare_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Preparar audio para faster-whisper (in-memory, sin I/O)

        Faster-whisper espera float32 normalizado [-1.0, 1.0]
        """
        # Convertir a float32 si es necesario
        if audio.dtype == np.int16:
            # int16 [-32768, 32767] -> float32 [-1.0, 1.0]
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Asegurar que esté normalizado
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val

        # Asegurar que sea mono (1D array)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        return audio
    
    def transcribe_streaming(
        self,
        audio_generator,
        sample_rate: int = 16000,
        min_audio_ms: int = 500,
        silence_threshold: float = 0.01,
        silence_duration_ms: int = 200
    ):
        """
        Transcripción con VAD streaming - comienza a procesar ANTES del silencio final.

        Optimización: En lugar de esperar a que termine el audio completo,
        detectamos cuando el usuario hace una pausa y comenzamos a transcribir.
        Ahorro estimado: ~50-100ms en comandos cortos.

        Args:
            audio_generator: Generador que yield chunks de audio
            sample_rate: Sample rate del audio
            min_audio_ms: Mínimo de audio antes de intentar transcribir
            silence_threshold: Umbral RMS para detectar silencio
            silence_duration_ms: ms de silencio para considerar fin de frase

        Yields:
            dict con {"partial": bool, "text": str, "is_final": bool}
        """
        if self._model is None:
            self.load()

        audio_buffer = []
        silence_samples = 0
        samples_per_ms = sample_rate // 1000
        silence_samples_threshold = silence_duration_ms * samples_per_ms
        min_samples = min_audio_ms * samples_per_ms

        total_samples = 0

        for chunk in audio_generator:
            if isinstance(chunk, np.ndarray):
                audio_buffer.append(chunk)
                total_samples += len(chunk)

                # Detectar silencio en el chunk actual
                rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))
                if rms < silence_threshold:
                    silence_samples += len(chunk)
                else:
                    silence_samples = 0

                # Si hay suficiente audio Y detectamos pausa, transcribir parcial
                if total_samples >= min_samples and silence_samples >= silence_samples_threshold:
                    # Concatenar y transcribir
                    full_audio = np.concatenate(audio_buffer)
                    text, _ = self.transcribe(full_audio, sample_rate)

                    if text.strip():
                        yield {
                            "partial": True,
                            "text": text.strip(),
                            "is_final": False,
                            "samples": total_samples
                        }

                    # Reset silence counter (pero mantener buffer para contexto)
                    silence_samples = 0

        # Transcripción final con todo el audio
        if audio_buffer:
            full_audio = np.concatenate(audio_buffer)
            text, elapsed_ms = self.transcribe(full_audio, sample_rate)

            yield {
                "partial": False,
                "text": text.strip(),
                "is_final": True,
                "samples": total_samples,
                "elapsed_ms": elapsed_ms
            }

    def transcribe_with_early_vad(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        chunk_ms: int = 100
    ) -> tuple[str, float, bool]:
        """
        Transcribir con VAD temprano - detecta fin de habla antes del timeout.

        En lugar de esperar el timeout completo de captura de audio,
        analiza el audio y comienza transcripción cuando detecta que
        el usuario terminó de hablar.

        Args:
            audio: Audio completo capturado
            sample_rate: Sample rate
            chunk_ms: Tamaño de chunk para análisis VAD

        Returns:
            (texto, tiempo_ms, early_detected) - early_detected=True si terminó antes
        """
        if self._model is None:
            self.load()

        # Analizar desde el final para encontrar dónde termina el habla
        chunk_samples = int(sample_rate * chunk_ms / 1000)
        silence_threshold = 0.015

        # Buscar desde el final hacia atrás
        speech_end = len(audio)
        consecutive_silence = 0
        min_silence_chunks = 3  # ~300ms de silencio

        for i in range(len(audio) - chunk_samples, 0, -chunk_samples):
            chunk = audio[i:i + chunk_samples]
            rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))

            if rms < silence_threshold:
                consecutive_silence += 1
                if consecutive_silence >= min_silence_chunks:
                    speech_end = i + chunk_samples * min_silence_chunks
                    break
            else:
                consecutive_silence = 0
                speech_end = i + chunk_samples

        # Transcribir solo la porción con habla
        early_detected = speech_end < len(audio) * 0.9
        audio_trimmed = audio[:speech_end] if early_detected else audio

        text, elapsed_ms = self.transcribe(audio_trimmed, sample_rate)

        if early_detected:
            logger.debug(f"VAD early: transcribiendo {speech_end}/{len(audio)} samples")

        return text, elapsed_ms, early_detected


class MoonshineSTT:
    """
    Alternativa: Moonshine STT (más rápido que Whisper)
    https://github.com/usefulsensors/moonshine
    """
    
    def __init__(self, model: str = "moonshine/base", device: str = "cuda:0"):
        self.model_name = model
        self.device = device
        self._model = None
    
    def load(self):
        """Cargar modelo Moonshine"""
        try:
            import moonshine_onnx
            
            logger.info(f"Cargando Moonshine: {self.model_name}")
            self._model = moonshine_onnx.load(self.model_name)
            logger.info("Moonshine cargado")
        except ImportError:
            logger.error("Moonshine no instalado: pip install moonshine-onnx")
            raise
    
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> tuple[str, float]:
        """Transcribir con Moonshine (muy rápido)"""
        if self._model is None:
            self.load()

        start = time.perf_counter()

        # Moonshine espera audio float32 normalizado
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if np.abs(audio).max() > 1.0:
            audio = audio / 32768.0

        text = self._model.transcribe(audio, sample_rate)

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug(f"Moonshine STT ({elapsed_ms:.0f}ms): {text[:50]}...")

        return text, elapsed_ms

    def transcribe_with_confidence(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> STTResult:
        """Transcribir devolviendo STTResult (confianza no disponible en Moonshine)."""
        text, ms = self.transcribe(audio, sample_rate)
        return STTResult(text=text, elapsed_ms=ms, no_speech_prob=None, avg_logprob=None)


def create_stt(config: dict) -> FastWhisperSTT | MoonshineSTT:
    """Factory para crear el STT según configuración"""
    model = config.get("model", "distil-whisper/distil-small.en")

    if "moonshine" in model.lower():
        return MoonshineSTT(
            model=model,
            device=config.get("device", "cuda:0")
        )
    else:
        return FastWhisperSTT(
            model=model,
            device=config.get("device", "cuda:0"),
            compute_type=config.get("compute_type", "float16"),
            language=config.get("language", "es"),
            beam_size=config.get("beam_size", 1),
            best_of=config.get("best_of", 1),
            initial_prompt=config.get("initial_prompt"),
            vad_filter=config.get("vad_filter", True),
            temperature_fallback=config.get("temperature_fallback", {}).get("enabled", False),
            fallback_temperatures=config.get("temperature_fallback", {}).get("temperatures"),
            compression_ratio_threshold=config.get("temperature_fallback", {}).get("compression_ratio_threshold", 2.0),
            log_prob_threshold=config.get("temperature_fallback", {}).get("log_prob_threshold", -3.0),
        )
