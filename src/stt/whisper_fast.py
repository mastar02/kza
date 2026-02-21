"""
Speech-to-Text Module
Transcripción de audio ultra-rápida con Whisper

Optimizado para latencia mínima:
- Procesamiento 100% en RAM (sin I/O de disco)
- VAD agresivo para reducir silencios
- Configuración de beam_size=1 para velocidad
"""

import logging
import time
from pathlib import Path
from typing import Optional, Union
import numpy as np

logger = logging.getLogger(__name__)


class FastWhisperSTT:
    """Speech-to-Text con Faster-Whisper optimizado para baja latencia"""
    
    def __init__(
        self,
        model: str = "distil-whisper/distil-small.en",
        device: str = "cuda:0",
        compute_type: str = "float16",
        language: str = "es"
    ):
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self._model = None
    
    def load(self):
        """Cargar modelo en GPU"""
        from faster_whisper import WhisperModel
        
        logger.info(f"Cargando Whisper: {self.model_name} en {self.device}")
        start = time.time()
        
        self._model = WhisperModel(
            self.model_name,
            device=self.device.replace("cuda:", "cuda"),
            device_index=int(self.device.split(":")[-1]) if ":" in self.device else 0,
            compute_type=self.compute_type
        )
        
        elapsed = time.time() - start
        logger.info(f"Whisper cargado en {elapsed:.1f}s")
    
    def transcribe(
        self,
        audio: Union[np.ndarray, str, Path],
        sample_rate: int = 16000
    ) -> tuple[str, float]:
        """
        Transcribir audio a texto - 100% en RAM

        Args:
            audio: Array de audio (float32/int16) o path al archivo
            sample_rate: Sample rate del audio (16000 recomendado)

        Returns:
            (texto, tiempo_ms)
        """
        if self._model is None:
            self.load()

        start = time.perf_counter()

        # Preparar audio para faster-whisper
        if isinstance(audio, np.ndarray):
            # Faster-whisper acepta numpy arrays directamente
            # Debe ser float32 normalizado entre -1.0 y 1.0
            audio_input = self._prepare_audio(audio)
        else:
            # Path a archivo - faster-whisper lo maneja internamente
            audio_input = str(audio)

        # Transcribir con configuración optimizada para velocidad
        segments, info = self._model.transcribe(
            audio_input,
            language=self.language,
            beam_size=1,              # Más rápido (sin beam search)
            best_of=1,                # Sin sampling múltiple
            temperature=0,            # Determinístico
            condition_on_previous_text=False,  # Más rápido para comandos cortos
            vad_filter=True,          # Filtra silencios
            vad_parameters={
                "min_silence_duration_ms": 300,   # Reducido de 500 (más agresivo)
                "speech_pad_ms": 100,             # Reducido de 200 (menos padding)
                "threshold": 0.5,                 # Umbral de detección de voz
            }
        )

        # Concatenar segmentos
        text = " ".join([s.text.strip() for s in segments])

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug(f"STT ({elapsed_ms:.0f}ms): {text[:50]}...")

        return text, elapsed_ms

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
        sample_rate: int = 16000
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
            language=config.get("language", "es")
        )
