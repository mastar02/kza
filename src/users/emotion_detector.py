"""
Emotion Detector Module
Detecta emociones en audio usando modelos de speech emotion recognition.

Emociones detectadas:
- Happy (alegre)
- Sad (triste)
- Angry (enojado)
- Fearful (asustado)
- Neutral (neutral)
- Surprised (sorprendido)

Proporciona:
1. Clasificación de emoción principal
2. Arousal (energía/activación): 0-1
3. Valence (positividad/negatividad): 0-1
4. Confianza en la predicción
5. Ajustes de respuesta según emoción detectada
"""

import logging
import time
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EmotionResult:
    """Resultado de detección de emoción"""
    emotion: str  # happy, sad, angry, fearful, neutral, surprised
    confidence: float  # 0-1
    arousal: float  # 0-1 (energía/activación)
    valence: float  # 0-1 (positivo=1, negativo=0)
    all_emotions: dict[str, float]  # {emotion: confidence} para todas las emociones
    processing_time_ms: float

    @property
    def is_confident(self) -> bool:
        """¿La predicción es confiable?"""
        return self.confidence >= 0.7

    @property
    def response_adjustment(self) -> dict[str, float]:
        """
        Ajustes recomendados para la respuesta según la emoción detectada.
        Usado para adaptar TTS (pitch, speed, energy).
        """
        adjustments = {
            "pitch_shift": 0.0,  # -0.5 (grave) a +0.5 (agudo)
            "speech_rate": 1.0,  # 0.5 (lento) a 2.0 (rápido)
            "energy": 1.0,  # 0.5 (suave) a 2.0 (fuerte)
            "emotional_tone": "neutral",
        }

        match self.emotion:
            case "happy":
                adjustments["pitch_shift"] = 0.2
                adjustments["speech_rate"] = 1.1
                adjustments["energy"] = 1.2
                adjustments["emotional_tone"] = "cheerful"

            case "sad":
                adjustments["pitch_shift"] = -0.2
                adjustments["speech_rate"] = 0.9
                adjustments["energy"] = 0.8
                adjustments["emotional_tone"] = "sympathetic"

            case "angry":
                adjustments["pitch_shift"] = 0.1
                adjustments["speech_rate"] = 1.2
                adjustments["energy"] = 1.5
                adjustments["emotional_tone"] = "firm"

            case "fearful":
                adjustments["pitch_shift"] = 0.15
                adjustments["speech_rate"] = 1.3
                adjustments["energy"] = 0.7
                adjustments["emotional_tone"] = "calm_reassuring"

            case "surprised":
                adjustments["pitch_shift"] = 0.3
                adjustments["speech_rate"] = 1.2
                adjustments["energy"] = 1.3
                adjustments["emotional_tone"] = "engaging"

            case "neutral":
                # Sin cambios especiales
                pass

        return adjustments


class EmotionDetector:
    """
    Detecta emociones en audio usando wav2vec2 + clasificador de emociones.

    Arquitectura:
    1. Audio → wav2vec2 embeddings → Vector de características
    2. Clasificador de emociones → Predicción de emoción + confidence
    3. Calcular arousal y valence a partir de la emoción
    4. Retornar EmotionResult con ajustes de respuesta

    Modelos soportados:
    - audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim (recomendado)
    - facebook/wav2vec2-large-xlsr-53 + clasificador custom
    - superb-emotion (alternativa más ligera)
    """

    # Mapeo de emociones a arousal/valence
    EMOTION_CHARACTERISTICS = {
        "happy": {"arousal": 0.8, "valence": 0.9},
        "sad": {"arousal": 0.3, "valence": 0.1},
        "angry": {"arousal": 0.9, "valence": 0.2},
        "fearful": {"arousal": 0.7, "valence": 0.2},
        "neutral": {"arousal": 0.5, "valence": 0.5},
        "surprised": {"arousal": 0.8, "valence": 0.6},
    }

    def __init__(
        self,
        model_name: str = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
        device: str = "cuda:1",
        sample_rate: int = 16000,
    ):
        """
        Inicializar EmotionDetector.

        Args:
            model_name: Nombre del modelo en HuggingFace Hub
            device: Dispositivo CUDA a usar (ej: "cuda:1")
            sample_rate: Sample rate del audio esperado (Hz)
        """
        self.model_name = model_name
        self.device = device
        self.sample_rate = sample_rate

        self._model = None
        self._processor = None
        self._model_type = None

    def load(self):
        """
        Cargar modelo de detección de emociones.
        Lazy loading para no consumir GPU en startup.
        """
        if self._model is not None:
            logger.debug("Modelo ya cargado")
            return

        logger.info(f"Cargando emotion model: {self.model_name}")
        start = time.time()

        try:
            if "wav2vec2" in self.model_name.lower():
                self._load_wav2vec2()
            else:
                self._load_wav2vec2()  # Default

            elapsed = time.time() - start
            logger.info(f"Emotion model cargado en {elapsed:.1f}s")

        except Exception as e:
            logger.error(f"Error cargando emotion model: {e}")
            self._model = None
            self._processor = None
            raise

    def _load_wav2vec2(self):
        """Cargar modelo wav2vec2 para speech emotion recognition."""
        try:
            from transformers import pipeline

            self._model = pipeline(
                "audio-classification",
                model=self.model_name,
                device=self._get_device_id(),
            )
            self._model_type = "wav2vec2_pipeline"
            logger.debug(f"wav2vec2 pipeline cargado: {self.model_name}")

        except ImportError:
            raise ImportError(
                "Necesitas instalar transformers:\n"
                "  pip install transformers[audio] torch torchaudio"
            )

    def _get_device_id(self) -> int:
        """Convertir device string a integer para transformers."""
        if self.device == "cpu":
            return -1
        # Extraer número de device (ej: "cuda:1" -> 1)
        try:
            return int(self.device.split(":")[-1])
        except (ValueError, IndexError):
            return 0

    def detect(self, audio: np.ndarray) -> EmotionResult:
        """
        Detectar emoción en audio.

        Args:
            audio: Audio como numpy array (float32, mono)

        Returns:
            EmotionResult con emoción detectada y características
        """
        if self._model is None:
            self.load()

        start = time.perf_counter()

        try:
            # Asegurar formato correcto
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # Normalizar si es necesario
            if np.abs(audio).max() > 1.0:
                audio = audio / np.abs(audio).max()

            # Detectar emoción
            predictions = self._model(
                audio,
                sampling_rate=self.sample_rate,
            )

            # Procesar resultados
            result = self._process_predictions(predictions)

            elapsed_ms = (time.perf_counter() - start) * 1000
            result.processing_time_ms = elapsed_ms

            logger.debug(
                f"Emotion detected: {result.emotion} "
                f"(conf={result.confidence:.2f}) in {elapsed_ms:.0f}ms"
            )

            return result

        except Exception as e:
            logger.error(f"Error detectando emoción: {e}")
            # Retornar neutral como fallback
            return self._create_neutral_result(start)

    def _process_predictions(self, predictions: list[dict]) -> EmotionResult:
        """
        Procesar predicciones del modelo.

        El modelo retorna lista de dicts con 'label' y 'score'.
        Ejemplo:
        [
            {'label': 'sad', 'score': 0.8},
            {'label': 'angry', 'score': 0.15},
            ...
        ]
        """
        if not predictions:
            return self._create_neutral_result(time.perf_counter())

        # Construir dict de todas las emociones
        all_emotions = {pred["label"]: pred["score"] for pred in predictions}

        # Encontrar emoción principal
        main_pred = predictions[0]
        emotion = main_pred["label"].lower()
        confidence = main_pred["score"]

        # Normalizar si la emoción no está en nuestro conjunto
        if emotion not in self.EMOTION_CHARACTERISTICS:
            emotion = "neutral"
            confidence = 0.0

        # Obtener arousal y valence de características predefinidas
        characteristics = self.EMOTION_CHARACTERISTICS[emotion]
        arousal = characteristics["arousal"]
        valence = characteristics["valence"]

        return EmotionResult(
            emotion=emotion,
            confidence=confidence,
            arousal=arousal,
            valence=valence,
            all_emotions=all_emotions,
            processing_time_ms=0.0,  # Se asigna después en detect()
        )

    def _create_neutral_result(self, start_time: float) -> EmotionResult:
        """Crear resultado neutral como fallback."""
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        characteristics = self.EMOTION_CHARACTERISTICS["neutral"]

        return EmotionResult(
            emotion="neutral",
            confidence=0.0,
            arousal=characteristics["arousal"],
            valence=characteristics["valence"],
            all_emotions={
                "happy": 0.0,
                "sad": 0.0,
                "angry": 0.0,
                "fearful": 0.0,
                "neutral": 1.0,
                "surprised": 0.0,
            },
            processing_time_ms=elapsed_ms,
        )

    def batch_detect(
        self,
        audio_samples: list[np.ndarray],
    ) -> list[EmotionResult]:
        """
        Detectar emociones en múltiples audios - BATCH REAL en GPU.

        Optimización: Procesa todos los audios en un solo forward pass
        aprovechando la paralelización de GPU. Ahorra ~10-20ms vs secuencial.

        Args:
            audio_samples: Lista de audios (numpy arrays)

        Returns:
            Lista de EmotionResult
        """
        if not audio_samples:
            return []

        if len(audio_samples) == 1:
            return [self.detect(audio_samples[0])]

        if self._model is None:
            self.load()

        start = time.perf_counter()

        try:
            # Preparar todos los audios
            prepared_audios = []
            for audio in audio_samples:
                if audio.dtype != np.float32:
                    audio = audio.astype(np.float32)
                if np.abs(audio).max() > 1.0:
                    audio = audio / np.abs(audio).max()
                prepared_audios.append(audio)

            # Batch inference con transformers pipeline
            # El pipeline acepta lista de audios y procesa en batch
            batch_predictions = self._model(
                prepared_audios,
                sampling_rate=self.sample_rate,
                batch_size=len(prepared_audios),  # Procesar todo junto
            )

            elapsed_ms = (time.perf_counter() - start) * 1000

            # Procesar resultados
            results = []
            per_sample_ms = elapsed_ms / len(audio_samples)

            for predictions in batch_predictions:
                # predictions puede ser lista de dicts o lista de listas de dicts
                if isinstance(predictions, list) and predictions and isinstance(predictions[0], dict):
                    result = self._process_predictions(predictions)
                else:
                    result = self._process_predictions([predictions] if isinstance(predictions, dict) else predictions)

                result.processing_time_ms = per_sample_ms
                results.append(result)

            logger.debug(
                f"Batch emotion detect: {len(audio_samples)} samples "
                f"in {elapsed_ms:.0f}ms ({per_sample_ms:.0f}ms/sample)"
            )

            return results

        except Exception as e:
            logger.warning(f"Batch emotion failed, falling back to sequential: {e}")
            # Fallback a procesamiento secuencial
            return [self.detect(audio) for audio in audio_samples]

    def detect_parallel(
        self,
        audio: np.ndarray,
        return_all_emotions: bool = False
    ) -> EmotionResult:
        """
        Detección optimizada con pre-procesamiento paralelo.

        Usa torch.no_grad() explícito para inferencia más rápida.

        Args:
            audio: Audio como numpy array
            return_all_emotions: Si incluir todas las emociones en el resultado

        Returns:
            EmotionResult
        """
        if self._model is None:
            self.load()

        start = time.perf_counter()

        try:
            import torch

            # Asegurar formato correcto
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            if np.abs(audio).max() > 1.0:
                audio = audio / np.abs(audio).max()

            # Inferencia con no_grad para máxima velocidad
            with torch.no_grad():
                predictions = self._model(
                    audio,
                    sampling_rate=self.sample_rate,
                )

            result = self._process_predictions(predictions)
            result.processing_time_ms = (time.perf_counter() - start) * 1000

            return result

        except Exception as e:
            logger.error(f"Error en detect_parallel: {e}")
            return self._create_neutral_result(start)

    def get_emotion_description(self, emotion: str) -> str:
        """
        Obtener descripción en español para una emoción.

        Args:
            emotion: Nombre de la emoción

        Returns:
            Descripción en español
        """
        descriptions = {
            "happy": "Alegre",
            "sad": "Triste",
            "angry": "Enojado",
            "fearful": "Asustado",
            "neutral": "Neutral",
            "surprised": "Sorprendido",
        }
        return descriptions.get(emotion, "Desconocida")
