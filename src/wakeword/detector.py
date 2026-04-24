"""
Wake Word Detector
Detecta palabras de activación usando modelos pre-entrenados o personalizados.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


class WakeWordDetector:
    """
    Detector de wake word con soporte para múltiples modelos.

    Soporta:
    - Modelos pre-entrenados de OpenWakeWord (hey_jarvis, alexa, etc.)
    - Modelos personalizados entrenados con WakeWordTrainer
    - Múltiples wake words simultáneos
    """

    # Modelos pre-entrenados disponibles en OpenWakeWord
    PRETRAINED_MODELS = [
        "hey_jarvis",
        "alexa",
        "hey_mycroft",
        "ok_nabu",
        "timer",
        "weather",
    ]

    def __init__(
        self,
        models: list[str] = None,
        custom_models_dir: str = "./models/wakeword",
        threshold: float = 0.5,
        refractory_period: float = 2.0,  # Segundos entre detecciones
        inference_framework: str = "onnx"
    ):
        """
        Args:
            models: Lista de modelos a usar (pre-entrenados o paths a .onnx)
            custom_models_dir: Directorio de modelos personalizados
            threshold: Umbral de detección (0-1)
            refractory_period: Tiempo mínimo entre detecciones
            inference_framework: Framework para inferencia ("onnx" o "tflite")
        """
        self.models = models or ["hey_jarvis"]
        self.custom_models_dir = Path(custom_models_dir)
        self.threshold = threshold
        self.refractory_period = refractory_period
        self.inference_framework = inference_framework

        self._oww_model = None
        self._custom_models: dict[str, any] = {}
        self._last_detection_time: dict[str, float] = {}
        self._loaded = False

        # Callbacks
        self._on_detection: Callable[[str, float], None] | None = None

    def load(self):
        """Cargar modelos de wake word"""
        logger.info(f"Cargando wake word detector con modelos: {self.models}")

        pretrained = []
        custom = []

        for model in self.models:
            if model in self.PRETRAINED_MODELS:
                pretrained.append(model)
            elif model.endswith(".onnx"):
                custom.append(model)
            else:
                # Verificar si es un modelo personalizado en el directorio
                custom_path = self.custom_models_dir / f"{model}.onnx"
                if custom_path.exists():
                    custom.append(str(custom_path))
                else:
                    logger.warning(f"Modelo no encontrado: {model}")

        # Cargar modelos pre-entrenados con OpenWakeWord
        if pretrained:
            self._load_openwakeword(pretrained)

        # Cargar modelos personalizados
        for model_path in custom:
            self._load_custom_model(model_path)

        self._loaded = True
        logger.info(f"Wake word detector cargado. Modelos activos: {self.get_active_models()}")

    def _load_openwakeword(self, models: list[str]):
        """Cargar modelos de OpenWakeWord"""
        try:
            from openwakeword import Model

            self._oww_model = Model(
                wakeword_models=models,
                inference_framework=self.inference_framework
            )
            logger.info(f"OpenWakeWord cargado: {models}")

        except ImportError:
            logger.error("OpenWakeWord no instalado. Ejecuta: pip install openwakeword")
            raise
        except Exception as e:
            logger.error(f"Error cargando OpenWakeWord: {e}")
            raise

    def _load_custom_model(self, model_path: str):
        """Cargar modelo personalizado ONNX"""
        try:
            import onnxruntime as ort

            path = Path(model_path)
            if not path.exists():
                logger.warning(f"Modelo personalizado no encontrado: {model_path}")
                return

            session = ort.InferenceSession(str(path))
            model_name = path.stem

            self._custom_models[model_name] = {
                "session": session,
                "path": str(path),
                "input_name": session.get_inputs()[0].name,
                "output_name": session.get_outputs()[0].name
            }

            logger.info(f"Modelo personalizado cargado: {model_name}")

        except ImportError:
            logger.error("onnxruntime no instalado. Ejecuta: pip install onnxruntime")
        except Exception as e:
            logger.error(f"Error cargando modelo {model_path}: {e}")

    def predict(self, audio_chunk: np.ndarray) -> dict[str, float]:
        """
        Procesar chunk de audio y detectar wake words.

        Args:
            audio_chunk: Audio como numpy array (float32, 16kHz, mono).
                Se acepta tanto float32 normalizado [-1,1] como int16-scale;
                OpenWakeWord necesita escala int16 internamente.

        Returns:
            Dict de {model_name: confidence_score}
        """
        if not self._loaded:
            self.load()

        results = {}

        # OpenWakeWord espera int16 (o float32 en rango [-32768,32767]).
        # sounddevice/faster-whisper entregan float32 normalizado [-1,1] → escalar antes.
        oww_input = audio_chunk
        if audio_chunk.dtype == np.float32 or audio_chunk.dtype == np.float64:
            max_abs = float(np.max(np.abs(audio_chunk))) if audio_chunk.size else 0.0
            if max_abs <= 1.5:  # está normalizado
                oww_input = (np.clip(audio_chunk, -1.0, 1.0) * 32767).astype(np.int16)

        # Predicción con OpenWakeWord
        if self._oww_model:
            oww_predictions = self._oww_model.predict(oww_input)
            results.update(oww_predictions)

        # Predicción con modelos personalizados
        for name, model_info in self._custom_models.items():
            try:
                score = self._predict_custom(model_info, audio_chunk)
                results[name] = score
            except Exception as e:
                logger.debug(f"Error en predicción {name}: {e}")
                results[name] = 0.0

        return results

    def _predict_custom(self, model_info: dict, audio: np.ndarray) -> float:
        """Predecir con modelo personalizado"""
        import torchaudio
        import torch

        # Convertir a mel spectrogram (mismo preprocesamiento que entrenamiento)
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        audio_tensor = torch.from_numpy(audio).unsqueeze(0)

        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_mels=40,
            n_fft=400,
            hop_length=160
        )
        mel = mel_transform(audio_tensor)

        # Inferencia ONNX
        session = model_info["session"]
        input_data = mel.numpy()

        outputs = session.run(
            [model_info["output_name"]],
            {model_info["input_name"]: input_data}
        )

        return float(outputs[0][0])

    def detect(self, audio_chunk: np.ndarray) -> tuple[str, float] | None:
        """
        Detectar wake word en chunk de audio.

        Args:
            audio_chunk: Audio como numpy array

        Returns:
            (model_name, confidence) si se detecta, None si no
        """
        predictions = self.predict(audio_chunk)
        current_time = time.time()

        for model_name, confidence in predictions.items():
            if confidence >= self.threshold:
                # Verificar período refractario
                last_time = self._last_detection_time.get(model_name, 0)
                if current_time - last_time >= self.refractory_period:
                    self._last_detection_time[model_name] = current_time

                    logger.debug(f"Wake word detectado: {model_name} ({confidence:.2f})")

                    # Callback si está configurado
                    if self._on_detection:
                        self._on_detection(model_name, confidence)

                    return model_name, confidence

        return None

    def on_detection(self, callback: Callable[[str, float], None]):
        """
        Registrar callback para cuando se detecta un wake word.

        Args:
            callback: Función que recibe (model_name, confidence)
        """
        self._on_detection = callback

    def get_active_models(self) -> list[str]:
        """Obtener lista de modelos activos"""
        models = []

        if self._oww_model:
            # Obtener nombres de modelos de OpenWakeWord
            for model_name in self.models:
                if model_name in self.PRETRAINED_MODELS:
                    models.append(model_name)

        models.extend(list(self._custom_models.keys()))

        return models

    def set_threshold(self, threshold: float, model_name: str = None):
        """
        Ajustar umbral de detección.

        Args:
            threshold: Nuevo umbral (0-1)
            model_name: Modelo específico (None para todos)
        """
        if model_name:
            # Por ahora aplicamos el mismo umbral a todos
            logger.info(f"Threshold para {model_name}: {threshold}")
        else:
            logger.info(f"Threshold global: {threshold}")

        self.threshold = threshold

    def reset(self):
        """Resetear estado del detector"""
        self._last_detection_time.clear()

        if self._oww_model:
            self._oww_model.reset()

    def get_info(self) -> dict:
        """Obtener información del detector"""
        return {
            "models": self.get_active_models(),
            "threshold": self.threshold,
            "refractory_period": self.refractory_period,
            "inference_framework": self.inference_framework,
            "loaded": self._loaded,
            "pretrained_count": 1 if self._oww_model else 0,
            "custom_count": len(self._custom_models)
        }
