"""
Wake Word Detector
Detecta palabras de activación usando modelos pre-entrenados o personalizados.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import Context
import logging
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


class _WakeLease:
    def __init__(self, owner):
        self.owner = owner
        self.active = True

class _WakeLifecycle:
    """Retain this adapter's real operations without owning its callers' data."""

    def _wake_initialize(self) -> None:
        self._wake_condition = threading.Condition()
        self._wake_local = threading.local()
        self._wake_active = 0
        self._wake_sealed = False
        self._wake_close_started = False
        self._wake_close_owner = None
        self._wake_close_done = threading.Event()
        self._wake_failure = None
        self._wake_closing = None

    def _wake_current(self):
        lease = getattr(self._wake_local, "lease", None)
        return lease if lease is not None and lease.owner is self and lease.active else None

    @contextmanager
    def _wake_admit(self):
        lease = self._wake_current()
        owned = lease is None
        if owned:
            with self._wake_condition:
                if self._wake_sealed:
                    raise RuntimeError("wake_closed")
                lease = _WakeLease(self)
                self._wake_active += 1
        previous = getattr(self._wake_local, "lease", None)
        self._wake_local.lease = lease
        try:
            yield
        finally:
            self._wake_local.lease = previous
            if owned:
                with self._wake_condition:
                    lease.active = False
                    self._wake_active -= 1
                    self._wake_condition.notify_all()

    def _wake_require_close_caller(self) -> None:
        if self._wake_current() is not None:
            raise RuntimeError("wake_close_from_worker")
        with self._wake_condition:
            if (self._wake_close_owner is threading.current_thread()
                    and not self._wake_close_done.is_set()):
                raise RuntimeError("wake_close_from_owner")

    def request_close(self) -> None:
        """Seal admission using memory only, without waiting for model locks."""
        with self._wake_condition:
            self._wake_sealed = True
            self._wake_condition.notify_all()

    def wait_closed(self) -> None:
        """Drain actual operations and drop only this adapter's Python references."""
        self._wake_require_close_caller()
        self.request_close()
        with self._wake_condition:
            own_close = not self._wake_close_started
            if own_close:
                self._wake_close_started = True
                self._wake_close_owner = threading.current_thread()
        if own_close:
            failure = None
            try:
                with self._wake_condition:
                    while self._wake_active:
                        self._wake_condition.wait()
                self._wake_release_resources()
            except BaseException:
                failure = "wake_close_failed"
            with self._wake_condition:
                self._wake_failure = failure
                self._wake_close_owner = None
                self._wake_close_done.set()
        else:
            self._wake_close_done.wait()
        if self._wake_failure is not None:
            raise RuntimeError(self._wake_failure)

    def close(self) -> None:
        """Synchronous retirement; async callers use begin_close or aclose."""
        self.wait_closed()

    def begin_close(self) -> asyncio.Task:
        """Return the sole retained owner; never cancel this owner directly."""
        self._wake_require_close_caller()
        loop = asyncio.get_running_loop()
        self.request_close()
        with self._wake_condition:
            if self._wake_closing is None:
                async def finish():
                    # Publish the task outside the condition even with eager tasks.
                    await asyncio.sleep(0)
                    await asyncio.to_thread(self.wait_closed)

                self._wake_closing = loop.create_task(finish(), context=Context())
                self._wake_closing.add_done_callback(
                    lambda done: None if done.cancelled() else done.exception(),
                    context=Context(),
                )
            return self._wake_closing

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


class WakeWordDetector(_WakeLifecycle):
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
        self._wake_initialize()
        self._model_lock = threading.RLock()
        self._wake_has_backend = False
        self._wake_custom_names: tuple[str, ...] = ()

        # Callbacks
        self._on_detection: Callable[[str, float], None] | None = None

    def load(self):
        """Cargar modelos de wake word"""
        with self._wake_admit():
            with self._model_lock:
                if self._loaded:
                    return
                logger.info(f"Cargando wake word detector con modelos: {self.models}")

                pretrained = []
                custom = []

                for model in self.models:
                    if model in self.PRETRAINED_MODELS:
                        pretrained.append(model)
                    elif model.endswith(".onnx"):
                        pretrained.append(model)  # KZA: .onnx -> openwakeword.Model (pipeline correcto)
                    else:
                        # Verificar si es un modelo personalizado en el directorio
                        custom_path = self.custom_models_dir / f"{model}.onnx"
                        if custom_path.exists():
                            pretrained.append(str(custom_path))
                        else:
                            logger.warning(f"Modelo no encontrado: {model}")

                # Cargar modelos pre-entrenados con OpenWakeWord
                if pretrained:
                    self._load_openwakeword(pretrained)

                # Cargar modelos personalizados
                for model_path in custom:
                    self._load_custom_model(model_path)

                with self._wake_condition:
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
            self._wake_sync_metadata()
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

            self._wake_sync_metadata()
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
        with self._wake_admit():
            with self._model_lock:
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
        with self._wake_admit():
            with self._model_lock:
                predictions = self.predict(audio_chunk)
                current_time = time.time()
                self._dbg_n = getattr(self, "_dbg_n", 0) + 1
                if predictions:
                    _mxk = max(predictions, key=predictions.get); _mxv = predictions[_mxk]
                    if self._dbg_n % 25 == 0 or _mxv > 0.05:
                        import numpy as _np
                        _ac = audio_chunk.astype(_np.float32) if audio_chunk.dtype != _np.float32 else audio_chunk
                        _rms = float(_np.sqrt(_np.mean(_ac**2))) if _ac.size else 0.0
                        logger.info(f"[oww-dbg] n={self._dbg_n} rms={_rms:.4f} max={_mxk}:{_mxv:.3f} thr={self.threshold}")

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
        with self._wake_admit():
            with self._model_lock:
                self._on_detection = callback

    def get_active_models(self) -> list[str]:
        """Obtener lista de modelos activos"""
        return self._wake_metadata()["models"]

    def set_threshold(self, threshold: float, model_name: str = None):
        """
        Ajustar umbral de detección.

        Args:
            threshold: Nuevo umbral (0-1)
            model_name: Modelo específico (None para todos)
        """
        with self._wake_admit():
            with self._model_lock:
                if model_name:
                    # Por ahora aplicamos el mismo umbral a todos
                    logger.info(f"Threshold para {model_name}: {threshold}")
                else:
                    logger.info(f"Threshold global: {threshold}")

                self.threshold = threshold

    def reset_refractory(self):
        """Liberar el período refractario sin tocar el estado del modelo.

        Usado por el AmbientGuard (2026-06-05): cuando el guard rechaza un
        wake (score < strict_wake_score), la detección NO debe consumir la
        ventana refractaria de 2s — un frame de TV a 0.5 abría el refractario
        y suprimía el "Nexa" real del usuario a 0.9 que llegaba 80ms después.
        """
        with self._wake_admit():
            with self._model_lock:
                self._last_detection_time.clear()

    def reset(self):
        """Resetear estado del detector"""
        with self._wake_admit():
            with self._model_lock:
                self._last_detection_time.clear()

                if self._oww_model:
                    self._oww_model.reset()

    def get_info(self) -> dict:
        """Obtener información del detector"""
        return self._wake_metadata()

    def _wake_sync_metadata(self) -> None:
        """Publish scalar/name metadata from the serialized native operation."""
        with self._wake_condition:
            self._wake_has_backend = self._oww_model is not None
            self._wake_custom_names = tuple(self._custom_models)

    def _wake_metadata(self) -> dict:
        """Read detached metadata without borrowing a backend or its native lock."""
        with self._wake_condition:
            models = ([name for name in self.models if name in self.PRETRAINED_MODELS]
                      if self._wake_has_backend else [])
            models.extend(self._wake_custom_names)
            return {
                "models": models,
                "threshold": self.threshold,
                "refractory_period": self.refractory_period,
                "inference_framework": self.inference_framework,
                "loaded": self._loaded,
                "pretrained_count": 1 if self._wake_has_backend else 0,
                "custom_count": len(self._wake_custom_names),
            }

    def _wake_release_resources(self) -> None:
        """Drop only this instance's references after admitted work really exits."""
        with self._model_lock:
            self._oww_model = None
            self._custom_models.clear()
            self._on_detection = None
            self._last_detection_time.clear()
            with self._wake_condition:
                self._wake_has_backend = False
                self._wake_custom_names = ()
                self._loaded = False
