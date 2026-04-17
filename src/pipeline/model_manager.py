"""
Model Manager - Gestión centralizada de modelos GPU.

Este módulo maneja la carga y descarga de todos los modelos que usan GPU,
permitiendo liberar VRAM cuando sea necesario (ej: entrenamiento nocturno).

Arquitectura GPU de día:
    GPU 0: TTS (Piper/VITS) - ~2GB
    GPU 1: Embeddings (BGE) + Speaker ID (ECAPA-TDNN) - ~2GB
    GPU 2: Router (Qwen 7B) - ~6GB
    GPU 3: Whisper (STT) + Emotion Detection - ~3GB

Uso:
    manager = ModelManager(config)
    await manager.load_all()      # Cargar todos los modelos
    await manager.unload_all()    # Descargar para liberar VRAM
    await manager.reload_all()    # Recargar después del entrenamiento
"""

import asyncio
import gc
import logging
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable
import time

logger = logging.getLogger(__name__)


class ModelState(StrEnum):
    """Estado de un modelo"""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    UNLOADING = "unloading"
    ERROR = "error"


@dataclass
class ModelInfo:
    """Información de un modelo"""
    name: str
    gpu_id: int
    vram_mb: int  # VRAM aproximada en MB
    state: ModelState = ModelState.UNLOADED
    model_instance: Any = None
    load_time: float = 0.0
    error_message: str | None = None


@dataclass
class ModelManagerConfig:
    """Configuración del ModelManager"""
    # TTS
    tts_model: str = "piper"  # "piper" o "vits"
    tts_gpu: int = 0
    tts_voice: str = "es_ES-mls_9972-medium"

    # Embeddings
    embeddings_model: str = "BAAI/bge-small-en-v1.5"
    embeddings_gpu: int = 1

    # Speaker ID
    speaker_id_model: str = "speechbrain/spkrec-ecapa-voxceleb"
    speaker_id_gpu: int = 1

    # Router
    router_model: str = "Qwen/Qwen2.5-7B-Instruct"
    router_gpu: int = 2
    router_enabled: bool = True

    # STT (Whisper)
    stt_model: str = "openai/whisper-medium"
    stt_gpu: int = 3

    # Emotion Detection
    emotion_model: str = "superb/wav2vec2-base-superb-er"
    emotion_gpu: int = 3
    emotion_enabled: bool = True

    # General
    lazy_load: bool = False  # Cargar modelos al arrancar o bajo demanda
    timeout_seconds: int = 120  # Timeout para carga de modelos


class ModelManager:
    """
    Gestiona la carga y descarga de todos los modelos GPU.

    Features:
    - Carga/descarga centralizada de modelos
    - Tracking de estado y VRAM
    - Callbacks para notificar cambios
    - Soporte para carga lazy (bajo demanda)
    - Limpieza de CUDA cache

    Uso típico:
        manager = ModelManager(config)

        # Al iniciar el sistema
        await manager.load_all()

        # Antes del entrenamiento nocturno
        await manager.unload_all()
        # ... entrenar ...
        await manager.reload_all()
    """

    def __init__(
        self,
        config: ModelManagerConfig = None,
        on_state_change: Callable[[str, ModelState], None] = None
    ):
        self.config = config or ModelManagerConfig()
        self.on_state_change = on_state_change

        # Registro de modelos
        self.models: dict[str, ModelInfo] = {}

        # Inicializar registro con modelos conocidos
        self._init_model_registry()

        # Estado global
        self._loading = False
        self._unloading = False

        logger.info("ModelManager inicializado")

    def _init_model_registry(self):
        """Inicializar registro de modelos conocidos"""
        models = [
            ModelInfo(
                name="tts",
                gpu_id=self.config.tts_gpu,
                vram_mb=2000,  # ~2GB
            ),
            ModelInfo(
                name="embeddings",
                gpu_id=self.config.embeddings_gpu,
                vram_mb=500,  # ~0.5GB
            ),
            ModelInfo(
                name="speaker_id",
                gpu_id=self.config.speaker_id_gpu,
                vram_mb=1500,  # ~1.5GB
            ),
            ModelInfo(
                name="stt",
                gpu_id=self.config.stt_gpu,
                vram_mb=2500,  # ~2.5GB (Whisper medium)
            ),
        ]

        # Modelos opcionales
        if self.config.router_enabled:
            models.append(ModelInfo(
                name="router",
                gpu_id=self.config.router_gpu,
                vram_mb=6000,  # ~6GB (7B model)
            ))

        if self.config.emotion_enabled:
            models.append(ModelInfo(
                name="emotion",
                gpu_id=self.config.emotion_gpu,
                vram_mb=500,  # ~0.5GB
            ))

        self.models = {m.name: m for m in models}

    # =========================================================================
    # LOADING
    # =========================================================================

    async def load_all(self) -> dict[str, bool]:
        """
        Cargar todos los modelos.

        Returns:
            Dict con nombre de modelo -> éxito
        """
        if self._loading:
            logger.warning("Ya hay una carga en progreso")
            return {}

        self._loading = True
        results = {}

        try:
            logger.info("📦 Cargando todos los modelos GPU...")
            start_time = time.time()

            # Cargar en paralelo por GPU para evitar contención
            gpus = set(m.gpu_id for m in self.models.values())

            for gpu_id in sorted(gpus):
                gpu_models = [
                    name for name, m in self.models.items()
                    if m.gpu_id == gpu_id
                ]

                logger.info(f"  GPU {gpu_id}: {gpu_models}")

                for name in gpu_models:
                    try:
                        success = await self._load_model(name)
                        results[name] = success
                    except Exception as e:
                        logger.error(f"Error cargando {name}: {e}")
                        results[name] = False

            elapsed = time.time() - start_time
            loaded = sum(1 for v in results.values() if v)
            logger.info(f"✓ Modelos cargados: {loaded}/{len(results)} en {elapsed:.1f}s")

        finally:
            self._loading = False

        return results

    async def _load_model(self, name: str) -> bool:
        """Cargar un modelo específico"""
        if name not in self.models:
            logger.error(f"Modelo desconocido: {name}")
            return False

        model_info = self.models[name]

        if model_info.state == ModelState.LOADED:
            logger.debug(f"{name} ya está cargado")
            return True

        model_info.state = ModelState.LOADING
        self._notify_state_change(name, ModelState.LOADING)

        start_time = time.time()

        try:
            # Cargar según el tipo de modelo
            if name == "tts":
                model_info.model_instance = await self._load_tts()
            elif name == "embeddings":
                model_info.model_instance = await self._load_embeddings()
            elif name == "speaker_id":
                model_info.model_instance = await self._load_speaker_id()
            elif name == "stt":
                model_info.model_instance = await self._load_stt()
            elif name == "router":
                model_info.model_instance = await self._load_router()
            elif name == "emotion":
                model_info.model_instance = await self._load_emotion()
            else:
                raise ValueError(f"Loader no implementado para: {name}")

            model_info.state = ModelState.LOADED
            model_info.load_time = time.time() - start_time
            model_info.error_message = None

            self._notify_state_change(name, ModelState.LOADED)
            logger.info(f"  ✓ {name} cargado en GPU {model_info.gpu_id} ({model_info.load_time:.1f}s)")

            return True

        except Exception as e:
            model_info.state = ModelState.ERROR
            model_info.error_message = str(e)
            self._notify_state_change(name, ModelState.ERROR)
            logger.error(f"  ✗ Error cargando {name}: {e}")
            return False

    async def _load_tts(self):
        """Cargar modelo TTS"""
        # Importar aquí para evitar dependencias circulares
        try:
            if self.config.tts_model == "piper":
                from piper import PiperVoice
                voice_path = f"./models/piper/{self.config.tts_voice}.onnx"
                return PiperVoice.load(voice_path)
            else:
                # VITS o alternativa
                import torch
                from TTS.api import TTS
                device = f"cuda:{self.config.tts_gpu}"
                tts = TTS("tts_models/es/css10/vits").to(device)
                return tts
        except ImportError as e:
            logger.warning(f"TTS no disponible: {e}")
            return None

    async def _load_embeddings(self):
        """Cargar modelo de embeddings"""
        try:
            from sentence_transformers import SentenceTransformer
            device = f"cuda:{self.config.embeddings_gpu}"
            model = SentenceTransformer(
                self.config.embeddings_model,
                device=device
            )
            return model
        except ImportError as e:
            logger.warning(f"Embeddings no disponible: {e}")
            return None

    async def _load_speaker_id(self):
        """Cargar modelo de identificación de hablante"""
        try:
            # Workaround: torchaudio 2.10 removed list_audio_backends()
            import torchaudio
            if not hasattr(torchaudio, "list_audio_backends"):
                torchaudio.list_audio_backends = lambda: ["ffmpeg"]

            from speechbrain.inference.speaker import EncoderClassifier
            device = f"cuda:{self.config.speaker_id_gpu}"
            model = EncoderClassifier.from_hparams(
                source=self.config.speaker_id_model,
                run_opts={"device": device}
            )
            return model
        except ImportError as e:
            logger.warning(f"Speaker ID no disponible: {e}")
            return None

    async def _load_stt(self):
        """Cargar modelo STT (Whisper)"""
        try:
            import torch
            from transformers import WhisperProcessor, WhisperForConditionalGeneration

            device = f"cuda:{self.config.stt_gpu}"

            processor = WhisperProcessor.from_pretrained(self.config.stt_model)
            model = WhisperForConditionalGeneration.from_pretrained(
                self.config.stt_model,
                torch_dtype=torch.float16
            ).to(device)

            return {"processor": processor, "model": model}
        except ImportError as e:
            logger.warning(f"STT no disponible: {e}")
            return None

    async def _load_router(self):
        """Cargar modelo Router (Qwen 7B)"""
        try:
            import torch
            from vllm import LLM

            model = LLM(
                model=self.config.router_model,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.85,
                dtype="half"
            )
            return model
        except ImportError as e:
            logger.warning(f"Router no disponible: {e}")
            return None

    async def _load_emotion(self):
        """Cargar modelo de detección de emociones"""
        try:
            import torch
            from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

            device = f"cuda:{self.config.emotion_gpu}"

            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                self.config.emotion_model
            )
            model = Wav2Vec2ForSequenceClassification.from_pretrained(
                self.config.emotion_model
            ).to(device)

            return {"feature_extractor": feature_extractor, "model": model}
        except ImportError as e:
            logger.warning(f"Emotion detection no disponible: {e}")
            return None

    # =========================================================================
    # UNLOADING
    # =========================================================================

    async def unload_all(self) -> dict[str, bool]:
        """
        Descargar todos los modelos para liberar VRAM.

        Returns:
            Dict con nombre de modelo -> éxito
        """
        if self._unloading:
            logger.warning("Ya hay una descarga en progreso")
            return {}

        self._unloading = True
        results = {}

        try:
            logger.info("🗑️ Descargando todos los modelos GPU...")
            start_time = time.time()

            for name in list(self.models.keys()):
                try:
                    success = await self._unload_model(name)
                    results[name] = success
                except Exception as e:
                    logger.error(f"Error descargando {name}: {e}")
                    results[name] = False

            # Limpiar CUDA cache
            self._clear_cuda_cache()

            elapsed = time.time() - start_time
            unloaded = sum(1 for v in results.values() if v)
            logger.info(f"✓ Modelos descargados: {unloaded}/{len(results)} en {elapsed:.1f}s")

        finally:
            self._unloading = False

        return results

    async def _unload_model(self, name: str) -> bool:
        """Descargar un modelo específico"""
        if name not in self.models:
            return False

        model_info = self.models[name]

        if model_info.state == ModelState.UNLOADED:
            return True

        model_info.state = ModelState.UNLOADING
        self._notify_state_change(name, ModelState.UNLOADING)

        try:
            # Eliminar referencia al modelo
            if model_info.model_instance is not None:
                # Si es un dict (como STT, emotion), limpiar cada componente
                if isinstance(model_info.model_instance, dict):
                    for key, value in model_info.model_instance.items():
                        if hasattr(value, 'cpu'):
                            value.cpu()
                        del value
                    model_info.model_instance.clear()
                else:
                    # Mover a CPU primero si tiene el método
                    if hasattr(model_info.model_instance, 'cpu'):
                        model_info.model_instance.cpu()
                    # Para vLLM
                    if hasattr(model_info.model_instance, 'shutdown'):
                        model_info.model_instance.shutdown()

                del model_info.model_instance
                model_info.model_instance = None

            # Forzar garbage collection
            gc.collect()

            model_info.state = ModelState.UNLOADED
            self._notify_state_change(name, ModelState.UNLOADED)
            logger.info(f"  ✓ {name} descargado")

            return True

        except Exception as e:
            model_info.state = ModelState.ERROR
            model_info.error_message = str(e)
            self._notify_state_change(name, ModelState.ERROR)
            logger.error(f"  ✗ Error descargando {name}: {e}")
            return False

    def _clear_cuda_cache(self):
        """Limpiar CUDA cache de todas las GPUs"""
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()

                # Forzar garbage collection
                gc.collect()
                torch.cuda.empty_cache()

                logger.info("CUDA cache limpiado")
        except Exception as e:
            logger.warning(f"No se pudo limpiar CUDA cache: {e}")

    # =========================================================================
    # RELOAD
    # =========================================================================

    async def reload_all(self) -> dict[str, bool]:
        """
        Recargar todos los modelos (después del entrenamiento).

        Equivalente a unload_all() + load_all()
        """
        logger.info("🔄 Recargando todos los modelos...")

        # Primero descargar todo (por si hay algo residual)
        await self.unload_all()

        # Esperar un momento para que CUDA libere memoria
        await asyncio.sleep(2)

        # Cargar todo de nuevo
        return await self.load_all()

    # =========================================================================
    # STATUS & UTILITIES
    # =========================================================================

    def get_model(self, name: str) -> Any:
        """Obtener instancia de un modelo cargado"""
        if name not in self.models:
            return None
        model_info = self.models[name]
        if model_info.state != ModelState.LOADED:
            return None
        return model_info.model_instance

    def get_status(self) -> dict:
        """Obtener estado de todos los modelos"""
        status = {
            "loading": self._loading,
            "unloading": self._unloading,
            "models": {}
        }

        total_vram = 0
        for name, info in self.models.items():
            status["models"][name] = {
                "state": info.state.value,
                "gpu": info.gpu_id,
                "vram_mb": info.vram_mb,
                "load_time": info.load_time,
                "error": info.error_message
            }
            if info.state == ModelState.LOADED:
                total_vram += info.vram_mb

        status["total_vram_mb"] = total_vram
        return status

    def get_vram_usage(self) -> dict[int, int]:
        """Obtener uso de VRAM por GPU (MB)"""
        usage = {}
        for info in self.models.values():
            if info.state == ModelState.LOADED:
                gpu = info.gpu_id
                usage[gpu] = usage.get(gpu, 0) + info.vram_mb
        return usage

    def is_all_loaded(self) -> bool:
        """Verificar si todos los modelos están cargados"""
        return all(
            m.state == ModelState.LOADED
            for m in self.models.values()
        )

    def is_all_unloaded(self) -> bool:
        """Verificar si todos los modelos están descargados"""
        return all(
            m.state == ModelState.UNLOADED
            for m in self.models.values()
        )

    def _notify_state_change(self, model_name: str, state: ModelState):
        """Notificar cambio de estado"""
        if self.on_state_change:
            try:
                self.on_state_change(model_name, state)
            except Exception as e:
                logger.warning(f"Error en callback de estado: {e}")


# =============================================================================
# Funciones de conveniencia para integración
# =============================================================================

_global_manager: ModelManager | None = None


def get_model_manager() -> ModelManager | None:
    """Obtener el manager global"""
    return _global_manager


def init_model_manager(config: ModelManagerConfig = None) -> ModelManager:
    """Inicializar el manager global"""
    global _global_manager
    _global_manager = ModelManager(config)
    return _global_manager


async def unload_all_models():
    """Función de conveniencia para descargar todos los modelos"""
    if _global_manager:
        await _global_manager.unload_all()


async def reload_all_models():
    """Función de conveniencia para recargar todos los modelos"""
    if _global_manager:
        await _global_manager.reload_all()
