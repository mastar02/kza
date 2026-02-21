"""
Pipeline Package
Procesamiento completo de voz para domótica inteligente.

Módulos:
- audio_manager: Captura de audio, detección de wake word, VAD
- command_processor: STT, identificación de usuario, clasificación de intent
- response_handler: TTS, streaming, enrutamiento a zonas
- voice_pipeline: Coordinador principal (orquestación)
- request_router: Command routing (orchestrated + legacy paths)
- model_manager: Gestión centralizada de modelos GPU (carga/descarga)
- feature_manager: Lifecycle for timers, intercom, notifications, alerts, HA
- audio_loop: Main audio capture and wake word detection loop
"""

from src.pipeline.audio_loop import AudioLoop
from src.pipeline.audio_manager import AudioManager
from src.pipeline.command_processor import CommandProcessor
from src.pipeline.response_handler import ResponseHandler
from src.pipeline.request_router import RequestRouter
from src.pipeline.voice_pipeline import VoicePipeline
from src.pipeline.feature_manager import FeatureManager
from src.pipeline.model_manager import (
    ModelManager,
    ModelManagerConfig,
    ModelState,
    init_model_manager,
    get_model_manager,
    unload_all_models,
    reload_all_models
)

__all__ = [
    "AudioLoop",
    "AudioManager",
    "CommandProcessor",
    "ResponseHandler",
    "RequestRouter",
    "VoicePipeline",
    "FeatureManager",
    "ModelManager",
    "ModelManagerConfig",
    "ModelState",
    "init_model_manager",
    "get_model_manager",
    "unload_all_models",
    "reload_all_models",
]
