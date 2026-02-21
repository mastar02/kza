"""
Pipeline Package
Procesamiento completo de voz para domótica inteligente.

Módulos:
- audio_manager: Captura de audio, detección de wake word, VAD
- command_processor: STT, identificación de usuario, clasificación de intent
- response_handler: TTS, streaming, enrutamiento a zonas
- voice_pipeline: Coordinador principal (orquestación)
- model_manager: Gestión centralizada de modelos GPU (carga/descarga)
"""

from src.pipeline.audio_manager import AudioManager
from src.pipeline.command_processor import CommandProcessor
from src.pipeline.response_handler import ResponseHandler
from src.pipeline.voice_pipeline import VoicePipeline
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
    "AudioManager",
    "CommandProcessor",
    "ResponseHandler",
    "VoicePipeline",
    "ModelManager",
    "ModelManagerConfig",
    "ModelState",
    "init_model_manager",
    "get_model_manager",
    "unload_all_models",
    "reload_all_models",
]
