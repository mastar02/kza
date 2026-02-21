#!/usr/bin/env python3
"""
Home Assistant Voice Control
Entry point principal
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.stt.whisper_fast import create_stt
from src.tts.piper_tts import create_tts
from src.vectordb.chroma_sync import ChromaSync
from src.home_assistant.ha_client import HomeAssistantClient
from src.llm.reasoner import LLMReasoner, FastRouter
from src.routines.routine_manager import RoutineManager
from src.pipeline.voice_pipeline import VoicePipeline
from src.pipeline.audio_loop import AudioLoop
from src.pipeline.audio_manager import AudioManager
from src.pipeline.command_processor import CommandProcessor
from src.pipeline.response_handler import ResponseHandler
from src.pipeline.feature_manager import FeatureManager
from src.pipeline.request_router import RequestRouter
from src.memory.memory_manager import MemoryManager
from src.users.speaker_identifier import SpeakerIdentifier
from src.users.user_manager import UserManager
from src.users.voice_enrollment import VoiceEnrollment
from src.users.emotion_detector import EmotionDetector
from src.monitoring.latency_monitor import LatencyMonitor
from src.analytics.event_logger import EventLogger
from src.analytics.pattern_analyzer import PatternAnalyzer
from src.analytics.suggestion_engine import SuggestionEngine
from src.audio.zone_manager import ZoneManager, Zone
from src.audio.ma1260_controller import MA1260Controller, MA1260Source
from src.audio.echo_suppressor import EchoSuppressor
from src.conversation.follow_up_mode import FollowUpMode
from src.orchestrator import MultiUserOrchestrator
from src.timers.named_timers import NamedTimerManager
from src.intercom.intercom_system import IntercomSystem
from src.notifications.smart_notifications import SmartNotificationManager
from src.alerts.alert_manager import AlertManager
from src.alerts.alert_scheduler import AlertScheduler

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/settings.yaml") -> dict:
    """Cargar configuración desde YAML"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Reemplazar variables de entorno
    def replace_env_vars(obj):
        if isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            var_name = obj[2:-1]
            return os.getenv(var_name, obj)
        elif isinstance(obj, dict):
            return {k: replace_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [replace_env_vars(item) for item in obj]
        return obj

    return replace_env_vars(config)


async def main():
    """Entry point principal"""

    # Cargar variables de entorno
    load_dotenv()

    # Cargar configuración
    config_path = os.getenv("CONFIG_PATH", "config/settings.yaml")
    logger.info(f"Cargando configuración: {config_path}")
    config = load_config(config_path)

    # Verificar Home Assistant
    ha_config = config.get("home_assistant", {})
    ha_url = ha_config.get("url")
    ha_token = ha_config.get("token")

    if not ha_url or not ha_token or ha_token.startswith("${"):
        logger.error("Configura HOME_ASSISTANT_URL y HOME_ASSISTANT_TOKEN")
        logger.error("   Copia .env.example a .env y completa los valores")
        sys.exit(1)

    # Crear componentes
    logger.info("Inicializando componentes...")

    # Home Assistant Client
    ha_client = HomeAssistantClient(
        url=ha_url,
        token=ha_token,
        timeout=ha_config.get("timeout", 2.0)
    )

    # Verificar conexión (async)
    if not await ha_client.test_connection():
        logger.error(f"No se puede conectar a Home Assistant: {ha_url}")
        sys.exit(1)
    logger.info(f"Conectado a Home Assistant: {ha_url}")

    # Speech-to-Text
    stt = create_stt(config.get("stt", {}))

    # Text-to-Speech
    tts = create_tts(config.get("tts", {}))

    # Vector Database
    vectordb_config = config.get("vectordb", {})
    embeddings_config = config.get("embeddings", {})
    chroma = ChromaSync(
        chroma_path=vectordb_config.get("path", "./data/chroma_db"),
        embedder_model=embeddings_config.get("model", "BAAI/bge-small-en-v1.5"),
        embedder_device=embeddings_config.get("device", "cuda:1")
    )

    # LLM Reasoner
    reasoner_config = config.get("reasoner", {})
    model_path = reasoner_config.get("model_path")

    if not model_path or not Path(model_path).exists():
        logger.warning(f"Modelo LLM no encontrado: {model_path}")
        logger.warning("   Ejecuta: ./scripts/download_models.sh")
        llm = None
    else:
        llm = LLMReasoner(
            model_path=model_path,
            n_ctx=reasoner_config.get("n_ctx", 8192),
            n_threads=reasoner_config.get("n_threads", 24)
        )

    # Routine Manager
    routine_manager = RoutineManager(ha_client, chroma, llm)

    # Fast Router (GPU 2) - para clasificación rápida con KV-Cache
    router_config = config.get("router", {})
    fast_router = None
    if router_config.get("enabled", True):
        fast_router = FastRouter(
            model=router_config.get("model", "Qwen/Qwen2.5-7B-Instruct"),
            device=router_config.get("device", "cuda:2"),
            gpu_memory_utilization=router_config.get("gpu_memory_utilization", 0.85),
            enable_prefix_caching=router_config.get("enable_prefix_caching", True)
        )
        logger.info(f"Fast router habilitado (prefix_caching={router_config.get('enable_prefix_caching', True)})")

    # Memory Manager - memoria contextual
    memory_config = config.get("memory", {})
    memory_manager = None
    if memory_config.get("enabled", True):
        memory_manager = MemoryManager(
            chroma_path=memory_config.get("chroma_path", "./data/memory_db"),
            preferences_path=memory_config.get("preferences_path", "./data/preferences.json"),
            short_term_size=memory_config.get("short_term_size", 10)
        )
        logger.info("Memory manager habilitado")

    # Speaker Identification (GPU 1)
    speaker_config = config.get("speaker_id", {})
    speaker_identifier = None
    user_manager = None
    voice_enrollment = None

    if speaker_config.get("enabled", True):
        speaker_identifier = SpeakerIdentifier(
            model_name=speaker_config.get("model", "speechbrain/spkrec-ecapa-voxceleb"),
            device=speaker_config.get("device", "cuda:1"),
            similarity_threshold=speaker_config.get("threshold", 0.75)
        )

        user_manager = UserManager(
            data_path=speaker_config.get("users_path", "./data/users.json")
        )

        voice_enrollment = VoiceEnrollment(
            user_manager=user_manager,
            speaker_identifier=speaker_identifier
        )
        logger.info("Speaker identification habilitado")

    # Emotion Detection (GPU 1 — shared with speaker ID)
    emotion_detector = EmotionDetector(
        device=speaker_config.get("device", "cuda:1"),
        sample_rate=16000,
    )
    logger.info("Emotion detector habilitado")

    # Voice Pipeline config
    wake_config = config.get("wake_word", {})
    latency_config = config.get("latency_targets", {})

    # Latency Monitor
    monitoring_config = config.get("monitoring", {})
    latency_monitor = None
    if monitoring_config.get("enabled", True):
        def on_latency_alert(record):
            logger.warning(f"Latencia alta: {record.total_ms:.0f}ms (target: {record.target_ms}ms)")

        latency_monitor = LatencyMonitor(
            db_path=monitoring_config.get("db_path", "./data/latency.db"),
            target_ms=float(latency_config.get("total", 300)),
            alert_callback=on_latency_alert
        )
        logger.info("Latency monitor habilitado")

    # Smart Automations - Event Logger y Suggestion Engine
    analytics_config = config.get("analytics", {})
    event_logger = None
    suggestion_engine = None

    if analytics_config.get("enabled", True):
        event_logger = EventLogger(
            db_path=analytics_config.get("events_db", "./data/events.db"),
            retention_days=analytics_config.get("retention_days", 90)
        )

        pattern_analyzer = PatternAnalyzer(event_logger)

        suggestion_engine = SuggestionEngine(
            event_logger=event_logger,
            pattern_analyzer=pattern_analyzer,
            suggestions_path=analytics_config.get("suggestions_path", "./data/suggestions.json"),
            min_confidence=analytics_config.get("min_confidence", 0.7)
        )
        logger.info("Smart automations habilitado")

    # Multi-Zone Audio System (Dayton MA1260)
    zones_config = config.get("zones", {})
    zone_manager = None

    if zones_config.get("enabled", False):
        # Crear controlador MA1260
        ma1260_config = zones_config.get("ma1260", {})
        ma1260 = MA1260Controller(
            connection_type=ma1260_config.get("connection_type", "simulation"),
            serial_port=ma1260_config.get("serial_port", "/dev/ttyUSB0"),
            baudrate=ma1260_config.get("baudrate", 9600),
            ip_address=ma1260_config.get("ip_address"),
            ip_port=ma1260_config.get("ip_port", 8080),
            audio_output_device=ma1260_config.get("audio_output_device"),
            default_source=MA1260Source(ma1260_config.get("default_source", 1))
        )

        # Crear zonas
        detection_config = zones_config.get("detection", {})
        zone_list = []
        for zone_cfg in zones_config.get("zone_list", []):
            zone = Zone(
                id=zone_cfg["id"],
                name=zone_cfg["name"],
                mic_device_index=zone_cfg["mic_device_index"],
                ma1260_zone=zone_cfg["ma1260_zone"],
                volume=zone_cfg.get("default_volume", 50),
                noise_floor=zone_cfg.get("noise_floor", 0.01),
                detection_threshold=detection_config.get("vad_threshold", 0.02)
            )
            zone_list.append(zone)

        # Crear zone manager
        zone_manager = ZoneManager(
            zones=zone_list,
            ma1260_controller=ma1260,
            detection_window_ms=detection_config.get("detection_window_ms", 500),
            priority_mode=detection_config.get("priority_mode", "loudest")
        )
        logger.info(f"Multi-zone audio habilitado ({len(zone_list)} zonas)")

    # ----------------------------------------------------------------
    # Build pipeline components (DI chain)
    # ----------------------------------------------------------------

    # TTS / streaming config
    tts_config = config.get("tts", {})
    streaming_config = tts_config.get("streaming", {})

    # Audio manager (wake word + VAD)
    audio_manager = AudioManager(
        zone_manager=zone_manager,
        wake_word_model=wake_config.get("model", "hey_jarvis"),
        wake_word_threshold=wake_config.get("threshold", 0.5),
        sample_rate=16000,
        command_duration=2.0,
    )

    # Command processor (STT + speaker ID + emotion)
    command_processor = CommandProcessor(
        stt=stt,
        speaker_identifier=speaker_identifier,
        user_manager=user_manager,
        emotion_detector=emotion_detector,
        sample_rate=16000,
    )

    # Response handler (TTS + streaming + zone routing)
    response_handler = ResponseHandler(
        tts=tts,
        zone_manager=zone_manager,
        llm=llm,
        streaming_enabled=streaming_config.get("enabled", True),
        streaming_buffer_ms=streaming_config.get("buffer_ms", 150),
        streaming_prebuffer_ms=streaming_config.get("prebuffer_ms", 80),
    )

    # Echo suppressor and follow-up mode (needed by AudioLoop)
    echo_suppressor = EchoSuppressor(sample_rate=16000)
    follow_up = FollowUpMode(follow_up_window=8.0)

    # Audio loop (capture + wake word + echo suppression + follow-up)
    audio_loop = AudioLoop(
        audio_manager=audio_manager,
        echo_suppressor=echo_suppressor,
        follow_up=follow_up,
        sample_rate=16000,
    )

    # Multi-user orchestrator (optional)
    orchestrator = None
    orchestrator_enabled = config.get("orchestrator", {}).get("enabled", True)
    if orchestrator_enabled:
        orchestrator = MultiUserOrchestrator(
            chroma_sync=chroma,
            ha_client=ha_client,
            routine_manager=routine_manager,
            router=fast_router,
            llm=llm,
            tts=tts,
            speaker_identifier=speaker_identifier,
            user_manager=user_manager,
        )

    # Request router (command routing: orchestrated + legacy paths)
    request_router = RequestRouter(
        command_processor=command_processor,
        response_handler=response_handler,
        audio_manager=audio_manager,
        orchestrator=orchestrator,
        orchestrator_enabled=orchestrator_enabled,
        chroma_sync=chroma,
        ha_client=ha_client,
        llm_reasoner=llm,
        fast_router=fast_router,
        memory_manager=memory_manager,
        user_manager=user_manager,
        enrollment=voice_enrollment,
        event_logger=event_logger,
        suggestion_engine=suggestion_engine,
        latency_monitor=latency_monitor,
        routine_manager=routine_manager,
        vector_search_threshold=vectordb_config.get("search_threshold", 0.65),
        latency_target_ms=latency_config.get("total", 300),
        suggestion_interval=analytics_config.get("suggestion_interval", 20),
    )

    # Feature subsystems (timers, intercom, notifications, alerts)
    alerts_config = config.get("alerts", {})
    alert_manager = None
    alert_scheduler = None
    timer_manager = None
    intercom = None
    notifications = None

    if alerts_config.get("enabled", True):
        general = alerts_config.get("general", {})
        alert_manager = AlertManager(
            cooldown_seconds=general.get("cooldown_seconds", 300.0),
            max_history=general.get("max_history", 1000),
        )

        sched = alerts_config.get("scheduler", {})
        if sched.get("enabled", True):
            alert_scheduler = AlertScheduler(
                alert_manager=alert_manager,
                security_interval=sched.get("security_interval_seconds", 60.0),
                pattern_interval=sched.get("pattern_interval_seconds", 300.0),
                device_interval=sched.get("device_interval_seconds", 600.0),
            )
        logger.info("Alert system enabled")

    timer_manager = NamedTimerManager()
    intercom = IntercomSystem(
        zone_manager=zone_manager,
        ha_client=ha_client,
    )
    notifications = SmartNotificationManager(
        user_manager=user_manager,
        ha_client=ha_client,
    )

    feature_manager = FeatureManager(
        timer_manager=timer_manager,
        intercom=intercom,
        notifications=notifications,
        alert_manager=alert_manager,
        alert_scheduler=alert_scheduler,
    )

    # Assemble the slim VoicePipeline
    pipeline = VoicePipeline(
        audio_loop=audio_loop,
        command_processor=command_processor,
        request_router=request_router,
        response_handler=response_handler,
        feature_manager=feature_manager,
        chroma_sync=chroma,
        memory_manager=memory_manager,
        orchestrator=orchestrator,
    )

    # Ejecutar
    try:
        await pipeline.run()
    except KeyboardInterrupt:
        logger.info("\nDeteniendo...")
        await pipeline.stop()


if __name__ == "__main__":
    asyncio.run(main())
