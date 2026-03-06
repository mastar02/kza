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
from src.training.conversation_collector import ConversationCollector
from src.training.habit_dataset_generator import HabitDatasetGenerator
from src.training.nightly_trainer import NightlyTrainer, NightlyConfig
from src.rooms.room_context import RoomContextManager, RoomConfig
from src.presence.presence_detector import PresenceDetector
from src.pipeline.multi_room_audio_loop import MultiRoomAudioLoop, RoomStream
from src.wakeword.detector import WakeWordDetector
from src.lists.list_store import ListStore
from src.lists.list_manager import ListManager
from src.reminders.reminder_store import ReminderStore
from src.reminders.reminder_manager import ReminderManager
from src.reminders.reminder_scheduler import ReminderScheduler

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

    # Lists & Reminders
    lists_config = config.get("lists", {})
    reminders_config = config.get("reminders", {})
    list_store = ListStore(lists_config.get("db_path", "./data/lists.db"))
    await list_store.initialize()
    list_manager = ListManager(store=list_store, ha_client=ha_client, config=lists_config)

    reminder_store = ReminderStore(reminders_config.get("db_path", "./data/reminders.db"))
    await reminder_store.initialize()
    reminder_manager = ReminderManager(store=reminder_store, config=reminders_config)
    reminder_scheduler = None  # Created after presence_detector is ready
    logger.info("Lists & reminders initialized")

    # Multi-room / multi-zone defaults (always defined for DI)
    room_context_manager = None
    presence_detector = None
    multi_room_loop = None

    # Multi-Zone Audio System (Dayton MA1260) — legacy fallback
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
    # Multi-Room Audio System (rooms-based, replaces legacy zones)
    # ----------------------------------------------------------------
    rooms_config = config.get("rooms", {})
    ROOMS_RESERVED_KEYS = {
        "enabled", "cross_validation", "fallback_room",
        "dedup_window_ms", "ma1260", "wake_word", "detection",
    }

    if rooms_config.get("enabled", False):
        # Presence detector
        presence_config = config.get("presence", {})
        if presence_config.get("enabled", False):
            presence_detector = PresenceDetector(
                user_manager=user_manager,
                ha_client=ha_client,
                away_timeout=presence_config.get("away_timeout", 300),
                just_arrived_duration=presence_config.get("just_arrived_duration", 300),
            )
            logger.info("Presence detector created")

        # Reminder scheduler (needs presence_detector)
        if reminder_store:
            reminder_scheduler = ReminderScheduler(
                store=reminder_store, tts=tts,
                presence_detector=presence_detector,
                ha_client=ha_client, config=reminders_config,
            )

        # Room context manager
        room_context_manager = RoomContextManager(
            presence_detector=presence_detector,
            ha_client=ha_client,
            cross_validation=rooms_config.get("cross_validation", True),
            fallback_room=rooms_config.get("fallback_room"),
        )

        # Wake word config for per-room detectors
        room_wake_cfg = rooms_config.get("wake_word", wake_config)
        room_detection_cfg = rooms_config.get("detection", {})

        # Build per-room configs, streams, and presence zones
        room_streams: dict[str, RoomStream] = {}

        for room_key, room_dict in rooms_config.items():
            if room_key in ROOMS_RESERVED_KEYS or not isinstance(room_dict, dict):
                continue

            # Build RoomConfig
            rc = RoomConfig(
                room_id=room_key,
                name=room_dict.get("name", room_key),
                display_name=room_dict.get("display_name", room_key),
                mic_device_index=room_dict.get("mic_device_index"),
                mic_device_name=room_dict.get("mic_device_name"),
                bt_adapter=room_dict.get("bt_adapter"),
                ma1260_zone=room_dict.get("ma1260_zone"),
                output_mode=room_dict.get("output_mode", "mono"),
                default_volume=room_dict.get("default_volume", 50),
                noise_floor=room_dict.get("noise_floor", 0.01),
                default_light=room_dict.get("default_light"),
                default_climate=room_dict.get("default_climate"),
                default_cover=room_dict.get("default_cover"),
                default_media_player=room_dict.get("default_media_player"),
                default_fan=room_dict.get("default_fan"),
                motion_sensor=room_dict.get("motion_sensor"),
                temperature_sensor=room_dict.get("temperature_sensor"),
                humidity_sensor=room_dict.get("humidity_sensor"),
                aliases=room_dict.get("aliases", []),
                tts_speaker=room_dict.get("tts_speaker"),
            )
            room_context_manager.add_room(rc)

            # Build RoomStream if room has a mic
            if rc.mic_device_index is not None:
                wake_detector = WakeWordDetector(
                    models=[room_wake_cfg.get("model", "hey_jarvis")],
                    threshold=room_wake_cfg.get("threshold", 0.5),
                )
                room_echo = EchoSuppressor(sample_rate=16000)
                room_streams[room_key] = RoomStream(
                    room_id=room_key,
                    device_index=rc.mic_device_index,
                    wake_detector=wake_detector,
                    echo_suppressor=room_echo,
                )

            # Register BLE zone for presence
            if rc.bt_adapter and presence_detector:
                presence_detector.add_zone(
                    zone_id=room_key,
                    zone_name=rc.name,
                    ble_adapter=rc.bt_adapter,
                    motion_sensor_entity=rc.motion_sensor,
                )

        # Build ZoneManager from rooms config if not already built from legacy zones
        if not zone_manager:
            ma1260_cfg = rooms_config.get("ma1260", {})
            ma1260 = MA1260Controller(
                connection_type=ma1260_cfg.get("connection_type", "simulation"),
                serial_port=ma1260_cfg.get("serial_port", "/dev/ttyUSB0"),
                baudrate=ma1260_cfg.get("baudrate", 9600),
                ip_address=ma1260_cfg.get("ip_address"),
                ip_port=ma1260_cfg.get("ip_port", 8080),
                audio_output_device=ma1260_cfg.get("audio_output_device"),
                default_source=MA1260Source(ma1260_cfg.get("default_source", 1)),
            )

            zone_list = []
            for room_key, room_dict in rooms_config.items():
                if room_key in ROOMS_RESERVED_KEYS or not isinstance(room_dict, dict):
                    continue
                ma_zone = room_dict.get("ma1260_zone")
                if ma_zone is not None:
                    zone = Zone(
                        id=room_key,
                        name=room_dict.get("name", room_key),
                        mic_device_index=room_dict.get("mic_device_index", 0),
                        ma1260_zone=ma_zone,
                        volume=room_dict.get("default_volume", 50),
                        noise_floor=room_dict.get("noise_floor", 0.01),
                        detection_threshold=room_detection_cfg.get("vad_threshold", 0.02),
                    )
                    zone_list.append(zone)

            if zone_list:
                zone_manager = ZoneManager(
                    zones=zone_list,
                    ma1260_controller=ma1260,
                    detection_window_ms=room_detection_cfg.get("detection_window_ms", 500),
                    priority_mode=room_detection_cfg.get("priority_mode", "loudest"),
                )
                logger.info(f"ZoneManager built from rooms config ({len(zone_list)} zones)")

        # Build MultiRoomAudioLoop if we have room streams
        if room_streams:
            multi_room_loop = MultiRoomAudioLoop(
                room_streams=room_streams,
                follow_up=FollowUpMode(follow_up_window=8.0),
                sample_rate=16000,
                command_duration=2.0,
                dedup_window_ms=rooms_config.get("dedup_window_ms", 200),
            )
            logger.info(
                f"MultiRoomAudioLoop created ({len(room_streams)} rooms: "
                f"{', '.join(room_streams.keys())})"
            )

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
            list_manager=list_manager,
            reminder_manager=reminder_manager,
        )

    # Request router (command routing: orchestrated + legacy paths)
    request_router = RequestRouter(
        command_processor=command_processor,
        response_handler=response_handler,
        audio_manager=audio_manager,
        room_context_manager=room_context_manager,
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

    # ----------------------------------------------------------------
    # Nightly Training (habit learning + QLoRA)
    # ----------------------------------------------------------------
    training_config = config.get("training", {})
    nightly_config_data = training_config.get("nightly", {})
    nightly_trainer = None

    if nightly_config_data.get("enabled", False):
        conversation_collector = ConversationCollector(
            data_dir=training_config.get("conversations_dir", "./data/conversations"),
            auto_save_interval=training_config.get("auto_save_interval", 10),
            max_conversations_in_memory=training_config.get("max_conversations_in_memory", 100),
        )

        habits_config = training_config.get("habits", {})
        habit_generator = HabitDatasetGenerator(
            data_dir=habits_config.get("data_dir", "./data/habit_training"),
            event_logger=event_logger,
            conversation_collector=conversation_collector,
            pattern_learner=None,
            min_confidence=habits_config.get("min_confidence", 0.6),
            synthetic_multiplier=habits_config.get("synthetic_multiplier", 3),
        )

        schedule = nightly_config_data.get("schedule", {})
        lora_cfg = nightly_config_data.get("lora_config", {})
        nightly_cfg = NightlyConfig(
            training_hour=schedule.get("hour", 3),
            training_minute=schedule.get("minute", 0),
            min_samples_to_train=nightly_config_data.get("min_samples_to_train", 20),
            include_unmarked=nightly_config_data.get("include_unmarked", False),
            max_samples_per_session=nightly_config_data.get("max_samples_per_session", 1000),
            base_model=nightly_config_data.get("base_model", "meta-llama/Llama-3.2-3B-Instruct"),
            use_qlora=nightly_config_data.get("qlora", {}).get("enabled", True),
            qlora_bits=nightly_config_data.get("qlora", {}).get("bits", 4),
            lora_r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("alpha", 32),
            lora_dropout=lora_cfg.get("dropout", 0.05),
            epochs=nightly_config_data.get("training_params", {}).get("epochs", 3),
            batch_size=nightly_config_data.get("training_params", {}).get("batch_size", 2),
            output_dir=nightly_config_data.get("output_dir", "./models/lora_adapters/nightly"),
            data_dir=nightly_config_data.get("data_dir", "./data/nightly_training"),
            conversations_dir=training_config.get("conversations_dir", "./data/conversations"),
        )

        def _nightly_alert_callback(level: str, message: str):
            """Sync callback for nightly trainer alerts."""
            log_fn = getattr(logger, level.lower(), logger.info)
            log_fn(f"[NightlyTrainer] {message}")

        nightly_trainer = NightlyTrainer(
            config=nightly_cfg,
            alert_callback=_nightly_alert_callback,
            habit_generator=habit_generator,
        )
        nightly_trainer.start_scheduler()
        logger.info("Nightly trainer scheduled")

    # Use MultiRoomAudioLoop if rooms are configured, else fall back to AudioLoop
    active_audio_loop = multi_room_loop if multi_room_loop else audio_loop

    # Assemble the slim VoicePipeline
    pipeline = VoicePipeline(
        audio_loop=active_audio_loop,
        command_processor=command_processor,
        request_router=request_router,
        response_handler=response_handler,
        feature_manager=feature_manager,
        chroma_sync=chroma,
        memory_manager=memory_manager,
        orchestrator=orchestrator,
    )

    # Start presence detector before pipeline
    if presence_detector:
        await presence_detector.start()
        logger.info("Presence detector started")

    if reminder_scheduler:
        asyncio.create_task(reminder_scheduler.start())
        logger.info("Reminder scheduler started")

    # Ejecutar
    try:
        await pipeline.run()
    except KeyboardInterrupt:
        logger.info("\nDeteniendo...")
        if presence_detector:
            await presence_detector.stop()
        await pipeline.stop()


if __name__ == "__main__":
    asyncio.run(main())
