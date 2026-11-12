#!/usr/bin/env python3
"""
Home Assistant Voice Control
Entry point principal
"""
# ruff: noqa: E402 -- the managed entry is armed before the (~5 s of) imports below.
if __name__ == "__main__":
    # U3b-0d1: unblock SIGTERM/SIGINT and record an early stop before any
    # heavy import; process_lifecycle is stdlib-only.
    from src.core.process_lifecycle import arm_process_entry
    _ENTRY = arm_process_entry()

from src.memory import output_registration_v2 as output_registration
import asyncio
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
import logging
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from dotenv import load_dotenv

from src.application import (
    ApplicationFactories, ApplicationGraph, ApplicationModels,
    HttpReasonerOwner, SpeakerModelOwner, STTExecutorOwner, TTSModelOwner, WakeModelOwner,
    HermesReasonerOwner as _HermesReasonerOwner,
)
from src.core.egress import CloudConsentGate, ConsentWithdrawalSource
from src.core.owned_work import OwnedBlockingWork
from src.core.paths import TRAINING_OUTPUT_NIGHTLY, models_path, speaker_savedir
from src.core.process_lifecycle import ProcessLifecycle, StopPhase, arm_process_entry
from src.core.runtime_admission import TICKET_ENV, seal_runtime_configuration, verify_runtime_admission
from src.core.state_configuration import ConfigurationChanges
from src.core.settings_schema import (
    DEFAULT_LOCAL_LLM_GATEWAY,
    check_unresolved_env_vars,
    resolve_env_vars,
    validate_settings,
)
from src.stt.whisper_fast import create_stt
from src.tts.piper_tts import create_tts
from src.vectordb.chroma_sync import ChromaSync
from src.home_assistant.ha_client import HomeAssistantClient
from src.llm.reasoner import LLMReasoner, FastRouter
from src.routines.routine_manager import RoutineManager
from src.routines.factory import build_routine_scheduler
from src.routines.routine_scheduler import RoutineScheduler
from src.home_assistant.home_actions import HomeActionService
from src.home_assistant.capability_service import HomeCapabilityService
from src.home_assistant.voice_resolver import HomeVoiceResolver
from src.home_assistant.control_policy import HomeControlPolicy
from src.pipeline.voice_pipeline import VoicePipeline
from src.pipeline.audio_loop import AudioLoop
from src.pipeline.audio_manager import AudioManager
from src.pipeline.command_processor import CommandProcessor
from src.pipeline.voice_services import build_voice_services
from src.pipeline.request_router import RequestRouter
from src.memory.services import build_memory_services
from src.memory.privacy_services import build_privacy_services
from src.memory.index_erasure import build_erasure_services
from src.ambient.erasure import build_capture_erasure
from src.users.speaker_identifier import SpeakerIdentifier
from src.users.voice_registry import build_voice_registry
from src.users.voice_attributor import build_voice_attribution
from src.users.voice_enrollment import VoiceEnrollment
from src.users.emotion_detector import EmotionDetector
from src.monitoring.latency_monitor import LatencyMonitor
from src.monitoring.loop_lag import LoopLagConfig, LoopLagProbe
from src.analytics.services import build_event_logger
from src.analytics.erasure import build_event_erasure
from src.obs.diagnostic_erasure import build_diagnostic_services
from src.analytics.suggestion_erasure import build_suggestion_erasure
from src.wakeword.clip_store import build_wake_clip_store
from src.orchestrator.context_erasure import build_context_erasure
from src.analytics.pattern_analyzer import PatternAnalyzer
from src.analytics.suggestion_engine import SuggestionEngine
from src.audio.zone_manager import ZoneManager, Zone
from src.audio.ma1260_controller import MA1260Controller, MA1260Source
from src.audio.echo_suppressor import EchoSuppressor
from src.audio.direction_observer import parse_direction_observation
from src.conversation.follow_up_mode import FollowUpMode
from src.orchestrator import MultiUserOrchestrator
from src.orchestrator.context_manager import ContextManager
from src.world.weather import DEFAULT_ENTITY as DEFAULT_WEATHER_ENTITY
from src.training.conversation_collector import build_conversation_collector
from src.training.stt_correction_collector import build_stt_correction_collector
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
from src.dashboard.api import DashboardAPI
from src.dashboard.authority import DashboardAuthority
from src.dashboard.home_api import build_home_services, build_microphone_rooms
from src.dashboard.live_erasure import build_live_services
from src.monitoring.health_aggregator import HealthAggregator
from src.nlu.command_gate import CommandAcceptanceGate

if TYPE_CHECKING:
    from src.hooks.registry import HookRegistry

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

    # Fail-fast al boot: una config rota aborta acá con el detalle de cada
    # campo inválido, en vez de morir a mitad de la cadena de DI. Los
    # placeholders ${VAR} sin resolver abortan solo si son de HA (esencial);
    # el resto loguea WARNING (p. ej. reasoner.http_base_url tiene su propio
    # fallback seguro más abajo, en el punto de consumo).
    config = resolve_env_vars(config)
    check_unresolved_env_vars(config)
    return validate_settings(config)


def _build_xvf_controllers(
    room_streams: dict, spe_cfg: dict, tuning_cfg: dict, direction_room_ids: tuple[str, ...],
) -> dict:
    """Construct one port-bound controller per room needing gate, tuning, or DoA.

    Construction never opens USB or starts the legacy SPENERGY poller.
    """
    from src.audio.xvf_controller import XvfController

    legacy_needed = spe_cfg.get('enabled', False) or (
        tuning_cfg.get('apply_on_start', False) and tuning_cfg.get('params')
    )
    controllers = {}
    for room_id, stream in room_streams.items():
        if not legacy_needed and room_id not in direction_room_ids:
            continue
        if not stream.mic_usb_port:
            logger.warning('[XVF] %s has no mic_usb_port; gate/tuning/direction unavailable', room_id)
            continue
        controllers[room_id] = XvfController(
            usb_port=stream.mic_usb_port,
            poll_interval_s=spe_cfg.get('poll_interval_s', 0.04),
            window_s=spe_cfg.get('window_s', 4.0),
        )
    return controllers


async def _warmup_models(stt, tts, speaker_identifier, emotion_detector, chroma, *,
                         voice_output=None, command_processor=None):
    """Precalentar modelos con input dummy para evitar cold-start lag.

    CUDA compila kernels la primera vez que se usa un modelo — este warmup
    hace que todas las compilaciones ocurran al startup en vez de en el
    primer comando real del usuario. Cada warmup está en try/except
    individual: si un modelo falla, se loguea y se sigue con el resto.

    Args:
        stt: Instancia de STT (FastWhisperSTT) con método transcribe.
        tts: Instancia de TTS con método synthesize.
        speaker_identifier: Opcional. Instancia con método get_embedding.
        emotion_detector: Opcional. Instancia con método detect.
        chroma: Opcional. Instancia de ChromaSync con método warmup_embedder().
        command_processor: Dueño de los modelos compartidos; retiene el trabajo
            nativo de warmup aunque se cancele el arranque de la aplicación.
    """
    import numpy as np
    silence = np.zeros(16000, dtype=np.float32)  # 1s @ 16kHz

    timings = {}

    async def run_model(model, function, *args, **kwargs):
        if command_processor is not None:
            return await command_processor.awarmup_model(model, function, *args, **kwargs)
        return await asyncio.to_thread(function, *args, **kwargs)

    # STT warmup — compila kernels de Whisper
    if stt is not None:
        t0 = time.perf_counter()
        try:
            await run_model(stt, stt.transcribe, silence, sample_rate=16000)
            timings["stt"] = (time.perf_counter() - t0) * 1000
        except Exception as e:
            logger.warning(f"Warmup STT skipped: {e}")

    # TTS warmup — compila kernels de Kokoro/Qwen3
    if tts is not None:
        t0 = time.perf_counter()
        try:
            if voice_output is not None:
                maybe = await voice_output.asynthesize("hola")
            else:
                maybe = await asyncio.to_thread(tts.synthesize, "hola") if hasattr(tts, "synthesize") else None
            if asyncio.iscoroutine(maybe):
                await maybe
            timings["tts"] = (time.perf_counter() - t0) * 1000
        except Exception as e:
            logger.warning(
                f"Warmup TTS skipped (non-fatal): {type(e).__name__}: {e}",
                exc_info=True,
            )

    # Speaker ID warmup — compila kernels de ECAPA-TDNN
    if speaker_identifier is not None:
        t0 = time.perf_counter()
        try:
            await run_model(speaker_identifier, speaker_identifier.get_embedding, silence)
            timings["speaker_id"] = (time.perf_counter() - t0) * 1000
        except Exception as e:
            logger.warning(f"Warmup speaker_id skipped: {e}")

    # Emotion detector warmup — compila kernels de wav2vec2
    if emotion_detector is not None:
        t0 = time.perf_counter()
        try:
            if hasattr(emotion_detector, "detect"):
                await run_model(emotion_detector, emotion_detector.detect, silence)
            timings["emotion"] = (time.perf_counter() - t0) * 1000
        except Exception as e:
            logger.warning(f"Warmup emotion skipped: {e}")

    # BGE-M3 warmup — compila kernels del embedder usado por ChromaDB.
    # Usa warmup_embedder() (vía property) en vez del guard lazy `_embedder`,
    # que daba False y salteaba el warmup → el primer comando pagaba ~48ms cold.
    if chroma is not None:
        try:
            timings["bge_m3"] = await asyncio.to_thread(chroma.warmup_embedder)
        except Exception as e:
            logger.warning(f"Warmup BGE-M3 skipped: {e}")

    summary = " ".join(f"{k}={v:.0f}ms" for k, v in timings.items())
    logger.info(f"Warmup: {summary}")


def _preflight_vram(threshold_mib: int = 1500) -> None:
    """Avisa (no bloquea) si la VRAM libre en algún device CUDA está al límite.

    Motivado por el OOM de 2026-05-29: reiniciar kza-voice con poca VRAM libre
    en cuda:1 daba un 'CUDA out of memory' críptico a mitad de la carga de
    modelos. Este preflight loguea un WARNING accionable ANTES de cargar, para
    que el operador libere VRAM o entienda el riesgo. No-op si torch no tiene
    CUDA (ej: dev en macOS). Ver project_stt_double_load_oom_2026-05-29.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return
        for i in range(torch.cuda.device_count()):
            free_b, _total_b = torch.cuda.mem_get_info(i)
            free_mib = free_b / (1024 * 1024)
            if free_mib < threshold_mib:
                logger.warning(
                    f"⚠️  VRAM ajustada en cuda:{i}: {free_mib:.0f} MiB libres "
                    f"(< {threshold_mib} MiB). Cargar los modelos de voz puede "
                    f"OOM-ear en este restart — liberá VRAM si el arranque falla."
                )
            else:
                logger.info(
                    f"Preflight VRAM cuda:{i}: {free_mib:.0f} MiB libres (OK)"
                )
    except Exception as e:
        logger.debug(f"Preflight VRAM skipped: {e}")


async def _initialize_routines(
    config: dict, *, ha_client, home_actions: HomeActionService, shutdown: AsyncExitStack,
    start: bool = True,
) -> RoutineScheduler | None:
    """Build persistent routines with shared Casa services and owned cleanup.

    Args:
        config: Central application configuration.
        ha_client: Shared Home Assistant client owned by main.
        home_actions: Shared Casa action service owned by main.
        shutdown: Application lifecycle stack that outlives web admission.
        start: Start immediately for existing callers. The application factory
            defers this until all construction and warmup have succeeded.

    Returns:
        An initialized scheduler, or None when routines are unavailable. Other
        application subsystems can continue after an initialization failure.
    """
    cleanup = AsyncExitStack()
    transferred = False
    try:
        routines_config = config.get("routines", {})
        scheduler = await build_routine_scheduler(
            db_path=routines_config.get("db_path", "./data/routines.db"),
            timezone=routines_config.get("timezone", "America/Montevideo"),
            ha_client=ha_client,
            home_actions=home_actions,
        )
        cleanup.push_async_callback(scheduler.store.close)
        cleanup.push_async_callback(scheduler.stop)
        if start:
            await scheduler.start()
        shutdown.push_async_callback(cleanup.pop_all().aclose)
        transferred = True
        return scheduler
    except Exception:
        # Keep private database paths and dependency exception text out of logs.
        logger.error("Routine subsystem unavailable after initialization failure")
        return None
    finally:
        if not transferred:
            # Startup cancellation must finish owned cleanup before main closes
            # shared Casa/HA, even if another cancellation arrives during I/O.
            closing = asyncio.create_task(cleanup.aclose())
            cancelled = False
            while not closing.done():
                try:
                    await asyncio.shield(closing)
                except asyncio.CancelledError:
                    cancelled = True
                except Exception:
                    # aclose has exhausted its callbacks; observe and report
                    # the failed task below without overriding degradation.
                    break
            try:
                closing.result()
            except Exception:
                logger.error("Routine subsystem cleanup failed")
            if cancelled:
                raise asyncio.CancelledError


async def _prune_retired_memory_owners(privacy_services, dashboard_authority) -> int:
    """Re-run the D7 retirement cascade at boot, after the authority is loaded (U3b-1b).

    Deletes facts still owned by a durably retired account (a crash between
    retirement and cleanup, or rows brought back by a restore). Failure keeps
    the rows and is logged: assignment to such a person is already refused,
    and account removal stays blocked by the owner proof. An authority that
    cannot answer counts as a failure (review R1). The privacy maintenance
    tick retries a pending prune, runs it hourly regardless, and operators
    see it pending (F2).
    """
    privacy_services.attach_retired_owner_prune(dashboard_authority.users_retired,
                                                dashboard_authority.memory_owner_retired)
    try:
        deleted = await privacy_services.acall(privacy_services.ledger.prune_retired_owners,
                                               dashboard_authority.users_retired,
                                               probe=dashboard_authority.memory_owner_retired)
    except Exception as error:
        logger.error(f"[MemoryOwners] retired-owner prune failed at boot: {type(error).__name__}")
        privacy_services.defer_retired_owner_prune()
        return 0
    if deleted:
        logger.info(f"[MemoryOwners] boot prune deleted {deleted} facts of retired accounts")
    return deleted


async def main(*, managed_exit: bool = False, lifecycle: ProcessLifecycle | None = None,
               build: Callable[[AsyncExitStack, ProcessLifecycle], Awaitable[ApplicationGraph]] | None = None,
               ) -> None:
    """Own shutdown even when construction or warmup fails before voice runs.

    SIGTERM is routed to the lifecycle's stop signal before the graph is
    built. A stop in any phase finishes construction, skips the remaining
    start-up and closes the owned stack normally, so a managed admission still
    writes its receipt.

    Args:
        managed_exit: Set by the managed entry. After a clean cleanup with a
            written receipt, flush and ``os._exit(0)`` so no executor or
            worker join can outlive the receipt. In-process callers return.
        lifecycle: Stop/readiness contract; a fresh one when omitted.
        build: Graph builder taking (shutdown, lifecycle); defaults to the
            central configuration and ``build_application``.
    """
    lifecycle = lifecycle if lifecycle is not None else ProcessLifecycle()
    lifecycle.stop.install(asyncio.get_running_loop())
    shutdown = AsyncExitStack()
    try:
        await _run_main(shutdown, lifecycle=lifecycle, build=build)
    finally:
        lifecycle.stop.enter(StopPhase.CLOSING)
        # Repeated cancellation must not close HA while action sends still drain.
        cleanup = asyncio.create_task(shutdown.aclose())
        cancelled = False
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                cancelled = True
        cleanup.result()
        if cancelled:
            raise asyncio.CancelledError
    if managed_exit and lifecycle.receipt_written:
        # N5: only here, after cleanup and the receipt; never in a callback.
        lifecycle.flush_for_exit()
        os._exit(0)


async def _build_configured_application(shutdown: AsyncExitStack,
                                        lifecycle: ProcessLifecycle | None) -> ApplicationGraph:
    load_dotenv()
    config_path = os.getenv("CONFIG_PATH", "config/settings.yaml")
    logger.info("Loading application configuration: %s", config_path)
    return await build_application(load_config(config_path), shutdown=shutdown, lifecycle=lifecycle)


async def _run_main(shutdown: AsyncExitStack, *, lifecycle: ProcessLifecycle | None = None,
                    build: Callable[..., Awaitable[ApplicationGraph]] | None = None) -> None:
    """Load central configuration, build the shared graph and run it."""
    stop = None if lifecycle is None else lifecycle.stop
    if stop is not None:
        stop.enter(StopPhase.CONSTRUCTION)
    if build is None and lifecycle is None:
        load_dotenv()
        config_path = os.getenv("CONFIG_PATH", "config/settings.yaml")
        logger.info("Loading application configuration: %s", config_path)
        graph = await build_application(load_config(config_path), shutdown=shutdown)
    else:
        graph = await (build or _build_configured_application)(shutdown, lifecycle)
    if lifecycle is not None and (graph.lifecycle is not lifecycle
                                  or getattr(graph.pipeline, "lifecycle", None) is not lifecycle):
        # A custom builder must wire both the graph and the pipeline it runs.
        raise RuntimeError("main_stop_signal_not_wired")
    if stop is not None and stop.requested:
        logger.info("[Stop] construction finished after a stop; skipping warmup and voice")
    else:
        if stop is not None:
            stop.enter(StopPhase.WARMUP)
        await graph.warmup()
    await graph.run()


def run_entry(lifecycle: ProcessLifecycle | None = None,
              build: Callable[[AsyncExitStack, ProcessLifecycle], Awaitable[ApplicationGraph]] | None = None,
              ) -> None:
    """The managed process entry: arm signals first, then run main()."""
    lifecycle = lifecycle if lifecycle is not None else arm_process_entry()
    asyncio.run(main(managed_exit=os.environ.get(TICKET_ENV) is not None,
                     lifecycle=lifecycle, build=build))


def boot_consent_decision(registration) -> dict | None:
    """The consent decision this run recorded in its journal (None when unmanaged)."""
    return None if registration is None else getattr(registration, "effective_decision", None)


def apply_boot_consent_decision(gate: CloudConsentGate, registration) -> None:
    """Close the runtime gate iff the recorded boot decision keeps a withdrawal in force.

    The decision is the coordinator's single source of truth (F10): it is
    None whenever reasoner.cloud.consent is not True (settling) or no marker
    is authoritative; ``basis`` actor or unverifiable closes the gate.
    """
    decision = boot_consent_decision(registration)
    if decision is None:
        return
    source = (ConsentWithdrawalSource.DURABLE_MARKER if decision["basis"] == "actor"
              else ConsentWithdrawalSource.MARKER_UNVERIFIABLE)
    gate.withdraw(source)
    logger.warning("[Consent] Cloud consent withdrawn at boot by the journal decision (basis=%s)",
                   decision["basis"])


def _reasoner_is_cloud(reasoner_config: dict, reasoner_mode: str) -> bool:
    """m4: Hermes is always cloud; HTTP is cloud unless declared LOCAL for its endpoint."""
    from src.llm.cloud_consent import is_cloud_endpoint
    if reasoner_mode == "hermes_cli":
        return True
    if reasoner_mode != "http":
        return False
    return is_cloud_endpoint(reasoner_config.get("http_base_url", DEFAULT_LOCAL_LLM_GATEWAY),
                             destination=reasoner_config.get("destination", "cloud"))


def load_policy_hooks(hooks_cfg: dict, shutdown: AsyncExitStack) -> "HookRegistry | None":
    """Load the policy hooks when ``hooks.enabled`` is true.

    Imports every module of ``hooks.policies_module`` (safety_alarm and
    tts_rewrite_es register at import), then registers the SQLite audit only
    when ``hooks.audit_sqlite.enabled`` is true (U3b-0e N1); its close goes on
    ``shutdown``.

    Args:
        hooks_cfg: The ``hooks`` section of the central configuration.
        shutdown: The application's owned exit stack.

    Returns:
        The global registry with at least one handler, or None.
    """
    hooks = None
    if hooks_cfg.get("enabled", False):
        import pkgutil
        import importlib

        try:
            from src.hooks import _global_registry
        except Exception as e:
            logger.error(
                f"[main] Hooks disabled — failed to import src.hooks: {e}",
                exc_info=True,
            )
            hooks = None
        else:
            # Per-policy iteration: a single broken policy must NOT disable the
            # entire hook system. We log each failure individually and continue.
            policies_module_path = hooks_cfg.get("policies_module", "src.policies")
            try:
                pkg = importlib.import_module(policies_module_path)
            except Exception as e:
                logger.error(
                    f"[main] Hooks disabled — failed to import policies package "
                    f"'{policies_module_path}': {e}",
                    exc_info=True,
                )
                hooks = None
            else:
                loaded, failed = [], []
                for mod_info in pkgutil.iter_modules(pkg.__path__):
                    mod_name = f"{policies_module_path}.{mod_info.name}"
                    try:
                        importlib.import_module(mod_name)
                        loaded.append(mod_info.name)
                    except Exception as e:
                        failed.append((mod_info.name, str(e)))
                        logger.error(
                            f"[main] Policy '{mod_name}' failed to load: {e}",
                            exc_info=True,
                        )
                if failed:
                    logger.error(
                        f"[main] Hooks PARTIAL: {len(failed)} policy(ies) failed: "
                        f"{[n for n, _ in failed]}; loaded: {loaded}"
                    )
                # U3b-0e N1: the audit has its own switch; importing the policy
                # registers nothing. Off: no handler and no path is opened.
                from src.policies.audit_sqlite import register_audit_sqlite
                audit_policy = register_audit_sqlite(_global_registry, hooks_cfg)
                if audit_policy is not None:
                    shutdown.push_async_callback(asyncio.to_thread, audit_policy.close)
                stats = _global_registry.get_stats()
                total_handlers = (
                    sum(stats["before_handler_count"].values())
                    + sum(stats["after_handler_count"].values())
                )
                if total_handlers == 0:
                    logger.error(
                        f"[main] Hooks ENABLED but zero handlers registered "
                        f"(loaded: {loaded}). Check policy files for errors. "
                        "Disabling hooks to avoid masquerade."
                    )
                    hooks = None
                else:
                    hooks = _global_registry
                    logger.info(
                        f"[main] Plugin hooks enabled — "
                        f"before_ha={stats['before_handler_count']['before_ha_action']} "
                        f"before_tts={stats['before_handler_count']['before_tts_speak']} "
                        f"after_handlers={sum(stats['after_handler_count'].values())} "
                        f"loaded={loaded}"
                    )
    return hooks


async def build_application(
    config: dict, *, shutdown: AsyncExitStack,
    factories: ApplicationFactories | None = None,
    lifecycle: ProcessLifecycle | None = None,
) -> ApplicationGraph:
    """Construct the actual application graph without starting house producers.

    Args:
        config: Validated central application configuration, or an explicit
            offline fixture. The factory never loads dotenv or another config.
        shutdown: Caller-owned stack; it must be closed on every exit, including
            factory cancellation. Private storage and retention start here.
        factories: Optional leaf adapters. Unspecified constructors resolve
            this module's globals at call time, including existing test patches.
        lifecycle: Optional process stop/readiness contract. The graph and the
            pipeline observe its stop signal; READY follows 'System ready'; a
            managed admission closes logging before its receipt.

    Returns:
        The same graph run by main, with operational startup and warmup deferred.
    """
    config = seal_runtime_configuration(config)
    admission = await asyncio.to_thread(verify_runtime_admission, config)
    destination_registration = None
    if admission is not None:
        admission.install_on(shutdown)
        if lifecycle is not None:
            lifecycle.attach_admission(admission)
        destination_registration = await admission.begin_destination_registration(config)
    # OQ-8: one runtime gate shared by every cloud-capable client; closed now
    # when the journal's recorded decision keeps a withdrawal in force.
    consent_gate = CloudConsentGate()
    apply_boot_consent_decision(consent_gate, destination_registration)
    factories = factories or ApplicationFactories()

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

    # Register authority ownership before every consumer. Opening the private
    # database is deferred until construction reaches the dashboard boundary.
    dashboard_authority = None
    configuration_changes = None
    user_manager = None
    if config.get("dashboard", {}).get("enabled", True):
        auth_config = config.get("dashboard", {}).get("auth", {})
        dashboard_authority = DashboardAuthority(
            auth_config.get("db_path", "./data/dashboard_auth.db"),
            session_ttl_s=auth_config.get("session_ttl_s", 30 * 86400),
            account_provider=lambda: None if user_manager is None else user_manager.active_accounts(),
        )
        shutdown.push_async_callback(dashboard_authority.aclose)
        await output_registration.attach_output(dashboard_authority, "authority_sessions")
        if admission is not None:
            configuration_changes = ConfigurationChanges(dashboard_authority, admission, consent_gate)
            configuration_changes.install_on(shutdown)
            # Registered before any write: the owner writes only at shutdown.
            await destination_registration.attach_configuration(configuration_changes, dashboard_authority)
    elif destination_registration is not None:
        # The unselected publisher is recorded; its existing file stays an obligation.
        await destination_registration.configuration_disabled()

    # Home Assistant Client
    ha_client = (factories.ha_client or HomeAssistantClient)(
        url=ha_url,
        token=ha_token,
        timeout=ha_config.get("timeout", 2.0),
        state_confirmation_timeout_s=ha_config.get("state_confirmation_timeout_s", 0.3),
        operation_confirmation_timeouts=ha_config.get("operation_confirmation_timeouts"),
    )
    shutdown.push_async_callback(ha_client.close)
    home_action_service = HomeActionService(
        ha_client, policy=HomeControlPolicy(selection_path=ha_config.get("control_selection_path")),
    )
    shutdown.push_async_callback(home_action_service.close)
    home_capabilities = HomeCapabilityService(ha_client, home_action_service.policy,
                                            policy_provider=lambda: home_action_service.policy)
    shutdown.push_async_callback(home_capabilities.aclose)

    # Verificar conexión (async)
    if await ha_client.test_connection():
        logger.info("Home Assistant connected")
    else:
        logger.warning("Home Assistant unavailable at startup; continuing local voice and dashboard")


    # Preflight de VRAM antes de cargar modelos pesados (aviso, no bloquea).
    (factories.preflight or _preflight_vram)()

    # Own each created Whisper adapter even if later construction fails.
    stt_executors = STTExecutorOwner()
    shutdown.push_async_callback(stt_executors.aclose)
    # Speech-to-Text
    stt = (factories.stt or create_stt)(config.get("stt", {}))
    stt_executors.add_model(stt)

    # STT shadow (A/B en vivo, opcional): 2º motor que transcribe el mismo
    # audio del command path solo para comparar/loguear ([STT-shadow]), sin
    # cambiar el STT activo. Se activa con stt.shadow_engine (default null).
    shadow_stt = None
    _shadow_engine = (config.get("stt", {}) or {}).get("shadow_engine")
    if _shadow_engine:
        shadow_stt = (factories.stt or create_stt)({**config.get("stt", {}), "engine": _shadow_engine})
        stt_executors.add_model(shadow_stt)
        logger.info(f"STT shadow habilitado: engine={_shadow_engine} (A/B en vivo)")

    # Retain the model even if output or a later factory cannot be constructed.
    tts_models = TTSModelOwner()
    shutdown.push_async_callback(tts_models.aclose)
    # Text-to-Speech
    tts = (factories.tts or create_tts)(config.get("tts", {}))
    tts_models.add_model(tts)

    # Vector Database — embedder_model y embedder_device son obligatorios.
    # Fuente de verdad: settings.yaml § embeddings (cuda:0, BAAI/bge-m3).
    vectordb_config = config.get("vectordb", {})
    embeddings_config = config.get("embeddings", {})
    if "model" not in embeddings_config or "device" not in embeddings_config:
        raise ValueError(
            "settings.yaml debe declarar embeddings.model y embeddings.device"
        )
    chroma = (factories.chroma or ChromaSync)(
        chroma_path=vectordb_config.get("path", "./data/chroma_db"),
        embedder_model=embeddings_config["model"],
        embedder_device=embeddings_config["device"],
        excluded_entities=vectordb_config.get("exclude_entities"),
        excluded_patterns=vectordb_config.get("exclude_patterns"),
    )
    shutdown.push_async_callback(chroma.aclose)

    # LLM Reasoner — HTTP al kza-72b.service (default) o local GGUF (fallback)
    from src.llm import HttpReasoner
    http_reasoners = HttpReasonerOwner()
    shutdown.push_async_callback(http_reasoners.aclose)
    hermes_reasoners = _HermesReasonerOwner()
    shutdown.push_async_callback(hermes_reasoners.aclose)
    reasoner_config = config.get("reasoner", {})
    reasoner_mode = reasoner_config.get("mode", "http")

    from src.llm.cloud_consent import resolve_reasoner_gate

    # Evaluate the declared backend, even for alternate modes: compaction
    # shares this grant. Loopback gateways do not acquire a LOCAL declaration.
    gate_allowed, _ = resolve_reasoner_gate(
        reasoner_config, reasoner_mode, DEFAULT_LOCAL_LLM_GATEWAY
    )
    if consent_gate.withdrawn and _reasoner_is_cloud(reasoner_config, reasoner_mode):
        # m4: only a cloud reasoner is withdrawn; the config dict is never mutated.
        logger.warning(
            "Reasoner cloud bloqueado: consentimiento retirado (registro duradero en el journal). "
            "Para reactivarlo, el operador debe aplicar consent en settings.yaml y volver a preparar la admisión."
        )
        gate_allowed = False

    if reasoner_mode == "http":
        # The resolver labels the gateway fallback CLOUD; consumers also
        # enforce the grant at every inference, including cached clients.
        if not gate_allowed:
            logger.warning(
                "Reasoner cloud bloqueado por falta de consent — slow path sin reasoner. "
                "Setear reasoner.cloud.consent=true en settings.yaml para activarlo."
            )
            llm = None
        else:
            llm = HttpReasoner(
                base_url=reasoner_config.get("http_base_url", DEFAULT_LOCAL_LLM_GATEWAY),
                model=reasoner_config.get("http_model"),
                timeout=reasoner_config.get("http_timeout", 120),
                idle_timeout_s=reasoner_config.get("idle_timeout_s"),
                api_style=reasoner_config.get("api_style", "completions"),
                api_key_env=reasoner_config.get("api_key_env"),
                verify_ssl=reasoner_config.get("verify_ssl", True),
                destination=reasoner_config.get("destination", "cloud"),
                cloud_consent=reasoner_config.get("cloud", {}).get("consent", False),
                consent_gate=consent_gate,
            )
            http_reasoners.add_model(llm)
            try:
                await asyncio.to_thread(llm.load)
                info = llm.get_info()
                logger.info(f"LLM reasoner (cloud) vía HTTP → {info['base_url']} modelo={info['model']}")
            except Exception as e:
                # El fallback hacia 7B ahora lo maneja LLMRouter (plan #1 OpenClaw).
                # Si el reasoner cloud no responde, llm queda None — el LLMRouter rota al 7B.
                logger.error(f"HttpReasoner cloud no contactable: {e}. llm=None — failover via LLMRouter")
                llm = None
    elif reasoner_mode == "hermes_cli":
        # Reasoner cloud vía subproceso (hermes -z --provider openai-codex),
        # sin base_url — ver resolve_reasoner_gate branch hermes_cli
        # (src/llm/cloud_consent.py) y docs/superpowers/specs/
        # 2026-08-10-hermes-cli-reasoner-design.md.
        from src.llm.hermes_reasoner import HermesCliReasoner, warn_if_router_timeout_misaligned
        if not gate_allowed:
            logger.warning(
                "Reasoner cloud (hermes_cli) bloqueado por falta de consent — slow path sin reasoner. "
                "Setear reasoner.cloud.consent=true en settings.yaml para activarlo."
            )
            llm = None
        else:
            hermes_timeout_s = reasoner_config.get("hermes_timeout_s", 90.0)
            # Boot-time guard (finding #4, review 2026-08-10): el checklist pre-flip
            # comentado más arriba en settings.yaml pide a mano subir
            # llm.failover.endpoints[reasoner_cloud].timeout_s a >= hermes_timeout_s
            # — acá lo enforzamos con un log ruidoso si alguien lo pisa, en vez de
            # confiar solo en que se lea el comentario. failover_cfg todavía no se
            # parseó a esta altura de main() (eso pasa más abajo, al construir
            # LLMRouter) — se lee el dict crudo de config acá porque es lo único
            # que hace falta para esta comparación.
            _router_ep_timeout_s = next(
                (
                    ep.get("timeout_s")
                    for ep in (config.get("llm", {}).get("failover", {}) or {}).get("endpoints", []) or []
                    if ep.get("id") == "reasoner_cloud"
                ),
                None,
            )
            warn_if_router_timeout_misaligned(hermes_timeout_s, _router_ep_timeout_s)
            llm = HermesCliReasoner(
                binary_path=reasoner_config.get("hermes_binary_path", "hermes"),
                provider=reasoner_config.get("hermes_provider", "openai-codex"),
                cloud_consent=reasoner_config.get("cloud", {}).get("consent", False),
                model=reasoner_config.get("hermes_model"),
                timeout_s=hermes_timeout_s,
                consent_gate=consent_gate,
            )
            hermes_reasoners.add_model(llm)
            try:
                await hermes_reasoners.load_model(llm)
                info = llm.get_info()
                logger.info(f"LLM reasoner (cloud) vía Hermes CLI → provider={info['provider']} modelo={info['model']}")
            except Exception as e:
                logger.error(f"HermesCliReasoner no disponible: {e}. llm=None — failover via LLMRouter")
                llm = None
    else:
        model_path = reasoner_config.get("model_path")
        if not model_path or not Path(model_path).exists():
            logger.warning(f"Modelo LLM local no encontrado: {model_path}")
            llm = None
        else:
            llm = LLMReasoner(
                model_path=model_path,
                lora_path=reasoner_config.get("lora_path"),
                lora_scale=reasoner_config.get("lora_scale", 1.0),
                n_ctx=reasoner_config.get("n_ctx", 32768),
                n_threads=reasoner_config.get("n_threads", 24),
                n_batch=reasoner_config.get("n_batch", 512),
                n_gpu_layers=reasoner_config.get("n_gpu_layers", 0),
                chat_format=reasoner_config.get("chat_format", "chatml"),
                rope_freq_base=reasoner_config.get("rope_freq_base", 1000000.0),
                rope_freq_scale=reasoner_config.get("rope_freq_scale", 1.0),
            )
            llm.load()
            logger.info("LLM 72B cargado en proceso (mode=local)")

    # Routine Manager
    routine_manager = RoutineManager(ha_client, chroma, llm)

    # Fast Router (GPU 2) - para clasificación rápida con KV-Cache
    router_config = config.get("router", {})
    fast_router = None
    if router_config.get("enabled", True):
        fast_router = FastRouter(
            base_url=router_config.get("base_url", "http://127.0.0.1:8101/v1"),
            model=router_config.get("model", "qwen2.5-7b-instruct"),
            timeout=router_config.get("timeout", 30),
            destination=router_config.get("destination", "cloud"),
            cloud_consent=reasoner_config.get("cloud", {}).get("consent", False),
            consent_gate=consent_gate,
        )
        shutdown.push_async_callback(fast_router.aclose)
        logger.info(f"Fast router (HTTP) → {router_config.get('base_url', 'http://127.0.0.1:8101/v1')}")

    # LLMRouter — candidate chain con cooldown y failover (plan #1 OpenClaw 2026-04-28).
    # Envuelve fast_router y llm (HttpReasoner 72B) para rotar automáticamente
    # cuando uno falla. El dispatcher usa router.complete(), no router.generate().
    llm_router = None
    llm_metrics_tracker = None
    failover_cfg = config.get("llm", {}).get("failover")
    if failover_cfg and fast_router is not None:
        from src.llm import build_llm_router
        from src.llm.metrics import LLMMetricsTracker
        llm_metrics_tracker = LLMMetricsTracker(window_s=300.0)
        clients = {"fast_router_7b": fast_router}
        if llm is not None:
            clients["reasoner_cloud"] = llm
        # Pegar tracker direct-on-client así captura llamadas que NO pasan por
        # LLMRouter (ej. el LLMCommandRouter NLU usa fast_router directo).
        for ep_id, client in clients.items():
            try:
                client._metrics_tracker = llm_metrics_tracker
                client._endpoint_id = ep_id
            except AttributeError:
                pass  # cliente no soporta tracker — skip silently
        try:
            llm_router = build_llm_router(failover_cfg, clients, metrics_tracker=llm_metrics_tracker)
            logger.info(
                f"LLMRouter listo con {len(clients)} endpoints "
                f"({', '.join(clients)})"
            )
        except Exception as e:
            logger.error(f"No se pudo construir LLMRouter, sigo con fast_router suelto: {e}")
            llm_router = None

    # Privacy authority always exists; semantic recall remains optional.
    privacy_services = await build_privacy_services(config, shutdown=shutdown)
    privacy_services.add_borrower(chroma.begin_close)
    privacy_services.add_borrower(http_reasoners.begin_close)
    privacy_services.add_borrower(hermes_reasoners.begin_close)
    http_reasoners.add_borrower(chroma.begin_close)
    hermes_reasoners.add_borrower(chroma.begin_close)
    privacy_services.add_borrower(stt_executors.begin_close)
    privacy_services.add_borrower(tts_models.begin_close)
    if fast_router is not None:
        privacy_services.add_borrower(fast_router.begin_close)
    memory_services = await build_memory_services(config, shutdown=shutdown, privacy=privacy_services)
    memory_manager = memory_services.manager if memory_services is not None else None
    erasure_services = await build_erasure_services(config, privacy=privacy_services, shutdown=shutdown,
        memory=memory_manager.long_term if memory_manager is not None else None)
    voice_registry = await build_voice_registry(privacy=privacy_services,owner=erasure_services,shutdown=shutdown)
    speaker_models = SpeakerModelOwner()
    shutdown.push_async_callback(speaker_models.aclose)
    speaker_models.add_borrower(voice_registry.begin_close)
    privacy_services.add_borrower(speaker_models.begin_close)
    wake_models = WakeModelOwner()
    shutdown.push_async_callback(wake_models.aclose)
    privacy_services.add_borrower(wake_models.begin_close)
    diagnostic_services = await build_diagnostic_services(config, privacy=privacy_services, owner=erasure_services)
    metrics_emitter = diagnostic_services.emitter
    capture_erasure = await build_capture_erasure(config, privacy=privacy_services, owner=erasure_services)
    event_erasure = await build_event_erasure(config, privacy=privacy_services, owner=erasure_services)
    suggestion_erasure = await build_suggestion_erasure(config, privacy=privacy_services, owner=erasure_services)
    context_erasure = await build_context_erasure(config, privacy=privacy_services, owner=erasure_services, memory=memory_manager)
    wake_clip_store = await build_wake_clip_store(config, privacy=privacy_services, shutdown=shutdown, owner=erasure_services)
    stt_correction_collector = await build_stt_correction_collector(config, store=wake_clip_store)
    conversation_collector = await build_conversation_collector(config, store=wake_clip_store)

    # LLM Command Router (Opción 2 NLU) — clasificador de comandos vía vLLM 7B.
    # Independiente del LLMRouter: este valida texto post-wake contra
    # alucinaciones de TV / replays / frases noise. Hereda el FastRouter
    # ya configurado contra el vLLM compartido en :8100.
    llm_command_router = None
    nlu_cfg = config.get("nlu", {}).get("llm_router", {})
    if nlu_cfg.get("enabled", False) and fast_router is not None:
        from src.nlu.llm_router import LLMCommandRouter
        # Cargar el FastRouter ahora si todavía no se hizo (lazy en su .load()).
        try:
            fast_router.load()
        except Exception as e:
            logger.warning(f"FastRouter .load() falló pre-LLMCommandRouter: {e}")
        llm_command_router = LLMCommandRouter(
            fast_router=fast_router,
            max_history=nlu_cfg.get("max_history", 5),
            history_ttl_s=nlu_cfg.get("history_ttl_s", 120.0),
            timeout_s=nlu_cfg.get("timeout_s", 1.5),
            max_tokens=nlu_cfg.get("max_tokens", 200),
            temperature=nlu_cfg.get("temperature", 0.0),
            source_policy=privacy_services.ledger,
        )
        shutdown.push_async_callback(llm_command_router.aclose)
        logger.info(
            f"LLMCommandRouter habilitado (timeout={nlu_cfg.get('timeout_s', 1.5)}s, "
            f"max_history={nlu_cfg.get('max_history', 5)})"
        )

    # Speaker Identification (cuda:0 — pipeline de audio consolidado)
    # Fuente de verdad: settings.yaml § speaker_id.
    speaker_config = config.get("speaker_id", {})
    speaker_identifier = None
    user_manager = await voice_registry.load_accounts(
        speaker_config.get('users_path', './data/users.json'),
        retirement_proof=(lambda identities: dashboard_authority.users_retired(identities)
            and voice_registry.ledger.people.user_links_cleared(identities)
            and privacy_services.ledger.owner_refs_cleared(identities)) if dashboard_authority is not None else None,
    )
    voice_enrollment = None

    if speaker_config.get("enabled", True):
        if "model" not in speaker_config or "device" not in speaker_config:
            raise ValueError(
                "settings.yaml debe declarar speaker_id.model y speaker_id.device"
            )
        speaker_identifier = SpeakerIdentifier(
            model_name=speaker_config["model"],
            device=speaker_config["device"],
            similarity_threshold=speaker_config.get("threshold", 0.75),
            savedir=speaker_savedir(config, speaker_config["model"]),
        )

        speaker_models.add_model(speaker_identifier)
        voice_registry.register_encoder(speaker_identifier)

        voice_enrollment = VoiceEnrollment(
            user_manager=user_manager,
            speaker_identifier=speaker_identifier,
            review_required=True,
        )
        logger.info("Speaker identification habilitado")

    # U3b-4b-2: the voice stage (attribution after the reply, verification
    # phrases) exists only with memory.voice_proposals.enabled true; it reuses
    # the registry's stored embeddings and adds no inference or device.
    # The whisper wake engine logs transcripts, so it keeps the stage off;
    # phrase words are kept away from this home's device and area names.
    voice_attributor = build_voice_attribution(
        privacy_services, voice_registry, dashboard_authority, shutdown=shutdown,
        wake_engine=config.get("rooms", {}).get("wake_word", {}).get("engine", "openwakeword"),
        home_names=home_capabilities.cached_names,
    )

    # Emotion Detection (cuda:0 — pipeline de audio consolidado)
    # Fuente de verdad: settings.yaml § emotion. Comparte GPU con speaker_id por
    # diseño (ambos consumen el mismo audio post-STT). Si difieren, alertamos.
    emotion_config = config.get("emotion", {})
    if not emotion_config.get("enabled", True):
        emotion_detector = None
        logger.info("Emotion detector deshabilitado por config (emotion.enabled=false)")
    else:
        if "model" not in emotion_config or "device" not in emotion_config:
            raise ValueError(
                "settings.yaml debe declarar emotion.model y emotion.device"
            )
        if speaker_config.get("enabled", True) and emotion_config["device"] != speaker_config.get("device"):
            logger.warning(
                f"emotion.device ({emotion_config['device']}) != speaker_id.device "
                f"({speaker_config.get('device')}); el diseño asume la misma GPU."
            )
        emotion_detector = EmotionDetector(
            model_name=emotion_config["model"],
            device=emotion_config["device"],
            sample_rate=emotion_config.get("sample_rate", 16000),
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

    # U3b-0d2: worst event-loop block, for the later watchdog sizing.
    loop_lag = LoopLagProbe(LoopLagConfig.from_config(monitoring_config.get("loop_lag")))

    # Smart Automations - Event Logger y Suggestion Engine
    analytics_config = config.get("analytics", {})
    event_logger = None
    suggestion_engine = None

    if analytics_config.get("enabled", True):
        event_logger = await build_event_logger(config, shutdown=shutdown,
            privacy=privacy_services, retention=event_erasure.retention)

        pattern_analyzer = PatternAnalyzer(event_logger)

        suggestion_engine = await output_registration.construct_suggestions(event_logger, SuggestionEngine,
            event_logger=event_logger,
            pattern_analyzer=pattern_analyzer,
            suggestions_path=analytics_config.get("suggestions_path", "./data/suggestions.json"),
        )
        suggestion_erasure.attach_engine(suggestion_engine)
        logger.info("Smart automations habilitado")

    # Lists & Reminders
    lists_config = config.get("lists", {})
    reminders_config = config.get("reminders", {})
    list_store = ListStore(lists_config.get("db_path", "./data/lists.db"))
    shutdown.push_async_callback(list_store.close)
    if destination_registration is not None:
        await destination_registration.register_store(list_store, "lists")
    await list_store.initialize()
    list_manager = ListManager(store=list_store, ha_client=ha_client, config=lists_config)

    reminder_wake_event = asyncio.Event()
    reminder_store = ReminderStore(reminders_config.get("db_path", "./data/reminders.db"),
                                   on_created=reminder_wake_event.set)
    shutdown.push_async_callback(reminder_store.close)
    if destination_registration is not None:
        await destination_registration.register_store(reminder_store, "reminders")
    await reminder_store.initialize()
    reminder_manager = ReminderManager(store=reminder_store, config=reminders_config, home_actions=home_action_service)
    reminder_scheduler = None  # Created after presence_detector is ready
    logger.info("Lists & reminders initialized")

    # Multi-room / multi-zone defaults (always defined for DI)
    room_context_manager = None
    presence_detector = None
    multi_room_loop = None
    ambient_path = None

    # Multi-Zone Audio System (Dayton MA1260) — legacy fallback
    zones_config = config.get("zones", {})
    zone_manager = None

    if zones_config.get("enabled", False):
        # Crear controlador MA1260
        ma1260_config = zones_config.get("ma1260", {})
        ma1260 = (factories.amplifier or MA1260Controller)(
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
        "telemetry", "barge_in", "stream_watchdog", "ambient_guard", "endpointing",
        "direction_observation",
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
            shutdown.push_async_callback(presence_detector.stop)
            logger.info("Presence detector created")

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

        # Recolectar TODOS los aliases de rooms (room_id + RoomConfig.aliases)
        # para inyectarlos al WhisperWakeDetector como known_room_aliases.
        # Usado por el TV-mode soft-reject (decisión 2-C, 2026-05-03): un wake
        # en TV-mode con audio degradado y SIN mención literal de room/entity
        # se rechaza como probable fantasma.
        _all_room_aliases: list[str] = []
        for _rk, _rd in rooms_config.items():
            if _rk in ROOMS_RESERVED_KEYS or not isinstance(_rd, dict):
                continue
            _all_room_aliases.append(_rk)
            _all_room_aliases.extend(_rd.get("aliases", []))
        _all_room_aliases_t = tuple(set(_all_room_aliases))

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
                mic_usb_port=room_dict.get("mic_usb_port"),
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

            # Keep configured USB microphones visible and retryable when absent.
            if rc.mic_device_index is not None or rc.mic_usb_port:
                wake_engine = room_wake_cfg.get("engine", "openwakeword")
                if wake_engine == "whisper":
                    # Speaker filter opcional: carga embedding si existe + enabled.
                    sf_cfg = room_wake_cfg.get("speaker_filter", {})
                    spk_emb = None
                    spk_ref_identifier = None
                    if sf_cfg.get("enabled", False) and speaker_identifier is not None:
                        emb_path = Path(sf_cfg.get("embedding_path", ""))
                        if emb_path.exists():
                            import numpy as _np
                            spk_emb = _np.load(str(emb_path))
                            spk_ref_identifier = speaker_identifier
                            logger.info(
                                f"Speaker filter activo para '{room_key}': "
                                f"embedding={emb_path} shape={spk_emb.shape} "
                                f"threshold={sf_cfg.get('threshold', 0.65)}"
                            )
                        else:
                            logger.warning(
                                f"Speaker filter enabled pero falta {emb_path}. "
                                f"Ejecutar: python -m scripts.enroll_voice --user <id>"
                            )

                    streaming = room_wake_cfg.get("streaming", False)
                    stt_cfg = config.get("stt", {})
                    wake_language = stt_cfg.get("language", "es")
                    # beam_size / initial_prompt: override por room o herencia del stt top.
                    wake_beam = room_wake_cfg.get("beam_size") or stt_cfg.get("beam_size", 1)
                    wake_prompt = room_wake_cfg.get("initial_prompt") or stt_cfg.get("initial_prompt")
                    if streaming:
                        from src.wakeword.streaming_whisper_wake import (
                            StreamingWhisperWakeDetector,
                        )
                        wake_detector = StreamingWhisperWakeDetector(
                            whisper_stt=stt,
                            wake_words=room_wake_cfg.get("words", ["nexa"]),
                            interval_ms=room_wake_cfg.get("streaming_interval_ms", 200),
                            window_s=room_wake_cfg.get("streaming_window_s", 2.0),
                            vad_threshold=room_wake_cfg.get("vad_threshold", 0.7),
                            min_rms=room_wake_cfg.get("min_rms", 0.025),
                            language=wake_language,
                            speaker_identifier=spk_ref_identifier,
                            speaker_embedding=spk_emb,
                            speaker_threshold=sf_cfg.get("threshold", 0.65),
                            speaker_min_audio_s=sf_cfg.get("min_audio_s", 0.8),
                            beam_size=wake_beam,
                            initial_prompt=wake_prompt,
                            use_silero_vad=room_wake_cfg.get("use_silero_vad", True),
                        )
                    else:
                        from src.wakeword.whisper_wake import WhisperWakeDetector
                        wake_detector = WhisperWakeDetector(
                            whisper_stt=stt,
                            wake_words=room_wake_cfg.get("words", ["nexa"]),
                            silence_end_ms=room_wake_cfg.get("silence_end_ms", 500),
                            min_utterance_ms=room_wake_cfg.get("min_utterance_ms", 250),
                            max_utterance_s=room_wake_cfg.get("max_utterance_s", 3.5),
                            vad_threshold=room_wake_cfg.get("vad_threshold", 0.7),
                            min_rms=room_wake_cfg.get("min_rms", 0.025),
                            fuzzy_threshold=room_wake_cfg.get("fuzzy_threshold", 0.75),
                            fuzzy_start_words=room_wake_cfg.get("fuzzy_start_words", 3),
                            language=wake_language,
                            speaker_identifier=spk_ref_identifier,
                            speaker_embedding=spk_emb,
                            speaker_threshold=sf_cfg.get("threshold", 0.65),
                            speaker_min_audio_s=sf_cfg.get("min_audio_s", 0.8),
                            beam_size=wake_beam,
                            initial_prompt=wake_prompt,
                            metrics_emitter=metrics_emitter,
                            room_id=room_key,
                            follow_up_window_s=room_wake_cfg.get("follow_up_window_s", 4.0),
                            follow_up_max_words=room_wake_cfg.get("follow_up_max_words", 3),
                            known_room_aliases=_all_room_aliases_t,
                            tv_mode_min_rms=room_wake_cfg.get("tv_mode_min_rms", 0.04),
                            tv_mode_min_audio_s=room_wake_cfg.get("tv_mode_min_audio_s", 1.0),
                            use_silero_vad=room_wake_cfg.get("use_silero_vad", True),
                        )
                elif wake_engine == "porcupine":
                    import os as _os
                    from src.wakeword.porcupine_wake import PorcupineWakeDetector
                    _pv = room_wake_cfg.get("porcupine", {})
                    wake_detector = PorcupineWakeDetector(
                        access_key=_os.environ.get(
                            _pv.get("access_key_env", "PORCUPINE_ACCESS_KEY"), ""
                        ),
                        keyword_path=(_pv["keyword_path"] if "keyword_path" in _pv
                                      else models_path(config, "wakeword", "nexa_es.ppn")),
                        model_path=(_pv["model_path"] if "model_path" in _pv
                                    else models_path(config, "wakeword", "porcupine_params_es.pv")),
                        sensitivity=_pv.get("sensitivity", 0.6),
                        wake_word=(room_wake_cfg.get("words") or ["nexa"])[0],
                    )
                else:
                    wake_detector = (factories.wake_detector or WakeWordDetector)(
                        models=[room_wake_cfg.get("model", "hey_jarvis")],
                        custom_models_dir=models_path(config, "wakeword"),
                        threshold=room_wake_cfg.get("threshold", 0.5),
                    )
                wake_models.add_model(wake_detector)
                room_echo = EchoSuppressor(sample_rate=16000)
                room_streams[room_key] = RoomStream(
                    room_id=room_key,
                    device_index=rc.mic_device_index,
                    wake_detector=wake_detector,
                    echo_suppressor=room_echo,
                    # L-3: canal de captura per-room (0=Conference, 1=ASR en el
                    # XVF3800 UA según doc Seeed — A/B con medición, fallback
                    # seguro a 0 si el device no tiene el canal).
                    capture_channel=room_dict.get("capture_channel", 0),
                    mic_usb_port=rc.mic_usb_port,
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
            ma1260 = (factories.amplifier or MA1260Controller)(
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
            early_cfg = rooms_config.get("wake_word", {})
            watchdog_cfg = rooms_config.get("stream_watchdog", {}) or {}
            endpointing_cfg = rooms_config.get("endpointing", {})
            # Barge-in (S3). `response_handler` se inyecta luego con
            # `attach_response_handler()` porque se construye más abajo.
            barge_in_cfg = rooms_config.get("barge_in", {}) or {}
            # Pre-gate SPENERGY (VAD por hardware del XVF3800). Fail-open: si no
            # se puede abrir el device (sin udev/permisos/pyusb), el gate queda OFF.
            # UN controller POR ROOM, bindeado al mic_usb_port de esa habitación
            # (2026-07-25). Antes se construía uno solo y usb.core.find() tomaba
            # el PRIMER XVF3800 que enumerara: al conectar el segundo mic
            # (cocina) el tuning podía escribirse sobre el mic equivocado y
            # dejar al otro en el preset de fábrica (MAXGAIN=64) en silencio.
            # Una room sin mic_usb_port no recibe gate ni tuning — nunca cae al
            # controller de otra habitación.
            spe_cfg = early_cfg.get("spenergy_gate", {}) or {}
            tuning_cfg = early_cfg.get("xvf_tuning", {}) or {}
            direction_cfg = rooms_config.get('direction_observation')
            _, direction_room_ids, _ = parse_direction_observation(
                direction_cfg, room_context_manager.get_all_rooms(),
            )
            xvf_controllers = _build_xvf_controllers(
                room_streams, spe_cfg, tuning_cfg, direction_room_ids,
            )
            # AmbientGuard (spec 2026-06-05): compuerta acústica integral.
            # Default enabled=false → pasivo. Umbrales: ver settings.yaml.
            from src.pipeline.ambient_guard import AmbientGuard, AmbientGuardConfig
            ag_cfg = rooms_config.get("ambient_guard", {}) or {}
            ambient_guard = AmbientGuard(
                config=AmbientGuardConfig(
                    enabled=ag_cfg.get("enabled", False),
                    strict_entry_rejects=ag_cfg.get("strict_entry_rejects", 4),
                    strict_entry_window_s=ag_cfg.get("strict_entry_window_s", 60.0),
                    strict_exit_quiet_s=ag_cfg.get("strict_exit_quiet_s", 120.0),
                    strict_wake_score=ag_cfg.get("strict_wake_score", 0.65),
                    strict_min_rms=ag_cfg.get("strict_min_rms", 0.0),
                    strict_min_spenergy=ag_cfg.get("strict_min_spenergy", 0.0),
                    cooldown_entry_rejects=ag_cfg.get("cooldown_entry_rejects", 6),
                    cooldown_entry_window_s=ag_cfg.get("cooldown_entry_window_s", 60.0),
                    cooldown_duration_s=ag_cfg.get("cooldown_duration_s", 30.0),
                    cooldown_override_score=ag_cfg.get("cooldown_override_score", 0.0),
                    strict_follow_up_grace_s=ag_cfg.get("strict_follow_up_grace_s", 12.0),
                    strict_vad_adaptive=ag_cfg.get("strict_vad_adaptive", False),
                    strict_wake_score_min=ag_cfg.get("strict_wake_score_min", 0.50),
                    strict_vad_lo=ag_cfg.get("strict_vad_lo", 0.30),
                    strict_vad_hi=ag_cfg.get("strict_vad_hi", 0.70),
                )
            )
            if ag_cfg.get("enabled", False):
                logger.info(
                    f"AmbientGuard ACTIVO: strict_wake_score="
                    f"{ag_cfg.get('strict_wake_score', 0.65)}, "
                    f"entry={ag_cfg.get('strict_entry_rejects', 4)} rechazos/"
                    f"{ag_cfg.get('strict_entry_window_s', 60.0):.0f}s"
                )
            # Captura de clips de wake (2026-06-12): dataset de re-entrenamiento
            # de nexa.onnx con audio real (falsos de TV + comandos far-field).
            wake_clip_writer = None
            clip_cfg = early_cfg.get("capture_clips", {}) or {}
            if clip_cfg.get("enabled", False):
                from src.wakeword.wake_clip_writer import WakeClipWriter
                clip_target = await wake_clip_store._work.run(wake_clip_store.register,
                    clip_cfg.get("dir", "data/wakeword_training/captured"))
                wake_clip_writer = WakeClipWriter(
                    directory=clip_target.root,
                    store=wake_clip_store,
                    target=clip_target,
                    max_files=int(clip_cfg.get("max_files", 2000)),
                    max_rejected_files=int(clip_cfg.get("max_rejected_files", 4000)),
                    max_missed_files=int(clip_cfg.get("max_missed_files", 300)),
                )
                logger.info(
                    f"WakeClipWriter ACTIVO: dir={clip_cfg.get('dir')} "
                    f"max_files={clip_cfg.get('max_files', 2000)}"
                )
            multi_room_loop = MultiRoomAudioLoop(
                privacy=privacy_services,
                room_streams=room_streams,
                microphone_rooms=build_microphone_rooms(room_context_manager.get_all_rooms()),
                microphone_interval_s=rooms_config.get("telemetry", {}).get("interval_s", 2.0),
                microphone_stale_after_s=rooms_config.get("telemetry", {}).get("stale_after_s", 6.0),
                follow_up=FollowUpMode(
                    follow_up_window=early_cfg.get("follow_up_window_s", 4.0)
                ),
                xvf_controllers=xvf_controllers,
                spenergy_threshold=spe_cfg.get("threshold", 100.0),
                spenergy_gate_enabled=spe_cfg.get("enabled", False),
                xvf_tuning=tuning_cfg,
                direction_observation=direction_cfg,
                direction_calibrations={
                    room_id: rooms_config[room_id].get('direction_calibration')
                    for room_id in room_context_manager.get_all_rooms()
                },
                ambient_guard=ambient_guard,
                wake_clip_writer=wake_clip_writer,
                sample_rate=16000,
                command_duration=2.0,
                dedup_window_ms=rooms_config.get("dedup_window_ms", 200),
                min_wake_rms=early_cfg.get("min_wake_rms", 0.0),
                wake_preroll_s=early_cfg.get("wake_preroll_s", 0.0),
                early_dispatch_enabled=early_cfg.get("early_dispatch", False),
                early_dispatch_interval_ms=early_cfg.get("early_dispatch_interval_ms", 400),
                early_dispatch_min_audio_s=early_cfg.get("early_dispatch_min_audio_s", 0.6),
                stt=stt,
                # VAD de fin-de-comando (QW-3 2026-06-04): antes quedaban en los
                # defaults hardcodeados del constructor — la config no llegaba.
                silence_threshold=endpointing_cfg.get("silence_threshold", 0.015),
                silence_duration_ms=endpointing_cfg.get("silence_duration_ms", 300),
                min_speech_ms=endpointing_cfg.get("min_speech_ms", 300),
                endpointing_enabled=endpointing_cfg.get("enabled", True),
                endpointing_short_ms=endpointing_cfg.get("short_ms", 150),
                endpointing_medium_ms=endpointing_cfg.get("medium_ms", 300),
                endpointing_long_ms=endpointing_cfg.get("long_ms", 500),
                barge_in_enabled=barge_in_cfg.get("enabled", False),
                barge_in_rms_threshold=barge_in_cfg.get("rms_threshold", 0.03),
                barge_in_min_duration_ms=barge_in_cfg.get("min_duration_ms", 200),
                stream_watchdog_enabled=watchdog_cfg.get("enabled", False),
                stream_watchdog_no_frames_timeout_s=watchdog_cfg.get("no_frames_timeout_s", 8.0),
                stream_watchdog_check_interval_s=watchdog_cfg.get("check_interval_s", 2.0),
                stream_watchdog_reopen_backoff_min_s=watchdog_cfg.get("reopen_backoff_min_s", 1.0),
                stream_watchdog_reopen_backoff_max_s=watchdog_cfg.get("reopen_backoff_max_s", 10.0),
                stream_watchdog_first_frame_grace_s=watchdog_cfg.get("first_frame_grace_s", 180.0),
                audio_health_path=watchdog_cfg.get(
                    "health_path", "./data/audio_health.json"
                ),
            )
            stt_executors.add_borrower(multi_room_loop.begin_close)
            speaker_models.add_borrower(multi_room_loop.begin_close)
            wake_models.add_borrower(multi_room_loop.begin_close)
            shutdown.push_async_callback(multi_room_loop.stop)
            logger.info(
                f"MultiRoomAudioLoop created ({len(room_streams)} rooms: "
                f"{', '.join(room_streams.keys())})"
            )

            # Ambient path (spec 2026-06-06): transcripción continua multi-pista
            # en cuda:0. Best-effort: si falla el build, el command path sigue.
            ambient_cfg = config.get("ambient", {}) or {}
            if ambient_cfg.get("enabled", False):
                try:
                    from src.ambient.transcriber import build_ambient_path
                    _store_fact = (
                        memory_manager.long_term.astore_fact
                        if memory_manager is not None else None
                    )
                    ambient_path = build_ambient_path(
                        ambient_cfg=ambient_cfg,
                        stt_base_cfg=config.get("stt", {}) or {},
                        room_ids=list(room_streams.keys()),
                        store_fact_fn=_store_fact,
                        privacy=privacy_services,
                        voice_registry=voice_registry,
                        register_stt=stt_executors.add_model,
                        register_stt_borrower=stt_executors.add_borrower,
                        register_speaker=speaker_models.add_model,
                        register_speaker_borrower=speaker_models.add_borrower,
                        config=config,
                    )
                    shutdown.push_async_callback(ambient_path.stop)
                    await capture_erasure.attach(ambient_path)
                    multi_room_loop.attach_ambient(
                        tap=ambient_path.tap,
                        transcriber=ambient_path.transcriber,
                    )
                    logger.info(
                        f"Ambient path construido ({len(room_streams)} rooms, "
                        f"shadow={ambient_cfg.get('shadow_mode', True)})"
                    )
                except Exception:
                    logger.exception(
                        "Ambient path no construido (best-effort) — "
                        "el pipeline de voz sigue sin él"
                    )
                    ambient_path = None

    # ----------------------------------------------------------------
    # Build pipeline components (DI chain)
    # ----------------------------------------------------------------

    # Audio manager (wake word + VAD)
    audio_manager = AudioManager(
        zone_manager=zone_manager,
        wake_models=wake_models,
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
        shadow_stt=shadow_stt,
        shadow_veto=bool((config.get("stt", {}) or {}).get("shadow_veto", False)),
        privacy=privacy_services,
        model_locks=multi_room_loop.native_model_locks if multi_room_loop is not None else None,
    )
    stt_executors.add_borrower(command_processor.begin_close)
    http_reasoners.add_borrower(command_processor.begin_close)
    hermes_reasoners.add_borrower(command_processor.begin_close)
    speaker_models.add_borrower(command_processor.begin_close)
    shutdown.push_async_callback(command_processor.aclose)

    # === Plan #3 OpenClaw — Plugin hooks (optional) ===
    hooks_cfg = config.get("hooks", {}) or {}
    self_warn_ms = float(hooks_cfg.get("before_handler_warn_ms", 5.0))
    hooks = load_policy_hooks(hooks_cfg, shutdown)

    # Response handler (TTS + streaming + zone routing)
    # El response_cache (S2) se inicializa post-warmup abajo y se inyecta via
    # attach_response_cache — esto mantiene el orden: DI → warmup → build cache.
    def _register_voice_output(begin_close: Callable[[], asyncio.Task]) -> None:
        """Capture the shared response drain before later builders run."""
        tts_models.add_borrower(begin_close)
        http_reasoners.add_borrower(begin_close)
        hermes_reasoners.add_borrower(begin_close)

    voice_services = build_voice_services(
        config, tts=tts, shutdown=shutdown, zone_manager=zone_manager, llm=llm,
        hooks=hooks, before_handler_warn_ms=self_warn_ms, user_manager=user_manager,
        ha_client=ha_client, home_actions=home_action_service, reminder_store=reminder_store,
        presence_detector=presence_detector, reminder_wake_event=reminder_wake_event,
        register_output=_register_voice_output,
    )
    response_handler = voice_services.responses
    reminder_scheduler = voice_services.reminders

    # Earcon "te oí, no te entendí" (2026-06-15): cargar asset y configurar
    # en el response_handler para que play_earcon() lo reproduzca en zonas.
    from src.tts.response_cache import load_earcon
    _earcon_cfg = config.get("command_gate", {}).get("earcon", {"enabled": False})
    _earcon_audio, _earcon_sr = load_earcon(
        _earcon_cfg.get("asset", "data/earcons/not_understood.wav")
    )
    if _earcon_audio is not None:
        response_handler.set_earcon(_earcon_audio, _earcon_sr)

    # Inyectar response_handler al MultiRoomAudioLoop para barge-in (S3).
    # Se construyó arriba sin response_handler por orden de DI; lo attacheamos
    # ahora que ambos existen.
    if multi_room_loop is not None:
        multi_room_loop.attach_response_handler(response_handler)

    # Echo suppressor and follow-up mode (needed by AudioLoop)
    echo_suppressor = EchoSuppressor(sample_rate=16000)
    follow_up = FollowUpMode(follow_up_window=8.0)

    # Audio loop (capture + wake word + echo suppression + follow-up)
    audio_loop = AudioLoop(
        audio_manager=audio_manager,
        echo_suppressor=echo_suppressor,
        follow_up=follow_up,
        sample_rate=16000,
        privacy=privacy_services,
    )
    if callable(getattr(type(audio_loop), "begin_close", None)):
        stt_executors.add_borrower(audio_loop.begin_close)
        speaker_models.add_borrower(audio_loop.begin_close)
        wake_models.add_borrower(audio_loop.begin_close)
    shutdown.push_async_callback(audio_loop.stop)

    # === Plan #2 OpenClaw — Compactor + Persister (opcionales) ===
    context_cfg = (config.get("orchestrator", {}) or {}).get("context", {}) or {}
    compaction_cfg = context_cfg.get("compaction", {}) or {}
    persistence_cfg = context_cfg.get("persistence", {}) or {}

    compactor = None
    persister = None
    compaction_threshold = compaction_cfg.get("threshold_turns", 6)
    keep_recent_turns = compaction_cfg.get("keep_recent_turns", 3)

    if compaction_cfg.get("enabled", False):
        from src.orchestrator import Compactor

        if reasoner_mode == "hermes_cli":
            # Sin base_url que resolver — resolve_compaction_endpoint asume HTTP
            # (ver su docstring) y no aplica acá. El compactor usa la MISMA clase
            # y config que el reasoner principal cuando el gate lo permite; si no,
            # no hay compactación (no tiene sentido forzar un segundo subproceso
            # hermes solo para esto cuando el reasoner principal ya está apagado
            # por falta de consent).
            from src.llm.hermes_reasoner import HermesCliReasoner
            if not gate_allowed:
                logger.warning(
                    "[main] Compactor (hermes_cli) deshabilitado — reasoner.cloud.consent=false."
                )
                compactor = None
            else:
                # timeout_s del subproceso: usar hermes_timeout_s (90s default, el mismo
                # valor que el reasoner principal), NO compaction_cfg.timeout_s (30s) —
                # ese 30s está tuneado para una llamada HTTP a un gateway ya caliente
                # (el path mode="http" preexistente), no para arrancar un proceso hermes
                # nuevo. El costo de arranque de hermes es el mismo compactando que
                # razonando.
                compaction_hermes_timeout_s = reasoner_config.get("hermes_timeout_s", 90.0)
                compaction_reasoner = HermesCliReasoner(
                    binary_path=reasoner_config.get("hermes_binary_path", "hermes"),
                    provider=reasoner_config.get("hermes_provider", "openai-codex"),
                    cloud_consent=reasoner_config.get("cloud", {}).get("consent", False),
                    model=reasoner_config.get("hermes_model"),
                    timeout_s=compaction_hermes_timeout_s,
                    consent_gate=consent_gate,
                )
                hermes_reasoners.add_model(compaction_reasoner)
                try:
                    await hermes_reasoners.load_model(compaction_reasoner)
                    # El timeout de Compactor limita al observador de complete;
                    # el dueño conserva el job nativo aunque ese observador se cancele.
                    # Mantener el máximo entre lo configurado y el presupuesto del
                    # subproceso evita agotar la espera antes de su propio timeout.
                    compactor = Compactor(
                        reasoner=compaction_reasoner,
                        max_summary_tokens=compaction_cfg.get("max_summary_tokens", 200),
                        timeout_s=max(
                            compaction_cfg.get("timeout_s", 30.0),
                            compaction_hermes_timeout_s,
                        ),
                    )
                    logger.info(
                        f"[main] Compactor enabled vía Hermes CLI (threshold={compaction_threshold}, "
                        f"keep_recent={keep_recent_turns})"
                    )
                except Exception as e:
                    logger.error(
                        f"[main] Compactor disabled — could not load Hermes reasoner: {e}",
                        exc_info=True,
                    )
                    compactor = None
        else:
            from src.llm.cloud_consent import resolve_compaction_endpoint
            from src.llm.reasoner import HttpReasoner

            # Cutover 2026-05-30 (Notion pág 8): el modelo local kza-llm-ik (:8200)
            # fue reemplazado por el gateway LiteLLM → MiniMax, que exige la virtual
            # key (api_key_env). Antes :8200 era local sin auth y este reasoner se
            # construía sin key → tras el cutover daba 400 "No connected db" y el
            # compactor quedaba disabled. Reusamos el gateway autenticado del
            # reasoner principal (instancia aparte = cache pool propio). Override
            # explícito vía orchestrator.context.compaction.{base_url,model}.
            #
            # El endpoint se resuelve vía resolve_compaction_endpoint, que respeta
            # gate_allowed (cloud.consent) — con consent:false esto degrada a un
            # LLM local en vez de heredar el reasoner cloud a ciegas (Critical
            # cerrado, ver .superpowers/sdd/compactor-consent/). Con consent:true
            # el resultado es byte-idéntico a la construcción de antes de este fix
            # (test_compaction_inherits_cloud_endpoint_when_gate_allows).
            compaction_endpoint = resolve_compaction_endpoint(
                compaction_cfg, reasoner_config, gate_allowed and not consent_gate.withdrawn
            )
            compaction_reasoner = HttpReasoner(
                base_url=compaction_endpoint.base_url,
                model=compaction_endpoint.model,
                api_style=compaction_endpoint.api_style,
                api_key_env=compaction_endpoint.api_key_env,
                destination=compaction_endpoint.destination,
                cloud_consent=compaction_endpoint.cloud_consent,
                timeout=compaction_cfg.get("timeout_s", 30.0),
                consent_gate=consent_gate,
            )
            http_reasoners.add_model(compaction_reasoner)
            try:
                await asyncio.to_thread(compaction_reasoner.load)
                compactor = Compactor(
                    reasoner=compaction_reasoner,
                    max_summary_tokens=compaction_cfg.get("max_summary_tokens", 200),
                    timeout_s=compaction_cfg.get("timeout_s", 30.0),
                )
                logger.info(
                    f"[main] Compactor enabled (threshold={compaction_threshold}, "
                    f"keep_recent={keep_recent_turns})"
                )
            except Exception as e:
                logger.error(
                    f"[main] Compactor disabled — could not load reasoner: {e}",
                    exc_info=True,
                )
                compactor = None

    # Persistence sin compaction guardaría el conversation_history crudo en
    # cada snapshot (los turnos no se compactan nunca), inflando el JSON sin
    # techo. Por eso es un hard requirement: persistence implica compaction.
    if persistence_cfg.get("enabled", False):
        if compactor is None:
            logger.error(
                "[main] context.persistence.enabled=true requires compaction.enabled=true. "
                "Disabling persistence."
            )
        else:
            from src.orchestrator import ContextPersister
            persister = ContextPersister(
                base_path=persistence_cfg.get("base_path", "data/contexts"),
                source_policy=privacy_services.ledger,
            )
            logger.info(f"[main] Context persister enabled at {persister.base_path}")

    # Multi-user orchestrator (optional)
    orchestrator = None
    orchestrator_enabled = config.get("orchestrator", {}).get("enabled", True)
    if orchestrator_enabled:
        orchestrator = MultiUserOrchestrator(
            chroma_sync=chroma,
            ha_client=ha_client,
            routine_manager=routine_manager,
            router=llm_router or fast_router,
            llm=llm,
            tts=tts,
            speaker_identifier=speaker_identifier,
            user_manager=user_manager,
            list_manager=list_manager,
            reminder_manager=reminder_manager,
            # Sin esto el dispatcher del orquestador no puede avisar de un
            # fallo de HA y el comando se pierde en silencio (fix 2026-07-25).
            response_handler=response_handler,
            memory_manager=memory_manager,
            privacy=privacy_services,
            response_timeout_s=config.get("orchestrator", {}).get("response_timeout_s", 180.0),
            home_actions=home_action_service,
            home_capabilities=home_capabilities,
            # Plan #2 OpenClaw
            compactor=compactor,
            persister=persister,
            compaction_threshold=compaction_threshold,
            keep_recent_turns=keep_recent_turns,
            hooks=hooks,
            before_handler_warn_ms=self_warn_ms,
            require_known_speaker_for_actions=config.get("security", {}).get(
                "require_known_speaker_for_actions", False
            ),
            # Kill switch del precheck de is_entity_available (default True).
            # Ver comentario en config/settings.yaml:home_assistant.
            unavailable_precheck_enabled=config.get("home_assistant", {}).get(
                "unavailable_precheck_enabled", True
            ),
            # Entidad del clima. Ver config/settings.yaml:home_assistant.
            weather_entity=config.get("home_assistant", {}).get(
                "weather_entity", DEFAULT_WEATHER_ENTITY
            ),
        )

        shutdown.push_async_callback(orchestrator.stop)
        if isinstance(orchestrator.context_manager, ContextManager):
            http_reasoners.add_borrower(orchestrator.context_manager.begin_close)
            hermes_reasoners.add_borrower(orchestrator.context_manager.begin_close)
        context_erasure.attach_manager(orchestrator.context_manager)

    # Request router (command routing: orchestrated + legacy paths)
    confidence_cfg = config.get("orchestrator", {}).get("confidence", {})
    _gate_cfg = config.get("command_gate", {})
    _resolved_wake_words = rooms_config.get("wake_word", {}).get(
        "words", wake_config.get("words", ["nexa"])
    )
    # Wake dedicado (openwakeword): el wake ya se confirmo acusticamente, el
    # texto del comando NO contiene la wake word -> la regla missing_wake del
    # gate seria un false-reject de todo comando valido. Solo aplicar la regla
    # cuando el engine es whisper (wake-as-STT, el texto SI contiene la wake).
    _wake_engine_gate = rooms_config.get("wake_word", {}).get("engine", "openwakeword")
    _gate_wake_words = _resolved_wake_words if _wake_engine_gate == "whisper" else ()
    command_gate = CommandAcceptanceGate(
        wake_words=_gate_wake_words,
        enforce_confidence=_gate_cfg.get("enforce_confidence", False),
        max_no_speech_prob=_gate_cfg.get("max_no_speech_prob", 0.60),
        min_avg_logprob=_gate_cfg.get("min_avg_logprob", -1.20),
        enforce_compression_ratio=_gate_cfg.get("enforce_compression_ratio", False),
        max_compression_ratio=_gate_cfg.get("max_compression_ratio", 2.2),
        # Leer config["stt"] directo: stt_cfg solo se asigna en el branch
        # wake_engine=="whisper" (nunca corre con openwakeword) y usarla acá
        # era UnboundLocalError al boot (review 2026-07-05).
        initial_prompt=config.get("stt", {}).get("initial_prompt"),
    )
    home_voice_resolver = HomeVoiceResolver(
        home_action_service, room_context_manager.get_all_rooms() if room_context_manager else {},
        capabilities=home_capabilities,
    )
    request_router = RequestRouter(
        features=voice_services.features,
        home_voice_resolver=home_voice_resolver,
        before_handler_warn_ms=self_warn_ms,
        unavailable_precheck_enabled=config.get("home_assistant", {}).get("unavailable_precheck_enabled", True),
        command_processor=command_processor,
        response_handler=response_handler,
        home_actions=home_action_service,
        audio_manager=audio_manager,
        room_context_manager=room_context_manager,
        orchestrator=orchestrator,
        orchestrator_enabled=orchestrator_enabled,
        chroma_sync=chroma,
        ha_client=ha_client,
        llm_reasoner=llm,
        fast_router=fast_router,
        memory_manager=memory_manager,
        privacy=privacy_services,
        voice_attributor=voice_attributor,
        user_manager=user_manager,
        enrollment=voice_enrollment,
        conversation_collector=conversation_collector,
        event_logger=event_logger,
        suggestion_engine=suggestion_engine,
        latency_monitor=latency_monitor,
        routine_manager=routine_manager,
        vector_search_threshold=vectordb_config.get("search_threshold", 0.65),
        latency_target_ms=latency_config.get("total", 300),
        suggestion_interval=analytics_config.get("suggestion_interval", 20),
        confidence_threshold=confidence_cfg.get("threshold", 0.75),
        metrics_emitter=metrics_emitter,
        wake_words=_resolved_wake_words,
        llm_command_router=llm_command_router,
        # Con engine acústico (openwakeword/porcupine) el wake ya se confirmó
        # fuera del texto → el grammar fastpath no exige "Nexa" transcripta.
        wake_acoustically_confirmed=(_wake_engine_gate != "whisper"),
        hooks=hooks,
        command_gate=command_gate,
        llm_min_command_confidence=nlu_cfg.get("min_command_confidence", 0.6),
        earcon_cfg=_earcon_cfg,
        asr_event_logger=event_logger,
    )

    shutdown.push_async_callback(request_router.aclose)
    query_work = getattr(request_router, "_query_work", None)
    if isinstance(query_work, OwnedBlockingWork):
        http_reasoners.add_borrower(query_work.begin_close)
        hermes_reasoners.add_borrower(query_work.begin_close)
    context_erasure.attach_router(request_router)
    suggestion_erasure.attach_router(request_router)

    # Wake textual (spec 2026-07-05): red de seguridad puramente textual sobre
    # el ambient path — "nexa" en una utterance no-TV dispara el command path
    # con el texto ya transcripto. Requiere el ambient_path (transcriber) Y el
    # request_router recién construido arriba; se conecta acá y no antes.
    # Best-effort, mismo patrón que el bloque ambient de más arriba.
    _textual_wake_cfg = (config.get("ambient", {}) or {}).get("textual_wake", {}) or {}
    if (
        ambient_path is not None
        and multi_room_loop is not None
        and _textual_wake_cfg.get("enabled", False)
    ):
        try:
            from src.ambient.textual_wake import TextualWakeDetector

            # Etapa A (docs/plans/2026-07-25-reentrenamiento-wake-nexa.md §2):
            # cada disparo textual que llega hasta acá es un falso negativo
            # real del wake acústico. -1.0 = sentinel explícito: el score
            # acústico real (WakeWordDetector.predict() corre continuo en el
            # command path, ver [oww-dbg] en detector.py) no está plomeado
            # hasta acá — no hay accessor tipo last_wake_score(room_id) como
            # sí existe last_command_dispatch_ts(room_id). No es "no existe",
            # es "no está expuesto" (review PR #16). El nombre del archivo no
            # debe fingir un score que no se pudo obtener.
            #
            # 7 días de campaña para juntar unos cientos de eventos raros: un
            # drop silencioso por cola llena arruina el conteo. Loguear.
            def _submit_missed(room_id: str, audio) -> None:
                if wake_clip_writer is None:
                    return
                if not wake_clip_writer.submit_missed(room_id, -1.0, audio):
                    logger.warning(
                        f"[TextualWake] missed clip descartado (cola llena) "
                        f"room={room_id} — dataset de Etapa A subcuenta"
                    )

            textual_wake_detector = TextualWakeDetector(
                dispatch_fn=request_router.process_command,
                last_acoustic_command_ts_fn=multi_room_loop.last_command_dispatch_ts,
                dedup_window_s=_textual_wake_cfg.get("dedup_window_s", 8.0),
                variants=tuple(_textual_wake_cfg.get("variants", ("nexa", "next up"))),
                max_edit_distance=_textual_wake_cfg.get("max_edit_distance", 1),
                min_vad=_textual_wake_cfg.get("min_vad", 0.50),
                on_missed_fn=_submit_missed,
            )
            ambient_path.transcriber.attach_textual_wake(textual_wake_detector)
            logger.info(
                f"[TextualWake] activo (rooms={list(room_streams.keys())})"
            )
        except Exception:
            logger.exception(
                "[TextualWake] no se pudo activar (best-effort) — "
                "el ambient path sigue sin él"
            )

    feature_manager = voice_services.features
    alert_manager = feature_manager.alert_manager

    # ----------------------------------------------------------------
    # Dashboard API + Health Aggregator
    # ----------------------------------------------------------------
    dashboard_config = config.get("dashboard", {})
    dashboard = None
    home_services = build_home_services(
        ha_client,
        room_context_manager.get_all_rooms() if room_context_manager else {},
        multi_room_loop.get_microphone_snapshot if multi_room_loop else None,
        actions=home_action_service,
        capabilities=home_capabilities,
    )
    routine_scheduler = await _initialize_routines(
        config, ha_client=ha_client, home_actions=home_action_service, shutdown=shutdown, start=False,
    )

    if dashboard_authority is not None:
        await dashboard_authority.initialize()
        await _prune_retired_memory_owners(privacy_services, dashboard_authority)

    live_event_bus = await build_live_services(config,privacy=privacy_services,
        owner=erasure_services,shutdown=shutdown)

    ambient_control = None
    if ambient_path is not None:
        from src.ambient.control import build_ambient_control
        ambient_control = await build_ambient_control(privacy=privacy_services,path=ambient_path,
            voices=voice_registry,shutdown=shutdown)

    from src.memory.capture_control import build_capture_control
    from src.memory.capture_native import build_capture_native_drains

    active_audio_loop = multi_room_loop if multi_room_loop else audio_loop
    capture_drains, capture_native_supported = build_capture_native_drains(
        model_owners=(stt_executors, tts_models, speaker_models, wake_models,
                      http_reasoners, hermes_reasoners),
        router=request_router, responses=response_handler,
        context_manager=orchestrator.context_manager if orchestrator else None,
    )
    capture_control = await build_capture_control(
        privacy=privacy_services, audio_loop=active_audio_loop,
        command_processor=command_processor, ambient_control=ambient_control,
        voices=voice_registry,
        collectors=(stt_correction_collector, conversation_collector),
        native_drains=capture_drains, native_supported=capture_native_supported,
        shutdown=shutdown,
    )

    if dashboard_config.get("enabled", True):
        health_aggregator = HealthAggregator(
            ha_client=ha_client,
            latency_monitor=latency_monitor,
            priority_queue=getattr(orchestrator, "_queue", None) if orchestrator else None,
            reminder_scheduler=reminder_scheduler,
            audio_health_provider=multi_room_loop.get_audio_health if multi_room_loop else None,
            loop_lag_provider=loop_lag.snapshot,
            # Only an index type that defines it (test doubles report nothing).
            command_index_provider=(chroma.index_health
                                    if callable(getattr(type(chroma), "index_health", None)) else None),
        )

        dashboard = DashboardAPI(
            authority=dashboard_authority,
            routine_scheduler=routine_scheduler,
            presence_detector=presence_detector,
            ha_client=ha_client,
            list_manager=list_manager,
            reminder_manager=reminder_manager,
            health_aggregator=health_aggregator,
            reminder_scheduler=reminder_scheduler,
            host=dashboard_config.get("host", "127.0.0.1"),
            port=dashboard_config.get("port", 9500),
            cors_config=dashboard_config.get("cors"),
            event_bus=live_event_bus,
            llm_router=llm_router,
            user_manager=user_manager,
            alert_manager=alert_manager,
            zone_manager=zone_manager,
            event_logger=event_logger,
            speaker_identifier=speaker_identifier,
            speaker_model_lock=command_processor.native_model_lock(speaker_identifier),
            llm_metrics=llm_metrics_tracker,
            observability_use_mocks=dashboard_config.get("observability_use_mocks", False),
            home_catalog=home_services.catalog,
            home_actions=home_action_service,
            microphone_snapshot_provider=home_services.microphone_snapshot_provider,
            home_enabled=dashboard_config.get("home", {}).get("enabled", False),
            memory_manager=memory_manager,
            voice_registry=voice_registry,
            ambient_control=ambient_control,
            capture_control=capture_control,
            configuration_changes=configuration_changes,
            consent_gate=consent_gate,
        )
        speaker_models.add_borrower(dashboard.admin_work.begin_close)
        shutdown.push_async_callback(dashboard.stop)
        if destination_registration is not None:
            # Original stop is installed first; attachment failure still drains it.
            await destination_registration.attach_accounts(user_manager, dashboard)
        logger.info(f"Dashboard API configured on {dashboard_config.get('host', '127.0.0.1')}:{dashboard_config.get('port', 9500)}")
    elif destination_registration is not None:
        # The real manager still read/imports accounts; no managed writer exists.
        await destination_registration.accounts_disabled(user_manager)

    # ----------------------------------------------------------------
    # Nightly Training (habit learning + QLoRA)
    # ----------------------------------------------------------------
    training_config = config.get("training", {})
    nightly_config_data = training_config.get("nightly", {})
    nightly_trainer = None

    if nightly_config_data.get("enabled", False):
        habits_config = training_config.get("habits", {})
        habit_generator = HabitDatasetGenerator(
            data_dir=habits_config.get("data_dir", "./data/habit_training"),
            event_logger=event_logger,
            conversation_collector=conversation_collector,
            pattern_learner=None,
            min_confidence=habits_config.get("min_confidence", 0.6),
            synthetic_multiplier=habits_config.get("synthetic_multiplier", 3),
            source_policy=privacy_services.ledger,
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
            output_dir=nightly_config_data.get("output_dir", TRAINING_OUTPUT_NIGHTLY),
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
            stt_correction_collector=stt_correction_collector,
            source_policy=privacy_services.ledger,
        )
        shutdown.push_async_callback(lambda: asyncio.to_thread(nightly_trainer.stop_scheduler))

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
        event_bus=live_event_bus,
        close_dependencies=(
            ((dashboard.admin_work.begin_close,) if dashboard is not None else ())
            + ((multi_room_loop.begin_close,) if multi_room_loop is not None else ())
            + (stt_executors.begin_close, tts_models.begin_close, speaker_models.begin_close,
               wake_models.begin_close, http_reasoners.begin_close, hermes_reasoners.begin_close)
        ),
    )
    if lifecycle is not None:
        pipeline.lifecycle = lifecycle

    shutdown.push_async_callback(pipeline.stop)

    graph = ApplicationGraph(
        config=config, pipeline=pipeline, ha_client=ha_client,
        home_actions=home_action_service, privacy=privacy_services,
        voice_registry=voice_registry, voice_services=voice_services,
        home_services=home_services, home_capabilities=home_capabilities, list_manager=list_manager,
        reminder_manager=reminder_manager, routine_scheduler=routine_scheduler,
        dashboard=dashboard, dashboard_authority=dashboard_authority,
        configuration_changes=configuration_changes, consent_gate=consent_gate,
        models=ApplicationModels(stt, tts, chroma, speaker_identifier, emotion_detector),
        warmup_models=_warmup_models, presence_detector=presence_detector,
        ambient_path=ambient_path, nightly_trainer=nightly_trainer,
        stt_executors=stt_executors, tts_models=tts_models, speaker_models=speaker_models,
        wake_models=wake_models, http_reasoners=http_reasoners,
        capture_control=capture_control, fallback_audio_loop=audio_loop,
        hermes_reasoners=hermes_reasoners,
        erasure_services=erasure_services,
        lifecycle=lifecycle,
        loop_lag=loop_lag,
    )
    from src.memory.producer_erasure import ProducerCoverage, build_producer_erasure

    coverage = ProducerCoverage(
        fallback_audio_loop=audio_loop, hermes_reasoners=hermes_reasoners,
        context_erasure=context_erasure, suggestion_erasure=suggestion_erasure,
        capture_erasure=capture_erasure,
        collectors=(stt_correction_collector, conversation_collector),
    )
    graph.producer_coverage = coverage
    graph.producer_erasure = await build_producer_erasure(
        privacy=privacy_services, owner=erasure_services, graph=graph, coverage=coverage,
    )
    shutdown.push_async_callback(graph.producer_erasure.aclose)
    if configuration_changes is not None:
        configuration_changes.attach_graph(graph)
    await output_registration.composed(destination_registration, graph)
    if admission is not None:
        admission.attach_graph(graph)
        await admission.revalidate_constructed()
    return graph


if __name__ == "__main__":
    run_entry(_ENTRY)
