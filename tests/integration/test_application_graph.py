"""Acceptance starts at the effective main factory, with only leaf adapters."""

import asyncio
from contextlib import AsyncExitStack
from dataclasses import replace
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import yaml

from src.application import ApplicationFactories
from src.home_assistant.control_policy import ActionOrigin, HomeActor
from src.home_assistant.home_actions import HomeActionRequest
from src.main import build_application
from src.memory.fact_ledger import MemoryReviewer
from src.pipeline.command_event import CommandEvent
from src.reminders.reminder_store import ReminderActionStatus, ReminderScope
from tests.mocks.application_graph import (
    OfflineEncoder, OfflineSTT, OfflineTTS, OfflineWake, offline_config, offline_home,
)
from tests.mocks.audio_output import AudioBackend
from tests.mocks.memory_sources import memory_source
from tests.mocks.speech import PCMEngine


@pytest.fixture
def leaves(monkeypatch, tmp_path):
    import src.vectordb.chroma_sync as index_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(index_module, "SentenceTransformer", lambda *args, **kwargs: OfflineEncoder())
    stt, tts = OfflineSTT(), OfflineTTS()
    return ApplicationFactories(stt=lambda _: stt, tts=lambda _: tts, preflight=lambda: None)


@pytest.mark.parametrize("rooms", [False, True])
async def test_real_factory_shares_household_output_privacy_and_managers(leaves, tmp_path, rooms):
    async with offline_home() as (url, observations):
        async with AsyncExitStack() as stack:
            graph = await build_application(offline_config(tmp_path, url, rooms=rooms),
                                            shutdown=stack, factories=leaves)
            pipeline = graph.pipeline
            router = pipeline.request_router
            orchestrator = pipeline._orchestrator
            actions = graph.home_actions
            authority = graph.privacy.ledger
            assert pipeline.command_processor is router.command_processor
            assert pipeline.response_handler is router.response_handler is graph.voice_services.responses
            assert pipeline.features is router.features is graph.voice_services.features
            assert pipeline._orchestrator is router._orchestrator
            assert pipeline.command_processor.stt is graph.models.stt
            assert pipeline.response_handler.tts is graph.models.tts
            assert pipeline.chroma is router.chroma is orchestrator.chroma is graph.models.chroma
            assert router.home_actions is orchestrator.home_actions is orchestrator.dispatcher.home_actions is actions
            assert graph.reminder_manager.home_actions is graph.routine_scheduler.executor.home_actions is actions
            assert graph.dashboard.home_actions is actions
            assert graph.dashboard.authority is graph.dashboard_authority
            assert graph.dashboard.app.state.dashboard_authority is graph.dashboard_authority
            assert graph.dashboard_authority._lease_fd is not None
            assert graph.dashboard.home_catalog._policy is actions.policy
            assert actions.ha is router.ha is orchestrator.ha is graph.dashboard.ha is graph.ha_client
            assert graph.list_manager is graph.dashboard.list_manager is orchestrator.dispatcher.list_manager
            assert graph.reminder_manager is graph.dashboard.reminder_manager is orchestrator.dispatcher.reminder_manager
            assert graph.dashboard.scheduler is graph.routine_scheduler
            assert graph.pipeline._event_bus is graph.dashboard.event_bus
            assert pipeline.command_processor._privacy is pipeline.audio_loop._privacy is router._privacy is graph.privacy
            assert router._source_policy is orchestrator._source_policy is authority
            assert orchestrator.context_manager.source_policy is orchestrator.queue.source_policy is authority
            assert graph.voice_registry.privacy is graph.privacy
            assert orchestrator.dispatcher.response_handler is graph.voice_services.responses
            # T030-U1: the orchestrated timer handler drives the one manager the FeatureManager starts.
            assert orchestrator.dispatcher.timer_manager is graph.voice_services.features.timer_manager
            assert orchestrator.dispatcher.intercom is graph.voice_services.features.intercom
            # P3-F1: one household clock, built once from routines.timezone, and one room namespace.
            # Imported here: compiled-restore children import this module's fixtures against
            # older compiled slots that predate src.core.household_clock.
            from src.core.household_clock import HouseholdClock

            clock = graph.household_clock
            assert isinstance(clock, HouseholdClock) and clock.timezone == "America/Montevideo"
            assert orchestrator.dispatcher.household_clock is graph.reminder_manager._household_clock is clock
            assert orchestrator.context_manager.household_clock is clock  # slow-path prompt time
            assert router.household_clock is clock  # legacy prompt time
            assert graph.routine_scheduler._clock == clock.now_utc  # one clock source (review R4)
            assert graph.routine_scheduler.timezone == clock.timezone
            assert orchestrator.dispatcher.room_namespace is graph.room_namespace
            assert graph.room_namespace.rooms == frozenset()
            assert graph.models.stt.loads == graph.models.tts.loads == 0
            assert not graph.routine_scheduler._running
            assert not pipeline._running
            assert graph.dashboard._server is None
            assert observations["service_calls"] == 0
            if rooms:
                reminders = graph.voice_services.reminders
                assert reminders is graph.dashboard.reminder_scheduler
                assert reminders._household_clock is clock
                assert reminders.home_actions is actions
                assert reminders._store is graph.reminder_manager._store
                assert reminders._tts is graph.voice_services.features.voice_delivery
                assert graph.voice_services.reminder_task is None
                # Q31: other activity freeing the output opens a new notice round.
                assert graph.voice_services.responses._output_free_listeners == [reminders.notify_output_available]
                # Q33: an output-free round waits for the active audio loop's follow-up conversation.
                follow_up = pipeline.audio_loop.follow_up
                assert reminders._conversation is follow_up
                assert follow_up._end_listeners == [reminders.notify_conversation_ended]
                # Q33-1 (review round 3): and for a follow-up command still being captured or dispatched.
                turns = pipeline.audio_loop.command_turns
                assert reminders._turns is turns
                assert turns._listeners == [reminders.notify_conversation_ended]
        assert graph.ha_client._session.closed
        assert graph.dashboard_authority._lease_fd is None
        assert not graph.privacy.resource_status()["writer_lease"]
        assert graph.voice_services.responses.get_delivery_status()["closed"]
        assert router._closed and pipeline.command_processor._closed
        assert pipeline.audio_loop._closed


async def test_factory_keeps_authority_open_until_all_consumers_have_closed(leaves, tmp_path, monkeypatch):
    from src.dashboard.authority import DashboardAuthority

    observations = []
    original = DashboardAuthority.aclose
    async def close(authority):
        observations.append((graph.ha_client._session.closed,
                             not graph.privacy.resource_status()["writer_lease"]))
        await original(authority)
    monkeypatch.setattr(DashboardAuthority, "aclose", close)
    async with offline_home() as (url, _):
        async with AsyncExitStack() as stack:
            graph = await build_application(offline_config(tmp_path, url), shutdown=stack, factories=leaves)
            await graph.dashboard.stop()
            assert graph.dashboard_authority._lease_fd is not None
        assert observations == [(True, True)]
        assert graph.dashboard_authority._lease_fd is None


@pytest.mark.parametrize("dashboard_enabled", [False, True])
async def test_factory_loads_canonical_accounts_without_constructing_biometric_models(
    leaves, tmp_path, monkeypatch, dashboard_enabled, dashboard_auth,
):
    import src.main as module
    from src.users.user_manager import PermissionLevel, UserManager

    legacy = UserManager(str(tmp_path / "users.json"))
    account, _ = legacy.add_user("Synthetic", PermissionLevel.ADULT, voice_embedding=np.ones(8, dtype=np.float32))
    previous = legacy.data_path.read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("disabled biometric dependency was constructed")

    monkeypatch.setattr(module, "SpeakerIdentifier", forbidden)
    monkeypatch.setattr(module, "VoiceEnrollment", forbidden)
    async with offline_home() as (url, observations):
        config = offline_config(tmp_path, url)
        config["dashboard"]["enabled"] = dashboard_enabled
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=leaves)
            users = graph.pipeline.command_processor.user_manager
            assert users.voice_review_managed
            assert users.get_user(account.user_id).name == "Synthetic"
            assert users.get_user(account.user_id).voice_embedding is None
            assert users.get_all_embeddings() == {}
            assert graph.models.speaker_identifier is None
            assert users.data_path == tmp_path / "users_accounts_v1.json"
            assert not users.data_path.exists()
            if dashboard_enabled:
                assert graph.dashboard.user_manager is users
                assert graph.dashboard_authority._account_provider() is users.active_accounts()
                targets = frozenset({account.user_id})
                assert not users._retirement_proof(targets)
                root = graph.dashboard_authority.resolve(dashboard_auth['Authorization'].split(' ', 1)[1])
                # U3b-1b: the production proof also requires no owner row for the account.
                # Imported here: compiled-slot rehearsals import this module from pre-U3b slots.
                from src.memory.fact_ledger import MemoryAssigner
                privacy = graph.voice_registry.privacy
                fact = await privacy.acall(lambda: privacy.ledger.confirm(privacy.ledger.store(
                    "Synthetic owned fact", "fact", 1.0, (memory_source(),), proposed=True).id,
                    MemoryReviewer("synthetic-reviewer")))
                await privacy.acall(lambda: privacy.ledger.assign_owner(
                    fact.id, account.user_id, fact.revision, 0, MemoryAssigner("synthetic-operator"),
                    retired=graph.dashboard_authority.memory_owner_retired,
                    active=graph.dashboard_authority.memory_owner_active))
                assert await graph.dashboard_authority.retire_user(root, account.user_id)
                await graph.voice_registry.acall(graph.voice_registry.ledger.people.reconcile)
                assert graph.dashboard_authority.users_retired(targets)
                assert not users._retirement_proof(targets)
                assert await privacy.acall(privacy.ledger.clear_user_facts, targets,
                                           graph.dashboard_authority.users_retired) == 1
                assert users._retirement_proof(targets)
            else:
                assert graph.dashboard is None
                assert users._retirement_proof is None
            created, _ = await graph.voice_registry.acall(users.add_user, "Another", PermissionLevel.ADULT)
            assert created is not None
            assert users.data_path.stat().st_mode & 0o777 == 0o600
            assert UserManager(str(legacy.data_path), voice_registry=graph.voice_registry).get_user(account.user_id).voice_embedding is None
            assert legacy.data_path.read_bytes() == previous
        assert observations["service_calls"] == 0


@pytest.mark.parametrize("close_path", ["factory_stack", "runtime"])
async def test_real_graph_drains_cancelled_admin_write_before_pipeline_and_authority_retirement(
    leaves, tmp_path, monkeypatch, dashboard_auth, close_path,
):
    import os
    from httpx import ASGITransport, AsyncClient

    entered, release = threading.Event(), threading.Event()
    fsync = os.fsync
    retired = []

    def hold(descriptor):
        fsync(descriptor)
        entered.set()
        assert release.wait(5)

    async with offline_home() as (url, observations):
        async with AsyncExitStack() as stack:
            graph = await build_application(offline_config(tmp_path, url), shutdown=stack, factories=leaves)
            original_close = graph.pipeline.command_processor.aclose

            async def close():
                retired.append(True)
                await original_close()

            monkeypatch.setattr(graph.pipeline.command_processor, "aclose", close)
            pipeline_entered = asyncio.Event()

            async def pipeline():
                pipeline_entered.set()
                await asyncio.Event().wait()

            async def serve():
                dashboard = graph.dashboard
                dashboard._serve_task = asyncio.current_task()
                dashboard._server = SimpleNamespace(should_exit=False)
                try:
                    while not dashboard._server.should_exit:
                        await asyncio.sleep(0.001)
                finally:
                    dashboard._serve_task = None

            running = None
            if close_path == "runtime":
                monkeypatch.setattr(graph.pipeline, "run", pipeline)
                monkeypatch.setattr(graph.dashboard, "start", serve)
                running = asyncio.create_task(graph.run())
                await asyncio.wait_for(pipeline_entered.wait(), 3)
            async with AsyncClient(transport=ASGITransport(app=graph.dashboard.app), base_url="http://testserver") as client:
                token = dashboard_auth["Authorization"].split(" ", 1)[1]
                assert (await client.post("/api/admin/auth/login", json={"token": token})).status_code == 200
                monkeypatch.setattr(os, "fsync", hold)
                pending = asyncio.create_task(client.post("/api/admin/users", json={"name": "Retained"}))
                closing = None
                try:
                    assert await asyncio.to_thread(entered.wait, 3)
                    pending.cancel()
                    await asyncio.gather(pending, return_exceptions=True)
                    if running is not None:
                        running.cancel()
                        closing = running
                    else:
                        closing = asyncio.create_task(stack.aclose())
                    await until(lambda: graph.dashboard.admin_work.status()["closed"])
                    if running is not None:
                        running.cancel()
                        await asyncio.sleep(0)
                    assert not closing.done()
                    assert graph.dashboard.admin_work.status()["operations"] == 1
                    assert retired == []
                    assert not graph.pipeline.request_router._closed
                    assert not graph.ha_client._session.closed
                    assert graph.dashboard_authority._lease_fd is not None
                    assert graph.privacy.resource_status()["writer_lease"]
                finally:
                    release.set()
                    await asyncio.gather(pending, return_exceptions=True)
                    if closing is not None:
                        await asyncio.gather(closing, return_exceptions=True)
                assert retired
                assert graph.dashboard.admin_work.status()["operations"] == 0
                assert graph.pipeline.request_router._closed
                assert graph.pipeline.command_processor.user_manager.get_user_by_name("Retained") is not None
        assert graph.dashboard_authority._lease_fd is None
        assert graph.ha_client._session.closed
        assert not graph.privacy.resource_status()["writer_lease"]
        assert observations["service_calls"] == 0


async def test_disabled_dashboard_opens_no_auth_database(leaves, tmp_path):
    async with offline_home() as (url, _):
        config = offline_config(tmp_path, url)
        config["dashboard"]["enabled"] = False
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=leaves)
            assert graph.dashboard is None and graph.dashboard_authority is None
        assert not (tmp_path / "dashboard-auth.db").exists()


async def test_phrase_reaches_actual_main_feature_and_shared_pcm_output(leaves, monkeypatch, tmp_path):
    import src.tts.piper_tts as output_module

    backend = AudioBackend()
    monkeypatch.setattr(output_module, "sd", backend)
    async with offline_home() as (url, observations):
        config = offline_config(tmp_path, url)
        config["orchestrator"]["enabled"] = False
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=leaves)
            await graph.voice_services.features.start()
            event = CommandEvent(np.full(16000, 0.1, dtype=np.float32), room_id="")
            result = await graph.pipeline.process_command(event)
            assert result["intent"] == "timer" and result["success"]
            assert result["outcome"]["persistence"] == "committed"
            assert graph.models.stt.calls == 1
            assert graph.models.tts.texts
            assert len(backend.heard()) > 0
            assert observations["service_calls"] == 0


def _timer_turn(room_id: str = "", **wake) -> CommandEvent:
    return CommandEvent(np.full(16000, 0.1, dtype=np.float32), room_id=room_id, **wake)


async def test_orchestrated_intercom_delivers_to_the_catalog_and_vetoes_ambient_turns(leaves, monkeypatch, tmp_path):
    import src.tts.piper_tts as output_module

    backend = AudioBackend()
    monkeypatch.setattr(output_module, "sd", backend)
    amplifier = MagicMock(audio_output_device=7)
    factories = replace(leaves, wake_detector=OfflineWake, amplifier=lambda **_: amplifier)
    async with offline_home() as (url, observations):
        config = offline_config(tmp_path, url, rooms=True)
        config["rooms"].update({
            "cocina": {"name": "Cocina", "aliases": ["ambiente"], "mic_device_index": 0, "ma1260_zone": 3},
            "hall": {"name": "Hall", "aliases": ["ambiente"], "mic_device_index": 1, "ma1260_zone": 2},
        })
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=factories)
            features = graph.voice_services.features
            await features.start()
            stt, tts = graph.models.stt, graph.models.tts

            for text in ("Avisa, en el fogon: Llego Ana", "Avisa, en el ambiente: Llego Ana"):
                stt.text = text
                clarification = await graph.pipeline.process_command(_timer_turn("hall", capture_trigger="wake"))
                assert clarification["outcome"]["status"] == "needs_input"
                assert "intercom_receipt" not in clarification
                assert "sala configurada" in clarification["response"]
            assert features.get_announcement_history() == []
            assert tts.texts == [clarification["response"]] * 2
            assert [call.args for call in amplifier.select_zone.call_args_list] == [(2,), (2,)]
            tts.texts.clear()
            amplifier.select_zone.reset_mock()

            stt.text = "Avisa, en la cocina: Llego Ana!"
            one = await graph.pipeline.process_command(_timer_turn("hall", capture_trigger="wake"))
            assert (one["intent"], one["path"], one["task_completed"]) == ("intercom", "fast_household", True)
            assert one["intercom_receipt"]["zones"] == {"zone_cocina": "confirmed"}
            assert tts.texts == ["Llego Ana!", "Anuncio reproducido."]
            assert [call.args for call in amplifier.select_zone.call_args_list] == [(3,), (2,)]

            amplifier.select_zone.reset_mock()
            tts.texts.clear()
            stt.text = "Anunciá en toda la casa: Merienda"
            all_rooms = await graph.pipeline.process_command(_timer_turn("hall", capture_trigger="wake"))
            assert all_rooms["task_completed"] and all_rooms["intercom_receipt"]["zones"] == {
                "zone_cocina": "confirmed", "zone_hall": "confirmed",
            }
            assert tts.texts == ["Merienda", "Merienda", "Anuncio reproducido."]
            assert [call.args for call in amplifier.select_zone.call_args_list] == [(3,), (2,), (2,)]
            assert len(backend.heard()) > 0

            history = features.get_announcement_history()
            assert len(history) == 2
            assert [item.announcement_id for item in history] == [
                result["intercom_receipt"]["announcement_id"] for result in (one, all_rooms)
            ]
            output = tts.texts.copy()
            for wake in ({"ambient_strict": True}, {"textual_wake": True}, {"capture_trigger": "follow_up"}):
                rejected = await graph.pipeline.process_command(_timer_turn("hall", **wake))
                assert rejected["intent"] == "llm_rejected:household" and not rejected["success"]
                assert "intercom_receipt" not in rejected
            assert tts.texts == output and features.get_announcement_history() == history

            stt.text = "Anunciá: El pedido está listo."
            literal = await graph.pipeline.process_command(_timer_turn("hall", capture_trigger="wake"))
            assert literal["task_completed"] and literal["intercom_receipt"]["zones"] == {
                "zone_cocina": "confirmed", "zone_hall": "confirmed",
            }
            assert tts.texts[len(output):] == ["El pedido está listo.", "El pedido está listo.", "Anuncio reproducido."]

            graph.voice_services.responses.zone_manager.remove_zone("cocina")
            partial = await graph.pipeline.process_command(_timer_turn("hall", capture_trigger="wake"))
            assert not partial["success"] and not partial["task_completed"]
            assert (partial["outcome"]["dispatch"], partial["outcome"]["effect"]) == ("partial", "partial")
            assert partial["intercom_receipt"]["zones"] == {"zone_cocina": "failed", "zone_hall": "confirmed"}
            assert "1 de 2 salas" in partial["response"]
            assert observations["service_calls"] == 0


async def test_legacy_zone_list_supports_programmatic_announcements_and_honest_results(leaves, monkeypatch, tmp_path):
    import src.tts.piper_tts as output_module

    backend = AudioBackend()
    monkeypatch.setattr(output_module, "sd", backend)
    amplifier = MagicMock(audio_output_device=7)
    factories = replace(leaves, amplifier=lambda **_: amplifier)
    async with offline_home() as (url, observations):
        config = offline_config(tmp_path, url)
        del config["rooms"]
        config["zones"] = {"enabled": True, "zone_list": [
            {"id": "zone_sala", "name": "Sala", "mic_device_index": 0, "ma1260_zone": 1},
            {"id": "zone_hall", "name": "Hall", "mic_device_index": 1, "ma1260_zone": 2},
        ]}
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=factories)
            features = graph.voice_services.features
            await features.start()
            voice, _, announcement = await features.intercom.announce_from_voice("en la sala: Hola")
            assert voice.status.value == "unsupported" and announcement is None

            broadcast = await features.announce("Cena lista")
            directed = await features.announce("Llegó Ana", zones=["zone_hall"])
            emergency = await features.announce_emergency("Aviso urgente")
            both = {"zone_sala": "confirmed", "zone_hall": "confirmed"}
            assert broadcast["success"] and broadcast["intercom_receipt"]["zones"] == both
            assert directed["success"] and directed["intercom_receipt"]["zones"] == {"zone_hall": "confirmed"}
            assert emergency["success"] and emergency["intercom_receipt"]["zones"] == both
            assert features.get_announcement_history()[-1].priority.value == "emergency"
            assert graph.models.tts.texts == ["Cena lista", "Cena lista", "Llegó Ana", "Aviso urgente", "Aviso urgente"]
            assert [call.args for call in amplifier.select_zone.call_args_list] == [(1,), (2,), (2,), (1,), (2,)]

            missing = await features.announce("No hay ruta", zones=["zone_missing"])
            assert not missing["success"] and missing["intercom_receipt"]["zones"] == {"zone_missing": "failed"}
            empty = await features.announce("Sin destinos", zones=[])
            assert not empty["success"] and empty["intercom_receipt"]["zones"] == {}

            backend.fail_start = True
            failed = await features.announce("Salida no disponible", zones=["zone_hall"])
            assert not failed["success"]
            assert failed["intercom_receipt"]["zones"]["zone_hall"] in {"failed", "uncertain"}
            assert amplifier.select_zone.call_count == 6

            def queue_full(_):
                raise asyncio.QueueFull

            monkeypatch.setattr(features.intercom._announcement_queue, "put_nowait", queue_full)
            rejected = await features.announce_emergency("Cola llena")
            assert not rejected["success"] and rejected["intercom_receipt"]["zones"] == {
                "zone_sala": "not_sent", "zone_hall": "not_sent",
            }
            assert amplifier.select_zone.call_count == 6
            assert observations["service_calls"] == 0


@pytest.mark.parametrize("orchestrated", [True, False])
async def test_unknown_voice_dnd_leaves_personal_settings_untouched(leaves, monkeypatch, tmp_path, orchestrated):
    import src.tts.piper_tts as output_module
    from src.notifications.smart_notifications import NotificationRule

    monkeypatch.setattr(output_module, "sd", AudioBackend())
    async with offline_home() as (url, observations):
        config = offline_config(tmp_path, url)
        config["orchestrator"]["enabled"] = orchestrated
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=leaves)
            notifications = graph.voice_services.features.notifications
            notifications.set_dnd("synthetic", True)
            notifications.set_user_preference("synthetic", "channel", "voice")
            rule = NotificationRule("synthetic", "Synthetic rule")
            notifications.add_rule("synthetic", rule)

            graph.models.stt.text = "Activá el modo no molestar"
            result = await graph.pipeline.process_command(_timer_turn(capture_trigger="wake"))
            assert result["intent"] == "notification_config"
            assert result["outcome"]["status"] == "unsupported"
            assert not result["success"] and result["outcome"]["persistence"] == "not_written"
            assert "iniciá sesión en la web" in result["response"]
            assert graph.models.tts.texts == [result["response"]]

            # Even a supplied identity is only a voice label, with no web account authority.
            if orchestrated:
                claimed = await graph.pipeline._orchestrator.process(user_id="synthetic", text="Desactivá el no molestar")
                assert claimed.outcome.status.value == "unsupported" and not claimed.success
                assert claimed.response == result["response"]
            assert notifications._user_dnd == {"synthetic": True}
            assert notifications._user_preferences == {"synthetic": {"channel": "voice"}}
            assert notifications._user_rules == {"synthetic": [rule]}
            assert observations["service_calls"] == 0


async def test_orchestrated_twin_reaches_the_main_timer_and_shared_pcm_output(leaves, monkeypatch, tmp_path):
    """T030-U1: the T010 journey above with orchestrator.enabled true: create, query, pause, resume, cancel."""
    import src.tts.piper_tts as output_module
    from src.timers.named_timers import TimerState

    backend = AudioBackend()
    monkeypatch.setattr(output_module, "sd", backend)
    async with offline_home() as (url, observations):
        config = offline_config(tmp_path, url)
        assert config["orchestrator"]["enabled"] is True
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=leaves)
            features = graph.voice_services.features
            timers = features.timer_manager
            router = graph.pipeline.request_router
            assert router.llm_command_router is None and graph.pipeline._orchestrator.dispatcher.router is None
            search = AsyncMock(side_effect=AssertionError("vector search reached"))
            monkeypatch.setattr(graph.models.chroma, "asearch_command", search)
            await features.start()

            created = await graph.pipeline.process_command(_timer_turn())  # "pon un timer de 5 minutos"
            assert (created["intent"], created["path"], created["success"]) == ("timer", "fast_timer", True)
            assert created["task_completed"] and created["outcome"]["persistence"] == "committed"
            (timer,) = timers.get_all_timers()
            # No zone routes in this graph: "default" is the one local output (R1: deliverable, not a null zone).
            assert features.voice_delivery.require_zone is False
            assert (timer.user_id, timer.zone_id, timer.duration_seconds, timer.name) == ("household", "default", 300, "")
            assert graph.models.tts.texts == ["Temporizador de 5 minutos en marcha."]
            assert graph.models.stt.calls == 1 and len(backend.heard()) > 0

            journey = [
                ("¿cuánto le falta al timer?", TimerState.RUNNING, "Al temporizador de 5 minutos, faltan "),
                ("pausá el timer", TimerState.PAUSED, "Pausé el temporizador de 5 minutos; le quedan "),
                ("seguí el timer", TimerState.RUNNING, "Sigue el temporizador de 5 minutos; faltan "),
                ("cancelá el timer", TimerState.CANCELLED, "Cancelé el temporizador de 5 minutos."),
            ]
            for text, state, spoken in journey:
                graph.models.stt.text = text
                result = await graph.pipeline.process_command(_timer_turn())
                assert (result["intent"], result["path"], result["task_completed"]) == ("timer", "fast_timer", True)
                if text != journey[0][0]:
                    assert result["outcome"]["persistence"] == "committed"
                assert timer.state is state
                assert graph.models.tts.texts[-1].startswith(spoken)
            assert len(timers.get_all_timers()) == 1
            assert observations["service_calls"] == 0
            search.assert_not_awaited()


async def test_stop_by_voice_while_the_timer_rings_on_the_shared_output(leaves, monkeypatch, tmp_path):
    """C02-TIM-0007 (T030-U2) on the real graph: the ring reaches the PCM output, 'pará el timer' silences it,
    nothing is left that could ring again, and the timer state is not touched (it already completed)."""
    import src.tts.piper_tts as output_module
    from src.timers.named_timers import RingDelivery, TimerState

    backend = AudioBackend()
    monkeypatch.setattr(output_module, "sd", backend)
    async with offline_home() as (url, observations):
        config = offline_config(tmp_path, url)
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=leaves)
            features = graph.voice_services.features
            timers = features.timer_manager
            await features.start()
            graph.models.stt.text = "poné un timer de 1 segundo"
            created = await graph.pipeline.process_command(_timer_turn())
            assert (created["path"], created["success"]) == ("fast_timer", True)
            (timer,) = timers.get_all_timers()
            ring = "¡Terminó el temporizador de 1 segundo!"
            for _ in range(100):
                if ring in graph.models.tts.texts:
                    break
                await asyncio.sleep(0.05)
            assert ring in graph.models.tts.texts and timers.ringing_timers() == [timer]
            heard = len(backend.heard())
            assert heard > 0
            graph.models.stt.text = "pará el timer"
            stopped = await graph.pipeline.process_command(_timer_turn())
            assert (stopped["intent"], stopped["path"], stopped["task_completed"]) == ("timer", "fast_timer", True)
            assert stopped["outcome"]["persistence"] == "not_required"
            assert graph.models.tts.texts[-1] == "Paré el temporizador de 1 segundo."
            assert (timer.state, timer.ring) == (TimerState.COMPLETED, RingDelivery.STOPPED)
            assert timers.ringing_timers() == [] and timers.get_status()["delivery_tasks"] == 0
            assert graph.models.tts.texts.count(ring) == 1
            assert observations["service_calls"] == 0


async def test_timer_voice_create_query_and_cancel_across_graph_restarts(leaves, monkeypatch, tmp_path):
    async with offline_home() as (url, observations):
        config = offline_config(tmp_path, url)
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=leaves)
            await graph.voice_services.features.start()
            graph.models.stt.text = "poné un temporizador de 5 minutos para la pasta"
            created = await graph.pipeline.process_command(_timer_turn())
            assert created["task_completed"] and created["outcome"]["persistence"] == "committed"
            (original,) = graph.voice_services.features.timer_manager.get_all_timers()

        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=leaves)
            features = graph.voice_services.features
            timers = features.timer_manager
            monkeypatch.setattr(timers, "_wall_clock", lambda: time.time() + 30)
            await features.start()
            (restored,) = timers.get_all_timers()
            assert (restored.timer_id, restored.ends_at) == (original.timer_id, original.ends_at)
            assert 0 < restored.remaining_seconds <= 270
            assert features.get_status()["timers"]["status"]["durable_active"] == 1
            graph.models.stt.text = "¿cuánto le falta al timer de pasta?"
            queried = await graph.pipeline.process_command(_timer_turn())
            assert queried["task_completed"] and "Al temporizador de pasta, faltan" in queried["response"]

            graph.models.stt.text = "cancelá el timer de pasta"
            cancelled = await graph.pipeline.process_command(_timer_turn())
            assert cancelled["task_completed"] and cancelled["outcome"]["persistence"] == "committed"

        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=leaves)
            await graph.voice_services.features.start()
            assert graph.voice_services.features.timer_manager.get_all_timers() == []
            graph.models.stt.text = "¿cuánto le falta al timer de pasta?"
            queried = await graph.pipeline.process_command(_timer_turn())
            assert not queried["task_completed"] and queried["response"] == "No encontré un temporizador de pasta."
            assert observations["service_calls"] == 0


async def test_timer_store_failure_speaks_no_success_and_leaves_no_timer(leaves, tmp_path):
    async with offline_home() as (url, _):
        async with AsyncExitStack() as stack:
            graph = await build_application(offline_config(tmp_path, url), shutdown=stack, factories=leaves)
            await graph.voice_services.features.start()
            store = graph.reminder_manager._store

            def reject_insert(db):
                db.execute("CREATE TRIGGER reject_timer BEFORE INSERT ON timers "
                           "BEGIN SELECT RAISE(ABORT, 'synthetic storage failure'); END")

            await store._work.write(reject_insert)
            graph.models.stt.text = "poné un temporizador de 5 minutos para la pasta"
            result = await graph.pipeline.process_command(_timer_turn())
            assert not result["task_completed"] and not result["success"]
            assert (result["outcome"]["status"], result["outcome"]["persistence"]) == ("uncertain", "uncertain")
            assert graph.models.tts.texts == ["No puedo confirmar que se haya guardado el cambio del temporizador."]
            timers = graph.voice_services.features.timer_manager
            assert timers.get_all_timers() == []
            assert timers.get_status()["persistence"] == "uncertain"
            assert await store._work.read(lambda db: db.execute("SELECT count(*) FROM timers").fetchone()[0]) == 0


@pytest.mark.parametrize(("command", "paused"), [
    ("pausá el timer de pasta", False),
    ("cancelá el timer de pasta", False),
    ("seguí el timer de pasta", True),
])
async def test_timer_update_failure_preserves_state_and_speaks_uncertainty(leaves, tmp_path, command, paused):
    from src.timers.named_timers import TimerState

    async with offline_home() as (url, _):
        async with AsyncExitStack() as stack:
            graph = await build_application(offline_config(tmp_path, url), shutdown=stack, factories=leaves)
            features = graph.voice_services.features
            await features.start()
            graph.models.stt.text = "poné un temporizador de 5 minutos para la pasta"
            created = await graph.pipeline.process_command(_timer_turn())
            assert created["task_completed"] and created["outcome"]["persistence"] == "committed"
            (timer,) = features.timer_manager.get_all_timers()
            if paused:
                graph.models.stt.text = "pausá el timer de pasta"
                held = await graph.pipeline.process_command(_timer_turn())
                assert held["task_completed"] and held["outcome"]["persistence"] == "committed"
            assert timer.state is (TimerState.PAUSED if paused else TimerState.RUNNING)
            before = (timer.state, timer.revision, timer.ends_at, timer.deadline)
            store = graph.reminder_manager._store

            def reject_update(db):
                db.execute("CREATE TRIGGER reject_timer_update BEFORE UPDATE ON timers "
                           "BEGIN SELECT RAISE(ABORT, 'synthetic storage failure'); END")

            await store._work.write(reject_update)
            spoken = len(graph.models.tts.texts)
            graph.models.stt.text = command
            result = await graph.pipeline.process_command(_timer_turn())
            assert not result["task_completed"] and not result["success"]
            assert (result["outcome"]["status"], result["outcome"]["persistence"]) == ("uncertain", "uncertain")
            assert graph.models.tts.texts[spoken:] == ["No puedo confirmar que se haya guardado el cambio del temporizador."]
            assert (timer.state, timer.revision, timer.ends_at, timer.deadline) == before
            assert features.timer_manager.get_all_timers() == [timer]
            row = await store._work.read(lambda db: tuple(db.execute(
                "SELECT state, revision, ends_at FROM timers WHERE id = ?", (timer.timer_id,)).fetchone()))
            assert row == before[:3]


async def test_timer_expired_during_downtime_delivers_one_notice_across_graph_restarts(leaves, monkeypatch, tmp_path):
    import src.tts.piper_tts as output_module

    backend = AudioBackend()
    monkeypatch.setattr(output_module, "sd", backend)
    async with offline_home() as (url, _):
        config = offline_config(tmp_path, url)
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=leaves)
            features = graph.voice_services.features
            await features.start()
            graph.models.stt.text = "poné un temporizador de 5 minutos para la pasta"
            created = await graph.pipeline.process_command(_timer_turn())
            assert created["task_completed"] and created["outcome"]["persistence"] == "committed"
            (timer,) = features.timer_manager.get_all_timers()
            assert await graph.reminder_manager._store.get_notice_rows() == []
            spoken = len(graph.models.tts.texts)
            heard = backend.heard().size

        expected_notice = "El temporizador de pasta terminó mientras el sistema estaba apagado."
        for _ in range(2):
            async with AsyncExitStack() as stack:
                graph = await build_application(config, shutdown=stack, factories=leaves)
                features = graph.voice_services.features
                timers = features.timer_manager
                monkeypatch.setattr(timers, "_wall_clock", lambda: timer.ends_at + 1)
                await features.start()
                await until(lambda: timers.get_status()["pending_notices"] == 0)
                assert timers.get_all_timers() == [] and timers.ringing_timers() == []
                (notice,) = await graph.reminder_manager._store.get_notice_rows()
                assert notice.request_id == f"notice:timer:{timer.timer_id}"
                assert notice.status is ReminderActionStatus.CONFIRMED
        assert graph.models.tts.texts[spoken:] == [expected_notice]
        assert backend.heard().size > heard


async def test_timer_recovery_failure_does_not_stop_other_features(leaves, tmp_path):
    async with offline_home() as (url, _):
        config = offline_config(tmp_path, url)
        config["alerts"]["enabled"] = True
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=leaves)
            store = graph.reminder_manager._store

            def add_invalid_running_timer(db):
                now = time.time()
                db.execute("INSERT INTO timers (id, label, actor, state, duration_s, created_at, updated_at) "
                           "VALUES ('invalid', '', 'household', 'running', 5, ?, ?)", (now, now))

            await store._work.write(add_invalid_running_timer)
            features = graph.voice_services.features
            await features.start()
            assert features.timer_manager.get_status()["persistence"] == "uncertain"
            assert not features.timer_manager.running
            assert features.intercom.get_status()["running"]
            assert features.notifications.get_status()["running"]
            await until(lambda: features.alert_scheduler.get_status()["running"])


async def test_listo_stops_the_ring_only_while_it_sounds_on_the_real_graph(leaves, monkeypatch, tmp_path):
    """T030-U2b (user answer morning_questions_20261119) on the graph main builds: the command gate gets the
    timer manager's ring check, so 'listo' is noise while a timer only runs and stops the ring when it sounds,
    with the reply and outcome of 'pará el timer'."""
    import src.tts.piper_tts as output_module
    from src.timers.named_timers import RingDelivery, TimerState

    backend = AudioBackend()
    monkeypatch.setattr(output_module, "sd", backend)
    async with offline_home() as (url, observations):
        config = offline_config(tmp_path, url)
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=leaves)
            features = graph.voice_services.features
            timers = features.timer_manager
            await features.start()
            graph.models.stt.text = "poné un timer de 1 segundo"
            created = await graph.pipeline.process_command(_timer_turn())
            assert (created["path"], created["success"]) == ("fast_timer", True)
            (timer,) = timers.get_all_timers()
            assert await timers.commit_command(lambda view: view.pause_timer(timer.timer_id))
            spoken = len(graph.models.tts.texts)
            graph.models.stt.text = "listo"
            early = await graph.pipeline.process_command(_timer_turn())
            assert early["intent"] == "gate_rejected:noise_phrase:'listo'"
            assert timer.state is TimerState.PAUSED and len(graph.models.tts.texts) == spoken
            assert await timers.commit_command(lambda view: view.resume_timer(timer.timer_id))
            ring = "¡Terminó el temporizador de 1 segundo!"
            for _ in range(100):
                if ring in graph.models.tts.texts:
                    break
                await asyncio.sleep(0.05)
            assert ring in graph.models.tts.texts and timers.ringing_timers() == [timer]
            stopped = await graph.pipeline.process_command(_timer_turn())
            assert (stopped["intent"], stopped["path"], stopped["task_completed"]) == ("timer", "fast_timer", True)
            assert stopped["outcome"]["persistence"] == "not_required"
            assert graph.models.tts.texts[-1] == "Paré el temporizador de 1 segundo."
            assert (timer.state, timer.ring) == (TimerState.COMPLETED, RingDelivery.STOPPED)
            assert timers.ringing_timers() == [] and timers.get_status()["delivery_tasks"] == 0
            assert graph.models.tts.texts.count(ring) == 1
            assert observations["service_calls"] == 0


async def test_replay_t010_bare_temporizador_phrase_creates_a_timer(leaves, monkeypatch, tmp_path):
    """T010 graph-third.log:80-83: 'temporizador de 5 minutos' reached vector search; now it is a timer."""
    async with offline_home() as (url, observations):
        async with AsyncExitStack() as stack:
            graph = await build_application(offline_config(tmp_path, url), shutdown=stack, factories=leaves)
            search = AsyncMock(side_effect=AssertionError("vector search reached"))
            monkeypatch.setattr(graph.models.chroma, "asearch_command", search)
            await graph.voice_services.features.start()
            graph.models.stt.text = "temporizador de 5 minutos"
            result = await graph.pipeline.process_command(_timer_turn())
            assert (result["path"], result["outcome"]["status"]) == ("fast_timer", "completed")
            assert [t.duration_seconds for t in graph.voice_services.features.timer_manager.get_all_timers()] == [300]
            search.assert_not_awaited()
            assert observations["service_calls"] == 0


@pytest.mark.parametrize("failure", ["never_started", "start_failed"])
async def test_timer_manager_not_running_never_confirms(leaves, monkeypatch, tmp_path, failure):
    """C02-TIM-0005: a FeatureManager start failure (or no start) gives a spoken refusal and no timer."""
    async with offline_home() as (url, observations):
        async with AsyncExitStack() as stack:
            graph = await build_application(offline_config(tmp_path, url), shutdown=stack, factories=leaves)
            features = graph.voice_services.features
            if failure == "start_failed":
                async def broken_load():
                    raise RuntimeError("synthetic startup failure")
                monkeypatch.setattr(features.intercom, "_load_zones", broken_load)
                with pytest.raises(RuntimeError, match="synthetic startup failure"):
                    await features.start()
            graph.models.stt.text = "poné un temporizador de 5 minutos"
            result = await graph.pipeline.process_command(_timer_turn())
            assert (result["path"], result["outcome"]["status"], result["outcome"]["persistence"]) == (
                "fast_timer", "failed", "not_written")
            assert not result["task_completed"] and not result["success"]
            assert graph.models.tts.texts == ["Ahora no puedo manejar temporizadores. Probá de nuevo en un rato."]
            assert features.timer_manager.get_all_timers() == []
            assert observations["service_calls"] == 0


async def test_timer_with_the_7b_unavailable(leaves, monkeypatch, tmp_path):
    """:8101 down: a strong acoustic wake of six words never calls it; a TV-signal turn fails closed."""
    import httpx
    from src.llm import reasoner

    completions = []
    actual_client = httpx.Client

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"object": "list", "data": [{"id": "synthetic-router", "object": "model"}]})
        completions.append(request.url.path)
        return httpx.Response(503, json={"error": "synthetic 7B unavailable"})

    class HttpxLeaf:
        @staticmethod
        def Client(*args, **kwargs):
            return actual_client(*args, transport=httpx.MockTransport(handler), **kwargs)

        def __getattr__(self, name):
            return getattr(httpx, name)

    monkeypatch.setattr(reasoner, "httpx", HttpxLeaf())
    monkeypatch.setattr(reasoner, "_resolve_api_key", lambda *args, **kwargs: "synthetic-token")
    async with offline_home() as (url, observations):
        config = offline_config(tmp_path, url)
        config["router"] = {"enabled": True, "base_url": "http://synthetic.invalid/v1", "model": "synthetic-router",
                            "destination": "local"}
        config["nlu"] = {"llm_router": {"enabled": True, "timeout_s": 1.0}}
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=leaves)
            assert graph.pipeline.request_router.llm_command_router is not None
            timers = graph.voice_services.features.timer_manager
            await graph.voice_services.features.start()
            graph.models.stt.text = "poné un temporizador de 5 minutos"
            strong = await graph.pipeline.process_command(_timer_turn())
            assert (strong["path"], strong["outcome"]["status"]) == ("fast_timer", "completed")
            assert completions == []
            graph.models.stt.text = "poné un temporizador de 10 minutos"
            weak = await graph.pipeline.process_command(_timer_turn(wake_score=0.42))
            assert weak["intent"] == "llm_rejected:household" and not weak["success"]
            assert len(completions) >= 1
            assert [t.duration_seconds for t in timers.get_all_timers()] == [300]
            assert observations["service_calls"] == 0


async def test_timer_label_never_reaches_timer_or_dispatcher_info_logs(leaves, tmp_path, caplog):
    """C02-PRV-0002 for the timer path (the transcript INFO lines of the router and gate are reported apart)."""
    async with offline_home() as (url, _):
        async with AsyncExitStack() as stack:
            graph = await build_application(offline_config(tmp_path, url), shutdown=stack, factories=leaves)
            await graph.voice_services.features.start()
            graph.models.stt.text = "poné un temporizador de 1 minuto para la pastilla"
            with caplog.at_level("DEBUG"):
                result = await graph.pipeline.process_command(_timer_turn())
            assert result["task_completed"]
            assert [t.name for t in graph.voice_services.features.timer_manager.get_all_timers()] == ["pastilla"]
            leaked = [r for r in caplog.records if r.levelno >= 20 and "pastilla" in r.getMessage()
                      and r.name.startswith(("src.timers", "src.orchestrator", "src.pipeline.feature_manager"))]
            assert leaked == []


@pytest.mark.parametrize(("utc", "text", "spoken", "wake_words"), [
    ((2026, 11, 18, 2, 59, 59), "¿Qué hora es?", "Son las 23:59.", None),
    ((2026, 11, 18, 3, 0, 0), "Nexa, ¿qué día es hoy?", "Hoy es miércoles 18 de noviembre.", None),
    # T037-U3-R1: an alias of the grammar and a configured rooms.wake_word.words entry.
    ((2026, 11, 18, 2, 59, 59), "Alexa, ¿qué hora es?", "Son las 23:59.", None),
    ((2026, 11, 18, 3, 0, 0), "Neska, ¿qué día es hoy?", "Hoy es miércoles 18 de noviembre.", ["nexa", "neska"]),
])
async def test_time_and_date_come_from_the_household_clock_through_the_orchestrated_graph(
        leaves, monkeypatch, tmp_path, utc, text, spoken, wake_words):
    """T037-U3: the built graph answers from its one household clock, with no model and no HA call."""
    from datetime import UTC, datetime

    from src.llm.reasoner import FastRouter
    import src.tts.piper_tts as output_module

    backend = AudioBackend()
    monkeypatch.setattr(output_module, "sd", backend)
    async with offline_home() as (url, observations):
        config = offline_config(tmp_path, url)
        if wake_words is not None:
            config["rooms"]["wake_word"] = {"words": wake_words}
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=leaves)
            dispatcher = graph.pipeline._orchestrator.dispatcher
            assert dispatcher.household_clock is graph.household_clock
            moment = datetime(*utc, tzinfo=UTC).timestamp()
            monkeypatch.setattr(graph.household_clock, "_time_source", lambda: moment)
            spy = MagicMock(spec=FastRouter)
            spy.complete = AsyncMock(side_effect=AssertionError("7B called"))
            spy.generate = MagicMock(side_effect=AssertionError("7B called"))
            dispatcher.router, dispatcher.use_router = spy, True
            graph.models.stt.text = text
            result = await graph.pipeline.process_command(CommandEvent(np.full(16000, 0.1, dtype=np.float32), room_id=""))
            assert (result["intent"], result["path"], result["response"]) == ("clock", "fast_clock", spoken)
            assert result["success"] and result["task_completed"]
            assert graph.models.tts.texts == [spoken] and len(backend.heard()) > 0
            spy.complete.assert_not_called()
            spy.generate.assert_not_called()
            assert observations["service_calls"] == 0


async def test_late_factory_failure_retires_router_and_prior_resources(leaves, monkeypatch, tmp_path):
    import src.main as module

    seen = {}
    real_router = module.RequestRouter

    def router(*args, **kwargs):
        seen["router"] = real_router(*args, **kwargs)
        return seen["router"]

    async def fail(*args, privacy, **kwargs):
        seen["privacy"] = privacy
        raise RuntimeError("synthetic_late_factory_failure")

    monkeypatch.setattr(module, "RequestRouter", router)
    monkeypatch.setattr(module, "build_live_services", fail)
    async with offline_home() as (url, observations):
        with pytest.raises(RuntimeError, match="synthetic_late_factory_failure"):
            async with AsyncExitStack() as stack:
                await build_application(offline_config(tmp_path, url), shutdown=stack, factories=leaves)
        assert seen["router"]._closed
        assert seen["router"].command_processor._closed
        assert seen["router"].response_handler.get_delivery_status()["closed"]
        assert seen["router"].ha._session.closed
        assert not seen["privacy"].resource_status()["writer_lease"]
        assert observations["service_calls"] == 0


async def test_run_starts_routines_then_drains_them_on_pipeline_failure(leaves, monkeypatch, tmp_path):
    async with offline_home() as (url, observations):
        async with AsyncExitStack() as stack:
            graph = await build_application(offline_config(tmp_path, url), shutdown=stack, factories=leaves)
            graph.dashboard = None
            reached = asyncio.Event()

            async def fail():
                assert graph.routine_scheduler._running
                reached.set()
                raise RuntimeError("synthetic_runtime_failure")

            monkeypatch.setattr(graph.pipeline, "run", fail)
            with pytest.raises(RuntimeError, match="synthetic_runtime_failure"):
                await graph.run()
            assert reached.is_set()
            assert not graph.routine_scheduler._running
            assert graph.pipeline.request_router._closed
            with pytest.raises(RuntimeError, match="application_graph_already_run"):
                await graph.run()
        assert not graph.routine_scheduler._running
        assert observations["service_calls"] == 0


async def test_routine_start_failure_disables_the_real_dashboard_facade(leaves, monkeypatch, tmp_path):
    async with offline_home() as (url, _):
        async with AsyncExitStack() as stack:
            graph = await build_application(offline_config(tmp_path, url), shutdown=stack, factories=leaves)
            scheduler = graph.routine_scheduler
            monkeypatch.setattr(scheduler, "start", AsyncMock(side_effect=RuntimeError("synthetic")))
            await graph._start_services()
            assert graph.routine_scheduler is None
            assert graph.dashboard.scheduler is None
            assert not scheduler._running


async def test_warmup_uses_same_models_and_no_operational_start(leaves, tmp_path):
    async with offline_home() as (url, observations):
        async with AsyncExitStack() as stack:
            graph = await build_application(offline_config(tmp_path, url), shutdown=stack, factories=leaves)
            warmup = AsyncMock()
            graph.config["warmup"]["enabled"] = True
            graph.warmup_models = warmup
            await graph.warmup()
            warmup.assert_awaited_once_with(graph.models.stt, graph.models.tts, None, None,
                                           graph.models.chroma, voice_output=graph.voice_services.responses,
                                           command_processor=graph.pipeline.command_processor)
            assert not graph.routine_scheduler._running
            assert observations["service_calls"] == 0


async def test_four_configured_inputs_share_output_privacy_and_native_locks(leaves, tmp_path):
    async with offline_home() as (url, observations):
        config = offline_config(tmp_path, url, rooms=True)
        room_ids = ("living", "hall", "escritorio", "cocina")
        for index, room_id in enumerate(room_ids):
            config["rooms"][room_id] = {"name": room_id, "mic_device_index": index,
                                         "capture_channel": 1}
        factories = replace(leaves, wake_detector=OfflineWake)
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=factories)
            loop = graph.pipeline.audio_loop
            assert set(loop.room_streams) == set(room_ids)
            # P3-F1: the namespace is the configured catalog, shared with the dispatcher.
            assert graph.room_namespace.rooms == frozenset(room_ids)
            assert graph.pipeline._orchestrator.dispatcher.room_namespace is graph.room_namespace
            assert graph.room_namespace.to_room("zone_hall") == "hall"
            assert all(stream.capture_channel == 1 for stream in loop.room_streams.values())
            assert all(stream.wake_detector.loads == 0 for stream in loop.room_streams.values())
            assert loop._response_handler is graph.voice_services.responses
            assert loop._privacy is graph.privacy
            assert graph.pipeline.command_processor._locks is loop.native_model_locks
            assert graph.home_services.microphone_snapshot_provider.__self__ is loop
            assert loop._streams == {}
            assert observations["service_calls"] == 0
        assert not graph.privacy.resource_status()["writer_lease"]


async def test_empty_selection_rejects_action_through_constructed_graph(leaves, tmp_path):
    async with offline_home() as (url, observations):
        async with AsyncExitStack() as stack:
            graph = await build_application(offline_config(tmp_path, url), shutdown=stack, factories=leaves)
            reads = observations["reads"]
            receipt = await graph.home_actions.execute(
                HomeActionRequest("light.synthetic", "turn_on", "denied-action", origin=ActionOrigin.VOICE),
                actor=HomeActor.household(),
            )
            assert (receipt.status, receipt.reason) == ("failed", "action_not_selected")
            assert observations == {"reads": reads, "service_calls": 0}


async def test_committed_settings_announce_a_due_shared_reminder_in_the_living(leaves, monkeypatch, tmp_path, caplog):
    """User decision 2026-11-16: shared reminders are spoken in the living until SCHEMA-V2 adds delivery_room.

    The committed reminders/tts blocks and the committed living/hall MA1260 zones go through the
    actual main factory, which must not warn about the reminder policy; the reminder must reach
    the living's amplifier zone, not the hall's.
    """
    import src.tts.piper_tts as output_module

    committed = yaml.safe_load((Path(__file__).resolve().parents[2] / "config/settings.yaml").read_text(encoding="utf-8"))
    backend = AudioBackend()
    monkeypatch.setattr(output_module, "sd", backend)
    amplifier = MagicMock()
    amplifier.audio_output_device = 7
    factories = replace(leaves, wake_detector=OfflineWake, amplifier=lambda **_: amplifier)
    async with offline_home() as (url, observations):
        config = offline_config(tmp_path, url, rooms=True)
        config["reminders"] = {**committed["reminders"], "db_path": str(tmp_path / "reminders.db")}
        config["tts"]["streaming"]["default_output_zone"] = committed["tts"]["streaming"]["default_output_zone"]
        for index, room_id in enumerate(("living", "hall")):
            config["rooms"][room_id] = {"name": committed["rooms"][room_id]["name"], "mic_device_index": index,
                                         "ma1260_zone": committed["rooms"][room_id]["ma1260_zone"]}
        living_zone = committed["rooms"]["living"]["ma1260_zone"]
        assert living_zone != committed["rooms"]["hall"]["ma1260_zone"]
        async with AsyncExitStack() as stack:
            caplog.clear()
            with caplog.at_level("WARNING", logger="src.reminders.reminder_scheduler"):
                graph = await build_application(config, shutdown=stack, factories=factories)
            assert [record.getMessage() for record in caplog.records
                    if record.name == "src.reminders.reminder_scheduler" and record.levelname == "WARNING"] == []
            reminders = graph.voice_services.reminders
            # The living room's own voice responses use this id (request_router: f"zone_{room_id}").
            assert reminders.output_zone == "zone_living"
            zones = graph.voice_services.responses.zone_manager
            assert zones.get_zone(reminders.output_zone) is zones.get_zone("living")
            store = graph.reminder_manager._store
            reminder = await store.create(user_id="synthetic", text="Regar las plantas", trigger_at=time.time() - 1,
                                          scope=ReminderScope.SHARED)
            assert await reminders._fire_reminder(reminder)
            assert (await store.get_by_id(reminder.id)).state == "fired"
            assert graph.models.tts.texts == ["Oye, recuerda Regar las plantas"]
            assert len(backend.heard()) > 0
            assert [call.args for call in amplifier.select_zone.call_args_list] == [(living_zone,)]
            assert observations["service_calls"] == 0


async def test_committed_settings_ring_a_timer_without_origin_in_the_living(leaves, monkeypatch, tmp_path):
    """T030-U1 R1: a timer turn with no origin room (room_id '', audio_loop) on the committed zones.

    With a ZoneManager the background delivery refuses 'default' (default_output_zone is null), so
    main hands the dispatcher the shared-notice zone (reminders.output_zone, the living): the ring
    must reach the living's amplifier zone instead of completing in silence.
    """
    import src.tts.piper_tts as output_module

    committed = yaml.safe_load((Path(__file__).resolve().parents[2] / "config/settings.yaml").read_text(encoding="utf-8"))
    assert committed["tts"]["streaming"]["default_output_zone"] is None
    backend = AudioBackend()
    monkeypatch.setattr(output_module, "sd", backend)
    amplifier = MagicMock()
    amplifier.audio_output_device = 7
    factories = replace(leaves, wake_detector=OfflineWake, amplifier=lambda **_: amplifier)
    async with offline_home() as (url, observations):
        config = offline_config(tmp_path, url, rooms=True)
        config["reminders"] = {**committed["reminders"], "db_path": str(tmp_path / "reminders.db")}
        config["tts"]["streaming"]["default_output_zone"] = committed["tts"]["streaming"]["default_output_zone"]
        for index, room_id in enumerate(("living", "hall")):
            config["rooms"][room_id] = {"name": committed["rooms"][room_id]["name"], "mic_device_index": index,
                                         "ma1260_zone": committed["rooms"][room_id]["ma1260_zone"]}
        living_zone = committed["rooms"]["living"]["ma1260_zone"]
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=factories)
            features = graph.voice_services.features
            assert features.voice_delivery.require_zone is True
            assert graph.pipeline._orchestrator.dispatcher.timer_fallback_zone == "zone_living"
            await features.start()
            graph.models.stt.text = "poné un timer de 1 segundo"
            result = await graph.pipeline.process_command(_timer_turn())
            assert (result["intent"], result["path"], result["success"]) == ("timer", "fast_timer", True)
            (timer,) = features.timer_manager.get_all_timers()
            assert timer.zone_id == "zone_living"
            ring = "¡Terminó el temporizador de 1 segundo!"
            for _ in range(100):
                if ring in graph.models.tts.texts:
                    break
                await asyncio.sleep(0.05)
            assert ring in graph.models.tts.texts
            assert amplifier.select_zone.call_args_list[-1].args == (living_zone,)
            assert observations["service_calls"] == 0


def patch_main(monkeypatch, config, leaves):
    """Keep the actual entrypoint/factory while replacing config and model leaves."""
    import src.main as module

    monkeypatch.setattr(module, "load_dotenv", lambda: None)
    monkeypatch.setattr(module, "load_config", lambda _: config)
    monkeypatch.setattr(module, "create_stt", leaves.stt)
    monkeypatch.setattr(module, "create_tts", leaves.tts)
    monkeypatch.setattr(module, "_preflight_vram", leaves.preflight)
    return module


async def until(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0.001)


async def test_real_main_builds_warms_starts_and_closes_the_same_graph(leaves, monkeypatch, tmp_path):
    async with offline_home() as (url, observations):
        config = offline_config(tmp_path, url)
        config["warmup"]["enabled"] = True
        module = patch_main(monkeypatch, config, leaves)
        real_build = module.build_application
        seen = []

        async def record(*args, **kwargs):
            graph = await real_build(*args, **kwargs)
            assert not graph.routine_scheduler._running
            assert graph.models.stt.calls == 0 and graph.models.tts.texts == []
            seen.append(graph)
            return graph

        async def terminal(pipeline):
            graph = seen[0]
            assert pipeline is graph.pipeline
            assert graph.models.stt.calls == 1 and graph.models.tts.texts == ["hola"]
            assert graph.routine_scheduler._running
            await asyncio.sleep(0)

        listener = AsyncMock()
        monkeypatch.setattr(module, "build_application", record)
        monkeypatch.setattr(module.VoicePipeline, "run", terminal)
        monkeypatch.setattr(module.DashboardAPI, "start", listener)
        await module.main()
        graph = seen[0]
        listener.assert_awaited_once()
        assert graph.pipeline.request_router._closed
        assert not graph.routine_scheduler._running
        assert graph.ha_client._session.closed
        assert not graph.privacy.resource_status()["writer_lease"]
        assert observations["service_calls"] == 0


async def test_real_main_cancellation_during_late_construction_closes_owners(leaves, monkeypatch, tmp_path):
    async with offline_home() as (url, observations):
        module = patch_main(monkeypatch, offline_config(tmp_path, url), leaves)
        real_router = module.RequestRouter
        entered = asyncio.Event()
        seen = {}

        def record_router(*args, **kwargs):
            seen["router"] = real_router(*args, **kwargs)
            return seen["router"]

        async def hold(*args, privacy, **kwargs):
            seen["privacy"] = privacy
            entered.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(module, "RequestRouter", record_router)
        monkeypatch.setattr(module, "build_live_services", hold)
        task = asyncio.create_task(module.main())
        await asyncio.wait_for(entered.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        router = seen["router"]
        assert router._closed and router.command_processor._closed
        assert router.response_handler.get_delivery_status()["closed"]
        assert router.ha._session.closed
        assert not seen["privacy"].resource_status()["writer_lease"]
        assert observations["service_calls"] == 0


@pytest.mark.parametrize("stage", ["stt", "tts"])
async def test_cancelled_real_main_warmup_retains_native_work_before_authority_closes(
    leaves, monkeypatch, tmp_path, stage,
):
    import src.tts.piper_tts as output_module

    backend = AudioBackend()
    monkeypatch.setattr(output_module, "sd", backend)
    async with offline_home() as (url, observations):
        config = offline_config(tmp_path, url)
        config["warmup"]["enabled"] = True
        module = patch_main(monkeypatch, config, leaves)
        if stage == "stt":
            engine = OfflineSTT(release=threading.Event())
            monkeypatch.setattr(module, "create_stt", lambda _: engine)
        else:
            engine = PCMEngine(frames=240)
            engine.release.clear()
            monkeypatch.setattr(module, "create_tts", lambda _: engine)
        real_build = module.build_application
        seen = []

        async def record(*args, **kwargs):
            graph = await real_build(*args, **kwargs)
            seen.append(graph)
            return graph

        listener, run = AsyncMock(), AsyncMock()
        monkeypatch.setattr(module, "build_application", record)
        monkeypatch.setattr(module.DashboardAPI, "start", listener)
        monkeypatch.setattr(module.VoicePipeline, "run", run)
        task = asyncio.create_task(module.main())
        try:
            await until(engine.entered.is_set)
            graph = seen[0]
            assert not graph.routine_scheduler._running
            task.cancel()
            await until(lambda: graph.pipeline.command_processor._closed if stage == "stt"
                        else graph.voice_services.responses.get_delivery_status()["closed"])
            assert not task.done()
            assert not graph.ha_client._session.closed
            assert graph.privacy.resource_status()["writer_lease"]
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
        finally:
            engine.release.set()
            result = await asyncio.gather(task, return_exceptions=True)
        assert isinstance(result[0], asyncio.CancelledError)
        assert graph.ha_client._session.closed
        assert not graph.privacy.resource_status()["writer_lease"]
        listener.assert_not_awaited()
        run.assert_not_awaited()
        assert observations["service_calls"] == 0
        assert backend.outputs == []


@pytest.mark.parametrize("stop_fails", [False, True])
async def test_repeated_runtime_cancel_observes_tasks_and_runs_all_stops(
    leaves, monkeypatch, tmp_path, stop_fails,
):
    async with offline_home() as (url, observations):
        async with AsyncExitStack() as stack:
            graph = await build_application(offline_config(tmp_path, url), shutdown=stack, factories=leaves)
            entered, stopping, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
            handles = []

            async def background():
                handles.append(asyncio.current_task())
                await asyncio.Event().wait()

            async def stop():
                stopping.set()
                await release.wait()
                if stop_fails:
                    raise RuntimeError("synthetic_stop_failure")

            async def pipeline():
                entered.set()
                await asyncio.Event().wait()

            ambient_stop = AsyncMock()
            graph.ambient_path = SimpleNamespace(start=background, stop=ambient_stop)
            monkeypatch.setattr(graph.dashboard, "start", background)
            monkeypatch.setattr(graph.dashboard, "stop", stop)
            monkeypatch.setattr(graph.pipeline, "run", pipeline)
            task = asyncio.create_task(graph.run())
            try:
                await entered.wait()
                await until(lambda: len(handles) == 2)
                task.cancel()
                await stopping.wait()
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done()
            finally:
                release.set()
                result = await asyncio.gather(task, return_exceptions=True)
            expected = RuntimeError if stop_fails else asyncio.CancelledError
            assert isinstance(result[0], expected)
            assert all(handle.done() for handle in handles)
            ambient_stop.assert_awaited_once()
            assert graph.pipeline.request_router._closed
            assert not graph.routine_scheduler._running
            assert observations["service_calls"] == 0
