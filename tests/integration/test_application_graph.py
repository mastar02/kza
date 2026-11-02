"""Acceptance starts at the effective main factory, with only leaf adapters."""

import asyncio
from contextlib import AsyncExitStack
from dataclasses import replace
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest

from src.application import ApplicationFactories
from src.home_assistant.control_policy import ActionOrigin, HomeActor
from src.home_assistant.home_actions import HomeActionRequest
from src.main import build_application
from src.pipeline.command_event import CommandEvent
from tests.mocks.application_graph import (
    OfflineEncoder, OfflineSTT, OfflineTTS, OfflineWake, offline_config, offline_home,
)
from tests.mocks.audio_output import AudioBackend
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
            assert graph.models.stt.loads == graph.models.tts.loads == 0
            assert not graph.routine_scheduler._running
            assert not pipeline._running
            assert graph.dashboard._server is None
            assert observations["service_calls"] == 0
            if rooms:
                reminders = graph.voice_services.reminders
                assert reminders is graph.dashboard.reminder_scheduler
                assert reminders.home_actions is actions
                assert reminders._store is graph.reminder_manager._store
                assert reminders._tts is graph.voice_services.features.voice_delivery
                assert graph.voice_services.reminder_task is None
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
                assert await graph.dashboard_authority.retire_user(root, account.user_id)
                await graph.voice_registry.acall(graph.voice_registry.ledger.people.reconcile)
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
            event = CommandEvent(np.full(16000, 0.1, dtype=np.float32), room_id="")
            result = await graph.pipeline.process_command(event)
            assert result["intent"] == "timer" and result["success"]
            assert graph.models.stt.calls == 1
            assert graph.models.tts.texts
            assert len(backend.heard()) > 0
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
