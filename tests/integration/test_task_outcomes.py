"""Operation and response receipts from the effective application factory."""

import asyncio
from contextlib import AsyncExitStack
import json
import sqlite3
import time
from unittest.mock import AsyncMock
from uuid import uuid4

from httpx import ASGITransport, AsyncClient
import numpy as np
import pytest

from src.application import ApplicationFactories
from src.core.task_outcome import DispatchState, EffectState
from src.home_assistant.control_policy import ActionOrigin, HomeActor
from src.home_assistant.home_actions import HomeActionRequest
from src.main import build_application
from src.pipeline.command_event import CommandEvent
from src.reminders.reminder_store import ReminderScope
from tests.mocks.application_graph import OfflineEncoder, OfflineSTT, OfflineTTS, offline_config, offline_home
from tests.mocks.audio_output import AudioBackend
from tests.mocks.task_transport import TaskTransport


@pytest.fixture
def leaves(monkeypatch, tmp_path):
    import src.tts.piper_tts as audio
    import src.vectordb.chroma_sync as index

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(index, "SentenceTransformer", lambda *a, **kw: OfflineEncoder())
    backend = AudioBackend()
    monkeypatch.setattr(audio, "sd", backend)
    return ApplicationFactories(stt=lambda _: OfflineSTT(), tts=lambda _: OfflineTTS(), preflight=lambda: None), backend


def event():
    return CommandEvent(np.full(16000, .1, dtype=np.float32), room_id="")


def selection(root, *, group=False):
    entities = [{"entity_id": name, "operations": ["turn_on", "turn_off"],
                 "live_verified": True, "indirect_effects_verified": True}
                for name in ["light.living", "light.second", "light.third"]]
    if group:
        entities.append({"entity_id": "light.group", "kind": "light_group", "operations": ["turn_on", "turn_off"],
                         "members": ["light.living", "light.second", "light.third"], "effects_verified": True,
                         "live_verified": True, "indirect_effects_verified": True})
    path = root / "selection.json"
    path.write_text(json.dumps({"selection_approved": True, "live_api_verified": True, "allowed_entities": entities}))
    path.chmod(0o600)
    return str(path)


@pytest.mark.parametrize("output_fails", [False, True])
async def test_committed_list_item_does_not_disappear_or_replay_after_speech_failure(leaves, tmp_path, output_fails):
    factories, backend = leaves
    backend.fail_start = output_fails
    async with offline_home() as (url, observed):
        config = offline_config(tmp_path, url)
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=factories)
            await graph.list_manager.create_list(None, "compras", shared=True, shared_only=True)
            graph.models.stt.text = "agrega leche a la lista de compras"
            result = await graph.pipeline.process_command(event())
            assert result["intent"] == "list" and result["handled"]
            assert result["task_completed"] and result["outcome"]["persistence"] == "committed"
            assert result["response_completed"] is not output_fails
            assert result["success"] is not output_fails
            assert [i.text for i in await graph.list_manager.get_items(None, "compras", shared_only=True)] == ["leche"]
            assert observed["service_calls"] == 0
        # Restart the real factory with the same stores; never replay the turn.
        async with AsyncExitStack() as stack:
            restored = await build_application(config, shutdown=stack, factories=factories)
            assert [i.text for i in await restored.list_manager.get_items(None, "compras", shared_only=True)] == ["leche"]


@pytest.mark.parametrize("text", ["vacía la lista de inexistente", "recuérdame tomar agua", "cancela el recordatorio de agua"])
async def test_missing_or_unsupported_object_never_counts_as_completed(leaves, tmp_path, text):
    factories, _ = leaves
    async with offline_home() as (url, _):
        async with AsyncExitStack() as stack:
            graph = await build_application(offline_config(tmp_path, url), shutdown=stack, factories=factories)
            graph.models.stt.text = text
            result = await graph.pipeline.process_command(event())
            assert result["handled"] and not result["task_completed"] and not result["success"]
            assert result["response_completed"]
            if text == "recuérdame tomar agua":
                # T031-U2 (deliberate update): creation by voice exists now, and a request without a time
                # asks for it (NEEDS_INPUT, was the interim UNSUPPORTED); nothing is written.
                assert result["outcome"]["status"] == "needs_input"
                assert result["outcome"]["persistence"] == "not_written"
            assert await graph.list_manager.get_all_lists("unknown") == []
            assert await graph.reminder_manager.get_active("unknown") == []
            assert await graph.reminder_manager.get_active(None, shared_only=True) == []


@pytest.mark.parametrize("orchestrated", [False, True])
@pytest.mark.parametrize("mode,dispatch,effect", [
    ("confirmed", "accepted", "observed"), ("unconfirmed", "accepted", "unconfirmed"),
    ("rejected", "rejected", "not_observed"), ("uncertain", "uncertain", "uncertain"),
    ("observation_error", "accepted", "unconfirmed"),
])
async def test_home_receipt_survives_both_router_modes(leaves, monkeypatch, tmp_path, orchestrated, mode, dispatch, effect):
    factories, _ = leaves
    transport = TaskTransport(mode)
    async with transport.serve() as url:
        config = offline_config(tmp_path, url)
        config["orchestrator"]["enabled"] = orchestrated
        config["home_assistant"].update(control_selection_path=selection(tmp_path), state_confirmation_timeout_s=.03)
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=factories)
            graph.models.stt.text = "prendé la luz del living"
            graph.voice_services.responses.set_earcon(np.ones(240, dtype=np.float32) * .1, 24000)
            command = {
                "entity_id": "light.living", "domain": "light", "service": "turn_on", "data": {},
                "description": "Luz encendida", "similarity": 1.0,
            }
            monkeypatch.setattr(graph.models.chroma, "asearch_command", AsyncMock(return_value=command))
            monkeypatch.setattr(graph.models.chroma, "search_command", lambda *a, **kw: command)
            result = await graph.pipeline.process_command(event(), request_id="server:known-turn")
            assert result["receipt"]["request_id"] == "server:known-turn"
            assert result["receipt"]["dispatch"] == dispatch
            assert result["receipt"]["effect"] == effect
            assert result["outcome"]["dispatch"] == dispatch and result["outcome"]["effect"] == effect
            assert len(transport.sends) == 1
            if mode == "confirmed":
                assert result["task_completed"] and result["success"]
                assert result["response_completed"] is None and not result["response_required"]
                assert graph.models.tts.texts == []
            else:
                assert not result["task_completed"] and not result["success"]
                assert result["response_required"]


async def test_cancelled_waiters_keep_same_server_receipt_and_one_send(leaves, tmp_path):
    factories, _ = leaves
    transport = TaskTransport(hold=True)
    async with transport.serve() as url:
        config = offline_config(tmp_path, url)
        config["home_assistant"]["control_selection_path"] = selection(tmp_path)
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=factories)
            request = HomeActionRequest("light.living", "turn_on", "server:stable", origin=ActionOrigin.VOICE)
            actor = HomeActor.household()
            waiters = [asyncio.create_task(graph.home_actions.execute(request, actor=actor)) for _ in range(2)]
            await asyncio.wait_for(transport.entered.wait(), timeout=2)
            for task in waiters:
                task.cancel()
            await asyncio.gather(*waiters, return_exceptions=True)
            transport.release.set()
            receipt = await graph.home_actions.execute(request, actor=actor)
            assert receipt.dispatch is DispatchState.ACCEPTED and receipt.effect is EffectState.OBSERVED
            assert len(transport.sends) == 1
            assert graph.home_actions.get_receipt(actor=actor, request_id=request.request_id) is receipt


@pytest.mark.parametrize("orchestrated", [False, True])
async def test_cancelled_pipeline_turn_can_rejoin_receipt_without_replaying_or_mixing_ids(
    leaves, monkeypatch, tmp_path, orchestrated,
):
    factories, _ = leaves
    transport = TaskTransport(hold=True)
    async with transport.serve() as url:
        config = offline_config(tmp_path, url)
        config["orchestrator"]["enabled"] = orchestrated
        config["home_assistant"]["control_selection_path"] = selection(tmp_path)
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=factories)
            graph.models.stt.text = "prendé la luz del living"
            command = {"entity_id": "light.living", "domain": "light", "service": "turn_on",
                       "data": {}, "description": "Luz encendida", "similarity": 1.0}
            monkeypatch.setattr(graph.models.chroma, "asearch_command", AsyncMock(return_value=command))
            monkeypatch.setattr(graph.models.chroma, "search_command", lambda *a, **kw: command)
            initial = asyncio.create_task(graph.pipeline.process_command(event(), request_id="server:cancelled"))
            await asyncio.wait_for(transport.entered.wait(), timeout=2)
            initial.cancel()
            with pytest.raises(asyncio.CancelledError):
                await initial
            retry = asyncio.create_task(graph.pipeline.process_command(event(), request_id="server:cancelled"))
            other = asyncio.create_task(graph.pipeline.process_command(event(), request_id="server:independent"))
            transport.release.set()
            results = await asyncio.wait_for(asyncio.gather(retry, other), timeout=3)
            assert [result["receipt"]["request_id"] for result in results] == ["server:cancelled", "server:independent"]
            assert all(result["task_completed"] for result in results)
            assert len(transport.sends) == 2
            from src.core.task_outcome import CURRENT_REQUEST_ID
            assert CURRENT_REQUEST_ID.get() is None


async def test_missing_http_object_has_no_successful_mutation_receipt(leaves, tmp_path, dashboard_auth):
    factories, _ = leaves
    async with offline_home() as (url, _):
        async with AsyncExitStack() as stack:
            graph = await build_application(offline_config(tmp_path, url), shutdown=stack, factories=factories)
            async with AsyncClient(transport=ASGITransport(app=graph.dashboard.app), base_url="http://test", headers=dashboard_auth) as client:
                assert (await client.delete("/api/reminders/missing")).status_code == 404
                assert (await client.delete("/api/lists/missing/items/missing")).status_code == 404
                created = await client.post("/api/lists", json={"name": "compras", "user_id": "fixture"})
                assert created.json()["outcome"]["persistence"] == "committed"


async def test_named_reminder_cancellation_cannot_cancel_an_unrelated_queued_request(leaves, tmp_path):
    factories, _ = leaves
    async with offline_home() as (url, _):
        async with AsyncExitStack() as stack:
            graph = await build_application(offline_config(tmp_path, url), shutdown=stack, factories=factories)
            reminder = await graph.reminder_manager.create("unknown", "tomar agua", time.time() + 3600,
                scope=ReminderScope.SHARED)
            source = graph.privacy.source_for_capture(str(uuid4()), time.time(), "")
            from src.orchestrator.priority_queue import Priority
            queued = graph.pipeline._orchestrator.queue.enqueue("unknown", "synthetic queued request",
                                                                Priority.LOW, memory_sources=(source,))
            graph.models.stt.text = "cancela el recordatorio de tomar agua"
            result = await graph.pipeline.process_command(event())
            assert result["intent"] == "reminder" and result["task_completed"]
            assert not queued.is_cancelled
            stored = await graph.reminder_manager._store.get_by_id(reminder.id)
            assert stored.state == "cancelled"


async def test_voice_cancel_of_a_persisted_shared_reminder_is_one_store_transition_and_survives_rebuild(
        leaves, tmp_path):
    """T032-U1 journey: a voseo cancel through the real graph, after and before a rebuild."""
    factories, _ = leaves
    async with offline_home() as (url, observed):
        config = offline_config(tmp_path, url)
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=factories)
            water = await graph.reminder_manager.create("household", "tomar agua", time.time() + 3600,
                                                        scope=ReminderScope.SHARED)
            trash = await graph.reminder_manager.create("household", "sacar la basura", time.time() + 7200,
                                                        scope=ReminderScope.SHARED)
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=factories)
            store = graph.reminder_manager._store
            store_cancel = AsyncMock(side_effect=store.cancel)
            store.cancel = store_cancel
            queue = graph.pipeline._orchestrator.queue
            source = graph.privacy.source_for_capture(str(uuid4()), time.time(), "")
            from src.orchestrator.priority_queue import Priority
            queued = queue.enqueue("unknown", "synthetic queued request", Priority.LOW, memory_sources=(source,))
            original_queue_cancel = queue.cancel_user_request
            queue_cancels = []

            def counted_queue_cancel(*args, **kwargs):
                queue_cancels.append(args)
                return original_queue_cancel(*args, **kwargs)
            queue.cancel_user_request = counted_queue_cancel
            search = AsyncMock(side_effect=graph.models.chroma.asearch_command)
            graph.models.chroma.asearch_command = search
            graph.models.stt.text = "borrá el recordatorio de tomar agua"
            result = await graph.pipeline.process_command(event())
            assert (result["path"], result["intent"]) == ("fast_reminder", "reminder")
            assert result["task_completed"] and result["outcome"]["persistence"] == "committed"
            assert result["response"].startswith("Cancelé el recordatorio tomar agua a las ")
            assert store_cancel.await_count == 1 and queue_cancels == [] and not queued.is_cancelled
            search.assert_not_awaited()
            assert observed["service_calls"] == 0
            # Repeating the request never reports a second cancellation and never reaches the store write.
            again = await graph.pipeline.process_command(event())
            assert not again["task_completed"] and again["outcome"]["status"] == "needs_input"
            assert store_cancel.await_count == 1 and queue_cancels == []
        async with AsyncExitStack() as stack:
            restored = await build_application(config, shutdown=stack, factories=factories)
            assert (await restored.reminder_manager._store.get_by_id(water.id)).state == "cancelled"
            assert [r.id for r in await restored.reminder_manager.get_active(None, shared_only=True)] == [trash.id]
            with sqlite3.connect(config["reminders"]["db_path"]) as db:
                assert db.execute("SELECT COUNT(*) FROM reminders").fetchone()[0] == 2


async def test_timer_recognition_and_duplicate_name_are_not_operation_completion(leaves, tmp_path):
    factories, _ = leaves
    async with offline_home() as (url, _):
        config = offline_config(tmp_path, url)
        config["orchestrator"]["enabled"] = False
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=factories)
            graph.models.stt.text = "pon un timer de cero minutos"
            invalid = await graph.pipeline.process_command(event())
            assert invalid["handled"] and not invalid["task_completed"]
            graph.models.stt.text = "timer pasta 5 minutos"
            created = await graph.pipeline.process_command(event())
            assert created["task_completed"] and created["outcome"]["persistence"] == "volatile"
            duplicate = await graph.pipeline.process_command(event())
            assert duplicate["handled"] and not duplicate["task_completed"]
            assert len(graph.voice_services.features.timer_manager.get_active_timers()) == 1


async def test_partial_group_preserves_each_leaf_and_failed_feedback(leaves, monkeypatch, tmp_path):
    factories, backend = leaves
    backend.fail_start = True
    transport = TaskTransport(modes={"light.second": "uncertain"})
    async with transport.serve() as url:
        config = offline_config(tmp_path, url)
        config["home_assistant"]["control_selection_path"] = selection(tmp_path, group=True)
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=factories)
            graph.models.stt.text = "prendé la luz del living"
            graph.voice_services.responses.set_earcon(np.ones(240, dtype=np.float32) * .1, 24000)
            command = {"entity_id": "light.group", "domain": "light", "service": "turn_on", "data": {},
                       "description": "Grupo encendido", "similarity": 1.0}
            monkeypatch.setattr(graph.models.chroma, "asearch_command", AsyncMock(return_value=command))
            monkeypatch.setattr(graph.models.chroma, "search_command", lambda *a, **kw: command)
            result = await graph.pipeline.process_command(event())
            assert result["receipt"]["reason"] == "partial_effects"
            assert result["outcome"]["effect"] == "partial"
            assert result["outcome"]["dispatch"] == "uncertain"
            assert not result["task_completed"] and not result["response_completed"]
            assert [leaf["status"] for leaf in result["receipt"]["effects"]] == ["confirmed", "uncertain", "not_executed"]
            assert [send[0] for send in transport.sends] == ["light.living", "light.second"]


async def test_optional_ha_replication_failure_does_not_erase_committed_list_item(leaves, tmp_path):
    factories, _ = leaves
    async with offline_home() as (url, _):
        async with AsyncExitStack() as stack:
            graph = await build_application(offline_config(tmp_path, url), shutdown=stack, factories=factories)
            from types import SimpleNamespace
            await graph.list_manager.create_list(None, "compras", shared=True, shared_only=True)
            graph.list_manager._ha_sync = SimpleNamespace(sync_add_item=AsyncMock(side_effect=RuntimeError("synthetic")))
            graph.models.stt.text = "agrega leche a la lista de compras"
            result = await graph.pipeline.process_command(event())
            assert result["task_completed"]
            assert result["outcome"]["persistence"] == "committed"
            assert result["outcome"]["replication"] == "uncertain"
            assert len(await graph.list_manager.get_items(None, "compras", shared_only=True)) == 1


@pytest.mark.parametrize("kind", ["list_item", "list", "clear_list", "reminder"])
async def test_stale_lookup_cannot_prove_a_committed_deletion(leaves, tmp_path, kind):
    factories, _ = leaves
    async with offline_home() as (url, _):
        async with AsyncExitStack() as stack:
            graph = await build_application(offline_config(tmp_path, url), shutdown=stack, factories=factories)
            if kind == "reminder":
                await graph.reminder_manager.create("unknown", "tomar agua", time.time() + 3600)
                graph.reminder_manager._store.cancel = AsyncMock(return_value=False)
                assert not await graph.reminder_manager.cancel_by_text("unknown", "tomar agua")
            else:
                await graph.list_manager.add_item("unknown", "leche", "compras")
                if kind == "list_item":
                    graph.list_manager._store.remove_item = AsyncMock(return_value=False)
                    result = await graph.list_manager.remove_item_with_outcome("unknown", "leche", "compras")
                    assert result.item is None and not result.outcome.task_completed
                    assert result.outcome.persistence.value == "not_written"
                elif kind == "clear_list":
                    graph.list_manager._store.clear_list = AsyncMock(return_value=False)
                    assert not await graph.list_manager.clear_list("unknown", "compras")
                else:
                    graph.list_manager._store.delete_list = AsyncMock(return_value=False)
                    assert not await graph.list_manager.delete_list("unknown", "compras")


@pytest.mark.parametrize(("text", "path", "intent", "status"), [
    ("recordame apagar la luz", "fast_reminder", "reminder", "unsupported"),
    ("Nexa, recordame prender la luz del living", "fast_reminder", "reminder", "unsupported"),
    # T030-U1: the graph is built but the FeatureManager was never started, so no timer is confirmed.
    ("poné un temporizador de 5 minutos", "fast_timer", "timer", "failed"),
    ("cancelá el timer y el recordatorio", "fast_household", "compound", "needs_input"),
    ("apagá la luz y pausá la música", "fast_household", "compound", "needs_input"),
])
async def test_claimed_household_turn_never_reaches_home_control(leaves, monkeypatch, tmp_path, text, path, intent,
                                                                 status):
    """P3-F2 (CORE-02): a household turn is claimed before the grammar, the 7B and vector search."""
    factories, _ = leaves
    transport = TaskTransport("confirmed")
    async with transport.serve() as url:
        config = offline_config(tmp_path, url)
        config["home_assistant"].update(control_selection_path=selection(tmp_path), state_confirmation_timeout_s=.03)
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=factories)
            command = {"entity_id": "light.living", "domain": "light", "service": "turn_off", "data": {},
                       "description": "Luz apagada", "similarity": 1.0}
            search = AsyncMock(return_value=command)
            monkeypatch.setattr(graph.models.chroma, "asearch_command", search)
            graph.models.stt.text = text
            result = await graph.pipeline.process_command(event())
            assert (result["path"], result["intent"]) == (path, intent)
            assert result["handled"] and not result["task_completed"] and not result["success"]
            assert result["outcome"]["status"] == status
            assert graph.voice_services.features.timer_manager.get_all_timers() == []
            assert result["response_completed"]
            assert transport.sends == []
            search.assert_not_awaited()
            assert await graph.reminder_manager.get_active(None) == []
            router = graph.pipeline.request_router
            classifier = router.household_intents
            assert classifier is router._orchestrator.dispatcher.household_intents
            assert classifier is router.command_gate._household_intents
