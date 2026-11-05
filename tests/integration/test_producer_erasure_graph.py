"""Public DI and real local producer receipts; opaque copies stay unaccepted."""
from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
import json
import os
import threading
from types import SimpleNamespace

import pytest

from src.audio.owned_input import InputRetirementError, _OWNED_INPUTS
from src.ambient.ambient_stt import AmbientSTT
from src.ambient.speaker_tagger import SpeakerTagger
from src.main import build_application
from src.memory.erasure_coordinator import ErasureCoordinator
from src.memory.erasure_journal import ErasureParticipant
from src.memory.privacy_services import PrivacyServices
from src.pipeline.command_processor import CommandProcessor
from src.stt.parakeet_stt import ParakeetSTT
from src.users.speaker_identifier import SpeakerIdentifier
from src.users.voice_registry import VoiceRegistry
from src.wakeword.detector import WakeWordDetector
from tests.integration.test_wake_graph_retirement import install_sdk
from tests.mocks.application_graph import offline_home
from tests.mocks.producer_erasure_graph import (
    ACTOR, CHILD_ENV, CHILD_PREFIX, CPUInputBackend, HeldAudio, NativeGate,
    application, configuration, launch, nominal_factories, observed_step,
    producer_step, real_ambient_application, source, unknown_child_command, until,
)


pytestmark = pytest.mark.asyncio


async def test_producer_receipt_completes_for_s_while_actual_u_stt_remains_held(tmp_path, monkeypatch):
    subject_gate, unrelated_gate = NativeGate(), NativeGate()
    async with application(tmp_path, monkeypatch, gates=(subject_gate, unrelated_gate)) as (g, models, home):
        processor = g.pipeline.command_processor
        assert g.producer_erasure.inventory.verify().valid
        subject, unrelated = source(g), source(g)
        original_u = unrelated.to_dict()
        subject_caller = launch(g, subject, .125)
        unrelated_caller = None
        try:
            await until(subject_gate.entered.is_set)
            assert subject_gate.scope == ((subject,), 0)
            unrelated_caller = launch(g, unrelated, .25)
            await until(lambda: processor.resource_status()["native"]["operations"] == 2)
            assert not unrelated_gate.entered.is_set()
            operation = await g.erasure_services.request(subject.conversation_id, ACTOR)
            subject_caller.cancel()
            await asyncio.gather(subject_caller, return_exceptions=True)
            pending = await observed_step(g, operation,
                lambda step: step["state"] == "pending" and bool(step["pending_count"]))
            assert pending["pending_count"] and processor.conversation_busy(subject.conversation_id)
            assert not subject_gate.returned.is_set()
            subject_gate.release.set()
            await until(unrelated_gate.entered.is_set)
            await until(lambda: not processor.conversation_busy(subject.conversation_id))
            complete = await observed_step(g, operation, lambda step: step["state"] == "complete")
            assert complete["pending_count"] == 0
            assert unrelated_gate.scope == ((unrelated,), 0)
            assert not unrelated_gate.returned.is_set() and not unrelated_caller.done()
            assert processor.conversation_busy(unrelated.conversation_id)
            assert g.privacy.sources_available((unrelated,)) and unrelated.to_dict() == original_u
            receipt = await g.privacy.acall(g.privacy.ledger.erasures.snapshot, operation["id"])
            assert receipt["state"] != "complete"
            for participant in ("diagnostics", "streams", "voices"):
                assert any(step["participant"] == participant and step["state"] != "complete"
                    for step in receipt["steps"])
            unrelated_gate.release.set()
            result = await unrelated_caller
            assert result.success and result.text == "Synthetic U command"
            await until(lambda: not processor.snapshot_operations())
            assert subject_gate.completed.is_set() and unrelated_gate.completed.is_set()
            assert models.leaf.calls == ["S", "U"]
            assert unrelated.to_dict() == original_u and home["service_calls"] == 0
            assert g.capture_control.snapshot()["target_muted"] is False
        finally:
            subject_gate.release.set()
            unrelated_gate.release.set()
            await asyncio.gather(*(task for task in (subject_caller, unrelated_caller)
                if task is not None), return_exceptions=True)


@pytest.mark.parametrize("enabled", [False, True])
async def test_disabled_ambient_differs_from_enabled_but_unconstructed_owner(tmp_path, monkeypatch, enabled):
    import src.ambient.transcriber as ambient_module
    attempts = []
    def unavailable_constructor(*args, **kwargs):
        attempts.append(True)
        raise RuntimeError("synthetic ambient constructor unavailable")
    monkeypatch.setattr(ambient_module, "build_ambient_path", unavailable_constructor)
    # A real MultiRoom branch attempts construction when enabled. It has a
    # nominal wake detector but is never started or connected to hardware.
    async with application(tmp_path, monkeypatch, ambient=enabled, rooms=True) as (g, _, _):
        assert bool(attempts) is enabled
        assert g.pipeline.audio_loop.room_streams
        assert g.ambient_path is None
        assert g.producer_erasure.inventory.verify().valid is not enabled
        operation = await g.erasure_services.request(source(g).conversation_id, ACTOR)
        if enabled:
            step = await observed_step(g, operation,
                lambda value: value["state"] == "pending" and value["cursor"] == "inventory_unknown")
            assert step["attempts"] > 0
        else:
            await observed_step(g, operation, lambda value: value["state"] == "complete")


@pytest.mark.parametrize("change", ["command_alias", "command_replacement", "foreign_authority", "coordinator"])
async def test_observed_binding_drift_cannot_be_resealed_by_restoring_old_fields(tmp_path, monkeypatch, change):
    async with application(tmp_path, monkeypatch) as (g, _, _):
        assert g.producer_erasure.inventory.verify().valid
        pipeline, owner = g.pipeline, g.erasure_services
        original_command = pipeline.command_processor
        async with AsyncExitStack() as extra:
            with monkeypatch.context() as patch:
                if change in {"command_alias", "command_replacement"}:
                    replacement = CommandProcessor(g.models.stt, privacy=g.privacy, close_timeout_s=.03)
                    extra.push_async_callback(replacement.aclose)
                    patch.setattr(pipeline.request_router, "command_processor", replacement)
                    if change == "command_replacement":
                        patch.setattr(pipeline, "command_processor", replacement)
                elif change == "foreign_authority":
                    foreign = PrivacyServices(str(tmp_path / "foreign-privacy.db"))
                    extra.push_async_callback(foreign.aclose)
                    await foreign.acall(foreign.initialize)
                    patch.setattr(original_command, "_privacy", foreign)
                else:
                    replacement = ErasureCoordinator(g.privacy)
                    extra.push_async_callback(replacement.aclose)
                    patch.setattr(g, "erasure_services", replacement)
                assert not g.producer_erasure.inventory.verify().valid
                operation = await owner.request(source(g).conversation_id, ACTOR)
                # Use the original real coordinator even while the live DI slot
                # is corrupt; no substitute driver or completion is supplied.
                async with asyncio.timeout(4):
                    while True:
                        await owner.reconcile()
                        before = await producer_step(g, operation)
                        if before and before["state"] == "pending" and before["cursor"] == "inventory_unknown":
                            break
                        await asyncio.sleep(.01)
            assert pipeline.command_processor is original_command
            assert g.erasure_services is owner
            assert not g.producer_erasure.inventory.verify().valid
            await observed_step(g, operation, lambda step:
                step["attempts"] > before["attempts"] and step["state"] == "pending"
                and step["cursor"] == "inventory_unknown")


async def test_unsupported_historical_producer_target_is_not_acknowledged(tmp_path, monkeypatch):
    async with application(tmp_path, monkeypatch) as (g, _, _):
        await g.privacy.acall(g.privacy.ledger.erasures.register_target,
            ErasureParticipant.PRODUCERS, 2, {"kind": "runtime_producers", "protocol": 2})
        operation = await g.erasure_services.request(source(g).conversation_id, ACTOR)
        await observed_step(g, operation, lambda step: step["state"] == "complete")
        unknown = await observed_step(g, operation,
            lambda step: step["last_error"] == "participant_unavailable", version=2)
        assert unknown["state"] != "complete" and unknown["version"] == 2


async def test_actual_unknown_input_callback_blocks_until_copy_and_thread_really_exit(tmp_path, monkeypatch):
    import src.pipeline.audio_loop as audio_module

    backend, gate = CPUInputBackend(), NativeGate()
    monkeypatch.setattr(audio_module.sd, "InputStream", backend.InputStream)
    async with application(tmp_path, monkeypatch, gates=(gate,)) as (g, _, _):
        loop = g.pipeline.audio_loop
        await loop._input.open()
        held = HeldAudio(gate)
        callback = threading.Thread(target=backend.stream.emit, args=(held,),
            name="synthetic-producer-callback")
        callback.start()
        try:
            await until(gate.entered.is_set)
            assert loop._room.callbacks.status()["active"] == 1
            origin = source(g)
            assert loop.conversation_busy(origin.conversation_id)
            operation = await g.erasure_services.request(origin.conversation_id, ACTOR)
            await observed_step(g, operation, lambda step:
                step["state"] == "pending" and bool(step["pending_count"]))
            assert callback.is_alive() and not gate.returned.is_set()
            assert loop._room.callbacks.status()["active"] == 1
            gate.release.set()
            await asyncio.to_thread(callback.join, 2)
            assert not callback.is_alive() and callback._handle.is_done()
            assert gate.completed.is_set() and loop._room.callbacks.status()["active"] == 0
            await observed_step(g, operation, lambda step: step["state"] == "complete")
        finally:
            gate.release.set()
            await asyncio.to_thread(callback.join, 2)
            assert not callback.is_alive(), "fixture callback must really exit before owner close"


async def test_legitimate_lazy_wake_is_owned_before_actual_sdk_constructor_returns(tmp_path, monkeypatch):
    sdk = install_sdk(monkeypatch)
    sdk.hold_constructors.add(0)
    async with application(tmp_path, monkeypatch) as (g, _, _):
        manager = g.pipeline.audio_loop.audio_manager
        owner, factory, model = manager.wake_inventory()
        assert owner is g.wake_models and factory is WakeWordDetector and model is None
        assert g.producer_erasure.inventory.verify().valid
        loading = asyncio.create_task(asyncio.to_thread(manager.load_wake_word))
        try:
            await until(lambda: bool(sdk.gates) and sdk.gates[0].ctor_entered.is_set())
            owner, factory, model = manager.wake_inventory()
            assert owner is g.wake_models and factory is WakeWordDetector
            assert type(model) is WakeWordDetector and not model._loaded
            assert any(candidate is model for candidate in owner._models.values())
            sdk.gates[0].ctor_release.set()
            await loading
            assert model._loaded and sdk.loads == 1
            assert g.producer_erasure.inventory.verify().valid
            # Existing AudioManager semantics create a backend on every
            # explicit load; this replacement must be owned but invalidate
            # the previously observed producer inventory.
            await asyncio.to_thread(manager.load_wake_word)
            replacement = manager.wake_inventory()[2]
            assert sdk.loads == 2 and replacement is not model
            assert any(candidate is replacement for candidate in owner._models.values())
            assert not g.producer_erasure.inventory.verify().valid
        finally:
            for gate in sdk.gates:
                gate.ctor_release.set()
                gate.predict_release.set()
                gate.reset_release.set()
            await asyncio.gather(loading, return_exceptions=True)


async def test_native_close_unknown_cannot_be_acknowledged(tmp_path, monkeypatch):
    if os.environ.get(CHILD_ENV) != "1":
        receipt = await asyncio.to_thread(unknown_child_command)
        assert receipt == {"ok": True, "unknown": True, "retirement_confirmed": False,
            "retained": True, "pointer_null": True, "native_close_calls": 1,
            "producer_complete": False, "cursor": "inventory_unknown", "writer_lease": True,
            "containment_exit_only": True}
        return

    import src.pipeline.audio_loop as audio_module
    backend = CPUInputBackend(fail_close=True)
    monkeypatch.setattr(audio_module.sd, "InputStream", backend.InputStream)
    # This child must retain the honestly uncertain owner.  OS exit is solely
    # containment, never a product retirement receipt or a lease-reset API.
    try:
        async with application(tmp_path, monkeypatch) as (g, _, _):
            try:
                stream_owner = g.pipeline.audio_loop._input
                await stream_owner.open()
                closing = stream_owner.begin_close()
                with pytest.raises(InputRetirementError):
                    await closing
                assert stream_owner.begin_close() is closing
                assert stream_owner.retirement_unknown and not stream_owner.retired
                assert stream_owner in _OWNED_INPUTS
                assert backend.stream.pointer is None and backend.stream.closes == 1
                assert not g.producer_erasure.inventory.verify().valid
                operation = await g.erasure_services.request(source(g).conversation_id, ACTOR)
                step = await observed_step(g, operation, lambda value:
                    value["state"] == "pending" and value["cursor"] == "inventory_unknown")
                receipt = {"ok": True, "unknown": stream_owner.retirement_unknown,
                    "retirement_confirmed": stream_owner.retired, "retained": stream_owner in _OWNED_INPUTS,
                    "pointer_null": backend.stream.pointer is None, "native_close_calls": backend.stream.closes,
                    "producer_complete": step["state"] == "complete", "cursor": step["cursor"],
                    "writer_lease": g.privacy.resource_status()["writer_lease"], "containment_exit_only": True}
            except BaseException as error:
                # Do not enter graph teardown with an honestly unknown input.
                print("\n" + CHILD_PREFIX + json.dumps({"ok": False,
                    "error_type": type(error).__name__}), flush=True)
                os._exit(1)
            print("\n" + CHILD_PREFIX + json.dumps(receipt), flush=True)
            os._exit(0)
    except BaseException as error:
        print("\n" + CHILD_PREFIX + json.dumps({"ok": False, "error_type": type(error).__name__}), flush=True)
        os._exit(1)


class LaterProducerSetupFailure(RuntimeError):
    pass


async def test_later_factory_failure_keeps_native_work_and_writer_lease_until_real_close(tmp_path, monkeypatch):
    import src.memory.producer_erasure as producer_module

    native = NativeGate()
    factories, models = nominal_factories(monkeypatch, tmp_path, subject=native)
    real_build = producer_module.build_producer_erasure
    retained = SimpleNamespace(graph=None, command=None)

    async def fail_after_real_binding(**kwargs):
        driver = await real_build(**kwargs)
        graph = kwargs["graph"]
        assert graph.producer_erasure is driver
        retained.graph = graph
        retained.command = launch(graph, source(graph), .125)
        await until(native.entered.is_set)
        raise LaterProducerSetupFailure("synthetic later factory boundary")

    monkeypatch.setattr(producer_module, "build_producer_erasure", fail_after_real_binding)
    async with offline_home() as (url, _):
        stack = AsyncExitStack()
        closing = None
        try:
            with pytest.raises(LaterProducerSetupFailure):
                await build_application(configuration(tmp_path, url), shutdown=stack, factories=factories)
            assert retained.graph is not None and not native.returned.is_set()
            closing = asyncio.create_task(stack.aclose())
            started = asyncio.Event()
            async def observe_close():
                started.set()
                return await asyncio.shield(closing)
            observer = asyncio.create_task(observe_close())
            await started.wait()
            with pytest.raises(TimeoutError):
                async with asyncio.timeout(.05):
                    await asyncio.shield(observer)
            observer.cancel()
            await asyncio.gather(observer, return_exceptions=True)
            assert not closing.done() and not native.returned.is_set()
            assert retained.graph.privacy.resource_status()["writer_lease"]
            assert retained.graph.privacy.ledger._db is not None
        finally:
            native.release.set()
            if retained.command is not None:
                await asyncio.gather(retained.command, return_exceptions=True)
            if closing is None:
                await stack.aclose()
            else:
                await closing
        assert native.completed.is_set()
        assert models.stt._model is None
        assert retained.graph.privacy.ledger._db is None
        assert not retained.graph.privacy.resource_status()["writer_lease"]


@pytest.mark.parametrize("binding", ["shared", "stt_policy", "tagger_policy", "voice_registry"])
async def test_real_ambient_adapters_require_shared_policy_and_registry(tmp_path, monkeypatch, binding):
    async with real_ambient_application(tmp_path, monkeypatch) as (g, _, home, sdk_loads):
        path = g.ambient_path
        transcriber = path.transcriber
        asr, tagger = transcriber._stt, transcriber._tagger
        assert type(asr) is AmbientSTT and type(tagger) is SpeakerTagger
        assert type(asr._stt) is ParakeetSTT and type(tagger._identifier) is SpeakerIdentifier
        assert asr._source_policy is tagger._source_policy is g.privacy.ledger
        assert tagger._voice_registry is g.voice_registry
        assert path.store.source_policy is g.privacy.ledger
        assert not transcriber._running and not transcriber._tasks and not transcriber._room_states
        assert asr._stt._model is None and tagger._identifier._model is None
        assert g.producer_erasure.inventory.verify().valid
        if binding == "shared":
            operation = await g.erasure_services.request(source(g).conversation_id, ACTOR)
            await observed_step(g, operation, lambda step: step["state"] == "complete")
        else:
            async with AsyncExitStack() as extra:
                foreign = PrivacyServices(str(tmp_path / "foreign-ambient-policy.db"))
                extra.push_async_callback(foreign.aclose)
                await foreign.acall(foreign.initialize)
                with monkeypatch.context() as patch:
                    if binding == "stt_policy":
                        patch.setattr(asr, "_source_policy", foreign.ledger)
                    elif binding == "tagger_policy":
                        patch.setattr(tagger, "_source_policy", foreign.ledger)
                    else:
                        registry = VoiceRegistry(foreign)
                        extra.push_async_callback(registry.aclose)
                        patch.setattr(tagger, "_voice_registry", registry)
                    # Every object remains nominal. Only its actual authority
                    # or registry identity differs from the application graph.
                    assert not g.producer_erasure.inventory.verify().valid
                    operation = await g.erasure_services.request(source(g).conversation_id, ACTOR)
                    step = await observed_step(g, operation, lambda value:
                        value["state"] == "pending" and value["cursor"] == "inventory_unknown")
                    assert step["attempts"] > 0
                assert asr._source_policy is tagger._source_policy is g.privacy.ledger
                assert tagger._voice_registry is g.voice_registry
                assert not g.producer_erasure.inventory.verify().valid
        assert not sdk_loads and home["service_calls"] == 0
