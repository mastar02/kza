"""Truthful closure observations over real graph/native/SDK ownership."""
import asyncio
import importlib.util
import time

import pytest

from src.memory import runtime_closure
from src.memory.runtime_closure import RuntimeClosureError, RuntimeClosureWitness
from src.stt.parakeet_stt import ParakeetSTT
from tests.mocks.memory_graph import Clock
from tests.mocks.restore_application_graph import application_graph


async def test_real_graph_and_public_sdk_close_produce_content_free_proof(tmp_path, monkeypatch):
    async with application_graph(tmp_path, monkeypatch, Clock(time.time())) as bundle:
        witness = RuntimeClosureWitness.capture(bundle.graph)
        chroma = bundle.graph.models.chroma
        assert chroma._client is None and not chroma._ready
        # Match main's capture-before-warmup order. Acquisition is the real
        # public API after the witness has installed its constructor observer.
        await asyncio.to_thread(chroma.initialize)
        client, closed = chroma._client, []
        original_close = client.close

        def observed_close():
            result = original_close()
            closed.append(True)
            return result

        monkeypatch.setattr(client, "close", observed_close)
        with pytest.raises(RuntimeClosureError, match="^runtime_closure_pre_close_required$"):
            witness.verify_closed()
        await witness.pre_close()
        assert bundle.graph.privacy.ledger.authority_available
        assert bundle.sdk_clients and all(not client.closed for client in bundle.sdk_clients)
    assert all(client.closed for client in bundle.sdk_clients)
    assert not bundle.graph.privacy.ledger.authority_available
    proof = witness.verify_closed()
    assert closed == [True] and proof["chroma_opened"] is True
    assert proof["chroma_never_opened"] is False
    assert proof["schema_version"] == 1 and proof["model_owner_count"] == 6
    assert proof["sdk_owner_count"] == 2 and proof["input_owner_count"] >= 1
    assert {key: proof[key] for key in ("bindings_unchanged", "model_owners_closed",
        "graph_closed", "privacy_closed", "sdk_closed", "native_closed")} == dict.fromkeys(
            ("bindings_unchanged", "model_owners_closed", "graph_closed",
             "privacy_closed", "sdk_closed", "native_closed"), True)


async def test_never_opened_chroma_is_sealed_without_a_fabricated_sdk_close(tmp_path, monkeypatch):
    async with application_graph(tmp_path, monkeypatch, Clock(time.time())) as bundle:
        chroma = bundle.graph.models.chroma
        assert chroma._client is None and not chroma._ready
        witness = RuntimeClosureWitness.capture(bundle.graph)
        await witness.pre_close()
    proof = witness.verify_closed()
    assert proof["chroma_opened"] is False and proof["chroma_never_opened"] is True
    assert chroma._admission_closed and chroma._client is None
    assert chroma._phase.value == "complete" and not chroma._client_closed
    assert proof["sdk_closed"] is True


async def test_uncertain_lazy_sdk_constructor_never_becomes_a_close_ack(tmp_path, monkeypatch):
    import src.vectordb.chroma_sync as index_module

    async with application_graph(tmp_path, monkeypatch, Clock(time.time())) as bundle:
        chroma = bundle.graph.models.chroma
        witness = RuntimeClosureWitness.capture(bundle.graph)

        def unreturned_client(**kwargs):
            raise RuntimeError("synthetic_constructor_without_handle")

        with monkeypatch.context() as patch:
            patch.setattr(index_module.chromadb, "PersistentClient", unreturned_client)
            with pytest.raises(RuntimeError, match="^chroma_sync_initialization_failed$"):
                await asyncio.to_thread(chroma.initialize)
        assert chroma._client is None
        with pytest.raises(RuntimeClosureError, match="^runtime_closure_sdk_binding_changed$"):
            await witness.pre_close()
    assert chroma._phase.value == "complete"
    with pytest.raises(RuntimeClosureError, match="^runtime_closure_pre_close_required$"):
        witness.verify_closed()


async def test_unreviewed_leaf_cannot_obtain_a_graph_closure_witness(tmp_path, monkeypatch):
    async with application_graph(tmp_path, monkeypatch, Clock(time.time())) as bundle:
        bundle.graph.stt_executors.add_model(object())
        assert bundle.graph.stt_executors._capture_unreviewed
        with pytest.raises(RuntimeClosureError, match="^runtime_closure_inventory_invalid$"):
            RuntimeClosureWitness.capture(bundle.graph)


async def test_binding_drift_is_terminal_even_if_original_leaf_is_restored(tmp_path, monkeypatch):
    async with application_graph(tmp_path, monkeypatch, Clock(time.time())) as bundle:
        witness = RuntimeClosureWitness.capture(bundle.graph)
        command = bundle.graph.pipeline.command_processor
        original = command.stt
        try:
            command.stt = object()
            with pytest.raises(RuntimeClosureError, match="^runtime_closure_binding_changed$"):
                await witness.pre_close()
        finally:
            command.stt = original
        with pytest.raises(RuntimeClosureError, match="^runtime_closure_witness_failed$"):
            await witness.pre_close()
    with pytest.raises(RuntimeClosureError, match="^runtime_closure_pre_close_required$"):
        witness.verify_closed()


async def test_replacing_memory_sdk_does_not_acknowledge_the_original(tmp_path, monkeypatch):
    async with application_graph(tmp_path, monkeypatch, Clock(time.time())) as bundle:
        witness = RuntimeClosureWitness.capture(bundle.graph)
        owner, original = witness._memory, witness._memory_client
        try:
            owner._client = object()
            with pytest.raises(RuntimeClosureError, match="^runtime_closure_sdk_binding_changed$"):
                await witness.pre_close()
        finally:
            owner._client = original
    assert bundle.sdk_clients and all(client.closed for client in bundle.sdk_clients)
    with pytest.raises(RuntimeClosureError, match="^runtime_closure_pre_close_required$"):
        witness.verify_closed()


async def test_native_success_requires_the_actual_terminal_close_receipt():
    model = ParakeetSTT()
    assert not runtime_closure._native_closed(model)
    await model.begin_close()
    assert runtime_closure._native_closed(model)
    # Fault injection only removes evidence, never manufactures native success.
    model._parakeet_close.result = ("synthetic_metadata_failure",)
    assert not runtime_closure._native_closed(model)
    assert not runtime_closure._native_closed(object())


async def test_real_native_release_failure_remains_unconfirmed(monkeypatch):
    model = ParakeetSTT()
    release = model._parakeet_release_resources

    def fail_after_release():
        release()
        raise RuntimeError("synthetic_release_failure")

    monkeypatch.setattr(model, "_parakeet_release_resources", fail_after_release)
    with pytest.raises(RuntimeError, match="^parakeet_close_failed$"):
        await model.begin_close()
    assert model._model is None
    assert model._parakeet_closing.done()
    assert not runtime_closure._native_closed(model)


async def test_trusted_alias_can_observe_existing_src_graph_without_patching_it(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("trusted_closure_bridge", runtime_closure.__file__)
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    async with application_graph(tmp_path, monkeypatch, Clock(time.time())) as bundle:
        ledger = bundle.graph.privacy.ledger
        witness = bridge.RuntimeClosureWitness.capture(bundle.graph)
        await witness.pre_close()
        assert bundle.graph.privacy.ledger is ledger and ledger.authority_available
    assert not ledger.authority_available
    proof = witness.verify_closed()
    assert all(proof[key] is True for key in ("bindings_unchanged", "model_owners_closed",
        "graph_closed", "privacy_closed", "sdk_closed", "native_closed"))


async def test_reviewed_lazy_wake_registration_is_included_before_close(tmp_path, monkeypatch):
    from tests.integration.test_wake_graph_retirement import install_sdk
    from tests.mocks.producer_erasure_graph import application

    sdk = install_sdk(monkeypatch)
    async with application(tmp_path, monkeypatch) as (graph, _, _):
        witness = RuntimeClosureWitness.capture(graph)
        manager = graph.fallback_audio_loop.audio_manager
        assert manager.wake_inventory()[2] is None
        try:
            await asyncio.to_thread(manager.load_wake_word)
            model = manager.wake_inventory()[2]
            assert model is not None and sdk.loads == 1
            assert graph.producer_erasure.inventory.verify().valid
            await witness.pre_close()
        finally:
            for gate in sdk.gates:
                gate.ctor_release.set()
                gate.predict_release.set()
                gate.reset_release.set()
    assert runtime_closure._native_closed(model)
    proof = witness.verify_closed()
    assert all(proof[key] is True for key in ("bindings_unchanged", "model_owners_closed",
        "graph_closed", "privacy_closed", "sdk_closed", "native_closed"))
