"""Truthful closure observations over real graph/native/SDK ownership."""
import asyncio
import importlib.util
import threading
import time

import pytest

from src.memory import runtime_closure
from src.memory.runtime_closure import RuntimeClosureError, RuntimeClosureWitness
from src.stt.parakeet_stt import ParakeetSTT
from tests.mocks.memory_graph import Clock
from tests.mocks.restore_application_graph import TOKEN, application_graph, until


async def test_real_graph_and_public_sdk_close_produce_content_free_proof(tmp_path, monkeypatch):
    async with application_graph(tmp_path, monkeypatch, Clock(time.time())) as bundle:
        witness = RuntimeClosureWitness.capture(bundle.graph)
        ha_session = bundle.graph.ha_client._session
        assert ha_session is not None and not ha_session.closed
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
    assert proof["sdk_owner_count"] == 3 and proof["input_owner_count"] >= 1
    assert proof["ha_session_acquired"] is True and proof["ha_never_acquired"] is False
    assert ha_session.closed and witness._ha.close_returned
    assert proof["authority_lease_acquired"] is True
    assert proof["capture_permissions_retired"] is True
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


async def test_capture_permission_waits_for_real_owner_return_and_public_retirement(tmp_path, monkeypatch):
    from src.core.owned_work import OwnedBlockingWork
    from src.memory.capture_control import CaptureController
    from tests.mocks.application_restore import root_lease

    entered, release = threading.Event(), threading.Event()
    work = OwnedBlockingWork(capacity=1)

    def native():
        entered.set()
        assert release.wait(5)
        return "actual callable returned"

    async with application_graph(tmp_path, monkeypatch, Clock(time.time())) as bundle:
        control = bundle.graph.capture_control
        actor = root_lease(bundle)
        actor = CaptureController.from_dashboard(actor.principal, actor.current)
        issued = await control.recorders.register(actor, kind="calibration")
        await control.recorders.activate(issued["token"], actor, kind="calibration")
        owner = asyncio.create_task(work.run(native))
        preparation = None
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            witness = RuntimeClosureWitness.capture(bundle.graph)
            preparation = asyncio.create_task(witness.pre_close())
            await until(lambda: control.recorders._sealed)
            assert control.recorders.pending(kind="calibration") == 1
            assert not preparation.done()
            assert bundle.graph.dashboard_authority.available
            assert bundle.privacy.resource_status()["writer_lease"]
            release.set()
            assert await owner == "actual callable returned"
            await work.aclose()
            assert await control.recorders.retire(issued["token"], kind="calibration") == {"retired": True}
            await asyncio.wait_for(asyncio.shield(preparation), 2)
        finally:
            release.set()
            await asyncio.gather(owner, return_exceptions=True)
            await work.aclose()
            await control.recorders.retire(issued["token"], kind="calibration")
            if preparation is not None:
                await asyncio.gather(preparation, return_exceptions=True)
    assert witness.verify_closed()["capture_permissions_retired"] is True


@pytest.mark.parametrize("state", ["active", "unknown"])
async def test_pending_remote_capture_cannot_become_a_universal_close_ack(tmp_path, monkeypatch, state):
    from src.core.owned_work import OwnedBlockingWork
    from src.memory.capture_control import CaptureController
    from src.memory.capture_recorders import CaptureRecorders
    from tests.mocks.application_restore import root_lease

    entered, release = threading.Event(), threading.Event()
    work = OwnedBlockingWork(capacity=1)

    def native():
        entered.set()
        assert release.wait(5)

    async with application_graph(tmp_path, monkeypatch, Clock(time.time())) as bundle:
        control = bundle.graph.capture_control
        actor = root_lease(bundle)
        actor = CaptureController.from_dashboard(actor.principal, actor.current)
        issued = await control.recorders.register(actor, kind="calibration")
        await control.recorders.activate(issued["token"], actor, kind="calibration")
        if state == "unknown":
            # Real local close/recovery preserves the remote permission.
            await control.recorders.aclose()
            recovered = CaptureRecorders(bundle.privacy, capture_allowed=lambda: control.capture_allowed)
            await recovered.initialize()
            control.recorders = recovered
            assert (await recovered.poll(issued["token"], actor, kind="calibration"))["state"] == "unknown"
        owner = asyncio.create_task(work.run(native))
        preparation = None
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            witness = RuntimeClosureWitness.capture(bundle.graph)
            preparation = asyncio.create_task(witness.pre_close())
            await until(lambda: control.recorders._sealed)
            with pytest.raises(TimeoutError):
                async with asyncio.timeout(.03):
                    await asyncio.shield(preparation)
            assert control.recorders.pending(kind="calibration") == 1
            assert not preparation.done()
            preparation.cancel()
            with pytest.raises(asyncio.CancelledError):
                await preparation
            with pytest.raises(RuntimeClosureError, match="^runtime_closure_pre_close_required$"):
                witness.verify_closed()
        finally:
            # Retirement is genuine cleanup after the actual callable returns;
            # it cannot repair this cancelled witness into a completion ACK.
            release.set()
            await asyncio.gather(owner, return_exceptions=True)
            await work.aclose()
            await control.recorders.retire(issued["token"], kind="calibration")
            if preparation is not None:
                await asyncio.gather(preparation, return_exceptions=True)
    with pytest.raises(RuntimeClosureError, match="^runtime_closure_pre_close_required$"):
        witness.verify_closed()


async def test_real_ha_session_closed_flag_does_not_replace_public_close_return(monkeypatch):
    from src.home_assistant.ha_client import HomeAssistantClient

    owner = HomeAssistantClient("http://127.0.0.1:1", "synthetic-only")
    witness = runtime_closure._HAClosure(owner)
    session = await owner._ensure_session()
    connector = session.connector
    actual_close = connector.close
    entered, release = asyncio.Event(), asyncio.Event()

    async def close_connector():
        await actual_close()
        entered.set()
        await release.wait()

    monkeypatch.setattr(connector, "close", close_connector)
    observer = asyncio.create_task(owner.close())
    try:
        await asyncio.wait_for(entered.wait(), 2)
        assert session.closed is True
        assert not witness.close_returned and not observer.done()
        with pytest.raises(RuntimeClosureError, match="^runtime_closure_ha_not_closed$"):
            witness.verify_closed()
        observer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await observer
        assert not owner._close_task.done()
    finally:
        release.set()
        await owner.close()
        await asyncio.gather(observer, return_exceptions=True)
    assert witness.close_returned and witness.verify_closed() is True


async def test_never_acquired_ha_is_sealed_without_inventing_a_session_close():
    from src.home_assistant.ha_client import HomeAssistantClient

    owner = HomeAssistantClient("http://127.0.0.1:1", "synthetic-only")
    witness = runtime_closure._HAClosure(owner)
    await owner.close()
    assert witness.verify_closed() is False
    assert owner._session is None and not witness.close_returned


async def test_real_loopback_ha_websocket_and_session_require_owned_retirement():
    from src.home_assistant.ha_client import HomeAssistantClient
    from tests.integration.test_ha_sdk_retirement import local_ha

    async with local_ha() as (url, connections):
        owner = HomeAssistantClient(url, "synthetic-only")
        witness = runtime_closure._HAClosure(owner)
        try:
            assert await owner.connect_websocket()
            assert connections and owner._owned_sockets and owner._sdk_retirements
        finally:
            await owner.close()
        assert witness.verify_closed() is True and witness.close_returned
        assert not owner._owned_sockets and not owner._sdk_retirements


async def test_borrowed_ha_session_is_not_acknowledged_as_owned():
    import aiohttp
    from src.home_assistant.ha_client import HomeAssistantClient

    async with aiohttp.ClientSession() as session:
        owner = HomeAssistantClient("http://127.0.0.1:1", "synthetic-only", session=session)
        try:
            with pytest.raises(RuntimeClosureError, match="^runtime_closure_ha_binding_changed$"):
                runtime_closure._HAClosure(owner)
        finally:
            await owner.close()
        assert session.closed is False


async def test_uncertain_ha_constructor_stays_unconfirmed_after_owner_close(monkeypatch):
    import aiohttp
    from src.home_assistant.ha_client import HomeAssistantClient

    owner = HomeAssistantClient("http://127.0.0.1:1", "synthetic-only")
    witness = runtime_closure._HAClosure(owner)
    actual_session = aiohttp.ClientSession
    acquired = []

    def uncertain(*args, **kwargs):
        acquired.append(actual_session(*args, **kwargs))
        raise RuntimeError("synthetic constructor did not return its handle")

    monkeypatch.setattr(aiohttp, "ClientSession", uncertain)
    try:
        with pytest.raises(RuntimeError, match="synthetic constructor did not return its handle"):
            await owner._ensure_session()
        await owner.close()
        assert owner._session is None and acquired and not acquired[0].closed
        with pytest.raises(RuntimeClosureError, match="^runtime_closure_ha_binding_changed$"):
            witness.verify_closed()
    finally:
        await owner.close()
        for session in acquired:
            await session.close()


async def test_replacing_real_ha_owner_cannot_borrow_original_close_receipt(tmp_path, monkeypatch):
    from src.home_assistant.ha_client import HomeAssistantClient

    async with application_graph(tmp_path, monkeypatch, Clock(time.time())) as bundle:
        original = bundle.graph.ha_client
        witness = RuntimeClosureWitness.capture(bundle.graph)
        replacement = HomeAssistantClient(original.url, "synthetic-only")
        try:
            await replacement._ensure_session()
            bundle.graph.ha_client = replacement
            with pytest.raises(RuntimeClosureError, match="^runtime_closure_binding_changed$"):
                await witness.pre_close()
        finally:
            bundle.graph.ha_client = original
            await replacement.close()


async def test_capture_registry_cannot_borrow_another_work_owners_close(tmp_path, monkeypatch):
    from src.core.owned_work import OwnedBlockingWork

    async with application_graph(tmp_path, monkeypatch, Clock(time.time())) as bundle:
        witness = RuntimeClosureWitness.capture(bundle.graph)
        recorders = bundle.graph.capture_control.recorders
        original = recorders._work
        replacement = OwnedBlockingWork(capacity=1)
        await replacement.aclose()
        try:
            recorders._work = replacement
            with pytest.raises(RuntimeClosureError, match="^runtime_closure_capture_binding_changed$"):
                await witness.pre_close()
        finally:
            recorders._work = original


async def test_authority_ack_waits_for_actual_native_return_and_writer_release(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    async with application_graph(tmp_path, monkeypatch, Clock(time.time())) as bundle:
        authority = bundle.graph.dashboard_authority
        witness = RuntimeClosureWitness.capture(bundle.graph)
        actual_transaction = authority._transaction

        def transaction(action, args, guard):
            if action == "login":
                entered.set()
                assert release.wait(5)
            return actual_transaction(action, args, guard)

        monkeypatch.setattr(authority, "_transaction", transaction)
        caller = asyncio.create_task(authority.login(TOKEN))
        observer = None
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            await witness.pre_close()
            observer = asyncio.create_task(authority.aclose())
            await until(lambda: authority._closing is not None)
            assert authority._lease_fd is not None and not authority._closing.done()
            with pytest.raises(RuntimeClosureError, match="^runtime_closure_authority_not_closed$"):
                witness._authority.verify_closed()
            observer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await observer
            assert authority._lease_fd is not None
        finally:
            release.set()
            await authority.aclose()
            await asyncio.gather(caller, return_exceptions=True)
            if observer is not None:
                await asyncio.gather(observer, return_exceptions=True)
    assert witness.verify_closed()["authority_lease_acquired"] is True


async def test_unreviewed_xvf_owner_does_not_receive_auxiliary_close_ack(tmp_path, monkeypatch):
    from src.audio.xvf_controller import XvfController

    async with application_graph(tmp_path, monkeypatch, Clock(time.time())) as bundle:
        loop = bundle.graph.pipeline.audio_loop
        controller = XvfController(usb_port="synthetic-never-opened")
        loop._xvf_by_room["synthetic-room"] = controller
        try:
            with pytest.raises(RuntimeClosureError, match="^runtime_closure_unsupported_auxiliary_native$"):
                RuntimeClosureWitness.capture(bundle.graph)
        finally:
            loop._xvf_by_room.pop("synthetic-room")
        assert controller._thread is None and controller._dev is None
