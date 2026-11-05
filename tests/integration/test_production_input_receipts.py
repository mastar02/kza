"""Checked input retirement through real loops, SQLite and capture control."""
from __future__ import annotations

import asyncio
import json
import os

import pytest

from tests.integration.test_global_capture_control import actor, capture_phase
from tests.mocks.production_input import NativeInputs, production_graph, run_isolated, until


TRANSPORTS = ["single", "multi"]


def strict_once(stream):
    assert stream.calls == [("stop", False), ("close", False)]


async def cancel_observer(task):
    async def observe():
        await asyncio.shield(task)

    observer = asyncio.create_task(observe())
    await asyncio.sleep(0)
    observer.cancel()
    await asyncio.gather(observer, return_exceptions=True)


@pytest.mark.parametrize("transport", TRANSPORTS)
@pytest.mark.asyncio
async def test_strict_retirement_and_authenticated_resume_use_fresh_native_handles(tmp_path, monkeypatch, transport):
    backend = NativeInputs()
    async with production_graph(tmp_path, monkeypatch, transport, backend) as g:
        assert await g.open() is not None
        owner, stream = g.owner(), backend.streams[0]
        await until(stream.callback_returned.is_set)
        assert stream.worker.is_alive() and not stream.worker._handle.is_done()
        assert g.control.snapshot()["supported"]
        g.control.request(True, g.control.snapshot()["revision"], actor())
        state = await capture_phase(g.control, "muted")
        await asyncio.shield(g.control._operation)
        state = g.control.snapshot()
        strict_once(stream)
        assert stream.worker._handle.is_done() and stream.native_returned.is_set()
        assert owner.retired and not owner.retirement_unknown
        assert owner.resource_status()["retirement_confirmed"]
        assert state["total_mute_confirmed"] and state["pending_operations"] == 0
        closing = owner.begin_close()
        assert owner.begin_close() is closing
        await asyncio.shield(closing)
        strict_once(stream)
        g.control.request(False, state["revision"], actor())
        await capture_phase(g.control, "running")
        assert await g.open() is not None
        assert g.owner() is not owner
        assert len(backend.streams) == 2 and backend.streams[1] is not stream
        assert backend.streams[1].worker.is_alive()
        assert stream.worker._handle.is_done()


@pytest.mark.parametrize("transport", TRANSPORTS)
@pytest.mark.parametrize("mode", ["stop-error", "stop-control"])
def test_stop_error_still_closes_once_but_does_not_silently_ack_or_resume(tmp_path, transport, mode):
    receipt = run_isolated(transport, mode, tmp_path)
    assert receipt["retired"] and not receipt["retirement_unknown"]
    assert receipt["close_attempts"] == 1 and receipt["native_exits"] == 1
    assert receipt["resume_rejected"] and receipt["writer_lease"]
    if mode == "stop-control":
        assert any(kind == "interrupt" for kind, code in receipt["failures"])


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_pointer_null_after_strict_close_error_is_not_a_retry_or_retirement_receipt(tmp_path, transport):
    receipt = run_isolated(transport, "close-error", tmp_path)
    assert receipt["pointer_null"] and receipt["actor_alive_before_fixture_join"]
    assert receipt["retirement_unknown"] and not receipt["retired"]
    assert receipt["close_attempts"] == 1 and receipt["native_exits"] == 1
    assert receipt["resume_rejected"] and receipt["writer_lease"]


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_unreturned_constructor_remains_unknown_after_native_future_terminal(tmp_path, transport):
    receipt = run_isolated(transport, "unreturned-constructor", tmp_path)
    assert receipt["actor_alive_before_fixture_join"]
    assert receipt["retirement_unknown"] and not receipt["retired"]
    assert receipt["close_attempts"] == 0 and receipt["native_exits"] == 1
    assert receipt["resume_rejected"] and receipt["writer_lease"]


@pytest.mark.parametrize("transport", TRANSPORTS)
@pytest.mark.asyncio
async def test_late_constructor_return_after_observer_cancel_is_retired_without_start(tmp_path, monkeypatch, transport):
    backend = NativeInputs(hold={"constructor"})
    async with production_graph(tmp_path, monkeypatch, transport, backend) as g:
        opening = asyncio.create_task(g.open())
        try:
            await until(backend.event("constructor").is_set)
            owner = g.owner()
            opening.cancel()
            await asyncio.gather(opening, return_exceptions=True)
            g.control.request(True, g.control.snapshot()["revision"], actor())
            closing = owner.begin_close()
            assert not closing.done()
            assert g.control.snapshot()["ingress_sealed"]
            assert not g.control.snapshot()["total_mute_confirmed"]
            assert owner.snapshot_operations()
        finally:
            backend.release_event("constructor").set()
            await asyncio.gather(opening, return_exceptions=True)
        state = await capture_phase(g.control, "muted")
        stream = backend.streams[0]
        assert stream.worker is None, "late sealed construction must not start a native actor"
        strict_once(stream)
        assert owner.retired and not owner.retirement_unknown
        assert state["total_mute_confirmed"]
        assert len(backend.streams) == 1


@pytest.mark.parametrize("transport", TRANSPORTS)
@pytest.mark.asyncio
async def test_delegated_start_failure_retains_known_stream_through_actual_close(tmp_path, monkeypatch, transport):
    backend = NativeInputs("delegated-start-error", hold={"start"})
    async with production_graph(tmp_path, monkeypatch, transport, backend) as g:
        opening = asyncio.create_task(g.open())
        try:
            await until(backend.event("start").is_set)
            stream, owner = backend.streams[0], g.owner()
            assert stream.worker.is_alive() and not stream.worker._handle.is_done()
            opening.cancel()
            await asyncio.gather(opening, return_exceptions=True)
            g.control.request(True, g.control.snapshot()["revision"], actor())
            assert not g.control.snapshot()["total_mute_confirmed"]
            assert not owner.begin_close().done()
        finally:
            backend.release_event("start").set()
            await asyncio.gather(opening, return_exceptions=True)
        state = await capture_phase(g.control, "muted")
        strict_once(stream)
        assert stream.worker._handle.is_done() and stream.native_returned.is_set()
        assert owner.retired and not owner.retirement_unknown
        assert state["total_mute_confirmed"] and len(backend.streams) == 1


async def _legacy_child(transport, tmp_path, patch):
    if transport == "single":
        from tests.integration.test_single_input_privacy import graph
        context = graph(tmp_path, patch)
    else:
        from tests.integration.test_capture_producer_privacy import graph
        from tests.integration.test_input_driver_pause import Driver, portaudio
        driver = Driver(fail="close")
        portaudio(patch, [driver])
        context = graph(tmp_path, controllers=False)
    async with context as g:
        if transport == "single":
            g.backend.close_failures = 2
            owner = g.loop._input
            closing = g.loop.begin_close()
        else:
            assert await g.loop._try_reopen_once(g.loop.room_streams["synthetic-a"])
            owner = g.loop._inputs._owners["synthetic-a"]
            closing = g.loop.begin_pause_inputs()
        results = await asyncio.gather(asyncio.shield(closing), return_exceptions=True)
        assert isinstance(results[0], BaseException)
        assert owner.retirement_unknown and not owner.retired
        assert owner.begin_close().done()
        assert g.privacy.resource_status()["writer_lease"]
        if transport == "single":
            attempts = g.gates["close"].calls
            assert attempts == 1 and g.gates["constructor"].calls == 1
            assert not g.backend.stream.closed
            assert g.loop.begin_close() is closing
            resume = g.loop.resume_admission
        else:
            attempts = driver.close_attempts
            assert attempts == 1 and driver.closed == 0
            assert g.loop.begin_pause_inputs() is closing
            resume = g.loop.resume_inputs
        with pytest.raises(RuntimeError):
            resume()
        privacy_failed = False
        try:
            await g.privacy.aclose()
        except BaseException:
            privacy_failed = True
        assert privacy_failed and g.privacy.resource_status()["writer_lease"]
        # Do not leave an async context that would reinterpret failed permanent
        # retirement as successful cleanup. The process boundary contains it.
        receipt = {"production_input_receipt": True, "transport": transport,
            "mode": "legacy-close", "valid": True, "ack": None,
            "ack_scope": "not_measured_legacy_loop_only",
            "close_attempts": attempts, "retirement_unknown": True, "retired": False,
            "writer_lease": True, "resume_rejected": True, "constructor_calls": 1}
        print(json.dumps(receipt), flush=True)
        os._exit(0)


async def _isolated_body(transport, mode, tmp_path):
    import httpx

    patch = pytest.MonkeyPatch()

    async def denied(self, request):
        raise AssertionError("Tests must inject a synthetic HTTPX transport")

    patch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", denied)
    if mode == "legacy-close":
        await _legacy_child(transport, tmp_path, patch)
        raise AssertionError("legacy child returned without its immutable receipt")
    backend = NativeInputs(mode, hold={"close"} if mode.startswith("stop-") else set())
    async with production_graph(tmp_path, patch, transport, backend) as g:
        opening = asyncio.create_task(g.open())
        try:
            if mode == "unreturned-constructor":
                await until(lambda: bool(backend.streams) and backend.streams[0].native_entered.is_set())
                await asyncio.gather(opening, return_exceptions=True)
            else:
                assert await opening is not None
            owner, stream = g.owner(), backend.streams[0]
            g.control.request(True, g.control.snapshot()["revision"], actor())
            closing = owner.begin_close()
            assert owner.begin_close() is closing
            if mode.startswith("stop-"):
                await until(backend.event("close").is_set)
                await cancel_observer(closing)
                assert not closing.done() and stream.worker.is_alive()
                assert not g.control.snapshot()["total_mute_confirmed"]
                backend.release_event("close").set()
            await capture_phase(g.control, "failed")
            await asyncio.shield(g.control._operation)
            results = await asyncio.gather(asyncio.shield(closing), return_exceptions=True)
            failure = results[0]
            assert isinstance(failure, RuntimeError) and isinstance(failure.failures, tuple)
            assert failure.__cause__ is None and failure.__context__ is None
            if mode == "stop-control":
                assert any(kind == "interrupt" for kind, code in failure.failures)
            status = owner.resource_status()
            assert status["failure_count"] >= 1
            assert not g.control.snapshot()["total_mute_confirmed"]
            assert g.control.snapshot()["ingress_sealed"]
            await until(lambda: not owner.snapshot_operations())
            if mode.startswith("stop-"):
                assert owner.retired and not owner.retirement_unknown
                assert stream.worker._handle.is_done()
            else:
                assert not owner.retired and owner.retirement_unknown
                assert stream.worker.is_alive() and not stream.worker._handle.is_done()
            alive_before = stream.worker.is_alive()
            attempts = sum(method == "close" for method, strict in stream.calls)
            expected = 0 if mode == "unreturned-constructor" else 1
            assert attempts == expected
            # The real authenticated global resume runs and must fail safely;
            # zero operations or a null public SDK pointer cannot replace owner.
            g.control.request(False, g.control.snapshot()["revision"], actor())
            transition = g.control._operation
            await asyncio.shield(transition)
            assert g.control.snapshot()["phase"] == "failed"
            assert g.control.snapshot()["ingress_sealed"]
            assert not g.control.snapshot()["total_mute_confirmed"]
            assert g.owner() is owner and len(backend.streams) == 1
            assert owner.begin_close() is closing
            assert sum(method == "close" for method, strict in stream.calls) == attempts
            policy, _revision = await g.privacy.acall(g.control._load_policy)
            assert policy == "muted"
            assert g.privacy.resource_status()["writer_lease"]
        finally:
            await asyncio.to_thread(backend.release)
            await asyncio.gather(opening, return_exceptions=True)
        assert stream.worker._handle.is_done() and stream.native_returned.is_set()
        assert not g.control.snapshot()["total_mute_confirmed"]
        assert owner.retired == mode.startswith("stop-")
        if owner.retirement_unknown:
            privacy_failed = False
            try:
                await g.privacy.aclose()
            except BaseException:
                privacy_failed = True
            assert privacy_failed and g.privacy.resource_status()["writer_lease"]
        receipt = {"production_input_receipt": True, "transport": transport, "mode": mode,
            "valid": True, "ack": g.control.snapshot()["total_mute_confirmed"], "retired": owner.retired,
            "retirement_unknown": owner.retirement_unknown, "close_attempts": attempts,
            "native_exits": 1, "actor_alive_before_fixture_join": alive_before,
            "pointer_null": stream.pointer is None, "resume_rejected": True,
            "writer_lease": True, "failures": list(failure.failures)}
        print(json.dumps(receipt), flush=True)
        os._exit(0)
