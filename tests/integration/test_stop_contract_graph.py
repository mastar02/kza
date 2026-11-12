"""U3b-0d1 in-process: the real graph, pipeline and dashboard honour one stop signal."""

import asyncio
from contextlib import AsyncExitStack
import os
import signal
import socket
import time
from types import SimpleNamespace

import pytest

from src.core.process_lifecycle import ProcessLifecycle, ReadinessNotifier, StopPhase, StopSignal
from src.dashboard import api as dashboard_api
from src.main import build_application
import src.main as main_module
from tests.integration.test_application_graph import leaves  # noqa: F401 - fixture
from tests.mocks.application_graph import offline_config, offline_home


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _hold_audio(graph, events):
    """Synthetic microphone leaf: the loop runs until it is cancelled."""
    loop = graph.pipeline.audio_loop

    async def start():
        events.append("audio_start")

    async def run():
        events.append("audio_run")
        try:
            await asyncio.Event().wait()
        finally:
            events.append("audio_cancelled")

    loop.start, loop.run = start, run


class _Pipe:
    def __init__(self):
        self.read, self.write = os.pipe()
        os.set_blocking(self.read, False)

    def lines(self):
        try:
            return os.read(self.read, 4096).decode().splitlines()
        except BlockingIOError:
            return []

    def close(self):
        os.close(self.read)
        os.close(self.write)


async def _until(predicate, timeout=5):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.005)


async def test_stop_before_run_skips_services_and_voice(leaves, tmp_path, monkeypatch):  # noqa: F811
    lifecycle = ProcessLifecycle()
    async with offline_home() as (url, observations):
        async with AsyncExitStack() as stack:
            graph = await build_application(offline_config(tmp_path, url), shutdown=stack,
                                            factories=leaves, lifecycle=lifecycle)
            assert graph.stop_signal is lifecycle.stop
            started = []
            monkeypatch.setattr(graph, "_start_services", lambda: started.append("services"))
            monkeypatch.setattr(graph.pipeline, "run", lambda: started.append("pipeline"))
            lifecycle.stop.request()
            await asyncio.wait_for(graph.run(), 10)
            assert started == []
        assert graph.pipeline.request_router._closed
        assert not graph.privacy.resource_status()["writer_lease"]
        assert observations["service_calls"] == 0


async def test_steady_state_stop_cancels_audio_and_ready_follows_system_ready(
        leaves, tmp_path, caplog):  # noqa: F811
    pipe = _Pipe()
    lifecycle = ProcessLifecycle(StopSignal(), ReadinessNotifier(pipe.write))
    events = []
    try:
        async with offline_home() as (url, _):
            async with AsyncExitStack() as stack:
                config = offline_config(tmp_path, url)
                config["dashboard"]["enabled"] = False
                graph = await build_application(config, shutdown=stack, factories=leaves, lifecycle=lifecycle)
                _hold_audio(graph, events)
                original = graph.pipeline.command_processor.aload_models

                async def load():
                    assert pipe.lines() == []
                    await original()
                    events.append("models_loaded")

                graph.pipeline.command_processor.aload_models = load
                caplog.set_level("INFO")
                running = asyncio.create_task(graph.run())
                await _until(lambda: "audio_run" in events)
                assert pipe.lines() == ["READY=1"]
                assert lifecycle.stop.phase is StopPhase.READY
                assert "System ready" in caplog.text
                assert events.index("models_loaded") < events.index("audio_start")
                lifecycle.stop.request()
                lifecycle.stop.request()
                await asyncio.wait_for(running, 10)
                assert events[-1] == "audio_cancelled"
                assert pipe.lines() == ["STOPPING=1"]
                assert lifecycle.stop.requested_phase is StopPhase.READY
    finally:
        pipe.close()


@pytest.mark.parametrize("step", ["models", "features"])
async def test_stop_during_pipeline_start_up_returns_before_ready(leaves, tmp_path, step):  # noqa: F811
    pipe = _Pipe()
    lifecycle = ProcessLifecycle(StopSignal(), ReadinessNotifier(pipe.write))
    events = []
    try:
        async with offline_home() as (url, _):
            async with AsyncExitStack() as stack:
                config = offline_config(tmp_path, url)
                config["dashboard"]["enabled"] = False
                graph = await build_application(config, shutdown=stack, factories=leaves, lifecycle=lifecycle)
                _hold_audio(graph, events)
                owner = graph.pipeline.command_processor if step == "models" else graph.pipeline.features
                name = "aload_models" if step == "models" else "start"
                original = getattr(owner, name)

                async def stop_inside(*args, **kwargs):
                    lifecycle.stop.request()
                    return await original(*args, **kwargs)

                setattr(owner, name, stop_inside)
                await asyncio.wait_for(graph.run(), 10)
                assert events == [] and not graph.pipeline._running
                assert pipe.lines() == ["STOPPING=1"]
                assert lifecycle.stop.requested_phase is StopPhase.PIPELINE
    finally:
        pipe.close()


async def test_blocked_capture_is_ready_with_status_and_waits_on_the_stop(
        leaves, tmp_path, monkeypatch):  # noqa: F811
    pipe = _Pipe()
    lifecycle = ProcessLifecycle(StopSignal(), ReadinessNotifier(pipe.write))
    started = []
    async with offline_home() as (url, _):
        async with AsyncExitStack() as stack:
            config = offline_config(tmp_path, url)
            config["dashboard"]["enabled"] = False
            graph = await build_application(config, shutdown=stack, factories=leaves, lifecycle=lifecycle)
            graph.capture_control = SimpleNamespace(snapshot=lambda: {
                "ingress_sealed": True, "supported": False, "error_code": None})
            ran = []
            monkeypatch.setattr(graph.pipeline, "run", lambda: ran.append(1))
            original = graph._start_services

            async def services():
                await original()
                started.append("services")

            monkeypatch.setattr(graph, "_start_services", services)
            try:
                running = asyncio.create_task(graph.run())
                await _until(lambda: lifecycle.stop.phase is StopPhase.CAPTURE_BLOCKED)
                await asyncio.sleep(0.05)
                assert not running.done() and started == ["services"]
                # Serving degraded: READY once, with the capture-blocked STATUS.
                assert pipe.lines() == ["READY=1", "STATUS=capture blocked"]
                assert lifecycle.notifier.sent == ("READY=1", "STATUS=capture blocked")
                lifecycle.stop.request()
                await asyncio.wait_for(running, 10)
                assert ran == [] and pipe.lines() == ["STOPPING=1"]
                assert lifecycle.stop.requested_phase is StopPhase.CAPTURE_BLOCKED
            finally:
                pipe.close()


async def test_main_in_process_returns_without_os_exit(leaves, tmp_path, monkeypatch):  # noqa: F811
    lifecycle = ProcessLifecycle()
    exits = []
    monkeypatch.setattr(main_module.os, "_exit", lambda code: exits.append(code))
    async with offline_home() as (url, _):
        config = offline_config(tmp_path, url)
        config["dashboard"]["enabled"] = False

        async def build(shutdown, current):
            graph = await build_application(config, shutdown=shutdown, factories=leaves, lifecycle=current)
            graph.pipeline.run = _stopping_run(current)
            return graph

        await asyncio.wait_for(main_module.main(lifecycle=lifecycle, build=build), 20)
        # Managed exit without a receipt (no admission) also returns normally.
        await asyncio.wait_for(main_module.main(managed_exit=True, lifecycle=ProcessLifecycle(),
                                                build=build), 20)
    assert exits == []


def _stopping_run(lifecycle):
    async def run():
        lifecycle.stop.request()
    return run


@pytest.mark.parametrize("miswiring", ["unwired", "graph", "pipeline"])
async def test_main_refuses_a_builder_that_does_not_wire_the_stop(leaves, tmp_path, miswiring):  # noqa: F811
    async with offline_home() as (url, _):
        async def build(shutdown, current):
            if miswiring == "unwired":
                return await build_application(offline_config(tmp_path, url), shutdown=shutdown,
                                               factories=leaves)
            graph = await build_application(offline_config(tmp_path, url), shutdown=shutdown,
                                            factories=leaves, lifecycle=current)
            if miswiring == "graph":
                # The pipeline is wired but the graph waits on another stop.
                graph.lifecycle = ProcessLifecycle()
            else:
                # The graph is wired but the pipeline it runs ignores stops.
                graph.pipeline.lifecycle = ProcessLifecycle()
            return graph

        with pytest.raises(RuntimeError, match="main_stop_signal_not_wired"):
            await main_module.main(lifecycle=ProcessLifecycle(), build=build)


async def test_stop_between_ready_and_audio_loop_run_closes_normally(leaves, tmp_path):  # noqa: F811
    pipe = _Pipe()
    lifecycle = ProcessLifecycle(StopSignal(), ReadinessNotifier(pipe.write))
    events = []
    try:
        async with offline_home() as (url, _):
            async with AsyncExitStack() as stack:
                config = offline_config(tmp_path, url)
                config["dashboard"]["enabled"] = False
                graph = await build_application(config, shutdown=stack, factories=leaves, lifecycle=lifecycle)
                _hold_audio(graph, events)
                loop = graph.pipeline.audio_loop

                async def start():
                    # The stop lands after READY, inside audio_loop.start().
                    events.append("audio_start")
                    lifecycle.stop.request()

                loop.start = start
                await asyncio.wait_for(graph.run(), 10)
                assert events == ["audio_start"]  # run() was never started
                assert pipe.lines() == ["READY=1", "STOPPING=1"]
                assert lifecycle.stop.requested_phase is StopPhase.READY
            assert graph.pipeline.request_router._closed
            assert graph.pipeline.audio_loop._closed
            assert not graph.privacy.resource_status()["writer_lease"]
    finally:
        pipe.close()


def test_dashboard_server_neutralizes_uvicorn_signal_capture():
    import uvicorn
    from uvicorn.server import Server

    assert uvicorn.__version__ == "0.49.0"
    assert callable(getattr(Server, "capture_signals", None))
    server_class = dashboard_api._owned_signal_server(uvicorn)
    assert issubclass(server_class, Server) and server_class is dashboard_api._owned_signal_server(uvicorn)
    server = server_class(uvicorn.Config(lambda scope, receive, send: None))
    before = signal.getsignal(signal.SIGTERM)
    with server.capture_signals():
        assert signal.getsignal(signal.SIGTERM) is before
    assert signal.getsignal(signal.SIGTERM) is before and server._captured_signals == []


async def test_dashboard_stop_bounds_and_awaits_an_open_websocket(leaves, tmp_path, dashboard_auth):  # noqa: F811
    import websockets

    port = _free_port()
    marks = []
    async with offline_home() as (url, _):
        async with AsyncExitStack() as stack:
            config = offline_config(tmp_path, url)
            config["dashboard"].update(host="127.0.0.1", port=port)
            graph = await build_application(config, shutdown=stack, factories=leaves)
            dashboard = graph.dashboard

            async def hold(websocket):
                await websocket.accept()
                marks.append("accepted")
                try:
                    await asyncio.Event().wait()
                finally:
                    await asyncio.sleep(0.3)
                    marks.append("handler_finally")

            from starlette.routing import WebSocketRoute
            # Ahead of the static mount, which would otherwise match first.
            dashboard.app.router.routes.insert(0, WebSocketRoute("/test-hold", hold))
            before = signal.getsignal(signal.SIGTERM)
            serving = asyncio.create_task(dashboard.start())
            await _until(lambda: dashboard._server is not None and dashboard._server.started, 10)
            assert signal.getsignal(signal.SIGTERM) is before
            client = await websockets.connect(f"ws://127.0.0.1:{port}/test-hold")
            try:
                await _until(lambda: marks == ["accepted"])
                started = time.monotonic()
                await asyncio.wait_for(dashboard.stop(), 15)
                elapsed = time.monotonic() - started
                assert marks == ["accepted", "handler_finally"]
                assert elapsed < dashboard_api.GRACEFUL_SHUTDOWN_SECONDS + dashboard_api.CONNECTION_DRAIN_SECONDS + 2
                await asyncio.wait_for(serving, 5)
                assert signal.getsignal(signal.SIGTERM) is before
            finally:
                await client.close()
