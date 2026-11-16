"""SC-4a: ApplicationGraph.run's verify_only branch and the VERIFY acknowledgement (R5-PRIV-2).

The acknowledgement is sent only from the verify branch, after it skipped
every ingress and house-effect start and before READY; a normal run never
sends it, even with a run_id attached. Fakes stand in for the graph's
services: the real graph is covered by tests/integration/test_sc4a_verify_boot.py.
"""
from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

import pytest

from src.application import ApplicationGraph
from src.core.process_lifecycle import ProcessLifecycle, ReadinessNotifier, StopPhase

pytestmark = pytest.mark.asyncio
RUN_ID = "9d" * 16


class Pipeline:
    def __init__(self, lifecycle, events):
        self.lifecycle, self.events = lifecycle, events

    async def prepare(self):
        self.events.append("prepare")

    async def run(self):
        self.events.append("pipeline.run")
        self.lifecycle.ready()
        await self.lifecycle.stop.wait()

    async def stop(self):
        self.events.append("pipeline.stop")


def _graph(events, lifecycle, *, verify_only):
    graph = ApplicationGraph(
        config={}, pipeline=Pipeline(lifecycle, events), ha_client=None, home_actions=None, privacy=None,
        voice_registry=None, voice_services=SimpleNamespace(), home_services=None, home_capabilities=None,
        list_manager=None, reminder_manager=None, routine_scheduler=None, dashboard=None,
        models=None, warmup_models=None, lifecycle=lifecycle, verify_only=verify_only, verify_run_id=RUN_ID)

    async def start_services():
        events.append("start_services")

    async def stop_services(dashboard_task, ambient_task):
        events.append("stop_services")

    graph._start_services = start_services
    graph._stop_services = stop_services
    async def ambient():
        events.append("ambient")

    graph.ambient_path = SimpleNamespace(start=ambient, stop=None)
    graph.loop_lag = SimpleNamespace(start=lambda: events.append("loop_lag"))
    return graph


async def _run(verify_only: bool) -> tuple[list, bytes]:
    read_fd, write_fd = os.pipe()
    try:
        lifecycle = ProcessLifecycle(notifier=ReadinessNotifier(write_fd))
        events = []
        graph = _graph(events, lifecycle, verify_only=verify_only)
        task = asyncio.create_task(graph.run())
        while lifecycle.stop.phase is not StopPhase.READY and not task.done():
            await asyncio.sleep(0.01)
        lifecycle.stop.request("test")
        await asyncio.wait_for(task, 5)
        return events, os.read(read_fd, 4096)
    finally:
        os.close(read_fd)
        os.close(write_fd)


async def test_verify_branch_prepares_acknowledges_then_ready_and_starts_nothing():
    events, lines = await _run(True)
    assert events == ["prepare", "stop_services"]
    assert lines.split(b"\n")[:2] == [b"VERIFY=" + RUN_ID.encode(), b"READY=1"]


async def test_a_normal_run_never_sends_verify_even_with_a_run_id():
    events, lines = await _run(False)
    assert events[:2] == ["loop_lag", "start_services"] and events[-1] == "stop_services"
    assert sorted(events[2:-1]) == ["ambient", "pipeline.run"]
    assert b"VERIFY" not in lines and b"READY=1" in lines


async def test_verify_branch_without_a_lifecycle_refuses():
    graph = _graph([], None, verify_only=True)
    graph.lifecycle = None
    with pytest.raises(RuntimeError, match="verify_requires_lifecycle"):
        await graph.run()


async def test_a_stop_before_the_verify_branch_sends_nothing():
    read_fd, write_fd = os.pipe()
    try:
        lifecycle = ProcessLifecycle(notifier=ReadinessNotifier(write_fd))
        lifecycle.stop.request("test")
        events = []
        await _graph(events, lifecycle, verify_only=True).run()
        assert events == ["stop_services"]
        assert os.read(read_fd, 4096) == b"STOPPING=1\n"  # no VERIFY, no READY
    finally:
        os.close(read_fd)
        os.close(write_fd)
