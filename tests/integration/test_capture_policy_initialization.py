"""Corrupt and late policy initialization stay sealed through real factory work."""
from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
import threading
import time
from uuid import uuid4

import pytest

from src.main import build_application
from src.memory.capture_control import CaptureControl
from src.memory.capture_recorders import CaptureRecorders
from src.pipeline.audio_loop import AudioLoop
from tests.integration.test_capability_application import configuration
from tests.integration.test_capture_producer_privacy import until
from tests.integration.test_global_capture_control import actor
from tests.integration.test_global_capture_http import application, leaves as capture_leaves
from tests.mocks.task_transport import TaskTransport

pytestmark = pytest.mark.asyncio


@pytest.fixture
def leaves(monkeypatch, tmp_path):
    return capture_leaves.__wrapped__(monkeypatch, tmp_path)


@pytest.mark.parametrize('corruption', ['missing_row', 'invalid_mode'])
async def test_established_corrupt_policy_fails_closed_without_repair(tmp_path, leaves, dashboard_auth, corruption):
    async with application(tmp_path, leaves) as first:
        assert first.capture_control.snapshot()['phase'] == 'running'

        def corrupt():
            with first.privacy.ledger._transaction() as db:
                if corruption == 'missing_row':
                    db.execute('DELETE FROM capture_runtime_control WHERE id=1')
                else:
                    # Deliberate fault injection, not a migration or repair.
                    db.execute('PRAGMA ignore_check_constraints=ON')
                    db.execute("UPDATE capture_runtime_control SET mode='synthetic-invalid' WHERE id=1")

        await first.privacy.acall(corrupt)
    async with application(tmp_path, leaves) as rebuilt:
        control = rebuilt.capture_control
        state = control.snapshot()
        assert state['phase'] == 'failed' and state['ingress_sealed']
        assert not state['total_mute_confirmed'] and not control.capture_allowed
        assert state['error_code'] == 'capture_control_state_invalid'
        assert rebuilt.pipeline.audio_loop._paused and rebuilt.pipeline.command_processor._paused
        assert not rebuilt._run_entered
        with pytest.raises(PermissionError):
            rebuilt.privacy.source_for_capture(str(uuid4()), time.time(), 'synthetic-corrupt')
        with pytest.raises(RuntimeError, match='capture_control_unavailable'):
            control.request(False, state['revision'], actor())
        with pytest.raises(RuntimeError, match='capture_control_state_invalid'):
            await rebuilt.privacy.acall(control._load_policy)


async def test_cancelled_factory_observer_keeps_late_sql_owned_during_privacy_close(tmp_path, leaves, dashboard_auth, monkeypatch):
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    controls, workers, starts = [], [], []
    load, start = CaptureControl._load_policy, AudioLoop.start

    def held(control):
        controls.append(control)
        workers.append(threading.current_thread())
        entered.set()
        assert release.wait(5), 'actual policy initializer was not released'
        try:
            return load(control)
        finally:
            returned.set()

    async def observe_start(loop):
        starts.append(loop)
        return await start(loop)

    monkeypatch.setattr(CaptureControl, '_load_policy', held)
    monkeypatch.setattr(AudioLoop, 'start', observe_start)
    stack = AsyncExitStack()
    building = closing = None
    try:
        async with TaskTransport().serve() as url:
            config, _ = configuration(tmp_path, url)
            building = asyncio.create_task(build_application(config, shutdown=stack, factories=leaves))
            try:
                await until(entered.is_set)
                control, = controls
                assert len(workers) == 1 and workers[0].ident is not None
                assert not returned.is_set() and not starts
                assert control.snapshot()['ingress_sealed'] and not control.capture_allowed
                assert not control.snapshot()['total_mute_confirmed']
                with pytest.raises(PermissionError):
                    control.privacy.source_for_capture(str(uuid4()), time.time(), 'synthetic-initializing')
                building.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await building
                closing = control.privacy.begin_close()
                assert not closing.done() and not returned.is_set()
                assert control.privacy.resource_status()['writer_lease']
            finally:
                release.set()
            async with asyncio.timeout(5):
                await asyncio.shield(closing)
            assert returned.is_set() and not starts
            assert control.snapshot()['ingress_sealed']
            assert not control.snapshot()['total_mute_confirmed']
            assert not control.privacy.resource_status()['writer_lease']
    finally:
        release.set()
        if building is not None:
            await asyncio.gather(building, return_exceptions=True)
        # Attempt every real callback even if an earlier close was unconfirmed.
        await asyncio.gather(stack.aclose(), return_exceptions=True)


async def test_cancelled_factory_observer_retains_actual_recorder_sql_return(tmp_path, leaves, dashboard_auth, monkeypatch):
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    controls, workers, starts = [], [], []
    load = CaptureControl._load_policy
    initialize, start = CaptureRecorders._initialize, AudioLoop.start

    def observe_control(control):
        controls.append(control)
        return load(control)

    def held(registry):
        result = initialize(registry)
        workers.append(threading.current_thread())
        entered.set()
        assert release.wait(5), 'actual recorder SQL initializer was not released'
        returned.set()
        return result

    async def observe_start(loop):
        starts.append(loop)
        return await start(loop)

    monkeypatch.setattr(CaptureControl, '_load_policy', observe_control)
    monkeypatch.setattr(CaptureRecorders, '_initialize', held)
    monkeypatch.setattr(AudioLoop, 'start', observe_start)
    stack = AsyncExitStack()
    building = closing = None
    try:
        async with TaskTransport().serve() as url:
            config, _ = configuration(tmp_path, url)
            building = asyncio.create_task(build_application(config, shutdown=stack, factories=leaves))
            try:
                await until(entered.is_set)
                control, = controls
                assert len(workers) == 1 and workers[0].ident is not None
                assert control.recorders._work.status()['operations'] > 0
                assert not returned.is_set() and not starts
                assert control.snapshot()['ingress_sealed'] and not control.capture_allowed
                assert not control.snapshot()['total_mute_confirmed']
                building.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await building
                closing = control.privacy.begin_close()
                assert not closing.done() and not returned.is_set()
                assert control.privacy.resource_status()['writer_lease']
            finally:
                release.set()
            async with asyncio.timeout(5):
                await asyncio.shield(closing)
            assert returned.is_set() and not starts
            assert control.recorders._work.status()['operations'] == 0
            assert control.snapshot()['ingress_sealed']
            assert not control.snapshot()['total_mute_confirmed']
            assert not control.privacy.resource_status()['writer_lease']
    finally:
        release.set()
        if building is not None:
            await asyncio.gather(building, return_exceptions=True)
        await asyncio.gather(stack.aclose(), return_exceptions=True)
