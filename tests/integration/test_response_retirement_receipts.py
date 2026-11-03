"""Actual Response owners with synthetic synthesis/output leaves only."""

import asyncio
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from src.core.voice_delivery import DeliveryState
from src.hooks import HookRegistry, TtsPayload, execute_after_event
from src.pipeline.response_handler import ResponseCloseTimeout
from tests.mocks.voice_graph import voice_graph


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(.002)


async def settle(owner, *tasks):
    """Release gates before calling this; never cancel the actual close owner."""
    await asyncio.gather(*tasks, return_exceptions=True)
    async with asyncio.timeout(3):
        await asyncio.shield(owner.begin_close())


@pytest.mark.asyncio
async def test_owned_after_tts_callback_drains_without_waiting_for_registry_peer(monkeypatch):
    registry = HookRegistry()
    own_entered, own_release = asyncio.Event(), asyncio.Event()
    peer_entered, peer_release = asyncio.Event(), asyncio.Event()

    async def after_tts(payload):
        if payload.text == 'synthetic peer':
            peer_entered.set()
            await peer_release.wait()
        else:
            own_entered.set()
            await own_release.wait()

    registry.register_after('tts', after_tts)
    execute_after_event(registry, 'tts', TtsPayload(0, 'synthetic peer', None, 0, True))
    await until(peer_entered.is_set)
    peers = set(registry._after_tasks)
    assert len(peers) == 1
    owner, _, backend, _, _ = voice_graph(monkeypatch, hooks=registry, shutdown_timeout_s=.02)
    speech = asyncio.create_task(owner.aspeak('synthetic owned', zone_id='zone_origin'))
    try:
        await until(own_entered.is_set)
        assert (await speech).delivery is DeliveryState.COMPLETE
        assert owner.get_delivery_status()['hook_tasks'] == 1
        closing = owner.begin_close()
        with pytest.raises(ResponseCloseTimeout):
            await owner.aclose()
        assert owner.begin_close() is closing and not closing.done()
        assert all(not task.done() for task in peers)
        own_release.set()
        await owner.wait_closed()
        assert closing.done() and closing.exception() is None
        assert owner.get_delivery_status()['hook_tasks'] == 0
        assert all(not task.done() and not task.cancelled() for task in peers)
        assert backend.outputs and all(output.closed for output in backend.outputs)
    finally:
        own_release.set()
        peer_release.set()
        await settle(owner, speech, *peers, *tuple(registry._after_tasks))


@pytest.mark.asyncio
async def test_timeout_and_cancelled_waiters_keep_same_actual_retirement(monkeypatch):
    owner, engine, backend, _, _ = voice_graph(monkeypatch, shutdown_timeout_s=.02)
    engine.release.clear()
    synthesis = asyncio.create_task(owner.asynthesize('synthetic cache'))
    waiter = None
    entered = asyncio.Event()

    async def wait_for_actual_exit():
        entered.set()
        await owner.wait_closed()

    try:
        await until(engine.entered.is_set)
        closing = owner.begin_close()
        with pytest.raises(ResponseCloseTimeout):
            await owner.aclose()
        assert owner.begin_close() is closing and not closing.done()
        waiter = asyncio.create_task(wait_for_actual_exit())
        await entered.wait()
        waiter.cancel()
        await asyncio.sleep(0)
        waiter.cancel()
        await asyncio.sleep(0)
        assert not waiter.done() and not closing.done()
        assert owner.get_delivery_status()['native_calls'] == 1
        engine.release.set()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        with pytest.raises(asyncio.CancelledError):
            await synthesis
        await owner.wait_closed()
        assert owner.begin_close() is closing and closing.exception() is None
        assert not backend.outputs
        with pytest.raises(RuntimeError, match='voice_delivery_closed'):
            await owner.asynthesize('new admission')
    finally:
        engine.release.set()
        await settle(owner, synthesis, *([waiter] if waiter is not None else []))


def mark_delivery_worker(monkeypatch, owner):
    marker = threading.local()
    worker = owner._worker

    def marked(session):
        marker.in_delivery = True
        try:
            return worker(session)
        finally:
            marker.in_delivery = False

    monkeypatch.setattr(owner, '_worker', marked)
    return marker


@pytest.mark.asyncio
async def test_abort_postlude_outlives_closed_output_and_finished_worker(monkeypatch):
    owner, _, backend, _, _ = voice_graph(monkeypatch, shutdown_timeout_s=.02)
    backend.stall = True
    marker = mark_delivery_worker(monkeypatch, owner)
    entered, release = threading.Event(), threading.Event()
    speech = asyncio.create_task(owner.aspeak('synthetic interrupted', zone_id='zone_origin'))
    try:
        await until(backend.created.is_set)
        output = backend.outputs[0]
        stop = output.owner.stop

        def held_stop():
            stop()
            if not getattr(marker, 'in_delivery', False):
                entered.set()
                assert release.wait(5), 'synthetic abort postlude was not released'

        monkeypatch.setattr(output.owner, 'stop', held_stop)
        closing = owner.begin_close()
        await until(entered.is_set)
        await until(lambda: owner.get_delivery_status()['workers'] == 0)
        assert output.closed and output.finished.is_set()
        assert owner.get_delivery_status()['aborts'] == 1
        assert not closing.done()
        with pytest.raises(ResponseCloseTimeout):
            await owner.aclose()
        assert owner.begin_close() is closing and not closing.cancelled()
        release.set()
        await owner.wait_closed()
        assert (await speech).delivery is DeliveryState.CANCELLED
        assert owner.get_delivery_status()['aborts'] == 0
    finally:
        release.set()
        await settle(owner, speech)


@pytest.mark.asyncio
async def test_sync_speak_controller_retains_worker_postlude_after_worker_telemetry(monkeypatch):
    owner, _, backend, _, _ = voice_graph(monkeypatch, shutdown_timeout_s=.02)
    entered, release = threading.Event(), threading.Event()
    finished = owner._worker_finished

    def held_finished(session):
        finished(session)
        entered.set()
        assert release.wait(5), 'synthetic sync speech postlude was not released'

    monkeypatch.setattr(owner, '_worker_finished', held_finished)
    speech = asyncio.create_task(asyncio.to_thread(owner.speak, 'synthetic sync', zone_id='zone_origin'))
    try:
        await until(entered.is_set)
        assert owner.get_delivery_status()['workers'] == 0
        assert owner.get_delivery_status()['controllers'] == 1
        closing = owner.begin_close()
        with pytest.raises(ResponseCloseTimeout):
            await owner.aclose()
        assert not closing.done() and not speech.done()
        release.set()
        await speech
        await owner.wait_closed()
        assert owner.begin_close() is closing and closing.exception() is None
        assert owner.get_delivery_status()['controllers'] == 0
        assert backend.outputs and all(output.closed for output in backend.outputs)
    finally:
        release.set()
        await settle(owner, speech)


@pytest.mark.asyncio
async def test_failed_abort_is_terminal_same_owner_after_physical_actors_exit(monkeypatch):
    owner, _, backend, _, _ = voice_graph(monkeypatch, shutdown_timeout_s=.02)
    backend.stall = True
    marker = mark_delivery_worker(monkeypatch, owner)
    speech = asyncio.create_task(owner.aspeak('synthetic failed abort', zone_id='zone_origin'))
    try:
        await until(backend.created.is_set)
        output = backend.outputs[0]
        stop = output.owner.stop

        def failed_stop():
            stop()
            if not getattr(marker, 'in_delivery', False):
                raise RuntimeError('synthetic_abort_postlude_failure')

        monkeypatch.setattr(output.owner, 'stop', failed_stop)
        closing = owner.begin_close()
        async with asyncio.timeout(3):
            with pytest.raises(RuntimeError, match='^voice_retirement_unconfirmed$'):
                await owner.wait_closed()
        assert output.closed and output.finished.is_set()
        assert output.thread is not None and not output.thread.is_alive()
        status = owner.get_delivery_status()
        assert status['failed_barriers'] > 0
        assert not any(status[key] for key in (
            'pending', 'controllers', 'workers', 'generations', 'blocked_generations',
            'aborts', 'native_calls', 'native_operations', 'native_retirements',
            'callbacks', 'hook_tasks',
        ))
        assert owner.begin_close() is closing and closing.done() and not closing.cancelled()
        with pytest.raises(RuntimeError, match='^voice_retirement_unconfirmed$'):
            await owner.aclose()
        assert owner.begin_close() is closing
    finally:
        await asyncio.gather(speech, return_exceptions=True)
        async with asyncio.timeout(3):
            await asyncio.gather(asyncio.shield(owner.begin_close()), return_exceptions=True)


async def failure_child(path):
    """Run only in the isolated child: a failed lease intentionally remains held."""
    from types import SimpleNamespace
    from weakref import ref

    import numpy as np

    from src.application import TTSModelOwner
    from src.memory.privacy_services import PrivacyServices
    from src.tts.piper_tts import KokoroTTS

    patch = pytest.MonkeyPatch()

    class Pipeline:
        def __init__(self, **kwargs):
            pass

        def __call__(self, text, *, voice):
            yield SimpleNamespace(audio=np.full(8192, .125, dtype=np.float32))

    patch.setitem(sys.modules, 'kokoro', SimpleNamespace(KPipeline=Pipeline))
    model = KokoroTTS(model='synthetic-only', device='cpu')
    model.load()
    model_ref = ref(model._model)
    owner, _, backend, _, _ = voice_graph(patch, shutdown_timeout_s=.02)
    owner.tts = model
    backend.stall = True
    marker = mark_delivery_worker(patch, owner)
    models = TTSModelOwner()
    models.add_model(model)
    models.add_borrower(owner.begin_close)
    privacy = PrivacyServices(str(path), close_timeout_s=.02)
    privacy.initialize()
    privacy.add_borrower(models.begin_close)
    speech = asyncio.create_task(owner.aspeak('synthetic failure', zone_id='zone_origin'))
    try:
        await until(backend.created.is_set)
        output = backend.outputs[0]
        stop = output.owner.stop

        def failed_stop():
            stop()
            if not getattr(marker, 'in_delivery', False):
                raise RuntimeError('synthetic_abort_postlude_failure')

        patch.setattr(output.owner, 'stop', failed_stop)
        closing = privacy.begin_close()
        async with asyncio.timeout(3):
            with pytest.raises(RuntimeError, match='^voice_retirement_unconfirmed$'):
                await asyncio.shield(closing)
        assert closing.done() and privacy.begin_close() is closing
        assert models.begin_close().done()
        assert isinstance(models.begin_close().exception(), RuntimeError)
        assert owner.begin_close().done()
        assert privacy.resource_status()['writer_lease']
        assert privacy.ledger._db is not None
        assert model_ref() is not None and model._model is model_ref()
        assert output.closed and not output.thread.is_alive()
        assert owner.get_delivery_status()['failed_barriers'] > 0
        assert owner.get_delivery_status()['native_calls'] == 0
        await asyncio.gather(speech, return_exceptions=True)
        print('failed_owner_models_and_privacy_lease_retained', flush=True)
    finally:
        # The child is the explicit final owner of this intentionally failed
        # lease. No private flags are reset to claim successful retirement.
        await asyncio.gather(speech, return_exceptions=True)
        patch.undo()


def test_failed_response_retains_selected_model_and_privacy_writer_lease(tmp_path):
    code = (
        'import asyncio, runpy, sys\n'
        'from pathlib import Path\n'
        'failure_child = runpy.run_path(sys.argv[1])["failure_child"]\n'
        'asyncio.run(failure_child(Path(sys.argv[2])))\n'
    )
    result = subprocess.run(
        [sys.executable, '-c', code, str(Path(__file__).resolve()), str(tmp_path / 'privacy.sqlite')],
        cwd=Path(voice_graph.__code__.co_filename).resolve().parents[2],
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'failed_owner_models_and_privacy_lease_retained' in result.stdout
