"""An uncertain native submission cannot expire the cache operation guard."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
import threading
from weakref import ref

import numpy as np

from src.pipeline.response_handler import ResponseHandler
from src.tts.response_cache import ResponseCache


async def until(predicate):
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(.005)


async def partial_submit_child():
    import src.tts.response_cache as module

    module.CANONICAL_PHRASES, module.TEMPLATES = ['Listo'], {}
    entered, check, checked, release = (threading.Event() for _ in range(4))
    errors = []

    class QueuedThenFailed(ThreadPoolExecutor):
        failed = False

        def submit(self, function, *args, **kwargs):
            future = super().submit(function, *args, **kwargs)
            if not self.failed:
                self.failed = True
                # The Future is deliberately not returned. Execution is still
                # possible, exactly as with a submit failure after enqueue.
                assert future is not None
                raise RuntimeError('synthetic submit failed after enqueue')
            return future

    class Engine:
        sample_rate = 24000

        def synthesize(self, _text):
            entered.set()
            assert check.wait(5), 'native guard check was not requested'
            for function in (cache.require_close_caller, handler.begin_close):
                try:
                    function()
                except RuntimeError as error:
                    errors.append(str(error))
                else:
                    errors.append('incorrectly allowed')
            checked.set()
            assert release.wait(5), 'synthetic native call was not released'
            return np.ones(64, dtype=np.float32)

    engine = Engine()
    handler = ResponseHandler(engine)
    cache = ResponseCache(engine, synthesize_callback=handler.asynthesize)
    handler.register_owned_response_cache(cache)
    handler.set_earcon(np.ones(64, dtype=np.float32), 24000)
    earcon = ref(handler._earcon_audio)
    asyncio.get_running_loop().set_default_executor(QueuedThenFailed(max_workers=2))
    building = asyncio.create_task(cache.build())
    try:
        await until(entered.is_set)
        await until(lambda: handler.get_delivery_status()['failed_barriers'] == 1)
        for _ in range(5):
            await asyncio.sleep(0)
        assert not building.done()
        check.set()
        await until(checked.is_set)
        assert errors == ['response_cache_close_from_operation'] * 2
        assert not handler.get_delivery_status()['closed']
        closing = handler.begin_close()
        for _ in range(5):
            await asyncio.sleep(0)
        assert not closing.done() and not building.done()
        assert cache._tts is engine and earcon() is not None
    finally:
        check.set()
        release.set()
    await until(lambda: any(exited.is_set() for _, exited in handler._native_tasks.values()))
    async with asyncio.timeout(5):
        await asyncio.shield(building)
    # The submit observer has no successful receipt. Its failed barrier and
    # resource ownership remain unconfirmed even after the native finally.
    assert not closing.done()
    assert handler.get_delivery_status()['failed_barriers'] == 1
    assert cache._tts is engine and earcon() is not None
    print('uncertain_submit_pin_released_only_by_actual_native_exit', flush=True)


def test_partial_submit_keeps_guard_and_resources_until_actual_native_exit():
    code = (
        'import asyncio, os, runpy, sys, traceback\n'
        'child = runpy.run_path(sys.argv[1])["partial_submit_child"]\n'
        'loop = asyncio.new_event_loop()\n'
        'asyncio.set_event_loop(loop)\n'
        'try:\n'
        '    loop.run_until_complete(child())\n'
        'except BaseException:\n'
        '    traceback.print_exc()\n'
        '    os._exit(1)\n'
        'os._exit(0)\n'
    )
    result = subprocess.run([sys.executable, '-c', code, str(Path(__file__).resolve())],
        cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'uncertain_submit_pin_released_only_by_actual_native_exit' in result.stdout
