"""Isolated real LOCAL FastRouter/SDK path with post-store metadata failure."""
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


_CHILD = r'''
import asyncio
import threading
import httpx
import openai
from src.core import native_executor
from src.llm import reasoner, router_lifecycle

MODE = __MODE__

async def main():
    request_entered = threading.Event()
    request_release = threading.Event()
    postlude_entered = threading.Event()
    postlude_release = threading.Event()
    clients, futures, workers, work_items, sdk_error_types = [], [], [], [], []

    def handle(request):
        if request.method == "GET":
            return httpx.Response(200, json={"object": "list", "data": [
                {"id": "synthetic-router", "object": "model"},
            ]})
        request_entered.set()
        assert request_release.wait(5), "synthetic request not released"
        return httpx.Response(400, json={"error": {
            "message": "synthetic private SDK error", "type": "invalid_request_error",
            "code": "synthetic",
        }})

    actual_http = httpx.Client
    def http_client(*args, **kwargs):
        result = actual_http(*args, transport=httpx.MockTransport(handle), **kwargs)
        clients.append(result)
        return result
    class HttpxProxy:
        Client = staticmethod(http_client)
        def __getattr__(self, name):
            return getattr(httpx, name)
    reasoner.httpx = HttpxProxy()
    reasoner._resolve_api_key = lambda *args, **kwargs: "synthetic-token"

    actual_worker = native_executor._NativeWorker
    class Worker(actual_worker):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            workers.append(self)
        def run(self):
            try:
                return super().run()
            finally:
                postlude_entered.set()
                assert postlude_release.wait(5), "synthetic postlude not released"
    native_executor._NativeWorker = Worker
    actual_submit = native_executor.NativeThreadPoolExecutor.submit
    def submit(executor, function, *args, **kwargs):
        work = getattr(function, "__self__", None)
        if isinstance(work, router_lifecycle._RouterWork):
            work_items.append(work)
        result = actual_submit(executor, function, *args, **kwargs)
        futures.append(result)
        return result
    native_executor.NativeThreadPoolExecutor.submit = submit

    class OnceAfterExit:
        def __init__(self, lock):
            self.lock = lock
            self.armed = True
        def __enter__(self):
            self.lock.acquire()
            return self
        def __exit__(self, kind, error, traceback):
            self.lock.release()
            if self.armed:
                self.armed = False
                raise RuntimeError("synthetic post-store lock failure")
            return False
    actual_store = router_lifecycle._RouterWork._store
    def store(work, *, value=None, error=None):
        try:
            if isinstance(error, openai.BadRequestError):
                sdk_error_types.append(type(error))
                if MODE == "abandoned-post-store":
                    work._lock = OnceAfterExit(work._lock)
            return actual_store(work, value=value, error=error)
        finally:
            # Instrumentation must not supply an extra SDK traceback loan.
            work = value = error = None
    router_lifecycle._RouterWork._store = store

    router = reasoner.FastRouter(
        base_url="http://synthetic.invalid/v1", model="synthetic-router",
        destination="local", cloud_consent=False,
    )
    request = closing = None
    try:
        request = asyncio.create_task(router.complete("synthetic private prompt"))
        assert await asyncio.to_thread(request_entered.wait, 3)
        assert clients and not clients[0].is_closed
        if MODE == "abandoned-post-store":
            request.cancel()
            try:
                await request
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError("cancelled observer must receive cancellation")
            assert work_items and work_items[0].observer is False
        request_release.set()
        if MODE == "normal-original":
            try:
                await request
            except openai.BadRequestError as error:
                assert type(error) is openai.BadRequestError
            else:
                raise AssertionError("live observer lost original SDK error class")
        else:
            # Observe actual Future completion without raising its retained
            # exception; the following assertion is the exact RED boundary.
            native_done = threading.Event()
            futures[0].add_done_callback(lambda _done: native_done.set())
            assert await asyncio.to_thread(native_done.wait, 3)
        assert sdk_error_types == [openai.BadRequestError]
        assert len(futures) == 1 and futures[0].done()
        assert futures[0].exception() is None, "raw metadata/SDK exception escaped native Future"
        outcome = futures[0].result()
        assert outcome.error is None and outcome.value is None
        assert work_items[0].finished and work_items[0].context is None
        assert work_items[0].inputs is None and work_items[0].router is None
        closing = asyncio.create_task(router.aclose())
        assert await asyncio.to_thread(postlude_entered.wait, 3)
        assert workers and not workers[0]._handle.is_done()
        assert not closing.done() and not clients[0].is_closed
        postlude_release.set()
        if MODE == "abandoned-post-store":
            try:
                await closing
            except RuntimeError as error:
                assert "router" in str(error)
            else:
                raise AssertionError("metadata failure falsely confirmed retirement")
            assert router in router_lifecycle._ROUTERS
            assert not clients[0].is_closed
            assert not router._router_session._sealed
        else:
            await closing
            assert router not in router_lifecycle._ROUTERS
            assert router._client is None and router._router_executor is None
            assert clients[0].is_closed
        assert all(worker._handle.is_done() for worker in workers)
    finally:
        request_release.set()
        postlude_release.set()
        if request is not None and not request.done():
            request.cancel()
        if request is not None:
            await asyncio.gather(request, return_exceptions=True)
        if closing is not None:
            await asyncio.gather(closing, return_exceptions=True)
        try:
            await router.aclose()
        except RuntimeError:
            # Intentionally retained uncertain owners are process-isolated;
            # no registry reset or external close fabricates a receipt.
            assert router._router_work_failure
        for worker in workers:
            if worker._handle.ident:
                worker._handle.join(3)
                assert worker._handle.is_done()

asyncio.run(main())
'''


@pytest.mark.parametrize("mode", ["normal-original", "abandoned-post-store"])
def test_real_router_native_future_never_retains_abandoned_sdk_error(mode):
    root = Path(__file__).resolve()
    while root.name != "kza" and root.parent != root:
        root = root.parent
    source = textwrap.dedent(_CHILD).replace("__MODE__", repr(mode))
    result = subprocess.run(
        [sys.executable, "-c", source], cwd=root, text=True,
        capture_output=True, timeout=25,
    )
    assert result.returncode == 0, result.stdout + result.stderr
