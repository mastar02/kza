"""Real FastRouter/OpenAI/HTTPX tests with synthetic transport leaves only."""

import asyncio
import json
from pathlib import Path
import subprocess
import sys
from contextvars import ContextVar
import os
import threading
from types import SimpleNamespace
import weakref

import httpx
import numpy as np
import openai
import pytest
import pytest_asyncio


class _PrivateControl(KeyboardInterrupt):
    pass


@pytest_asyncio.fixture
async def router_sdk(monkeypatch):
    from src.llm import reasoner
    from src.core import native_executor

    state = SimpleNamespace(
        routers=[], clients=[], sdk_clients=[], workers=[], futures=[], work=[],
        events=[], gates={}, tasks=[], payloads=[], requests=[], mode="ok",
        start_calls=0, context=ContextVar("router_test_context", default="empty"), contexts=[],
    )

    def hold(phase):
        gate = threading.Event(), threading.Event()
        state.gates[phase] = gate
        return gate

    def hit(phase):
        state.events.append(phase)
        if phase in state.gates:
            entered, release = state.gates[phase]
            entered.set()
            assert release.wait(5), f"synthetic {phase} not released"

    def handle(request):
        state.requests.append((request.method, request.url.path))
        if request.method == "GET":
            return httpx.Response(200, json={"object": "list", "data": [
                {"id": "synthetic-router", "object": "model"},
            ]})
        state.contexts.append(state.context.get())
        hit("request")
        if state.mode == "bad-request":
            return httpx.Response(400, json={"error": {
                "message": "synthetic request rejected", "type": "invalid_request_error", "code": "synthetic",
            }})
        if state.mode == "control":
            payload = np.full(32, .375, dtype=np.float32)
            state.payloads.append(weakref.ref(payload))
            raise _PrivateControl(payload)
        return httpx.Response(200, json={
            "id": "synthetic", "object": "text_completion", "created": 0,
            "model": "synthetic-router", "choices": [
                {"index": 0, "text": "synthetic response", "finish_reason": "stop"},
            ], "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        })

    class Transport(httpx.MockTransport):
        def close(self):
            hit("http-close")
            return super().close()

    actual_http = httpx.Client

    def client(*args, **kwargs):
        state.events.append("http-ctor")
        value = actual_http(*args, transport=Transport(handle), **kwargs)
        state.clients.append(value)
        return value

    class HttpxProxy:
        Client = staticmethod(client)

        def __getattr__(self, name):
            return getattr(httpx, name)

    monkeypatch.setattr(reasoner, "httpx", HttpxProxy())

    def api_key(*args, **kwargs):
        state.events.append("resolve-key")
        return "synthetic-token"

    monkeypatch.setattr(reasoner, "_resolve_api_key", api_key)
    actual_openai, original_close = openai.OpenAI, openai.OpenAI.close

    def sdk_client(*args, **kwargs):
        state.events.append("sdk-ctor")
        hit("sdk-before-ctor")
        value = actual_openai(*args, **kwargs)
        state.sdk_clients.append(value)
        return value

    def sdk_close(client):
        hit("sdk-close-enter")
        result = original_close(client)
        hit("sdk-close-return")
        return result

    monkeypatch.setattr(openai, "OpenAI", sdk_client)
    monkeypatch.setattr(actual_openai, "close", sdk_close)
    actual_worker = native_executor._NativeWorker

    class Worker(actual_worker):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            state.workers.append(self)

        def start(self):
            state.start_calls += 1
            return super().start()

        def run(self):
            try:
                return super().run()
            finally:
                hit("native-postlude")

    monkeypatch.setattr(native_executor, "_NativeWorker", Worker)
    original_submit = native_executor.NativeThreadPoolExecutor.submit

    def submit(executor, function, *args, **kwargs):
        work = getattr(function, "__self__", None)
        if work is not None and hasattr(work, "outcome"):
            state.work.append(work)
        future = original_submit(executor, function, *args, **kwargs)
        state.futures.append(future)
        return future

    monkeypatch.setattr(native_executor.NativeThreadPoolExecutor, "submit", submit)

    def router(*, local=True):
        value = reasoner.FastRouter(
            base_url="http://synthetic.invalid/v1", model="synthetic-router",
            destination="local" if local else "cloud", cloud_consent=False,
        )
        state.routers.append(value)
        return value

    def spawn(awaitable):
        task = asyncio.create_task(awaitable)
        state.tasks.append(task)
        return task

    state.router, state.spawn, state.hold, state.hit = router, spawn, hold, hit
    state.reasoner, state.native = reasoner, native_executor
    yield state
    for _, release in state.gates.values():
        release.set()
    for task in state.tasks:
        if not task.done():
            task.cancel()
    if state.tasks:
        await asyncio.gather(*state.tasks, return_exceptions=True)
    failures = []
    for value in state.routers:
        try:
            await value.aclose()
        except BaseException as error:
            failures.append(error)
    for worker in state.workers:
        try:
            if worker._handle.ident:
                worker._handle.join(3)
                assert worker._handle.is_done()
        except BaseException as error:
            failures.append(error)
    if failures:
        raise BaseExceptionGroup("router synthetic SDK cleanup failed", failures)


async def _caught(awaitable):
    try:
        await awaitable
    except BaseException as error:
        return error
    raise AssertionError("expected actual SDK/request failure")


def _retired(state, router):
    assert router._client is None and router._router_executor is None
    assert router._router_work == set()
    assert all(worker._handle.is_done() for worker in state.workers)
    assert all(client.is_closed for client in state.clients)
    assert all(future.done() and future.exception() is None for future in state.futures)
    assert all(work.finished and work.router is None and work.inputs is None and work.context is None
               and work.outcome.error is None and work.outcome.value is None for work in state.work)


@pytest.mark.asyncio
async def test_cloud_denial_precedes_all_sdk_key_and_transport_construction(router_sdk):
    state = router_sdk
    router = state.router(local=False)
    retained = await _caught(router.complete("synthetic prompt"))
    from src.core.egress import EgressDenied

    assert isinstance(retained, EgressDenied)
    assert state.clients == state.sdk_clients == []
    assert "resolve-key" not in state.events and state.requests == []
    await router.aclose()
    _retired(state, router)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["bad-request", "control"])
async def test_current_complete_observer_receives_original_sdk_error_class(router_sdk, mode):
    state = router_sdk
    router = state.router()
    state.mode = mode
    retained = await _caught(router.complete("synthetic prompt"))
    assert type(retained) is (openai.BadRequestError if mode == "bad-request" else _PrivateControl)
    assert all(future.exception() is None for future in state.futures)
    assert all(work.outcome.error is None for work in state.work)
    await router.aclose()
    _retired(state, router)


@pytest.mark.asyncio
@pytest.mark.parametrize("late_mode", ["ok", "control"])
async def test_cancelled_complete_retains_real_callable_and_native_postlude_before_http_close(router_sdk, late_mode):
    state = router_sdk
    router = state.router()
    entered, release = state.hold("request")
    postlude_entered, postlude_release = state.hold("native-postlude")
    state.context.set("caller-context")
    request = state.spawn(router.complete("synthetic held prompt"))
    assert await asyncio.to_thread(entered.wait, 5)
    assert state.contexts == ["caller-context"]
    assert state.clients and not state.clients[0].is_closed
    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request
    assert len(router._router_work) == 1
    assert not state.futures[0].done()
    closing = state.spawn(router.aclose())
    await asyncio.sleep(0)
    assert not closing.done() and "sdk-close-enter" not in state.events
    state.mode = late_mode
    release.set()
    assert await asyncio.to_thread(postlude_entered.wait, 5)
    assert not state.workers[0]._handle.is_done()
    assert state.work[0].finished and state.futures[0].done()
    assert state.futures[0].exception() is None
    assert state.work[0].outcome.value is None and state.work[0].outcome.error is None
    assert state.work[0].inputs is None and state.work[0].router is None
    assert all(reference() is None for reference in state.payloads)
    assert not state.clients[0].is_closed and "sdk-close-enter" not in state.events
    assert not closing.done()
    postlude_release.set()
    await closing
    assert state.events.index("native-postlude") < state.events.index("sdk-close-enter")
    assert state.events.count("http-close") == 1
    _retired(state, router)


"""Appendable native Router queue, concurrency and partial-client cases."""


@pytest.mark.asyncio
async def test_native_router_keeps_default_worker_count_and_two_real_http_calls_in_parallel(router_sdk, monkeypatch):
    state = router_sdk
    router = state.router()
    router.load()
    entered, release = state.hold("request")
    both = threading.Event()
    arrivals, lock = [], threading.Lock()
    transport_type = type(state.clients[0]._transport)
    original = transport_type.handle_request

    def handle(transport, request):
        if request.method == "POST":
            with lock:
                arrivals.append(threading.current_thread())
                if len(arrivals) == 2:
                    both.set()
        return original(transport, request)

    monkeypatch.setattr(transport_type, "handle_request", handle)
    first = state.spawn(router.complete("first synthetic prompt"))
    second = state.spawn(router.complete("second synthetic prompt"))
    assert await asyncio.to_thread(both.wait, 5), "complete calls were serialized behind one native worker"
    assert await asyncio.to_thread(entered.wait, 5)
    assert arrivals[0] is not arrivals[1]
    expected = min(32, (os.process_cpu_count() or 1) + 4)
    assert router._router_executor._max_workers == expected and expected > 1
    assert sum(work.started for work in state.work) == 2
    release.set()
    assert await asyncio.gather(first, second) == ["synthetic response", "synthetic response"]
    await router.aclose()
    _retired(state, router)


@pytest.mark.asyncio
async def test_cancel_before_native_queue_start_discards_inputs_without_http_dispatch(router_sdk):
    state = router_sdk
    router = state.router()
    assert await router.complete("prime") == "synthetic response"
    executor = router._router_executor
    expected = min(32, (os.process_cpu_count() or 1) + 4)
    assert executor._max_workers == expected
    all_entered, release = state.hold("queue-blockers")
    arrivals, lock = [], threading.Lock()

    def block():
        with lock:
            arrivals.append(threading.current_thread())
            if len(arrivals) == expected:
                all_entered.set()
        assert release.wait(5), "synthetic queued-work blocker was not released"

    blockers = [executor.submit(block) for _ in range(expected)]
    assert await asyncio.to_thread(all_entered.wait, 5)
    before = list(state.requests)
    request = state.spawn(router.complete("must never dispatch"))
    await asyncio.sleep(0)
    work, future = state.work[-1], state.futures[-1]
    assert not work.started and not future.running()
    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request
    assert work.discarded and work.finished
    assert work.inputs is None and work.context is None and work.router is None
    release.set()
    await asyncio.gather(*(asyncio.wrap_future(future) for future in [*blockers, future]))
    assert state.requests == before
    assert work.outcome.value is None and work.outcome.error is None
    await router.aclose()
    _retired(state, router)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["resolve-key", "sdk-ctor"])
async def test_key_resolution_precedes_http_and_partial_openai_constructor_retires_real_http(router_sdk, monkeypatch, phase):
    state = router_sdk
    router = state.router()

    class ExpectedFailure(RuntimeError):
        pass

    if phase == "resolve-key":
        def resolve(*args, **kwargs):
            state.events.append("resolve-key-failed")
            raise ExpectedFailure("synthetic key resolution failure")

        monkeypatch.setattr(state.reasoner, "_resolve_api_key", resolve)
    else:
        def construct(*args, **kwargs):
            state.events.append("sdk-ctor-failed")
            assert kwargs["http_client"] is state.clients[0]
            raise ExpectedFailure("synthetic SDK constructor failure")

        monkeypatch.setattr(openai, "OpenAI", construct)
    with pytest.raises(ExpectedFailure):
        router.load()
    if phase == "resolve-key":
        assert state.clients == [] and "http-ctor" not in state.events
    else:
        assert state.events.index("resolve-key") < state.events.index("http-ctor") < state.events.index("sdk-ctor-failed")
        assert len(state.clients) == 1 and not state.clients[0].is_closed
        assert state.sdk_clients == []
        assert len(router._router_session._engines) == 1
    await router.aclose()
    assert state.events.count("http-close") == (0 if phase == "resolve-key" else 1)
    _retired(state, router)


def _router_native_child(mode):
    import asyncio
    import os
    import threading
    import httpx
    import openai
    import pytest
    from src.llm import reasoner
    from src.core import native_executor

    patch = pytest.MonkeyPatch()
    requests, workers, starts, guards, failures, results = [], [], [], [], [], []
    clients, closed = [], []
    request_entered, request_release = threading.Event(), threading.Event()
    postlude_entered, postlude_release = threading.Event(), threading.Event()
    closing_started = threading.Event()
    if mode != "start-control":
        request_release.set()
        postlude_release.set()
    holder = []

    def emit(reason, **scalars):
        print("ROUTER_NATIVE=" + json.dumps({"mode": mode, "reason": reason, **scalars}), flush=True)

    def require_callback_rejection():
        try:
            holder[0].close()
        except RuntimeError as error:
            guards.append(str(error))
            assert str(error).startswith("router_close_from_")
        else:
            raise AssertionError("Router callback was allowed to wait on its own producer/closer")

    def handle(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "synthetic-router", "object": "model"}], "object": "list"})
        requests.append(request.url.path)
        request_entered.set()
        if mode == "request-callback":
            require_callback_rejection()
            assert not holder[0]._adapter_sealed
        assert request_release.wait(5), "synthetic HTTP body not released"
        return httpx.Response(200, json={"id": "synthetic", "object": "text_completion", "created": 0,
            "model": "synthetic-router", "choices": [{"index": 0, "text": "synthetic response", "finish_reason": "stop"}]})

    class Transport(httpx.MockTransport):
        def close(self):
            closed.append(True)
            return super().close()

    actual_http = httpx.Client

    def client(*args, **kwargs):
        value = actual_http(*args, transport=Transport(handle), **kwargs)
        clients.append(value)
        return value

    class HttpxProxy:
        Client = staticmethod(client)

        def __getattr__(self, name):
            return getattr(httpx, name)

    patch.setattr(reasoner, "httpx", HttpxProxy())
    patch.setattr(reasoner, "_resolve_api_key", lambda *args, **kwargs: "synthetic-token")
    original_sdk_close = openai.OpenAI.close

    def sdk_close(sdk):
        if mode == "close-callback":
            require_callback_rejection()
        return original_sdk_close(sdk)

    patch.setattr(openai.OpenAI, "close", sdk_close)
    actual_worker = native_executor._NativeWorker

    class Worker(actual_worker):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            workers.append(self)

        def start(self):
            starts.append(True)
            super().start()
            if mode == "start-control":
                assert request_entered.wait(3), "real native callable did not dispatch before injected control"
                assert self._handle.ident and self._started.is_set()
                raise KeyboardInterrupt

        def run(self):
            try:
                return super().run()
            finally:
                postlude_entered.set()
                assert postlude_release.wait(5), "actual native Thread postlude not released"

    patch.setattr(native_executor, "_NativeWorker", Worker)

    async def exercise():
        router = reasoner.FastRouter(base_url="http://synthetic.invalid/v1", model="synthetic-router",
                                     destination="local", cloud_consent=False)
        holder.append(router)
        router.load()
        if mode == "start-control":
            try:
                await router.complete("first actual dispatch")
            except BaseException as error:
                results.append(type(error).__name__)
            assert results == ["KeyboardInterrupt"]
            try:
                await router.complete("must not retry uncertain dispatch")
            except RuntimeError as error:
                results.append(type(error).__name__)
            else:
                raise AssertionError("broken executor accepted a second dispatch")
            assert len(starts) == 1 and len(requests) == 1
        else:
            assert await router.complete("synthetic callback request") == "synthetic response"
        closing_started.set()
        await router.aclose()
        assert all(worker._handle.is_done() for worker in workers)
        assert all(client.is_closed for client in clients) and closed == [True]
        assert router._router_executor is None and router._client is None and not router._router_work
        if mode != "start-control":
            assert len(guards) == 1

    def run():
        try:
            asyncio.run(exercise())
        except BaseException as error:
            failures.append((type(error).__name__, str(error)))

    caller = threading.Thread(target=run, daemon=True, name="router-isolated-observer")
    caller.start()
    if mode == "start-control":
        assert closing_started.wait(5), f"real dispatch/control phase failed: {failures}"
        assert workers[0]._handle.ident and not workers[0]._handle.is_done()
        assert not caller._handle.is_done() and closed == []
        request_release.set()
        assert postlude_entered.wait(5), f"worker did not reach actual exit postlude: {failures}"
        assert not workers[0]._handle.is_done() and not caller._handle.is_done() and closed == []
        postlude_release.set()
    caller._handle.join(2)
    if not caller._handle.is_done():
        emit("callback_self_wait", actual_native_observer_exited=False,
             started_native_handles=[worker._handle.ident for worker in workers],
             request_count=len(requests), close_count=len(closed), guard_count=len(guards))
        os._exit(1)
    assert failures == [], failures
    emit("confirmed", actual_native_observer_exited=True,
         actual_pool_native_exited=all(worker._handle.is_done() for worker in workers),
         start_calls=len(starts), request_count=len(requests), close_count=len(closed), guard_count=len(guards))
    os._exit(0)


def _run_router_native_child(mode):
    root = next(parent for parent in Path(__file__).resolve().parents
                if (parent / "tools/benchmark_latency.py").is_file())
    script = "import runpy,sys; m=runpy.run_path(sys.argv[1]); m['_router_native_child'](sys.argv[2])"
    result = subprocess.run([sys.executable, "-c", script, str(Path(__file__).resolve()), mode],
                            cwd=root, capture_output=True, text=True, timeout=20)
    receipts = [json.loads(line.removeprefix("ROUTER_NATIVE=")) for line in result.stdout.splitlines()
                if line.startswith("ROUTER_NATIVE=")]
    assert receipts, f"fixture failed before native receipt: {result.stdout}\n{result.stderr}"
    assert result.returncode == 0, f"{receipts[-1]}\n{result.stderr}"
    assert receipts[-1]["reason"] == "confirmed", receipts[-1]


def test_real_native_start_then_control_waits_callable_and_postlude_without_retry():
    _run_router_native_child("start-control")


@pytest.mark.parametrize("mode", ["request-callback", "close-callback"])
def test_router_native_request_and_sdk_closer_callbacks_cannot_wait_on_their_own_owner(mode):
    _run_router_native_child(mode)
