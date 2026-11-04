"""Actual graph/FastRouter ownership, with synthetic model/HTTP leaves."""

import asyncio
from contextlib import AsyncExitStack
import threading
from types import SimpleNamespace

import httpx
import openai
import pytest
import pytest_asyncio

from src.main import build_application
from tests.integration.test_application_graph import leaves as application_leaves
from tests.mocks.application_graph import offline_config, offline_home

leaves = application_leaves
pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures("dashboard_auth")]


class _RecordingStack(AsyncExitStack):
    def __init__(self):
        super().__init__()
        self.registered = []

    def push_async_callback(self, callback, *args, **kwargs):
        self.registered.append(callback)
        return super().push_async_callback(callback, *args, **kwargs)


@pytest_asyncio.fixture
async def graph_router(monkeypatch):
    import src.main as main_module
    from src.llm import reasoner

    state = SimpleNamespace(routers=[], clients=[], owners=[], body_results=[], fail_ctor=False, registered_before_load=[], stack=None,
                            events=[], borrower_entered=threading.Event(), borrower_release=threading.Event(),
                            borrower_exited=threading.Event(), request_entered=asyncio.Event(), nlu_entered=asyncio.Event())
    actual_router = reasoner.FastRouter

    def construct(*args, **kwargs):
        value = actual_router(*args, **kwargs)
        state.routers.append(value)
        return value

    monkeypatch.setattr(main_module, "FastRouter", construct)
    original_sdk = openai.OpenAI

    def sdk(*args, **kwargs):
        actual = state.routers[-1]
        state.owners.extend(owner for owner in actual._router_session._engines.values() if owner not in state.owners)
        registered = any(getattr(callback, "__self__", None) is actual
                         for callback in state.stack.registered)
        state.registered_before_load.append(registered)
        if state.fail_ctor:
            raise RuntimeError("synthetic partial router load")
        return original_sdk(*args, **kwargs)

    monkeypatch.setattr(openai, "OpenAI", sdk)
    monkeypatch.setattr(reasoner, "_resolve_api_key", lambda *args, **kwargs: "synthetic-token")
    actual_http = httpx.Client

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"object": "list", "data": [
                {"id": "synthetic-router", "object": "model"},
            ]})
        return httpx.Response(200, json={"id": "synthetic", "object": "text_completion", "created": 0,
            "model": "synthetic-router", "choices": [{"index": 0, "text": "synthetic response", "finish_reason": "stop"}]})

    def http_client(*args, **kwargs):
        client = actual_http(*args, transport=httpx.MockTransport(handler), **kwargs)
        state.clients.append(client)
        return client

    class HttpxProxy:
        Client = staticmethod(http_client)

        def __getattr__(self, name):
            return getattr(httpx, name)

    monkeypatch.setattr(reasoner, "httpx", HttpxProxy())
    original_fast_close = actual_router.aclose

    async def fast_close(router):
        state.events.append("fast-enter")
        await original_fast_close(router)
        state.events.append("fast-exit")

    monkeypatch.setattr(actual_router, "aclose", fast_close)
    original_request_close = main_module.RequestRouter.aclose

    async def request_close(router):
        state.events.append("request-enter")
        state.request_entered.set()
        await original_request_close(router)
        state.events.append("request-exit")

    monkeypatch.setattr(main_module.RequestRouter, "aclose", request_close)
    from src.nlu.llm_router import LLMCommandRouter
    original_nlu_close = LLMCommandRouter.aclose

    async def nlu_close(router):
        state.events.append("nlu-enter")
        state.nlu_entered.set()
        await original_nlu_close(router)
        state.events.append("nlu-exit")

    monkeypatch.setattr(LLMCommandRouter, "aclose", nlu_close)
    yield state
    state.borrower_release.set()
    failures = []
    for router in state.routers:
        try:
            await original_fast_close(router)
        except BaseException as error:
            failures.append(error)
    if failures:
        raise BaseExceptionGroup("synthetic graph router cleanup failed", failures)


def _router_config(tmp_path, url):
    config = offline_config(tmp_path, url)
    config["router"] = {"enabled": True, "base_url": "http://synthetic.invalid/v1",
                        "model": "synthetic-router", "destination": "local"}
    config["nlu"] = {"llm_router": {"enabled": True}}
    return config


async def test_graph_registers_real_fast_router_before_partial_load_failure(leaves, graph_router, tmp_path, monkeypatch):
    import src.nlu.llm_router as nlu_module

    state = graph_router
    state.fail_ctor = True

    def later_failure(*args, **kwargs):
        assert kwargs["fast_router"] is state.routers[0]
        raise RuntimeError("synthetic later NLU constructor failure")

    monkeypatch.setattr(nlu_module, "LLMCommandRouter", later_failure)
    async with offline_home() as (url, observations):
        stack = _RecordingStack()
        state.stack = stack
        with pytest.raises(RuntimeError, match="synthetic later NLU constructor failure"):
            async with stack:
                await build_application(_router_config(tmp_path, url), shutdown=stack, factories=leaves)
        assert len(state.routers) == len(state.clients) == 1
        # Genuine baseline RED: the real partial HTTP client remains owned/open,
        # rather than a missing close API or only an unregistered callback flag.
        assert state.clients[0].is_closed
        assert state.routers[0]._client is None
        assert len(state.owners) == 1
        assert all(owner._confirmed and owner._handle.is_done() for owner in state.owners)
        assert state.registered_before_load == [True]
        assert observations["service_calls"] == 0


async def test_graph_lifo_drains_real_consumers_before_fast_router_native_and_http_close(leaves, graph_router, tmp_path):
    state = graph_router
    async with offline_home() as (url, observations):
        stack = _RecordingStack()
        state.stack = stack
        async with stack:
            graph = await build_application(_router_config(tmp_path, url), shutdown=stack, factories=leaves)
            fast = state.routers[0]
            classifier = graph.pipeline.request_router.llm_command_router
            assert classifier.router is fast
            assert await fast.complete("prime real native router") == "synthetic response"
            workers = tuple(fast._router_executor._threads)

            def borrowed_classification():
                state.borrower_entered.set()
                try:
                    assert state.borrower_release.wait(5), "synthetic admitted classifier not released"
                    result = fast.generate(["synthetic command text"])
                    state.body_results.append(result)
                    return result
                finally:
                    state.borrower_exited.set()

            producer = asyncio.create_task(classifier._work.run(borrowed_classification))
            closing = None
            try:
                assert await asyncio.to_thread(state.borrower_entered.wait, 5)
                closing = asyncio.create_task(stack.aclose())
                await asyncio.wait_for(state.request_entered.wait(), 5)
                await asyncio.wait_for(state.nlu_entered.wait(), 5)
                assert not closing.done() and "fast-enter" not in state.events
                assert not state.clients[0].is_closed
            finally:
                state.borrower_release.set()
                producer_result = (await asyncio.gather(producer, return_exceptions=True))[0]
                if closing is not None:
                    await closing
            assert producer_result == ["synthetic response"] or isinstance(producer_result, asyncio.CancelledError)
            assert state.body_results == [["synthetic response"]] and state.borrower_exited.is_set()
            assert all(worker._handle.is_done() for worker in workers)
            assert state.clients[0].is_closed and fast._client is None
            assert max(index for index, event in enumerate(state.events) if event == "request-exit") < state.events.index("fast-enter")
            assert max(index for index, event in enumerate(state.events) if event == "nlu-exit") < state.events.index("fast-enter")
            assert state.events.count("fast-enter") == state.events.count("fast-exit") == 1
        assert observations["service_calls"] == 0


async def test_graph_privacy_writer_waits_for_cancelled_router_native_and_postlude(leaves, graph_router, tmp_path, monkeypatch):
    from src.llm import reasoner
    from src.core import native_executor

    state = graph_router
    request_entered, request_release = threading.Event(), threading.Event()
    postlude_entered, postlude_release = threading.Event(), threading.Event()
    fast_begin = asyncio.Event()
    native_workers, ledger_closed = [], []
    actual_type = reasoner.FastRouter
    original_begin = actual_type.begin_close

    def begin(router):
        task = original_begin(router)
        if router in state.routers:
            fast_begin.set()
        return task

    # Register this real observer before build, so PrivacyServices' WeakMethod
    # refers to it as well as the later AsyncExitStack callback path.
    monkeypatch.setattr(actual_type, "begin_close", begin)
    original_worker = native_executor._NativeWorker

    class Worker(original_worker):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            native_workers.append(self)

        def run(self):
            try:
                return super().run()
            finally:
                postlude_entered.set()
                assert postlude_release.wait(5), "real Router native postlude not released"

    monkeypatch.setattr(native_executor, "_NativeWorker", Worker)
    original_request = httpx.MockTransport.handle_request

    def request(transport, request):
        if request.method == "POST":
            request_entered.set()
            assert request_release.wait(5), "actual Router HTTP request not released"
        return original_request(transport, request)

    monkeypatch.setattr(httpx.MockTransport, "handle_request", request)
    async with offline_home() as (url, observations):
        stack = _RecordingStack()
        state.stack = stack
        async with stack:
            graph = await build_application(_router_config(tmp_path, url), shutdown=stack, factories=leaves)
            fast = state.routers[0]
            ledger = graph.privacy.ledger
            original_ledger_close = type(ledger).close

            def close_ledger(actual):
                if actual is ledger:
                    ledger_closed.append(True)
                return original_ledger_close(actual)

            monkeypatch.setattr(type(ledger), "close", close_ledger)
            lease = graph.privacy._lease
            assert lease is not None and graph.privacy.resource_status()["writer_lease"]
            observer = asyncio.create_task(fast.complete("synthetic held graph request"))
            closing = None
            try:
                assert await asyncio.to_thread(request_entered.wait, 5)
                observer.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await observer
                assert fast._router_work and native_workers
                closing = asyncio.create_task(stack.aclose())
                # Without the privacy borrower, this event arrives only at the
                # earlier stack callback, after the ledger/lease were released.
                await asyncio.wait_for(fast_begin.wait(), 5)
                assert not closing.done()
                assert graph.privacy._lease == lease
                assert graph.privacy.resource_status()["writer_lease"]
                assert ledger_closed == []
                request_release.set()
                assert await asyncio.to_thread(postlude_entered.wait, 5)
                assert not native_workers[0]._handle.is_done()
                assert graph.privacy._lease == lease and ledger_closed == []
                assert graph.privacy.resource_status()["writer_lease"]
            finally:
                request_release.set()
                postlude_release.set()
                if not observer.done():
                    observer.cancel()
                await asyncio.gather(observer, return_exceptions=True)
                if closing is not None:
                    await closing
            assert all(worker._handle.is_done() for worker in native_workers)
            assert ledger_closed == [True]
            assert not graph.privacy.resource_status()["writer_lease"]
            assert graph.privacy._lease is None
            assert observations["service_calls"] == 0
