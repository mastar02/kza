"""Private R graph contracts with actual factory/owners and synthetic SDKs.

Keep beside the core contract test, or place in tests/integration after root
copies the core to tests/unit/llm. Failed owners remain until isolated OS exit.
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace

import httpx
import pytest

_core_path = Path(__file__).with_name("test_http_reasoner_lifecycle.py")
if not _core_path.exists():
    _core_path = Path(__file__).resolve().parents[2] / "tests/unit/llm/test_http_reasoner_lifecycle.py"
_core = runpy.run_path(str(_core_path))
Probe, Gate, ANSWER = (_core[name] for name in ("Probe", "Gate", "ANSWER"))


class Postlude:
    """Hold inside SDK.close after actual HTTP close, before SDK returns."""

    def __init__(self, sdk):
        self.sdk = sdk
        self.entered, self.release, self.exited = (threading.Event() for _ in range(3))
        self.thread = None
        original = sdk.close

        def close():
            original()
            self.thread = threading.current_thread()
            self.entered.set()
            try:
                self.release.wait()
            finally:
                self.exited.set()

        sdk.close = close


class GraphProbe:
    def __init__(self, patch, root, *, hold_closes=False):
        from src.application import ApplicationFactories, HttpReasonerOwner
        from src.llm import reasoner
        from src.main import build_application
        import src.vectordb.chroma_sync as index_module
        from tests.mocks.application_graph import OfflineEncoder, OfflineSTT, OfflineTTS

        self.patch, self.root = patch, root
        self.sync = Probe(patch)
        self.wrappers, self.load_threads, self.registered, self.postludes = [], [], {}, []
        self.owner = self.graph = self.privacy = None
        self.ledger_closed, self.request_clients = [], []
        self.owner_entered = asyncio.Event()
        self.loop_thread = threading.current_thread()
        self.build_application = build_application
        patch.chdir(root)
        patch.setattr(index_module, "SentenceTransformer", lambda *args, **kwargs: OfflineEncoder())

        class Collection:
            def count(self):
                return 0

            def get(self, *args, **kwargs):
                return {"ids": [], "documents": [], "metadatas": [], "embeddings": []}

            def query(self, *args, **kwargs):
                return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        class ChromaSDK:
            def __init__(self, *args, **kwargs):
                self.collections = {}

            def get_or_create_collection(self, name, **kwargs):
                return self.collections.setdefault(name, Collection())

            def get_collection(self, name, **kwargs):
                # The command-index resolver opens a pointer target without creating it.
                if name not in self.collections:
                    from chromadb.errors import NotFoundError
                    raise NotFoundError(f"Collection [{name}] does not exist")
                return self.collections[name]

            def close(self):
                pass  # Synthetic leaf, no invented model-disposal protocol.

        patch.setattr(index_module.chromadb, "PersistentClient", ChromaSDK)
        stt, tts = OfflineSTT(), OfflineTTS()
        self.factories = ApplicationFactories(stt=lambda _: stt, tts=lambda _: tts, preflight=lambda: None)
        original_add = HttpReasonerOwner.add_model
        original_load = reasoner.HttpReasoner.load
        original_begin = HttpReasonerOwner.begin_close
        state = self

        def add(owner, model):
            original_add(owner, model)
            if isinstance(model, reasoner.HttpReasoner):
                assert owner._models[id(model)] is model
                state.registered[id(model)] = owner
                state.owner = owner

        def load(wrapper):
            assert id(wrapper) in state.registered  # BEFORE any SDK load work.
            worker = threading.current_thread()
            assert worker is not state.loop_thread
            state.wrappers.append(wrapper)
            state.load_threads.append(worker)
            result = original_load(wrapper)
            if hold_closes:
                state.postludes.append(Postlude(wrapper._client))
            return result

        def begin(owner):
            task = original_begin(owner)
            if owner is state.owner:
                state.owner_entered.set()
            return task

        patch.setattr(HttpReasonerOwner, "add_model", add)
        patch.setattr(HttpReasonerOwner, "begin_close", begin)
        patch.setattr(reasoner.HttpReasoner, "load", load)

    def config(self, url):
        from tests.mocks.application_graph import offline_config

        config = offline_config(self.root, url)
        config["dashboard"]["enabled"] = False
        config["reasoner"] = {"mode": "http", "http_base_url": "http://127.0.0.1:8200/v1",
                              "http_model": "synthetic-main", "destination": "local", "cloud": {"consent": False}}
        config["orchestrator"] = {"enabled": True, "context": {"compaction": {
            "enabled": True, "base_url": "http://127.0.0.1:8201/v1", "model": "synthetic-compactor",
            "destination": "local", "timeout_s": 30,
        }}}
        return config

    def acquired(self, count=2):
        assert len(self.wrappers) == len(self.load_threads) == len(self.sync.sdks) == len(self.sync.http) == count
        assert len(self.registered) == count
        assert self.wrappers[0].base_url == "http://127.0.0.1:8200/v1"
        if count == 2:
            assert self.wrappers[1].base_url == "http://127.0.0.1:8201/v1"
            assert self.wrappers[0] is not self.wrappers[1]
        assert all(thread is not self.loop_thread for thread in self.load_threads)

    def observe_privacy(self, privacy):
        self.privacy = privacy
        self.ledger, self.lease = privacy.ledger, privacy._lease
        assert self.lease is not None and privacy.resource_status()["writer_lease"]
        original = type(self.ledger).close
        state = self

        def close(actual):
            if actual is state.ledger:
                state.ledger_closed.append(True)
            return original(actual)

        self.patch.setattr(type(self.ledger), "close", close)

    def authority_retained(self):
        assert self.privacy._lease == self.lease
        assert self.privacy.resource_status()["writer_lease"]
        assert self.ledger_closed == []

    def retired(self):
        assert all(sdk.close_calls == 1 for sdk in self.sync.sdks)
        assert all(raw.close_calls == 1 and raw.is_closed for raw in self.sync.http)
        assert all(wrapper._client is None for wrapper in self.wrappers)
        assert all(postlude.exited.is_set() for postlude in self.postludes)


async def _event(event):
    if isinstance(event, asyncio.Event):
        await event.wait()
    else:
        assert await asyncio.to_thread(event.wait, 6), "actual SDK boundary not reached"


async def _caught(awaitable):
    try:
        return await awaitable
    except BaseException as error:
        return error


def _watchdog(state):
    finished = threading.Event()

    def observe():
        if not finished.wait(18):
            print("HTTP_GRAPH_CONTRACT=" + json.dumps({"reason": "watchdog_pending_actual_owner",
                "owned_clients": len(state.sync.sdks),
                "sdk_close_calls": [sdk.close_calls for sdk in state.sync.sdks],
                "sdk_postludes_entered": [gate.entered.is_set() for gate in state.postludes],
                "sdk_postludes_exited": [gate.exited.is_set() for gate in state.postludes],
                "ledger_close_calls": len(state.ledger_closed)}), flush=True)
            os._exit(1)

    threading.Thread(target=observe, daemon=True, name="http-graph-contract-watchdog").start()
    return finished


def _child(kind, argument, temporary):
    async def run():
        from tests.mocks.application_graph import offline_home

        patch = pytest.MonkeyPatch()
        state = GraphProbe(patch, Path(temporary), hold_closes=kind in {"partial", "held"})
        finished = _watchdog(state)
        async with offline_home() as (url, observations):
            if kind == "registered":
                async with AsyncExitStack() as stack:
                    graph = await state.build_application(state.config(url), shutdown=stack, factories=state.factories)
                    state.acquired()
                    assert graph.http_reasoners is state.owner
                    assert tuple(state.owner._models.values()) == tuple(state.wrappers)
                    assert graph.voice_services.responses.llm is graph.pipeline.response_handler.llm is state.wrappers[0]
                    assert graph.pipeline._orchestrator.context_manager.compactor.reasoner is state.wrappers[1]
                    assert not any(raw.is_closed for raw in state.sync.http)
                state.retired()
            elif kind == "partial":
                await _partial(state, url, argument)
            elif kind == "held":
                route, cancels = argument
                await _held(state, url, route, cancels)
            elif kind == "failed":
                await _failed(state, url)
                raise AssertionError("failed owner must remain until isolated OS exit")
            else:
                raise AssertionError("unknown private graph contract")
            assert observations["service_calls"] == 0
        print("HTTP_GRAPH_CONTRACT=" + json.dumps({"reason": "confirmed", "kind": kind,
            "loaded_on_actual_workers": True, "all_owned_handles_retired": True,
            "ledger_close_calls": len(state.ledger_closed)}), flush=True)
        finished.set()
        patch.undo()

    try:
        asyncio.run(run())
    except BaseException as error:
        frames = []
        trace = error.__traceback__
        while trace is not None:
            frames.append({"file": Path(trace.tb_frame.f_code.co_filename).name,
                           "function": trace.tb_frame.f_code.co_name, "line": trace.tb_lineno})
            trace = trace.tb_next
        print("HTTP_GRAPH_CONTRACT=" + json.dumps({"reason": "assertion_or_lifecycle_failure",
              "error_type": type(error).__name__, "kind": kind, "frames": frames}), flush=True)
        os._exit(1)


async def _partial(state, url, stage):
    import src.main as main

    class LaterFactoryFailure(RuntimeError):
        pass

    if stage == "after-main":
        async def fail_privacy(*args, **kwargs):
            raise LaterFactoryFailure("synthetic after-main construction failure")

        state.patch.setattr(main, "build_privacy_services", fail_privacy)
        expected = 1
    else:
        original_privacy = main.build_privacy_services

        async def privacy(*args, **kwargs):
            result = await original_privacy(*args, **kwargs)
            state.observe_privacy(result)
            return result

        def fail_router(*args, **kwargs):
            raise LaterFactoryFailure("synthetic after-compactor construction failure")

        state.patch.setattr(main, "build_privacy_services", privacy)
        state.patch.setattr(main, "RequestRouter", fail_router)
        expected = 2

    async def build():
        async with AsyncExitStack() as stack:
            return await state.build_application(state.config(url), shutdown=stack, factories=state.factories)

    building = asyncio.create_task(_caught(build()))
    while len(state.postludes) < expected:
        assert not building.done(), "factory ended before acquiring expected cohorts"
        await asyncio.sleep(0)
    for gate in state.postludes:
        await _event(gate.entered)
        assert gate.thread is not state.loop_thread and not gate.thread._handle.is_done()
    state.acquired(expected)
    assert not building.done()
    assert all(raw.is_closed for raw in state.sync.http)
    if state.privacy is not None:
        state.authority_retained()
    for index, gate in enumerate(state.postludes):
        gate.release.set()
        await _event(gate.exited)
        if index + 1 < expected:
            assert not building.done()
            state.authority_retained()
    result = await building
    assert isinstance(result, LaterFactoryFailure)
    state.retired()
    if state.privacy is not None:
        assert state.ledger_closed == [True] and state.privacy._lease is None


async def _held(state, url, route, cancels):
    from src.orchestrator.context_manager import ConversationTurn
    from tests.mocks.memory_sources import memory_source

    stack = AsyncExitStack()
    graph = await state.build_application(state.config(url), shutdown=stack, factories=state.factories)
    state.graph = graph
    state.acquired()
    state.observe_privacy(graph.privacy)
    main, compactor = state.wrappers
    manager = graph.pipeline._orchestrator.context_manager
    assert manager.compactor.reasoner is compactor
    create = Gate(state.sync)
    prepare = Gate(state.sync)
    state.sync.on_create = create
    client_close_entered, client_close_release, client_close_exited = (asyncio.Event() for _ in range(3))

    def no_network(request):
        raise AssertionError("graph SDK leaves must not send requests")

    class Client:
        def __init__(self):
            self.http = httpx.AsyncClient(transport=httpx.MockTransport(no_network))
            self.close_calls = 0
            state.request_clients.append(self)

        def prepare(self, style):
            assert style == compactor.api_style == "chat"
            prepare(self)

        async def models(self):
            return [compactor.model]

        async def complete(self, options, *, style, max_chars):
            return SimpleNamespace(text="synthetic valid compacted summary", tokens=4)

        async def close(self):
            self.close_calls += 1
            client_close_entered.set()
            try:
                await client_close_release.wait()
                await self.http.aclose()
            finally:
                client_close_exited.set()

    state.patch.setattr(compactor, "_new_async_client", lambda base_url: Client())
    primary = state.sync.start(lambda: main.generate("synthetic held primary"), "http-graph-primary")
    await _event(create.entered)
    context = manager.get_or_create("synthetic-compaction-user")
    source = memory_source(now=graph.privacy.ledger.clock())
    assert graph.privacy.ledger.sources_available((source,))
    assert context.sources_available((source,))
    context.conversation_history.append(ConversationTurn("user", "synthetic compaction turn", sources=(source,)))
    compaction = asyncio.create_task(manager._compact_background(context.user_id, keep_recent=0))
    await _event(prepare.entered)
    assert compaction in manager._active_compactions
    closing = asyncio.create_task(stack.aclose() if route == "stack" else graph._stop_services(None, None))
    await state.owner_entered.wait()
    observer = asyncio.create_task(state.owner.aclose())
    for unused in range(cancels):
        await asyncio.sleep(0)
        observer.cancel()
    assert not closing.done() and not observer.done()
    state.authority_retained()
    assert all(sdk.close_calls == 0 for sdk in state.sync.sdks)
    create.release.set()
    await asyncio.to_thread(primary.join)
    assert primary.error is None and primary.value == ANSWER.strip()
    assert primary.thread._handle.is_done()
    assert not closing.done()
    state.authority_retained()
    # Cancellation of compaction's observer cannot acknowledge native prepare.
    prepare.release.set()
    await client_close_entered.wait()
    assert prepare.exited.is_set() and not client_close_exited.is_set()
    assert not closing.done() and all(sdk.close_calls == 0 for sdk in state.sync.sdks)
    state.authority_retained()
    client_close_release.set()
    await client_close_exited.wait()
    for gate in state.postludes:
        await _event(gate.entered)
        assert gate.thread is not state.loop_thread and not gate.thread._handle.is_done()
    assert all(raw.is_closed for raw in state.sync.http)
    assert not closing.done() and not observer.done()
    state.authority_retained()
    # Both cohort closes begin before either actual SDK postlude is released.
    state.postludes[0].release.set()
    await _event(state.postludes[0].exited)
    assert not state.postludes[1].exited.is_set() and not closing.done()
    state.authority_retained()
    state.postludes[1].release.set()
    await closing
    observer_result = await _caught(observer)
    assert isinstance(observer_result, asyncio.CancelledError) if cancels else observer_result is None
    await asyncio.gather(compaction, return_exceptions=True)
    assert not manager._active_compactions
    assert state.request_clients and all(client.close_calls == 1 and client.http.is_closed for client in state.request_clients)
    state.retired()
    await stack.aclose()  # Runtime service stop precedes authority retirement.
    assert state.ledger_closed == [True] and graph.privacy._lease is None


async def _failed(state, url):
    stack = AsyncExitStack()
    graph = await state.build_application(state.config(url), shutdown=stack, factories=state.factories)
    state.acquired()
    state.observe_privacy(graph.privacy)
    failed = state.sync.sdks[1]
    original = failed.close
    attempts = []

    def failed_close():
        attempts.append(True)
        original()
        raise RuntimeError("synthetic graph compactor close failure")

    failed.close = failed_close
    first = await _caught(stack.aclose())
    second = await _caught(state.owner.aclose())
    assert isinstance(first, BaseException) and isinstance(second, BaseException)
    assert attempts == [True] and failed.close_calls == 1 and failed.http.is_closed
    assert state.wrappers[1]._client is failed
    state.authority_retained()
    print("HTTP_GRAPH_CONTRACT=" + json.dumps({"reason": "confirmed", "kind": "failed",
          "actual_sdk_close_returned_error": True, "sdk_close_attempts": len(attempts),
          "writer_lease_retained": True, "ledger_close_calls": len(state.ledger_closed)}), flush=True)
    os._exit(0)  # No retry, reset, forced ledger close or false receipt.


def _isolated(kind, argument=None):
    root = _core["_root"]()
    script = "import json,runpy,sys; m=runpy.run_path(sys.argv[1]); m['_child'](sys.argv[2],json.loads(sys.argv[3]),sys.argv[4])"
    with tempfile.TemporaryDirectory(prefix="kza-http-graph-synthetic-") as temporary:
        result = subprocess.run([sys.executable, "-c", script, str(Path(__file__).resolve()), kind,
                                 json.dumps(argument), temporary], cwd=root, capture_output=True, text=True, timeout=25)
    rows = [json.loads(line.split("=", 1)[1]) for line in result.stdout.splitlines() if line.startswith("HTTP_GRAPH_CONTRACT=")]
    assert result.returncode == 0 and rows and rows[-1]["reason"] == "confirmed", (result.stdout, result.stderr)


def test_factory_registers_both_actual_reasoners_before_load_on_native_workers():
    _isolated("registered")


@pytest.mark.parametrize("stage", ["after-main", "after-compactor"])
def test_partial_factory_stack_waits_real_sdk_close_postludes_and_preserves_authority(stage):
    _isolated("partial", stage)


@pytest.mark.parametrize("route", ["stack", "runtime"])
@pytest.mark.parametrize("cancels", [0, 3])
def test_graph_and_privacy_wait_both_actual_operations_and_sdk_postludes_through_observer_cancellation(route, cancels):
    _isolated("held", [route, cancels])


def test_graph_failed_compactor_retirement_remains_observable_and_keeps_privacy_until_process_exit():
    _isolated("failed")
