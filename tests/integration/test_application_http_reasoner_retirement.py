"""Actual HttpReasoner acquisition, synthetic SDK leaves and real HTTPX."""

from types import SimpleNamespace
from contextlib import AsyncExitStack

import httpx
import openai
import pytest

from src.llm import reasoner
from src.main import build_application
from tests.integration.test_application_graph import leaves as application_leaves
from tests.mocks.application_graph import offline_config, offline_home

leaves = application_leaves


class SyntheticSDKFailure(RuntimeError):
    pass


@pytest.fixture
def http_leaves(monkeypatch):
    state = SimpleNamespace(http=[], sdks=[], events=[], fail="", wrappers=[])
    actual_http = httpx.Client

    def unexpected_network(request):
        raise AssertionError("synthetic acquisition must not issue network requests")

    class OwnedHTTP(actual_http):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, transport=httpx.MockTransport(unexpected_network), **kwargs)
            self.retirement_calls = 0
            state.http.append(self)
            state.events.append("http-created")

        def close(self):
            self.retirement_calls += 1
            state.events.append("http-close")
            return super().close()

    class HttpxProxy:
        Client = OwnedHTTP

        def __getattr__(self, name):
            return getattr(httpx, name)

    class SDK:
        def __init__(self, raw):
            self.http, self.retirement_calls = raw, 0
            self.models = SimpleNamespace(list=self.discover)

        def discover(self):
            state.events.append("discovery-entered")
            if state.fail == "discovery":
                raise SyntheticSDKFailure("synthetic model discovery failure")
            return SimpleNamespace(data=[SimpleNamespace(id="synthetic-http")])

        def close(self):
            self.retirement_calls += 1
            state.events.append("sdk-close")
            self.http.close()

        @property
        def is_closed(self):
            return self.http.is_closed

    def sdk(*args, **kwargs):
        raw = kwargs["http_client"]
        assert raw is state.http[-1]
        state.events.append("sdk-constructor-entered")
        if state.fail == "constructor":
            raise SyntheticSDKFailure("synthetic SDK constructor failure")
        value = SDK(raw)
        state.sdks.append(value)
        return value

    monkeypatch.setattr(reasoner, "httpx", HttpxProxy())
    monkeypatch.setattr(reasoner, "_resolve_api_key", lambda *args, **kwargs: "synthetic-token")
    monkeypatch.setattr(openai, "OpenAI", sdk)
    yield state
    # Assertion occurs before fixture cleanup: actual application retirement
    # must close acquired handles while this observer keeps them visible.
    failures = []
    for value in [*state.sdks, *state.http]:
        if not value.is_closed:
            try:
                value.close()
            except BaseException as error:
                failures.append(error)
    if failures:
        raise BaseExceptionGroup("synthetic acquired-handle cleanup failed", failures)


def _wrapper(state):
    value = reasoner.HttpReasoner(base_url="http://synthetic.invalid/v1", model="synthetic-http",
                                destination="local", cloud_consent=False)
    state.wrappers.append(value)
    return value


class LaterFactoryFailure(RuntimeError):
    pass


@pytest.fixture
def graph_http(http_leaves, leaves, monkeypatch):
    import src.vectordb.chroma_sync as index_module

    state = http_leaves
    actual_load = reasoner.HttpReasoner.load

    def load(value):
        state.wrappers.append(value)
        return actual_load(value)

    monkeypatch.setattr(reasoner.HttpReasoner, "load", load)

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
            pass  # Synthetic SDK leaf, never a model or real Chroma backend.

    monkeypatch.setattr(index_module.chromadb, "PersistentClient", ChromaSDK)
    return state


def _config(tmp_path, ha_url):
    config = offline_config(tmp_path, ha_url)
    config["reasoner"] = {"mode": "http", "http_base_url": "http://127.0.0.1:8200/v1",
                          "http_model": "synthetic-main", "destination": "local", "cloud": {"consent": False}}
    config["orchestrator"] = {"enabled": True, "context": {"compaction": {
        "enabled": True, "base_url": "http://127.0.0.1:8201/v1", "model": "synthetic-compactor",
        "destination": "local", "timeout_s": 30,
    }}}
    return config


def _acquired(state, count):
    assert len(state.wrappers) == len(state.http) == len(state.sdks) == count
    assert state.wrappers[0].base_url == "http://127.0.0.1:8200/v1"
    if count == 2:
        assert state.wrappers[1].base_url == "http://127.0.0.1:8201/v1"
        assert state.wrappers[0] is not state.wrappers[1]
    assert all(wrapper._client is sdk for wrapper, sdk in zip(state.wrappers, state.sdks))


def _retired(state):
    # These are the old acquired handles. Missing new lifecycle APIs cannot
    # produce this RED; the original factory/stack must actually retire them.
    assert all(value.is_closed for value in state.http)
    assert all(value.is_closed and value.retirement_calls == 1 for value in state.sdks)
    assert all(value.retirement_calls == 1 for value in state.http)
    assert all(wrapper._client is None for wrapper in state.wrappers)


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["after-main", "after-compactor"])
async def test_actual_factory_retires_acquired_http_reasoners_on_later_failure(graph_http, leaves, tmp_path, monkeypatch, stage):
    import src.main as main_module

    state = graph_http
    if stage == "after-main":
        async def fail_privacy(*args, **kwargs):
            _acquired(state, 1)
            raise LaterFactoryFailure("synthetic failure after main reasoner")

        monkeypatch.setattr(main_module, "build_privacy_services", fail_privacy)
    else:
        def fail_request_router(*args, **kwargs):
            _acquired(state, 2)
            raise LaterFactoryFailure("synthetic failure after compactor reasoner")

        monkeypatch.setattr(main_module, "RequestRouter", fail_request_router)
    async with offline_home() as (url, observations):
        with pytest.raises(LaterFactoryFailure, match="synthetic failure after"):
            async with AsyncExitStack() as shutdown:
                await build_application(_config(tmp_path, url), shutdown=shutdown, factories=leaves)
        assert len(state.sdks) == (1 if stage == "after-main" else 2)
        assert observations["service_calls"] == 0
        _retired(state)


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["stack", "runtime-then-stack"])
async def test_actual_graph_retires_both_main_and_compactor_http_reasoners(graph_http, leaves, tmp_path, path):
    state = graph_http
    async with offline_home() as (url, observations):
        async with AsyncExitStack() as shutdown:
            graph = await build_application(_config(tmp_path, url), shutdown=shutdown, factories=leaves)
            _acquired(state, 2)
            assert not any(client.is_closed for client in state.http)
            if path == "runtime-then-stack":
                await graph._stop_services(None, None)
        assert observations["service_calls"] == 0
        _retired(state)




