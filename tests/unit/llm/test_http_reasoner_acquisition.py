"""Actual HttpReasoner acquisition, synthetic SDK leaves and real HTTPX."""

from types import SimpleNamespace

import httpx
import openai
import pytest

from src.llm import reasoner


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


def test_try_connect_retires_actual_raw_http_after_sdk_constructor_failure(http_leaves):
    state = http_leaves
    state.fail = "constructor"
    wrapper = _wrapper(state)
    with pytest.raises(SyntheticSDKFailure, match="synthetic SDK constructor failure"):
        wrapper._try_connect(wrapper.base_url, wrapper.model)
    assert state.events[:2] == ["http-created", "sdk-constructor-entered"]
    assert len(state.http) == 1 and state.sdks == [] and wrapper._client is None
    assert state.http[0].is_closed  # Genuine RED on the old acquired HTTPX handle.
    assert state.http[0].retirement_calls == 1


def test_load_retires_actual_sdk_and_http_after_model_discovery_failure(http_leaves):
    state = http_leaves
    state.fail = "discovery"
    wrapper = _wrapper(state)
    with pytest.raises(SyntheticSDKFailure, match="synthetic model discovery failure"):
        wrapper.load()
    assert state.events[:3] == ["http-created", "sdk-constructor-entered", "discovery-entered"]
    assert len(state.http) == len(state.sdks) == 1 and wrapper._client is None
    assert state.sdks[0].is_closed and state.http[0].is_closed
    assert state.sdks[0].retirement_calls == state.http[0].retirement_calls == 1
