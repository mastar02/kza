"""Post-adoption R contracts; API absence is NOT baseline RED evidence.

The immutable six old-path REDs live separately in r/baseline-tests.py and
r/factory-baseline-tests.py. All SDK/network leaves below are synthetic.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import httpx
import openai
import pytest

from src.llm import reasoner


ANSWER = "synthetic answer " * 20
PROMPT = "synthetic lifecycle input"


class Call:
    def __init__(self, function, name="http-contract-caller"):
        self.value = self.error = None

        def run():
            try:
                self.value = function()
            except BaseException as error:
                self.error = error

        self.thread = threading.Thread(target=run, name=name, daemon=True)
        self.thread.start()

    def join(self, timeout=3):
        self.thread._handle.join(timeout)
        assert self.thread._handle.is_done(), "actual test native caller did not exit"
        return self


class Gate:
    def __init__(self, probe):
        self.entered, self.release, self.exited = (threading.Event() for _ in range(3))
        probe.release_gates.append(self.release)

    def __call__(self, *args):
        self.entered.set()
        try:
            self.release.wait()
        finally:
            self.exited.set()


class Probe:
    def __init__(self, patch):
        self.patch = patch
        self.http, self.sdks, self.wrappers, self.streams, self.calls = [], [], [], [], []
        self.release_gates = []
        self.on_http = self.on_sdk = self.on_discover = self.on_create = None
        self.on_parse = self.on_close = self.on_transport_close = self.on_stream_close = None
        self.on_next = None
        actual_http = httpx.Client
        probe = self

        class Transport(httpx.BaseTransport):
            def handle_request(self, request):
                raise AssertionError("contract leaves must not make real network requests")

            def close(self):
                probe.invoke("on_transport_close", self.client)

        class HTTP(actual_http):
            def __init__(self, *args, **kwargs):
                transport = Transport()
                super().__init__(*args, transport=transport, **kwargs)
                transport.client = self
                self.close_calls = 0
                probe.http.append(self)
                probe.invoke("on_http", self)

            def close(self):
                self.close_calls += 1
                return super().close()

        class HttpxProxy:
            Client = HTTP

            def __getattr__(self, name):
                return getattr(httpx, name)

        class Choice:
            def __init__(self, sdk):
                self.sdk = sdk

            @property
            def text(self):
                probe.invoke("on_parse", self.sdk)
                return ANSWER

            @property
            def message(self):
                probe.invoke("on_parse", self.sdk)
                return SimpleNamespace(content=ANSWER)

        class Stream:
            def __init__(self, sdk):
                self.sdk = sdk
                self.close_calls = self.iter_calls = self.iterator_exits = 0
                self.parts = ["first " * 50, "second " * 50, "third " * 50]
                probe.streams.append(self)

            def __iter__(self):
                self.iter_calls += 1

                def iterate():
                    try:
                        for part in self.parts:
                            probe.invoke("on_next", self)
                            yield SimpleNamespace(choices=[SimpleNamespace(text=part, finish_reason=None)])
                    finally:
                        self.iterator_exits += 1

                return iterate()  # Distinct real Python iterator, like SDK Stream.

            def close(self):
                self.close_calls += 1
                probe.invoke("on_stream_close", self)

        class SDK:
            def __init__(self, raw):
                self.http, self.close_calls = raw, 0
                self.models = SimpleNamespace(list=self.discover)
                self.completions = SimpleNamespace(create=self.create)
                self.chat = SimpleNamespace(completions=self.completions)

            def discover(self):
                probe.invoke("on_discover", self)
                return SimpleNamespace(data=[SimpleNamespace(id="synthetic-http")])

            def create(self, **options):
                probe.invoke("on_create", self)
                if options.get("stream"):
                    return Stream(self)
                return SimpleNamespace(choices=[Choice(self)], usage=SimpleNamespace(completion_tokens=4))

            def close(self):
                self.close_calls += 1
                probe.invoke("on_close", self)
                self.http.close()

            @property
            def is_closed(self):
                return self.http.is_closed

        def construct(*args, **kwargs):
            probe.invoke("on_sdk", kwargs["http_client"])
            value = SDK(kwargs["http_client"])
            probe.sdks.append(value)
            return value

        self.sdk_type = SDK
        patch.setattr(reasoner, "httpx", HttpxProxy())
        patch.setattr(reasoner, "_resolve_api_key", lambda *args, **kwargs: "synthetic-token")
        patch.setattr(openai, "OpenAI", construct)

    def invoke(self, name, value):
        callback = getattr(self, name)
        if callback is not None:
            return callback(value)

    def wrapper(self, **options):
        value = reasoner.HttpReasoner(base_url="http://synthetic.invalid/v1", model="synthetic-http",
                                    destination="local", cloud_consent=False, **options)
        self.wrappers.append(value)
        return value

    def start(self, function, name="http-contract-caller"):
        call = Call(function, name)
        self.calls.append(call)
        return call

    def release(self):
        for event in self.release_gates:
            event.set()

    def cleanup(self):
        self.release()
        failures = []
        for call in self.calls:
            try:
                call.join()
            except BaseException as error:
                failures.append(error)
        for wrapper in self.wrappers:
            try:
                wrapper.close()
            except BaseException as error:
                failures.append(error)
        if failures:
            raise BaseExceptionGroup("synthetic HTTP contract cleanup failed", failures)


@pytest.fixture
def probe(monkeypatch):
    value = Probe(monkeypatch)
    yield value
    value.cleanup()


def observe_sync_wait(patch, wrapper):
    lifecycle = wrapper._http_lifecycle
    original = lifecycle._adapter_condition
    waiting = threading.Event()

    class Condition:
        def __enter__(self):
            return original.__enter__()

        def __exit__(self, *args):
            return original.__exit__(*args)

        def __getattr__(self, name):
            return getattr(original, name)

        def wait(self, *args, **kwargs):
            waiting.set()
            return original.wait(*args, **kwargs)

    patch.setattr(lifecycle, "_adapter_condition", Condition())
    return waiting


@pytest.mark.parametrize("stage", ["http", "sdk", "discover", "create", "parse", "close"])
def test_sync_close_observes_actual_sdk_boundary_before_retiring_handles(probe, stage):
    wrapper = probe.wrapper()
    if stage in {"create", "parse", "close"}:
        wrapper.load()
    gate = Gate(probe)
    setattr(probe, "on_" + stage, gate)
    waiting = observe_sync_wait(probe.patch, wrapper)
    if stage == "close":
        first = probe.start(wrapper.close, "http-first-close")
    else:
        operation = (lambda: wrapper.generate(PROMPT)) if stage in {"create", "parse"} else wrapper.load
        first = probe.start(operation)
    closer = None
    try:
        assert gate.entered.wait(3), "actual SDK boundary never entered"
        closer = probe.start(wrapper.close, "http-concurrent-close")
        assert waiting.wait(3), "actual synchronous drain did not observe live work"
        assert not closer.thread._handle.is_done() and not gate.exited.is_set()
        assert not any(value.is_closed for value in probe.http)
    finally:
        gate.release.set()
        first.join()
        if closer is not None:
            closer.join()
    assert first.error is None and closer.error is None
    if stage in {"create", "parse"}:
        assert first.value == ANSWER.strip()
    assert all(value.is_closed and value.close_calls == 1 for value in probe.http)
    assert all(value.close_calls == 1 for value in probe.sdks)
    assert wrapper._client is None


@pytest.mark.parametrize("stage", ["http", "sdk", "discover", "create", "parse"])
def test_reentrant_sdk_callback_rejects_close_before_sealing_admitted_work(stage):
    _run_isolated("_reentrant_sdk_child", stage)


def _reentrant_sdk_child(stage):
    patch = pytest.MonkeyPatch()
    probe = Probe(patch)
    wrapper = probe.wrapper()
    if stage in {"create", "parse"}:
        wrapper.load()
    rejections = []
    finished = _watchdog(lambda: {"stage": stage, "rejected": len(rejections)})

    def callback(value):
        try:
            wrapper.close()
        except RuntimeError as error:
            assert str(error).endswith("_close_from_worker")
            rejections.append(threading.current_thread())
        else:
            raise AssertionError("SDK callback closed its own operation")

    setattr(probe, "on_" + stage, callback)
    call = probe.start(lambda: wrapper.generate(PROMPT))
    call.join(6)  # Watchdog supplies the failure receipt if reentry self-waits.
    assert call.error is None and call.value == ANSWER.strip()
    assert rejections and all(thread is call.thread for thread in rejections)
    setattr(probe, "on_" + stage, None)
    assert wrapper.generate(PROMPT) == ANSWER.strip()  # Failed guard did not seal.
    wrapper.close()
    assert all(value.is_closed for value in probe.http)
    print("HTTP_CONTRACT=" + json.dumps({"reason": "confirmed", "stage": stage,
          "actual_native_callback_exited": True, "guard_rejections": len(rejections)}), flush=True)
    finished.set()
    patch.undo()


def test_explicit_load_reconnects_and_retires_every_owned_client(probe):
    wrapper = probe.wrapper()
    wrapper.load()
    first = wrapper._client
    wrapper.load()
    second = wrapper._client
    assert first is not second and len(probe.sdks) == 2
    assert not any(value.is_closed for value in probe.http)
    wrapper.close()
    assert all(value.is_closed and value.close_calls == 1 for value in probe.http)
    assert [value.close_calls for value in probe.sdks] == [1, 1]


def test_concurrent_explicit_loads_do_not_serialize_or_close_previous_publication(probe):
    wrapper = probe.wrapper()
    first_gate, second_gate = Gate(probe), Gate(probe)

    def discovery(sdk):
        (first_gate if sdk is probe.sdks[0] else second_gate)()

    probe.on_discover = discovery
    first = probe.start(wrapper.load, "http-load-A")
    second = None
    try:
        assert first_gate.entered.wait(3)
        second = probe.start(wrapper.load, "http-load-B")
        assert second_gate.entered.wait(3), "second load serialized behind first SDK callback"
        second_gate.release.set()
        second.join()
        assert second.error is None and wrapper._client is probe.sdks[1]
        assert not any(value.is_closed for value in probe.http)
    finally:
        first_gate.release.set()
        second_gate.release.set()
        first.join()
        if second is not None:
            second.join()
    assert first.error is None and wrapper._client is probe.sdks[0]
    wrapper.close()
    assert all(value.is_closed and value.close_calls == 1 for value in probe.http)


def test_swapped_borrowed_sdk_is_not_closed_or_cleared_as_an_owned_identity(probe):
    wrapper = probe.wrapper()
    wrapper.load()
    owned = wrapper._client
    raw = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200)))
    borrowed = probe.sdk_type(raw)
    try:
        wrapper._client = borrowed
        wrapper.close()
        assert owned.close_calls == 1 and owned.is_closed
        assert borrowed.close_calls == 0 and not raw.is_closed
        assert wrapper._client is borrowed
    finally:
        borrowed.close()  # This test is the borrowed peer's actual owner.


@pytest.mark.parametrize("exhaust", [False, True])
def test_sync_stream_keeps_public_payload_and_retires_distinct_iterator_and_source(probe, exhaust):
    wrapper = probe.wrapper()
    stream = wrapper.generate_stream(PROMPT)
    assert probe.http == []  # Preserve lazy first-pull load.
    first = next(stream)
    assert first["token"] and first["text"] == first["token"]
    assert wrapper._http_lifecycle._adapter_current() is None
    source = probe.streams[0]
    if exhaust:
        rows = [first, *list(stream)]
        assert "".join(row["token"] for row in rows) == "".join(source.parts)
        assert rows[-1]["text"] == "".join(source.parts)
        assert [row["token_count"] for row in rows] == list(range(1, len(rows) + 1))
    else:
        stream.close()
    assert source.iter_calls == source.iterator_exits == source.close_calls == 1
    wrapper.close()
    assert probe.sdks[0].close_calls == probe.http[0].close_calls == 1


def _root():
    return next(parent for parent in Path(__file__).resolve().parents
                if (parent / "src/llm/reasoner.py").is_file())


def _run_isolated(name, argument=""):
    script = "import os,runpy,sys; m=runpy.run_path(sys.argv[1]); m[sys.argv[2]](sys.argv[3]); os._exit(0)"
    result = subprocess.run([sys.executable, "-c", script, str(Path(__file__).resolve()), name, argument],
                            cwd=_root(), capture_output=True, text=True, timeout=12)
    receipts = [json.loads(line.split("=", 1)[1]) for line in result.stdout.splitlines()
                if line.startswith("HTTP_CONTRACT=")]
    assert receipts, f"fixture failed before contract receipt: {result.stdout}\n{result.stderr}"
    assert result.returncode == 0 and receipts[-1]["reason"] == "confirmed", f"{receipts[-1]}\n{result.stderr}"


def _watchdog(snapshot):
    finished = threading.Event()

    def wait():
        if not finished.wait(5):
            import os
            print("HTTP_CONTRACT=" + json.dumps({"reason": "actual_owner_or_consumer_hung", **snapshot()}), flush=True)
            os._exit(1)  # Do not rescue lifecycle flags or a live owner.

    threading.Thread(target=wait, name="http-contract-watchdog", daemon=True).start()
    return finished


def _sync_consumer_child(mode):
    patch = pytest.MonkeyPatch()
    probe = Probe(patch)
    wrapper = probe.wrapper()
    stream = wrapper.generate_stream(PROMPT)
    rows, rejected, threads = [], [], []
    finished = _watchdog(lambda: {"mode": mode, "rejected": len(rejected),
                                 "sdk_close_calls": [sdk.close_calls for sdk in probe.sdks]})

    def pull_and_close():
        rows.append(next(stream))
        try:
            wrapper.close()
        except RuntimeError as error:
            assert str(error).endswith("_close_from_worker")
            rejected.append(True)
        else:
            raise AssertionError("suspended latest consumer waited on itself")

    if mode == "migrate":
        first = probe.start(lambda: rows.append(next(stream)), "http-stream-A")
        first.join()
        assert first.error is None
        threads.append(first.thread)
    consumer = probe.start(pull_and_close, "http-stream-B")
    consumer.join(6)
    threads.append(consumer.thread)
    assert consumer.error is None and rejected == [True]
    assert not probe.http[0].is_closed
    assert wrapper.generate(PROMPT) == ANSWER.strip()
    stream.close()  # Independent real current Thread retires the stream.
    wrapper.close()
    assert all(thread._handle.is_done() for thread in threads)
    assert probe.streams[0].iterator_exits == probe.streams[0].close_calls == 1
    assert probe.http[0].is_closed
    print("HTTP_CONTRACT=" + json.dumps({"reason": "confirmed", "mode": mode,
          "actual_native_consumers_exited": True, "source_and_iterator_retired": True}), flush=True)
    finished.set()
    patch.undo()


@pytest.mark.parametrize("mode", ["same", "migrate"])
def test_sync_stream_latest_consumer_guard_is_finite_and_tracks_migration(mode):
    _run_isolated("_sync_consumer_child", mode)


def _close_callback_child(unused):
    patch = pytest.MonkeyPatch()
    probe = Probe(patch)
    wrapper = probe.wrapper()
    wrapper.load()
    rejected = []
    finished = _watchdog(lambda: {"sdk_close_calls": probe.sdks[0].close_calls,
                                 "rejected": len(rejected)})

    def callback(sdk):
        try:
            wrapper.close()
        except RuntimeError as error:
            assert str(error).endswith(("_close_from_worker", "_close_from_owner"))
            rejected.append(threading.current_thread())
        else:
            raise AssertionError("actual SDK disposer recursively closed its owner")

    probe.on_close = callback
    closing = probe.start(wrapper.close)
    closing.join(6)
    assert closing.error is None and len(rejected) == 1
    assert probe.sdks[0].close_calls == probe.http[0].close_calls == 1
    assert probe.http[0].is_closed
    print("HTTP_CONTRACT=" + json.dumps({"reason": "confirmed", "sdk_disposer_guard": True,
          "actual_native_call_exited": True}), flush=True)
    finished.set()
    patch.undo()


def test_actual_sdk_disposer_callback_cannot_wait_on_its_own_reasoner():
    _run_isolated("_close_callback_child")


def _failed_close_child(unused):
    patch = pytest.MonkeyPatch()
    probe = Probe(patch)
    wrapper = probe.wrapper()
    wrapper.load()
    sdk = probe.sdks[0]
    physical_calls = []
    finished = _watchdog(lambda: {"sdk_close_calls": sdk.close_calls,
                                 "physical_close_calls": len(physical_calls)})

    def failed_transport(raw):
        assert raw.is_closed  # Real HTTPX publishes CLOSED before transport.close.
        physical_calls.append(True)
        raise RuntimeError("synthetic real transport close failure")

    probe.on_transport_close = failed_transport
    failures = []
    for attempt in range(2):
        try:
            wrapper.close()
        except BaseException as error:
            failures.append(type(error).__name__)
        else:
            raise AssertionError("CLOSED flag or retry fabricated successful retirement")
    assert len(failures) == 2 and sdk.close_calls == 1
    assert physical_calls == [True] and probe.http[0].close_calls == 1
    assert probe.http[0].is_closed and wrapper._client is sdk
    # Failed owner/model/bundle deliberately remain strong until OS process exit.
    print("HTTP_CONTRACT=" + json.dumps({"reason": "confirmed", "physical_calls": 1,
          "closed_flag_not_ack": True, "failed_owner_retained": True}), flush=True)
    finished.set()
    patch.undo()


def test_closed_flag_before_real_transport_failure_is_terminal_without_disposal_retry():
    _run_isolated("_failed_close_child")


def _single_pool_stream_child(unused):
    async def run():
        patch = pytest.MonkeyPatch()
        probe = Probe(patch)
        wrapper = probe.wrapper()
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="http-contract-only-worker")
        asyncio.get_running_loop().set_default_executor(executor)
        stream = wrapper.generate_stream(PROMPT)
        first = await asyncio.to_thread(next, stream)
        assert first["token"]
        close_entered, close_exited = threading.Event(), threading.Event()
        finished = _watchdog(lambda: {"stream_close_entered": close_entered.is_set(),
                                     "stream_close_exited": close_exited.is_set(),
                                     "sdk_close_calls": [sdk.close_calls for sdk in probe.sdks]})
        closing = wrapper.begin_close()
        assert not closing.done()

        def close_stream():
            close_entered.set()
            try:
                stream.close()
            finally:
                close_exited.set()

        # A blocking work-wait job submitted by begin_close would occupy this
        # sole worker ahead of the very stream cleanup needed by that wait.
        await asyncio.to_thread(close_stream)
        await asyncio.shield(closing)
        assert close_exited.is_set()
        assert probe.streams[0].iterator_exits == probe.streams[0].close_calls == 1
        assert probe.http[0].is_closed
        workers = tuple(executor._threads)
        executor.shutdown(wait=True)  # The TEST owns this executor, not wrapper.
        assert len(workers) == 1 and workers[0]._handle.is_done()
        print("HTTP_CONTRACT=" + json.dumps({"reason": "confirmed", "one_worker": True,
              "actual_stream_cleanup_exited": True, "test_owned_executor_exited": True}), flush=True)
        finished.set()
        patch.undo()

    asyncio.run(run())


def test_async_begin_close_keeps_only_default_worker_available_for_sync_stream_cleanup():
    _run_isolated("_single_pool_stream_child")


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["close", "wait_closed"])
async def test_sync_close_on_running_loop_rejects_before_any_terminal_seal(probe, method):
    wrapper = probe.wrapper()
    with pytest.raises(RuntimeError):
        getattr(wrapper, method)()
    assert await asyncio.to_thread(wrapper.generate, PROMPT) == ANSWER.strip()
    await wrapper.aclose()
    assert probe.http[0].is_closed
