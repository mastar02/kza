"""Post-adoption async R task/child-loan contracts; never missing-API REDs."""

import asyncio
import json
from pathlib import Path
import runpy
import subprocess
import sys
import threading
from types import SimpleNamespace

import httpx
import pytest


# Staged/live files travel together. Definitions load; tests/fixtures do not run.
_core = runpy.run_path(str(Path(__file__).with_name("test_http_reasoner_lifecycle.py")))
Probe, Gate = _core["Probe"], _core["Gate"]
ANSWER, PROMPT = _core["ANSWER"], _core["PROMPT"]
_watchdog = _core["_watchdog"]


class AsyncProbe:
    def __init__(self, patch, *, idle_timeout_s=None):
        self.sync = Probe(patch)
        self.wrapper = self.sync.wrapper(idle_timeout_s=idle_timeout_s)
        self.clients, self.streams = [], []
        self.on_construct = self.on_prepare = None
        self.on_models = self.on_complete = self.on_stream = self.on_read = None
        self.on_client_close = self.on_stream_close = None
        owner = self

        class Stream:
            def __init__(self):
                self.close_calls = self.index = 0
                self.parts = ["first " * 50, "second " * 50, "third " * 50]
                owner.streams.append(self)

            def __aiter__(self):
                return self

            async def __anext__(self):
                await owner.invoke("on_read", self)
                if self.index < len(self.parts):
                    part = self.parts[self.index]
                    self.index += 1
                    return SimpleNamespace(text=part, finish_reason=None, empty=False)
                if self.index == len(self.parts):
                    self.index += 1
                    return SimpleNamespace(text="", finish_reason="stop", empty=False)
                raise StopAsyncIteration

            async def close(self):
                self.close_calls += 1
                await owner.invoke("on_stream_close", self)

        class Client:
            def __init__(self):
                self.http = httpx.AsyncClient(transport=httpx.MockTransport(owner.no_network))
                self.close_calls = 0
                owner.clients.append(self)
                if owner.on_construct is not None:
                    owner.on_construct(self)

            def prepare(self, style):
                assert style == "completions"
                if owner.on_prepare is not None:
                    owner.on_prepare(self)

            async def models(self):
                await owner.invoke("on_models", self)
                return ["synthetic-http"]

            async def complete(self, options, *, style, max_chars):
                assert options["model"] == "synthetic-http" and options["stream"] is False
                await owner.invoke("on_complete", self)
                return SimpleNamespace(text=ANSWER, tokens=4)

            async def stream(self, options, *, style, max_chars):
                assert options["model"] == "synthetic-http" and options["stream"] is True
                await owner.invoke("on_stream", self)
                return Stream()

            async def close(self):
                self.close_calls += 1
                await owner.invoke("on_client_close", self)
                await self.http.aclose()

        def construct(base_url):
            assert base_url == self.wrapper.base_url
            return Client()

        patch.setattr(self.wrapper, "_new_async_client", construct)

    @staticmethod
    def no_network(request):
        raise AssertionError("async contract leaves must not issue network requests")

    async def invoke(self, name, value):
        callback = getattr(self, name)
        if callback is not None:
            await callback(value)

    def assert_retired(self):
        assert all(client.close_calls == 1 and client.http.is_closed for client in self.clients)
        assert all(stream.close_calls == 1 for stream in self.streams)
        assert self.sync.http == self.sync.sdks == []  # Request-local != sync cache.


def _root():
    return next(parent for parent in Path(__file__).resolve().parents
                if (parent / "src/llm/reasoner.py").is_file())


def _isolated(name, argument=""):
    script = ("import os,runpy,sys,json\nm=runpy.run_path(sys.argv[1])\ntry:\n"
              " m[sys.argv[2]](sys.argv[3])\nexcept BaseException as error:\n"
              " print('HTTP_CONTRACT='+json.dumps({'reason':'contract_failed','error_type':type(error).__name__}),flush=True)\n"
              " os._exit(1)\nos._exit(0)")
    result = subprocess.run([sys.executable, "-c", script, str(Path(__file__).resolve()), name, argument],
                            cwd=_root(), capture_output=True, text=True, timeout=12)
    receipts = [json.loads(line.split("=", 1)[1]) for line in result.stdout.splitlines()
                if line.startswith("HTTP_CONTRACT=")]
    assert receipts, f"fixture failed before async contract receipt: {result.stdout}\n{result.stderr}"
    assert result.returncode == 0 and receipts[-1]["reason"] == "confirmed", f"{receipts[-1]}\n{result.stderr}"


def _callback_child(stage):
    async def run():
        patch = pytest.MonkeyPatch()
        state = AsyncProbe(patch, idle_timeout_s=5 if stage == "read" else None)
        wrapper = state.wrapper
        rejected = []
        initiating_task = asyncio.current_task()
        finished = _watchdog(lambda: {"stage": stage, "guard_rejections": len(rejected),
                                     "clients_closed": [client.http.is_closed for client in state.clients]})

        def native_guard(value):
            try:
                wrapper.close()
            except RuntimeError as error:
                assert str(error).endswith("_close_from_worker")
                rejected.append(("Thread", threading.current_thread()))
            else:
                raise AssertionError("actual constructor/prepare worker closed itself")

        async def task_guard(value):
            try:
                await wrapper.aclose()
            except RuntimeError as error:
                assert str(error).endswith("_close_from_worker")
                rejected.append(("Task", asyncio.current_task()))
            else:
                raise AssertionError("actual SDK/read/cleanup Task closed its own token")

        if stage in {"construct", "prepare"}:
            setattr(state, "on_" + stage, native_guard)
        else:
            setattr(state, "on_" + stage, task_guard)
        if stage in {"stream", "read", "stream_close"}:
            rows = [row async for row in wrapper.astream(PROMPT)]
            assert "".join(row["token"] for row in rows) == "".join(state.streams[0].parts)
        else:
            assert await wrapper.complete(PROMPT) == ANSWER.strip()
        assert rejected
        assert all(thread is not threading.current_thread() for kind, thread in rejected if kind == "Thread")
        if stage in {"read", "stream_close", "client_close"}:
            assert all(task is not initiating_task for kind, task in rejected if kind == "Task")
        setattr(state, "on_" + stage, None)
        assert wrapper._http_lifecycle._adapter_current() is None
        expected = "".join(state.streams[0].parts).strip() if wrapper.idle_timeout_s else ANSWER.strip()
        assert await wrapper.complete(PROMPT) == expected  # Guard did not seal the configured path.
        await wrapper.aclose()
        state.assert_retired()
        print("HTTP_CONTRACT=" + json.dumps({"reason": "confirmed", "stage": stage,
              "actual_child_guard_visible": True, "request_clients_retired": len(state.clients)}), flush=True)
        finished.set()
        patch.undo()

    asyncio.run(run())


@pytest.mark.parametrize("stage", ["construct", "prepare", "models", "complete", "stream", "read", "stream_close", "client_close"])
def test_actual_async_constructor_read_and_owned_close_callbacks_are_visible_to_guard(stage):
    _isolated("_callback_child", stage)


def _discard_child(unused):
    async def run():
        patch = pytest.MonkeyPatch()
        state = AsyncProbe(patch)
        wrapper = state.wrapper
        gate = Gate(state.sync)
        state.on_prepare = gate
        rejections = []
        finished = _watchdog(lambda: {"phase": "discard", "guard_rejections": len(rejections),
                                     "constructor_exited": gate.exited.is_set()})

        async def guard(client):
            try:
                await wrapper.aclose()
            except RuntimeError as error:
                assert str(error).endswith("_close_from_worker")
                rejections.append(asyncio.current_task())
            else:
                raise AssertionError("discard_client cleanup Task waited on its parent")

        state.on_client_close = guard
        observer = asyncio.create_task(wrapper.complete(PROMPT))
        while not gate.entered.is_set():
            await asyncio.sleep(.001)
        observer.cancel()
        await asyncio.sleep(0)
        observer.cancel()
        gate.release.set()
        with pytest.raises(asyncio.CancelledError):
            await observer
        assert gate.exited.is_set() and len(rejections) == 1
        assert rejections[0] is not observer
        state.on_prepare = state.on_client_close = None
        assert await wrapper.complete(PROMPT) == ANSWER.strip()
        await wrapper.aclose()
        state.assert_retired()
        print("HTTP_CONTRACT=" + json.dumps({"reason": "confirmed", "discard_client_guard": True,
              "native_constructor_exited": True, "repeated_observer_cancel": True}), flush=True)
        finished.set()
        patch.undo()

    asyncio.run(run())


def test_cancelled_constructor_discard_child_keeps_guard_and_real_cleanup_through_repeated_cancel():
    _isolated("_discard_child")


def _migration_child(cancel_text):
    async def run():
        patch = pytest.MonkeyPatch()
        state = AsyncProbe(patch)
        wrapper = state.wrapper
        stream = wrapper.astream(PROMPT)
        rows, rejected = [], []
        finished = _watchdog(lambda: {"phase": "async_migration", "rejected": len(rejected),
                                     "source_close_calls": [source.close_calls for source in state.streams]})
        rows.append(await anext(stream))  # TaskA, then public yield.
        assert wrapper._http_lifecycle._adapter_current() is None
        try:
            await wrapper.aclose()
        except RuntimeError as error:
            assert str(error).endswith("_close_from_worker")
            rejected.append("A")
        else:
            raise AssertionError("latest async consumer A was not guarded")

        async def consume_b():
            rows.append(await anext(stream))
            assert wrapper._http_lifecycle._adapter_current() is None
            try:
                await wrapper.aclose()
            except RuntimeError as error:
                assert str(error).endswith("_close_from_worker")
                rejected.append("B")
            else:
                raise AssertionError("latest async consumer B was not guarded")

        task_b = asyncio.create_task(consume_b())
        await task_b
        assert rejected == ["A", "B"]
        began = asyncio.Event()
        original_begin = wrapper.begin_close

        def observe_begin():
            try:
                return original_begin()
            finally:
                began.set()

        patch.setattr(wrapper, "begin_close", observe_begin)
        # Unrelated TaskC shares the loop Thread and ordinary inherited Context.
        # A suspended stream must not leave Thread/TLS/ContextVar ownership.
        closing = asyncio.create_task(wrapper.aclose())
        await began.wait()
        for unused in range(int(cancel_text)):
            closing.cancel()
            await asyncio.sleep(0)
        await asyncio.create_task(stream.aclose())  # Independent TaskD cleanup.
        if int(cancel_text):
            with pytest.raises(asyncio.CancelledError):
                await closing
        else:
            await closing
        state.assert_retired()
        assert rows[0]["token"] and rows[0]["text"] == rows[0]["token"]
        print("HTTP_CONTRACT=" + json.dumps({"reason": "confirmed", "Task_A_B_migration": True,
              "unrelated_Task_C_allowed": True, "Task_D_cleanup_exited": True,
              "observer_cancellations": int(cancel_text)}), flush=True)
        finished.set()
        patch.undo()

    asyncio.run(run())


@pytest.mark.parametrize("cancels", [0, 3])
def test_async_stream_migration_and_unrelated_same_loop_closer_keep_real_task_identity(cancels):
    _isolated("_migration_child", str(cancels))
