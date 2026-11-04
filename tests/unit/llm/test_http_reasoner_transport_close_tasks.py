"""R GREEN contracts for actual HTTPGenerationClient nested close Tasks.

No new hook is called by this fixture. The actual _new_async_client path must
install its own production hook on the actual HTTPGenerationClient instance.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import threading
from types import SimpleNamespace

import httpx
import pytest

_core_path = Path(__file__).with_name("test_http_reasoner_lifecycle.py")
if not _core_path.exists():
    _core_path = Path(__file__).resolve().parents[2] / "tests/unit/llm/test_http_reasoner_lifecycle.py"
_core = runpy.run_path(str(_core_path))
Probe = _core["Probe"]


def _child(phase):
    async def run():
        from src.llm import reasoner
        from src.llm.http_generation import HTTPGenerationClient

        patch = pytest.MonkeyPatch()
        sync = Probe(patch)
        wrapper = sync.wrapper()
        state = SimpleNamespace(enabled=True, requests=[], clients=[], bodies=[], responses=[], guarded=[], operation=None)
        finished = threading.Event()

        def watchdog():
            if not finished.wait(15):
                print("HTTP_CLOSE_TASK_CONTRACT=" + json.dumps({"reason": "close_callback_self_wait",
                    "phase": phase, "guards": len(state.guarded),
                    "body_close_entered": [body.entered for body in state.bodies],
                    "body_close_exited": [body.exited for body in state.bodies]}), flush=True)
                os._exit(1)

        threading.Thread(target=watchdog, daemon=True, name="http-transport-contract-watchdog").start()

        async def attempt(role):
            task = asyncio.current_task()
            assert task is not state.operation, "callback did not run in a real inner close Task"
            try:
                await wrapper.aclose()
            except RuntimeError as error:
                assert str(error).endswith("_close_from_worker")
            else:
                raise AssertionError("transport close callback elected its own instance drain")
            assert not wrapper._http_lifecycle._adapter_closing
            state.guarded.append((role, task))

        class Body(httpx.AsyncByteStream):
            def __init__(self, chunks, target):
                self.chunks, self.target = chunks, target
                self.close_calls = 0
                self.entered = self.exited = False
                state.bodies.append(self)

            async def __aiter__(self):
                for chunk in self.chunks:
                    yield chunk

            async def aclose(self):
                self.close_calls += 1
                self.entered = True
                try:
                    if self.target:
                        await attempt("raw-byte-stream")
                finally:
                    self.exited = True

        class Response(httpx.Response):
            async def aclose(self):
                if not self.is_closed and self.guard_response:
                    await attempt("response")
                return await super().aclose()

        def handle(request):
            assert request.url.host == "synthetic.invalid"
            assert request.url.path in {"/v1/models", "/v1/completions"}
            state.requests.append((request.method, request.url.path))
            is_models = request.method == "GET"
            target = state.enabled and (is_models if phase == "models-json" else not is_models)
            status, headers = 200, {"content-type": "application/json"}
            if is_models:
                chunks = [json.dumps({"data": [{"id": "synthetic-http"}]}).encode()]
            elif phase == "stream-body":
                headers = {"content-type": "text/event-stream"}
                chunks = [("data: " + json.dumps({"choices": [{"index": 0, "text": "stream " * 50,
                            "finish_reason": None}]}) + "\n\n").encode(),
                          ("data: " + json.dumps({"choices": [{"index": 0, "text": "",
                            "finish_reason": "stop"}]}) + "\n\n").encode(), b"data: [DONE]\n\n"]
            else:
                chunks = [json.dumps({"choices": [{"index": 0, "text": " synthetic completion ",
                           "finish_reason": "stop"}], "usage": {"completion_tokens": 4}}).encode()]
            if target and phase == "send-status":
                status = 503
            if target and phase == "send-encoding":
                headers["content-encoding"] = "gzip"
            body = Body(chunks, target)
            response = Response(status, headers=headers, stream=body)
            response.guard_response = target and phase in {"send-status", "send-encoding", "stream-content-type"}
            state.responses.append(response)
            return response

        class Client(HTTPGenerationClient):
            def __init__(self, *args, **kwargs):
                # Preserve the actual reasoner factory and forward optional
                # lifecycle constructor arguments without inventing a hook.
                assert "http_client" not in kwargs
                raw = httpx.AsyncClient(transport=httpx.MockTransport(handle))
                super().__init__(*args, http_client=raw, **kwargs)
                self.raw, self.close_calls = raw, 0
                state.clients.append(self)

            async def close(self):
                self.close_calls += 1
                await super().close()

        patch.setattr(reasoner, "HTTPGenerationClient", Client)
        state.operation = asyncio.current_task()
        failure = None
        if phase in {"stream-body", "stream-content-type"}:
            stream = wrapper.astream("synthetic transport close request")
            try:
                first = await anext(stream)
                assert phase == "stream-body" and first["token"]
                assert wrapper._http_lifecycle._adapter_current() is None
            except BaseException as error:
                failure = error
            finally:
                await stream.aclose()
        else:
            try:
                result = await wrapper.complete("synthetic transport close request")
                assert result == "synthetic completion"
            except BaseException as error:
                failure = error
        if phase == "send-status":
            assert isinstance(failure, httpx.HTTPStatusError)
        elif phase == "send-encoding":
            assert isinstance(failure, ValueError) and str(failure) == "unsupported_generation_encoding"
        elif phase == "stream-content-type":
            assert isinstance(failure, ValueError) and str(failure) == "invalid_generation_content_type"
        else:
            assert failure is None
        assert state.requests == [("GET", "/v1/models"), ("POST", "/v1/completions")]
        roles = {role for role, task in state.guarded}
        assert "raw-byte-stream" in roles
        if phase in {"send-status", "send-encoding", "stream-content-type"}:
            assert roles == {"response", "raw-byte-stream"}
        assert all(task is not state.operation and task.done() for role, task in state.guarded)
        assert all(body.close_calls == 1 and body.exited for body in state.bodies)
        assert all(response.is_closed for response in state.responses)
        assert all(isinstance(client, HTTPGenerationClient) and client.close_calls == 1 and client.raw.is_closed
                   for client in state.clients)
        assert not wrapper._http_lifecycle._adapter_closing
        # A finite rejection must leave this actual instance usable.
        state.enabled = False
        if phase == "stream-body":
            # Keep the same public streaming wire shape for this fixture.
            follow = wrapper.astream("synthetic followup after close callback guard")
            assert (await anext(follow))["token"]
            await follow.aclose()
        else:
            assert await wrapper.complete("synthetic followup after close callback guard") == "synthetic completion"
        await wrapper.aclose()
        assert all(client.close_calls == 1 and client.raw.is_closed for client in state.clients)
        assert sync.http == sync.sdks == []  # Async request clients are independent.
        print("HTTP_CLOSE_TASK_CONTRACT=" + json.dumps({"reason": "confirmed", "phase": phase,
            "real_http_generation_client": True, "actual_inner_close_tasks": len(state.guarded),
            "guarded_roles": sorted(roles), "all_real_httpx_clients_closed": True}), flush=True)
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
        print("HTTP_CLOSE_TASK_CONTRACT=" + json.dumps({"reason": "assertion_or_lifecycle_failure",
              "error_type": type(error).__name__, "phase": phase, "frames": frames}), flush=True)
        os._exit(1)


@pytest.mark.parametrize("phase", ["models-json", "complete-json", "send-status", "send-encoding",
                                  "stream-content-type", "stream-body"])
def test_real_http_generation_transport_close_tasks_cannot_close_their_own_reasoner(phase):
    script = "import runpy,sys; m=runpy.run_path(sys.argv[1]); m['_child'](sys.argv[2])"
    result = subprocess.run([sys.executable, "-c", script, str(Path(__file__).resolve()), phase],
                            cwd=_core["_root"](), capture_output=True, text=True, timeout=22)
    rows = [json.loads(line.split("=", 1)[1]) for line in result.stdout.splitlines() if line.startswith("HTTP_CLOSE_TASK_CONTRACT=")]
    assert result.returncode == 0 and rows and rows[-1]["reason"] == "confirmed", (result.stdout, result.stderr)
