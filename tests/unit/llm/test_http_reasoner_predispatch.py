"""Exact R prepare/discard dispatch faults, isolated without cleanup rescue."""

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
import weakref

import httpx
import pytest

_core_path = Path(__file__).with_name("test_http_reasoner_lifecycle.py")
if not _core_path.exists():
    _core_path = Path(__file__).resolve().parents[2] / "tests/unit/llm/test_http_reasoner_lifecycle.py"
_core = runpy.run_path(str(_core_path))


async def _caught(awaitable):
    try:
        return await awaitable
    except BaseException as error:
        return error


def _leaves(error):
    if isinstance(error, BaseExceptionGroup):
        return [leaf for child in error.exceptions for leaf in _leaves(child)]
    return [error]


def _child(stage, dispatch):
    async def run():
        from src.llm import http_reasoner_lifecycle as implementation
        from src.llm import reasoner
        from src.llm.http_generation import HTTPGenerationClient

        patch = pytest.MonkeyPatch()
        sync = _core["Probe"](patch)
        wrapper = sync.wrapper()
        owner = wrapper._http_lifecycle
        state = SimpleNamespace(injected=False, reservation=None, holder=None, lease=None,
            pending=None, scheduled=None, preparing=None, client_refs=[], raw_refs=[],
            close_calls=0, requests=0, native_thread=None)
        entered, release, exited = (threading.Event() for _ in range(3))
        finished = threading.Event()
        injected = KeyboardInterrupt()
        target = "_Initialization._prepare" if stage == "initialization" else "HttpReasonerLifecycle._discard"

        def watchdog():
            if not finished.wait(15):
                print("HTTP_PREDISPATCH_CONTRACT=" + json.dumps({"reason": "owner_or_actor_hung",
                    "stage": stage, "dispatch": dispatch, "fault_fired": state.injected,
                    "native_entered": entered.is_set(), "native_exited": exited.is_set(),
                    "physical_client_close_calls": state.close_calls}), flush=True)
                os._exit(1)

        threading.Thread(target=watchdog, daemon=True, name="http-dispatch-contract-watchdog").start()

        def no_network(request):
            state.requests += 1
            raise AssertionError("dispatch contracts must not send model requests")

        class Client(HTTPGenerationClient):
            def __init__(self, *args, **kwargs):
                raw = httpx.AsyncClient(transport=httpx.MockTransport(no_network))
                super().__init__(*args, http_client=raw, **kwargs)
                state.client_refs.append(weakref.ref(self))
                state.raw_refs.append(weakref.ref(raw))

            def prepare(self, style):
                super().prepare(style)
                state.native_thread = threading.current_thread()
                entered.set()
                try:
                    release.wait()
                finally:
                    exited.set()

            async def close(self):
                state.close_calls += 1
                await super().close()

        patch.setattr(reasoner, "HTTPGenerationClient", Client)
        actual_create_task = asyncio.create_task

        class AsyncioProxy:
            def __getattr__(self, name):
                return getattr(asyncio, name)

            def create_task(self, pending, *args, **kwargs):
                code = getattr(pending, "cr_code", None)
                name = getattr(code, "co_qualname", None)
                if name != target or state.injected:
                    task = actual_create_task(pending, *args, **kwargs)
                    if name == "_Initialization._prepare":
                        state.preparing = task
                    return task
                frame = pending.cr_frame.f_locals
                holder = frame["self"] if stage == "initialization" else frame["initialization"]
                lease = holder.lease if stage == "initialization" else frame["lease"]
                reservation = frame["reservation"]
                # Inspect the REAL published lease/reservation before invoking
                # the actual dispatch factory. No detached metadata aliases.
                assert holder in owner._initializing
                assert lease in owner._adapter_leases and lease.owner is owner
                assert reservation in lease.entries
                assert reservation.active and not reservation.started and not reservation.finished
                assert reservation.task is None and reservation.awaitable is pending
                state.injected = True
                state.holder, state.lease, state.reservation, state.pending = holder, lease, reservation, pending
                if dispatch == "after":
                    state.scheduled = actual_create_task(pending, *args, **kwargs)
                    if stage == "initialization":
                        state.preparing = state.scheduled
                raise injected

        # Replaces only reasoner's module reference, never global asyncio or
        # fixture Task creation. The helper still uses the real event loop.
        patch.setattr(reasoner, "asyncio", AsyncioProxy())
        operation = asyncio.create_task(_caught(wrapper.complete("synthetic exact dispatch request")))
        if stage == "discard":
            assert await asyncio.to_thread(entered.wait, 5), "actual native preparation never entered"
            assert state.client_refs and state.client_refs[0]() is not None
            assert state.native_thread is not threading.current_thread()
            assert not state.native_thread._handle.is_done()
            operation.cancel()  # Actual cancellation selects the discard path.
        result = await operation
        assert state.injected and result is injected
        expected = "http_reasoner_prepare_dispatch_unconfirmed" if stage == "initialization" else "http_reasoner_discard_dispatch_unconfirmed"
        assert owner._dispatch_uncertain and owner in implementation._UNCERTAIN_HTTP_OWNERS
        assert state.holder in owner._initializing
        assert state.scheduled is not None if dispatch == "after" else state.scheduled is None

        async def rejected_close():
            closing = owner.begin_close()
            failure = await _caught(wrapper.aclose())
            leaves = _leaves(failure)
            assert isinstance(failure, BaseException) and all(isinstance(leaf, RuntimeError) for leaf in leaves)
            assert expected + ":RuntimeError" in {str(leaf) for leaf in leaves}
            assert closing.done() and not closing.cancelled()
            direct = await _caught(closing)
            assert isinstance(direct, RuntimeError) and str(direct) == "http_reasoner_close_failed"
            return closing

        closing = await rejected_close()  # Finite FAILURE, never a retirement ACK.
        if stage == "discard":
            client, raw = state.client_refs[0](), state.raw_refs[0]()
            assert client is not None and raw is not None and not raw.is_closed
            assert state.holder.resources.client is client and state.close_calls == 0
            assert not exited.is_set() and not state.preparing.done()
            # Release only the TEST's native leaf gate. Do not close the acquired
            # client, reset reservations or remove the failed lifecycle owner.
            release.set()
            await state.preparing
            assert exited.is_set() and state.holder.native_finished
            if state.scheduled is not None:
                await state.scheduled
                assert state.reservation.finished and not state.reservation.active
                work = state.holder.close_work
                assert work in owner._close_work and work.resource is client
                assert not work.closed_receipt and work.failures
            else:
                assert state.reservation.active and not state.reservation.finished
                assert state.reservation.awaitable is state.pending
            assert state.close_calls == 0 and not raw.is_closed
            assert state.holder.resources.client is client
        else:
            if state.scheduled is not None:
                await state.scheduled
                assert state.reservation.started and state.reservation.finished and not state.reservation.active
            else:
                assert state.reservation.active and not state.reservation.started and not state.reservation.finished
                assert state.reservation.awaitable is state.pending
            # A Task receipt is not a native-client receipt: the actual late
            # prepare observes the seal and never enters its native constructor.
            assert not entered.is_set() and state.client_refs == state.raw_refs == []
        assert await rejected_close() is closing  # Same failed attempt; no retry.
        assert owner in implementation._UNCERTAIN_HTTP_OWNERS and owner._dispatch_uncertain
        assert state.close_calls == state.requests == 0
        assert sync.http == sync.sdks == []
        print("HTTP_PREDISPATCH_CONTRACT=" + json.dumps({"reason": "confirmed", "stage": stage,
            "dispatch": dispatch, "reservation_visible_before_factory": True,
            "actual_scheduled_task_finished": state.scheduled is not None and state.scheduled.done(),
            "native_prepare_exited": exited.is_set(), "acquired_client_retained": stage == "discard",
            "close_reported_failure": True, "physical_client_close_calls": state.close_calls}), flush=True)
        finished.set()
        # Unconfirmed coroutine/owner and any real acquired HTTP client remain
        # strongly retained. No fixture cleanup after this elected failure.
        os._exit(0)

    try:
        asyncio.run(run())
    except BaseException as error:
        frames = []
        trace = error.__traceback__
        while trace is not None:
            frames.append({"file": Path(trace.tb_frame.f_code.co_filename).name,
                           "function": trace.tb_frame.f_code.co_name, "line": trace.tb_lineno})
            trace = trace.tb_next
        print("HTTP_PREDISPATCH_CONTRACT=" + json.dumps({"reason": "assertion_or_lifecycle_failure",
              "error_type": type(error).__name__, "stage": stage, "dispatch": dispatch, "frames": frames}), flush=True)
        os._exit(1)


@pytest.mark.parametrize("stage", ["initialization", "discard"])
@pytest.mark.parametrize("dispatch", ["before", "after"])
def test_prepare_and_discard_reserve_before_actual_task_dispatch_and_never_ack_uncertainty(stage, dispatch):
    script = "import runpy,sys; m=runpy.run_path(sys.argv[1]); m['_child'](sys.argv[2],sys.argv[3])"
    result = subprocess.run([sys.executable, "-c", script, str(Path(__file__).resolve()), stage, dispatch],
                            cwd=_core["_root"](), capture_output=True, text=True, timeout=22)
    rows = [json.loads(line.split("=", 1)[1]) for line in result.stdout.splitlines() if line.startswith("HTTP_PREDISPATCH_CONTRACT=")]
    assert result.returncode == 0 and rows and rows[-1]["reason"] == "confirmed", (result.stdout, result.stderr)
