"""Actual async-token drain observation; no sleep-based early-success oracle."""

import asyncio
import json
from pathlib import Path
import runpy
import subprocess
import sys

import pytest


_async = runpy.run_path(str(Path(__file__).with_name("test_http_reasoner_async_lifecycle.py")))
AsyncProbe, Gate = _async["AsyncProbe"], _async["Gate"]
Probe = _async["Probe"]
ANSWER, PROMPT = _async["ANSWER"], _async["PROMPT"]
_watchdog = _async["_watchdog"]


def _root():
    return next(parent for parent in Path(__file__).resolve().parents
                if (parent / "src/llm/reasoner.py").is_file())


def _isolated(stage, cancels):
    script = ("import os,runpy,sys,json\nm=runpy.run_path(sys.argv[1])\ntry:\n"
              " m['_held_child'](sys.argv[2],int(sys.argv[3]))\nexcept BaseException as error:\n"
              " print('HTTP_CONTRACT='+json.dumps({'reason':'contract_failed','error_type':type(error).__name__}),flush=True)\n"
              " os._exit(1)\nos._exit(0)")
    result = subprocess.run([sys.executable, "-c", script, str(Path(__file__).resolve()), stage, str(cancels)],
                            cwd=_root(), capture_output=True, text=True, timeout=12)
    receipts = [json.loads(line.split("=", 1)[1]) for line in result.stdout.splitlines()
                if line.startswith("HTTP_CONTRACT=")]
    assert receipts, f"fixture failed before held contract receipt: {result.stdout}\n{result.stderr}"
    assert result.returncode == 0 and receipts[-1]["reason"] == "confirmed", f"{receipts[-1]}\n{result.stderr}"


def _held_child(stage, cancels):
    async def run():
        patch = pytest.MonkeyPatch()
        state = AsyncProbe(patch)
        wrapper = state.wrapper
        await asyncio.to_thread(wrapper.load)  # Actual cached sync cohort too.
        cached = state.sync.sdks[0]
        entered, release, exited = (asyncio.Event() for _ in range(3))
        native_gate = Gate(state.sync) if stage == "prepare" else None

        async def gate(resource):
            entered.set()
            try:
                await release.wait()
            finally:
                exited.set()

        if native_gate is not None:
            state.on_prepare = native_gate
        else:
            setattr(state, "on_" + stage, gate)
        observed_live_drain = asyncio.Event()
        lifecycle = wrapper._http_lifecycle
        actual_leases = lifecycle._adapter_leases
        owner = {}

        class ObservedActualLeases:
            def __iter__(self):
                try:
                    task = asyncio.current_task()
                except RuntimeError:
                    task = None
                if task is owner.get("task") and task is not None and any(lease.active for lease in actual_leases):
                    observed_live_drain.set()
                return iter(actual_leases)

            def __getattr__(self, name):
                return getattr(actual_leases, name)

            def __len__(self):
                return len(actual_leases)

            def __contains__(self, value):
                return value in actual_leases

        # Delegates every mutation/read to the ORIGINAL live set. No copied
        # leases, fake owner, flags or detached aliases manufacture progress.
        patch.setattr(lifecycle, "_adapter_leases", ObservedActualLeases())
        original_begin = wrapper.begin_close

        def begin():
            task = original_begin()
            owner["task"] = task
            return task

        patch.setattr(wrapper, "begin_close", begin)

        def cached_close(sdk):
            assert sdk is cached
            assert state.clients and all(client.close_calls == 1 and client.http.is_closed for client in state.clients)
            assert all(stream.close_calls == 1 for stream in state.streams)
            assert native_gate.exited.is_set() if native_gate is not None else exited.is_set()

        state.sync.on_close = cached_close
        finished = _watchdog(lambda: {"stage": stage, "real_async_drain_observed": observed_live_drain.is_set(),
                                     "cached_sdk_close_calls": cached.close_calls,
                                     "request_clients_closed": [client.http.is_closed for client in state.clients]})

        async def invoke():
            if stage == "read":
                return [row async for row in wrapper.astream(PROMPT)]
            return await wrapper.complete(PROMPT)

        operation = asyncio.create_task(invoke())
        if native_gate is not None:
            while not native_gate.entered.is_set():
                await asyncio.sleep(.001)
        else:
            await entered.wait()
        closing = asyncio.create_task(wrapper.aclose())
        await observed_live_drain.wait()  # Actual retained owner examined live lease.
        assert not operation.done() and not closing.done()
        assert cached.close_calls == 0 and not cached.http.is_closed
        for unused in range(cancels):
            closing.cancel()
            await asyncio.sleep(0)
        if native_gate is not None:
            native_gate.release.set()
        else:
            release.set()
        result = await operation
        if stage == "read":
            assert "".join(row["token"] for row in result) == "".join(state.streams[0].parts)
        else:
            assert result == ANSWER.strip()
        if cancels:
            with pytest.raises(asyncio.CancelledError):
                await closing
        else:
            await closing
        assert cached.close_calls == cached.http.close_calls == 1 and cached.http.is_closed
        assert all(client.close_calls == 1 and client.http.is_closed for client in state.clients)
        assert wrapper._client is None
        print("HTTP_CONTRACT=" + json.dumps({"reason": "confirmed", "stage": stage,
              "actual_live_lease_observed_by_retirement_owner": True,
              "request_cleanup_before_cached_sdk_close": True, "observer_cancellations": cancels}), flush=True)
        finished.set()
        patch.undo()

    asyncio.run(run())


@pytest.mark.parametrize("stage", ["prepare", "complete", "read", "client_close"])
@pytest.mark.parametrize("cancels", [0, 3])
def test_instance_close_waits_actual_async_constructor_body_read_and_owned_client_cleanup(stage, cancels):
    _isolated(stage, cancels)


def test_failed_explicit_reload_closes_only_failed_attempt_and_keeps_prior_cache_retryable(monkeypatch):
    class ReloadError(RuntimeError):
        pass

    state = Probe(monkeypatch)
    wrapper = state.wrapper()
    try:
        wrapper.load()
        first = wrapper._client

        def discovery(sdk):
            if sdk is state.sdks[1]:
                raise ReloadError("synthetic discovery failed after a successful load")

        state.on_discover = discovery
        with pytest.raises(ReloadError):
            wrapper.load()
        assert wrapper._client is first and not first.is_closed
        assert state.sdks[1].close_calls == 1 and state.sdks[1].is_closed
        state.on_discover = None
        wrapper.load()
        assert wrapper._client is state.sdks[2] and not first.is_closed
        wrapper.close()
        assert [sdk.close_calls for sdk in state.sdks] == [1, 1, 1]
        assert all(http.close_calls == 1 and http.is_closed for http in state.http)
    finally:
        state.cleanup()
