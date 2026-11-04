"""Hermes owns actual borrowed-pool calls independently of their observers."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import MethodType

import pytest


_CALLER = ContextVar("synthetic_hermes_caller", default=False)
CASES = (
    "concurrent", "external-close", "reentry", "cancel-control", "error-clean",
    "load-call", "stream-yield",
)


async def _until(predicate):
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(.005)


async def _scenario(case):
    from src.llm.hermes_reasoner import HermesCliReasoner

    loop = asyncio.get_running_loop()
    pool = ThreadPoolExecutor(max_workers=16, thread_name_prefix="hermes-test")
    loop.set_default_executor(pool)
    gate = threading.Event()
    entered = []
    observers = []
    closing = None
    callback_receipts = []
    context_reads = []
    native_error = OSError("synthetic_native_error")
    reentry = {}
    model = HermesCliReasoner(provider="synthetic", cloud_consent=True)
    original_handler = loop.get_exception_handler()
    unexpected = []
    loop.set_exception_handler(lambda _loop, info: unexpected.append(info))

    def native(self, prompt=None):
        context_reads.append(_CALLER.get())
        entered.append(threading.current_thread())
        if case == "reentry":
            async def from_worker():
                worker_loop = asyncio.get_running_loop()
                original_factory = worker_loop.get_task_factory()
                creations = []

                def factory(loop, coroutine, **kwargs):
                    creations.append(True)
                    return asyncio.Task(coroutine, loop=loop, **kwargs)

                worker_loop.set_task_factory(factory)
                try:
                    try:
                        await self.aclose()
                    except RuntimeError:
                        reentry["rejected"] = True
                    else:
                        raise AssertionError("active worker closed its own instance")
                finally:
                    worker_loop.set_task_factory(original_factory)
                reentry["new_close_tasks"] = len(creations)

            asyncio.run(from_worker())
        assert gate.wait(10), "fixture did not release actual native callable"
        if case == "cancel-control":
            raise KeyboardInterrupt("synthetic abandoned native control")
        if case == "error-clean":
            raise native_error
        callback_receipts.append(threading.current_thread())
        return None if case == "load-call" else ("synthetic result", 0)

    if case == "load-call":
        model.load = MethodType(native, model)
    else:
        model._run = MethodType(native, model)

    marker = _CALLER.set(True)
    try:
        if case == "stream-yield":
            gate.set()
            stream = model.generate_stream("synthetic sync input")
            assert next(stream) == {
                "token": "synthetic result", "text": "synthetic result", "token_count": 1,
            }
            assert context_reads == [True]  # Sync calls keep their caller context.
            owner = model.begin_close()
            await model.aclose()
            assert owner.done() and owner.exception() is None
            stream.close()
            assert await loop.run_in_executor(pool, lambda: 7) == 7
            return {"case": case, "admitted_native_calls": 1,
                    "original_close_owner_succeeded": True,
                    "borrowed_pool_still_usable": True,
                    "yield_did_not_keep_worker_authority": True}
        count = 9 if case == "concurrent" else 1
        for _ in range(count):
            operation = model.aload() if case == "load-call" else model.complete("synthetic input")
            observers.append(asyncio.create_task(operation))
        await _until(lambda: len(entered) == count)
        assert len(set(entered)) == count
        assert context_reads == [False] * count

        if case in {"concurrent", "cancel-control", "load-call"}:
            for observer in observers:
                observer.cancel()
            cancelled = await asyncio.gather(*observers, return_exceptions=True)
            assert all(isinstance(value, asyncio.CancelledError) for value in cancelled)

        if case == "reentry":
            await _until(lambda: "new_close_tasks" in reentry)
            assert reentry == {"rejected": True, "new_close_tasks": 0}
            observers.append(asyncio.create_task(model.complete("another admitted call")))
            await _until(lambda: len(entered) == 2)
            count = 2  # Rejecting self-close must not seal the instance.
            assert context_reads == [False, False]

        owner = model.begin_close()
        assert model.begin_close() is owner
        closing = asyncio.create_task(model.aclose())
        # The original owner, not an observer facade, must remain pending.
        await asyncio.sleep(0)
        assert not owner.done() and not closing.done()
        if case in {"concurrent", "load-call"}:
            closing.cancel()
            await asyncio.sleep(0)
            closing.cancel()
            await asyncio.sleep(0)
            assert not closing.done() and not owner.cancelled()

        gate.set()
        results = await asyncio.gather(*observers, return_exceptions=True)
        if case == "error-clean":
            assert results[0] is native_error
        elif case not in {"concurrent", "cancel-control", "load-call"}:
            assert results == ["synthetic result"] * count

        close_result = await asyncio.gather(closing, return_exceptions=True)
        assert owner.done() and not owner.cancelled() and owner.exception() is None
        if case in {"concurrent", "load-call"}:
            assert isinstance(close_result[0], asyncio.CancelledError)
        else:
            assert close_result == [None]
        if case not in {"cancel-control", "error-clean"}:
            assert len(callback_receipts) == count

        before = len(entered)
        with pytest.raises(RuntimeError):
            await model.complete("rejected after close")
        if case != "load-call":
            with pytest.raises(RuntimeError):
                model.generate("rejected sync after close")
        assert len(entered) == before
        # Production borrows this pool; closing an instance cannot dispose it.
        assert await loop.run_in_executor(pool, lambda: 7) == 7
        await asyncio.sleep(0)
        assert unexpected == []
        return {"case": case, "admitted_native_calls": count,
                "original_close_owner_succeeded": True,
                "borrowed_pool_still_usable": True,
                "actual_native_context_empty": True}
    finally:
        _CALLER.reset(marker)
        gate.set()
        if observers:
            await asyncio.gather(*observers, return_exceptions=True)
        if closing is not None:
            await asyncio.gather(closing, return_exceptions=True)
        await loop.shutdown_default_executor(timeout=5)
        assert all(not thread.is_alive() for thread in pool._threads)
        loop.set_exception_handler(original_handler)


def _child(case):
    receipt = asyncio.run(_scenario(case))
    print("hermes-call-receipt:" + json.dumps(receipt), flush=True)
    os._exit(0)


@pytest.mark.parametrize("case", CASES)
def test_actual_native_call_lifecycle(case):
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; from tests.unit.llm.test_hermes_call_lifecycle import _child; "
         "_child(sys.argv[1])", case],
        cwd=Path(__file__).parents[3], stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads(next(
        line.split(":", 1)[1] for line in result.stdout.splitlines()
        if line.startswith("hermes-call-receipt:")
    ))
    assert receipt["case"] == case
    assert receipt["original_close_owner_succeeded"]
    assert receipt["borrowed_pool_still_usable"]
