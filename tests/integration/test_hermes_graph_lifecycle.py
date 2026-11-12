"""Actual Hermes graph/job receipts with an inert test executable."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack
from contextvars import ContextVar
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from unittest.mock import patch

import pytest


CASES = ("registration", "stack", "runtime", "after-main", "after-compactor", "failed-cleanup")
MARKER = ContextVar("synthetic_hermes_graph_marker", default=False)


async def _until(predicate, *, timeout=12):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "actual graph phase was not reached"
        await asyncio.sleep(.005)


async def _caught(task):
    try:
        return await asyncio.shield(task)
    except BaseException as error:
        return error


class _Pipe:
    def __init__(self, raw):
        self.raw, self.returns = raw, 0

    def __getattr__(self, name):
        return getattr(self.raw, name)

    def close(self):
        self.raw.close()
        self.returns += 1


def _child(base: Path, case: str):
    from src.llm import hermes_reasoner as module
    import src.main as main_module
    import src.vectordb.chroma_sync as index_module
    from tests.integration import test_application_graph as application_fixtures
    from tests.integration.test_wake_graph_retirement import configured
    from tests.mocks.application_graph import offline_home

    assert case in CASES
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    fixture_patch = pytest.MonkeyPatch()
    providers = application_fixtures.leaves.__wrapped__(fixture_patch, base)

    class Collection:
        def count(self):
            return 0

        def get(self, *args, **kwargs):
            return {"ids": [], "documents": [], "metadatas": [], "embeddings": []}

        def query(self, *args, **kwargs):
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    class ChromaLeaf:
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
            pass  # Synthetic SDK leaf, not invented model disposal.

    fixture_patch.setattr(index_module.chromadb, "PersistentClient", ChromaLeaf)
    executable = base / "inert-hermes"
    executable.write_text(
        "#!" + sys.executable + "\n"
        "import json, os, sys, time\nfrom pathlib import Path\n"
        "if sys.argv[1:3] == ['auth', 'status']:\n"
        " print('synthetic'); raise SystemExit(0)\n"
        "assert '-z' in sys.argv\n"
        "base = Path(" + repr(str(base)) + ")\n"
        "pid = str(os.getpid())\n"
        "(base / ('ready-' + pid)).write_text('ready')\n"
        "release = base / ('release-' + pid)\n"
        "deadline = time.monotonic() + 20\n"
        "while not release.exists():\n"
        " if time.monotonic() >= deadline: raise SystemExit(97)\n"
        " time.sleep(.01)\n"
        "usage = Path(sys.argv[sys.argv.index('--usage-file') + 1])\n"
        "usage.write_text(json.dumps({'tokens': {'total': 42}}))\n"
        "print('  synthetic graph result  ')\n"
    )
    executable.chmod(0o700)
    real_submit = ThreadPoolExecutor.submit
    real_popen = subprocess.Popen
    real_killpg, real_getpgid = os.killpg, os.getpgid
    real_init, real_load = module.HermesCliReasoner.__init__, module.HermesCliReasoner.load
    real_run, real_begin = module.HermesCliReasoner._run, module.HermesCliReasoner.begin_close
    local = threading.local()
    instances, loads, jobs = [], [], []
    model_tasks, completion_receipts = {}, []
    privacy = graph = None
    ledger_close_entries = []
    primary = OSError("synthetic_graph_communicate_failure")
    cleanup = OSError("synthetic_graph_signal_failure")
    fixture_pool = None
    fixture_errors, observers = [], []
    closing = build_task = external_close = None
    stack = None

    def submit(executor, function, /, *args, **kwargs):
        # Forward ALL work; associate its actual Future at real _run entry.
        published, slot = threading.Event(), {}

        def invoke():
            assert published.wait(10), "actual Future was not published"
            previous = getattr(local, "submission", None)
            local.submission = slot
            try:
                return function(*args, **kwargs)
            finally:
                local.submission = previous

        future = real_submit(executor, invoke)
        slot["future"] = future
        published.set()
        return future

    def initialize(instance, *args, **kwargs):
        real_init(instance, *args, **kwargs)
        instances.append(instance)

    def load(instance):
        result = real_load(instance)  # Actual inert auth process, not a stub.
        loads.append(instance)
        return result

    def begin(instance):
        task = real_begin(instance)
        previous = model_tasks.get(id(instance))
        assert previous is None or previous is task
        if previous is None:
            model_tasks[id(instance)] = task

            def completed(actual):
                selected_jobs = [job for job in jobs if job["instance"] is instance]
                completion_receipts.append({
                    "instance": instance, "task": actual,
                    "native_done": all(job["future"].done() for job in selected_jobs),
                    "postludes_exited": all(job["postlude_exited"].is_set()
                                            for job in selected_jobs),
                })

            task.add_done_callback(completed)
        return task

    def run(instance, prompt):
        slot = getattr(local, "submission", None)
        assert slot is not None and "future" in slot
        job = {
            "instance": instance, "future": slot["future"],
            "thread": threading.current_thread(), "proc": None,
            "postlude_entered": threading.Event(),
            "postlude_release": threading.Event(),
            "postlude_exited": threading.Event(),
        }
        jobs.append(job)
        assert MARKER.get() is False, "caller ContextVar reached native work"
        previous = getattr(local, "job", None)
        local.job = job
        try:
            value = real_run(instance, prompt)
            job["postlude_entered"].set()
            try:
                assert job["postlude_release"].wait(20), "native postlude not released"
            finally:
                job["postlude_exited"].set()
            return value
        finally:
            local.job = previous

    def popen(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        if "-z" not in args[0]:
            return proc
        job = getattr(local, "job", None)
        assert job is not None
        job.update(
            proc=proc, wait=proc.wait,
            raw_stdout=proc.stdout, raw_stderr=proc.stderr,
            ready=base / ("ready-" + str(proc.pid)),
            release=base / ("release-" + str(proc.pid)),
            usage=Path(args[0][args[0].index("--usage-file") + 1]),
            wait_returns=0,
        )
        stdout, stderr = _Pipe(proc.stdout), _Pipe(proc.stderr)
        proc.stdout, proc.stderr = stdout, stderr
        job.update(stdout=stdout, stderr=stderr)

        def wait(*a, **kw):
            value = job["wait"](*a, **kw)
            job["wait_returns"] += 1
            return value

        proc.wait = wait
        if case == "failed-cleanup":
            def communicate(*a, **kw):
                deadline = time.monotonic() + 10
                while not job["ready"].exists():
                    assert proc.poll() is None
                    assert time.monotonic() < deadline
                    time.sleep(.005)
                assert proc.poll() is None and real_getpgid(proc.pid) == proc.pid
                raise primary

            proc.communicate = communicate
        return proc

    def killpg(pgid, sig):
        if case == "failed-cleanup" and any(
            job["proc"] is not None and job["proc"].pid == pgid for job in jobs
        ):
            assert sig == signal.SIGKILL
            raise cleanup
        return real_killpg(pgid, sig)

    def selected(instance):
        matching = [job for job in jobs if job["instance"] is instance]
        assert len(matching) <= 1
        return matching[0] if matching else None

    def ready(instance):
        job = selected(instance)
        return job is not None and job["proc"] is not None and job["ready"].exists()

    def retained():
        assert privacy is not None and privacy.resource_status()["writer_lease"]
        assert privacy.ledger._db is not None
        assert privacy.ledger._db.execute("SELECT 1").fetchone()[0] == 1
        assert ledger_close_entries == []

    def observe_privacy(actual):
        nonlocal privacy
        privacy = actual
        ledger = actual.ledger
        original = type(ledger).close

        def close(value):
            if value is ledger:
                ledger_close_entries.append(True)
            return original(value)

        fixture_patch.setattr(type(ledger), "close", close)

    def assert_process_retired(job):
        assert job["wait_returns"] >= 1
        assert job["stdout"].returns >= 1 and job["stderr"].returns >= 1
        assert job["raw_stdout"].closed and job["raw_stderr"].closed
        assert not job["usage"].exists()
        assert all(owner.proc is not job["proc"] for owner in module._HERMES_PROCESS_OWNERS)

    async def finish_job(job):
        job["release"].touch()
        await _until(job["postlude_entered"].is_set)
        assert_process_retired(job)
        assert not job["future"].done()

    async def scenario():
        nonlocal graph, stack, closing, build_task, external_close, fixture_pool
        from src.main import build_application

        partial_error = RuntimeError("synthetic later Hermes factory failure")
        original_privacy = main_module.build_privacy_services

        def start_sync(instance):
            future = fixture_pool.submit(instance.generate, "synthetic partial request")
            observers.append(future)
            return future

        async def privacy_builder(*args, **kwargs):
            if case == "after-main":
                start_sync(instances[0])
                await _until(lambda: ready(instances[0]))
                raise partial_error
            value = await original_privacy(*args, **kwargs)
            observe_privacy(value)
            return value

        def later_router(*args, **kwargs):
            assert len(instances) == 2
            future = start_sync(instances[1])
            deadline = time.monotonic() + 10
            while not ready(instances[1]):
                assert not future.done(), "job ended before real child readiness"
                assert time.monotonic() < deadline
                time.sleep(.005)
            raise partial_error

        fixture_patch.setattr(main_module, "build_privacy_services", privacy_builder)
        if case.startswith("after-"):
            fixture_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hermes-fixture")
        if case == "after-compactor":
            fixture_patch.setattr(main_module, "RequestRouter", later_router)

        async with offline_home() as (url, observations):
            config = configured(base, url, rooms=False)
            config["reasoner"] = {
                "mode": "hermes_cli", "hermes_binary_path": str(executable),
                "hermes_provider": "synthetic", "hermes_model": "synthetic-model",
                "hermes_timeout_s": 20, "cloud": {"consent": True},
            }
            config["orchestrator"] = {"enabled": True, "context": {"compaction": {
                "enabled": True, "timeout_s": 20,
            }}}
            stack = AsyncExitStack()
            await stack.__aenter__()
            try:
                if case.startswith("after-"):
                    async def build():
                        async with stack:
                            return await build_application(config, shutdown=stack, factories=providers)

                    build_task = asyncio.create_task(build())
                    expected = 1 if case == "after-main" else 2
                    await _until(lambda: len(instances) == expected and bool(jobs)
                                 and any(id(value) in model_tasks for value in instances))
                    assert len(loads) == expected
                    assert not build_task.done()
                    job = jobs[0]
                    assert job["future"].running() and not job["future"].done()
                    if privacy is not None:
                        retained()
                    await finish_job(job)
                    assert not build_task.done()
                    if privacy is not None:
                        retained()
                    job["postlude_release"].set()
                    result = await asyncio.wait_for(_caught(build_task), 10)
                    assert result is partial_error
                    await _until(job["future"].done)
                    assert all(id(value) in model_tasks for value in instances)
                else:
                    graph = await build_application(config, shutdown=stack, factories=providers)
                    assert len(instances) == len(loads) == 2
                    main = graph.pipeline._orchestrator.llm
                    compactor = graph.pipeline._orchestrator.context_manager.compactor.reasoner
                    assert main is instances[0] and compactor is instances[1] and main is not compactor
                    assert type(main) is type(compactor) is module.HermesCliReasoner
                    if case == "registration":
                        assert jobs == []
                        await stack.aclose()
                    elif case == "failed-cleanup":
                        token = MARKER.set(True)
                        try:
                            caller = asyncio.create_task(main.complete("synthetic failed request"))
                        finally:
                            MARKER.reset(token)
                        observers.append(caller)
                        result = await asyncio.wait_for(_caught(caller), 10)
                        assert result is primary
                        job = selected(main)
                        assert job is not None
                        await _until(job["future"].done)
                        assert job["proc"].poll() is None
                        owners = [owner for owner in module._HERMES_PROCESS_OWNERS
                                  if owner.proc is job["proc"]]
                        assert len(owners) == 1 and not owners[0].retired
                        closing = asyncio.create_task(stack.aclose())
                        result = await asyncio.wait_for(_caught(closing), 10)
                        assert isinstance(result, BaseException)
                        actual = model_tasks[id(main)]
                        assert actual.done() and not actual.cancelled()
                        assert type(actual.exception()) is RuntimeError
                        assert actual.exception().args == ("hermes_retirement_unconfirmed",)
                        assert main.begin_close() is actual
                        assert owners[0] in module._HERMES_PROCESS_OWNERS
                        retained()
                    else:
                        for instance in instances:
                            token = MARKER.set(True)
                            try:
                                observers.append(asyncio.create_task(instance.complete("synthetic held request")))
                            finally:
                                MARKER.reset(token)
                        await _until(lambda: all(ready(instance) for instance in instances))
                        assert len(jobs) == 2
                        for observer in observers:
                            observer.cancel()
                        results = await asyncio.gather(*observers, return_exceptions=True)
                        assert all(isinstance(value, asyncio.CancelledError) for value in results)
                        assert all(job["future"].running() and not job["future"].done() for job in jobs)
                        closing = asyncio.create_task(
                            stack.aclose() if case == "stack" else graph._stop_services(None, None)
                        )
                        await _until(lambda: bool(model_tasks))
                        first = next(value for value in instances if id(value) in model_tasks)
                        actual = model_tasks[id(first)]
                        external_close = asyncio.create_task(first.aclose())
                        for unused in range(2):
                            await asyncio.sleep(0)
                            assert external_close.cancel()
                            await asyncio.sleep(0)
                            assert not external_close.done() and first.begin_close() is actual
                        assert not closing.done()
                        retained()
                        for job in jobs:
                            await finish_job(job)
                        assert not closing.done()
                        retained()
                        order = jobs if case == "stack" else list(reversed(jobs))
                        order[0]["postlude_release"].set()
                        await _until(order[0]["future"].done)
                        assert not order[1]["future"].done() and not closing.done()
                        retained()
                        order[1]["postlude_release"].set()
                        await asyncio.wait_for(asyncio.shield(closing), 10)
                        assert isinstance(await _caught(external_close), asyncio.CancelledError)
                        await stack.aclose()

                if case != "failed-cleanup":
                    assert all(id(value) in model_tasks for value in instances)
                    assert all(task.done() and not task.cancelled() and task.exception() is None
                               for task in model_tasks.values())
                    await asyncio.sleep(0)  # Deliver actual Task done callbacks.
                    assert len(completion_receipts) == len(instances)
                    assert all(receipt["postludes_exited"] for receipt in completion_receipts)
                    if not case.startswith("after-"):
                        assert all(receipt["native_done"] for receipt in completion_receipts)
                    if privacy is not None:
                        assert not privacy.resource_status()["writer_lease"]
                        assert privacy.ledger._db is None
                assert observations["service_calls"] == 0
                return {"case": case, "product_assertions_passed": True,
                        "actual_instances": len(instances), "native_jobs": len(jobs),
                        "failed_owners_retained": case == "failed-cleanup"}
            finally:
                # All fixture receipts follow product observations. Never reset
                # production owners or turn fixture cleanup into product success.
                for job in jobs:
                    job["postlude_release"].set()
                    if job["proc"] is not None:
                        job["release"].touch()
                for job in jobs:
                    proc = job["proc"]
                    if proc is None:
                        continue
                    try:
                        if proc.poll() is None:
                            assert real_getpgid(proc.pid) == proc.pid
                            real_killpg(proc.pid, signal.SIGKILL)
                        job["wait"](timeout=3)
                    except BaseException as error:
                        fixture_errors.append(type(error).__name__)
                    finally:
                        try:
                            job["raw_stdout"].close()
                        finally:
                            job["raw_stderr"].close()
                        job["usage"].unlink(missing_ok=True)
                await _until(lambda: all(job["future"].done() for job in jobs))
                for job in jobs:
                    try:
                        job["future"].result()
                    except BaseException as error:
                        if case != "failed-cleanup" or error is not primary:
                            fixture_errors.append(type(error).__name__)
                for pending in (build_task, closing, external_close):
                    if pending is not None and not pending.done():
                        await asyncio.wait_for(_caught(pending), 10)
                if not case.startswith("after-"):
                    if case != "failed-cleanup" or closing is None:
                        await stack.aclose()
                if fixture_pool is not None:
                    fixture_pool.shutdown(wait=True)
                assert fixture_errors == []

    with patch.object(ThreadPoolExecutor, "submit", submit), \
            patch.object(module.HermesCliReasoner, "__init__", initialize), \
            patch.object(module.HermesCliReasoner, "load", load), \
            patch.object(module.HermesCliReasoner, "_run", run), \
            patch.object(module.HermesCliReasoner, "begin_close", begin), \
            patch.object(module.subprocess, "Popen", popen), \
            patch.object(module.os, "killpg", killpg):
        receipt = asyncio.run(scenario())
    for thread in {job["thread"] for job in jobs}:
        thread.join(timeout=3)
        assert not thread.is_alive()
    assert all(job["future"].done() for job in jobs)
    receipt["fixture_real_futures_pipes_and_threads_retired"] = True
    print("hermes-graph-receipt:" + json.dumps(receipt), flush=True)
    # Failed W owners/authority remain retained: exit contains them, not retires.
    os._exit(0)


@pytest.mark.usefixtures("dashboard_auth")
@pytest.mark.parametrize("case", CASES)
def test_actual_hermes_graph_lifetime(tmp_path, case):
    command = [
        sys.executable, "-c",
        "from pathlib import Path; import sys; "
        "from tests.integration.test_hermes_graph_lifecycle import _child; "
        "_child(Path(sys.argv[1]), sys.argv[2])",
        str(tmp_path), case,
    ]
    result = subprocess.run(
        command, cwd=Path(__file__).resolve().parents[2],
        stdin=subprocess.DEVNULL, capture_output=True, text=True,
        timeout=60, check=False,
    )
    (tmp_path / "child.stdout.log").write_text(result.stdout)
    (tmp_path / "child.stderr.log").write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    line = next(value for value in result.stdout.splitlines()
                if value.startswith("hermes-graph-receipt:"))
    receipt = json.loads(line.split(":", 1)[1])
    assert receipt["case"] == case and receipt["product_assertions_passed"]
    assert receipt["fixture_real_futures_pipes_and_threads_retired"]
