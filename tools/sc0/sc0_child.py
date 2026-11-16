"""SC-0 (2) compiled child: the real managed entry, no traffic, every SQL write traced.

Derived from tests/mocks/stop_contract_child.py (scenario "ready"): runs inside
a prepared slot (cwd = the compiled runtime) under a real admission ticket,
with the same synthetic leaves (no models, no GPU, no microphone, loopback
home and model servers). The only additions are measurement hooks, installed
as monkeypatches here, never in src:

* every sqlite3 connection gets a trace callback that appends each write
  statement (INSERT/UPDATE/DELETE/REPLACE) to ``--trace-file`` (outside the
  state) as a JSON line with its database, verb, target table, the lifecycle
  phase it ran in, the thread and the innermost ``src/`` frames. Values are
  never recorded (literals are masked);
* ``--phase-file`` gets ``ready`` (time of READY) and ``steady`` markers;
  the parent sends SIGTERM after ``steady`` plus its settle interval.

The stop then goes through the real managed path (Application.run's finally,
the exit stack, RuntimeClosureWitness, receipt) and os._exit(0).
"""
import argparse
import contextvars
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import threading
import time
import traceback

RUNTIME = Path.cwd().resolve()
sys.path[:0] = [str(RUNTIME)]

parser = argparse.ArgumentParser()
parser.add_argument("--repo", required=True)
parser.add_argument("--phase-file", required=True)
parser.add_argument("--notify-fifo", default="")
parser.add_argument("--graph-root", required=True)
parser.add_argument("--dashboard-port", type=int, required=True)
parser.add_argument("--trace-file", required=True)
parser.add_argument("--ambient", choices=("idle", "real"), default="idle")
parser.add_argument("--probe-none-until-ready", action="store_true")
ARGS = parser.parse_args()
T0 = time.monotonic()

if ARGS.notify_fifo:
    os.environ["KZA_NOTIFY_FD"] = str(os.open(ARGS.notify_fifo, os.O_WRONLY | os.O_NONBLOCK))

from src.core.process_lifecycle import arm_process_entry  # noqa: E402

LIFECYCLE = arm_process_entry()
STATE = {"ready": False}


def phase(name: str, value: object = "") -> None:
    with open(ARGS.phase_file, "a") as out:
        out.write(f"{name} {value}\n")
        out.flush()


phase("pid", os.getpid())

# ---------------------------------------------------------------- SQL trace
_WRITE = re.compile(r"^\s*(?:WITH\b.*?\)\s*)?(INSERT(?:\s+OR\s+\w+)?\s+INTO|REPLACE\s+INTO|UPDATE(?:\s+OR\s+\w+)?|DELETE\s+FROM)"
                    r"\s+[\"'`\[]?([A-Za-z_][\w]*)", re.IGNORECASE | re.DOTALL)
_TRACE_LOCK = threading.Lock()
_TRACE_OUT = open(ARGS.trace_file, "a", buffering=1)
_REAL_CONNECT = sqlite3.connect


def _lifecycle_phase() -> str:
    if LIFECYCLE.stop.requested:
        return "close"
    return "after_ready" if STATE["ready"] else "construction"


CALLER = contextvars.ContextVar("sc0_caller", default=None)


def _frames(skip: int = 2, limit: int = 8) -> list[str]:
    frames = []
    for frame in reversed(traceback.extract_stack()[:-skip]):
        name = frame.filename
        marker = 0 if name.startswith("src/") else name.rfind("/src/") + 1
        if marker > 0 or name.startswith("src/"):
            frames.append(f"{name[marker:].replace('.pyc', '.py')}:{frame.lineno}:{frame.name}")
            if len(frames) == limit:
                break
    return frames


def _submitter() -> dict:
    import asyncio
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    coro = None if task is None else task.get_coro()
    return {"task": None if task is None else task.get_name(),
            "task_coro": getattr(coro, "__qualname__", None),
            "frames": _frames(skip=3, limit=10)}


def _with_caller(original):
    async def wrapped(*args, **kwargs):
        token = CALLER.set(((CALLER.get() or [])[-2:] + [_submitter()]))
        try:
            return await original(*args, **kwargs)
        finally:
            CALLER.reset(token)
    wrapped.__wrapped__ = original
    return wrapped


def _database_name(args, kwargs) -> str:
    target = kwargs.get("database", args[0] if args else "")
    text = os.fsdecode(target) if isinstance(target, (str, bytes, os.PathLike)) else str(target)
    if text.startswith("file:"):
        text = text[5:].split("?", 1)[0]
    root = str(Path(ARGS.graph_root).resolve().parents[1])  # the slot's state/
    return text[len(root) + 1:] if text.startswith(root + os.sep) else text


def _traced_connect(*args, **kwargs):
    connection = _REAL_CONNECT(*args, **kwargs)
    database = _database_name(args, kwargs)
    if database.endswith("chroma.sqlite3") or database in ("", ":memory:"):
        return connection

    def trace(statement: str) -> None:
        match = _WRITE.match(statement)
        if match is None:
            return
        record = {"t": round(time.monotonic() - T0, 4), "phase": _lifecycle_phase(),
                  "db": database, "verb": re.sub(r"\s+", " ", match.group(1).upper()),
                  "table": match.group(2), "thread": threading.current_thread().name,
                  "frames": _frames(),
                  "caller": CALLER.get() or SUBMITTED.get(threading.current_thread().name)}
        with _TRACE_LOCK:
            _TRACE_OUT.write(json.dumps(record) + "\n")

    connection.set_trace_callback(trace)
    return connection


sqlite3.connect = _traced_connect
import asyncio as _asyncio  # noqa: E402

_asyncio.to_thread = _with_caller(_asyncio.to_thread)
from src.core import owned_work as _owned_work  # noqa: E402

import aiosqlite.core as _aiosqlite_core  # noqa: E402

SUBMITTED: dict = {}
_aio_execute = _aiosqlite_core.Connection._execute


async def _aio_traced(self, fn, *args, **kwargs):
    # aiosqlite runs each statement on its connection's own thread, which
    # does not see the loop's context: remember the submitting coroutine.
    SUBMITTED[self._thread.name] = (CALLER.get() or [])[-2:] + [_submitter()]
    return await _aio_execute(self, fn, *args, **kwargs)


_aiosqlite_core.Connection._execute = _aio_traced

for _name in ("run", "run_async", "native_step"):
    setattr(_owned_work.OwnedBlockingWork, _name,
            _with_caller(getattr(_owned_work.OwnedBlockingWork, _name)))
# ------------------------------------------------------------------------

sys.path.append(ARGS.repo)

import asyncio  # noqa: E402

import pytest  # noqa: E402

import src.main as main  # noqa: E402
from tests.mocks.application_graph import offline_home  # noqa: E402
from tests.mocks.managed_slot_graph import _get, _leaves, _variable  # noqa: E402
from tests.mocks.producer_erasure_graph import construction_sdk_guards, nominal_factories  # noqa: E402
from tests.mocks.restore_application_graph import (  # noqa: E402
    MODEL_ENV, TOKEN, configuration as restore_configuration, local_models, sdk_leaves,
)

_ready = LIFECYCLE.ready


def _observed_ready():
    _ready()
    STATE["ready"] = True
    phase("ready", round(time.monotonic() - T0, 3))


LIFECYCLE.ready = _observed_ready
_blocked = LIFECYCLE.capture_blocked


def _observed_blocked():
    _blocked()
    STATE["ready"] = True
    phase("ready", round(time.monotonic() - T0, 3))
    phase("capture_blocked")


LIFECYCLE.capture_blocked = _observed_blocked

if ARGS.probe_none_until_ready:
    # Deferred retired-owner prune (design: "a prune deferred because the
    # authority probe answered None"): until READY the authority cannot tell.
    from src.dashboard.authority import DashboardAuthority

    _retired, _probe = DashboardAuthority.users_retired, DashboardAuthority.memory_owner_retired

    def _users_retired(self, identities):
        return False if not STATE["ready"] else _retired(self, identities)

    def _owner_retired(self, identities):
        return None if not STATE["ready"] else _probe(self, identities)

    DashboardAuthority.users_retired = _users_retired
    DashboardAuthority.memory_owner_retired = _owner_retired


def configuration(root, home_url, model_url, *, port=ARGS.dashboard_port):
    config = restore_configuration(Path(root), home_url, model_url)
    config["dashboard"].update(enabled=True, host="127.0.0.1", port=port)
    return config


def resolve_configuration(root: Path, home_url: str, model_url: str) -> dict:
    import yaml

    path = root.parents[1] / "config/settings.yaml"
    raw = yaml.safe_load(path.read_text())
    generated = configuration(root, home_url, model_url)
    for location, value in _leaves(raw):
        if type(value) is str and value.startswith("${KZA_SLOT_BIND_"):
            assert value == "${" + _variable(location) + "}"
            os.environ[_variable(location)] = _get(generated, location)
    return main.load_config(str(path))


def hold_audio(graph) -> None:
    loop = graph.pipeline.audio_loop

    async def start():
        return None

    async def run():
        phase("steady", round(time.monotonic() - T0, 3))
        await asyncio.Event().wait()

    loop.start, loop.run = start, run


def main_entry() -> None:
    patch = pytest.MonkeyPatch()
    root = Path(ARGS.graph_root).resolve()
    patch.setenv("KZA_DASHBOARD_TOKEN", TOKEN)
    patch.setenv(MODEL_ENV, "synthetic-local-key")
    factories, _ = nominal_factories(patch, root)
    assert not construction_sdk_guards(patch)
    sdk_leaves(patch, root / "chroma")

    async def build(shutdown, lifecycle):
        home_url, _ = await shutdown.enter_async_context(offline_home())
        model_url, _ = await shutdown.enter_async_context(local_models())
        config = resolve_configuration(root, home_url, model_url)
        graph = await main.build_application(config, shutdown=shutdown, factories=factories,
                                             lifecycle=lifecycle)
        phase("constructed", round(time.monotonic() - T0, 3))
        hold_audio(graph)
        if graph.ambient_path is not None and ARGS.ambient == "idle":
            async def ambient_idle():
                await asyncio.Event().wait()
            graph.ambient_path.start = ambient_idle
        return graph

    main.run_entry(LIFECYCLE, build=build)
    phase("returned")


if __name__ == "__main__":
    main_entry()
