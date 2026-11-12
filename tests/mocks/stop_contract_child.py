"""U3b-0d1 compiled child: the real managed entry with synthetic leaves.

Runs inside a prepared slot (cwd = the compiled runtime) under a real
admission ticket. ``src`` is imported only from the runtime's bytecode; the
checkout contributes the ``tests`` fixtures. The parent sends real SIGTERMs
when this child reports a phase on ``--phase-file``, and reads READY from the
FIFO given in ``--notify-fifo`` (handed to the child as KZA_NOTIFY_FD).
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

RUNTIME = Path.cwd().resolve()
sys.path[:0] = [str(RUNTIME)]

parser = argparse.ArgumentParser()
parser.add_argument("--repo", required=True)
parser.add_argument("--scenario", required=True)
parser.add_argument("--phase-file", required=True)
parser.add_argument("--notify-fifo", required=True)
parser.add_argument("--graph-root", required=True)
parser.add_argument("--dashboard-port", type=int, required=True)
ARGS = parser.parse_args()

# The parent holds the FIFO's read end open, so this never blocks.
os.environ["KZA_NOTIFY_FD"] = str(os.open(ARGS.notify_fifo, os.O_WRONLY | os.O_NONBLOCK))

from src.core.process_lifecycle import arm_process_entry, blocked_stop_signals  # noqa: E402

LIFECYCLE = arm_process_entry()


def phase(name: str, value: object = "") -> None:
    with open(ARGS.phase_file, "a") as out:
        out.write(f"{name} {value}\n")
        out.flush()


phase("pid", os.getpid())
phase("mask", ",".join(str(int(item)) for item in sorted(blocked_stop_signals())) or "empty")

sys.path.append(ARGS.repo)

if ARGS.scenario == "entry":
    phase("entry")
    deadline = time.monotonic() + 30
    while not LIFECYCLE.stop.requested and time.monotonic() < deadline:
        time.sleep(0.01)
    assert LIFECYCLE.stop.requested, "early SIGTERM not recorded"

import asyncio  # noqa: E402
import importlib.machinery  # noqa: E402

import pytest  # noqa: E402

import src.main as main  # noqa: E402
from tests.mocks.application_graph import offline_home  # noqa: E402
from tests.mocks.producer_erasure_graph import construction_sdk_guards, nominal_factories  # noqa: E402
from tests.mocks.restore_application_graph import (  # noqa: E402
    MODEL_ENV, TOKEN, configuration as restore_configuration, local_models, sdk_leaves,
)
from tests.mocks.managed_slot_graph import _get, _leaves, _variable  # noqa: E402

for name, module in list(sys.modules.items()):
    if name == "src" or name.startswith("src."):
        origin = getattr(module, "__file__", None)
        assert origin is None or Path(origin).resolve().is_relative_to(RUNTIME), name
assert isinstance(main.__loader__, importlib.machinery.SourcelessFileLoader)


def configuration(root, home_url, model_url, *, port=ARGS.dashboard_port):
    """Every serving store under the state; the dashboard on a private port."""
    config = restore_configuration(Path(root), home_url, model_url)
    config["dashboard"].update(enabled=True, host="127.0.0.1", port=port)
    return config


def resolve_configuration(root: Path, home_url: str, model_url: str) -> dict:
    """Bind the canonical template's placeholders, then use main's own loader."""
    import yaml

    path = root.parents[1] / "config/settings.yaml"
    raw = yaml.safe_load(path.read_text())
    generated = configuration(root, home_url, model_url)
    for location, value in _leaves(raw):
        if type(value) is str and value.startswith("${KZA_SLOT_BIND_"):
            assert value == "${" + _variable(location) + "}"
            os.environ[_variable(location)] = _get(generated, location)
    return main.load_config(str(path))


async def hold_until_stopped(name: str) -> None:
    phase(name)
    async with asyncio.timeout(30):
        while not LIFECYCLE.stop.requested:
            await asyncio.sleep(0.005)


def wrap(owner, attribute: str, name: str) -> None:
    original = getattr(owner, attribute)

    async def hooked(*args, **kwargs):
        await hold_until_stopped(name)
        return await original(*args, **kwargs)

    setattr(owner, attribute, hooked)


class _OneLight:
    async def get_domotics_entities(self):
        return [{"entity_id": "light.living", "attributes": {"friendly_name": "Living"}}]

    async def get_services_by_domain(self):
        return {"light": ["turn_on"]}


class _HungLLM:
    async def complete(self, prompt, **kwargs):
        phase("syncing")
        await asyncio.Event().wait()


class _ManyPhrases:
    async def complete(self, prompt, **kwargs):
        return json.dumps([{"entity_id": "light.living", "domain": "light", "service": "turn_on",
                            "description": "Encender", "phrases": [f"prende el living {n}" for n in range(300)]}])


def hold_audio(graph) -> None:
    loop = graph.pipeline.audio_loop

    async def start():
        if ARGS.scenario == "audio_start":
            # The stop lands after READY, before audio_loop.run() starts.
            await hold_until_stopped("audio_start")

    async def run():
        if ARGS.scenario == "executor":
            # An in-flight default-executor job asyncio.run would join for 60 s.
            asyncio.get_running_loop().run_in_executor(None, time.sleep, 60)
        elif ARGS.scenario in {"sync", "sync_encode"}:
            # U3b-0d2 S2: a command sync whose cloud call never answers ("sync"),
            # or one held in the owned worker's BGE-M3 encodes ("sync_encode").
            chroma = graph.models.chroma
            llm = _HungLLM()
            if ARGS.scenario == "sync_encode":
                llm = _ManyPhrases()
                encode = chroma._embedder.encode

                def slow_encode(*args, **kwargs):
                    phase("syncing")
                    time.sleep(0.1)
                    return encode(*args, **kwargs)
                chroma._embedder.encode = slow_encode
            hold_audio.sync = asyncio.create_task(chroma.sync_commands(_OneLight(), llm))
            hold_audio.sync.add_done_callback(lambda task: phase(
                "sync_ended", "cancelled" if task.cancelled() else getattr(task.exception(), "code", "ok")))
        elif ARGS.scenario == "writer":
            # A writable descriptor left open under the state.
            hold_audio.writer = open(Path(ARGS.graph_root) / "held-writer.bin", "ab")
        phase("steady")
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
    if ARGS.scenario == "construction":
        wrap(main, "build_privacy_services", "construction")

    async def build(shutdown, lifecycle):
        home_url, _ = await shutdown.enter_async_context(offline_home())
        model_url, _ = await shutdown.enter_async_context(local_models())
        config = resolve_configuration(root, home_url, model_url)
        graph = await main.build_application(config, shutdown=shutdown, factories=factories,
                                             lifecycle=lifecycle)
        hold_audio(graph)
        if graph.ambient_path is not None:
            async def ambient_idle():
                await asyncio.Event().wait()
            graph.ambient_path.start = ambient_idle
        warmup = graph.warmup

        async def observed_warmup():
            phase("warmup_entered")
            return await warmup()
        graph.warmup = observed_warmup
        if ARGS.scenario == "warmup":
            wrap(graph, "warmup", "warmup")
        elif ARGS.scenario == "services":
            wrap(graph, "_start_services", "services")
        elif ARGS.scenario == "pipeline":
            wrap(graph.pipeline.command_processor, "aload_models", "pipeline")
        elif ARGS.scenario == "blocked":
            # Keep the real owner (the closure witness pins its identity);
            # only its observation reports an unsupported, sealed ingress.
            control = graph.capture_control
            assert control is not None
            original = control.snapshot

            def snapshot():
                phase("blocked")
                return {**original(), "ingress_sealed": True, "supported": False, "error_code": None}
            control.snapshot = snapshot
        elif ARGS.scenario == "hung":
            async def never():
                phase("hanging")
                await asyncio.Event().wait()
            graph.pipeline.features.stop = never
        return graph

    main.run_entry(LIFECYCLE, build=build)
    # Reached only without a receipt (no os._exit).
    phase("returned")


if __name__ == "__main__":
    main_entry()
