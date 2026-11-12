"""U3b-0d1 compiled child for the import window: the real ``src.main`` module entry.

``src.main`` runs as ``__main__`` (runpy, as ``python -m src.main`` does) from
the compiled runtime, so its module-top arming is the code under test. A
test-only import hook holds the import of ``src.application`` until the
parent's SIGTERM is recorded. Synthetic leaves are prepared when main's
default build calls load_dotenv (wrapped before runpy), then the default
build (CONFIG_PATH and load_config) runs.
Nothing from ``src`` is imported here before runpy starts.
"""
import argparse
import asyncio
import importlib.abc
import json
import os
from pathlib import Path
import runpy
import signal
import sys
import threading
import time

RUNTIME = Path.cwd().resolve()
sys.path[:0] = [str(RUNTIME)]
HOME, MODEL = "http://slot-home.invalid", "http://slot-model.invalid"

parser = argparse.ArgumentParser()
parser.add_argument("--repo", required=True)
parser.add_argument("--phase-file", required=True)
parser.add_argument("--notify-fifo", required=True)
parser.add_argument("--graph-root", required=True)
parser.add_argument("--bindings", required=True)
ARGS = parser.parse_args()
os.environ["KZA_NOTIFY_FD"] = str(os.open(ARGS.notify_fifo, os.O_WRONLY | os.O_NONBLOCK))


def phase(name: str, value: object = "") -> None:
    with open(ARGS.phase_file, "a") as out:
        out.write(f"{name} {value}\n")


def _main_globals() -> dict:
    return sys.modules["__main__"].__dict__


def _serve(ready: threading.Event, urls: dict) -> None:
    from tests.mocks.application_graph import offline_home
    from tests.mocks.restore_application_graph import local_models

    async def serve():
        async with offline_home() as (home_url, _), local_models() as (model_url, _):
            urls.update(home=home_url, model=model_url)
            ready.set()
            await asyncio.Event().wait()

    asyncio.run(serve())


def _prepare_leaves() -> None:
    import pytest
    from tests.mocks.producer_erasure_graph import construction_sdk_guards, nominal_factories
    from tests.mocks.restore_application_graph import MODEL_ENV, TOKEN, sdk_leaves

    root = Path(ARGS.graph_root).resolve()
    patch = pytest.MonkeyPatch()
    patch.setenv("KZA_DASHBOARD_TOKEN", TOKEN)
    patch.setenv(MODEL_ENV, "synthetic-local-key")
    factories, _ = nominal_factories(patch, root)
    assert not construction_sdk_guards(patch)
    sdk_leaves(patch, root / "chroma")
    # Leaf constructors main resolves from its own (still importing) globals.
    _main_globals().update(create_stt=factories.stt, create_tts=factories.tts)
    ready, urls = threading.Event(), {}
    threading.Thread(target=_serve, args=(ready, urls), daemon=True).start()
    assert ready.wait(30), "fixture servers did not start"
    for variable, template in json.loads(Path(ARGS.bindings).read_text()).items():
        os.environ[variable] = template.replace(HOME, urls["home"]).replace(MODEL, urls["model"])
    os.environ["CONFIG_PATH"] = str(root.parents[1] / "config/settings.yaml")


class ImportWindow(importlib.abc.MetaPathFinder):
    """Hold one of src.main's own imports until the stop is recorded."""

    held = False

    def find_spec(self, name, path=None, target=None):
        if name == "src.application" and not self.held:
            self.held = True
            entry = _main_globals().get("_ENTRY")
            assert entry is not None, "src.main did not arm before its imports"
            blocked = signal.pthread_sigmask(signal.SIG_BLOCK, [])
            phase("mask", ",".join(str(int(s)) for s in sorted(blocked) if s in (signal.SIGTERM, signal.SIGINT))
                  or "empty")
            phase("importing")
            deadline = time.monotonic() + 30
            while not entry.stop.requested and time.monotonic() < deadline:
                time.sleep(0.01)
            assert entry.stop.requested, "SIGTERM during import not recorded"
        return None


def _wrap_load_dotenv() -> None:
    import dotenv

    original = dotenv.load_dotenv

    def load_dotenv(*args, **kwargs):
        _prepare_leaves()
        phase("imported")
        return original(*args, **kwargs)

    dotenv.load_dotenv = load_dotenv


phase("pid", os.getpid())
sys.path.append(ARGS.repo)
_wrap_load_dotenv()
sys.meta_path.insert(0, ImportWindow())
runpy.run_module("src.main", run_name="__main__", alter_sys=True)
phase("returned")
