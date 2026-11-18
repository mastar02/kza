"""SC4C-Q3: no earlier heap contents in the Chroma files the managed child writes.

The residue tests plant a canary in freed glibc heap chunks (libc malloc/free
through ctypes) and then let a fresh process open an existing Chroma store
with a plain ``count()``: Chroma 1.5.5 initializes the HNSW index of a segment
without a persisted id map again in that process and writes its malloc'd
buffers whole. ``glibc.malloc.arena_max=1`` is a harness device only: it puts
Chroma's own threads on the arena the canary was planted in, which a real
service reaches by sharing and reusing arenas among its many threads. The
control run (no hygiene) must find the canary, so a clean result is never a
planting that did not happen.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import managed_launcher as launcher
from scripts import stage_runtime  # noqa: F401 -- binds the launcher's stage (child_environment)
from src.core import entry_guard
from src.core import heap_hygiene as hh

REPO = Path(__file__).resolve().parents[3]
GLIBC = hh._glibc()
needs_glibc = pytest.mark.skipif(not GLIBC, reason="glibc allocator tunables (the server)")
HYGIENE = "glibc.malloc.perturb=165:glibc.malloc.tcache_count=0"
CANARY = "SC4CQ3CANARY"
DIMENSION = 1024  # BGE-M3, production's embedding size

CREATE_STORE = "\n".join([
    "import random, sys, chromadb",
    "from chromadb.config import Settings",
    "client = chromadb.PersistentClient(path=sys.argv[1], settings=Settings(anonymized_telemetry=False))",
    "collection = client.create_collection('heap-residue', embedding_function=None)",
    f"collection.add(ids=['fact-1'], embeddings=[[random.random() for _ in range({DIMENSION})]],",
    "               documents=['a live fact'])",
    "client.close()",
])

# Opens the store with a count() after planting the canary in freed heap chunks of the sizes the index
# initialization asks for (data_level0: 100 elements; length: 100 floats), then prints its own probe.
OPEN_STORE = "\n".join([
    "import ctypes, json, sys, chromadb",
    "from chromadb.config import Settings",
    "client = chromadb.PersistentClient(path=sys.argv[1], settings=Settings(anonymized_telemetry=False))",
    "libc = ctypes.CDLL(None)",
    "libc.malloc.restype, libc.malloc.argtypes = ctypes.c_void_p, [ctypes.c_size_t]",
    "libc.free.restype, libc.free.argtypes = None, [ctypes.c_void_p]",
    "libc.free(libc.malloc(20 << 20))",  # a freed mapping raises glibc's dynamic mmap threshold
    f"level0 = 100 * ((2 * 16) * 4 + 4 + {DIMENSION} * 4 + 8)",
    "planted, fences = [], []",
    "for size in (400, level0):",
    "    for _ in range(8):",
    "        pointer = libc.malloc(size)",
    f"        ctypes.memmove(pointer, ({CANARY!r}.encode() * (size // {len(CANARY)} + 1))[:size], size)",
    "        planted.append(pointer)",
    "        fences.append(libc.malloc(64))",
    "for pointer in planted:",
    "    libc.free(pointer)",
    "client.get_collection('heap-residue', embedding_function=None).count()",
    "client.close()",
])


def _environment(tunables: str | None) -> dict:
    environment = {key: os.environ[key] for key in ("PATH", "LANG", "LC_ALL", "HOME", "TMPDIR") if key in os.environ}
    if tunables is not None:
        environment[hh.GLIBC_TUNABLES_ENV] = tunables
    return environment


def _python(source: str, *args: str, tunables: str | None, cwd: Path | None = None, extra: dict | None = None):
    return subprocess.run([sys.executable, "-I", "-B", "-c", source, *args], capture_output=True, text=True,
                          env={**_environment(tunables), **(extra or {})}, cwd=cwd, timeout=300, check=False)


def _canaries(store: Path) -> dict[str, int]:
    return {path.name: path.read_bytes().count(CANARY.encode())
            for path in sorted(store.rglob("*")) if path.is_file() and path.name != "chroma.sqlite3"}


def _residue_after_open(tmp_path: Path, name: str, tunables: str) -> dict[str, int]:
    store = tmp_path / name
    created = _python(CREATE_STORE, str(store), tunables=HYGIENE)
    assert created.returncode == 0, created.stderr[-2000:]
    assert _canaries(store) and not any(_canaries(store).values())
    opened = _python(OPEN_STORE, str(store), tunables=tunables)
    assert opened.returncode == 0, opened.stderr[-2000:]
    return _canaries(store)


def _launcher_tunables(environ: dict) -> str:
    child = launcher.child_environment(environ, Path("/r"), {"run_id": "a" * 32}, 5, 6, None)
    return child[hh.GLIBC_TUNABLES_ENV]


# --- the tunables the child gets ------------------------------------------------------------------

def test_with_heap_hygiene_appends_ours_and_replaces_same_named_entries():
    assert hh.with_heap_hygiene(None) == HYGIENE
    assert hh.with_heap_hygiene("") == HYGIENE
    assert hh.with_heap_hygiene("glibc.malloc.arena_max=2:glibc.malloc.perturb=7:glibc.malloc.tcache_count=7"
                                ) == "glibc.malloc.arena_max=2:" + HYGIENE


def test_the_launcher_gives_the_child_the_hygiene_tunables_and_keeps_the_unit_s_others():
    assert launcher.HEAP_HYGIENE_TUNABLES == hh.HEAP_HYGIENE_TUNABLES
    assert launcher.GLIBC_TUNABLES_ENV == hh.GLIBC_TUNABLES_ENV
    assert _launcher_tunables({"PATH": "/bin"}) == HYGIENE
    for inherited in ("glibc.malloc.perturb=0", "glibc.malloc.arena_max=2:glibc.malloc.tcache_count=7", ""):
        assert _launcher_tunables({hh.GLIBC_TUNABLES_ENV: inherited}) == hh.with_heap_hygiene(inherited)


def test_the_ticket_literal_is_the_entry_guard_s():
    assert hh.TICKET_ENV == entry_guard.TICKET_ENV == launcher.TICKET_ENV


# --- the managed entry refuses without them -------------------------------------------------------

def test_the_managed_child_refuses_when_malloc_can_return_old_contents(monkeypatch):
    monkeypatch.setattr(hh, "heap_hygiene_active", lambda: False)
    with pytest.raises(hh.HeapHygieneRefused) as refused:
        hh.refuse_unprotected_managed_entry({hh.TICKET_ENV: "{}"})
    assert refused.value.code == 78 and str(refused.value) == hh.REFUSAL_CODE


@pytest.mark.parametrize("environ, active", [({}, False), ({hh.TICKET_ENV: "{}"}, True),
                                             ({hh.TICKET_ENV: "{}"}, None)],
                         ids=["unmanaged", "managed_with_hygiene", "managed_outside_glibc"])
def test_no_refusal_without_a_ticket_with_hygiene_or_outside_glibc(monkeypatch, environ, active):
    monkeypatch.setattr(hh, "heap_hygiene_active", lambda: active)
    assert hh.refuse_unprotected_managed_entry(environ) is None


@needs_glibc
def test_src_main_refuses_a_managed_start_without_the_tunables_before_arming(tmp_path):
    """The real entry: 'python -m src.main' with a ticket and the default allocator exits 78 at once,
    before arming (nothing is written to KZA_NOTIFY_FD) and before any store opens."""
    read, write = os.pipe()
    try:
        completed = subprocess.run(
            [sys.executable, "-B", "-P", "-m", "src.main"], cwd=REPO, capture_output=True, text=True,
            env={**_environment("glibc.malloc.perturb=165"), "PYTHONPATH": str(REPO),
                 hh.TICKET_ENV: "{}", "KZA_NOTIFY_FD": str(write), "CONFIG_PATH": str(tmp_path / "absent.yaml")},
            pass_fds=(write,), timeout=120, check=False)
    finally:
        os.close(write)
    try:
        os.set_blocking(read, False)
        try:
            notified = os.read(read, 4096)
        except BlockingIOError:
            notified = b""
    finally:
        os.close(read)
    assert completed.returncode == 78, completed.stderr[-2000:]
    assert hh.REFUSAL_CODE in completed.stderr
    assert notified == b""


# --- the probe ------------------------------------------------------------------------------------

PROBE = "\n".join([
    "import json, sys",
    "sys.path.insert(0, sys.argv[1])",
    "from src.core.heap_hygiene import heap_hygiene_active",
    "print(json.dumps(heap_hygiene_active()))",
])


@needs_glibc
@pytest.mark.parametrize("tunables, active", [
    (None, False),
    ("glibc.malloc.perturb=165", False),                      # the per-thread cache returns chunks unfilled
    ("glibc.malloc.tcache_count=0", False),
    ("glibc.malloc.perturb=7:glibc.malloc.tcache_count=0", False),  # another pattern than ours
    (HYGIENE, True),
], ids=["default", "perturb_only", "no_cache_only", "other_pattern", "hygiene"])
def test_the_probe_is_true_only_under_the_hygiene_tunables(tunables, active):
    completed = _python(PROBE, str(REPO), tunables=tunables)
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert json.loads(completed.stdout) is active


@needs_glibc
def test_the_probe_is_true_under_what_the_launcher_sets():
    completed = _python(PROBE, str(REPO), tunables=_launcher_tunables({hh.GLIBC_TUNABLES_ENV: "glibc.malloc.perturb=0"}))
    assert json.loads(completed.stdout) is True


def test_outside_glibc_the_probe_says_unknown(monkeypatch):
    monkeypatch.setattr(hh, "_glibc", lambda: False)
    assert hh.heap_hygiene_active() is None


# --- the residue itself ---------------------------------------------------------------------------

@needs_glibc
def test_an_open_without_hygiene_writes_planted_heap_contents_into_the_index(tmp_path):
    """Control: the planting reaches Chroma's index initialization (what the fix has to prevent)."""
    residue = _residue_after_open(tmp_path, "control", "glibc.malloc.arena_max=1")
    assert residue["data_level0.bin"] > 0


@needs_glibc
def test_an_open_under_the_launcher_s_environment_writes_no_heap_contents(tmp_path):
    """The managed child's environment (what child_environment gives it), plus the harness device."""
    tunables = hh.with_heap_hygiene(None)
    launched = _launcher_tunables({hh.GLIBC_TUNABLES_ENV: "glibc.malloc.arena_max=1"})
    assert launched.startswith("glibc.malloc.arena_max=1:") and launched.endswith(tunables)
    residue = _residue_after_open(tmp_path, "managed", launched)
    assert residue and not any(residue.values()), residue
    # Every byte of the reinitialized buffers is the allocation pattern (not only canary-free).
    store = tmp_path / "managed"
    [level0] = store.rglob("data_level0.bin")
    [length] = store.rglob("length.bin")
    assert set(level0.read_bytes()) == {0x5A} and set(length.read_bytes()) == {0x5A}

