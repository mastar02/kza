"""U3b-0e3: the adoption operator commands on a production-shaped root written by b0866e8's own code.

Adoption design v3 sub-unit U3b-0e3 acceptance, one group per item:
retire-legacy-training (2c), adoption-dry-run (T-D1..T-D4, tmpfs, swap, no
root writes), adoption-prepare (marked slot, bound plan, no CAS, audit hook),
adoption-commit (binding re-verified before the CAS; records, seal, head,
marker, 'current'), the adoption kind and the terminal aborted head,
adoption-abort (section 7) and CONTROLLER_EPOCH 2. Every command also
refuses as the wrong user, on the wrong root and when run twice.

The R1 archive is built from this working tree (clone overlay, as the
compiled launcher rehearsals do); the schema-only child runs from it.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import sqlite3
import stat
import subprocess
import sys
import tarfile
import time

import pytest

from scripts import adoption
from scripts import adoption_commands as commands
from scripts import managed_launcher as launcher
from scripts import stage_runtime as stage
from scripts.build_runtime import build_runtime
from tests.mocks import adoption_legacy as fixture
from tests.mocks import launcher_rehearsal

REPOSITORY = Path(__file__).resolve().parents[2]
CONTAINER = fixture.C_SLOT
DECISION = "u3b-0e-P1-D-20261112"
WINDOW = "r1-window-20261120"
VOICE = launcher.UNIT


# --- fixtures --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    """The R1 archive of this working tree (controller epoch 2) and an epoch-1 copy of it."""
    base = tmp_path_factory.mktemp("r1-artifact")
    clone = base / "clone"
    subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout", str(REPOSITORY), str(clone)],
                   check=True, timeout=120)
    revision = launcher_rehearsal.overlay_commit(REPOSITORY, clone)
    archive = base / "r1.tar.gz"
    manifest = build_runtime(clone, archive, revision)
    old = base / "epoch1.tar.gz"
    _rewrite_release(archive, old, controller_epoch=1)
    return {"archive": archive, "epoch1": old, "commit": manifest["git_commit"]}


def _rewrite_release(source: Path, destination: Path, **changes) -> None:
    buffer = io.BytesIO()
    with tarfile.open(source, "r:gz") as bundle, tarfile.open(fileobj=buffer, mode="w") as output:
        for member in bundle:
            data = bundle.extractfile(member).read()
            if member.name == "release.json":
                data = json.dumps(dict(json.loads(data), **changes), indent=2).encode()
                member.size = len(data)
            output.addfile(member, io.BytesIO(data))
    destination.write_bytes(gzip.compress(buffer.getvalue()))


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    return fixture.b0866e8_tree(tmp_path_factory.mktemp("b0866e8") / "tree")


class Units:
    """ServiceManager double: per-unit properties, reverse dependencies, enable and reload log."""

    def __init__(self) -> None:
        self.properties = {
            VOICE: {"LoadState": "loaded", "ActiveState": "inactive", "UnitFileState": "disabled",
                    "InvocationID": "", "InactiveEnterTimestampMonotonic": "1000"},
            "kza-audio-watchdog.timer": {"LoadState": "loaded", "ActiveState": "inactive",
                                         "UnitFileState": "enabled", "InvocationID": "",
                                         "InactiveEnterTimestampMonotonic": "900"},
            "kza-llm-ik.service": {"LoadState": "loaded", "ActiveState": "active",
                                   "ExecStart": "{ path=/usr/bin/llama-server ; argv[]=/usr/bin/llama-server -m "
                                                "/srv/models/qwen.gguf ; ignore_errors=no }",
                                   "Environment": "", "WorkingDirectory": "/srv"},
        }
        self.enabled: list[str] = []
        self.reloads = 0

    def user_show(self, unit: str, properties: tuple[str, ...]) -> dict:
        values = self.properties.get(unit, {"LoadState": "not-found"})
        return {name: values.get(name, "") for name in properties}

    def daemon_reload(self) -> bool:
        self.reloads += 1
        return True

    def user_enable(self, unit: str) -> bool:
        self.enabled.append(unit)
        self.properties[unit]["UnitFileState"] = "enabled"
        return True


class World:
    """One production-shaped root: the two legacy slots, 'current' on 63ea8d3, the checkout's models."""

    def __init__(self, base: Path, tree: Path, artifacts: dict, **build) -> None:
        self.base = base
        self.layout = fixture.build_legacy(base / "legacy", tree, **build)
        self.root = base / "legacy" / "releases"
        runtime = self.root / "slots" / CONTAINER / "runtime"
        runtime.mkdir()
        (runtime / "release.json").write_text(json.dumps({"git_commit": "63ea8d3" + "0" * 33, "files": {}}))
        (self.root / "current").symlink_to(Path("slots") / CONTAINER / "runtime")
        self.scratch = base / "tmpfs"
        self.scratch.mkdir(mode=0o700)
        self.paths = commands.ProductionPaths(os.fspath(self.layout.models_root), adoption.PRODUCTION_LOGS,
                                              adoption.PRODUCTION_METRICS)
        self.units = Units()
        self.archive = artifacts["archive"]
        self.artifacts = artifacts
        self.unit_dir = base / "unit.d"
        self.unit_dir.mkdir()

    def retire(self, **options):
        return commands.retire_legacy_training(self.root, decision_id=options.pop("decision_id", DECISION),
                                               container_slot=CONTAINER, manager=self.units, paths=self.paths,
                                               **options)

    def dry_run(self, slot: str = "r1", **options):
        return commands.adoption_dry_run(self.root, container_slot=CONTAINER,
                                         archive=options.pop("archive", self.archive), slot=slot,
                                         scratch_parent=options.pop("scratch_parent", self.scratch),
                                         python=sys.executable, manager=self.units, paths=self.paths,
                                         tmpfs_required=options.pop("tmpfs_required", False), **options)

    def prepare(self, slot: str = "r1", **options):
        return commands.adoption_prepare(self.root, container_slot=CONTAINER,
                                         archive=options.pop("archive", self.archive), slot=slot,
                                         decision_id=options.pop("decision_id", WINDOW),
                                         scratch_parent=self.scratch, python=sys.executable, manager=self.units,
                                         paths=options.pop("paths", self.paths),
                                         tmpfs_required=options.pop("tmpfs_required", False), **options)

    def install_controller(self):
        return launcher.install_controller(self.root, self.archive)

    def commit(self, plan_sha256: str):
        return commands.adoption_commit(self.root, confirm=plan_sha256, manager=self.units, paths=self.paths)

    def abort(self):
        return commands.adoption_abort(self.root, manager=self.units, unit_dir=self.unit_dir, paths=self.paths)

    def adopt(self, slot: str = "r1") -> dict:
        prepared = self.prepare(slot)
        if not (self.root / "admission" / "controller.json").exists():
            self.install_controller()
        result = self.commit(prepared["plan_sha256"])
        return dict(result, plan_sha256=prepared["plan_sha256"])

    def plan(self, plan_sha256: str) -> dict:
        return json.loads((stage.adoption_directory(self.root) / ("plan-" + plan_sha256 + ".json")).read_text())

    def head(self):
        return stage.read_admission_head(self.root)


@pytest.fixture
def world(tmp_path, tree, artifacts):
    return World(tmp_path.resolve(), tree, artifacts)


@pytest.fixture
def adapter_world(tmp_path, tree, artifacts):
    return World(tmp_path.resolve(), tree, artifacts, adapters=True)


def tree_manifest(root: Path) -> dict:
    """T-D1: (kind, inode, size, mtime_ns, mode, link, xattrs) of every entry and of ``root``; atime ignored."""
    found = {}
    listxattr = getattr(os, "listxattr", None)
    for path in [Path(root), *sorted(Path(root).rglob("*"))]:
        info = path.lstat()
        attributes = None
        if listxattr is not None:
            try:
                attributes = sorted(listxattr(path, follow_symlinks=False))
            except OSError:
                attributes = None
        found[path.relative_to(root).as_posix()] = (
            stat.S_IFMT(info.st_mode), info.st_ino, info.st_size, info.st_mtime_ns, info.st_mode,
            os.readlink(path) if path.is_symlink() else None,
            None if path.is_symlink() or not path.is_file() else hashlib.sha256(path.read_bytes()).hexdigest(),
            attributes)
    return found


def _source_tree(world: World) -> dict:
    return {name: tree_manifest(world.base / "legacy" / name) for name in ("releases", "kza", "logs")}


# --- T-D2: a process-wide audit hook, active only inside ``watching`` ---------------------------------

_WATCH = {"active": False, "base": None, "allowed": None, "seen": []}
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
_PATH_EVENTS = {"os.mkdir": (0,), "os.rename": (0, 1), "os.replace": (0, 1), "os.remove": (0,),
                "os.rmdir": (0,), "os.symlink": (1,), "os.link": (1,), "os.chmod": (0,), "os.utime": (0,),
                "os.truncate": (0,), "shutil.rmtree": (0,)}


def _watch_hook(event, args):
    if not _WATCH["active"]:
        return
    paths = ()
    if event == "open" and args:
        mode, flags = (args[1] if len(args) > 1 else None), (args[2] if len(args) > 2 else 0)
        if (isinstance(mode, str) and any(letter in mode for letter in "wax+")) or (
                type(flags) is int and flags & _WRITE_FLAGS):
            paths = (args[0],)
    elif event == "sqlite3.connect" and args:
        paths = (args[0],)
    elif event in _PATH_EVENTS:
        paths = tuple(args[index] for index in _PATH_EVENTS[event] if index < len(args))
    for value in paths:
        path = adoption._event_path(value)
        if path is None:
            continue
        inside = path == _WATCH["base"] or path.startswith(_WATCH["base"] + "/")
        allowed = any(path == item or path.startswith(item + "/") for item in _WATCH["allowed"])
        if inside and not allowed:
            _WATCH["seen"].append((event, os.path.relpath(path, _WATCH["base"])))


sys.addaudithook(_watch_hook)


class watching:
    """Record write-mode opens, path mutations and SQLite connects under ``base`` outside ``allowed``."""

    def __init__(self, base: Path, *allowed: Path) -> None:
        self.base, self.allowed = os.path.realpath(base), tuple(os.path.realpath(item) for item in allowed)

    def __enter__(self):
        _WATCH.update(active=True, base=self.base, allowed=self.allowed, seen=[])
        return _WATCH["seen"]

    def __exit__(self, *exc):
        _WATCH["active"] = False


def _fake_proc(tmp_path: Path, monkeypatch, *, active: bool) -> None:
    swaps = tmp_path / "swaps"
    swaps.write_text("Filename Type Size Used Priority\n" + ("/swap.img file 8388604 0 -2\n" if active else ""))
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       131072000 kB\nSwapTotal:       8388604 kB\nSwapFree:        8388604 kB\n"
                       if active else "MemTotal:       131072000 kB\nSwapTotal:             0 kB\nSwapFree:              0 kB\n")
    monkeypatch.setattr(commands, "PROC_SWAPS", os.fspath(swaps))
    monkeypatch.setattr(commands, "PROC_MEMINFO", os.fspath(meminfo))


# --- retire-legacy-training (W1b, P1=D, 2c) ----------------------------------------------------------

def test_retire_requires_a_valid_decision_id_before_anything(adapter_world):
    before = _source_tree(adapter_world)
    for decision in ("", "x", "decision with spaces", None):
        with pytest.raises(commands.CommandRefusal, match="^runtime_admission_retire_decision$"):
            adapter_world.retire(decision_id=decision)
    assert _source_tree(adapter_world) == before


def test_retire_deletes_only_the_lora_roots_with_a_content_free_record(adapter_world):
    world = adapter_world
    models = world.layout.models_root
    outside_before = commands.models_manifest(models, models / "lora_adapters")
    whisper = tree_manifest(models / "whisper-v3-turbo")
    result = world.retire()
    assert not (models / "lora_adapters").exists()
    assert commands.models_manifest(models, models / "lora_adapters") == outside_before
    assert tree_manifest(models / "whisper-v3-turbo") == whisper
    path = stage.adoption_directory(world.root) / result["record"]
    raw = path.read_bytes()
    record = json.loads(raw)
    assert set(record) == commands.RETIREMENT_KEYS
    assert record["decision_id"] == DECISION and record["kind"] == "retire_legacy_training"
    assert [item["class"] for item in record["roots"]] == ["lora_adapters", "lora_adapters/nightly",
                                                           "lora_adapters/whisper"]
    assert all(item["absence"] == "enoent" for item in record["roots"])
    assert record["roots"][0]["pre"] == {"exists": True, "files": 1, "bytes": 4}
    assert record["models_manifest_sha256"]["before"] == record["models_manifest_sha256"]["after"]
    assert b"custom_adapter" not in raw and b"safetensors" not in raw  # content-free: no file names
    assert record["units_checked"] == [VOICE]
    with pytest.raises(ValueError, match="runtime_admission_immutable_conflict"):
        stage._atomic_record(path, dict(record, decision_id="other"), immutable=True)
    # Twice: nothing left to retire.
    with pytest.raises(commands.CommandRefusal, match="^runtime_admission_retire_not_needed$"):
        world.retire()


def _mount_crossing(world, monkeypatch):
    inner = world.layout.models_root / "lora_adapters" / "custom_adapter"
    original = os.lstat

    def lstat(path, *args, **kwargs):
        info = original(path, *args, **kwargs)
        if os.fspath(path) == os.fspath(inner):
            values = list(info)
            values[stat.ST_DEV] = info.st_dev + 1
            return os.stat_result(values)
        return info
    monkeypatch.setattr(commands.os, "lstat", lstat)


@pytest.mark.parametrize("case,code", [
    ("symlink", "runtime_admission_retire_special"),
    ("fifo", "runtime_admission_retire_special"),
    ("mount", "runtime_admission_retire_mount"),
    ("root_link", "runtime_admission_retire_root_special"),
    ("model_path", "runtime_admission_retire_model_under_root:extra_models.adapter_base"),
    ("lora_unit", "runtime_admission_retire_unit_under_root:kza-llm-ik.service"),
])
def test_retire_scope_guards_refuse_before_any_deletion(adapter_world, monkeypatch, case, code):
    world = adapter_world
    models = world.layout.models_root
    lora = models / "lora_adapters"
    units = ()
    if case == "symlink":
        (lora / "custom_adapter" / "link").symlink_to(models / "whisper-v3-turbo" / "model.bin")
    elif case == "fifo":
        os.mkfifo(lora / "nightly-fifo")
    elif case == "mount":
        _mount_crossing(world, monkeypatch)
    elif case == "root_link":
        os.rename(lora, models / "lora-real")
        lora.symlink_to("lora-real")
    elif case == "model_path":
        settings = world.layout.legacy_state / "config" / "settings.yaml"
        settings.write_text(settings.read_text() + "\nextra_models:\n  adapter_base: ./models/lora_adapters/base.gguf\n")
    else:
        world.units.properties["kza-llm-ik.service"]["ExecStart"] = (
            "{ path=/usr/bin/llama-server ; argv[]=/usr/bin/llama-server --lora "
            + os.fspath(lora / "custom_adapter") + " ; ignore_errors=no }")
        units = ("kza-llm-ik.service",)
    before = tree_manifest(models)
    with pytest.raises(commands.CommandRefusal, match="^" + code + "$"):
        world.retire(units=units)
    assert tree_manifest(models) == before
    assert not stage.adoption_directory(world.root).exists() or not list(
        stage.adoption_directory(world.root).glob("retire-*"))


def test_retire_refuses_when_anything_outside_the_roots_changed(adapter_world, monkeypatch):
    world = adapter_world
    original = stage.make_writable_then_remove

    def remove_too_much(path):
        original(path)
        (world.layout.models_root / "whisper-v3-turbo" / "model.bin").unlink()
    monkeypatch.setattr(stage, "make_writable_then_remove", remove_too_much)
    with pytest.raises(commands.CommandRefusal, match="^runtime_admission_retire_outside_changed$"):
        world.retire()
    assert not list(stage.adoption_directory(world.root).glob("retire-*"))


@pytest.mark.parametrize("state,code", [({"ActiveState": "active"}, "runtime_admission_adoption_unit_active"),
                                        ({"UnitFileState": "enabled"}, "runtime_admission_adoption_unit_enabled")])
def test_retire_needs_the_unit_stopped_and_disabled(adapter_world, state, code):
    adapter_world.units.properties[VOICE].update(state)
    with pytest.raises(commands.CommandRefusal, match="^" + code + "$"):
        adapter_world.retire()
    assert (adapter_world.layout.models_root / "lora_adapters" / "custom_adapter").is_dir()


# --- adoption-dry-run (T-D1..T-D4, P4) ---------------------------------------------------------------

def test_dry_run_reads_only_reports_swap_sizes_and_census(world, tmp_path, monkeypatch):
    """T-D1 (tree manifest of root and sources), T-D2 (audit hook), T-D4 (success), swap and sizes."""
    _fake_proc(tmp_path, monkeypatch, active=True)
    before = _source_tree(world)
    with watching(world.base, world.scratch) as seen:
        code, report = world.dry_run()
    assert seen == []
    assert _source_tree(world) == before
    assert not (world.root / "admission").exists() and not (world.root / ".admission.lock").exists()
    assert not (world.root / ".stage.lock").exists() and not (world.root / "adoption").exists()
    assert code == 0 and report["problems"] == []
    assert report["swap"] == {"known": True, "active": True, "devices": 1, "total_kib": 8388604,
                              "free_kib": 8388604, "exposure": "tmpfs scratch pages may be written to swap"}
    assert report["sizes"]["files"] > 0 and report["sizes"]["bytes"] > 0
    assert report["free"]["scratch_ok"] is True
    assert report["scratch"] == {"kind": "other", "removed": True}
    assert os.listdir(world.scratch) == []
    assert report["journal_files"] == 1
    assert report["adoption"]["guard"]["refused"] == 0
    assert report["legacy_path_census"]["legacy_checkout_targets"] == 0
    assert report["units"]["reverse_dependencies"] == []
    assert report["code_index"] == {"unit": commands.CODE_INDEX_UNIT, "state": "absent"}


def test_dry_run_reports_no_swap(world, tmp_path, monkeypatch):
    _fake_proc(tmp_path, monkeypatch, active=False)
    _code, report = world.dry_run()
    assert report["swap"]["active"] is False and report["swap"]["exposure"] == "none"


def test_dry_run_leaves_a_pending_r_a_untouched(world):
    """T-D3: a running head with its receipt and a -wal beside a slot database stay byte-identical."""
    admission = world.root / "admission"
    (admission / "receipts").mkdir(parents=True)
    run_id = "a" * 32
    (admission / "head.json").write_text(json.dumps({"schema_version": 1, "generation": 3, "binding_id": "b" * 64,
                                                     "status": "running", "run_id": run_id}))
    (admission / "receipts" / (run_id + ".json")).write_text('{"schema_version": 1}')
    pending = world.root / "slots" / "managed" / "state" / "data"
    pending.mkdir(parents=True)
    with sqlite3.connect(pending / "memory.db") as db:
        db.execute("CREATE TABLE t (x)")
    db.close()
    (pending / "memory.db-wal").write_bytes(b"uncheckpointed frames of the run the receipt closed")
    before = _source_tree(world)
    with watching(world.base, world.scratch) as seen:
        code, report = world.dry_run()
    assert seen == []
    assert _source_tree(world) == before
    assert code == 78 and "runtime_admission_adoption_head_present" in report["problems"]
    assert report["head"] == {"generation": 3, "status": "running", "aborted": False}
    assert os.listdir(world.scratch) == []


def test_dry_run_removes_its_scratch_after_a_refusal(world, monkeypatch):
    """T-D4 (refusal): the library refuses mid-pipeline; the report carries the census so far."""
    def refuse(*_args, **_kwargs):
        raise adoption.AdoptionRefusal("runtime_admission_adoption_residue:voice_profile")
    monkeypatch.setattr(adoption, "relocate", refuse)
    code, report = world.dry_run()
    assert code == 78 and report["problems"] == ["runtime_admission_adoption_residue:voice_profile"]
    assert report["adoption"]["refused"] == "runtime_admission_adoption_residue:voice_profile"
    assert report["adoption"]["source_files"] > 0  # the content-free census so far
    assert report["scratch"]["removed"] is True and os.listdir(world.scratch) == []


_SIGTERM_DRIVER = r"""
import json, os, signal, sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from scripts import adoption, adoption_commands as commands
spec = json.loads(sys.argv[2])
class Units:
    def user_show(self, unit, properties):
        values = {"LoadState": "loaded", "ActiveState": "inactive", "UnitFileState": "disabled"} \
            if unit == "kza-voice.service" else {"LoadState": "not-found"}
        return {name: values.get(name, "") for name in properties}
def wait(*args, **kwargs):
    Path(spec["ready"]).write_text("1")
    time.sleep(120)
adoption.relocate = wait
signal.signal(signal.SIGTERM, commands._terminate)
paths = commands.ProductionPaths(spec["models"], adoption.PRODUCTION_LOGS, adoption.PRODUCTION_METRICS)
try:
    commands.adoption_dry_run(spec["root"], container_slot=spec["container"], archive=spec["archive"], slot="r1",
                              scratch_parent=spec["scratch"], python=sys.executable, manager=Units(),
                              paths=paths, tmpfs_required=False)
except commands._Terminated:
    raise SystemExit(3)
raise SystemExit(0)
"""


def test_dry_run_removes_its_scratch_after_sigterm(world):
    """T-D4 (SIGTERM): the handler unwinds through the dry-run's finally."""
    ready = world.base / "ready"
    spec = {"root": os.fspath(world.root), "container": CONTAINER, "archive": os.fspath(world.archive),
            "scratch": os.fspath(world.scratch), "models": os.fspath(world.layout.models_root),
            "ready": os.fspath(ready)}
    process = subprocess.Popen([sys.executable, "-c", _SIGTERM_DRIVER, os.fspath(REPOSITORY), json.dumps(spec)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    deadline = time.monotonic() + 120
    while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert ready.exists(), process.communicate()[1][-2000:]
    assert os.listdir(world.scratch) != []  # the private scratch exists while the dry run waits
    process.send_signal(signal.SIGTERM)
    _out, error = process.communicate(timeout=60)
    assert process.returncode == 3, error[-2000:]
    assert os.listdir(world.scratch) == []


def test_dry_run_requires_tmpfs_in_production(world):
    """P4=A: outside a tmpfs under /run/user/<uid> the dry run refuses before creating anything."""
    with pytest.raises(commands.CommandRefusal, match="^runtime_admission_adoption_scratch_not_tmpfs$"):
        world.dry_run(tmpfs_required=True)
    assert os.listdir(world.scratch) == []
    assert commands.scratch_kind(world.scratch) == "other"


def test_dry_run_surfaces_the_legacy_path_census(tmp_path, tree, artifacts, tmp_path_factory):
    """0e2 F2: a pre-10-28 (legacy checkout) target locator is reported by participant and refused."""
    pre_slot = fixture.b0866e8_tree(tmp_path_factory.mktemp("30da279") / "tree", fixture.PRE_SLOT)
    world = World(tmp_path.resolve(), tree, artifacts, pre_slot=pre_slot)
    code, report = world.dry_run()
    assert code == 78
    assert "runtime_admission_adoption_foreign_target:semantic_index:legacy_checkout" in report["problems"]
    census = report["legacy_path_census"]
    assert census["legacy_checkout_targets"] == 1 and census["needs_reviewed_rule"] == ["semantic_index"]
    assert census["descriptor_classes"]["semantic_index"] == {"legacy_checkout": 1, "legacy_state": 1}


def test_dry_run_reports_the_erasure_journal_file_count(world):
    """0d4a F4: a stray copy of the erasure journal is counted and named by its R-b refusal code."""
    data = world.layout.legacy_state / "data"
    (data / "stray-copy.db").write_bytes((data / "memory_privacy.db").read_bytes())
    code, report = world.dry_run()
    assert code == 78
    assert report["journal_files"] == 2
    assert "runtime_admission_recovery_erasure_journal_ambiguous" in report["problems"]


def test_dry_run_reports_reverse_dependencies_and_the_code_index_config(world):
    world.units.properties[VOICE]["WantedBy"] = "default.target kza-dashboard.service"
    world.units.properties["kza-audio-watchdog.timer"]["Triggers"] = "kza-audio-watchdog.service"
    world.units.properties["kza-audio-watchdog.service"] = {
        "LoadState": "loaded", "ExecStart": "{ path=/usr/bin/python3 ; argv[]=/usr/bin/python3 tools/watchdog.py "
                                            "; ignore_errors=no }",
        "Environment": "PYTHONPATH=" + os.fspath(world.root / "current"), "WorkingDirectory": "/srv"}
    world.units.properties["kza-backup.timer"] = {"LoadState": "loaded", "ActiveState": "inactive",
                                                  "ExecStart": "", "Environment": "", "WorkingDirectory": ""}
    world.units.properties[commands.CODE_INDEX_UNIT] = {
        "LoadState": "loaded", "Environment": "CONFIG_PATH=config/settings.yaml",
        "WorkingDirectory": os.fspath(world.layout.container)}
    code, report = world.dry_run(timers=("kza-audio-watchdog.timer", "kza-backup.timer"))
    assert code == 78
    assert report["timer_dispositions"] == [
        {"unit": "kza-audio-watchdog.timer", "references": ["managed_code"], "disposition": "review"},
        {"unit": "kza-backup.timer", "references": [], "disposition": "restart unchanged"}]
    assert report["units"]["reverse_dependencies"] == ["kza-dashboard.service"]
    assert report["code_index"]["config_class"] == "legacy_state"
    assert {"runtime_admission_adoption_reverse_dependency",
            "runtime_admission_adoption_code_index_reads_slot"} <= set(report["problems"])


# --- adoption-prepare (W2) ---------------------------------------------------------------------------

def test_prepare_builds_a_marked_slot_and_a_fully_bound_plan(world):
    current = os.readlink(world.root / "current")
    source = _source_tree(world)
    source.pop("releases")
    legacy = tree_manifest(world.layout.legacy_state)
    result = world.prepare()
    plan = world.plan(result["plan_sha256"])
    assert stage._digest(plan) == result["plan_sha256"]
    state = world.root / "slots" / "r1" / "state"
    marker = json.loads((state / adoption.COPY_MARKER).read_text())
    assert marker == {"schema_version": 1, "plan_sha256": plan["prepare_id"]}
    # No CAS, 'current' unchanged, nothing selected, sources untouched.
    assert world.head() is None and not (world.root / "admission" / "head.json").exists()
    assert os.readlink(world.root / "current") == current
    assert {name: tree_manifest(world.base / "legacy" / name) for name in ("kza", "logs")} == source
    assert tree_manifest(world.layout.legacy_state) == legacy
    assert not (world.root / "adoption").exists() and os.listdir(world.scratch) == []
    # The plan binds the full binding and every section 5 item.
    observation: dict = {}
    assert plan["prepared"]["binding"] == stage._binding_snapshot(world.root, "r1", artifacts=observation)
    assert plan["prepared"]["artifact_observation_sha256"] == stage._digest(
        observation[os.fspath(world.root / "slots" / "r1" / "runtime")])
    assert plan["prepared"]["manifest_sha256"] == adoption.manifest_digest(commands.prepared_manifest(state))
    assert plan["source"]["manifest_sha256"] == adoption.manifest_digest(
        adoption.source_manifest(world.layout, plan["source"]["metrics_path"]))
    assert set(plan) >= {"readlinks", "current_readlink", "expected_head", "archive", "retirement", "tables",
                         "confirm_foreign", "p_answers", "units", "window_decision_id", "prepared"}
    assert plan["archive"]["sha256"] == hashlib.sha256(world.archive.read_bytes()).hexdigest()
    assert plan["units"]["units"][VOICE]["InactiveEnterTimestampMonotonic"] == "1000"
    assert plan["p_answers"] == list(commands.P_ANSWERS) and plan["window_decision_id"] == WINDOW
    assert plan["report"]["guard"]["refused"] == 0 and plan["report"]["guard"]["connects_observed"] > 0
    # The runtime is read-only (0e1) and the slot is never served while marked.
    with pytest.raises(PermissionError):
        (world.root / "slots" / "r1" / "runtime" / "probe").write_text("x")
    assert adoption.COPY_MARKER not in json.dumps(result)


def test_prepare_runs_under_the_source_audit_hook(world, monkeypatch):
    original = adoption.recover_copies

    def touch_source(state):
        sqlite3.connect(world.layout.container / "data" / "lists.db").close()
        return original(state)
    monkeypatch.setattr(adoption, "recover_copies", touch_source)
    legacy = tree_manifest(world.layout.legacy_state)
    with pytest.raises(adoption.AdoptionRefusal, match="^runtime_admission_adoption_source_access:sqlite3.connect$"):
        world.prepare()
    assert tree_manifest(world.layout.legacy_state) == legacy
    assert not (world.root / "slots" / "r1").exists() and not (world.root / "adoption").exists()
    assert os.listdir(world.scratch) == []
    assert not list(stage.adoption_directory(world.root).glob("plan-*"))


@pytest.mark.parametrize("change,code", [
    (("voice", "ActiveState", "active"), "runtime_admission_adoption_unit_active"),
    (("voice", "UnitFileState", "enabled"), "runtime_admission_adoption_unit_enabled"),
    (("voice", "WantedBy", "multi-user.target"), "runtime_admission_adoption_reverse_dependency"),
    (("timer", "ActiveState", "active"), "runtime_admission_adoption_timer_active:kza-audio-watchdog.timer"),
])
def test_prepare_refuses_unless_the_window_stopped_everything(world, change, code):
    unit, key, value = change
    world.units.properties[VOICE if unit == "voice" else "kza-audio-watchdog.timer"][key] = value
    with pytest.raises(commands.CommandRefusal, match="^" + code + "$"):
        world.prepare(timers=("kza-audio-watchdog.timer",))
    assert not (world.root / "slots" / "r1").exists() and not (world.root / "admission").exists()


def test_prepare_refuses_twice_and_an_existing_slot(world):
    world.prepare()
    with pytest.raises(commands.CommandRefusal, match="^runtime_admission_adoption_prepared_pending$"):
        world.prepare("r2")
    with pytest.raises(commands.CommandRefusal, match="^runtime_admission_adoption_prepared_pending$"):
        world.prepare("r1")
    assert not (world.root / "slots" / "r2").exists()


def test_prepare_refuses_files_under_a_lora_root_without_retirement(adapter_world):
    with pytest.raises(adoption.AdoptionRefusal, match="^runtime_admission_adoption_legacy_training_present$"):
        adapter_world.prepare()
    assert not (adapter_world.root / "slots" / "r1").exists()


def test_prepare_refuses_the_models_root_mismatch_and_an_old_archive(world):
    """N10 (models_root equals the container's resolved models link) and CONTROLLER_EPOCH 2."""
    other = commands.ProductionPaths("/home/kza/kza/models", adoption.PRODUCTION_LOGS, adoption.PRODUCTION_METRICS)
    with pytest.raises(commands.CommandRefusal, match="^runtime_admission_adoption_models_root$"):
        world.prepare(paths=other)
    with pytest.raises(commands.CommandRefusal, match="^runtime_admission_adoption_archive_epoch$"):
        world.prepare(archive=world.artifacts["epoch1"])
    with pytest.raises(commands.CommandRefusal, match="^runtime_admission_adoption_decision$"):
        world.prepare(decision_id="no")
    with pytest.raises(commands.CommandRefusal, match="^runtime_admission_adoption_scratch_not_tmpfs$"):
        world.prepare(tmpfs_required=True)  # P4=A: production prepare also drops on tmpfs
    assert not (world.root / "admission").exists() and not (world.root / "slots" / "r1").exists()


# --- adoption-commit (W4) ----------------------------------------------------------------------------

def test_commit_writes_the_records_seal_head_and_publishes_current(adapter_world):
    world = adapter_world
    retired = world.retire()
    prepared = world.prepare()
    plan = world.plan(prepared["plan_sha256"])
    world.install_controller()
    legacy = tree_manifest(world.layout.legacy_state)
    result = world.commit(prepared["plan_sha256"])
    head = world.head()
    assert head == {"schema_version": 1, "generation": 1, "binding_id": head["binding_id"], "status": "closed",
                    "run_id": None}
    companion = stage.read_manual_head(world.root, head)
    assert companion["kind"] == "adoption" and companion["previous"] is None
    seal = json.loads(stage.adoption_seal_path(world.root, companion["authority_seal"]).read_text())
    assert set(seal) == stage.ADOPTION_SEAL_KEYS and result["seal"] == companion["authority_seal"]
    state = world.root / "slots" / "r1" / "state"
    retirement_path = stage.adoption_directory(world.root) / retired["record"]
    assert seal["plan_sha256"] == prepared["plan_sha256"]
    assert seal["archive_sha256"] == hashlib.sha256(world.archive.read_bytes()).hexdigest()
    assert seal["controller_commit"] == world.artifacts["commit"]
    assert seal["source_manifest_sha256"] == plan["source"]["manifest_sha256"]
    assert seal["prepared_manifest_sha256"] == plan["prepared"]["manifest_sha256"]
    assert seal["committed_manifest_sha256"] == adoption.manifest_digest(
        [entry for entry in commands.prepared_manifest(state) if entry[0] != ".kza-state-binding.json"])
    assert seal["state_manifest_sha256"] == stage.state_manifest_sha256(state)
    assert seal["closed_state_sha256"] == plan["prepared"]["closed_state_sha256"]
    assert seal["artifact_observation_sha256"] == plan["prepared"]["artifact_observation_sha256"]
    assert seal["transform_sha256"] == plan["tables"]["transform_sha256"] and seal["tables"] == adoption.table_digests()
    assert seal["retirement_record_sha256"] == hashlib.sha256(retirement_path.read_bytes()).hexdigest()
    assert seal["window_decision_id"] == WINDOW and seal["p_answers"] == list(commands.P_ANSWERS)
    assert seal["previous_abort"] is None and seal["closed_head"] == head
    record = json.loads(stage.adoption_record_path(world.root, head).read_text())
    assert record["plan_sha256"] == prepared["plan_sha256"] and record["seal_sha256"] == result["seal"]
    assert retired["record"] in record["legacy_training"] and DECISION in record["legacy_training"]
    assert not (state / adoption.COPY_MARKER).exists()
    assert json.loads((state / ".kza-state-binding.json").read_text())["binding_id"] == head["binding_id"]
    assert os.readlink(world.root / "current") == "slots/r1/runtime"
    assert tree_manifest(world.layout.legacy_state) == legacy
    # Twice: refused, nothing changes.
    with pytest.raises(commands.CommandRefusal, match="^runtime_admission_adoption_already_committed$"):
        world.commit(prepared["plan_sha256"])
    assert world.head() == head
    # The adoption head serves: activation selects it again and begin_run mints a ticket.
    with stage.AdmissionController(world.root) as controller:
        controller.activate("r1", managed=True)
        _binding, running, previous, _ticket = controller.begin_run()
    assert running["generation"] == 3 and previous["kind"] == "selection"


def _touch_same_stat(path: Path) -> None:
    info = path.stat()
    data = path.read_bytes()
    path.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns))


@pytest.mark.parametrize("item", ["source", "source_same_stat", "prepared", "prepared_same_stat", "units",
                                  "retirement", "current",
                                  "controller", "controller_commit", "archive", "layout"])
def test_commit_refuses_any_binding_difference_before_the_cas(adapter_world, item):
    world = adapter_world
    if item == "archive":  # a private copy: the module's archive stays intact
        copy = world.base / "r1.tar.gz"
        copy.write_bytes(world.archive.read_bytes())
        world.archive = copy
    world.retire()
    prepared = world.prepare()
    if item == "controller_commit":  # an epoch-2 controller of another commit
        other = world.base / "other.tar.gz"
        _rewrite_release(world.archive, other, git_commit="f" * 40)
        launcher.install_controller(world.root, other)
    elif item != "controller":
        world.install_controller()
    state = world.root / "slots" / "r1" / "state"
    if item == "source":
        with open(world.layout.legacy_state / "logs" / "voice_assistant.log", "a") as stream:
            stream.write("late line\n")
    elif item == "source_same_stat":
        _touch_same_stat(world.layout.legacy_state / "logs" / "voice_assistant.log")
    elif item == "prepared":
        with open(state / "config" / "settings.yaml", "a") as stream:
            stream.write("# edited\n")
    elif item == "prepared_same_stat":  # only the full-file hash sees it (not the binding, not lstat)
        _touch_same_stat(state / "data" / "ambient_audio" / "policy_v1" / "live.flac")
    elif item == "units":
        world.units.properties[VOICE]["InactiveEnterTimestampMonotonic"] = "2000"
    elif item == "retirement":
        (world.layout.models_root / "lora_adapters" / "nightly").mkdir(parents=True)
        (world.layout.models_root / "lora_adapters" / "nightly" / "adapter.bin").write_bytes(b"x")
    elif item == "current":
        os.unlink(world.root / "current")
        (world.root / "current").symlink_to(world.root / "slots" / CONTAINER / "runtime")
    elif item == "archive":
        with open(world.archive, "ab") as stream:
            stream.write(b"\0")
    elif item == "layout":
        os.unlink(world.layout.container / "logs")
        os.symlink("../../" + fixture.B_SLOT + "/state/./logs", world.layout.container / "logs")  # same target
    with pytest.raises(commands.CommandRefusal, match="^runtime_admission_adoption_plan_mismatch:"
                       + item.removesuffix("_same_stat").removesuffix("_commit") + "$"):
        world.commit(prepared["plan_sha256"])
    assert world.head() is None
    assert not (world.root / "admission" / "manual-components").exists()
    assert not list(stage.adoption_directory(world.root).glob("adoption-*"))
    assert (state / adoption.COPY_MARKER).exists()
    assert os.path.realpath(world.root / "current") == os.fspath(world.root / "slots" / CONTAINER / "runtime")


def test_commit_refuses_an_unknown_plan(world):
    world.prepare()
    world.install_controller()
    with pytest.raises(commands.CommandRefusal, match="^runtime_admission_adoption_plan_unknown$"):
        world.commit("0" * 64)
    with pytest.raises(commands.CommandRefusal, match="^runtime_admission_adoption_confirm$"):
        world.commit("not-a-digest")
    assert world.head() is None


# --- the adoption kind and the aborted head ---------------------------------------------------------

def test_the_adoption_kind_is_only_valid_at_the_first_head_or_after_a_bound_abort(world):
    world.adopt()
    head = world.head()
    companion = stage.read_manual_head(world.root, head)
    # A crafted adoption companion at generation 2 without a previous head is refused.
    forged_head = dict(head, generation=2)
    path = world.root / "admission" / "manual-components" / "heads" / (stage._digest(forged_head) + ".json")
    path.write_text(json.dumps(dict(companion, head=forged_head)))
    with pytest.raises(ValueError, match="^runtime_admission_adoption_kind_invalid$"):
        stage.read_manual_head(world.root, forged_head)  # the seal names another head
    # Even with a well-formed seal of its own, generation 2 without a bound abort is refused.
    seal = json.loads(stage.adoption_seal_path(world.root, companion["authority_seal"]).read_text())
    forged_seal = dict(seal, closed_head=forged_head)
    forged_id = stage._digest(forged_seal)
    stage.adoption_seal_path(world.root, forged_id).write_text(json.dumps(forged_seal))
    os.unlink(path)
    path.write_text(json.dumps(dict(companion, head=forged_head, authority_seal=forged_id)))
    assert stage._read_adoption_seal(world.root, forged_id)["closed_head"] == forged_head
    with pytest.raises(ValueError, match="^runtime_admission_adoption_kind_invalid$"):
        stage.read_manual_head(world.root, forged_head)
    # Re-adoption over a head that was never aborted is refused at prepare and in the writer.
    world.units.properties[VOICE]["UnitFileState"] = "disabled"
    with pytest.raises(commands.CommandRefusal, match="^runtime_admission_adoption_head_present$"):
        world.prepare("r2")
    binding = stage.read_binding(world.root, head["binding_id"])
    with pytest.raises(ValueError, match="^runtime_admission_adoption_abort_missing$"):
        stage.commit_adoption_head(world.root, head, binding, seal_fields={}, record_fields={})
    assert world.head() == head


def test_an_aborted_head_is_terminal_and_a_bound_re_adoption_follows_it(world):
    first = world.adopt()
    aborted = world.head()
    world.abort()
    with stage.AdmissionController(world.root) as controller:
        with pytest.raises(ValueError, match="^runtime_admission_adoption_aborted_head$"):
            controller.begin_run()
        with pytest.raises(ValueError, match="^runtime_admission_adoption_aborted_head$"):
            controller.activate("r1", managed=True)
    assert world.head() == aborted
    world.units.properties[VOICE]["UnitFileState"] = "disabled"
    second = world.adopt("r2")
    head = world.head()
    assert head["generation"] == 2 and second["plan_sha256"] != first["plan_sha256"]
    companion = stage.read_manual_head(world.root, head)
    assert companion["kind"] == "adoption"
    assert companion["previous"] == stage.manual_reference(stage.read_manual_head(world.root, aborted))
    seal = json.loads(stage.adoption_seal_path(world.root, companion["authority_seal"]).read_text())
    assert seal["previous_abort"] == stage._digest(stage.read_abort_record(world.root, aborted))
    # The re-adoption stands on that abort record: without it the companion no longer validates.
    os.unlink(stage.abort_record_path(world.root, aborted))
    with pytest.raises(ValueError, match="^runtime_admission_adoption_abort_missing$"):
        stage.read_manual_head(world.root, head)


# --- adoption-abort (section 7, P6=A) ----------------------------------------------------------------

def test_abort_restores_the_legacy_unit_before_any_managed_run(world):
    adopted = world.adopt()
    head = world.head()
    world.units.properties[VOICE]["UnitFileState"] = "disabled"
    legacy = tree_manifest(world.layout.legacy_state)
    result = world.abort()
    assert os.readlink(world.root / "current") == "slots/" + CONTAINER + "/runtime"
    assert not (world.root / "slots" / "r1").exists()
    assert world.units.enabled == [VOICE] and result["enabled"] is True
    record_path = stage.abort_record_path(world.root, head)
    assert record_path.name == "abort-" + stage._digest(head) + ".json" == result["abort_record"]
    record = stage.read_abort_record(world.root, head)
    assert set(record) == stage.ABORT_KEYS and record["plan_sha256"] == adopted["plan_sha256"]
    assert record["legacy_conf"] is None and record["slot"] == "r1"
    assert world.head() == head
    assert tree_manifest(world.layout.legacy_state) == legacy
    with pytest.raises(commands.CommandRefusal, match="^runtime_admission_adoption_already_aborted$"):
        world.abort()


def test_abort_renders_legacy_conf_when_the_launcher_was_installed(world):
    world.adopt()
    head = world.head()
    (world.unit_dir / "managed.conf").write_text("[Service]\nExecStart=managed\n")
    (world.unit_dir / "release.conf.unmanaged").write_text(
        "[Service]\nEnvironment=PYTHONPATH=/home/kza/releases/current CONFIG_PATH=config/settings.yaml\n")
    stage._atomic_record(world.root / "admission" / "launcher" / "install-20261120T000000Z000000.json", {
        "schema_version": 1, "legacy_exec_start": ["/home/kza/kza/.venv/bin/python", "-B", "-P", "-m", "src.main"],
        "properties": {"Type": "simple", "NotifyAccess": "none", "RestartPreventExitStatus": "",
                       "StandardOutput": "append:/home/kza/logs/console.log", "StandardError": "inherit",
                       "WatchdogUSec": "0", "TimeoutStartUSec": "1min 30s", "TimeoutStopUSec": "1min 30s",
                       "Environment_sha256": "0" * 64}}, immutable=True)
    world.abort()
    assert not (world.unit_dir / "managed.conf").exists() and (world.unit_dir / "managed.conf.disabled").exists()
    text = (world.unit_dir / "legacy.conf").read_text()
    assert "Environment=PYTHONPATH=" + os.fspath(world.root / "slots" / CONTAINER / "runtime") in text
    assert "WorkingDirectory=" + os.fspath(world.layout.container) in text
    record = stage.read_abort_record(world.root, head)
    assert record["legacy_conf"]["sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert world.units.reloads == 1


@pytest.mark.parametrize("case,code", [
    ("no_head", "runtime_admission_adoption_abort_head"),
    ("unit_active", "runtime_admission_adoption_unit_active"),
    ("selected", "runtime_admission_adoption_abort_head"),
    ("run_record", "runtime_admission_adoption_abort_after_run"),
    ("legacy_changed", "runtime_admission_adoption_abort_legacy_changed"),
])
def test_abort_refuses_outside_the_section_7_conditions(world, case, code):
    if case == "no_head":
        world.prepare()  # prepared, never committed: W-abort-1 territory, not this command
        with pytest.raises(commands.CommandRefusal, match="^" + code + "$"):
            world.abort()
        assert world.head() is None and (world.root / "slots" / "r1").is_dir() and world.units.enabled == []
        return
    world.adopt()
    if case == "unit_active":
        world.units.properties[VOICE]["ActiveState"] = "active"
    elif case == "selected":
        with stage.AdmissionController(world.root) as controller:
            controller.activate("r1", managed=True)
    elif case == "run_record":
        stage._atomic_record(world.root / "admission" / "runs" / ("c" * 32 + ".json"),
                             {"schema_version": 1, "run_id": "c" * 32, "generation": 2})
    else:
        _touch_same_stat(world.layout.legacy_state / "logs" / "voice_assistant.log")  # not SQLite, same stat
    head = world.head()
    with pytest.raises(commands.CommandRefusal, match="^" + code + "$"):
        world.abort()
    assert world.head() == head and not stage.head_aborted(world.root, head)
    assert (world.root / "slots" / "r1").is_dir() and os.readlink(world.root / "current") == "slots/r1/runtime"
    assert world.units.enabled == []


def test_abort_of_a_prepared_slot_before_commit(world):
    """W-abort-1: the marked slot is removed, 'current' never changed, the unit is re-enabled."""
    legacy = tree_manifest(world.layout.legacy_state)
    with pytest.raises(commands.CommandRefusal, match="^runtime_admission_adoption_abort_prepared_unmarked$"):
        commands.adoption_abort_prepared(world.root, slot=fixture.B_SLOT, manager=world.units)  # never the legacy
    assert tree_manifest(world.layout.legacy_state) == legacy
    world.prepare()
    result = commands.adoption_abort_prepared(world.root, slot="r1", manager=world.units)
    assert result == {"removed": ["slots/r1"], "enabled": True}
    assert not (world.root / "slots" / "r1").exists()
    assert os.readlink(world.root / "current") == "slots/" + CONTAINER + "/runtime"
    assert [path.name.startswith("abort-prepared-r1-") for path in stage.adoption_directory(world.root).iterdir()
            if path.name.startswith("abort-")] == [True]
    with pytest.raises(commands.CommandRefusal, match="^runtime_admission_adoption_nothing_prepared$"):
        commands.adoption_abort_prepared(world.root, slot="r1", manager=world.units)
    # A committed slot is not 'prepared'.
    world.units.properties[VOICE]["UnitFileState"] = "disabled"
    world.adopt("r2")
    with pytest.raises(commands.CommandRefusal, match="^runtime_admission_adoption_abort_prepared_committed$"):
        commands.adoption_abort_prepared(world.root, slot="r2", manager=world.units)


def test_operator_paths_write_only_under_the_root_and_the_scratch(adapter_world):
    """Retire deletes only the LoRA roots (P1=D); prepare and commit write only under the root and the
    scratch; abort writes only under the root and, for legacy.conf, the unit directory."""
    world = adapter_world
    lora = world.layout.models_root / "lora_adapters"
    with watching(world.base, world.root) as seen:  # the hook itself observes this process
        (world.base / "probe.txt").write_text("x")
    assert seen == [("open", "probe.txt")]
    with watching(world.base, world.root, lora) as seen:
        world.retire()
    assert seen == []
    with watching(world.base, world.root, world.scratch) as seen:
        prepared = world.prepare()
    assert seen == []
    world.install_controller()
    with watching(world.base, world.root) as seen:
        world.commit(prepared["plan_sha256"])
    assert seen == []
    world.units.properties[VOICE]["UnitFileState"] = "disabled"
    (world.unit_dir / "managed.conf").write_text("[Service]\n")
    stage._atomic_record(world.root / "admission" / "launcher" / "install-20261120T000000Z000000.json", {
        "schema_version": 1, "legacy_exec_start": ["/usr/bin/python3", "-m", "src.main"],
        "properties": {"Environment_sha256": "0" * 64}}, immutable=True)
    with watching(world.base, world.root, world.unit_dir) as seen:
        world.abort()
    assert seen == []
    assert (world.unit_dir / "legacy.conf").exists()


# --- wrong user, wrong root, CONTROLLER_EPOCH ---------------------------------------------------------

def _calls(world):
    return {
        "retire": lambda: world.retire(),
        "dry-run": lambda: world.dry_run(),
        "prepare": lambda: world.prepare(),
        "commit": lambda: world.commit("0" * 64),
        "abort": lambda: world.abort(),
        "abort-prepared": lambda: commands.adoption_abort_prepared(world.root, slot="r1", manager=world.units),
    }


@pytest.mark.parametrize("command", ["retire", "dry-run", "prepare", "commit", "abort", "abort-prepared"])
@pytest.mark.parametrize("euid", ["other", "root"])
def test_every_command_refuses_the_wrong_user_before_any_access(adapter_world, monkeypatch, command, euid):
    before = _source_tree(adapter_world)
    monkeypatch.setattr(commands, "_euid", lambda: 0 if euid == "root" else os.geteuid() + 1)
    with pytest.raises(commands.CommandRefusal, match="^runtime_admission_adoption_wrong_user$"):
        _calls(adapter_world)[command]()
    assert _source_tree(adapter_world) == before


@pytest.mark.parametrize("command", ["retire", "dry-run", "prepare", "commit", "abort", "abort-prepared"])
@pytest.mark.parametrize("case", ["link", "no_slots", "group_writable", "relative"])
def test_every_command_refuses_the_wrong_root(adapter_world, command, case):
    world = adapter_world
    if case == "link":
        (world.base / "alias").symlink_to(world.root)
        world.root = world.base / "alias"
        code = "runtime_admission_adoption_root"
    elif case == "no_slots":
        world.root = world.base / "legacy" / "kza"
        code = "runtime_admission_adoption_root"
    elif case == "group_writable":
        os.chmod(world.root, 0o775)
        code = "runtime_admission_adoption_root_mode"
    else:
        world.root = Path("legacy") / "releases"
        code = "runtime_admission_adoption_root"
    before = _source_tree(world)
    with pytest.raises(commands.CommandRefusal, match="^" + code + "$"):
        _calls(world)[command]()
    assert _source_tree(world) == before


def test_controller_epoch_two_and_an_epoch_one_install_refuses(world, artifacts):
    assert stage.CONTROLLER_EPOCH == 2
    with pytest.raises(launcher.LauncherRefusal, match="runtime_admission_controller_epoch_too_old"):
        launcher.install_controller(world.root, artifacts["epoch1"])
    assert not (world.root / "admission" / "controller.json").exists()
    assert launcher.install_controller(world.root, artifacts["archive"])["controller_epoch"] == 2


# --- the CLI (stage_runtime loads the commands by path, as the compiled controller does) --------------

def _cli(world, *arguments, root=None):
    return subprocess.run([sys.executable, "-I", "-B", os.fspath(REPOSITORY / "scripts" / "stage_runtime.py"),
                           "--root", os.fspath(root or world.root), "--offline", *arguments],
                          capture_output=True, text=True, timeout=300)


def test_the_cli_registers_every_command_and_refuses_with_78(world):
    result = _cli(world, "retire-legacy-training", "--container-slot", CONTAINER)
    assert result.returncode == 2 and "--decision" in result.stderr
    result = _cli(world, "adoption-dry-run", "--container-slot", CONTAINER, "--archive", os.fspath(world.archive),
                  "--slot", "r1", "--scratch", os.fspath(world.scratch))
    assert result.returncode == 78
    assert json.loads(result.stderr.splitlines()[-1]) == {"refused": "runtime_admission_adoption_models_root"}
    result = _cli(world, "adoption-commit", "--confirm", "0" * 64, root=world.base / "legacy" / "kza")
    assert result.returncode == 78
    assert json.loads(result.stderr.splitlines()[-1]) == {"refused": "runtime_admission_adoption_root"}
    result = _cli(world, "adoption-abort")
    assert result.returncode == 2
    for command in commands.COMMANDS:
        assert _cli(world, command, "--help").returncode == 0
    assert not (world.root / "admission").exists() and os.listdir(world.scratch) == []
