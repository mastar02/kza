"""Adoption operator commands (U3b-0e3): retire, dry-run, prepare, commit and abort.

Adoption design v3 (data/autonomous-roadmap-20261102/t040-t041/u3b-0e-adoption-design-v3.json)
sections 2c, 4, 5, 6 and 7, with the user's final answers P1-P8. The library
(:mod:`scripts.adoption`) does the copy and the R1 steps; this module adds
the window around it:

* ``retire-legacy-training --decision <id>`` (W1b, P1=D): deletes the three
  reviewed LoRA roots under the container's resolved models link after the
  2c scope guards, and writes an immutable, content-free record.
* ``adoption-dry-run`` (advisory, P4=A): the whole pipeline on a private
  tmpfs scratch; no lock and no write under the root (0d P3 pattern); swap,
  sizes, the legacy-path census and the erasure-journal count are reported.
* ``adoption-prepare`` (W2): the same pipeline into ``<root>/adoption/<slot>``,
  moved into ``slots/<slot>`` by one same-filesystem rename (the prepared
  manifest pins inodes). The slot keeps the adoption-copy marker; the plan
  binds everything commit re-verifies. No CAS; 'current' is unchanged.
* ``adoption-commit --confirm <plan_sha256>`` (W4): re-verifies every plan
  item, removes the copy marker, then writes the binding, the adoption seal
  and record, the 'adoption' manual head and the CAS, and publishes 'current'.
* ``adoption-abort`` (W-abort-2, P6=A) and ``adoption-abort --prepared-slot``
  (W-abort-1).

Every command refuses as root, as a user other than the root's owner, on a
root that is not the managed layout, and when run twice. Outputs are
content-free JSON: counts, classes, digests and names of units and slots.
"""
from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import sqlite3
import stat
import sys
import tempfile
import time

stage = None  # scripts/stage_runtime
launcher = None  # scripts/managed_launcher
adoption = None  # scripts/adoption


def bind(stage_module, launcher_module, adoption_module) -> None:
    """Use these implementations (stage_runtime binds by path when compiled)."""
    global stage, launcher, adoption
    stage, launcher, adoption = stage_module, launcher_module, adoption_module


if __package__ == "scripts":  # in-process users (tests)
    from scripts import adoption as _adoption_module  # noqa: E402
    from scripts import managed_launcher as _launcher_module  # noqa: E402
    from scripts import stage_runtime as _stage_module  # noqa: E402

    bind(_stage_module, _launcher_module, _adoption_module)


class CommandRefusal(ValueError):
    """A content-free refusal; ``str(error)`` is the code, ``report`` what was found so far."""

    def __init__(self, code: str, report: dict | None = None) -> None:
        super().__init__(code)
        self.report = report


def _refuse(code: str, report: dict | None = None):
    raise CommandRefusal(code, report)


COMMANDS = ("retire-legacy-training", "adoption-dry-run", "adoption-prepare", "adoption-commit", "adoption-abort")
SCHEMA_VERSION = 1
# The user's final answers (user-decisions-20261110.json u3b_0e_adoption), bound in plan and seal.
P_ANSWERS = ("P1=D", "P2=A", "P3=A", "P4=A", "P5=A", "P6=A", "P7=B", "P8=A")
DECISION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{2,79}")  # = uninstall-launcher's decision id
_HEX64 = re.compile("[0-9a-f]{64}")
UNIT_PROPERTIES = ("LoadState", "ActiveState", "UnitFileState", "InvocationID", "InactiveEnterTimestampMonotonic")
# Who can start or restart kza-voice (U3b-0e3 review F5): the reverse dependency properties, the
# .timer/.path/.socket that trigger it, OnFailure=/OnSuccess= handlers, and the units it is PartOf
# (PartOf is read on kza-voice itself; ConsistsOf lists the units that are PartOf kza-voice).
REVERSE_PROPERTIES = ("WantedBy", "RequiredBy", "BoundBy", "ConsistsOf", "RequisiteOf", "UpheldBy",
                      "TriggeredBy", "OnFailureOf", "OnSuccessOf", "PartOf")
REVERSE_ALLOWED = frozenset({"default.target"})
CODE_INDEX_UNIT = "kza-code-index.service"
# The llama-server fast-path NLU unit on :8101 (config/settings.yaml, docs/SERVER_CONVENTIONS.md):
# retire-legacy-training always reads it, whatever --unit lists (review F4).
LLM_FAST_UNIT = "kza-llm-fast.service"
# What makes a user unit relevant to the LoRA roots besides a path under the models root.
_UNIT_NEEDLES = ("llama-server", "--lora", "lora_adapters")
_CONTAINER_TOOLS = frozenset({"podman", "docker"})
_ENV_REFERENCE = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")
_ENV_FILE_LIMIT = 1 << 20
WORK_DIRECTORY = "adoption"  # <root>/adoption/<slot>: built here, then renamed into slots/<slot>
RETIREMENT_PREFIX = "retire-legacy-training-"
RETIRE_INTENT_PREFIX = "retire-intent-"
COMMIT_INTENT_PREFIX = "commit-intent-"
PLAN_PREFIX = "plan-"
PREPARED_ABORT_PREFIX = "abort-prepared-"
# Records whose presence means a managed run happened (P6: then only L4).
RUN_RECORD_DIRECTORIES = ("runs", "receipts", "receipt-states", "recoveries", "closures", "seal-intents")
# Free space (checked as a threshold, never hashed): 2 x the source plus a margin.
ROOT_FREE_MARGIN = 1 << 30
SCRATCH_FREE_MARGIN = 256 << 20
PROC_SWAPS = "/proc/swaps"
PROC_MEMINFO = "/proc/meminfo"
# Config keys whose b0866e8 value IS a retired root or the retired adapter (2a rows rewrite them).
RETIRED_CONFIG_KEYS = frozenset({("llm", "lora_path"), ("training", "lora", "output_dir"),
                                 ("training", "nightly", "output_dir"),
                                 ("training", "nightly", "whisper_finetune", "output_dir")})


def _euid() -> int:
    return os.geteuid()


def _free_bytes(path: Path) -> int:
    info = os.statvfs(path)
    return info.f_bavail * info.f_frsize


@dataclass(frozen=True)
class ProductionPaths:
    """The 2a table's absolute targets; production uses the adoption library's constants."""

    models_root: str
    logs_root: str
    legacy_metrics: str

    def rows(self):
        return adoption.transform_table(self.models_root, self.logs_root, self.legacy_metrics)

    def as_dict(self) -> dict:
        return {"models_root": self.models_root, "logs_root": self.logs_root, "legacy_metrics": self.legacy_metrics}


def production_paths() -> ProductionPaths:
    return ProductionPaths(adoption.PRODUCTION_MODELS, adoption.PRODUCTION_LOGS, adoption.PRODUCTION_METRICS)


# --- root, user, layout ------------------------------------------------------------------------------

def checked_root(root) -> Path:
    """The managed root: absolute, real (no link anywhere), owned by this non-root user, with slots/."""
    raw = Path(root)
    if not raw.is_absolute():
        _refuse("runtime_admission_adoption_root")
    try:
        info = os.lstat(raw)
    except OSError:
        _refuse("runtime_admission_adoption_root")
    if not stat.S_ISDIR(info.st_mode) or os.path.realpath(raw) != os.path.normpath(raw):
        _refuse("runtime_admission_adoption_root")
    euid = _euid()
    if euid == 0 or info.st_uid != euid:
        _refuse("runtime_admission_adoption_wrong_user")
    if info.st_mode & 0o022:
        _refuse("runtime_admission_adoption_root_mode")
    try:
        slots = os.lstat(raw / "slots")
    except OSError:
        _refuse("runtime_admission_adoption_root")
    if not stat.S_ISDIR(slots.st_mode) or slots.st_uid != euid:
        _refuse("runtime_admission_adoption_root")
    return Path(os.path.normpath(raw))


def _slot(name: str, code: str) -> str:
    try:
        return stage._slot_name(name)
    except (ValueError, TypeError):
        _refuse(code)


def source_layout(root: Path, container_slot: str, paths: ProductionPaths):
    """The SourceLayout read from the container's links (lstat/readlink only) and the N10 check.

    The legacy state must be another slot's state under this root; the
    models link must resolve to the 2a table's models root (N10); the
    checkout, metrics and code-index roots are its siblings, as in
    production (/home/kza/kza, /home/kza/logs, /home/kza/code-index).
    """
    name = _slot(container_slot, "runtime_admission_adoption_container_slot")
    container = root / "slots" / name / "state"
    try:
        if not stat.S_ISDIR(os.lstat(container).st_mode):
            _refuse("runtime_admission_adoption_source_layout")
    except OSError:
        _refuse("runtime_admission_adoption_source_layout")

    def resolved(link: str) -> Path:
        path = container / link
        if not os.path.islink(path):
            _refuse("runtime_admission_adoption_source_layout")
        return Path(os.path.realpath(path))
    legacy_state = resolved("config").parent
    if (legacy_state.name != "state" or legacy_state.parent.parent != root / "slots"
            or legacy_state.parent.name == name):
        _refuse("runtime_admission_adoption_source_layout")
    models = resolved("models")
    if os.fspath(models) != os.path.normpath(paths.models_root):
        _refuse("runtime_admission_adoption_models_root")
    checkout = models.parent
    layout = adoption.SourceLayout(container=container, legacy_state=legacy_state, models_root=models,
                                   legacy_checkout=checkout, metrics_root=checkout.parent / "logs",
                                   code_index=checkout.parent / "code-index")
    links = adoption.read_source_layout(layout)
    return layout, links


def layout_record(layout) -> dict:
    return {key: os.fspath(getattr(layout, key)) for key in
            ("container", "legacy_state", "models_root", "legacy_checkout", "metrics_root", "code_index")}


def current_link(root: Path, container_slot: str) -> str:
    """The raw readlink of 'current'; it must resolve to the container slot's runtime."""
    current = root / "current"
    if not os.path.islink(current):
        _refuse("runtime_admission_adoption_current")
    if os.path.realpath(current) != os.fspath(root / "slots" / container_slot / "runtime"):
        _refuse("runtime_admission_adoption_current")
    return os.readlink(current)


# --- units (W0/W1 observations; the plan binds them) -----------------------------------------------

def observe_units(manager, timers: tuple[str, ...], *, window: bool) -> tuple[dict, list[str]]:
    """kza-voice and the window-stop timers, plus kza-voice's reverse dependencies.

    ``window`` (prepare, commit, retire): kza-voice must be stopped and
    disabled (W1) and every timer inactive. Any unit other than
    default.target pulling kza-voice in is a problem in every mode (W0).
    """
    observed, problems = {}, []
    for unit in (launcher.UNIT, *timers):
        shown = manager.user_show(unit, UNIT_PROPERTIES)
        if shown is None:
            _refuse("runtime_admission_adoption_units_unavailable")
        observed[unit] = dict(shown)
    reverse = manager.user_show(launcher.UNIT, REVERSE_PROPERTIES)
    if reverse is None:
        _refuse("runtime_admission_adoption_units_unavailable")
    pulled = sorted({token for value in reverse.values() for token in value.split()} - REVERSE_ALLOWED)
    voice = observed[launcher.UNIT]
    for unit in (launcher.UNIT, *timers):  # a misspelled timer is 'not-found' and 'inactive' (review F4)
        if observed[unit]["LoadState"] != "loaded":
            problems.append("runtime_admission_adoption_unit_not_loaded:" + unit)
    if window:
        if voice["ActiveState"] not in ("inactive", "failed"):
            problems.append("runtime_admission_adoption_unit_active")
        if voice["UnitFileState"] != "disabled":
            problems.append("runtime_admission_adoption_unit_enabled")
        for timer in timers:
            if observed[timer]["ActiveState"] != "inactive":
                problems.append("runtime_admission_adoption_timer_active:" + timer)
    if pulled:
        problems.append("runtime_admission_adoption_reverse_dependency")
    return {"units": observed, "reverse_dependencies": pulled}, problems


def code_index_check(manager, unit: str, layout) -> tuple[dict, list[str]]:
    """2a: the code_index subtree is deleted only if the code-index unit does not read the slot config."""
    shown = manager.user_show(unit, ("LoadState", "Environment", "WorkingDirectory"))
    if shown is None:
        _refuse("runtime_admission_adoption_units_unavailable")
    if shown["LoadState"] != "loaded":
        return {"unit": unit, "state": "absent"}, []
    try:
        tokens = shlex.split(shown["Environment"])
    except ValueError:
        return {"unit": unit, "state": "loaded", "config_class": "unparsed"}, [
            "runtime_admission_adoption_code_index_unparsed"]
    values = [token.split("=", 1)[1] for token in tokens if token.startswith("CONFIG_PATH=")]
    directory = shown["WorkingDirectory"].lstrip("!-") or "/"
    path = os.path.realpath(os.path.join(directory, values[-1] if values else "config/settings.yaml"))
    klass = adoption.classify(path, layout)
    problems = ["runtime_admission_adoption_code_index_reads_slot"] if klass in ("container", "legacy_state") else []
    return {"unit": unit, "state": "loaded", "config_class": klass}, problems


def timer_dispositions(manager, timers: tuple[str, ...], root: Path, layout) -> list[dict]:
    """W0 (N8): 'restart unchanged' only for a timer whose units reference no KZA code path and no legacy state.

    A timer and every unit it triggers are read (ExecStart arguments,
    Environment values, WorkingDirectory), lexically and resolved: the
    managed root's 'current' or slots, the legacy checkout's src, the
    container or the legacy state make it 'review' (a reviewed retarget, or
    leave disabled with a reason). Only classes are reported, never values.
    """
    classes = (("managed_code", [os.fspath(root / "current"), os.fspath(root / "slots")]),
               ("legacy_checkout_code", [os.fspath(Path(layout.legacy_checkout) / "src")]),
               ("legacy_state", [os.fspath(layout.container), os.fspath(layout.legacy_state)]))
    found = []
    for timer in timers:
        shown = manager.user_show(timer, ("Triggers",))
        if shown is None:
            _refuse("runtime_admission_adoption_units_unavailable")
        references = set()
        for unit in (timer, *shown["Triggers"].split()):
            values = manager.user_show(unit, ("ExecStart", "Environment", "WorkingDirectory"))
            if values is None:
                _refuse("runtime_admission_adoption_units_unavailable")
            base = values["WorkingDirectory"].lstrip("!-") or "/"
            try:
                tokens = launcher.exec_start_argv(values["ExecStart"]) if values["ExecStart"] else []
                tokens += shlex.split(values["Environment"])
            except (ValueError, launcher.LauncherRefusal):
                references.add("unparsed")
                continue
            for token in [*tokens, base]:
                for piece in {token, token.split("=", 1)[-1]}:
                    if "/" not in piece:
                        continue
                    lexical = os.path.normpath(os.path.join(base, piece))
                    for candidate in {lexical, os.path.realpath(lexical)}:
                        references.update(name for name, prefixes in classes if _under(candidate, prefixes))
        found.append({"unit": timer, "references": sorted(references),
                      "disposition": "review" if references else "restart unchanged"})
    return found


# --- scratch (P4=A), swap, sizes -------------------------------------------------------------------

def swap_report() -> dict:
    """Swap exposure of the tmpfs scratch (P4): tmpfs pages can be written to an active swap."""
    try:
        devices = [line for line in Path(PROC_SWAPS).read_text().splitlines()[1:] if line.strip()]
        meminfo = Path(PROC_MEMINFO).read_text()
    except OSError:
        return {"known": False}
    values = {}
    for name in ("SwapTotal", "SwapFree"):
        match = re.search(rf"^{name}:\s+(\d+)\s+kB$", meminfo, re.M)
        values[name] = int(match.group(1)) if match else None
    return {"known": True, "active": bool(devices), "devices": len(devices), "total_kib": values["SwapTotal"],
            "free_kib": values["SwapFree"],
            "exposure": "tmpfs scratch pages may be written to swap" if devices else "none"}


def scratch_kind(parent: Path) -> str:
    """'tmpfs' when ``parent`` is on tmpfs inside /run/user/<uid> (P4=A); 'other' otherwise."""
    real = os.path.realpath(parent)
    base = f"/run/user/{os.getuid()}"
    if (real == base or real.startswith(base + "/")) and adoption._mount_type(real) == "tmpfs":
        return "tmpfs"
    return "other"


def checked_scratch(parent, *, tmpfs_required: bool) -> tuple[Path, str]:
    parent = Path(parent)
    try:
        info = os.lstat(parent)
    except OSError:
        _refuse("runtime_admission_adoption_scratch")
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        _refuse("runtime_admission_adoption_scratch")
    kind = scratch_kind(parent)
    if tmpfs_required and kind != "tmpfs":
        _refuse("runtime_admission_adoption_scratch_not_tmpfs")
    return Path(os.path.realpath(parent)), kind


def source_size(layout) -> dict:
    files = size = 0
    for path in adoption._walk_files(layout.legacy_state):
        files, size = files + 1, size + path.lstat().st_size
    return {"files": files, "bytes": size}


def journal_file_count(state: Path) -> int:
    """SQLite files of the copy holding any erasure-journal table (0d4a review F4)."""
    count = 0
    for path in launcher._sqlite_files(Path(state)):
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)) as db:
            try:
                names = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            except sqlite3.DatabaseError:
                continue
        if names & launcher._JOURNAL_TABLES:
            count += 1
    return count


def _remove_private(path: Path) -> bool:
    try:
        stage.make_writable_then_remove(path)
    except FileNotFoundError:
        return True
    except (OSError, ValueError):
        return False
    return not os.path.lexists(path)


# --- archive, runtime, retirement record -----------------------------------------------------------

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def read_r1_archive(archive) -> tuple[dict, dict, dict]:
    """(manifest, files, summary) of a verified R1 archive: controller epoch and launcher contract current."""
    archive = Path(os.path.abspath(archive))
    try:
        manifest, files = stage._read_archive(archive)
    except (OSError, ValueError, RuntimeError):
        _refuse("runtime_admission_adoption_archive")
    if (manifest.get("controller_epoch") != stage.CONTROLLER_EPOCH
            or manifest.get("launcher_contract", 0) < stage.LAUNCHER_CONTRACT):
        _refuse("runtime_admission_adoption_archive_epoch")
    summary = {"path": os.fspath(archive), "sha256": _sha256(archive), "git_commit": manifest["git_commit"],
               "controller_epoch": manifest["controller_epoch"]}
    return manifest, files, summary


def write_runtime(manifest: dict, files: dict, runtime: Path) -> None:
    runtime.mkdir(mode=0o700)
    for name, data in files.items():
        destination = runtime / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    (runtime / "release.json").write_text(json.dumps(manifest, indent=2) + "\n")


def latest_retirement(root: Path) -> tuple[dict | None, str | None, str | None]:
    """The newest retire-legacy-training record: (record, sha256 of its bytes, file name)."""
    directory = stage.adoption_directory(root)
    if not os.path.isdir(directory):
        return None, None, None
    names = sorted(name for name in os.listdir(directory)
                   if name.startswith(RETIREMENT_PREFIX) and name.endswith(".json"))
    if not names:
        return None, None, None
    path = directory / names[-1]
    record = stage._read_json(path)
    if (set(record) != RETIREMENT_KEYS or record.get("schema_version") != 1
            or record.get("kind") != "retire_legacy_training" or type(record.get("roots")) is not list):
        _refuse("runtime_admission_adoption_retirement_invalid")
    return record, _sha256(path), names[-1]


def retirement_view(record: dict | None) -> dict | None:
    """The library's retirement shape (scripts/adoption.legacy_training_disposition)."""
    if record is None:
        return None
    return {"schema_version": 1, "decision_id": record["decision_id"],
            "roots": [{"root": item["root"], "absence": item["absence"]} for item in record["roots"]]}


def retirement_absence(record: dict | None) -> list[list[str]] | None:
    """lstat each retired root now: 'enoent', 'empty' or 'present' (the plan binds it)."""
    if record is None:
        return None
    found = []
    for item in record["roots"]:
        root = item["root"]
        try:
            info = os.lstat(root)
        except FileNotFoundError:
            found.append([root, "enoent"])
            continue
        found.append([root, "empty" if stat.S_ISDIR(info.st_mode) and not os.listdir(root) else "present"])
    return found


# --- retire-legacy-training (W1b, P1=D, design 2c) ---------------------------------------------------

RETIREMENT_KEYS = frozenset({"schema_version", "kind", "decision_id", "container_slot", "roots", "intent",
                             "models_manifest_sha256", "models_entries", "units_checked", "units",
                             "config_keys_checked", "fsynced", "retired_at"})


def _tree_census(root: Path, device: int) -> dict:
    """lstat walk of one retired root: link, special or mount crossing refuses before any deletion."""
    try:
        info = os.lstat(root)
    except FileNotFoundError:
        return {"exists": False, "files": 0, "bytes": 0}
    if not stat.S_ISDIR(info.st_mode):
        _refuse("runtime_admission_retire_root_special")
    if info.st_dev != device or os.path.ismount(root):
        _refuse("runtime_admission_retire_mount")
    files = size = 0
    for directory, names, filenames in os.walk(root, followlinks=False):
        for name in names + filenames:
            path = os.path.join(directory, name)
            entry = os.lstat(path)
            if stat.S_ISLNK(entry.st_mode) or not (stat.S_ISDIR(entry.st_mode) or stat.S_ISREG(entry.st_mode)):
                _refuse("runtime_admission_retire_special")
            if entry.st_dev != device or (stat.S_ISDIR(entry.st_mode) and os.path.ismount(path)):
                _refuse("runtime_admission_retire_mount")
            if stat.S_ISREG(entry.st_mode):
                files, size = files + 1, size + entry.st_size
    return {"exists": True, "files": files, "bytes": size}


def models_manifest(models: Path, excluded: Path) -> list[list]:
    """lstat manifest of <models>/ minus the retired tree: links as their text, never followed."""
    entries = []
    for directory, names, filenames in os.walk(models, followlinks=False):
        if Path(directory) == excluded:
            names[:] = []
            continue
        names[:] = [name for name in names if Path(directory, name) != excluded]
        for name in sorted(names + filenames):
            path = Path(directory, name)
            if path == excluded:
                continue
            info = os.lstat(path)
            link = os.readlink(path) if stat.S_ISLNK(info.st_mode) else None
            entries.append([path.relative_to(models).as_posix(), stat.S_IFMT(info.st_mode), info.st_mode,
                            info.st_ino, info.st_size, info.st_mtime_ns, link])
    return sorted(entries, key=lambda item: item[0])


def _under(path: str, roots: list[str]) -> bool:
    return any(path == root or path.startswith(root.rstrip("/") + "/") for root in roots)


def _path_candidates(value: str, base: Path) -> list[str]:
    """Absolute or relative path-like pieces of one value (an '--opt=value' token counts twice)."""
    found = []
    for piece in {value, value.split("=", 1)[1] if "=" in value else value}:
        if "/" in piece:
            found.append(os.path.realpath(os.path.join(base, piece)))
    return found


def _config_leaves(value, key=()):
    if isinstance(value, dict):
        for name, item in value.items():
            yield from _config_leaves(item, (*key, str(name)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _config_leaves(item, (*key, str(index)))
    elif isinstance(value, str):
        yield key, value


def config_paths_under(layout, roots: list[str]) -> tuple[list[str], int]:
    """Config keys (never values) whose path resolves under a retired root; relative to the service cwd."""
    import yaml

    path = Path(layout.legacy_state) / "config" / "settings.yaml"
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        config = yaml.safe_load(stream.read()) or {}
    offending, checked = set(), 0
    for key, value in _config_leaves(config):
        if key in RETIRED_CONFIG_KEYS:
            continue
        checked += 1
        if any(_under(candidate, roots) for candidate in _path_candidates(value, Path(layout.container))):
            offending.add(".".join(key))
    return sorted(offending), checked


def _env_file_values(unit: str, specification: str) -> tuple[dict, list[str]]:
    """KEY=VALUE pairs of one EnvironmentFile= (O_NOFOLLOW, bounded); values stay in memory, keys are reported."""
    path = specification.rsplit(" (ignore_errors=", 1)[0].strip()
    optional = specification.endswith("(ignore_errors=yes)") or path.startswith("-")
    path = path.lstrip("-")
    if not os.path.isabs(path):
        _refuse("runtime_admission_retire_environment_file:" + unit)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        if optional:
            return {}, []
        _refuse("runtime_admission_retire_environment_file:" + unit)
    except OSError:
        _refuse("runtime_admission_retire_environment_file:" + unit)
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            _refuse("runtime_admission_retire_environment_file:" + unit)
        raw = stream.read(_ENV_FILE_LIMIT + 1)
    if len(raw) > _ENV_FILE_LIMIT:
        _refuse("runtime_admission_retire_environment_file:" + unit)
    values = {}
    try:
        lines = raw.decode().splitlines()
    except UnicodeDecodeError:
        _refuse("runtime_admission_retire_environment_file:" + unit)
    for line in lines:
        if not line.strip() or line.lstrip().startswith(("#", ";")):
            continue
        match = adoption._ENV_LINE.match(line)
        if match is None:
            _refuse("runtime_admission_retire_environment_file:" + unit)
        value = match.group(2).strip()
        try:
            value = " ".join(shlex.split(value)) if value[:1] in ("'", '"') else value
        except ValueError:
            _refuse("runtime_admission_retire_environment_file:" + unit)
        values[match.group(1)] = value
    return values, sorted(values)


def _expanded(unit: str, token: str, environment: dict) -> str:
    def substitute(match):
        name = match.group(1) or match.group(2)
        if name not in environment:
            _refuse("runtime_admission_retire_unit_unresolved:" + unit)
        return environment[name]
    return _ENV_REFERENCE.sub(substitute, token)


def _container_volumes(unit: str, argv: list[str]) -> list[str]:
    """Host sides of a podman/docker run's binds; a named or unparsed volume is returned as ''."""
    hosts = []
    for index, token in enumerate(argv):
        value = None
        if token in ("-v", "--volume", "--mount") and index + 1 < len(argv):
            value, kind = argv[index + 1], token
        elif token.startswith(("--volume=", "--mount=")):
            kind, value = token.split("=", 1)
        elif token.startswith("-v") and len(token) > 2:
            value, kind = token[2:], "-v"
        if value is None:
            continue
        if kind == "--mount":
            fields = dict(item.split("=", 1) for item in value.split(",") if "=" in item)
            host = fields.get("source", fields.get("src", ""))
        else:
            host = value.split(":", 1)[0]
        hosts.append(host if host.startswith("/") else "")
    return hosts


def unit_view(manager, unit: str) -> dict:
    """One user unit as retire reads it: LoadState, expanded argv tokens, env file keys, container binds.

    ExecStart is shown unexpanded by systemd: ${VAR} and $VAR are resolved
    from Environment= and every EnvironmentFile= (read O_NOFOLLOW; only key
    names are kept). An unresolvable reference or env file refuses.
    """
    shown = manager.user_show(unit, ("LoadState", "Environment", "WorkingDirectory"))
    commands_shown = manager.user_show_all(unit, "ExecStart")
    files = manager.user_show_all(unit, "EnvironmentFiles")
    if shown is None or commands_shown is None or files is None:
        _refuse("runtime_admission_adoption_units_unavailable")
    base = shown["WorkingDirectory"].lstrip("!-~") or "/"
    environment, keys = {}, []
    try:
        for token in shlex.split(shown["Environment"]):
            name, _sep, value = token.partition("=")
            environment[name] = value
    except ValueError:
        _refuse("runtime_admission_retire_unit_unparsed:" + unit)
    for specification in files:
        values, names = _env_file_values(unit, specification)
        environment.update(values)
        keys += names
    tokens, binds, raw = [], [], []
    for value in commands_shown:
        if not value:
            continue
        try:
            argv = launcher.exec_start_argv(value)
        except launcher.LauncherRefusal:
            _refuse("runtime_admission_retire_unit_unparsed:" + unit)
        argv = [_expanded(unit, token, environment) for token in argv]
        raw += argv
        if argv and os.path.basename(argv[0]) in _CONTAINER_TOOLS:
            binds += _container_volumes(unit, argv)
        tokens += argv
    tokens += list(environment.values()) + [base]
    return {"unit": unit, "load_state": shown["LoadState"], "base": base, "tokens": tokens, "raw": raw,
            "environment_file_keys": sorted(set(keys)), "container_binds": binds}


def _covers(path: str, roots: list[str]) -> bool:
    """``path`` is under a root, or a root is under ``path`` (a bind of a parent exposes the root)."""
    return _under(path, roots) or any(_under(root, [path]) for root in roots)


def _relevant(view: dict, models: str, checkout: str) -> bool:
    text = " ".join(view["raw"] + view["tokens"])
    if any(needle in text for needle in _UNIT_NEEDLES):
        return True
    if any(not host or _covers(os.path.realpath(host), [models, checkout]) for host in view["container_binds"]):
        return True  # a named or unparsed volume cannot be resolved here: the W0 list must name the unit
    return any(_covers(candidate, [models, checkout]) for token in view["tokens"]
               for candidate in _path_candidates(token, Path(view["base"])))


def units_under(views: list[dict], roots: list[str]) -> list[str]:
    """Units whose argument, Environment(File) value or WorkingDirectory resolves under a root,
    or whose container bind covers one or cannot be resolved."""
    offending = []
    for view in views:
        paths = [candidate for token in view["tokens"] for candidate in _path_candidates(token, Path(view["base"]))]
        if (any(_under(candidate, roots) for candidate in paths)
                or any(not host or _covers(os.path.realpath(host), roots) for host in view["container_binds"])):
            offending.append(view["unit"])
    return offending


def discover_units(manager, listed: tuple[str, ...], layout) -> tuple[list[dict], dict]:
    """Every user service that can reach the models (review F4): read, never trusted from --unit alone.

    Always kza-voice and the :8101 llama-server unit; plus each listed unit;
    plus every user service whose command, environment (files included) or
    working directory names llama-server, --lora, lora_adapters, a container
    bind, the models root or the legacy checkout. A discovered unit missing
    from ``listed`` refuses (the W0 list must name it); a named unit that is
    not loaded refuses.
    """
    names = manager.user_units()
    if names is None:
        _refuse("runtime_admission_adoption_units_unavailable")
    models = os.path.realpath(layout.models_root)
    checkout = os.path.realpath(layout.legacy_checkout)
    named = sorted({launcher.UNIT, LLM_FAST_UNIT, *listed})
    views = {}
    for unit in named:
        view = unit_view(manager, unit)
        if view["load_state"] != "loaded":
            _refuse("runtime_admission_retire_unit_not_loaded:" + unit)
        views[unit] = view
    for unit in sorted(set(names) - set(named)):
        if not unit.endswith(".service"):
            continue
        view = unit_view(manager, unit)
        if view["load_state"] == "loaded" and _relevant(view, models, checkout):
            _refuse("runtime_admission_retire_unit_not_listed:" + unit)
    summary = {"discovered": len(names), "checked": named,
               "environment_file_keys": {unit: view["environment_file_keys"] for unit, view in views.items()
                                         if view["environment_file_keys"]},
               "container_units": sorted(unit for unit, view in views.items() if view["container_binds"])}
    return list(views.values()), summary


RETIRE_INTENT_KEYS = frozenset({"schema_version", "kind", "decision_id", "container_slot", "roots", "pre",
                                "models_manifest_sha256", "units", "config_keys_checked", "intended_at"})


def _retire_intents(root: Path) -> list[tuple[str, dict]]:
    directory = stage.adoption_directory(root)
    if not os.path.isdir(directory):
        return []
    found = []
    for name in sorted(os.listdir(directory)):
        if name.startswith(RETIRE_INTENT_PREFIX) and name.endswith(".json"):
            intent = stage._read_json(directory / name)
            if (set(intent) != RETIRE_INTENT_KEYS or intent.get("schema_version") != 1
                    or intent.get("kind") != "retire_intent"):
                _refuse("runtime_admission_retire_intent_invalid")
            found.append((name, intent))
    return found


def pending_retire_intent(root: Path) -> tuple[str, dict] | tuple[None, None]:
    """The retire intent no retirement record completes yet (a crash between deletion and record, F6)."""
    intents = _retire_intents(root)
    if not intents:
        return None, None
    directory = stage.adoption_directory(root)
    completed = set()
    for name in os.listdir(directory):
        if name.startswith(RETIREMENT_PREFIX) and name.endswith(".json"):
            completed.add(stage._read_json(directory / name).get("intent"))
    pending = [(name, intent) for name, intent in intents if name not in completed]
    if len(pending) > 1:
        _refuse("runtime_admission_retire_intent_invalid")
    return pending[0] if pending else (None, None)


def retire_legacy_training(root, *, decision_id: str, container_slot: str, manager, units: tuple[str, ...] = (),
                           paths: ProductionPaths | None = None, wall=time.time) -> dict:
    """W1b: delete the LoRA roots by recorded user decision (P1=D); unrecoverable.

    The roots are the reviewed legacy_training tails (lora_adapters, its
    nightly and whisper) under the container's resolved models link, never a
    path read from data. Refuses before deleting anything on: a link,
    device, FIFO or socket inside, a mount crossing, a configured model path
    or a kza unit argument (llama-server --lora included, EnvironmentFile
    values resolved, container binds) under a root, a user unit that can
    reach the models and is missing from ``units``, or when nothing needs
    retiring. An immutable intent (decision, roots, pre-census, manifest
    digest) is written before the deletion; a re-run after a crash completes
    from it (review F6). The models tree outside the roots must be unchanged.
    """
    if type(decision_id) is not str or DECISION.fullmatch(decision_id) is None:
        _refuse("runtime_admission_retire_decision")
    paths = paths or production_paths()
    root = checked_root(root)
    layout, _links = source_layout(root, container_slot, paths)
    _observed, problems = observe_units(manager, (), window=True)
    if problems:
        _refuse(problems[0])
    with stage.AdmissionController(root):
        head = stage.read_admission_head(root)
        if head is not None and not stage.head_aborted(root, head):
            _refuse("runtime_admission_retire_head_present")
        intent_name, intent = pending_retire_intent(root)
        if intent is not None and (intent["decision_id"] != decision_id or intent["container_slot"] != container_slot):
            _refuse("runtime_admission_retire_intent_mismatch")
        models = Path(os.path.realpath(layout.models_root))
        base = models / adoption.LEGACY_TRAINING_OUTSIDE[0][0]
        device = os.lstat(models).st_dev
        tails = [tail for tail, _new in adoption.LEGACY_TRAINING_OUTSIDE]
        roots = [os.fspath(models / tail) for tail in tails]
        census = {tail: _tree_census(models / tail, device) for tail in tails}
        if intent is None and (not census[tails[0]]["exists"] or not any(item["files"] for item in census.values())):
            _refuse("runtime_admission_retire_not_needed")
        if intent is not None and intent["roots"] != roots:
            _refuse("runtime_admission_retire_intent_mismatch")
        keys, checked = config_paths_under(layout, roots)
        if keys:
            _refuse("runtime_admission_retire_model_under_root:" + keys[0])
        views, unit_summary = discover_units(manager, tuple(units), layout)
        offending = units_under(views, roots)
        if offending:
            _refuse("runtime_admission_retire_unit_under_root:" + offending[0])
        before = models_manifest(models, base)
        if intent is None:
            intent = {"schema_version": 1, "kind": "retire_intent", "decision_id": decision_id,
                      "container_slot": container_slot, "roots": roots, "pre": census,
                      "models_manifest_sha256": stage._digest(before), "units": unit_summary,
                      "config_keys_checked": checked, "intended_at": launcher._now_stamp(wall())}
            intent_name = RETIRE_INTENT_PREFIX + intent["intended_at"] + ".json"
            stage._atomic_record(stage.adoption_directory(root) / intent_name, intent, immutable=True)
        elif stage._digest(before) != intent["models_manifest_sha256"]:
            _refuse("runtime_admission_retire_outside_changed")
        if os.path.lexists(base):
            stage.make_writable_then_remove(base)  # the three roots all live under lora_adapters
        launcher._fsync_path(models)
        after = models_manifest(models, base)
        if after != before:
            _refuse("runtime_admission_retire_outside_changed")
        record_roots = []
        for tail, root_path in zip(tails, roots):
            if os.path.lexists(root_path):
                _refuse("runtime_admission_retire_absence")
            record_roots.append({"class": tail, "root": root_path, "pre": intent["pre"][tail], "absence": "enoent"})
        stamp = launcher._now_stamp(wall())
        record = {"schema_version": 1, "kind": "retire_legacy_training", "decision_id": decision_id,
                  "container_slot": container_slot, "roots": record_roots, "intent": intent_name,
                  "models_manifest_sha256": {"before": intent["models_manifest_sha256"],
                                             "after": stage._digest(after)},
                  "models_entries": len(after), "units_checked": unit_summary["checked"],
                  "units": unit_summary, "config_keys_checked": checked, "fsynced": [os.fspath(models)],
                  "retired_at": stamp}
        name = RETIREMENT_PREFIX + stamp + ".json"
        stage._atomic_record(stage.adoption_directory(root) / name, record, immutable=True)
    return {"record": name, "decision_id": decision_id, "intent": intent_name,
            "roots": [{"class": item["class"], "files": item["pre"]["files"], "bytes": item["pre"]["bytes"],
                       "absence": item["absence"]} for item in record_roots],
            "models_entries_unchanged": len(after), "units_checked": unit_summary["checked"]}


# --- the shared pipeline ----------------------------------------------------------------------------

def _head_state(root: Path) -> tuple[dict | None, str | None]:
    """(head, problem): adoption needs no head, or an aborted adoption head (re-adoption)."""
    head = stage.read_admission_head(root)
    if head is None:
        return None, None
    if head["status"] != "closed" or not stage.head_aborted(root, head):
        return head, "runtime_admission_adoption_head_present"
    return head, None


def _head_slot(root: Path) -> str | None:
    head = stage.read_admission_head(root)
    return None if head is None else stage.read_binding(root, head["binding_id"])["slot"]


def _pending_prepared(root: Path) -> list[str]:
    """Marked (never committed) slots, unmarked slots of a commit that never reached its CAS, leftovers.

    A commit removes the copy marker after its intent record (review F1); an
    unmarked slot named by an intent and bound by no head is a full copy
    of the production state outside every lineage until W-abort-1 removes it.
    """
    found = []
    intended = {intent["slot"] for intent in _commit_intents(root)}
    bound = _head_slot(root) if intended else None
    for entry in sorted(os.listdir(root / "slots")):
        if os.path.lexists(root / "slots" / entry / "state" / adoption.COPY_MARKER):
            found.append("slots/" + entry)
        elif entry in intended and entry != bound:
            found.append("slots/" + entry)
    work = root / WORK_DIRECTORY
    if os.path.lexists(work) and (not os.path.isdir(work) or os.path.islink(work) or os.listdir(work)):
        found.append(WORK_DIRECTORY)
    return found


def _identifier(*parts) -> str:
    return hashlib.sha256(json.dumps([*parts, os.urandom(16).hex()], sort_keys=True).encode()).hexdigest()


def _census_summary(report: dict) -> dict:
    """What the operator must see before the window (0e2 F2): legacy-path targets and .env classes."""
    foreign = report.get("foreign_targets") or []
    return {"descriptor_classes": report.get("descriptor_classes"), "foreign_targets": foreign,
            "legacy_checkout_targets": sum(1 for _participant, klass in foreign if klass == "legacy_checkout"),
            "needs_reviewed_rule": sorted({participant for participant, _klass in foreign}),
            "env": report.get("env")}


# --- adoption-dry-run (advisory, 0d P3 pattern) -----------------------------------------------------

_VOLATILE = ("timings", "free", "scratch", "dry_run_sha256")


def adoption_dry_run(root, *, container_slot: str, archive, slot: str, scratch_parent, python: str, manager,
                     timers: tuple[str, ...] = (), code_index_unit: str = CODE_INDEX_UNIT,
                     paths: ProductionPaths | None = None, tmpfs_required: bool = True,
                     wall=time.time, clock=time.monotonic) -> tuple[int, dict]:
    """The adoption on a private tmpfs scratch; nothing under the root is created, opened for write or locked.

    Returns (exit code, content-free report): 0 when the plan would be
    clean, 78 with the problems listed otherwise. The scratch is removed
    after success, refusal and SIGTERM (the CLI turns SIGTERM into an
    exception, so this function's ``finally`` runs).
    """
    started = clock()
    paths = paths or production_paths()
    root = checked_root(root)
    slot = _slot(slot, "runtime_admission_adoption_slot")
    report: dict = {"schema_version": SCHEMA_VERSION, "kind": "adoption_dry_run", "container_slot": container_slot,
                    "slot": slot, "p_answers": list(P_ANSWERS)}
    problems: list[str] = []
    warnings: list[str] = []
    layout, links = source_layout(root, container_slot, paths)
    report["readlinks"] = links
    try:
        current_link(root, container_slot)
        report["current"] = "container_runtime"
    except CommandRefusal as refusal:
        report["current"] = "other"
        problems.append(str(refusal))
    head, problem = _head_state(root)
    report["head"] = None if head is None else {"generation": head["generation"], "status": head["status"],
                                                "aborted": stage.head_aborted(root, head)}
    if problem:
        problems.append(problem)
    if os.path.lexists(root / "slots" / slot):
        problems.append("runtime_admission_adoption_slot_exists")
    pending = _pending_prepared(root)
    report["pending_prepared"] = len(pending)
    if pending:
        problems.append("runtime_admission_adoption_prepared_pending")
    report["units"], unit_problems = observe_units(manager, timers, window=False)
    report["code_index"], index_problems = code_index_check(manager, code_index_unit, layout)
    report["timer_dispositions"] = timer_dispositions(manager, timers, root, layout)
    problems += unit_problems + index_problems
    report["swap"] = swap_report()
    if pending_retire_intent(root)[1] is not None:
        problems.append("runtime_admission_adoption_retirement_incomplete")
    record, record_sha, record_name = latest_retirement(root)
    report["retirement"] = None if record is None else {"record": record_name, "sha256": record_sha,
                                                        "decision_id": record["decision_id"],
                                                        "absence": retirement_absence(record)}
    manifest, files, report["archive"] = read_r1_archive(archive)
    report["sizes"] = source_size(layout)
    parent, kind = checked_scratch(scratch_parent, tmpfs_required=tmpfs_required)
    for other in (root, *adoption.source_roots(layout)):
        resolved = os.path.realpath(other)
        if _under(os.fspath(parent), [resolved]) or _under(resolved, [os.fspath(parent)]):
            _refuse("runtime_admission_adoption_scratch_overlap", report)
    need = 2 * report["sizes"]["bytes"] + SCRATCH_FREE_MARGIN
    free = _free_bytes(parent)
    report["free"] = {"scratch_bytes": free, "scratch_needed": need, "scratch_ok": free >= need}
    if free < need:
        problems.append("runtime_admission_adoption_scratch_space")
    private = Path(tempfile.mkdtemp(prefix=".kza-adoption-dry-", dir=parent))
    removed = False
    try:
        runtime = private / "runtime"
        write_runtime(manifest, files, runtime)
        drop = private / "drop"
        drop.mkdir(mode=0o700)
        dry_id = _identifier("dry-run", os.fspath(root), slot, report["archive"]["sha256"])
        if kind != "tmpfs":
            adoption.write_scratch_marker(drop, dry_id)
        try:
            adopted = adoption.adopt_copy(layout, work=private / "work", scratch=drop,
                                          state_root=root / "slots" / slot / "state", plan_sha256=dry_id,
                                          adoption_time=float(wall()), python=python, runtime_root=runtime,
                                          retirement=retirement_view(record), rows=paths.rows(), advisory=True)
        except adoption.AdoptionRefusal as refusal:
            adopted = dict(getattr(refusal, "report", None) or {}, refused=str(refusal))
            problems.append(str(refusal))
        adopted.pop("plan_sha256", None)
        problems += adopted.get("refusals", [])
        if any(adopted.get("source_moving", {}).values()):
            warnings.append("advisory: source moving")
        if (adopted.get("legacy_training") or {}).get("retirement") == "assumed":
            warnings.append("advisory: legacy_training assumed retired (P1=D pending, W1b)")
        report["adoption"] = adopted
        report["legacy_path_census"] = _census_summary(adopted)
        if (private / "work").is_dir():
            report["journal_files"] = journal_file_count(private / "work")
            if report["journal_files"] > 1:
                problems.append("runtime_admission_recovery_erasure_journal_ambiguous")
    finally:
        removed = _remove_private(private)
    report["scratch"] = {"kind": kind, "removed": removed}
    if not removed:
        problems.append("runtime_admission_adoption_scratch_not_removed")
    report["problems"] = problems
    report["warnings"] = warnings
    report["dry_run_sha256"] = stage._digest({key: value for key, value in report.items() if key not in _VOLATILE})
    report["timings"] = {"seconds": round(clock() - started, 3)}
    return (0 if not problems else int(launcher.ExitCode.REFUSED)), report


# --- adoption-prepare (W2) --------------------------------------------------------------------------

def prepared_manifest(state: Path) -> list[list]:
    """The slot state's full-file manifest without ``-shm`` files.

    A read-only reader of a WAL database (the binding snapshot itself)
    creates or touches its -shm, which SQLite rebuilds and never treats as
    authoritative (as stage_runtime.state_manifest_sha256, review K1). Every
    other file, -wal included, keeps (path, inode, size, mtime_ns, sha256).
    """
    return [entry for entry in adoption.file_manifest(state) if not entry[0].endswith("-shm")]


def _plan_path(root: Path, plan_sha256: str) -> Path:
    return stage.adoption_directory(root) / (PLAN_PREFIX + plan_sha256 + ".json")


def source_manifest_digest(layout, metrics_path: str) -> str:
    """The full-file source manifest's digest, recomputed now (read-only).

    Only the digest is kept under admission/ (plan, seal, adoption record):
    the per-file list (names, inodes, hashes) is never written, so nothing
    names a legacy file once P7 retires the legacy state (review F10).
    """
    return adoption.manifest_digest(adoption.source_manifest(layout, metrics_path))


def adoption_prepare(root, *, container_slot: str, archive, slot: str, decision_id: str, scratch_parent,
                     python: str, manager, timers: tuple[str, ...] = (), code_index_unit: str = CODE_INDEX_UNIT,
                     paths: ProductionPaths | None = None, tmpfs_required: bool = True, wall=time.time) -> dict:
    """W2: build the adopted slot (marked, never selected) and write the plan commit will verify."""
    if type(decision_id) is not str or DECISION.fullmatch(decision_id) is None:
        _refuse("runtime_admission_adoption_decision")
    paths = paths or production_paths()
    root = checked_root(root)
    slot = _slot(slot, "runtime_admission_adoption_slot")
    units, problems = observe_units(manager, timers, window=True)
    layout, links = source_layout(root, container_slot, paths)
    index, index_problems = code_index_check(manager, code_index_unit, layout)
    if problems or index_problems:
        _refuse((problems + index_problems)[0])
    manifest, files, archive_summary = read_r1_archive(archive)
    parent, kind = checked_scratch(scratch_parent, tmpfs_required=tmpfs_required)
    size = source_size(layout)
    if _free_bytes(root) < 2 * size["bytes"] + ROOT_FREE_MARGIN:
        _refuse("runtime_admission_adoption_root_space")
    if _free_bytes(parent) < 2 * size["bytes"] + SCRATCH_FREE_MARGIN:
        _refuse("runtime_admission_adoption_scratch_space")
    with stage.AdmissionController(root):
        head, problem = _head_state(root)
        if problem:
            _refuse(problem)
        abort_sha = None if head is None else stage._digest(stage.read_abort_record(root, head))
        if _pending_prepared(root):
            _refuse("runtime_admission_adoption_prepared_pending")
        target = root / "slots" / slot
        if os.path.lexists(target):
            _refuse("runtime_admission_adoption_slot_exists")
        current = current_link(root, container_slot)
        if pending_retire_intent(root)[1] is not None:
            _refuse("runtime_admission_adoption_retirement_incomplete")
        record, record_sha, record_name = latest_retirement(root)
        prepare_id = _identifier("prepare", os.fspath(root), slot, archive_summary["sha256"])
        staging = root / WORK_DIRECTORY / slot
        stage._durable_parents(staging.parent)
        private = Path(tempfile.mkdtemp(prefix=".kza-adoption-", dir=parent))
        built = False
        try:
            drop = private / "drop"
            drop.mkdir(mode=0o700)
            if kind != "tmpfs":
                adoption.write_scratch_marker(drop, prepare_id)
            staging.mkdir(mode=0o700)
            runtime = staging / "runtime"
            write_runtime(manifest, files, runtime)
            for name in ("config", "data"):
                (runtime / name).symlink_to(Path("..") / "state" / name)
            adoption_time = float(wall())
            secret: dict = {}
            report = adoption.adopt_copy(layout, work=staging / "state", scratch=drop, state_root=target / "state",
                                         plan_sha256=prepare_id, adoption_time=adoption_time, python=python,
                                         runtime_root=runtime, retirement=retirement_view(record),
                                         rows=paths.rows(), private=secret)
            if os.path.lexists(staging / "state" / ".env"):
                (runtime / ".env").symlink_to(Path("..") / "state" / ".env")
            stage.make_runtime_read_only(runtime)
            if os.lstat(staging).st_dev != os.lstat(root / "slots").st_dev:
                _refuse("runtime_admission_adoption_cross_filesystem")
            os.rename(staging, target)  # one rename: every inode the prepared manifest pins is kept
            launcher._fsync_path(root / "slots")
            launcher._fsync_path(staging.parent)
            built = True
            if adoption.manifest_digest(adoption.file_manifest(target / "state")) != report["prepared_manifest_sha256"]:
                _refuse("runtime_admission_adoption_rename_changed")
            observation: dict = {}
            binding = stage._binding_snapshot(root, slot, artifacts=observation)
            # After the snapshot: its read-only images may leave a WAL database's empty -wal.
            entries = prepared_manifest(target / "state")
            committed = [entry for entry in entries if entry[0] != adoption.COPY_MARKER]
            source_digest = adoption.manifest_digest(secret["source_manifest"])
            if source_digest != report["source_manifest_sha256"]:
                _refuse("runtime_admission_adoption_manifest_invalid")
            plan = {
                "schema_version": SCHEMA_VERSION, "kind": "adoption_plan", "prepare_id": prepare_id,
                "root": os.fspath(root), "slot": slot, "container_slot": container_slot,
                "layout": layout_record(layout), "readlinks": links, "current_readlink": current,
                "expected_head": head, "previous_abort_sha256": abort_sha, "archive": archive_summary,
                "paths": paths.as_dict(),
                "source": {"manifest_sha256": source_digest, "files": len(secret["source_manifest"]),
                           "metrics_path": secret["metrics_path"]},
                "retirement": None if record is None else {
                    "record": record_name, "sha256": record_sha, "decision_id": record["decision_id"],
                    "absence": retirement_absence(record)},
                "tables": {"transform_sha256": report["transform_sha256"], "tables": report["tables"]},
                "confirm_foreign": report["confirm_foreign"], "p_answers": list(P_ANSWERS),
                "window_decision_id": decision_id, "units": units, "adoption_time": adoption_time,
                "timer_dispositions": timer_dispositions(manager, timers, root, layout),
                "code_index": index,
                "report": report,
                "prepared": {"manifest_sha256": adoption.manifest_digest(entries),
                             "library_manifest_sha256": report["prepared_manifest_sha256"],
                             "committed_manifest_sha256": adoption.manifest_digest(committed),
                             "state_manifest_sha256": stage.state_manifest_sha256(target / "state"),
                             "binding": binding, "binding_sha256": stage._digest(binding),
                             "closed_state_sha256": stage.closed_state_sha256(binding),
                             "artifact_observation_sha256": stage._digest(observation[os.fspath(target / "runtime")]),
                             "copy_marker": prepare_id},
                "prepared_at": launcher._now_stamp(adoption_time)}
            plan_sha256 = stage._digest(plan)
            stage._atomic_record(_plan_path(root, plan_sha256), plan, immutable=True)
        except BaseException:
            _remove_private(target if built else staging)
            raise
        finally:
            _remove_private(private)
            if os.path.isdir(staging.parent) and not os.listdir(staging.parent):
                os.rmdir(staging.parent)
    return {"plan_sha256": plan_sha256, "slot": slot, "source_files": plan["source"]["files"],
            "retirement": report["legacy_training"]["retirement"], "seed": report["seed"],
            "residue": report["residue"], "legacy_path_census": _census_summary(report),
            "scratch": kind, "head": "none" if head is None else "aborted"}


# --- adoption-commit (W4) ---------------------------------------------------------------------------

def load_plan(root: Path, plan_sha256: str) -> dict:
    if type(plan_sha256) is not str or _HEX64.fullmatch(plan_sha256) is None:
        _refuse("runtime_admission_adoption_confirm")
    try:
        plan = stage._read_json(_plan_path(root, plan_sha256))
    except FileNotFoundError:
        _refuse("runtime_admission_adoption_plan_unknown")
    if stage._digest(plan) != plan_sha256 or plan.get("kind") != "adoption_plan" or plan.get("root") != os.fspath(root):
        _refuse("runtime_admission_adoption_plan_unknown")
    return plan


def _adoption_records(root: Path) -> list[tuple[Path, dict]]:
    directory = stage.adoption_directory(root)
    if not os.path.isdir(directory):
        return []
    return [(directory / name, stage._read_json(directory / name)) for name in sorted(os.listdir(directory))
            if name.startswith("adoption-") and name.endswith(".json")]


def _mismatch(item: str):
    _refuse("runtime_admission_adoption_plan_mismatch:" + item)


def _layout_of(plan: dict):
    values = plan["layout"]
    return adoption.SourceLayout(**{key: Path(value) for key, value in values.items()})


def _without_binding_marker(root: Path, state: Path, binding: dict) -> list[list]:
    """prepared_manifest(state) without the state's binding marker, which must name exactly ``binding``.

    commit_adoption_head writes the marker just before its CAS; a commit
    re-run after a crash there finds it (review F1). Any other content refuses.
    """
    entries = prepared_manifest(state)
    path = state / stage._MARKER
    if not os.path.lexists(path):
        return entries
    if os.path.islink(path) or stage._read_json(path) != {
            "schema_version": 1, "admission_root": str(root), "binding_id": stage._digest(binding),
            "state_root": binding["state_root"]}:
        _refuse("runtime_admission_adoption_binding_marker")
    return [entry for entry in entries if entry[0] != stage._MARKER]


def _check_prepared(root: Path, plan: dict) -> bool:
    """The prepared slot is byte-for-byte the planned one; returns whether the copy marker is still there."""
    prepared = plan["prepared"]
    state = root / "slots" / plan["slot"] / "state"
    marker = state / adoption.COPY_MARKER
    try:
        digest = adoption.manifest_digest(_without_binding_marker(root, state, prepared["binding"]))
    except (OSError, ValueError, CommandRefusal):
        _mismatch("prepared")
    marked = os.path.lexists(marker)
    if marked and os.path.lexists(state / stage._MARKER):
        _mismatch("prepared")
    if marked:
        try:
            value = json.loads(Path(marker).read_text())
        except (OSError, ValueError):
            _mismatch("prepared")
        if (digest != prepared["manifest_sha256"] or value != {"schema_version": 1, "plan_sha256": plan["prepare_id"]}
                or stage.state_manifest_sha256(state) != prepared["state_manifest_sha256"]):
            _mismatch("prepared")
    elif digest != prepared["committed_manifest_sha256"]:
        _mismatch("prepared")
    observation: dict = {}
    try:
        binding = stage._binding_snapshot(root, plan["slot"], artifacts=observation)
    except (OSError, ValueError):
        _mismatch("prepared")
    if (binding != prepared["binding"] or stage.closed_state_sha256(binding) != prepared["closed_state_sha256"]
            or stage._digest(observation[os.fspath(root / "slots" / plan["slot"] / "runtime")])
            != prepared["artifact_observation_sha256"]):
        _mismatch("prepared")
    return marked


def verify_plan(root: Path, plan: dict, manager, paths: ProductionPaths) -> bool:
    """Every plan item, re-read now (design v3 section 5); returns whether the copy marker is present.

    Order: cheap reads first, the full-file source re-hash and the prepared
    slot last. Any difference refuses runtime_admission_adoption_plan_mismatch:<item>
    before any record or CAS.
    """
    head = stage.read_admission_head(root)
    if head != plan["expected_head"]:
        _mismatch("head")
    if head is not None and (not stage.head_aborted(root, head)
                             or stage._digest(stage.read_abort_record(root, head)) != plan["previous_abort_sha256"]):
        _mismatch("head")
    try:
        launcher.check_controller_epoch(root)
    except launcher.LauncherRefusal:
        _mismatch("controller")
    controller = stage._read_json(root / "admission" / "controller.json")
    if controller.get("controller_epoch") != stage.CONTROLLER_EPOCH or controller.get("commit") != plan["archive"]["git_commit"]:
        _mismatch("controller")
    units, problems = observe_units(manager, tuple(item for item in plan["units"]["units"] if item != launcher.UNIT),
                                    window=True)
    if problems or units != plan["units"]:
        _mismatch("units")
    try:
        if current_link(root, plan["container_slot"]) != plan["current_readlink"]:
            _mismatch("current")
    except CommandRefusal:
        _mismatch("current")
    if paths.as_dict() != plan["paths"]:
        _mismatch("tables")
    try:
        layout, links = source_layout(root, plan["container_slot"], paths)
    except (CommandRefusal, adoption.AdoptionRefusal):
        _mismatch("layout")
    if links != plan["readlinks"] or layout_record(layout) != plan["layout"]:
        _mismatch("layout")
    record, record_sha, record_name = latest_retirement(root)
    retirement = plan["retirement"]
    if (record is None) != (retirement is None) or (record is not None and (
            record_sha != retirement["sha256"] or record_name != retirement["record"]
            or retirement_absence(record) != retirement["absence"]
            or any(state == "present" for _root, state in retirement["absence"]))):
        _mismatch("retirement")
    if pending_retire_intent(root)[1] is not None:
        _mismatch("retirement")
    if retirement is None:  # review F7a: the roots prepare saw empty or absent must still hold no file
        models = Path(os.path.realpath(layout.models_root))
        try:
            if any(adoption._root_census(models / tail)["files"] for tail, _new in adoption.LEGACY_TRAINING_OUTSIDE):
                _mismatch("retirement")
        except adoption.AdoptionRefusal:
            _mismatch("retirement")
    timers = tuple(item for item in plan["units"]["units"] if item != launcher.UNIT)
    if timer_dispositions(manager, timers, root, layout) != plan["timer_dispositions"]:
        _mismatch("units")  # review F7b
    index, index_problems = code_index_check(manager, plan["code_index"]["unit"], layout)
    if index_problems or index != plan["code_index"]:
        _mismatch("code_index")  # review F7c
    archive = Path(plan["archive"]["path"])
    try:
        if _sha256(archive) != plan["archive"]["sha256"]:
            _mismatch("archive")
    except OSError:
        _mismatch("archive")
    if (adoption.transform_digest(paths.rows()) != plan["tables"]["transform_sha256"]
            or adoption.table_digests() != plan["tables"]["tables"]
            or plan["confirm_foreign"] != [] or plan["p_answers"] != list(P_ANSWERS)):
        _mismatch("tables")
    try:
        now = source_manifest_digest(layout, plan["source"]["metrics_path"])
    except (OSError, ValueError):
        _mismatch("source")
    if now != plan["source"]["manifest_sha256"]:
        _mismatch("source")
    return _check_prepared(root, plan)


def _publish(root: Path, slot: str) -> None:
    with stage._locked_root(root):
        stage._publish_selection(root, slot)
        launcher._fsync_path(root)


def _commit_intent_path(root: Path, plan_sha256: str) -> Path:
    return stage.adoption_directory(root) / (COMMIT_INTENT_PREFIX + plan_sha256 + ".json")


COMMIT_INTENT_KEYS = frozenset({"schema_version", "kind", "plan_sha256", "slot", "prepare_id", "committed_at"})


def _commit_intents(root: Path) -> list[dict]:
    directory = stage.adoption_directory(root)
    if not os.path.isdir(directory):
        return []
    found = []
    for name in sorted(os.listdir(directory)):
        if name.startswith(COMMIT_INTENT_PREFIX) and name.endswith(".json"):
            intent = stage._read_json(directory / name)
            if (set(intent) != COMMIT_INTENT_KEYS or intent.get("schema_version") != 1
                    or name != COMMIT_INTENT_PREFIX + str(intent.get("plan_sha256")) + ".json"):
                _refuse("runtime_admission_adoption_commit_intent_invalid")
            found.append(intent)
    return found


def _commit_intent(root: Path, plan: dict, plan_sha256: str, wall) -> dict:
    """The immutable commit intent: written once before the copy marker goes; every re-run reuses it.

    It fixes committed_at, so every record commit_adoption_head writes is
    the same on a re-run and the immutable writes match (review F1).
    """
    path = _commit_intent_path(root, plan_sha256)
    if os.path.lexists(path):
        intent = stage._read_json(path)
        if (set(intent) != COMMIT_INTENT_KEYS or intent["plan_sha256"] != plan_sha256
                or intent["slot"] != plan["slot"] or intent["prepare_id"] != plan["prepare_id"]):
            _refuse("runtime_admission_adoption_commit_intent_invalid")
        return intent
    intent = {"schema_version": 1, "kind": "adoption_commit_intent", "plan_sha256": plan_sha256,
              "slot": plan["slot"], "prepare_id": plan["prepare_id"],
              "committed_at": launcher._now_stamp(float(wall()))}
    stage._atomic_record(path, intent, immutable=True)
    return intent


def _prepared_aborted(root: Path, plan_sha256: str) -> bool:
    directory = stage.adoption_directory(root)
    if not os.path.isdir(directory):
        return False
    return any(stage._read_json(directory / name).get("plan_sha256") == plan_sha256
               for name in os.listdir(directory) if name.startswith(PREPARED_ABORT_PREFIX))


def adoption_commit(root, *, confirm: str, manager, paths: ProductionPaths | None = None, wall=time.time) -> dict:
    """W4: re-verify the whole plan, then the records, seal, adoption head, marker removal and 'current'.

    Recoverable at every crash point (review F1): the commit intent is
    written before the copy marker is removed and fixes every record field;
    a re-run before the CAS re-verifies the whole plan (an unmarked slot and
    the matching binding marker included) and re-enters
    commit_adoption_head, whose immutable writes then match; a re-run after
    the CAS only publishes 'current'. W-abort-1 removes a slot whose CAS
    never happened.
    """
    paths = paths or production_paths()
    root = checked_root(root)
    with stage.AdmissionController(root):
        plan = load_plan(root, confirm)
        slot = plan["slot"]
        if _prepared_aborted(root, confirm):
            _refuse("runtime_admission_adoption_plan_aborted")
        head = stage.read_admission_head(root)
        for _path, record in _adoption_records(root):
            if record.get("plan_sha256") != confirm:
                continue
            runtime = os.fspath(root / "slots" / slot / "runtime")
            if head == record.get("head"):
                if not stage.head_aborted(root, head) and os.path.realpath(root / "current") != runtime:
                    _publish(root, slot)  # resume: CAS done, 'current' not yet published
                    return {"head": head, "slot": slot, "resumed": "publish"}
                _refuse("runtime_admission_adoption_already_committed")
            if head != plan["expected_head"]:
                _refuse("runtime_admission_adoption_already_committed")
            # else: the records exist but the CAS never happened; re-verify and re-enter below.
        marked = verify_plan(root, plan, manager, paths)
        intent = _commit_intent(root, plan, confirm, wall)
        state = root / "slots" / slot / "state"
        if marked:
            os.unlink(state / adoption.COPY_MARKER)
            launcher._fsync_path(state)
        if (adoption.manifest_digest(_without_binding_marker(root, state, plan["prepared"]["binding"]))
                != plan["prepared"]["committed_manifest_sha256"]):
            _mismatch("prepared")
        served_manifest = stage.state_manifest_sha256(state)
        retirement = plan["retirement"]
        report = plan["report"]
        controller = stage._read_json(root / "admission" / "controller.json")
        seal_fields = {
            "plan_sha256": confirm, "archive_sha256": plan["archive"]["sha256"],
            "controller_commit": controller["commit"],
            "source_manifest_sha256": plan["source"]["manifest_sha256"],
            "prepared_manifest_sha256": plan["prepared"]["manifest_sha256"],
            "committed_manifest_sha256": plan["prepared"]["committed_manifest_sha256"],
            "state_manifest_sha256": served_manifest,
            "closed_state_sha256": plan["prepared"]["closed_state_sha256"],
            "artifact_observation_sha256": plan["prepared"]["artifact_observation_sha256"],
            "transform_sha256": plan["tables"]["transform_sha256"], "tables": plan["tables"]["tables"],
            "retirement_record_sha256": None if retirement is None else retirement["sha256"],
            "window_decision_id": plan["window_decision_id"], "p_answers": list(P_ANSWERS)}
        decision = report["legacy_training"]["retirement"]
        record_fields = {
            "kind": "adoption", "plan_sha256": confirm, "slot": slot, "container_slot": plan["container_slot"],
            "current_before": plan["current_readlink"], "source_manifest_sha256": plan["source"]["manifest_sha256"],
            "source_files": plan["source"]["files"], "metrics_path": plan["source"]["metrics_path"],
            "layout": plan["layout"],
            "retirement_record": None if retirement is None else retirement["record"],
            "legacy_training": ("no retirement needed" if decision == "not_needed" else
                                "the pending legacy_training steps complete at the first managed run because of "
                                "retirement record " + retirement["record"] + " (decision " + decision + ")"),
            "seeded": report["seed"], "requeued": report["residue"],
            "committed_at": intent["committed_at"]}
        head = stage.commit_adoption_head(root, plan["expected_head"], plan["prepared"]["binding"],
                                          seal_fields=seal_fields, record_fields=record_fields)
        _publish(root, slot)
    return {"head": head, "slot": slot, "seal": stage.read_manual_head(root, head)["authority_seal"],
            "record": stage.adoption_record_path(root, head).name, "current": "slots/" + slot + "/runtime"}


# --- adoption-abort (W-abort-2, P6=A; and W-abort-1) ------------------------------------------------

def _runs_happened(root: Path) -> bool:
    for name in RUN_RECORD_DIRECTORIES:
        directory = root / "admission" / name
        if os.path.lexists(directory) and (not os.path.isdir(directory) or os.listdir(directory)):
            return True
    return False


def _restore_current(root: Path, raw: str) -> None:
    with stage._locked_root(root):
        with tempfile.TemporaryDirectory(prefix=".abort-", dir=root) as temporary:
            link = Path(temporary) / "current"
            link.symlink_to(raw)
            os.replace(link, root / "current")
        launcher._fsync_path(root)


def _legacy_conf(root: Path, unit_dir: Path, container_slot: str, manager) -> dict | None:
    """As uninstall-launcher renders it, when install-launcher already ran (W6)."""
    managed, disabled, legacy = unit_dir / "managed.conf", unit_dir / "managed.conf.disabled", unit_dir / "legacy.conf"
    if not os.path.lexists(managed) and not os.path.lexists(disabled):
        return None
    record = launcher._latest_install(root)
    runtime, state, _commit = launcher._verify_legacy_slot(root, container_slot, launcher.LEGACY_COMMITS)
    text = launcher.render_legacy_conf(record, runtime, state, unit_dir / "release.conf.unmanaged")
    if os.path.lexists(managed):
        os.replace(managed, disabled)
    # A re-run after a stop between the rename and the abort record re-renders and re-verifies (review F11).
    if not os.path.lexists(legacy) or os.path.islink(legacy) or Path(legacy).read_text() != text:
        launcher._write_text_atomic(legacy, text)
    if not manager.daemon_reload(strict=True):  # review F3: exit 0 only
        _refuse("runtime_admission_adoption_daemon_reload")
    shown = manager.user_show(launcher.UNIT, launcher.RECORDED_PROPERTIES)
    return {"sha256": hashlib.sha256(text.encode()).hexdigest(),
            "mismatches": launcher._legacy_mismatches(record, shown, runtime, state)}


def _require_inactive(manager) -> dict:
    shown = manager.user_show(launcher.UNIT, UNIT_PROPERTIES)
    if shown is None:
        _refuse("runtime_admission_adoption_units_unavailable")
    if shown["ActiveState"] not in ("inactive", "failed"):
        _refuse("runtime_admission_adoption_unit_active")
    return shown


def _enable(manager) -> bool:
    return bool(manager.user_enable(launcher.UNIT))


def adoption_abort(root, *, manager, unit_dir, paths: ProductionPaths | None = None, wall=time.time) -> dict:
    """W-abort-2 (section 7, P6=A): back to the legacy unit before any managed run, without the phrase.

    Requires: kza-voice inactive; the head is exactly an adoption head
    (closed) with its adoption record; no run, receipt, recovery or closure
    record exists at all; the full-file source manifest equals the recorded
    one. Then: legacy.conf if managed.conf exists, 'current' restored, the
    abort record (keyed by the head digest; begin_run refuses that head from
    now on), the never-served slot removed, kza-voice re-enabled (not started).
    """
    paths = paths or production_paths()
    root = checked_root(root)
    unit_dir = Path(unit_dir)
    voice = _require_inactive(manager)
    with stage.AdmissionController(root):
        head = stage.read_admission_head(root)
        if head is None or head["status"] != "closed" or not os.path.lexists(stage.adoption_record_path(root, head)):
            _refuse("runtime_admission_adoption_abort_head")
        if stage.read_manual_head(root, head)["kind"] != "adoption":
            _refuse("runtime_admission_adoption_abort_head")
        record = stage._read_json(stage.adoption_record_path(root, head))
        slot = record["slot"]
        target = root / "slots" / slot
        if stage.head_aborted(root, head):
            stage.read_abort_record(root, head)
            # Resume only while this abort is the newest adoption event (review F11): a plan built
            # over the aborted head means a later window started, and enabling kza-voice then is wrong.
            if any(plan["expected_head"] == head for _digest, plan in _plans(root)):
                _refuse("runtime_admission_adoption_already_aborted")
            pending = os.path.lexists(target) or voice["UnitFileState"] != "enabled"
            if not pending:
                _refuse("runtime_admission_adoption_already_aborted")
            if os.path.lexists(target):
                stage.make_writable_then_remove(target)
                launcher._fsync_path(root / "slots")
            return {"head": head, "resumed": True, "slot_removed": True, "enabled": _enable(manager)}
        if _runs_happened(root):
            _refuse("runtime_admission_adoption_abort_after_run")
        layout = adoption.SourceLayout(**{key: Path(value) for key, value in record["layout"].items()})
        try:
            now = source_manifest_digest(layout, record["metrics_path"])
        except (OSError, ValueError):
            _refuse("runtime_admission_adoption_abort_legacy_changed")
        if now != record["source_manifest_sha256"]:
            _refuse("runtime_admission_adoption_abort_legacy_changed")
        legacy_conf = _legacy_conf(root, unit_dir, record["container_slot"], manager)
        _restore_current(root, record["current_before"])
        stage.record_adoption_abort(root, head, {
            "plan_sha256": record["plan_sha256"], "slot": slot, "current_restored": record["current_before"],
            "legacy_conf": legacy_conf, "source_manifest_sha256": record["source_manifest_sha256"],
            "aborted_at": launcher._now_stamp(float(wall()))})
        stage.make_writable_then_remove(target)
        launcher._fsync_path(root / "slots")
    enabled = _enable(manager)
    return {"head": head, "abort_record": stage.abort_record_path(root, head).name, "slot_removed": True,
            "current": "restored", "legacy_conf": legacy_conf, "enabled": enabled}


def _plans(root: Path) -> list[tuple[str, dict]]:
    directory = stage.adoption_directory(root)
    if not os.path.isdir(directory):
        return []
    found = []
    for name in sorted(os.listdir(directory)):
        if name.startswith(PLAN_PREFIX) and name.endswith(".json"):
            digest = name[len(PLAN_PREFIX):-len(".json")]
            found.append((digest, load_plan(root, digest)))
    return found


def _plan_of_prepare(root: Path, slot: str, prepare_id) -> str | None:
    """The plan a marked slot was prepared for (None when prepare stopped before writing it)."""
    return next((digest for digest, plan in _plans(root)
                 if plan["slot"] == slot and plan["prepare_id"] == prepare_id), None)


def _uncommitted_plan(root: Path, slot: str) -> str | None:
    """The plan whose commit intent names this unmarked slot and whose committed manifest it still is."""
    state = root / "slots" / slot / "state"
    for intent in reversed(_commit_intents(root)):
        if intent["slot"] != slot:
            continue
        plan = load_plan(root, intent["plan_sha256"])
        try:
            digest = adoption.manifest_digest(_without_binding_marker(root, state, plan["prepared"]["binding"]))
        except (OSError, ValueError):
            continue
        if digest == plan["prepared"]["committed_manifest_sha256"]:
            return intent["plan_sha256"]
    return None


def adoption_abort_prepared(root, *, slot: str, manager, wall=time.time) -> dict:
    """W-abort-1: remove a prepared, never-committed slot (or a prepare's leftover work directory).

    The retirement stays (irreversible, by decision). 'current' never
    changed. kza-voice is re-enabled, not started. An unmarked slot is
    removed only when a commit intent names it, its state is still that
    plan's committed manifest and no head binds it (a commit stopped before
    its CAS, review F1); the plan is then refused by adoption-commit.
    """
    root = checked_root(root)
    slot = _slot(slot, "runtime_admission_adoption_slot")
    _require_inactive(manager)
    with stage.AdmissionController(root):
        head = stage.read_admission_head(root)
        if head is not None and stage.read_binding(root, head["binding_id"])["slot"] == slot:
            _refuse("runtime_admission_adoption_abort_prepared_committed")
        target, work = root / "slots" / slot, root / WORK_DIRECTORY / slot
        if os.path.realpath(root / "current") == os.fspath(target / "runtime"):
            _refuse("runtime_admission_adoption_abort_prepared_committed")
        removed, plan_sha256 = [], None
        if os.path.lexists(target):
            marker = target / "state" / adoption.COPY_MARKER
            if os.path.lexists(marker):
                plan_sha256 = _plan_of_prepare(root, slot, stage._read_json(marker).get("plan_sha256"))
            else:
                plan_sha256 = _uncommitted_plan(root, slot)  # a commit that never reached its CAS (F1)
                if plan_sha256 is None:
                    _refuse("runtime_admission_adoption_abort_prepared_unmarked")
            removed.append("slots/" + slot)
        if os.path.lexists(work):
            removed.append(WORK_DIRECTORY + "/" + slot)
        if not removed:
            _refuse("runtime_admission_adoption_nothing_prepared")
        stamp = launcher._now_stamp(float(wall()))
        stage._atomic_record(stage.adoption_directory(root) / (PREPARED_ABORT_PREFIX + slot + "-" + stamp + ".json"),
                             {"schema_version": 1, "kind": "abort_prepared", "slot": slot, "removed": removed,
                              "plan_sha256": plan_sha256, "head": head, "aborted_at": stamp}, immutable=True)
        for relative in removed:
            stage.make_writable_then_remove(root / relative)
        launcher._fsync_path(root / "slots")
        if os.path.isdir(root / WORK_DIRECTORY) and not os.listdir(root / WORK_DIRECTORY):
            os.rmdir(root / WORK_DIRECTORY)
    return {"removed": removed, "enabled": _enable(manager)}


# --- CLI (dispatched from stage_runtime.main) -------------------------------------------------------

def add_commands(commands) -> None:
    """Register the adoption subcommands on stage_runtime's parser (names: design v3 section 6 N12e)."""
    retire = commands.add_parser("retire-legacy-training", help="W1b: retire the LoRA roots (P1=D, recorded decision)")
    retire.add_argument("--decision", required=True)
    retire.add_argument("--container-slot", required=True)
    retire.add_argument("--unit", action="append", default=[], help="Every enabled kza unit recorded at W0")
    for name, text in (("adoption-dry-run", "Advisory adoption on a private tmpfs scratch (read-only)"),
                       ("adoption-prepare", "W2: build the marked adoption slot and print plan_sha256")):
        parser = commands.add_parser(name, help=text)
        parser.add_argument("--container-slot", required=True)
        parser.add_argument("--archive", type=Path, required=True)
        parser.add_argument("--slot", required=True)
        parser.add_argument("--scratch", type=Path, default=Path(f"/run/user/{os.getuid()}"))
        parser.add_argument("--python", default=sys.executable)
        parser.add_argument("--window-timer", action="append", default=[])
        parser.add_argument("--code-index-unit", default=CODE_INDEX_UNIT)
        if name == "adoption-prepare":
            parser.add_argument("--decision", required=True, help="The window authorization decision id")
    commit = commands.add_parser("adoption-commit", help="W4: verify the plan, CAS the adoption head, publish")
    commit.add_argument("--confirm", required=True, help="plan_sha256 printed by adoption-prepare")
    abort = commands.add_parser("adoption-abort", help="W-abort-2 (or W-abort-1 with --prepared-slot)")
    group = abort.add_mutually_exclusive_group(required=True)
    group.add_argument("--unit-dir", type=Path)
    group.add_argument("--prepared-slot")


class _Terminated(BaseException):
    """SIGTERM/SIGINT during a command: unwinds through every ``finally`` (scratch removal)."""


def _terminate(signum, _frame):
    raise _Terminated(signum)


def run_command(args, *, manager=None) -> int:
    """Execute one adoption subcommand; returns the process exit code (78 on refusal)."""
    manager = manager if manager is not None else launcher.ServiceManager()
    previous = {signum: signal.signal(signum, _terminate) for signum in (signal.SIGTERM, signal.SIGINT)}
    report = None
    try:
        if args.command == "retire-legacy-training":
            result = retire_legacy_training(args.root, decision_id=args.decision, container_slot=args.container_slot,
                                            manager=manager, units=tuple(args.unit))
        elif args.command == "adoption-dry-run":
            code, result = adoption_dry_run(args.root, container_slot=args.container_slot, archive=args.archive,
                                            slot=args.slot, scratch_parent=args.scratch, python=args.python,
                                            manager=manager, timers=tuple(args.window_timer),
                                            code_index_unit=args.code_index_unit)
            sys.stdout.write(json.dumps(result, sort_keys=True, default=str) + "\n")
            if code:
                sys.stderr.write(json.dumps({"refused": result["problems"][0], "problems": result["problems"]}) + "\n")
            return code
        elif args.command == "adoption-prepare":
            result = adoption_prepare(args.root, container_slot=args.container_slot, archive=args.archive,
                                      slot=args.slot, decision_id=args.decision, scratch_parent=args.scratch,
                                      python=args.python, manager=manager, timers=tuple(args.window_timer),
                                      code_index_unit=args.code_index_unit)
        elif args.command == "adoption-commit":
            result = adoption_commit(args.root, confirm=args.confirm, manager=manager)
        elif args.prepared_slot is not None:
            result = adoption_abort_prepared(args.root, slot=args.prepared_slot, manager=manager)
        else:
            result = adoption_abort(args.root, manager=manager, unit_dir=args.unit_dir)
    except _Terminated:
        sys.stderr.write(json.dumps({"refused": "runtime_admission_adoption_terminated"}) + "\n")
        return int(launcher.ExitCode.REFUSED)
    except (ValueError, launcher.LauncherRefusal) as error:
        code = getattr(error, "code", None) or str(error)
        report = getattr(error, "report", None)
        refused = {"refused": code[:200]}
        if report:
            refused["report"] = report
        sys.stderr.write(json.dumps(refused, sort_keys=True, default=str) + "\n")
        return int(launcher.ExitCode.REFUSED)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    sys.stdout.write(json.dumps(result, sort_keys=True, default=str) + "\n")
    if args.command == "adoption-abort" and (result.get("legacy_conf") or {}).get("mismatches"):
        sys.stderr.write(json.dumps({"refused": "runtime_admission_adoption_legacy_conf_mismatch"}) + "\n")
        return int(launcher.ExitCode.REFUSED)
    if args.command.startswith("adoption-abort") and result.get("enabled") is False:
        sys.stderr.write(json.dumps({"refused": "runtime_admission_adoption_enable_failed"}) + "\n")
        return int(launcher.ExitCode.REFUSED)
    return 0
