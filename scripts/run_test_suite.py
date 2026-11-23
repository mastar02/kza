#!/usr/bin/env python3
"""Portable multi-worker test suite tier runner for KZA (T090-U1).

Supports Mac (Tiers 0 and 1 with pytest-xdist) and Linux kza-yo (Tiers 0, 1, and
Tier 2 full gate with parallel non-systemd core lane + dedicated systemd lane +
compiled-solo lane, automatic /dev/shm basetemp and journald NRestarts check).

Never imports src/ so entry_guard remains strictly respected.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import datetime as dt
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from typing import Literal, TextIO

REPO_ROOT = Path(__file__).resolve().parents[1]
MIN_SHM_FREE_BYTES = 16 * 1024**3  # 16 GiB safety margin on /dev/shm
RAM_BACKED_FSTYPES = frozenset({"tmpfs", "ramfs"})
CHILD_TERM_GRACE_S = 10.0
SHUTDOWN_SIGNALS = tuple(
    getattr(signal, name) for name in ("SIGHUP", "SIGTERM", "SIGINT") if hasattr(signal, name)
)
DEFAULT_SOLO_WINDOW_FILE = Path("/home/yo/projects/runs/SOLO_WINDOW")
TIER2_STEP_NAMES = {
    "node": "frontend-node",
    "a": "lane-a-core",
    "b-linux": "lane-b-linux",
    "b-sc6b1": "lane-b-sc6b1",
    "b-sc6b2": "lane-b-sc6b2",
    "c": "lane-c-compiled-solo",
}
TIER2_B_LANES = ("b-linux", "b-sc6b1", "b-sc6b2")
HOST_INTERFERENCE_UNITS = (
    "thermouy-tier2-nightly.service",
    "thermouy-tier2-nightly.timer",
    "thermouy-ci-prune.service",
    "thermouy-ci-prune.timer",
)
INHERITED_OVERRIDE_VARS: tuple[str, ...] = (
    "PYTEST_ADDOPTS",
    "KZA_SC6B_MEASURE",
    "KZA_REQUIRE_REAL_SYSTEMD",
    "KZA_LAUNCHER_ARTIFACTS_CACHE",
    "KZA_COMPACTION_ARTIFACT",
    "KZA_COMPACTION_ARTIFACT_EPOCH4",
)

# Five Linux/systemd integration modules reserved for Carril B on kza-yo.
LINUX_SYSTEMD_MODULES: tuple[str, ...] = (
    "tests/integration/test_linux_compaction_units.py",
    "tests/integration/test_linux_owner_compaction.py",
    "tests/integration/test_linux_adoption_window.py",
    "tests/integration/test_linux_managed_launcher.py",
    "tests/integration/test_linux_sc6b_release_rehearsal.py",
)

# Timing-budget compiled rehearsals run in a separate lane after the main parallel load.
COMPILED_SOLO_TESTS: tuple[str, ...] = (
    "tests/integration/test_copy_observation.py::test_compiled_existing_copy_observations_preserve_current_authority_without_grants",
    "tests/integration/test_copy_collector_cleanup.py::test_current_authority_cleans_mixed_collector_copies_and_drains_cancelled_native",
    "tests/integration/test_compiled_material_reconciliation.py::test_compiled_material_rollback_and_forward_preserve_latest_authority",
    "tests/integration/test_checkpoint_blob_cleanup.py::test_compiled_checkpoint_cleanup_unlinks_only_withdrawn_blob_and_preserves_historical_pins",
    "tests/integration/test_checkpoint_binary_lineage.py::test_compiled_binary_source_lineage_distinguishes_withdrawn_shared_and_manual",
    "tests/integration/test_compiled_output_registration_compatibility.py::test_compiled_v1_v2_histories_share_original_closed_controller",
    "tests/integration/test_compiled_slot_rollback.py::test_compiled_slot_rollback_preserves_latest_privacy_authority",
    "tests/integration/test_pre_u3b_owner_derivation.py::test_pre_u3b_state_derives_with_one_lookup_index_and_the_empty_owner_schema",
    "tests/integration/test_p6_bytecode_restore_reproducibility.py",
)

COMPILED_SOLO_MODULES: tuple[str, ...] = tuple(
    dict.fromkeys(nodeid.split("::", 1)[0] for nodeid in COMPILED_SOLO_TESTS)
)

DEFAULT_ENV: dict[str, str] = {
    "PYTHONFAULTHANDLER": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "ANONYMIZED_TELEMETRY": "False",
}


def default_workers(platform: str | None = None, cpu_count: int | None = None) -> int:
    plat = platform or sys.platform
    cpus = cpu_count or (os.cpu_count() or 4)
    if plat == "darwin":
        return max(1, min(8, cpus))
    # Cap Linux workers at 8 so RSS stays comfortably inside user@ MemoryHigh=32 GiB.
    return max(1, min(8, cpus // 2 if cpus >= 16 else cpus))


def resolve_solo_workers(
    explicit: int | None,
    *,
    workers: int,
    has_xdist: bool,
) -> int:
    """Resolve effective xdist workers for ``lane-c-compiled-solo``.

    Defaults to ``2`` when pytest-xdist is available and ``--workers >= 2`` so
    both workers run with dedicated CPU/disk after Lane A and Lane B complete,
    staying safely inside the real 55s ``advance-N`` phase budget; otherwise ``1``.
    """
    if explicit is not None:
        if explicit < 1:
            raise ValueError("--solo-workers must be >= 1")
        return explicit if has_xdist else 1
    if has_xdist and workers >= 2:
        return 2
    return 1


def _shm_has_capacity(shm_root: Path = Path("/dev/shm"), min_shm_bytes: int = MIN_SHM_FREE_BYTES) -> bool:
    if not shm_root.is_dir():
        return False
    try:
        statvfs = os.statvfs(shm_root)
        return (statvfs.f_bavail * statvfs.f_frsize) >= min_shm_bytes
    except OSError:
        return False


def _unescape_mount_field(field: str) -> str:
    """Decode the octal escapes (``\\040`` for a space) used by /proc/mounts."""
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), field)


def is_ram_backed_path(path: Path, *, mounts_file: Path = Path("/proc/mounts")) -> bool:
    """Return whether ``path`` lives on a tmpfs/ramfs mount (page cache charged to the cgroup).

    Uses the longest matching mount point in ``mounts_file``; returns ``False`` when
    the mount table is unavailable (non-Linux hosts).
    """
    try:
        resolved = os.path.realpath(path)
        lines = mounts_file.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return False
    best_len = -1
    best_type = ""
    for line in lines:
        fields = line.split()
        if len(fields) < 3:
            continue
        mount_point = os.path.normpath(_unescape_mount_field(fields[1]))
        prefix = mount_point if mount_point == "/" else mount_point + "/"
        if (resolved == mount_point or resolved.startswith(prefix)) and len(mount_point) > best_len:
            best_len, best_type = len(mount_point), fields[2]
    return best_type in RAM_BACKED_FSTYPES


def choose_basetemp_root(
    explicit: Path | None,
    out_dir: Path,
    *,
    tier: int = 0,
    platform: str | None = None,
    shm_root: Path = Path("/dev/shm"),
    min_shm_bytes: int = MIN_SHM_FREE_BYTES,
) -> tuple[Path, bool]:
    """Select basetemp root; use /dev/shm only for Tier 0 on Linux when free.

    Tier 1 and Tier 2 retain tens of GiB across the full pytest session, which can
    exhaust cgroup MemoryHigh/tmpfs RAM; those runs reserve a unique per-run
    directory under ``out_dir/basetemp-<pid>-*`` via ``mkdtemp`` and are cleaned up
    on exit so only logs/XML/summary remain without colliding or following symlinks.
    Returns ``(root_path, cleanup_on_exit)``.
    """
    if explicit is not None:
        resolved = Path(explicit).resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved, False

    plat = platform or sys.platform
    if tier == 0 and plat.startswith("linux") and _shm_has_capacity(shm_root, min_shm_bytes):
        shm_tmp = Path(tempfile.mkdtemp(prefix=f"kza-pytest-{os.getpid()}-", dir=shm_root))
        return shm_tmp, True

    out_dir_resolved = Path(out_dir).resolve()
    out_dir_resolved.mkdir(parents=True, exist_ok=True)
    disk_tmp = Path(tempfile.mkdtemp(prefix=f"basetemp-{os.getpid()}-", dir=out_dir_resolved))
    return disk_tmp, True


def choose_cache_dir(
    explicit: Path | None,
    *,
    platform: str | None = None,
    shm_root: Path = Path("/dev/shm"),
    min_shm_bytes: int = MIN_SHM_FREE_BYTES,
    home_dir: Path | None = None,
) -> Path:
    """Select persistent disk cache directory so artifacts stay off tmpfs RAM."""
    del platform, shm_root, min_shm_bytes
    if explicit is not None:
        return Path(explicit).resolve()
    base_home = home_dir or Path.home()
    return (base_home / ".cache" / "kza-runtime-artifacts").resolve()


def sanitize_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Build deterministic runner env stripping inherited pytest/gate overrides."""
    cleaned = dict(base_env if base_env is not None else os.environ)
    for var in INHERITED_OVERRIDE_VARS:
        cleaned.pop(var, None)
    cleaned.update(DEFAULT_ENV)
    return cleaned


@dataclass
class SoloWindowGuard:
    """Host SOLO_WINDOW + flock exclusion handle released cleanly in nested finally."""

    marker_path: Path | None = None
    lock_fh: TextIO | None = None

    def release(self) -> None:
        try:
            if self.marker_path is not None:
                self.marker_path.unlink(missing_ok=True)
        finally:
            if self.lock_fh is not None:
                try:
                    fcntl.flock(self.lock_fh.fileno(), fcntl.LOCK_UN)
                finally:
                    self.lock_fh.close()
                    self.lock_fh = None


def acquire_solo_window(
    *,
    tier: int,
    platform: str | None = None,
) -> SoloWindowGuard:
    """Enforce mutual exclusion on Linux via the single canonical host lock + SOLO_WINDOW.

    All Linux tiers (0, 1 and 2) share ``DEFAULT_SOLO_WINDOW_FILE`` and
    ``DEFAULT_SOLO_WINDOW_FILE.parent / "feature-pytest.lock"`` so no invocation
    can pass an alternate lock path, overlap, or start unlocked before Tier 2.
    Tiers 0/1 hold ``LOCK_SH`` while Tier 2 holds ``LOCK_EX`` + ``O_EXCL`` marker.
    """
    plat = platform or sys.platform
    if not plat.startswith("linux"):
        return SoloWindowGuard()

    target_solo = DEFAULT_SOLO_WINDOW_FILE
    if not target_solo.parent.is_dir():
        raise RuntimeError(
            f"Required host exclusion directory missing at {target_solo.parent}; "
            f"refusing unlocked Tier {tier} run on Linux"
        )

    lock_path = target_solo.parent / "feature-pytest.lock"
    lock_fh = lock_path.open("a+", encoding="utf-8")
    lock_mode = (fcntl.LOCK_EX if tier == 2 else fcntl.LOCK_SH) | fcntl.LOCK_NB
    try:
        fcntl.flock(lock_fh.fileno(), lock_mode)
    except OSError as exc:
        lock_fh.close()
        raise RuntimeError(
            f"Active test run holds host lock {lock_path}; refusing concurrent Tier {tier} run"
        ) from exc

    if target_solo.exists() or target_solo.is_symlink():
        try:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
        finally:
            lock_fh.close()
        raise RuntimeError(f"SOLO_WINDOW active at {target_solo}; refusing concurrent Tier {tier} run")

    if tier != 2:
        return SoloWindowGuard(marker_path=None, lock_fh=lock_fh)

    try:
        fd = os.open(target_solo, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as exc:
        try:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
        finally:
            lock_fh.close()
        raise RuntimeError(f"SOLO_WINDOW already active at {target_solo}; refusing concurrent Tier 2 run") from exc

    try:
        fh = os.fdopen(fd, "w", encoding="utf-8")
    except BaseException:
        try:
            os.close(fd)
        finally:
            try:
                target_solo.unlink(missing_ok=True)
            finally:
                try:
                    fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
                finally:
                    lock_fh.close()
        raise

    try:
        with fh:
            fh.write(
                f"run_test_suite tier=2 pid={os.getpid()} started={dt.datetime.now(dt.timezone.utc).isoformat()}\n"
            )
            fh.flush()
    except BaseException:
        try:
            target_solo.unlink(missing_ok=True)
        finally:
            try:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
            finally:
                lock_fh.close()
        raise
    return SoloWindowGuard(marker_path=target_solo, lock_fh=lock_fh)


def safe_rmtree(path: Path) -> None:
    """Remove directory trees containing 0555 directories without touching hardlinked file modes or symlinks."""
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink():
        path.unlink()
        return
    for _root, _dirs, _files, fd in os.fwalk(path, follow_symlinks=False):
        mode = stat.S_IMODE(os.fstat(fd).st_mode)
        if (mode & stat.S_IRWXU) != stat.S_IRWXU:
            os.fchmod(fd, mode | stat.S_IRWXU)
    shutil.rmtree(path, ignore_errors=False)


def _xdist_available(python_exe: Path) -> bool:
    if Path(python_exe).resolve() == Path(sys.executable).resolve():
        return importlib.util.find_spec("xdist") is not None
    chk = subprocess.run(
        [str(python_exe), "-c", "import xdist"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return chk.returncode == 0


def build_pytest_command(
    python_exe: Path,
    *,
    targets: list[str],
    workers: int,
    basetemp: Path,
    junit_xml: Path | None = None,
    ignores: tuple[str, ...] = (),
    deselects: tuple[str, ...] = (),
    k_expr: str | None = None,
    use_xdist: bool = True,
    dist_mode: str = "worksteal",
    extra_args: tuple[str, ...] = (),
) -> list[str]:
    cmd = [str(python_exe), "-m", "pytest", "-p", "no:cacheprovider"]
    if use_xdist and workers > 1:
        cmd.extend(["-n", str(workers), f"--dist={dist_mode}"])
    cmd.extend(["-o", "faulthandler_timeout=120", f"--basetemp={basetemp}"])
    if junit_xml is not None:
        cmd.append(f"--junitxml={junit_xml}")
    for ignore in ignores:
        cmd.append(f"--ignore={ignore}")
    for nodeid in deselects:
        cmd.extend(["--deselect", nodeid])
    if k_expr:
        cmd.extend(["-k", k_expr])
    cmd.extend(extra_args)
    cmd.extend(targets)
    return cmd


def journald_nrestarts() -> int | None:
    if not sys.platform.startswith("linux"):
        return None
    try:
        res = subprocess.run(
            ["systemctl", "show", "systemd-journald", "-p", "NRestarts", "--value"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if res.returncode != 0:
        return None
    raw = res.stdout.strip()
    if not raw.isdigit():
        return None
    return int(raw)


@dataclass(frozen=True)
class HostTimerCheck:
    """Result of inspecting one ThermoUY unit."""

    unit: str
    scope: Literal["user", "system", "not-found", "unavailable"]
    load_state: str | None
    active_state: str | None
    next_elapse_utc: str | None
    unit_id: str | None
    sub_state: str | None
    error: str | None


def _show_host_unit(unit: str, *, user: bool, env: dict[str, str]) -> tuple[dict[str, str], str | None]:
    command = ["systemctl", "--user"] if user else ["systemctl"]
    command.extend(["show", unit, "-p", "Id,LoadState,ActiveState,SubState,NextElapseUSecRealtime"])
    manager = "user" if user else "system"
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5, env=env)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {}, f"systemctl {manager} manager unavailable ({type(exc).__name__}: {exc})"

    properties = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    load_state = properties.get("LoadState")
    if load_state == "not-found":
        return properties, None
    if result.returncode != 0:
        detail = result.stderr.strip() or "no error output"
        return properties, f"systemctl {manager} manager returned {result.returncode}: {detail}"
    if not load_state:
        return properties, f"systemctl {manager} manager did not report LoadState"
    return properties, None


def check_host_interference(quiet_window_minutes: int) -> tuple[list[str], list[HostTimerCheck]]:
    """Return blocking reasons and the inspection result for each ThermoUY unit."""
    if not sys.platform.startswith("linux"):
        return [], []

    # systemctl formats this D-Bus property as a local wall-clock timestamp.
    env = dict(os.environ, LC_ALL="C", TZ="UTC")
    now = dt.datetime.now(dt.UTC)
    interference: list[str] = []
    checked: list[HostTimerCheck] = []
    for unit in HOST_INTERFERENCE_UNITS:
        scope: Literal["user", "system", "not-found", "unavailable"] = "user"
        properties, error = _show_host_unit(unit, user=True, env=env)
        if properties.get("LoadState") == "not-found":
            scope = "system"
            properties, error = _show_host_unit(unit, user=False, env=env)

        load_state = properties.get("LoadState")
        active_state = properties.get("ActiveState")
        sub_state = properties.get("SubState")
        timestamp = properties.get("NextElapseUSecRealtime", "")
        try:
            next_elapse = dt.datetime.strptime(timestamp.partition(" ")[2], "%Y-%m-%d %H:%M:%S UTC").replace(
                tzinfo=dt.UTC
            )
        except ValueError:
            next_elapse = None

        if error is not None:
            scope = "unavailable"
            interference.append(f"{unit}: {error}")
        elif load_state == "not-found":
            scope = "not-found"
        elif unit.endswith(".service") and active_state in {"active", "activating", "deactivating"}:
            interference.append(f"{unit} is {active_state} ({sub_state or 'unknown'})")
        elif unit.endswith(".timer") and next_elapse is not None:
            if 0 <= (next_elapse - now).total_seconds() <= quiet_window_minutes * 60:
                interference.append(f"{unit} is scheduled for {timestamp} within {quiet_window_minutes} minutes")

        checked.append(
            HostTimerCheck(
                unit=unit,
                scope=scope,
                load_state=load_state,
                active_state=active_state,
                next_elapse_utc=next_elapse.isoformat() if next_elapse is not None else None,
                unit_id=properties.get("Id"),
                sub_state=sub_state,
                error=error,
            )
        )
    return interference, checked


def _read_git(repo: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def read_git_commit(repo: Path) -> str | None:
    """Read the checked-out SHA for the run summary when Git is available."""
    sha = _read_git(repo, "rev-parse", "--verify", "HEAD")
    return sha if sha is not None and re.fullmatch(r"[0-9a-f]{40}", sha) else None


def read_git_dirty(repo: Path) -> bool | None:
    """Report whether the index or tracked files differ from HEAD."""
    status = _read_git(repo, "status", "--porcelain", "--untracked-files=no")
    return bool(status) if status is not None else None


_ACTIVE_CHILDREN: dict[int, subprocess.Popen] = {}
_ACTIVE_CHILDREN_LOCK = threading.RLock()  # re-entrant: the signal handler may fire while it is held
_SHUTDOWN = threading.Event()


def _killpg(pgid: int, sig: int) -> None:
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def terminate_active_children(grace_s: float = CHILD_TERM_GRACE_S) -> None:
    """SIGTERM every live pytest process group, then SIGKILL survivors after ``grace_s``.

    Process groups are signalled (not just the leader) so pytest-xdist workers cannot
    outlive the runner and keep tmpfs/RAM pinned after the session closes.
    """
    _SHUTDOWN.set()
    with _ACTIVE_CHILDREN_LOCK:
        children = dict(_ACTIVE_CHILDREN)
    for pgid in children:
        _killpg(pgid, signal.SIGTERM)
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline and any(proc.poll() is None for proc in children.values()):
        time.sleep(0.1)
    for pgid in children:
        _killpg(pgid, signal.SIGKILL)


def _raise_on_shutdown_signal(signum: int, _frame: object) -> None:
    terminate_active_children()
    raise SystemExit(128 + signum)


def install_shutdown_handlers() -> dict[int, object]:
    """Route SIGHUP/SIGTERM/SIGINT to an orderly shutdown so ``finally`` cleanup runs.

    Without this an SSH logout (SIGHUP) kills the runner outright, skipping basetemp
    and SOLO_WINDOW cleanup. Returns the previous handlers (empty off the main thread).
    """
    _SHUTDOWN.clear()
    if threading.current_thread() is not threading.main_thread():
        return {}
    return {sig: signal.signal(sig, _raise_on_shutdown_signal) for sig in SHUTDOWN_SIGNALS}


def restore_shutdown_handlers(previous: dict[int, object]) -> None:
    for sig, handler in previous.items():
        signal.signal(sig, handler)


def _run_subprocess(
    name: str,
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
) -> dict:
    started = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    returncode = 128 + signal.SIGTERM
    with log_path.open("wb") as log_file:
        with _ACTIVE_CHILDREN_LOCK:
            proc = None
            if not _SHUTDOWN.is_set():
                # Own session/process group so shutdown can reap pytest and its xdist workers together.
                proc = subprocess.Popen(
                    cmd,
                    cwd=cwd,
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                _ACTIVE_CHILDREN[proc.pid] = proc
        if proc is not None:
            try:
                returncode = proc.wait()
            except BaseException:
                _killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                raise
            finally:
                with _ACTIVE_CHILDREN_LOCK:
                    _ACTIVE_CHILDREN.pop(proc.pid, None)
    return {
        "name": name,
        "command": cmd,
        "exit_code": returncode,
        "duration_s": round(time.monotonic() - started, 2),
        "log": str(log_path),
    }


def _run_node_frontend(repo: Path, out_dir: Path, env: dict[str, str] | None = None) -> dict:
    tests = sorted(str(p.relative_to(repo)) for p in (repo / "tests" / "frontend").glob("*.cjs"))
    log_path = out_dir / "node.log"
    if not tests:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("No frontend .cjs tests found\n")
        return {"name": "frontend-node", "exit_code": 0, "duration_s": 0.0, "log": str(log_path)}
    return _run_subprocess(
        "frontend-node", ["node", "--test", *tests], cwd=repo, env=env if env is not None else dict(os.environ), log_path=log_path
    )


def _is_valid_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def evaluate_summary(
    *,
    steps: list[dict],
    before_nrestarts: int | None = None,
    after_nrestarts: int | None = None,
    require_nrestarts: bool = False,
) -> tuple[int, bool]:
    """Return ``(overall_exit_code, nrestarts_ok)`` across executed steps."""
    if require_nrestarts or before_nrestarts is not None or after_nrestarts is not None:
        nrestarts_ok = (
            _is_valid_int(before_nrestarts)
            and _is_valid_int(after_nrestarts)
            and after_nrestarts == before_nrestarts
        )
    else:
        nrestarts_ok = True
    overall_ok = nrestarts_ok and all(step.get("exit_code") == 0 for step in steps)
    return (0 if overall_ok else 1), nrestarts_ok


def should_prebuild(tier: int, *, prebuild: bool = False, no_prebuild: bool = False) -> bool:
    """Return whether standard runtime artifacts should be prebuilt before running pytest."""
    if prebuild:
        return True
    if no_prebuild:
        return False
    return tier in (1, 2)


def run_tier2_lanes(
    *,
    repo: Path,
    python_exe: Path,
    env: dict[str, str],
    basetemp_root: Path,
    out_dir: Path,
    workers: int,
    has_xdist: bool,
    run_node: bool,
    solo_workers: int | None = None,
    selected_lanes: tuple[str, ...] = tuple(TIER2_STEP_NAMES),
) -> list[dict]:
    """Run selected Tier 2 steps, with Lane C after the parallel group.

    Lane B stays strictly serial (``use_xdist=False``) to preserve ``daemon-reload``
    ordering and ``before_NRestarts == after_NRestarts`` on ``systemd-journald``.
    Lane C runs only after the selected Node, A and B steps exit, using ``--solo-workers``
    (default 2 when xdist is available and ``workers >= 2``, else 1) with
    ``--dist=load`` so compiled rehearsals retain quiet CPU/disk budgets.
    """
    env_systemd = dict(env, KZA_REQUIRE_REAL_SYSTEMD="1")
    effective_solo_workers = resolve_solo_workers(
        solo_workers,
        workers=workers,
        has_xdist=has_xdist,
    )

    def run_lane_a() -> dict:
        cmd_a = build_pytest_command(
            python_exe,
            targets=["tests"],
            workers=workers,
            basetemp=basetemp_root / "lane-a",
            junit_xml=out_dir / "tier2-core.xml",
            ignores=LINUX_SYSTEMD_MODULES,
            deselects=COMPILED_SOLO_TESTS,
            use_xdist=has_xdist,
        )
        name = TIER2_STEP_NAMES["a"]
        return _run_subprocess(name, cmd_a, cwd=repo, env=env, log_path=out_dir / f"{name}.log")

    def run_lane_b() -> list[dict]:
        lane_steps = []
        sc6b_module = "tests/integration/test_linux_sc6b_release_rehearsal.py"
        linux_modules = [module for module in LINUX_SYSTEMD_MODULES if module != sc6b_module]
        first_phase = "test_01 or test_02 or test_03 or test_04 or test_05 or test_06"
        phase_specs = (
            ("b-linux", linux_modules, "tier2-linux.xml", None),
            ("b-sc6b1", [sc6b_module], "tier2-sc6b1.xml", first_phase),
            ("b-sc6b2", [sc6b_module], "tier2-sc6b2.xml", f"not ({first_phase})"),
        )
        for lane, targets, xml_name, k_expr in phase_specs:
            if lane not in selected_lanes:
                continue
            name = TIER2_STEP_NAMES[lane]
            cmd = build_pytest_command(
                python_exe,
                targets=targets,
                workers=1,
                basetemp=basetemp_root / name,
                junit_xml=out_dir / xml_name,
                k_expr=k_expr,
                use_xdist=False,
            )
            lane_steps.append(
                _run_subprocess(name, cmd, cwd=repo, env=env_systemd, log_path=out_dir / f"{name}.log")
            )
        return lane_steps

    steps: list[dict] = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        fut_node = pool.submit(_run_node_frontend, repo, out_dir, env) if run_node and "node" in selected_lanes else None
        fut_a = pool.submit(run_lane_a) if "a" in selected_lanes else None
        fut_b = pool.submit(run_lane_b) if any(lane in selected_lanes for lane in TIER2_B_LANES) else None
        if fut_node is not None:
            steps.append(fut_node.result())
        if fut_a is not None:
            steps.append(fut_a.result())
        if fut_b is not None:
            steps.extend(fut_b.result())

    if "c" in selected_lanes:
        cmd_c = build_pytest_command(
            python_exe,
            targets=list(COMPILED_SOLO_TESTS),
            workers=effective_solo_workers,
            basetemp=basetemp_root / "lane-c-compiled",
            junit_xml=out_dir / "tier2-compiled-solo.xml",
            use_xdist=has_xdist and effective_solo_workers > 1,
            dist_mode="load",
            extra_args=("-q",),
        )
        name = TIER2_STEP_NAMES["c"]
        steps.append(_run_subprocess(name, cmd_c, cwd=repo, env=env, log_path=out_dir / f"{name}.log"))
    return steps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", type=int, choices=(0, 1, 2), required=True, help="Execution tier: 0=targeted, 1=fast-non-systemd, 2=full-linux-gate")
    parser.add_argument("--workers", type=int, default=default_workers(), help="pytest-xdist worker count for core non-systemd lane")
    parser.add_argument(
        "--solo-workers",
        type=int,
        default=None,
        help="pytest-xdist worker count for Tier 2 lane-c-compiled-solo (default: 2 when xdist is available and --workers >= 2, else 1)",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="Directory for logs, XML reports, and summary.json")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Shared runtime artifact cache directory")
    parser.add_argument("--basetemp", type=Path, default=None, help="Explicit pytest basetemp root")
    parser.add_argument(
        "--allow-tmpfs-basetemp",
        action="store_true",
        help="Permit an explicit --basetemp on tmpfs/ramfs for Tier 1/2 (default: refused; it pins RAM and starves the cgroup)",
    )
    parser.add_argument("--targets", nargs="*", default=[], help="Target test files/dirs for Tier 0")
    prebuild_group = parser.add_mutually_exclusive_group()
    prebuild_group.add_argument("--prebuild", action="store_true", help="Prebuild runtime artifacts cache before running (default for Tiers 1 and 2)")
    prebuild_group.add_argument("--no-prebuild", action="store_true", help="Skip prebuilding runtime artifacts cache (default lazy cache for Tier 0)")
    parser.add_argument(
        "--lanes",
        nargs="+",
        default=["all"],
        metavar="LANE",
        help="Tier 2 steps, separated by commas or spaces: all (default), node, a, b, b-linux, b-sc6b1, b-sc6b2, c",
    )
    parser.add_argument(
        "--quiet-window-minutes",
        type=int,
        default=45,
        help="Look ahead this many minutes for ThermoUY timers before Tier 2 B/C steps (default: 45)",
    )
    parser.add_argument(
        "--ignore-host-timers",
        action="store_true",
        help="Skip the Tier 2 ThermoUY service and timer preflight",
    )
    parser.add_argument("--no-node", action="store_true", help="Skip node --test tests/frontend/*.cjs")
    parser.add_argument("--json", action="store_true", help="Print JSON summary to stdout")
    args = parser.parse_args(argv)

    repo = REPO_ROOT
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.solo_workers is not None and args.solo_workers < 1:
        parser.error("--solo-workers must be >= 1")
    if args.quiet_window_minutes < 0:
        parser.error("--quiet-window-minutes must be >= 0")
    lane_tokens = " ".join(args.lanes).replace(",", " ").split()
    if not lane_tokens:
        parser.error("--lanes requires at least one lane")
    invalid_lanes = set(lane_tokens) - TIER2_STEP_NAMES.keys() - {"all", "b"}
    if invalid_lanes:
        parser.error(f"unknown --lanes token(s): {', '.join(sorted(invalid_lanes))}")
    if args.tier != 2 and lane_tokens != ["all"]:
        parser.error("--lanes selection requires --tier 2")
    if args.tier == 0 and not args.targets:
        parser.error("--tier 0 requires at least one path in --targets")
    if (
        args.basetemp is not None
        and args.tier >= 1
        and not args.allow_tmpfs_basetemp
        and is_ram_backed_path(args.basetemp)
    ):
        parser.error(
            f"--basetemp {args.basetemp} is on tmpfs/ramfs; Tier {args.tier} retains tens of GiB and, with "
            "MemorySwapMax=0, pins RAM until the directory is removed. Use a disk path (or omit --basetemp), "
            "or pass --allow-tmpfs-basetemp to override"
        )

    lanes = set(TIER2_STEP_NAMES if "all" in lane_tokens else lane_tokens)
    if "b" in lanes:
        lanes.update(TIER2_B_LANES)
    if args.no_node:
        lanes.discard("node")
    selected_lanes = tuple(lane for lane in TIER2_STEP_NAMES if lane in lanes)
    if args.tier == 2:
        if not selected_lanes:
            parser.error("--lanes and --no-node leave no Tier 2 steps selected")
        if not sys.platform.startswith("linux"):
            parser.error("--tier 2 (Linux real systemd gate) must run on Linux (kza-yo)")
        selected_steps = [TIER2_STEP_NAMES[lane] for lane in selected_lanes]
    else:
        selected_steps = [] if args.no_node else ["frontend-node"]
        selected_steps.append("tier0-pytest" if args.tier == 0 else "tier1-core")
    git_commit = read_git_commit(repo)
    git_dirty = read_git_dirty(repo)

    run_stamp = time.strftime("%Y%m%d-%H%M%S")
    if args.out_dir is not None:
        out_dir = Path(args.out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        runs_root = Path.home() / ".local" / "state" / "kza-test-runs"
        runs_root.mkdir(parents=True, exist_ok=True)
        out_dir = Path(tempfile.mkdtemp(prefix=f"tier{args.tier}-{run_stamp}-", dir=runs_root)).resolve()

    cache_dir = choose_cache_dir(args.cache_dir)
    python_exe = Path(sys.executable)
    has_xdist = _xdist_available(python_exe)
    if args.workers > 1 and not has_xdist:
        print("WARNING: pytest-xdist not installed; running single-worker", file=sys.stderr)
    solo_workers = resolve_solo_workers(args.solo_workers, workers=args.workers, has_xdist=has_xdist)
    if (args.solo_workers or 0) > 1 and not has_xdist:
        print("WARNING: pytest-xdist not installed; running lane-c-compiled-solo single-worker", file=sys.stderr)

    env = sanitize_env()
    env["KZA_SUITE_ROOT"] = str(repo)
    env["KZA_SUITE_OUT"] = str(out_dir)

    t_total_start = time.monotonic()
    prebuild_s = 0.0
    basetemp_root: Path | None = None
    cleanup_basetemp = False
    solo_guard = SoloWindowGuard()
    before_nr: int | None = None
    after_nr: int | None = None
    steps: list[dict] = []
    host_interference: list[str] = []
    host_timers_checked: list[HostTimerCheck] = []

    previous_handlers = install_shutdown_handlers()
    try:
        solo_guard = acquire_solo_window(tier=args.tier)
        if args.tier == 2 and not args.ignore_host_timers and lanes.intersection((*TIER2_B_LANES, "c")):
            host_interference, host_timers_checked = check_host_interference(args.quiet_window_minutes)
        if host_interference:
            print(
                "Tier 2 aborted before pytest by the ThermoUY host preflight: " + ", ".join(host_interference),
                file=sys.stderr,
            )
        else:
            basetemp_root, cleanup_basetemp = choose_basetemp_root(args.basetemp, out_dir, tier=args.tier)
            before_nr = journald_nrestarts() if args.tier == 2 else None

            if should_prebuild(args.tier, prebuild=args.prebuild, no_prebuild=args.no_prebuild):
                if str(repo) not in sys.path:
                    sys.path.insert(0, str(repo))
                from tests.mocks.runtime_artifact_cache import prebuild_standard_artifacts

                t_pre = time.monotonic()
                cache_env = prebuild_standard_artifacts(repo, cache_dir)
                prebuild_s = round(time.monotonic() - t_pre, 2)
                env.update(cache_env)
            else:
                env["KZA_RUNTIME_ARTIFACT_CACHE_DIR"] = str(cache_dir)

            if args.tier == 0:
                if not args.no_node:
                    steps.append(_run_node_frontend(repo, out_dir, env))
                cmd = build_pytest_command(
                    python_exe,
                    targets=list(args.targets),
                    workers=args.workers,
                    basetemp=basetemp_root / "tier0",
                    junit_xml=out_dir / "tier0.xml",
                    use_xdist=has_xdist,
                )
                steps.append(_run_subprocess("tier0-pytest", cmd, cwd=repo, env=env, log_path=out_dir / "tier0.log"))

            elif args.tier == 1:
                if not args.no_node:
                    steps.append(_run_node_frontend(repo, out_dir, env))
                cmd = build_pytest_command(
                    python_exe,
                    targets=["tests"],
                    workers=args.workers,
                    basetemp=basetemp_root / "tier1",
                    junit_xml=out_dir / "tier1.xml",
                    ignores=LINUX_SYSTEMD_MODULES,
                    deselects=COMPILED_SOLO_TESTS,
                    use_xdist=has_xdist,
                )
                steps.append(_run_subprocess("tier1-core", cmd, cwd=repo, env=env, log_path=out_dir / "tier1-core.log"))

            else:
                steps.extend(
                    run_tier2_lanes(
                        repo=repo,
                        python_exe=python_exe,
                        env=env,
                        basetemp_root=basetemp_root,
                        out_dir=out_dir,
                        workers=args.workers,
                        has_xdist=has_xdist,
                        run_node=not args.no_node,
                        selected_lanes=selected_lanes,
                        solo_workers=solo_workers,
                    )
                )
                after_nr = journald_nrestarts()
    finally:
        try:
            if cleanup_basetemp and basetemp_root is not None:
                safe_rmtree(basetemp_root)
        finally:
            try:
                solo_guard.release()
            finally:
                restore_shutdown_handlers(previous_handlers)

    overall_exit, nrestarts_ok = evaluate_summary(
        steps=steps,
        before_nrestarts=before_nr,
        after_nrestarts=after_nr,
        require_nrestarts=(args.tier == 2),
    )
    if host_interference:
        overall_exit = 2
    summary = {
        "tier": args.tier,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "selected_lanes": selected_steps,
        "ignore_host_timers": args.ignore_host_timers,
        "host_timers_checked": [asdict(check) for check in host_timers_checked],
        "host_interference": host_interference,
        "platform": sys.platform,
        "workers": args.workers if has_xdist else 1,
        "solo_workers": solo_workers,
        "xdist_enabled": has_xdist and args.workers > 1,
        "prebuild_s": prebuild_s,
        "total_wall_s": round(time.monotonic() - t_total_start, 2),
        "basetemp_root": str(basetemp_root),
        "cache_dir": str(cache_dir),
        "before_NRestarts": before_nr,
        "after_NRestarts": after_nr,
        "nrestarts_ok": nrestarts_ok,
        "exit_code": overall_exit,
        "steps": steps,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"[Tier {args.tier}] wall={summary['total_wall_s']}s prebuild={prebuild_s}s exit={overall_exit} out={out_dir}")
        for st in steps:
            print(f"  - {st['name']}: exit={st['exit_code']} duration={st['duration_s']}s")
    return overall_exit


if __name__ == "__main__":
    raise SystemExit(main())
