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
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
MIN_SHM_FREE_BYTES = 16 * 1024**3  # 16 GiB safety margin on /dev/shm
DEFAULT_SOLO_WINDOW_FILE = Path("/home/yo/projects/runs/SOLO_WINDOW")
INHERITED_OVERRIDE_VARS: tuple[str, ...] = (
    "PYTEST_ADDOPTS",
    "KZA_SC6B_MEASURE",
    "KZA_REQUIRE_REAL_SYSTEMD",
)

# Five Linux/systemd integration modules reserved for Carril B on kza-yo.
LINUX_SYSTEMD_MODULES: tuple[str, ...] = (
    "tests/integration/test_linux_compaction_units.py",
    "tests/integration/test_linux_owner_compaction.py",
    "tests/integration/test_linux_adoption_window.py",
    "tests/integration/test_linux_managed_launcher.py",
    "tests/integration/test_linux_sc6b_release_rehearsal.py",
)

# Exact seven timing-budget compiled rehearsals run solo after main parallel load.
COMPILED_SOLO_TESTS: tuple[str, ...] = (
    "tests/integration/test_copy_observation.py::test_compiled_existing_copy_observations_preserve_current_authority_without_grants",
    "tests/integration/test_copy_collector_cleanup.py::test_current_authority_cleans_mixed_collector_copies_and_drains_cancelled_native",
    "tests/integration/test_compiled_material_reconciliation.py::test_compiled_material_rollback_and_forward_preserve_latest_authority",
    "tests/integration/test_checkpoint_blob_cleanup.py::test_compiled_checkpoint_cleanup_unlinks_only_withdrawn_blob_and_preserves_historical_pins",
    "tests/integration/test_checkpoint_binary_lineage.py::test_compiled_binary_source_lineage_distinguishes_withdrawn_shared_and_manual",
    "tests/integration/test_compiled_output_registration_compatibility.py::test_compiled_v1_v2_histories_share_original_closed_controller",
    "tests/integration/test_compiled_slot_rollback.py::test_compiled_slot_rollback_preserves_latest_privacy_authority",
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
    return max(1, min(12, cpus // 2 if cpus >= 16 else cpus))


def _shm_has_capacity(shm_root: Path = Path("/dev/shm"), min_shm_bytes: int = MIN_SHM_FREE_BYTES) -> bool:
    if not shm_root.is_dir():
        return False
    try:
        statvfs = os.statvfs(shm_root)
        return (statvfs.f_bavail * statvfs.f_frsize) >= min_shm_bytes
    except OSError:
        return False


def choose_basetemp_root(
    explicit: Path | None,
    out_dir: Path,
    *,
    platform: str | None = None,
    shm_root: Path = Path("/dev/shm"),
    min_shm_bytes: int = MIN_SHM_FREE_BYTES,
) -> tuple[Path, bool]:
    """Select basetemp root; prefer /dev/shm on Linux when enough RAM is free.

    Returns ``(root_path, cleanup_on_exit)``.
    """
    if explicit is not None:
        resolved = Path(explicit).resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved, False

    plat = platform or sys.platform
    if plat.startswith("linux") and _shm_has_capacity(shm_root, min_shm_bytes):
        shm_tmp = shm_root / f"kza-pytest-{os.getpid()}-{int(time.time())}"
        shm_tmp.mkdir(parents=True, exist_ok=True)
        return shm_tmp, True

    fallback = (out_dir / "basetemp").resolve()
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback, False


def choose_cache_dir(
    explicit: Path | None,
    *,
    platform: str | None = None,
    shm_root: Path = Path("/dev/shm"),
    min_shm_bytes: int = MIN_SHM_FREE_BYTES,
    home_dir: Path | None = None,
) -> Path:
    """Select shared artifact cache directory using the same /dev/shm capacity guard."""
    if explicit is not None:
        return Path(explicit).resolve()
    plat = platform or sys.platform
    if plat.startswith("linux") and _shm_has_capacity(shm_root, min_shm_bytes):
        return (shm_root / "kza-artifact-cache").resolve()
    base_home = home_dir or Path.home()
    return (base_home / ".cache" / "kza-runtime-artifacts").resolve()


def sanitize_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Build deterministic runner env stripping inherited pytest/gate overrides."""
    cleaned = dict(base_env if base_env is not None else os.environ)
    for var in INHERITED_OVERRIDE_VARS:
        cleaned.pop(var, None)
    cleaned.update(DEFAULT_ENV)
    return cleaned


def acquire_solo_window(
    solo_file: Path,
    *,
    tier: int,
    platform: str | None = None,
) -> Path | None:
    """Enforce SOLO_WINDOW host exclusion on Linux (fail-closed if already held)."""
    plat = platform or sys.platform
    if not plat.startswith("linux"):
        return None
    if solo_file.exists() or solo_file.is_symlink():
        raise RuntimeError(f"SOLO_WINDOW active at {solo_file}; refusing concurrent Tier {tier} run")
    if tier != 2:
        return None
    if solo_file != DEFAULT_SOLO_WINDOW_FILE:
        solo_file.parent.mkdir(parents=True, exist_ok=True)
    elif not solo_file.parent.is_dir():
        return None
    try:
        fd = os.open(solo_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as exc:
        raise RuntimeError(f"SOLO_WINDOW already active at {solo_file}; refusing concurrent Tier 2 run") from exc
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(f"run_test_suite tier=2 pid={os.getpid()} started={dt.datetime.now(dt.timezone.utc).isoformat()}\n")
    return solo_file


def safe_rmtree(path: Path) -> None:
    """Remove directory trees containing read-only (0555/0444) slot trees without following symlinks."""
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink():
        path.unlink()
        return
    for _root, _dirs, files, fd in os.fwalk(path, follow_symlinks=False):
        mode = stat.S_IMODE(os.fstat(fd).st_mode)
        if (mode & stat.S_IRWXU) != stat.S_IRWXU:
            os.fchmod(fd, mode | stat.S_IRWXU)
        for name in files:
            st = os.lstat(name, dir_fd=fd)
            if not stat.S_ISLNK(st.st_mode):
                fmode = stat.S_IMODE(st.st_mode)
                if (fmode & stat.S_IWUSR) == 0:
                    try:
                        os.chmod(name, fmode | stat.S_IWUSR, dir_fd=fd)
                    except OSError:
                        pass
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
    extra_args: tuple[str, ...] = (),
) -> list[str]:
    cmd = [str(python_exe), "-m", "pytest", "-p", "no:cacheprovider"]
    if use_xdist and workers > 1:
        cmd.extend(["-n", str(workers), "--dist=worksteal"])
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
    res = subprocess.run(
        ["systemctl", "show", "systemd-journald", "-p", "NRestarts", "--value"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if res.returncode != 0:
        return None
    raw = res.stdout.strip()
    if not raw.isdigit():
        return None
    return int(raw)


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
    with log_path.open("wb") as log_file:
        proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    return {
        "name": name,
        "command": cmd,
        "exit_code": proc.returncode,
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
) -> list[dict]:
    """Run Node, Lane A (xdist core) and Lane B (serial real-systemd) concurrently, then Lane C strictly solo."""
    env_systemd = dict(env, KZA_REQUIRE_REAL_SYSTEMD="1")

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
        return _run_subprocess("lane-a-core", cmd_a, cwd=repo, env=env, log_path=out_dir / "lane-a-core.log")

    def run_lane_b() -> list[dict]:
        lane_steps = []
        linux_non_sc6b = [m for m in LINUX_SYSTEMD_MODULES if not m.endswith("test_linux_sc6b_release_rehearsal.py")]
        cmd_b0 = build_pytest_command(
            python_exe,
            targets=linux_non_sc6b,
            workers=1,
            basetemp=basetemp_root / "lane-b-linux",
            junit_xml=out_dir / "tier2-linux.xml",
            use_xdist=False,
        )
        lane_steps.append(_run_subprocess("lane-b-linux", cmd_b0, cwd=repo, env=env_systemd, log_path=out_dir / "lane-b-linux.log"))

        sc6b_mod = ["tests/integration/test_linux_sc6b_release_rehearsal.py"]
        k1 = "test_01 or test_02 or test_03 or test_04 or test_05 or test_06"
        cmd_b1 = build_pytest_command(
            python_exe,
            targets=sc6b_mod,
            workers=1,
            basetemp=basetemp_root / "lane-b-sc6b1",
            junit_xml=out_dir / "tier2-sc6b1.xml",
            k_expr=k1,
            use_xdist=False,
        )
        lane_steps.append(_run_subprocess("lane-b-sc6b1", cmd_b1, cwd=repo, env=env_systemd, log_path=out_dir / "lane-b-sc6b1.log"))

        k2 = f"not ({k1})"
        cmd_b2 = build_pytest_command(
            python_exe,
            targets=sc6b_mod,
            workers=1,
            basetemp=basetemp_root / "lane-b-sc6b2",
            junit_xml=out_dir / "tier2-sc6b2.xml",
            k_expr=k2,
            use_xdist=False,
        )
        lane_steps.append(_run_subprocess("lane-b-sc6b2", cmd_b2, cwd=repo, env=env_systemd, log_path=out_dir / "lane-b-sc6b2.log"))
        return lane_steps

    steps: list[dict] = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        fut_node = pool.submit(_run_node_frontend, repo, out_dir, env) if run_node else None
        fut_a = pool.submit(run_lane_a)
        fut_b = pool.submit(run_lane_b)
        if fut_node is not None:
            steps.append(fut_node.result())
        res_a = fut_a.result()
        res_b_list = fut_b.result()
    steps.append(res_a)
    steps.extend(res_b_list)

    # Start Lane C only after BOTH Lane A and Lane B have exited so CPU/I/O are completely quiet.
    cmd_c = build_pytest_command(
        python_exe,
        targets=list(COMPILED_SOLO_TESTS),
        workers=1,
        basetemp=basetemp_root / "lane-c-compiled",
        junit_xml=out_dir / "tier2-compiled-solo.xml",
        use_xdist=False,
        extra_args=("-q",),
    )
    res_c = _run_subprocess("lane-c-compiled-solo", cmd_c, cwd=repo, env=env, log_path=out_dir / "lane-c-compiled-solo.log")
    steps.append(res_c)
    return steps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", type=int, choices=(0, 1, 2), required=True, help="Execution tier: 0=targeted, 1=fast-non-systemd, 2=full-linux-gate")
    parser.add_argument("--workers", type=int, default=default_workers(), help="pytest-xdist worker count for core non-systemd lane")
    parser.add_argument("--out-dir", type=Path, default=None, help="Directory for logs, XML reports, and summary.json")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Shared runtime artifact cache directory")
    parser.add_argument("--basetemp", type=Path, default=None, help="Explicit pytest basetemp root")
    parser.add_argument("--solo-window-file", type=Path, default=DEFAULT_SOLO_WINDOW_FILE, help="Host SOLO_WINDOW exclusion lock file path on Linux")
    parser.add_argument("--targets", nargs="*", default=[], help="Target test files/dirs for Tier 0")
    prebuild_group = parser.add_mutually_exclusive_group()
    prebuild_group.add_argument("--prebuild", action="store_true", help="Prebuild runtime artifacts cache before running (default for Tiers 1 and 2)")
    prebuild_group.add_argument("--no-prebuild", action="store_true", help="Skip prebuilding runtime artifacts cache (default lazy cache for Tier 0)")
    parser.add_argument("--no-node", action="store_true", help="Skip node --test tests/frontend/*.cjs")
    parser.add_argument("--json", action="store_true", help="Print JSON summary to stdout")
    args = parser.parse_args(argv)

    repo = REPO_ROOT
    if args.tier == 0 and not args.targets:
        parser.error("--tier 0 requires at least one path in --targets")
    if args.tier == 2 and not sys.platform.startswith("linux"):
        parser.error("--tier 2 (Linux real systemd gate) must run on Linux (kza-yo)")

    run_stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = (args.out_dir or (Path.home() / ".local" / "state" / "kza-test-runs" / f"tier{args.tier}-{run_stamp}")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = choose_cache_dir(args.cache_dir)
    python_exe = Path(sys.executable)
    has_xdist = _xdist_available(python_exe)
    if args.workers > 1 and not has_xdist:
        print("WARNING: pytest-xdist not installed; running single-worker", file=sys.stderr)

    env = sanitize_env()
    env["KZA_SUITE_ROOT"] = str(repo)
    env["KZA_SUITE_OUT"] = str(out_dir)

    t_total_start = time.monotonic()
    prebuild_s = 0.0
    basetemp_root: Path | None = None
    cleanup_basetemp = False
    acquired_solo: Path | None = None
    before_nr: int | None = None
    steps: list[dict] = []

    try:
        acquired_solo = acquire_solo_window(args.solo_window_file, tier=args.tier)
        basetemp_root, cleanup_basetemp = choose_basetemp_root(args.basetemp, out_dir)
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
                )
            )
    finally:
        if cleanup_basetemp and basetemp_root is not None:
            safe_rmtree(basetemp_root)
        if acquired_solo is not None:
            acquired_solo.unlink(missing_ok=True)

    after_nr = journald_nrestarts() if args.tier == 2 else None
    overall_exit, nrestarts_ok = evaluate_summary(
        steps=steps,
        before_nrestarts=before_nr,
        after_nrestarts=after_nr,
        require_nrestarts=(args.tier == 2),
    )
    summary = {
        "tier": args.tier,
        "platform": sys.platform,
        "workers": args.workers if has_xdist else 1,
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
