"""Tests for the portable tier runner and the CI test workflow."""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import shlex
import stat
import subprocess
import time
from types import SimpleNamespace

import pytest
import yaml

from scripts import run_test_suite as runner


def test_default_workers_caps_cleanly():
    assert runner.default_workers(platform="darwin", cpu_count=12) == 8
    assert runner.default_workers(platform="linux", cpu_count=48) == 8
    assert runner.default_workers(platform="linux", cpu_count=4) == 4


def test_build_pytest_command_tier1_excludes_systemd_and_compiled_solo(tmp_path):
    cmd = runner.build_pytest_command(
        Path("/venv/bin/python"),
        targets=["tests"],
        workers=8,
        basetemp=tmp_path / "bt",
        junit_xml=tmp_path / "tier1.xml",
        ignores=runner.LINUX_SYSTEMD_MODULES,
        deselects=runner.COMPILED_SOLO_TESTS,
        use_xdist=True,
    )
    assert cmd[:6] == ["/venv/bin/python", "-m", "pytest", "-p", "no:cacheprovider", "-n"]
    assert "8" in cmd and "--dist=worksteal" in cmd
    for mod in runner.LINUX_SYSTEMD_MODULES:
        assert f"--ignore={mod}" in cmd
    assert cmd.count("--deselect") == len(runner.COMPILED_SOLO_TESTS)


def test_choose_basetemp_root_prefers_shm_on_linux_when_free(tmp_path, monkeypatch):
    shm = tmp_path / "dev_shm"
    shm.mkdir()
    monkeypatch.setattr(runner.os, "statvfs", lambda _p: SimpleNamespace(f_bavail=32 * 1024**3, f_frsize=1))

    bt, should_clean = runner.choose_basetemp_root(
        None, tmp_path / "out", tier=0, platform="linux", shm_root=shm
    )
    assert bt.parent == shm and bt.exists() and should_clean is True

    for full_tier in (1, 2):
        expected_parent = (tmp_path / f"out-{full_tier}").resolve()
        bt_tier, clean_tier = runner.choose_basetemp_root(
            None, expected_parent, tier=full_tier, platform="linux", shm_root=shm
        )
        assert bt_tier.parent == expected_parent and bt_tier.name.startswith("basetemp-")
        assert bt_tier.exists() and clean_tier is True

    monkeypatch.setattr(runner.os, "statvfs", lambda _p: SimpleNamespace(f_bavail=1 * 1024**3, f_frsize=1))
    low_parent = (tmp_path / "out-low").resolve()
    bt_fallback, should_clean_fb = runner.choose_basetemp_root(
        None, low_parent, tier=0, platform="linux", shm_root=shm
    )
    assert bt_fallback.parent == low_parent and bt_fallback.name.startswith("basetemp-") and should_clean_fb is True

    explicit_bt, clean_explicit = runner.choose_basetemp_root(
        tmp_path / "explicit-bt", tmp_path / "out-explicit", tier=2, platform="linux", shm_root=shm
    )
    assert explicit_bt == (tmp_path / "explicit-bt").resolve() and clean_explicit is False


def test_choose_basetemp_root_never_adopts_preexisting_or_symlinked_basetemp(tmp_path, monkeypatch):
    shm = tmp_path / "dev_shm"
    shm.mkdir()
    monkeypatch.setattr(runner.os, "statvfs", lambda _p: SimpleNamespace(f_bavail=32 * 1024**3, f_frsize=1))

    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "keep.txt"
    sentinel.write_text("keep")

    out_dir = tmp_path / "shared-out"
    out_dir.mkdir()
    os.symlink(outside, out_dir / "basetemp")

    bt1, clean1 = runner.choose_basetemp_root(None, out_dir, tier=2, platform="linux", shm_root=shm)
    bt2, clean2 = runner.choose_basetemp_root(None, out_dir, tier=2, platform="linux", shm_root=shm)
    assert clean1 is True and clean2 is True
    assert bt1 != bt2 and bt1.parent == out_dir.resolve() and not bt1.is_symlink()

    runner.safe_rmtree(bt1)
    runner.safe_rmtree(bt2)
    assert sentinel.read_text() == "keep"
    assert (out_dir / "basetemp").is_symlink()


def test_safe_rmtree_removes_readonly_slot_directories(tmp_path):
    tree = tmp_path / "basetemp" / "slot" / "runtime"
    tree.mkdir(parents=True)
    f = tree / "release.json"
    f.write_text("{}")
    os.chmod(f, 0o444)
    os.chmod(tree, 0o555)
    os.chmod(tree.parent, 0o555)

    runner.safe_rmtree(tmp_path / "basetemp")
    assert not (tmp_path / "basetemp").exists()


def test_evaluate_summary_requires_zero_exit_and_stable_journald():
    assert runner.evaluate_summary(
        steps=[{"name": "a", "exit_code": 0}],
        before_nrestarts=111,
        after_nrestarts=111,
    ) == (0, True)

    assert runner.evaluate_summary(
        steps=[{"name": "a", "exit_code": 0}],
        before_nrestarts=111,
        after_nrestarts=112,
    ) == (1, False)

    assert runner.evaluate_summary(
        steps=[{"name": "a", "exit_code": 1}],
        before_nrestarts=111,
        after_nrestarts=111,
    ) == (1, True)


def test_should_prebuild_defaults_lazy_on_tier0_and_enabled_on_tier1_tier2():
    assert runner.should_prebuild(0) is False
    assert runner.should_prebuild(0, prebuild=True) is True
    assert runner.should_prebuild(1) is True
    assert runner.should_prebuild(2) is True
    assert runner.should_prebuild(1, no_prebuild=True) is False
    assert runner.should_prebuild(2, no_prebuild=True) is False


def test_evaluate_summary_tier2_rejects_missing_or_partial_nrestarts():
    for pair in ((None, None), (4, None), (None, 4), (111, 112), (True, True)):
        assert runner.evaluate_summary(
            steps=[{"name": "lane-a", "exit_code": 0}],
            before_nrestarts=pair[0],
            after_nrestarts=pair[1],
            require_nrestarts=True,
        ) == (1, False)
    assert runner.evaluate_summary(
        steps=[{"name": "lane-a", "exit_code": 0}],
        before_nrestarts=111,
        after_nrestarts=111,
        require_nrestarts=True,
    ) == (0, True)


def test_choose_cache_dir_follows_shm_capacity(tmp_path, monkeypatch):
    shm = tmp_path / "dev_shm"
    shm.mkdir()
    monkeypatch.setattr(runner.os, "statvfs", lambda _p: SimpleNamespace(f_bavail=32 * 1024**3, f_frsize=1))
    expected_disk = (tmp_path / ".cache" / "kza-runtime-artifacts").resolve()
    assert runner.choose_cache_dir(None, platform="linux", shm_root=shm, home_dir=tmp_path) == expected_disk
    explicit = tmp_path / "custom-cache"
    assert runner.choose_cache_dir(explicit, platform="linux", shm_root=shm, home_dir=tmp_path) == explicit.resolve()


def test_sanitize_env_strips_inherited_overrides():
    cleaned = runner.sanitize_env(
        {
            "PATH": "/usr/bin",
            "PYTEST_ADDOPTS": "-n 4 -k bad",
            "KZA_SC6B_MEASURE": "1",
            "KZA_REQUIRE_REAL_SYSTEMD": "1",
            "KZA_LAUNCHER_ARTIFACTS_CACHE": "/tmp/fake",
            "KZA_COMPACTION_ARTIFACT": "/tmp/x5.tar.gz",
            "KZA_COMPACTION_ARTIFACT_EPOCH4": "/tmp/x4.tar.gz",
        }
    )
    for var in runner.INHERITED_OVERRIDE_VARS:
        assert var not in cleaned
    assert cleaned["PYTHONFAULTHANDLER"] == "1"


def test_acquire_solo_window_blocks_concurrent_runs_and_releases(tmp_path, monkeypatch):
    import pytest

    solo = tmp_path / "runs" / "SOLO_WINDOW"
    solo.parent.mkdir(parents=True)
    monkeypatch.setattr(runner, "DEFAULT_SOLO_WINDOW_FILE", solo)

    # Alternate path arguments / kwargs must be rejected at API level too.
    with pytest.raises(TypeError):
        runner.acquire_solo_window(tmp_path / "other" / "SOLO_WINDOW", tier=2, platform="linux")  # type: ignore[misc]
    with pytest.raises(TypeError):
        runner.acquire_solo_window(solo_file=tmp_path / "other" / "SOLO_WINDOW", tier=2, platform="linux")  # type: ignore[call-arg]

    guard = runner.acquire_solo_window(tier=2, platform="linux")
    assert guard.marker_path == solo and solo.is_file()
    try:
        for tier in (0, 1, 2):
            try:
                runner.acquire_solo_window(tier=tier, platform="linux")
                raise AssertionError("expected RuntimeError when SOLO_WINDOW is held")
            except RuntimeError as exc:
                assert "lock" in str(exc).lower() or "solo_window" in str(exc).lower()
    finally:
        guard.release()
    assert not solo.exists()

    # An active Tier 0/1 holding LOCK_SH must block Tier 2 from starting mid-run.
    tier1_guard = runner.acquire_solo_window(tier=1, platform="linux")
    assert tier1_guard.marker_path is None and tier1_guard.lock_fh is not None
    try:
        try:
            runner.acquire_solo_window(tier=2, platform="linux")
            raise AssertionError("expected Tier 2 to fail closed while Tier 1 holds LOCK_SH")
        except RuntimeError as exc:
            assert "lock" in str(exc).lower()
    finally:
        tier1_guard.release()

    # Missing host directory must fail closed for ALL tiers (0, 1, and 2) on Linux.
    missing_default = tmp_path / "missing" / "runs" / "SOLO_WINDOW"
    monkeypatch.setattr(runner, "DEFAULT_SOLO_WINDOW_FILE", missing_default)
    for tier in (0, 1, 2):
        try:
            runner.acquire_solo_window(tier=tier, platform="linux")
            raise AssertionError(f"expected Tier {tier} to fail closed when host dir is missing")
        except RuntimeError as exc:
            assert "host exclusion directory" in str(exc).lower()


def test_acquire_solo_window_cleans_marker_and_lock_if_write_fails(tmp_path, monkeypatch):
    solo = tmp_path / "runs" / "SOLO_WINDOW"
    solo.parent.mkdir(parents=True)
    monkeypatch.setattr(runner, "DEFAULT_SOLO_WINDOW_FILE", solo)

    class _FailWriter:
        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def write(self, _text):
            raise OSError(28, "No space left on device")

        def flush(self):
            pass

    real_fdopen = os.fdopen

    def fail_fdopen(fd, *args, **kwargs):
        os.close(fd)
        return _FailWriter()

    monkeypatch.setattr(runner.os, "fdopen", fail_fdopen)
    try:
        runner.acquire_solo_window(tier=2, platform="linux")
        raise AssertionError("expected OSError")
    except OSError:
        pass
    monkeypatch.setattr(runner.os, "fdopen", real_fdopen)
    assert not solo.exists()

    # Subsequent Tier 2 acquisition must still succeed cleanly.
    guard = runner.acquire_solo_window(tier=2, platform="linux")
    try:
        assert solo.exists()
    finally:
        guard.release()


def test_run_tier2_lanes_waits_for_a_and_b_and_runs_exact_compiled_nodeids(tmp_path, monkeypatch):
    timeline: list[tuple[str, str]] = []
    recorded_cmds: dict[str, list[str]] = {}
    recorded_envs: dict[str, dict[str, str]] = {}

    def fake_run(name: str, cmd: list[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> dict:
        timeline.append(("start", name))
        if name == "lane-b-sc6b2":
            time.sleep(0.05)
        recorded_cmds[name] = list(cmd)
        recorded_envs[name] = dict(env)
        timeline.append(("end", name))
        return {"name": name, "exit_code": 0, "duration_s": 0.01, "log": str(log_path)}

    monkeypatch.setattr(runner, "_run_subprocess", fake_run)
    env = runner.sanitize_env({"PYTEST_ADDOPTS": "-n 9", "KZA_SC6B_MEASURE": "1"})
    steps = runner.run_tier2_lanes(
        repo=tmp_path,
        python_exe=Path("/venv/bin/python"),
        env=env,
        basetemp_root=tmp_path / "bt",
        out_dir=tmp_path / "out",
        workers=8,
        has_xdist=True,
        run_node=True,
    )
    assert [s["name"] for s in steps][-1] == "lane-c-compiled-solo"
    end_a = timeline.index(("end", "lane-a-core"))
    end_b2 = timeline.index(("end", "lane-b-sc6b2"))
    start_c = timeline.index(("start", "lane-c-compiled-solo"))
    assert start_c > end_a and start_c > end_b2

    # Lane B stays strictly serial (no xdist) to protect daemon-reload and systemd-journald NRestarts.
    for lane_b_step in ("lane-b-linux", "lane-b-sc6b1", "lane-b-sc6b2"):
        assert "-n" not in recorded_cmds[lane_b_step]

    cmd_c = recorded_cmds["lane-c-compiled-solo"]
    p6_restore = "tests/integration/test_p6_bytecode_restore_reproducibility.py"
    assert p6_restore in cmd_c
    cmd_a = recorded_cmds["lane-a-core"]
    assert any(option == "--deselect" and target == p6_restore for option, target in zip(cmd_a, cmd_a[1:]))
    compiled_targets = [target for target in cmd_c if target.startswith("tests/integration/")]
    assert compiled_targets == list(runner.COMPILED_SOLO_TESTS)
    assert len(compiled_targets) == 9
    n_idx = cmd_c.index("-n")
    assert cmd_c[n_idx + 1] == "2"
    assert "--dist=load" in cmd_c
    assert "KZA_REQUIRE_REAL_SYSTEMD" not in recorded_envs["lane-a-core"]
    assert "KZA_REQUIRE_REAL_SYSTEMD" not in recorded_envs["lane-c-compiled-solo"]
    assert recorded_envs["lane-b-linux"]["KZA_REQUIRE_REAL_SYSTEMD"] == "1"


def test_resolve_solo_workers_defaults_and_overrides():
    assert runner.resolve_solo_workers(None, workers=8, has_xdist=True) == 2
    assert runner.resolve_solo_workers(None, workers=2, has_xdist=True) == 2
    assert runner.resolve_solo_workers(None, workers=1, has_xdist=True) == 1
    assert runner.resolve_solo_workers(None, workers=8, has_xdist=False) == 1
    assert runner.resolve_solo_workers(1, workers=8, has_xdist=True) == 1
    assert runner.resolve_solo_workers(3, workers=8, has_xdist=True) == 3
    assert runner.resolve_solo_workers(3, workers=8, has_xdist=False) == 1
    with pytest.raises(ValueError, match="solo-workers"):
        runner.resolve_solo_workers(0, workers=8, has_xdist=True)


def test_build_pytest_command_supports_load_dist_mode(tmp_path):
    cmd = runner.build_pytest_command(
        Path("/venv/bin/python"),
        targets=list(runner.COMPILED_SOLO_TESTS),
        workers=2,
        basetemp=tmp_path / "bt",
        use_xdist=True,
        dist_mode="load",
    )
    n_idx = cmd.index("-n")
    assert cmd[n_idx + 1] == "2"
    assert "--dist=load" in cmd
    assert "--dist=worksteal" not in cmd


@pytest.mark.parametrize(
    ("workers", "has_xdist", "solo_workers", "expected_n"),
    [
        (8, True, 1, None),
        (1, True, None, None),
        (8, False, None, None),
        (8, True, 3, "3"),
    ],
)
def test_run_tier2_lanes_respects_solo_workers_and_keeps_lane_b_serial(
    tmp_path, monkeypatch, workers, has_xdist, solo_workers, expected_n
):
    recorded_cmds: dict[str, list[str]] = {}

    def fake_run(name: str, cmd: list[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> dict:
        recorded_cmds[name] = list(cmd)
        return {"name": name, "exit_code": 0, "duration_s": 0.01, "log": str(log_path)}

    monkeypatch.setattr(runner, "_run_subprocess", fake_run)
    runner.run_tier2_lanes(
        repo=tmp_path,
        python_exe=Path("/venv/bin/python"),
        env=runner.sanitize_env({}),
        basetemp_root=tmp_path / "bt",
        out_dir=tmp_path / "out",
        workers=workers,
        has_xdist=has_xdist,
        run_node=False,
        solo_workers=solo_workers,
    )
    for lane_b_step in ("lane-b-linux", "lane-b-sc6b1", "lane-b-sc6b2"):
        assert "-n" not in recorded_cmds[lane_b_step]
    cmd_c = recorded_cmds["lane-c-compiled-solo"]
    if expected_n is None:
        assert "-n" not in cmd_c
        assert "--dist=load" not in cmd_c
    else:
        n_idx = cmd_c.index("-n")
        assert cmd_c[n_idx + 1] == expected_n
        assert "--dist=load" in cmd_c


def test_cli_forwards_solo_workers_and_rejects_non_positive(tmp_path, monkeypatch):
    solo = tmp_path / "runs" / "SOLO_WINDOW"
    solo.parent.mkdir(parents=True)
    out_dir = tmp_path / "out"
    seen_kwargs: list[dict] = []

    monkeypatch.setattr(runner, "DEFAULT_SOLO_WINDOW_FILE", solo)
    monkeypatch.setattr(runner.sys, "platform", "linux")
    monkeypatch.setattr(runner, "_xdist_available", lambda _p: True)
    monkeypatch.setattr(runner, "journald_nrestarts", lambda: 42)
    monkeypatch.setattr(
        runner,
        "run_tier2_lanes",
        lambda **kw: seen_kwargs.append(kw) or [{"name": "lane-c-compiled-solo", "exit_code": 0, "duration_s": 0.01, "log": "x"}],
    )

    rc = runner.main(
        [
            "--tier",
            "2",
            "--workers",
            "6",
            "--solo-workers",
            "1",
            "--no-prebuild",
            "--no-node",
            "--ignore-host-timers",
            "--out-dir",
            str(out_dir),
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )
    assert rc == 0
    assert seen_kwargs[0]["solo_workers"] == 1
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["solo_workers"] == 1

    with pytest.raises(SystemExit):
        runner.main(["--tier", "0", "--targets", "tests/a.py", "--solo-workers", "0"])


@pytest.fixture
def tier2_cli(tmp_path, monkeypatch):
    solo = tmp_path / "runs" / "SOLO_WINDOW"
    solo.parent.mkdir()
    out_dir = tmp_path / "out"

    def read_nrestarts():
        assert solo.exists()
        return 42

    monkeypatch.setattr(runner, "DEFAULT_SOLO_WINDOW_FILE", solo)
    monkeypatch.setattr(runner.sys, "platform", "linux")
    monkeypatch.setattr(runner, "_xdist_available", lambda _p: True)
    monkeypatch.setattr(runner, "journald_nrestarts", read_nrestarts)
    args = [
        "--tier",
        "2",
        "--workers",
        "6",
        "--no-prebuild",
        "--out-dir",
        str(out_dir),
        "--cache-dir",
        str(tmp_path / "cache"),
    ]
    return args, out_dir, solo


@pytest.mark.parametrize(
    ("lane_args", "options", "expected"),
    [
        ([], [], ["frontend-node", "lane-a-core", "lane-b-linux", "lane-b-sc6b1", "lane-b-sc6b2", "lane-c-compiled-solo"]),
        (["all"], ["--no-node"], ["lane-a-core", "lane-b-linux", "lane-b-sc6b1", "lane-b-sc6b2", "lane-c-compiled-solo"]),
        (["node"], [], ["frontend-node"]),
        (["c,a"], [], ["lane-a-core", "lane-c-compiled-solo"]),
        (["c b-sc6b2"], [], ["lane-b-sc6b2", "lane-c-compiled-solo"]),
        (["b-linux"], [], ["lane-b-linux"]),
        (["b-sc6b1"], [], ["lane-b-sc6b1"]),
        (["c"], ["--ignore-host-timers"], ["lane-c-compiled-solo"]),
        (
            ["c,b-sc6b1", "b", "a", "node"],
            ["--no-node"],
            ["lane-a-core", "lane-b-linux", "lane-b-sc6b1", "lane-b-sc6b2", "lane-c-compiled-solo"],
        ),
    ],
)
def test_tier2_cli_runs_selected_steps_in_canonical_order(tier2_cli, monkeypatch, lane_args, options, expected):
    args, out_dir, solo = tier2_cli
    timeline = []
    checks = []
    recorded_commands = {}

    def fake_run(name, cmd, *, cwd, env, log_path):
        assert solo.exists()
        timeline.append(("start", name))
        if name != "lane-c-compiled-solo":
            time.sleep(0.01)
        recorded_commands[name] = cmd
        timeline.append(("end", name))
        return {"name": name, "exit_code": 0, "duration_s": 0.01}

    def fake_check(minutes):
        assert solo.exists()
        assert not timeline
        checks.append(minutes)
        return [], []

    monkeypatch.setattr(runner, "_run_subprocess", fake_run)
    monkeypatch.setattr(runner, "check_host_interference", fake_check)
    if lane_args:
        args += ["--lanes", *lane_args]
    assert runner.main(args + options) == 0

    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["selected_lanes"] == expected == [step["name"] for step in summary["steps"]]
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", "HEAD"], cwd=runner.REPO_ROOT, text=True
    ).strip()
    assert summary["git_commit"] == git_commit
    assert len(git_commit) == 40
    tracked_status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=runner.REPO_ROOT, text=True
    )
    assert summary["git_dirty"] is bool(tracked_status)
    assert summary["before_NRestarts"] == summary["after_NRestarts"] == 42
    assert summary["nrestarts_ok"]
    assert summary["host_interference"] == []
    assert summary["ignore_host_timers"] is ("--ignore-host-timers" in options)
    assert summary["host_timers_checked"] == []
    assert not solo.exists()
    needs_check = any(name.startswith(("lane-b-", "lane-c-")) for name in expected)
    assert checks == ([45] if needs_check and "--ignore-host-timers" not in options else [])

    if "lane-c-compiled-solo" in expected:
        start_c = timeline.index(("start", "lane-c-compiled-solo"))
        assert all(timeline.index(("end", name)) < start_c for name in expected[:-1])
    lane_b = [name for name in expected if name.startswith("lane-b-")]
    assert [name for event, name in timeline if event == "start" and name.startswith("lane-b-")] == lane_b
    assert all(timeline.index(("end", a)) < timeline.index(("start", b)) for a, b in zip(lane_b, lane_b[1:]))
    for name in lane_b:
        cmd = recorded_commands[name]
        assert "-n" not in cmd
        if name.startswith("lane-b-sc6b"):
            assert cmd[-1] == "tests/integration/test_linux_sc6b_release_rehearsal.py"
            first_phase = "test_01 or test_02 or test_03 or test_04 or test_05 or test_06"
            phase = first_phase if name == "lane-b-sc6b1" else f"not ({first_phase})"
            assert cmd[cmd.index("-k") + 1] == phase


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--tier", "2", "--lanes", "a,unknown"], "unknown --lanes token(s): unknown"),
        (["--tier", "2", "--lanes", ", "], "--lanes requires at least one lane"),
        (["--tier", "2", "--lanes"], "expected at least one argument"),
        (["--tier", "2", "--lanes", "node", "--no-node"], "leave no Tier 2 steps selected"),
        (["--tier", "0", "--lanes", "a", "--targets", "tests/a.py"], "--lanes selection requires --tier 2"),
        (["--tier", "1", "--lanes", "c"], "--lanes selection requires --tier 2"),
        (["--tier", "2", "--quiet-window-minutes", "-1"], "--quiet-window-minutes must be >= 0"),
    ],
)
def test_cli_rejects_invalid_lane_options(args, message, capsys):
    with pytest.raises(SystemExit) as exc:
        runner.main(args)
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def test_git_metadata_returns_none_outside_repository(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    assert runner.read_git_commit(tmp_path) is None
    assert runner.read_git_dirty(tmp_path) is None


@pytest.mark.parametrize(("tier", "pytest_step"), [(0, "tier0-pytest"), (1, "tier1-core")])
@pytest.mark.parametrize("no_node", [False, True])
def test_cli_summarizes_selected_lanes_for_lower_tiers(tmp_path, monkeypatch, tier, pytest_step, no_node):
    out_dir = tmp_path / "out"
    calls = []

    def fake_run(name, cmd, *, cwd, env, log_path):
        calls.append(name)
        return {"name": name, "exit_code": 0, "duration_s": 0.01}

    def unexpected_check(*_args):
        pytest.fail("host timers are only inspected for Tier 2 lanes B and C")

    monkeypatch.setattr(runner.sys, "platform", "darwin")
    monkeypatch.setattr(runner, "_xdist_available", lambda _p: False)
    monkeypatch.setattr(runner, "_run_subprocess", fake_run)
    monkeypatch.setattr(runner, "check_host_interference", unexpected_check)
    args = [
        "--tier", str(tier),
        "--workers", "1",
        "--lanes", "all",
        "--no-prebuild",
        "--out-dir", str(out_dir),
        "--cache-dir", str(tmp_path / "cache"),
    ]
    if tier == 0:
        args += ["--targets", "tests/a.py"]
    if no_node:
        args += ["--no-node"]
    assert runner.main(args) == 0

    summary = json.loads((out_dir / "summary.json").read_text())
    expected = [pytest_step] if no_node else ["frontend-node", pytest_step]
    assert summary["selected_lanes"] == calls == expected
    assert not summary["ignore_host_timers"]
    assert summary["host_timers_checked"] == []
    assert summary["host_interference"] == []


def test_read_git_dirty_ignores_untracked_files_and_reports_index_changes(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "sample.py").write_text("x = 1\n")
    assert runner.read_git_dirty(tmp_path) is False
    subprocess.run(["git", "add", "--", "sample.py"], cwd=tmp_path, check=True)
    assert runner.read_git_dirty(tmp_path) is True


def test_tier2_partial_run_fails_when_journald_restarts(tier2_cli, monkeypatch):
    args, out_dir, solo = tier2_cli
    readings = iter((42, 43))

    def read_nrestarts():
        assert solo.exists()
        return next(readings)

    def fake_run(name, cmd, *, cwd, env, log_path):
        return {"name": name, "exit_code": 0, "duration_s": 0.01}

    monkeypatch.setattr(runner, "journald_nrestarts", read_nrestarts)
    monkeypatch.setattr(runner, "_run_subprocess", fake_run)
    assert runner.main(args + ["--lanes", "a"]) == 1

    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["before_NRestarts"] == 42
    assert summary["after_NRestarts"] == 43
    assert not summary["nrestarts_ok"]
    assert not solo.exists()


@pytest.fixture
def fake_systemctl(monkeypatch):
    real_run = subprocess.run
    responses = {
        ("user", unit): f"Id={unit}\nLoadState=loaded\nActiveState=inactive\nSubState=dead\n"
        for unit in runner.HOST_INTERFERENCE_UNITS
    }
    queried = []

    def run(cmd, **kwargs):
        if cmd[0] != "systemctl":
            return real_run(cmd, **kwargs)
        user = cmd[1] == "--user"
        scope = "user" if user else "system"
        prefix = ["systemctl", "--user"] if user else ["systemctl"]
        unit = cmd[len(prefix) + 1]
        assert cmd == [*prefix, "show", unit, "-p", "Id,LoadState,ActiveState,SubState,NextElapseUSecRealtime"]
        assert unit in runner.HOST_INTERFERENCE_UNITS
        assert kwargs["env"]["TZ"] == "UTC"
        assert kwargs["env"]["LC_ALL"] == "C"
        assert kwargs["capture_output"] and kwargs["text"] and kwargs["timeout"] > 0
        queried.append((scope, unit))
        response = responses.get((scope, unit))
        if isinstance(response, Exception):
            raise response
        if isinstance(response, subprocess.CompletedProcess):
            return response
        if response is None:
            missing = f"Id={unit}\nLoadState=not-found\nActiveState=inactive\nSubState=dead\n"
            return subprocess.CompletedProcess(cmd, 4, stdout=missing, stderr=f"Unit {unit} could not be found.")
        return subprocess.CompletedProcess(cmd, 0, stdout=response, stderr="")

    monkeypatch.setattr(runner.sys, "platform", "linux")
    monkeypatch.setattr(runner.subprocess, "run", run)
    return responses, queried


@pytest.mark.parametrize("state", ["active", "activating", "deactivating"])
def test_check_host_interference_reports_busy_service(fake_systemctl, state):
    responses, queried = fake_systemctl
    service = "thermouy-ci-prune.service"
    responses["user", service] = f"Id={service}\nLoadState=loaded\nActiveState={state}\nSubState=running\n"
    interference, checked = runner.check_host_interference(45)
    assert interference == [f"{service} is {state} (running)"]
    assert queried == [
        ("user", "thermouy-tier2-nightly.service"),
        ("user", "thermouy-tier2-nightly.timer"),
        ("user", "thermouy-ci-prune.service"),
        ("user", "thermouy-ci-prune.timer"),
    ]
    check = next(item for item in checked if item.unit == service)
    assert (check.scope, check.load_state, check.active_state) == ("user", "loaded", state)


@pytest.mark.parametrize(
    ("offset_minutes", "quiet_window", "interferes"),
    [(20, 45, True), (20, 10, False), (90, 45, False), (-5, 45, False), (None, 45, False)],
)
def test_check_host_interference_compares_timer_to_utc(fake_systemctl, offset_minutes, quiet_window, interferes):
    responses, _ = fake_systemctl
    timer = "thermouy-tier2-nightly.timer"
    timestamp = "n/a"
    next_elapse_utc = None
    if offset_minutes is not None:
        trigger = dt.datetime.now(dt.UTC).replace(microsecond=0) + dt.timedelta(minutes=offset_minutes)
        timestamp = f"{trigger:%a %Y-%m-%d %H:%M:%S} UTC"
        next_elapse_utc = trigger.isoformat()
    responses["user", timer] = (
        f"NextElapseUSecRealtime={timestamp}\nId={timer}\nLoadState=loaded\nActiveState=active\nSubState=waiting\n"
    )
    expected = [f"{timer} is scheduled for {timestamp} within {quiet_window} minutes"] if interferes else []
    interference, checked = runner.check_host_interference(quiet_window)
    assert interference == expected
    check = next(item for item in checked if item.unit == timer)
    assert check.next_elapse_utc == next_elapse_utc


def test_check_host_interference_allows_missing_units(fake_systemctl):
    responses, queried = fake_systemctl
    responses.clear()
    interference, checked = runner.check_host_interference(45)
    assert interference == []
    assert queried == [(scope, unit) for unit in runner.HOST_INTERFERENCE_UNITS for scope in ("user", "system")]
    assert [(item.unit, item.scope, item.load_state) for item in checked] == [
        (unit, "not-found", "not-found") for unit in runner.HOST_INTERFERENCE_UNITS
    ]


def test_check_host_interference_falls_back_to_system_for_user_unit_not_found(fake_systemctl):
    responses, queried = fake_systemctl
    service = "thermouy-tier2-nightly.service"
    responses["user", service] = f"Id={service}\nLoadState=not-found\nActiveState=inactive\nSubState=dead\n"
    responses["system", service] = f"Id={service}\nLoadState=loaded\nActiveState=active\nSubState=running\n"

    interference, checked = runner.check_host_interference(45)

    assert interference == [f"{service} is active (running)"]
    assert queried == [("user", service), ("system", service)] + [
        ("user", unit) for unit in runner.HOST_INTERFERENCE_UNITS[1:]
    ]
    assert [(item.scope, item.load_state) for item in checked] == [("system", "loaded")] + [("user", "loaded")] * 3
    assert checked[0].active_state == "active"


@pytest.mark.parametrize(
    ("scope", "response", "detail"),
    [
        ("user", FileNotFoundError("systemctl"), "FileNotFoundError"),
        ("system", subprocess.TimeoutExpired(["systemctl"], 5), "TimeoutExpired"),
        (
            "user",
            subprocess.CompletedProcess(["systemctl"], 1, stdout="", stderr="Failed to connect to bus"),
            "Failed to connect to bus",
        ),
    ],
)
def test_tier2_unavailable_preflight_blocks_unless_explicitly_ignored(
    tier2_cli, fake_systemctl, monkeypatch, capsys, scope, response, detail
):
    args, out_dir, solo = tier2_cli
    responses, queried = fake_systemctl
    unit = "thermouy-tier2-nightly.service"
    if scope == "system":
        del responses["user", unit]
    responses[scope, unit] = response
    started = []

    def fake_run(name, cmd, *, cwd, env, log_path):
        started.append(name)
        return {"name": name, "exit_code": 0, "duration_s": 0.01}

    monkeypatch.setattr(runner, "_run_subprocess", fake_run)
    args += ["--lanes", "c"]
    assert runner.main(args) == 2
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["exit_code"] == 2 and summary["steps"] == started == []
    assert not summary["ignore_host_timers"]
    assert len(summary["host_timers_checked"]) == 4
    check = next(item for item in summary["host_timers_checked"] if item["unit"] == unit)
    assert check["scope"] == "unavailable"
    assert check["load_state"] is check["active_state"] is check["next_elapse_utc"] is None
    assert scope in check["error"] and detail in check["error"]
    reason = f"{unit}: {check['error']}"
    assert summary["host_interference"] == [reason]
    assert reason in capsys.readouterr().err
    assert queried[:2] == [
        ("user", unit), (scope, unit if scope == "system" else "thermouy-tier2-nightly.timer")
    ]
    assert not solo.exists()

    queried.clear()
    assert runner.main(args + ["--ignore-host-timers"]) == 0
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["ignore_host_timers"] is True
    assert summary["host_timers_checked"] == summary["host_interference"] == queried == []
    assert started == ["lane-c-compiled-solo"]
    assert not solo.exists()


def test_journald_nrestarts_allows_missing_systemctl(monkeypatch):
    def unavailable(*args, **kwargs):
        raise FileNotFoundError("systemctl")

    monkeypatch.setattr(runner.sys, "platform", "linux")
    monkeypatch.setattr(runner.subprocess, "run", unavailable)
    assert runner.journald_nrestarts() is None


def test_tier2_preflight_aborts_and_releases_solo_window(tier2_cli, fake_systemctl, monkeypatch, capsys):
    args, out_dir, solo = tier2_cli
    responses, queried = fake_systemctl
    timer = "thermouy-ci-prune.timer"
    trigger = dt.datetime.now(dt.UTC).replace(microsecond=0) + dt.timedelta(minutes=5)
    responses["user", timer] = (
        f"NextElapseUSecRealtime={trigger:%a %Y-%m-%d %H:%M:%S} UTC\n"
        f"Id={timer}\nLoadState=loaded\nActiveState=active\nSubState=waiting\n"
    )

    original_check = runner.check_host_interference

    def check_under_lock(minutes):
        assert solo.exists()
        with pytest.raises(RuntimeError, match="holds host lock"):
            runner.acquire_solo_window(tier=1)
        return original_check(minutes)

    def unexpected_run(*args, **kwargs):
        pytest.fail("preflight must abort before launching any lane")

    monkeypatch.setattr(runner, "check_host_interference", check_under_lock)
    monkeypatch.setattr(runner, "_run_subprocess", unexpected_run)
    assert runner.main(args + ["--lanes", "a,b-sc6b1,c", "--quiet-window-minutes", "12"]) == 2

    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["exit_code"] == 2
    assert summary["selected_lanes"] == ["lane-a-core", "lane-b-sc6b1", "lane-c-compiled-solo"]
    assert summary["steps"] == []
    assert not summary["nrestarts_ok"]
    assert summary["ignore_host_timers"] is False
    assert len(summary["host_timers_checked"]) == 4
    check = next(item for item in summary["host_timers_checked"] if item["unit"] == timer)
    assert check == {
        "unit": timer,
        "unit_id": timer,
        "scope": "user",
        "load_state": "loaded",
        "active_state": "active",
        "sub_state": "waiting",
        "next_elapse_utc": trigger.isoformat(),
        "error": None,
    }
    (reason,) = summary["host_interference"]
    assert "thermouy-ci-prune.timer" in reason and "within 12 minutes" in reason
    assert reason in capsys.readouterr().err
    assert len(queried) == 4
    assert not solo.exists()
    runner.acquire_solo_window(tier=2).release()


def test_safe_rmtree_ignores_symlinks_to_outside_files(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("keep")
    cache_tar = tmp_path / "cache.tar.gz"
    cache_tar.write_bytes(b"tar")
    os.chmod(cache_tar, 0o444)

    tree = tmp_path / "tree"
    (tree / "sub").mkdir(parents=True)
    linked_tar = tree / "sub" / "ro.tar.gz"
    os.link(cache_tar, linked_tar)
    os.symlink(outside, tree / "sub" / "link")
    os.chmod(tree / "sub", 0o555)

    runner.safe_rmtree(tree)
    assert not tree.exists()
    assert outside.read_text() == "keep"
    assert stat.S_IMODE(cache_tar.stat().st_mode) == 0o444


def test_main_reads_nrestarts_inside_solo_window_and_releases_on_rmtree_error(tmp_path, monkeypatch):
    solo = tmp_path / "runs" / "SOLO_WINDOW"
    solo.parent.mkdir(parents=True)
    out_dir = tmp_path / "out"
    journald_states: list[bool] = []

    monkeypatch.setattr(runner, "DEFAULT_SOLO_WINDOW_FILE", solo)
    monkeypatch.setattr(runner.sys, "platform", "linux")
    monkeypatch.setattr(runner, "_xdist_available", lambda _p: True)
    monkeypatch.setattr(
        runner,
        "choose_basetemp_root",
        lambda *_a, **_kw: ((tmp_path / "bt").mkdir(parents=True, exist_ok=True) or (tmp_path / "bt"), True),
    )

    def fake_nrestarts():
        journald_states.append(solo.exists())
        return 111

    monkeypatch.setattr(runner, "journald_nrestarts", fake_nrestarts)
    monkeypatch.setattr(
        runner,
        "run_tier2_lanes",
        lambda **_kw: [{"name": "lane-a", "exit_code": 0, "duration_s": 0.01, "log": "x"}],
    )
    monkeypatch.setattr(runner, "safe_rmtree", lambda _p: (_ for _ in ()).throw(OSError("rmtree failed")))

    try:
        runner.main(
            [
                "--tier",
                "2",
                "--no-prebuild",
                "--no-node",
                "--ignore-host-timers",
                "--out-dir",
                str(out_dir),
                "--cache-dir",
                str(tmp_path / "cache"),
            ]
        )
        raise AssertionError("expected OSError from safe_rmtree")
    except OSError as exc:
        assert "rmtree failed" in str(exc)

    assert journald_states == [True, True]
    assert not solo.exists()


def test_same_second_runs_allocate_distinct_out_dir_and_basetemp(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(runner.time, "strftime", lambda *_a: "20261120-080000")
    monkeypatch.setattr(runner.sys, "platform", "darwin")

    seen_cmds: list[list[str]] = []
    monkeypatch.setattr(
        runner,
        "_run_subprocess",
        lambda name, cmd, *, cwd, env, log_path: (seen_cmds.append(list(cmd)) or {"name": name, "exit_code": 0, "duration_s": 0.0, "log": str(log_path)}),
    )
    monkeypatch.setattr(runner, "_xdist_available", lambda _p: False)

    assert runner.main(["--tier", "0", "--no-node", "--targets", "tests/a.py", "--cache-dir", str(tmp_path / "c")]) == 0
    assert runner.main(["--tier", "0", "--no-node", "--targets", "tests/b.py", "--cache-dir", str(tmp_path / "c")]) == 0
    assert len(seen_cmds) == 2
    bt1 = next(arg for arg in seen_cmds[0] if arg.startswith("--basetemp="))
    bt2 = next(arg for arg in seen_cmds[1] if arg.startswith("--basetemp="))
    xml1 = next(arg for arg in seen_cmds[0] if arg.startswith("--junitxml="))
    xml2 = next(arg for arg in seen_cmds[1] if arg.startswith("--junitxml="))
    assert bt1 != bt2 and xml1 != xml2


def test_cli_refuses_solo_window_override_flag():
    import pytest
    with pytest.raises(SystemExit):
        runner.main(["--tier", "0", "--targets", "tests/a.py", "--solo-window-file", "/tmp/other"])


def test_main_cleans_shm_basetemp_when_prebuild_fails(tmp_path, monkeypatch):
    shm = tmp_path / "dev_shm"
    shm.mkdir()
    out_dir = tmp_path / "out"
    solo = tmp_path / "runs" / "SOLO_WINDOW"
    solo.parent.mkdir(parents=True)
    monkeypatch.setattr(runner, "DEFAULT_SOLO_WINDOW_FILE", solo)
    monkeypatch.setattr(runner.sys, "platform", "linux")
    monkeypatch.setattr(runner.os, "statvfs", lambda _p: SimpleNamespace(f_bavail=32 * 1024**3, f_frsize=1))
    monkeypatch.setattr(runner, "_shm_has_capacity", lambda *_args, **_kw: True)
    monkeypatch.setattr(
        runner,
        "choose_basetemp_root",
        lambda explicit, od, **_kw: (
            (shm / "reserved-bt").mkdir(parents=True, exist_ok=True) or (shm / "reserved-bt"),
            True,
        ),
    )

    from tests.mocks import runtime_artifact_cache as cache_mod
    monkeypatch.setattr(cache_mod, "prebuild_standard_artifacts", lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("boom")))

    try:
        runner.main(
            [
                "--tier",
                "1",
                "--no-node",
                "--prebuild",
                "--out-dir",
                str(out_dir),
                "--cache-dir",
                str(tmp_path / "cache"),
            ]
        )
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "boom" in str(exc)
    assert not (shm / "reserved-bt").exists()
    guard = runner.acquire_solo_window(tier=2, platform="linux")
    guard.release()


def test_main_disk_basetemp_cleanup_preserves_symlink_on_prebuild_error(tmp_path, monkeypatch):
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "keep.txt"
    sentinel.write_text("keep")

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    os.symlink(outside, out_dir / "basetemp")

    solo = tmp_path / "runs" / "SOLO_WINDOW"
    solo.parent.mkdir(parents=True)
    monkeypatch.setattr(runner, "DEFAULT_SOLO_WINDOW_FILE", solo)
    monkeypatch.setattr(runner.sys, "platform", "linux")

    from tests.mocks import runtime_artifact_cache as cache_mod
    monkeypatch.setattr(cache_mod, "prebuild_standard_artifacts", lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("boom")))

    try:
        runner.main(
            [
                "--tier",
                "1",
                "--no-node",
                "--prebuild",
                "--out-dir",
                str(out_dir),
                "--cache-dir",
                str(tmp_path / "cache"),
            ]
        )
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "boom" in str(exc)

    assert not list(out_dir.glob("basetemp-*"))
    assert (out_dir / "basetemp").is_symlink()
    assert sentinel.read_text() == "keep"


@pytest.fixture(scope="module")
def ci_workflow() -> dict:
    """Load the checked-in GitHub Actions workflow with a safe YAML parser."""
    path = runner.REPO_ROOT / ".github" / "workflows" / "ci.yml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _ci_pytest_command(workflow: dict, suite: str) -> list[str]:
    steps = workflow["jobs"][f"{suite}-tests"]["steps"]
    test_step = next(step for step in steps if step.get("name") == f"Run {suite} tests")
    return shlex.split(test_step["run"])


@pytest.mark.parametrize("suite", ["unit", "safety", "integration"])
def test_ci_pytest_jobs_use_worksteal(ci_workflow: dict, suite: str) -> None:
    command = _ci_pytest_command(ci_workflow, suite)
    assert command[:2] == ["pytest", f"tests/{suite}/"]
    worker_option = command.index("-n")
    assert command[worker_option + 1] == "auto"
    assert "--dist=worksteal" in command


def test_ci_integration_excludes_real_systemd_modules(ci_workflow: dict) -> None:
    command = _ci_pytest_command(ci_workflow, "integration")
    ignored = {arg.removeprefix("--ignore=") for arg in command if arg.startswith("--ignore=")}
    assert ignored == set(runner.LINUX_SYSTEMD_MODULES)


def test_ci_safety_failures_block_the_job(ci_workflow: dict) -> None:
    job = ci_workflow["jobs"]["safety-tests"]
    assert "continue-on-error" not in job
    assert all("continue-on-error" not in step for step in job["steps"])


def test_ci_keeps_nightly_training_triggers(ci_workflow: dict) -> None:
    # PyYAML's YAML 1.1 resolver reads an unquoted GitHub Actions "on" key as True.
    triggers = ci_workflow.get("on", ci_workflow.get(True))
    assert triggers["schedule"]
    assert "workflow_dispatch" in triggers
    job = ci_workflow["jobs"]["nightly-training-test"]
    assert job["if"] == "github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'"
    assert any(
        step.get("name") == "Simulate nightly training cycle" and step.get("run")
        for step in job["steps"]
    )
