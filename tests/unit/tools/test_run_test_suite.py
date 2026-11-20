"""Tests for the portable multi-worker tier runner scripts/run_test_suite.py (T090-U1)."""

from __future__ import annotations

import os
from pathlib import Path
import stat
from types import SimpleNamespace

from scripts import run_test_suite as runner


def test_default_workers_caps_cleanly():
    assert runner.default_workers(platform="darwin", cpu_count=12) == 8
    assert runner.default_workers(platform="linux", cpu_count=48) == 12
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
        None, tmp_path / "out", platform="linux", shm_root=shm
    )
    assert bt.parent == shm and bt.exists() and should_clean is True

    monkeypatch.setattr(runner.os, "statvfs", lambda _p: SimpleNamespace(f_bavail=1 * 1024**3, f_frsize=1))
    bt_fallback, should_clean_fb = runner.choose_basetemp_root(
        None, tmp_path / "out", platform="linux", shm_root=shm
    )
    assert bt_fallback == (tmp_path / "out" / "basetemp").resolve() and should_clean_fb is False


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
    assert runner.choose_cache_dir(None, platform="linux", shm_root=shm, home_dir=tmp_path) == (
        shm / "kza-artifact-cache"
    ).resolve()

    monkeypatch.setattr(runner.os, "statvfs", lambda _p: SimpleNamespace(f_bavail=1 * 1024**3, f_frsize=1))
    assert runner.choose_cache_dir(None, platform="linux", shm_root=shm, home_dir=tmp_path) == (
        tmp_path / ".cache" / "kza-runtime-artifacts"
    ).resolve()


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
            import time
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

    cmd_c = recorded_cmds["lane-c-compiled-solo"]
    assert all("::" in target for target in cmd_c if target.startswith("tests/integration/"))
    assert len([t for t in cmd_c if t.startswith("tests/integration/")]) == len(runner.COMPILED_SOLO_TESTS)
    assert "-n" not in cmd_c
    assert "KZA_REQUIRE_REAL_SYSTEMD" not in recorded_envs["lane-a-core"]
    assert "KZA_REQUIRE_REAL_SYSTEMD" not in recorded_envs["lane-c-compiled-solo"]
    assert recorded_envs["lane-b-linux"]["KZA_REQUIRE_REAL_SYSTEMD"] == "1"


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
        lambda explicit, od: (
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
