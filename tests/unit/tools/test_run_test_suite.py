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
