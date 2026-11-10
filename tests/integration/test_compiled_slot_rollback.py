"""A compiled two-root rehearsal retains latest privacy authority on activation."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import stage_runtime as stage
from scripts.build_runtime import build_runtime
from tests.mocks.slot_rollback import exercise_slot_rollback


UNMANAGED_REVISION = "9b3170294eb69a53dd27b63949c09a63a2efed54"


def test_compiled_slot_rollback_preserves_latest_privacy_authority(tmp_path):
    exercise_slot_rollback(tmp_path)


def _tree(root, *states):
    """admission/** bytes, the current link and every state marker/ownership file."""
    tree = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if relative in {".admission.lock", ".stage.lock"} or relative.startswith("slots/"):
            continue
        tree[relative] = (("link", os.readlink(path)) if path.is_symlink() else ("dir",)
                          if path.is_dir() else ("file", path.read_bytes()))
    for state in states:
        for name in (stage._MARKER, ".kza-target-ownership.json"):
            path = state / name
            tree[str(path)] = path.read_bytes() if path.exists() else None
    return tree


def test_former_unmanaged_rehearsal_slot_is_refused_with_nothing_changed(tmp_path, monkeypatch):
    """The slot the rehearsals used before U3b-0a can no longer be selected or run managed."""
    repo = Path(__file__).resolve().parents[2]
    if subprocess.run(["git", "cat-file", "-e", UNMANAGED_REVISION + "^{commit}"], cwd=repo).returncode:
        pytest.skip("9b31702 is not available in this clone")
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout", str(repo), str(clone)],
                   check=True, timeout=120)
    legacy_archive = tmp_path / "legacy.tar.gz"
    manifest = build_runtime(clone, legacy_archive, UNMANAGED_REVISION)
    assert "scripts/stage_runtime.pyc" in manifest["files"]
    assert "src/core/runtime_admission.pyc" not in manifest["files"]
    current_archive = tmp_path / "current.tar.gz"
    build_runtime(repo, current_archive)
    source = tmp_path / "offline"
    (source / "config").mkdir(parents=True)
    (source / "data").mkdir()
    (source / "config" / "settings.yaml").write_text("schema: unmanaged-refusal\n")
    root = tmp_path / "staging"
    stage.prepare_slot(current_archive, root, "old", source)
    root = root.resolve()
    with stage.AdmissionController(root) as controller:
        assert controller.activate("old", managed=True) is None  # Bootstrap closed head.
    head = stage.read_admission_head(root)
    assert head["status"] == "closed"
    state = root / "slots/old/state"
    stage.prepare_slot(legacy_archive, root, "legacy", state)
    legacy = root / "slots/legacy/state"
    launched = []
    original = subprocess.Popen
    monkeypatch.setattr(stage.subprocess, "Popen",
                        lambda *args, **kwargs: launched.append(args) or original(*args, **kwargs))
    before = _tree(root, state, legacy)
    with stage.AdmissionController(root) as controller:
        with pytest.raises(ValueError, match="^runtime_admission_candidate_unmanaged$"):
            controller.activate("legacy")
        with pytest.raises(ValueError, match="^runtime_admission_candidate_unmanaged$"):
            controller.run("legacy", [sys.executable, "-c", "raise SystemExit(0)"], timeout=5)
    assert _tree(root, state, legacy) == before
    assert stage.read_admission_head(root) == head  # Never 'running' or 'unknown'.
    assert launched == []  # Decided from the verified manifest; neither probe nor child started.
