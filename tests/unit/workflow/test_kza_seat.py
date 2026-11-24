"""Unit tests for scripts/kza_seat.py using real temporary git repositories."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import kza_seat  # noqa: E402


def _init_git_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", "main"], cwd=str(repo), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@kza.local"], cwd=str(repo), check=True)
    subprocess.run(["git", "config", "user.name", "KZA Test"], cwd=str(repo), check=True)
    (repo / "README.md").write_text("# KZA Test Repo\n", encoding="utf-8")
    (repo / "herdr" / "templates").mkdir(parents=True, exist_ok=True)
    (repo / "herdr" / "templates" / "kza-qa.md").write_text("---\nname: kza-qa\n---\nQA body\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md", "herdr/templates/kza-qa.md"], cwd=str(repo), check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), check=True, capture_output=True)


def test_validate_alias_rejects_foreign_prefixes() -> None:
    assert kza_seat.validate_alias("kza-qa-pr1") == "kza-qa-pr1"
    for bad in ("tu-qa", "gsm-dev", "hl-ops", "random-alias", "kza-UPPER"):
        with pytest.raises(ValueError):
            kza_seat.validate_alias(bad)


def test_sync_templates_links_only_kza_and_protects_non_symlinks(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    target = tmp_path / "herdr-templates"
    target.mkdir()

    # Pre-existing foreign template must NEVER be touched.
    tu_file = target / "tu-qa.md"
    tu_file.write_text("do not touch", encoding="utf-8")

    # Existing regular file matching kza-qa.md must fail without --force.
    existing_kza = target / "kza-qa.md"
    existing_kza.write_text("regular file", encoding="utf-8")

    with pytest.raises(RuntimeError, match="non-symlink"):
        kza_seat.sync_templates(repo_root=repo, target_dir=target, force=False)

    # With force=True it replaces kza-qa.md with symlink and leaves tu-qa.md untouched.
    res = kza_seat.sync_templates(repo_root=repo, target_dir=target, force=True)
    assert res["linked"] == ["kza-qa.md"]
    assert existing_kza.is_symlink()
    assert existing_kza.resolve() == (repo / "herdr" / "templates" / "kza-qa.md").resolve()
    assert tu_file.read_text(encoding="utf-8") == "do not touch"
    assert not tu_file.is_symlink()


def test_open_rollback_when_hc_run_fails(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    wt_root = tmp_path / "worktrees"
    state_dir = tmp_path / "state"

    orig_run = kza_seat._run

    def _fake_run(cmd: list[str], **kwargs):
        if cmd[:2] == ["hc", "run"]:
            return subprocess.CompletedProcess(cmd, 1, "", "hc daemon error")
        return orig_run(cmd, **kwargs)

    with mock.patch.object(kza_seat, "_run", side_effect=_fake_run):
        with pytest.raises(RuntimeError, match="hc run"):
            kza_seat.open_seat(
                alias="kza-dev-fail",
                role="kza-dev",
                task="Implement feature",
                offline=True,
                allow_quota_emergency=True,
                space_id="sp-test",
                parent_terminal_id="t0",
                repo_root=repo,
                worktree_root=wt_root,
                state_dir=state_dir,
            )

    # Verify rollback removed BOTH the worktree directory AND the newly created branch.
    assert not (wt_root / "kza-dev-fail").exists()
    branches = subprocess.run(
        ["git", "branch", "--list", "seat/kza-dev-fail"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert branches == ""


def test_close_guards_working_foreign_dirt_and_merge_ancestry(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    wt_root = tmp_path / "worktrees"
    state_dir = tmp_path / "state"

    orig_run = kza_seat._run

    def _fake_open_run(cmd: list[str], **kwargs):
        if cmd[:2] == ["hc", "run"]:
            return subprocess.CompletedProcess(cmd, 0, json.dumps({"address": "t99"}), "")
        if cmd[:2] == ["hc", "kill"] or cmd[:2] == ["hc", "rm"]:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return orig_run(cmd, **kwargs)

    with (
        mock.patch.object(kza_seat, "_run", side_effect=_fake_open_run),
        mock.patch.object(kza_seat, "_rename_herdr_tab_no_focus", return_value="tb99"),
    ):
        seat = kza_seat.open_seat(
            alias="kza-dev-guard",
            role="kza-dev",
            task="Test close guards",
            offline=True,
            allow_quota_emergency=True,
            space_id="sp-test",
            parent_terminal_id="t0",
            repo_root=repo,
            worktree_root=wt_root,
            state_dir=state_dir,
        )

    wt = Path(seat["worktree"])
    # Commit a change on seat/kza-dev-guard so it is NOT yet an ancestor of main.
    (wt / "feature.txt").write_text("new feature\n", encoding="utf-8")
    subprocess.run(["git", "add", "feature.txt"], cwd=str(wt), check=True)
    subprocess.run(["git", "commit", "-m", "feat"], cwd=str(wt), check=True, capture_output=True)

    # Guard 1: refuse closing when terminal is still `working`.
    with mock.patch.object(
        kza_seat, "_hc_get_terminal", return_value={"state": "live", "status": {"state": "working"}}
    ):
        with pytest.raises(RuntimeError, match="working"):
            kza_seat.close_seat(
                alias="kza-dev-guard",
                merged=True,
                base_ref="main",
                repo_root=repo,
                state_dir=state_dir,
            )

    # Guard 1b: refuse closing when report file is missing or empty.
    with mock.patch.object(
        kza_seat, "_hc_get_terminal", return_value={"state": "live", "status": {"state": "done"}}
    ):
        with pytest.raises(RuntimeError, match="missing or empty report"):
            kza_seat.close_seat(
                alias="kza-dev-guard",
                merged=True,
                base_ref="main",
                repo_root=repo,
                state_dir=state_dir,
            )

    rpath = Path(seat["report_path"])
    rpath.parent.mkdir(parents=True, exist_ok=True)
    rpath.write_text("# Dev Report\nImplemented feature.txt cleanly.\n", encoding="utf-8")

    # Guard 2: detect foreign dirt outside ensure_manifest (specifically `.serena/memories/extra.md`).
    serena_mem = wt / ".serena" / "memories" / "extra.md"
    serena_mem.parent.mkdir(parents=True, exist_ok=True)
    serena_mem.write_text("stray memory", encoding="utf-8")
    with mock.patch.object(
        kza_seat, "_hc_get_terminal", return_value={"state": "live", "status": {"state": "done"}}
    ):
        with pytest.raises(RuntimeError, match="extra.md"):
            kza_seat.close_seat(
                alias="kza-dev-guard",
                merged=True,
                base_ref="main",
                repo_root=repo,
                state_dir=state_dir,
            )
    serena_mem.unlink()

    # Guard 3: refuse `--merged` when branch is NOT an ancestor of main.
    with mock.patch.object(
        kza_seat, "_hc_get_terminal", return_value={"state": "live", "status": {"state": "done"}}
    ):
        with pytest.raises(RuntimeError, match="NOT an ancestor"):
            kza_seat.close_seat(
                alias="kza-dev-guard",
                merged=True,
                base_ref="main",
                repo_root=repo,
                state_dir=state_dir,
            )

    subprocess.run(["git", "merge", "--ff-only", "seat/kza-dev-guard"], cwd=str(repo), check=True, capture_output=True)
    with (
        mock.patch.object(kza_seat, "_hc_get_terminal", return_value={"state": "live", "status": {"state": "done"}}),
        mock.patch.object(kza_seat, "_run", side_effect=_fake_open_run),
    ):
        closed = kza_seat.close_seat(
            alias="kza-dev-guard",
            merged=True,
            base_ref="main",
            repo_root=repo,
            state_dir=state_dir,
        )
    assert closed["status"] == "closed"
    assert closed["close_mode"] == "merged"
    assert closed["outcome_recorded"] is True
    assert not wt.exists()

    # Verify outcome was written once and is not duplicated on subsequent record_outcome(skip_if_exists=True).
    lines = [json.loads(l) for l in (state_dir / "decisions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    outcomes = [o for o in lines if o.get("type") == "outcome" and o.get("decision_id") == seat["decision_id"]]
    assert len(outcomes) == 1
    assert outcomes[0]["round1_approved"] is True


def test_open_seat_requires_exclude_family_on_gate_roles(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    wt_root = tmp_path / "worktrees"
    state_dir = tmp_path / "state"

    for role in ("kza-qa", "kza-privacy"):
        with pytest.raises(ValueError, match="exclude-family"):
            kza_seat.open_seat(
                alias=f"{role}-nofam",
                role=role,
                task="Audit diff",
                offline=True,
                allow_quota_emergency=True,
                repo_root=repo,
                worktree_root=wt_root,
                state_dir=state_dir,
                dry_run=True,
            )

    dry = kza_seat.open_seat(
        alias="kza-qa-cross",
        role="kza-qa",
        task="Audit diff",
        exclude_families=["google"],
        offline=True,
        allow_quota_emergency=True,
        repo_root=repo,
        worktree_root=wt_root,
        state_dir=state_dir,
        dry_run=True,
    )
    assert dry["dry_run"] is True
    assert dry["decision"]["excluded_families"] == ["google"]

    dry_override = kza_seat.open_seat(
        alias="kza-privacy-same",
        role="kza-privacy",
        task="Audit diff",
        allow_same_family=True,
        offline=True,
        allow_quota_emergency=True,
        repo_root=repo,
        worktree_root=wt_root,
        state_dir=state_dir,
        dry_run=True,
    )
    assert dry_override["dry_run"] is True
