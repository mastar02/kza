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
            approved=True,
            rounds=1,
            base_ref="main",
            repo_root=repo,
            state_dir=state_dir,
        )
    assert closed["status"] == "closed"
    assert closed["close_mode"] == "merged"
    assert closed["outcome_recorded"] is True
    assert not wt.exists()

    # Verify outcome was written once and a repeat call with skip_if_exists=True does not duplicate it.
    dup = kza_seat.kza_router.record_outcome(
        decision_id=seat["decision_id"],
        rounds=1,
        approved=True,
        state_dir=state_dir,
        skip_if_exists=True,
    )
    assert dup.get("already_recorded") is True
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

    # Whitespace-only exclude_families must also be rejected before routing.
    with pytest.raises(ValueError, match="exclude-family"):
        kza_seat.open_seat(
            alias="kza-qa-blankfam",
            role="kza-qa",
            task="Audit diff",
            exclude_families=["  "],
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


def test_reused_alias_archives_stale_report_and_outcome_errors_preserve_closed_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    repo = tmp_path / "repo"
    _init_git_repo(repo)
    wt_root = tmp_path / "worktrees"
    state_dir = tmp_path / "state"

    # Pre-seed an old report from a previous seat with the same alias.
    stale_report = state_dir / "reports" / "kza-qa-reuse.md"
    stale_report.parent.mkdir(parents=True, exist_ok=True)
    stale_report.write_text("# Old Report\nAPROBADO\n", encoding="utf-8")

    seen_envs: list[dict[str, str] | None] = []
    orig_run = kza_seat._run

    def _fake_run(cmd: list[str], **kwargs):
        if cmd[:2] == ["hc", "run"]:
            seen_envs.append(kwargs.get("env"))
            return subprocess.CompletedProcess(cmd, 0, json.dumps({"address": "t88"}), "")
        if cmd[:2] in (["hc", "kill"], ["hc", "rm"]):
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return orig_run(cmd, **kwargs)

    monkeypatch.setenv("CODEX_HOME", "/tmp/other-codex")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/tmp/other-claude")

    with (
        mock.patch.object(kza_seat, "_run", side_effect=_fake_run),
        mock.patch.object(kza_seat, "_rename_herdr_tab_no_focus", return_value="tb88"),
    ):
        seat = kza_seat.open_seat(
            alias="kza-qa-reuse",
            role="kza-qa",
            task="Audit new SHA",
            exclude_families=["google"],
            offline=True,
            allow_quota_emergency=True,
            space_id="sp-test",
            parent_terminal_id="t0",
            repo_root=repo,
            worktree_root=wt_root,
            state_dir=state_dir,
        )

    assert not stale_report.exists()
    archived = list((state_dir / "reports" / "archive").glob("kza-qa-reuse-*.md"))
    assert len(archived) == 1
    assert seen_envs and seen_envs[0] is not None
    assert "CODEX_HOME" not in seen_envs[0]
    assert "CLAUDE_CONFIG_DIR" not in seen_envs[0]

    # Placing a file with mtime older than opened_epoch must be rejected as stale.
    stale_report.write_text("# Backdated Report\nAPROBADO\n", encoding="utf-8")
    old_ts = float(seat["opened_epoch"]) - 10.0
    os.utime(stale_report, (old_ts, old_ts))
    with mock.patch.object(
        kza_seat, "_hc_get_terminal", return_value={"state": "live", "status": {"state": "done"}}
    ):
        with pytest.raises(RuntimeError, match="stale report"):
            kza_seat.close_seat(
                alias="kza-qa-reuse",
                done_readonly=True,
                repo_root=repo,
                state_dir=state_dir,
            )

    # Fresh report written now succeeds, and invalid rounds fails BEFORE removing worktree.
    stale_report.write_text("# Fresh Report\nAPROBADO\n", encoding="utf-8")
    with pytest.raises(ValueError, match="rounds must be >= 1"):
        kza_seat.close_seat(
            alias="kza-qa-reuse",
            done_readonly=True,
            rounds=0,
            approved=True,
            repo_root=repo,
            state_dir=state_dir,
        )
    assert Path(seat["worktree"]).exists()

    with pytest.raises(ValueError, match="--approved and --rounds"):
        kza_seat.close_seat(
            alias="kza-qa-reuse",
            done_readonly=True,
            rounds=2,
            repo_root=repo,
            state_dir=state_dir,
        )
    assert Path(seat["worktree"]).exists()

    with pytest.raises(ValueError, match="--approved and --rounds"):
        kza_seat.close_seat(
            alias="kza-qa-reuse",
            done_readonly=True,
            approved=True,
            repo_root=repo,
            state_dir=state_dir,
        )
    assert Path(seat["worktree"]).exists()

    # If record_outcome fails with OSError, seat must be persisted as closed with outcome_recorded=False.
    with (
        mock.patch.object(kza_seat, "_hc_get_terminal", return_value={"state": "live", "status": {"state": "done"}}),
        mock.patch.object(kza_seat, "_run", side_effect=_fake_run),
        mock.patch.object(kza_seat.kza_router, "record_outcome", side_effect=OSError("disk full")),
    ):
        with pytest.raises(RuntimeError, match="recording outcome failed"):
            kza_seat.close_seat(
                alias="kza-qa-reuse",
                done_readonly=True,
                approved=True,
                rounds=1,
                repo_root=repo,
                state_dir=state_dir,
            )

    saved = json.loads((state_dir / "seats.json").read_text(encoding="utf-8"))["kza-qa-reuse"]
    assert saved["status"] == "closed"
    assert saved["outcome_recorded"] is False
    assert "disk full" in saved["outcome_error"]
    assert not Path(seat["worktree"]).exists()


def test_parallel_open_during_close_outcome_is_preserved_and_legacy_created_at_checked(
    tmp_path: Path,
) -> None:
    import os

    repo = tmp_path / "repo"
    _init_git_repo(repo)
    wt_root = tmp_path / "worktrees"
    state_dir = tmp_path / "state"
    orig_run = kza_seat._run
    counter = {"n": 0}

    def _fake_run(cmd: list[str], **kwargs):
        if cmd[:2] == ["hc", "run"]:
            counter["n"] += 1
            return subprocess.CompletedProcess(cmd, 0, json.dumps({"address": f"t{counter['n']}"}), "")
        if cmd[:2] in (["hc", "kill"], ["hc", "rm"]):
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return orig_run(cmd, **kwargs)

    with (
        mock.patch.object(kza_seat, "_run", side_effect=_fake_run),
        mock.patch.object(kza_seat, "_rename_herdr_tab_no_focus", return_value="tb1"),
    ):
        seat1 = kza_seat.open_seat(
            alias="kza-qa-par1",
            role="kza-qa",
            task="Audit A",
            exclude_families=["google"],
            offline=True,
            allow_quota_emergency=True,
            space_id="sp-test",
            parent_terminal_id="t0",
            repo_root=repo,
            worktree_root=wt_root,
            state_dir=state_dir,
        )

    rep1 = Path(seat1["report_path"])
    rep1.write_text("# Report 1\nAPROBADO\n", encoding="utf-8")

    orig_record = kza_seat.kza_router.record_outcome

    def _interleaved_record_outcome(**kwargs):
        with (
            mock.patch.object(kza_seat, "_run", side_effect=_fake_run),
            mock.patch.object(kza_seat, "_rename_herdr_tab_no_focus", return_value="tb2"),
        ):
            kza_seat.open_seat(
                alias="kza-qa-par2",
                role="kza-qa",
                task="Audit B",
                exclude_families=["google"],
                offline=True,
                allow_quota_emergency=True,
                space_id="sp-test",
                parent_terminal_id="t0",
                repo_root=repo,
                worktree_root=wt_root,
                state_dir=state_dir,
            )
        return orig_record(**kwargs)

    with (
        mock.patch.object(kza_seat, "_hc_get_terminal", return_value={"state": "live", "status": {"state": "done"}}),
        mock.patch.object(kza_seat, "_run", side_effect=_fake_run),
        mock.patch.object(kza_seat.kza_router, "record_outcome", side_effect=_interleaved_record_outcome),
    ):
        closed1 = kza_seat.close_seat(
            alias="kza-qa-par1",
            done_readonly=True,
            approved=True,
            rounds=1,
            repo_root=repo,
            state_dir=state_dir,
        )

    assert closed1["status"] == "closed"
    assert closed1["outcome_recorded"] is True
    seats_after = json.loads((state_dir / "seats.json").read_text(encoding="utf-8"))
    assert "kza-qa-par1" in seats_after and seats_after["kza-qa-par1"]["status"] == "closed"
    assert "kza-qa-par2" in seats_after and seats_after["kza-qa-par2"]["status"] == "open"

    # Legacy seat without opened_epoch still rejects reports predating created_at.
    seats_after["kza-qa-par2"].pop("opened_epoch", None)
    (state_dir / "seats.json").write_text(json.dumps(seats_after), encoding="utf-8")
    rep2 = Path(seats_after["kza-qa-par2"]["report_path"])
    rep2.write_text("# Old legacy report\nAPROBADO\n", encoding="utf-8")
    old_ts = kza_seat.kza_quota._parse_timestamp(seats_after["kza-qa-par2"]["created_at"]) - 30.0
    os.utime(rep2, (old_ts, old_ts))

    with mock.patch.object(
        kza_seat, "_hc_get_terminal", return_value={"state": "live", "status": {"state": "done"}}
    ):
        with pytest.raises(RuntimeError, match="stale report"):
            kza_seat.close_seat(
                alias="kza-qa-par2",
                done_readonly=True,
                repo_root=repo,
                state_dir=state_dir,
            )


def test_concurrent_same_alias_open_and_close_serialization(tmp_path: Path) -> None:
    import threading

    repo = tmp_path / "repo"
    _init_git_repo(repo)
    wt_root = tmp_path / "worktrees"
    state_dir = tmp_path / "state"

    orig_route = kza_seat.kza_router.route_task
    orig_run = kza_seat._run
    routing_entered = threading.Event()
    allow_first_route = threading.Event()
    counter = {"n": 0, "route_calls": 0}
    lock = threading.Lock()

    def _slow_first_route(**kwargs):
        with lock:
            counter["route_calls"] += 1
            call_num = counter["route_calls"]
        if call_num == 1:
            routing_entered.set()
            allow_first_route.wait(timeout=5.0)
        return orig_route(**kwargs)

    def _fake_run(cmd: list[str], **kwargs):
        if cmd[:2] == ["hc", "run"]:
            with lock:
                counter["n"] += 1
                tid = f"t{counter['n']}"
            return subprocess.CompletedProcess(cmd, 0, json.dumps({"address": tid}), "")
        if cmd[:2] in (["hc", "kill"], ["hc", "rm"]):
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return orig_run(cmd, **kwargs)

    results: list[tuple[str, object]] = []

    def _open_worker(tag: str) -> None:
        try:
            res = kza_seat.open_seat(
                alias="kza-qa-race",
                role="kza-qa",
                task=f"Audit race {tag}",
                exclude_families=["google"],
                offline=True,
                allow_quota_emergency=True,
                space_id="sp-test",
                parent_terminal_id="t0",
                repo_root=repo,
                worktree_root=wt_root,
                state_dir=state_dir,
            )
            results.append(("ok", res))
        except Exception as exc:
            results.append(("err", exc))

    with (
        mock.patch.object(kza_seat.kza_router, "route_task", side_effect=_slow_first_route),
        mock.patch.object(kza_seat, "_run", side_effect=_fake_run),
        mock.patch.object(kza_seat, "_rename_herdr_tab_no_focus", return_value="tb-race"),
    ):
        t1 = threading.Thread(target=_open_worker, args=("one",))
        t2 = threading.Thread(target=_open_worker, args=("two",))
        t1.start()
        assert routing_entered.wait(timeout=5.0)
        t2.start()
        # Let thread 2 block on _alias_lock while thread 1 finishes opening & worker writes its report.
        allow_first_route.set()
        t1.join(timeout=10.0)
        t2.join(timeout=10.0)

    oks = [r for kind, r in results if kind == "ok"]
    errs = [r for kind, r in results if kind == "err"]
    assert len(oks) == 1 and len(errs) == 1
    assert "already open" in str(errs[0])

    # Winner writes its report; second failed open must NOT have archived/removed it.
    winner = oks[0]
    assert isinstance(winner, dict)
    rep_path = Path(winner["report_path"])
    rep_path.write_text("# Winner Report\nAPROBADO\n", encoding="utf-8")

    # Concurrent close_seat calls on the same alias: exactly 1 succeeds, 1 rejects 'already closed', 1 outcome row.
    close_results: list[tuple[str, object]] = []

    def _close_worker() -> None:
        try:
            c = kza_seat.close_seat(
                alias="kza-qa-race",
                done_readonly=True,
                approved=True,
                rounds=1,
                repo_root=repo,
                state_dir=state_dir,
            )
            close_results.append(("ok", c))
        except Exception as exc:
            close_results.append(("err", exc))

    with (
        mock.patch.object(kza_seat, "_hc_get_terminal", return_value={"state": "live", "status": {"state": "done"}}),
        mock.patch.object(kza_seat, "_run", side_effect=_fake_run),
    ):
        c1 = threading.Thread(target=_close_worker)
        c2 = threading.Thread(target=_close_worker)
        c1.start()
        c2.start()
        c1.join(timeout=10.0)
        c2.join(timeout=10.0)

    c_oks = [r for kind, r in close_results if kind == "ok"]
    c_errs = [r for kind, r in close_results if kind == "err"]
    assert len(c_oks) == 1 and len(c_errs) == 1
    assert "already closed" in str(c_errs[0])
    assert rep_path.is_file()

    lines = [json.loads(l) for l in (state_dir / "decisions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    outcomes = [o for o in lines if o.get("type") == "outcome" and o.get("decision_id") == winner["decision_id"]]
    assert len(outcomes) == 1


def test_open_seat_routing_summary_keeps_worker_task_intact(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    wt_root = tmp_path / "worktrees"
    state_dir = tmp_path / "state"

    dry = kza_seat.open_seat(
        alias="kza-qa-summary",
        role="kza-qa",
        task="Full worker instructions with /Users/gabriel/secret/path and SHA 1234567890abcdef1234567890abcdef12345678",
        routing_summary="Audit NEXO privacy boundaries across access and capture modules",
        exclude_families=["google"],
        offline=True,
        allow_quota_emergency=True,
        repo_root=repo,
        worktree_root=wt_root,
        state_dir=state_dir,
        dry_run=True,
    )
    assert dry["decision"]["sanitized_summary"] == "Audit NEXO privacy boundaries across access and capture modules"
