"""Unit tests for scripts/kza_safety_guard.py and scripts/kza_precompact.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import kza_precompact  # noqa: E402
import kza_safety_guard  # noqa: E402


@pytest.mark.parametrize(
    ("tool_name", "cmd"),
    [
        ("Bash", "git stash"),
        ("Bash", "git stash push -m wip"),
        ("shell_command", "git add -A"),
        ("run_shell", "git add ."),
        ("muse_shell", "command git add --all"),
        ("Bash", "echo ok\ngit push origin main"),
        ("Bash", "bash -c 'git push origin HEAD:main'"),
        ("Bash", "sh -lc \"env FOO=1 git stash\""),
        ("Bash", "git reset --hard HEAD~1"),
        ("Bash", "git clean -fd"),
        ("Bash", "git add .env"),
        ("Bash", "git add .env.production"),
        ("Bash", "git add data/users.json"),
        ("Bash", "rm -rf /home/kza/app/src"),
        ("Bash", "rm -rf /home/kza"),
        ("Bash", "sudo -u kza systemctl restart kza-voice"),
        ("Bash", "rm ~/.config/herdr-core/templates/tu-qa.md"),
        ("Bash", "security add-generic-password -s kza-openai-decisions -w"),
    ],
)
def test_safety_guard_blocks_prohibited_commands(tool_name: str, cmd: str) -> None:
    res = kza_safety_guard.evaluate_hook_payload(
        {"tool_name": tool_name, "tool_input": {"command": cmd}}
    )
    assert res.allowed is False, f"Expected block for ({tool_name}, {cmd!r})"


@pytest.mark.parametrize(
    "cmd",
    [
        "git status -sb",
        "git add scripts/kza_quota.py tests-v2/workflow/test_kza_quota.py",
        "git push -u origin feat/kza-multi-console-workflow",
        "pytest tests-v2/workflow/",
        "python3 scripts/kza_seat.py sync-templates",
        "git add models/.gitkeep",
        "git add .env.example",
    ],
)
def test_safety_guard_allows_safe_commands(cmd: str) -> None:
    res = kza_safety_guard.evaluate_hook_payload(
        {"tool_name": "Bash", "tool_input": {"command": cmd}}
    )
    assert res.allowed is True, f"Unexpected block for {cmd!r}: {res.reason}"


def test_safety_guard_blocks_implicit_git_push_on_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    subprocess.run(["git", "init", "-b", "main"], cwd=str(tmp_path), check=True, capture_output=True)
    monkeypatch.chdir(tmp_path)
    for cmd in ("git push", "git push origin"):
        res = kza_safety_guard.evaluate_hook_payload({"tool_name": "Bash", "tool_input": {"command": cmd}})
        assert res.allowed is False and res.rule_id == "git-push-main-implicit"


def test_safety_guard_blocks_forbidden_write_paths_and_cli_exit_code() -> None:
    for bad_path in (
        "/home/kza/app/src/main.py",
        "/Users/gabriel/.config/herdr-core/templates/tu-dev.md",
        "/Users/gabriel/.herdr/worktrees/thermouy-backend/foo.py",
        "/Users/gabriel/Documents/Proyectos/kza/.env",
    ):
        res = kza_safety_guard.evaluate_hook_payload(
            {"tool_name": "Edit", "tool_input": {"file_path": bad_path}}
        )
        assert res.allowed is False

    # End-to-end CLI invocation via stdin returns exit 2 + JSON deny payload.
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "kza_safety_guard.py")],
        input=json.dumps({"tool_name": "Bash", "tool_input": {"command": "git stash"}}),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    out = json.loads(proc.stdout)
    assert out["decision"] == "block"
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_precompact_emits_structured_recovery_context(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-b", "feat/demo"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@kza.local"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=str(tmp_path), check=True)
    (tmp_path / "a.txt").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "add", "a.txt"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(tmp_path), check=True, capture_output=True)

    summary = kza_precompact.build_recovery_summary(
        cwd=tmp_path,
        env={
            "HC_TERMINAL_ID": "t6k",
            "KZA_SEAT_ALIAS": "kza-qa-demo",
            "KZA_SEAT_ROLE": "kza-qa",
            "KZA_REPORT_PATH": "/tmp/report.md",
        },
    )
    assert "feat/demo" in summary
    assert "kza-qa-demo" in summary
    assert "/tmp/report.md" in summary
    assert "no push a main" in summary
