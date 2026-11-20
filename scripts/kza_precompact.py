#!/usr/bin/env python3
"""PreCompact state-recovery hook for KZA multi-console sessions.

Emits a concise recovery snapshot (branch, HEAD, dirty summary, seat metadata,
and core KZA invariants) so context compaction never loses active worktree or
reporting obligations.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def _git(args: list[str], cwd: Path) -> str:
    """Run a read-only git command returning stripped stdout or empty string."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return proc.stdout.strip() if proc.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def build_recovery_summary(cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
    """Build a compact markdown state recovery snapshot for PreCompact."""
    work_dir = cwd or Path.cwd()
    env_map = env if env is not None else dict(os.environ)

    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], work_dir) or "unknown"
    head = _git(["rev-parse", "--short", "HEAD"], work_dir) or "unknown"
    status_lines = [
        line
        for line in _git(["status", "--porcelain=v1"], work_dir).splitlines()
        if line.strip()
    ]
    tracked_dirty = [ln for ln in status_lines if not ln.startswith("?? ")]
    untracked = [ln for ln in status_lines if ln.startswith("?? ")]

    seat_alias = env_map.get("KZA_SEAT_ALIAS", "").strip()
    seat_role = env_map.get("KZA_SEAT_ROLE", "").strip()
    report_path = env_map.get("KZA_REPORT_PATH", "").strip()
    parent_tid = env_map.get("KZA_PARENT_TERMINAL_ID", "").strip()
    term_id = env_map.get("HC_TERMINAL_ID", "").strip()

    lines = [
        "[KZA PreCompact Recovery]",
        f"- cwd: {work_dir}",
        f"- git: branch={branch} head={head} tracked_dirty={len(tracked_dirty)} untracked={len(untracked)}",
    ]
    if tracked_dirty:
        preview = ", ".join(ln[3:] for ln in tracked_dirty[:8])
        suffix = f" (+{len(tracked_dirty) - 8} more)" if len(tracked_dirty) > 8 else ""
        lines.append(f"- modified: {preview}{suffix}")
    if seat_alias or seat_role or term_id:
        lines.append(
            f"- seat: alias={seat_alias or 'orchestrator'} role={seat_role or 'orchestrator'} terminal={term_id or 'none'} parent={parent_tid or 'none'}"
        )
    if report_path:
        lines.append(f"- report_path: {report_path} (escribir reporte completo y avisar <=6 líneas vía `hc send`)")
    lines.append(
        "- invariantes: no push a main; no tocar /home/kza/* ni prod; no git stash/add -A; no tocar templates/worktrees ajenos (tu-*/gsm-*)."
    )
    return "\n".join(lines)


def main() -> int:
    """Read optional hook JSON payload from stdin and emit PreCompact context."""
    cwd: Path | None = None
    raw = sys.stdin.read()
    if raw.strip():
        try:
            payload: dict[str, Any] = json.loads(raw)
            if isinstance(payload, dict) and isinstance(payload.get("cwd"), str) and payload["cwd"]:
                cwd = Path(payload["cwd"])
        except json.JSONDecodeError:
            pass

    summary = build_recovery_summary(cwd=cwd)
    output = {
        "hookSpecificOutput": {
            "hookEventName": "PreCompact",
            "additionalContext": summary,
        }
    }
    sys.stdout.write(json.dumps(output, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
