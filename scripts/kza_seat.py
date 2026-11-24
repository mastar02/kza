#!/usr/bin/env python3
"""Isolated worktree + `herdr-app`/`hc` seat controller for KZA.

Subcommands:
- `open`: route dynamically (or accept explicit candidate), create isolated
  worktree under `~/.herdr/worktrees/kza/<alias>`, enforce `dirty=0` on
  detached worktrees, launch via `hc run --bg`, rename tab in `herdr-app`
  without stealing focus, and rollback automatically if `hc run` fails.
- `status`: display active KZA seats with live `hc` state and report status.
- `close`: enforce safety guards (block closing `working`/`blocked` seats
  without `--abandon`, detect unexpected dirt outside `ensure_manifest`
  including `.serena/memories/*`, require `--merged` [with `merge-base
  --is-ancestor` check] or `--merged-squash` for write roles, and
  `--done-readonly` for read-only roles).
- `sync-templates`: symlink `herdr/templates/kza-*.md` into
  `~/.config/herdr-core/templates/`, refusing to overwrite non-symlinks
  without `--force` and never touching templates of other projects.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shlex
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import kza_quota  # noqa: E402
import kza_router  # noqa: E402

PROJECT_SLUG = "kza"
TEMPLATE_PREFIX = "kza-"
DEFAULT_SPACE_NAME = "kza"
DEFAULT_SPACE_ID_FALLBACK = "spd6ecf159033f133cd717f40ab9f9dfdb"

DEFAULT_WORKTREE_ROOT = Path.home() / ".herdr" / "worktrees" / PROJECT_SLUG
DEFAULT_STATE_DIR = kza_quota.DEFAULT_STATE_DIR
DEFAULT_HERDR_TEMPLATES_DIR = Path.home() / ".config" / "herdr-core" / "templates"

ALIAS_RE = re.compile(r"^kza-[a-z0-9][a-z0-9_-]{1,27}$")
FORBIDDEN_PREFIXES = ("tu-", "gsm-", "hl-", "sm-")
CROSS_MODEL_GATE_ROLES: frozenset[str] = frozenset({"kza-qa", "kza-privacy"})

# Files expected to exist/appear transiently in a worktree without counting as foreign dirt.
DEFAULT_ENSURE_MANIFEST: frozenset[str] = frozenset(
    {
        ".claude/scheduled_tasks.lock",
        ".DS_Store",
    }
)


def _repo_root(cwd: Path | None = None) -> Path:
    """Resolve the git top-level directory for KZA."""
    proc = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=str(cwd or Path.cwd()),
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(proc.stdout.strip()).resolve()


def _clean_launch_env() -> dict[str, str]:
    """Return process environment stripped of inherited profile overrides (`CODEX_HOME`, `CLAUDE_CONFIG_DIR`)."""
    cleaned = dict(os.environ)
    cleaned.pop("CODEX_HOME", None)
    cleaned.pop("CLAUDE_CONFIG_DIR", None)
    return cleaned


def _run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    timeout: float = 30.0,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run subprocess command capturing stdout/stderr."""
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=check,
        env=env,
    )


def validate_alias(alias: str) -> str:
    """Validate KZA seat alias (must start with `kza-`, never another project's prefix)."""
    if any(alias.startswith(pfx) for pfx in FORBIDDEN_PREFIXES):
        raise ValueError(f"Forbidden cross-project alias '{alias}'.")
    if not ALIAS_RE.match(alias):
        raise ValueError(
            f"Invalid alias '{alias}'. Must match {ALIAS_RE.pattern} (lowercase, starting with 'kza-')."
        )
    return alias


def _seats_file(state_dir: Path) -> Path:
    return state_dir / "seats.json"


def _load_seats(state_dir: Path) -> dict[str, dict[str, Any]]:
    path = _seats_file(state_dir)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_seats(state_dir: Path, seats: dict[str, dict[str, Any]]) -> None:
    kza_quota._atomic_write_json(_seats_file(state_dir), seats)


@contextmanager
def _locked_seats_txn(state_dir: Path) -> Iterator[dict[str, dict[str, Any]]]:
    """Acquire an exclusive flock on `<state_dir>/seats.lock`, reload seats, and atomically save on exit."""
    kza_quota._ensure_private_dir(state_dir)
    lock_path = state_dir / "seats.lock"
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        seats = _load_seats(state_dir)
        yield seats
        _save_seats(state_dir, seats)
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def resolve_herdr_space_id(space_name: str = DEFAULT_SPACE_NAME) -> str:
    """Resolve `herdr-app` space ID by name without disturbing other spaces."""
    env_space = os.environ.get("KZA_HERDR_SPACE_ID", "").strip()
    if env_space:
        return env_space
    try:
        proc = _run(["herdr-app", "ls"], check=False, timeout=5.0)
        if proc.returncode == 0 and proc.stdout.strip():
            data = json.loads(proc.stdout)
            for sp in data.get("spaces", []):
                if sp.get("name") == space_name and sp.get("id"):
                    return str(sp["id"])
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        pass
    return DEFAULT_SPACE_ID_FALLBACK


def _hc_get_terminal(terminal_id: str) -> dict[str, Any] | None:
    """Query `hc get <tid> --json` safely."""
    try:
        proc = _run(["hc", "get", terminal_id, "--json"], check=False, timeout=5.0)
        if proc.returncode == 0 and proc.stdout.strip():
            return json.loads(proc.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        pass
    return None


def _rename_herdr_tab_no_focus(terminal_id: str, alias: str, wait_sec: float = 2.5) -> str | None:
    """Rename the herdr-app tab hosting `terminal_id` to `alias` without stealing focus."""
    deadline = time.time() + wait_sec
    tab_id: str | None = None
    while time.time() < deadline:
        info = _hc_get_terminal(terminal_id)
        if info:
            labels = info.get("labels") or {}
            tid = labels.get("herdr.tab")
            if isinstance(tid, str) and tid:
                tab_id = tid
                break
        # Fallback: scan `herdr-app ls`
        try:
            proc = _run(["herdr-app", "ls"], check=False, timeout=5.0)
            if proc.returncode == 0 and proc.stdout.strip():
                ls_data = json.loads(proc.stdout)
                for sp in ls_data.get("spaces", []):
                    for tab in sp.get("tabs", []):
                        for term in tab.get("terminals", []):
                            if term.get("terminal") == terminal_id and tab.get("id"):
                                tab_id = str(tab["id"])
                                break
                        if tab_id:
                            break
                    if tab_id:
                        break
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
            pass
        if tab_id:
            break
        time.sleep(0.25)

    if not tab_id:
        return None
    _run(["herdr-app", "do", "tab.rename", "--id", tab_id, "--name", alias], check=False, timeout=5.0)
    return tab_id


def check_worktree_dirt(
    worktree_path: Path,
    ensure_manifest: set[str] | frozenset[str] | None = None,
) -> list[str]:
    """Return list of dirty/untracked paths outside `ensure_manifest` (includes `.serena/memories/*`)."""
    allowed = set(DEFAULT_ENSURE_MANIFEST)
    if ensure_manifest:
        allowed.update(ensure_manifest)

    proc = _run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=worktree_path,
        check=True,
    )
    foreign: list[str] = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        # Porcelain v1 format: XY <path> (or XY <old> -> <new>)
        rel = line[3:].strip()
        if " -> " in rel:
            rel = rel.split(" -> ", 1)[1].strip()
        rel = rel.strip('"')
        if rel not in allowed:
            foreign.append(rel)
    return foreign


def _read_template_body(repo_root: Path, role: str) -> str:
    """Read role template markdown body (stripping YAML frontmatter if present)."""
    tpath = repo_root / "herdr" / "templates" / f"{role}.md"
    if not tpath.is_file():
        return ""
    text = tpath.read_text(encoding="utf-8")
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            return text[end + 5 :].strip()
    return text.strip()


def build_worker_prompt(
    *,
    alias: str,
    role: str,
    runtime: str,
    worktree_path: Path,
    head_sha: str,
    report_path: Path,
    parent_terminal_id: str,
    task: str,
    template_body: str = "",
) -> str:
    """Build the self-contained launch prompt for a KZA worker seat."""
    read_only = role in kza_router.READ_ONLY_ROLES

    header = (
        f"[KZA SEAT {alias} | role={role} | runtime={runtime} | mode={'read-only' if read_only else 'read-write'}]\n"
        f"- Worktree: {worktree_path}\n"
        f"- Exact HEAD SHA: {head_sha}\n"
        f"- Report path: {report_path}\n"
        f"- Parent terminal: {parent_terminal_id or 'NONE'}\n"
        f"- Protocolo de cierre obligatorio:\n"
        f"  1. Escribí tu reporte completo UNA SOLA VEZ en `{report_path}`.\n"
        f"  2. Notificá al padre con un sobre de <= 6 líneas vía:\n"
        f"     `hc send {parent_terminal_id or '$HC_TERMINAL_ID'} \"<sobre <= 6 líneas con HEAD, veredicto/estado y ruta de reporte>\"`\n"
        f"  3. Nunca hagas `git stash`, `git add -A`, push a `main` ni toques `/home/kza/*` o templates/worktrees ajenos.\n"
    )
    parts = [header]
    if template_body:
        parts.append("## Guía del rol\n" + template_body)
    parts.append("## Tarea asignada\n" + task.strip())
    return "\n\n".join(parts) + "\n"


def open_seat(
    *,
    alias: str,
    role: str,
    task: str,
    ref: str = "HEAD",
    branch: str | None = None,
    complexity: str = "medium",
    exclude_families: list[str] | None = None,
    allow_same_family: bool = False,
    allow_quota_emergency: bool = False,
    offline: bool = False,
    space_id: str | None = None,
    parent_terminal_id: str | None = None,
    repo_root: Path | None = None,
    worktree_root: Path | None = None,
    state_dir: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Open an isolated worktree + `hc` seat for a KZA specialist."""
    validate_alias(alias)
    if role not in kza_router.ALL_ROLES:
        raise ValueError(f"Unknown role '{role}'. Expected one of {sorted(kza_router.ALL_ROLES)}.")
    norm_excluded = sorted({f.strip().lower() for f in (exclude_families or []) if isinstance(f, str) and f.strip()})
    if role in CROSS_MODEL_GATE_ROLES and not norm_excluded and not allow_same_family:
        raise ValueError(
            f"Role '{role}' requires at least one --exclude-family for strict cross-model auditing "
            "(or pass --allow-same-family explicitly)."
        )

    rroot = repo_root or _repo_root()
    wroot = worktree_root or Path(os.environ.get("KZA_WORKTREE_ROOT", str(DEFAULT_WORKTREE_ROOT)))
    sdir = state_dir or kza_router._state_dir()
    reports_dir = sdir / "reports"
    kza_quota._ensure_private_dir(reports_dir)

    seats = _load_seats(sdir)
    if alias in seats and seats[alias].get("status") == "open":
        raise RuntimeError(f"Seat '{alias}' is already open.")

    wt_path = wroot / alias
    if wt_path.exists():
        raise RuntimeError(f"Worktree path already exists: {wt_path}")

    read_only = role in kza_router.READ_ONLY_ROLES
    target_branch = None if read_only else (branch or f"seat/{alias}")

    decision = kza_router.route_task(
        role=role,
        summary=task,
        complexity=complexity,
        exclude_families=norm_excluded,
        allow_quota_emergency=allow_quota_emergency,
        offline=offline,
        state_dir=sdir,
        record_decision=not dry_run,
    )

    resolved_space = space_id or resolve_herdr_space_id()
    parent_tid = parent_terminal_id if parent_terminal_id is not None else os.environ.get("HC_TERMINAL_ID", "")
    report_path = reports_dir / f"{alias}.md"
    template_body = _read_template_body(rroot, role)

    if dry_run:
        return {
            "dry_run": True,
            "alias": alias,
            "role": role,
            "read_only": read_only,
            "worktree": str(wt_path),
            "branch": target_branch,
            "ref": ref,
            "space_id": resolved_space,
            "report_path": str(report_path),
            "decision": decision,
        }

    # Archive any leftover report for a reused alias so close_seat never accepts stale output.
    if report_path.exists():
        archive_dir = reports_dir / "archive"
        kza_quota._ensure_private_dir(archive_dir)
        old_mtime = int(report_path.stat().st_mtime)
        archived_path = archive_dir / f"{alias}-{old_mtime}-{uuid.uuid4().hex[:6]}.md"
        report_path.replace(archived_path)

    opened_epoch = time.time()
    wroot.mkdir(parents=True, exist_ok=True)
    created_worktree = False
    created_branch = False

    def _rollback() -> None:
        if created_worktree:
            _run(["git", "worktree", "remove", "--force", str(wt_path)], cwd=rroot, check=False)
        if created_branch and target_branch:
            _run(["git", "branch", "-D", target_branch], cwd=rroot, check=False)

    try:
        if read_only:
            _run(["git", "worktree", "add", "--detach", str(wt_path), ref], cwd=rroot, check=True)
            created_worktree = True
        else:
            _run(["git", "worktree", "add", "-b", target_branch, str(wt_path), ref], cwd=rroot, check=True)
            created_worktree = True
            created_branch = True

        # Guard: detached worktrees MUST start with tracked dirty=0 (never overwrite tracked files).
        if read_only:
            tracked_dirty = _run(
                ["git", "status", "--porcelain=v1", "--untracked-files=no"],
                cwd=wt_path,
                check=True,
            ).stdout.strip()
            if tracked_dirty:
                raise RuntimeError(
                    f"Refusing to launch detached seat '{alias}': tracked tree is dirty:\n{tracked_dirty}"
                )

        head_sha = _run(["git", "rev-parse", "HEAD"], cwd=wt_path, check=True).stdout.strip()
        prompt_text = build_worker_prompt(
            alias=alias,
            role=role,
            runtime=decision["runtime"],
            worktree_path=wt_path,
            head_sha=head_sha,
            report_path=report_path,
            parent_terminal_id=parent_tid,
            task=task,
            template_body=template_body,
        )

        # Build `hc run` command passing safe runtime flags + launch prompt as task argument.
        hc_cmd = [
            "hc",
            "run",
            "--bg",
            "--json",
            "--name",
            alias,
            "--cwd",
            str(wt_path),
            "--label",
            f"project={PROJECT_SLUG}",
            "--label",
            f"herdr.space={resolved_space}",
            "--label",
            f"kza.role={role}",
            "--label",
            f"kza.alias={alias}",
            "--label",
            f"kza.family={decision['family']}",
            "--label",
            f"kza.decision={decision['decision_id']}",
        ]
        if parent_tid:
            hc_cmd.extend(["--parent", parent_tid])

        # Pass runtime argv + initial prompt directly so every runtime receives its task cleanly.
        runtime_argv = list(decision["cli_command"]) + [prompt_text]
        hc_cmd.append("--")
        hc_cmd.extend(runtime_argv)

        proc = _run(hc_cmd, cwd=rroot, check=False, timeout=20.0, env=_clean_launch_env())
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(f"`hc run` failed (rc={proc.returncode}): {err}")

        hc_payload = json.loads(proc.stdout)
        terminal_id = str(hc_payload.get("address") or hc_payload.get("terminal") or hc_payload.get("id") or "")
        if not terminal_id:
            raise RuntimeError(f"`hc run` returned no terminal address: {proc.stdout[:200]}")

        tab_id = _rename_herdr_tab_no_focus(terminal_id, alias)

        seat_record = {
            "alias": alias,
            "role": role,
            "read_only": read_only,
            "status": "open",
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
            "opened_epoch": opened_epoch,
            "terminal_id": terminal_id,
            "tab_id": tab_id,
            "space_id": resolved_space,
            "parent_terminal_id": parent_tid,
            "worktree": str(wt_path),
            "branch": target_branch,
            "base_ref": ref,
            "head_sha": head_sha,
            "report_path": str(report_path),
            "decision_id": decision["decision_id"],
            "selected_candidate": decision["selected_candidate"],
            "selected_account": decision.get("selected_account"),
            "runtime": decision["runtime"],
            "model": decision["model"],
            "family": decision["family"],
            "effective_effort": decision.get("effective_effort", decision.get("effort")),
            "routing_source": decision["source"],
            "ensure_manifest": sorted(DEFAULT_ENSURE_MANIFEST),
        }
        with _locked_seats_txn(sdir) as latest_seats:
            if alias in latest_seats and latest_seats[alias].get("status") == "open":
                raise RuntimeError(f"Seat '{alias}' is already open.")
            latest_seats[alias] = seat_record
        return seat_record
    except Exception:
        _rollback()
        raise


def close_seat(
    *,
    alias: str,
    merged: bool = False,
    merged_squash: bool = False,
    done_readonly: bool = False,
    abandon: bool = False,
    allow_missing_report: bool = False,
    rounds: int | None = None,
    approved: bool | None = None,
    outcome_notes: str = "",
    record_seat_outcome: bool = True,
    base_ref: str = "HEAD",
    repo_root: Path | None = None,
    state_dir: Path | None = None,
) -> dict[str, Any]:
    """Close a KZA seat enforcing live status, report freshness, cleanliness, and merge safety guards."""
    validate_alias(alias)
    if rounds is not None and rounds < 1:
        raise ValueError("rounds must be >= 1.")
    if (approved is None) != (rounds is None):
        raise ValueError(
            "Both --approved and --rounds must be provided together when recording an outcome on close."
        )

    rroot = repo_root or _repo_root()
    sdir = state_dir or kza_router._state_dir()
    seats = _load_seats(sdir)
    if alias not in seats:
        raise RuntimeError(f"Seat '{alias}' not found in {sdir / 'seats.json'}.")

    seat = seats[alias]
    if seat.get("status") != "open":
        raise RuntimeError(f"Seat '{alias}' is already closed (status={seat.get('status')}).")

    read_only = bool(seat.get("read_only"))
    if not abandon:
        if read_only and not done_readonly:
            raise RuntimeError(
                f"Read-only seat '{alias}' requires --done-readonly (or --abandon) to close."
            )
        if not read_only and not (merged or merged_squash):
            raise RuntimeError(
                f"Write seat '{alias}' requires --merged, --merged-squash, or --abandon to close."
            )

    # Guard 1: refuse closing working/blocked seats without --abandon.
    terminal_id = str(seat.get("terminal_id") or "")
    live_info = _hc_get_terminal(terminal_id) if terminal_id else None
    if live_info and live_info.get("state") == "live":
        st_obj = live_info.get("status")
        live_state = st_obj.get("state") if isinstance(st_obj, dict) else str(st_obj or "")
        if live_state in ("working", "blocked") and not abandon:
            raise RuntimeError(
                f"Refusing to close seat '{alias}': terminal {terminal_id} is '{live_state}'. "
                "Wait for completion or pass --abandon."
            )

    # Guard 1b: require non-empty fresh report file unless --abandon or --allow-missing-report.
    if not abandon and not allow_missing_report:
        report_raw = seat.get("report_path")
        rpath = Path(report_raw) if report_raw else None
        if not rpath or not rpath.is_file() or rpath.stat().st_size == 0:
            raise RuntimeError(
                f"Refusing to close seat '{alias}': missing or empty report at '{report_raw}'. "
                "Ensure the worker wrote its report, or pass --allow-missing-report / --abandon."
            )
        min_epoch = seat.get("opened_epoch")
        if min_epoch is None:
            min_epoch = kza_quota._parse_timestamp(seat.get("created_at"))
        if min_epoch is not None and rpath.stat().st_mtime < float(min_epoch):
            raise RuntimeError(
                f"Refusing to close seat '{alias}': stale report at '{report_raw}' predates this seat launch."
            )

    wt_path = Path(seat["worktree"])
    manifest = set(seat.get("ensure_manifest") or DEFAULT_ENSURE_MANIFEST)

    # Guard 2: detect foreign dirt outside ensure_manifest (including .serena/memories/*).
    if wt_path.is_dir() and not abandon:
        foreign = check_worktree_dirt(wt_path, ensure_manifest=manifest)
        if foreign:
            preview = ", ".join(foreign[:10])
            raise RuntimeError(
                f"Refusing to close seat '{alias}': unexpected dirty/untracked files outside manifest: {preview}. "
                "Commit/clean them or pass --abandon."
            )

    # Guard 3: for write roles with --merged, verify ancestry via `git merge-base --is-ancestor`.
    branch = seat.get("branch")
    if not read_only and merged and not abandon:
        if not branch:
            raise RuntimeError(f"Write seat '{alias}' has no recorded branch to verify merge ancestry.")
        anc = _run(["git", "merge-base", "--is-ancestor", branch, base_ref], cwd=rroot, check=False)
        if anc.returncode != 0:
            raise RuntimeError(
                f"Refusing to close '{alias}' with --merged: branch '{branch}' is NOT an ancestor of '{base_ref}'. "
                "Merge first, use --merged-squash if squash-merged, or pass --abandon."
            )

    # Terminate and clean up `hc` terminal if still present.
    if terminal_id:
        _run(["hc", "kill", terminal_id], check=False, timeout=10.0)
        _run(["hc", "rm", terminal_id], check=False, timeout=10.0)

    # Remove worktree cleanly.
    if wt_path.exists():
        rm_cmd = ["git", "worktree", "remove"]
        if abandon:
            rm_cmd.append("--force")
        rm_cmd.append(str(wt_path))
        _run(rm_cmd, cwd=rroot, check=True)

    # Delete seat branch after verified merge or explicit abandon.
    if branch:
        del_flag = "-d" if (merged and not abandon) else "-D"
        _run(["git", "branch", del_flag, branch], cwd=rroot, check=False)

    decision_id = str(seat.get("decision_id") or "")
    closed_at = datetime.now(tz=timezone.utc).isoformat()
    close_mode = (
        "abandon"
        if abandon
        else ("done-readonly" if done_readonly else ("merged" if merged else "merged-squash"))
    )
    new_status = "abandoned" if abandon else "closed"

    with _locked_seats_txn(sdir) as latest_seats:
        cur = latest_seats.get(alias)
        if cur and (not decision_id or cur.get("decision_id") == decision_id):
            seat = {**cur, "status": new_status, "closed_at": closed_at, "close_mode": close_mode}
        else:
            seat["status"] = new_status
            seat["closed_at"] = closed_at
            seat["close_mode"] = close_mode
        latest_seats[alias] = seat

    if record_seat_outcome and decision_id and approved is not None:
        eff_rounds = rounds if rounds is not None else 1
        try:
            kza_router.record_outcome(
                decision_id=decision_id,
                rounds=eff_rounds,
                approved=approved,
                notes=outcome_notes or f"seat:{alias}:{close_mode}",
                state_dir=sdir,
                skip_if_exists=True,
            )
        except OSError as exc:
            with _locked_seats_txn(sdir) as latest_seats:
                cur = latest_seats.get(alias)
                if cur and cur.get("decision_id") == decision_id and cur.get("status") != "open":
                    cur["outcome_recorded"] = False
                    cur["outcome_error"] = str(exc)
                    latest_seats[alias] = cur
                    seat = dict(cur)
            raise RuntimeError(
                f"Seat '{alias}' closed cleanly, but recording outcome failed: {exc}"
            ) from exc

        with _locked_seats_txn(sdir) as latest_seats:
            cur = latest_seats.get(alias)
            if cur and cur.get("decision_id") == decision_id and cur.get("status") != "open":
                cur["outcome_recorded"] = True
                cur.pop("outcome_error", None)
                latest_seats[alias] = cur
                seat = dict(cur)

    return seat


def seat_status(*, state_dir: Path | None = None) -> list[dict[str, Any]]:
    """Return status list for tracked KZA seats enriched with live `hc` state."""
    sdir = state_dir or kza_router._state_dir()
    seats = _load_seats(sdir)
    rows: list[dict[str, Any]] = []
    for alias, seat in sorted(seats.items()):
        tid = str(seat.get("terminal_id") or "")
        live = _hc_get_terminal(tid) if (tid and seat.get("status") == "open") else None
        live_state = None
        if live:
            st = live.get("status")
            live_state = st.get("state") if isinstance(st, dict) else live.get("state")
        rpath = Path(seat["report_path"]) if seat.get("report_path") else None
        rows.append(
            {
                **seat,
                "live_hc_state": live_state,
                "report_exists": bool(rpath and rpath.is_file()),
                "report_bytes": rpath.stat().st_size if (rpath and rpath.is_file()) else 0,
            }
        )
    return rows


def sync_templates(
    *,
    repo_root: Path | None = None,
    target_dir: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Symlink `herdr/templates/kza-*.md` into `~/.config/herdr-core/templates/`.

    Safety invariants:
    - Only touches files matching `kza-*.md` (never touches `tu-*`, `gsm-*`, etc.).
    - Refuses to overwrite existing non-symlink files unless `force=True`.
    """
    rroot = repo_root or _repo_root()
    src_dir = rroot / "herdr" / "templates"
    dst_dir = target_dir or DEFAULT_HERDR_TEMPLATES_DIR
    if not src_dir.is_dir():
        raise RuntimeError(f"Source templates directory not found: {src_dir}")

    dst_dir.mkdir(parents=True, exist_ok=True)
    linked: list[str] = []
    skipped: list[str] = []

    sources = sorted(src_dir.glob(f"{TEMPLATE_PREFIX}*.md"))
    if not sources:
        raise RuntimeError(f"No {TEMPLATE_PREFIX}*.md templates found in {src_dir}")

    for src in sources:
        if not src.name.startswith(TEMPLATE_PREFIX):
            continue
        dst = dst_dir / src.name
        if dst.exists() or dst.is_symlink():
            if dst.is_symlink():
                if dst.resolve() == src.resolve():
                    skipped.append(src.name)
                    continue
                dst.unlink()
            else:
                if not force:
                    raise RuntimeError(
                        f"Refusing to overwrite non-symlink template {dst}. Pass --force to replace."
                    )
                dst.unlink()
        dst.symlink_to(src.resolve())
        linked.append(src.name)

    return {"source_dir": str(src_dir), "target_dir": str(dst_dir), "linked": linked, "unchanged": skipped}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for KZA seat controller."""
    parser = argparse.ArgumentParser(description="KZA isolated worktree and herdr-app/hc seat controller.")
    parser.add_argument("--state-dir", type=Path, default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    p_open = sub.add_parser("open", help="Open a new isolated specialist seat.")
    p_open.add_argument("--alias", required=True, help="Seat alias (must start with 'kza-').")
    p_open.add_argument("--role", required=True, choices=sorted(kza_router.ALL_ROLES))
    p_open.add_argument("--task", required=True, help="Task instructions for the worker.")
    p_open.add_argument("--ref", default="HEAD", help="Git ref/SHA to check out.")
    p_open.add_argument("--branch", default=None, help="Branch name for write roles (default seat/<alias>).")
    p_open.add_argument("--complexity", choices=("low", "medium", "high"), default="medium")
    p_open.add_argument("--exclude-family", action="append", default=[], choices=kza_router.FAMILIES)
    p_open.add_argument("--allow-same-family", action="store_true", help="Override cross-model guard on gate roles.")
    p_open.add_argument("--allow-quota-emergency", action="store_true")
    p_open.add_argument("--offline", action="store_true")
    p_open.add_argument("--space-id", default=None)
    p_open.add_argument("--dry-run", action="store_true")
    p_open.add_argument("--json", action="store_true")

    p_status = sub.add_parser("status", help="Show tracked KZA seats.")
    p_status.add_argument("--all", action="store_true", help="Include closed/abandoned seats.")
    p_status.add_argument("--json", action="store_true")

    p_close = sub.add_parser("close", help="Close a seat with mandatory safety guards.")
    p_close.add_argument("--alias", required=True)
    p_close.add_argument("--merged", action="store_true", help="Verify branch ancestry before closing.")
    p_close.add_argument("--merged-squash", action="store_true", help="Close squash-merged write seat.")
    p_close.add_argument("--done-readonly", action="store_true", help="Close completed read-only seat.")
    p_close.add_argument("--abandon", action="store_true", help="Force close even if working/blocked/dirty.")
    p_close.add_argument("--allow-missing-report", action="store_true", help="Allow closing without a non-empty report file.")
    p_close.add_argument("--rounds", type=int, default=None, help="Number of rounds before approval (requires --approved).")
    p_close.add_argument("--approved", type=kza_router._parse_bool, default=None, help="Whether seat deliverable was approved (true|false).")
    p_close.add_argument("--outcome-notes", default="", help="Optional outcome note saved to decisions.jsonl.")
    p_close.add_argument("--no-outcome", action="store_true", help="Skip recording outcome on close.")
    p_close.add_argument("--base-ref", default="HEAD", help="Base ref for --merged ancestry check.")
    p_close.add_argument("--json", action="store_true")

    p_sync = sub.add_parser("sync-templates", help="Symlink herdr/templates/kza-*.md into herdr-core.")
    p_sync.add_argument("--target-dir", type=Path, default=None)
    p_sync.add_argument("--force", action="store_true", help="Overwrite non-symlink kza-*.md files.")
    p_sync.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)

    try:
        if args.command == "open":
            res = open_seat(
                alias=args.alias,
                role=args.role,
                task=args.task,
                ref=args.ref,
                branch=args.branch,
                complexity=args.complexity,
                exclude_families=args.exclude_family,
                allow_same_family=args.allow_same_family,
                allow_quota_emergency=args.allow_quota_emergency,
                offline=args.offline,
                space_id=args.space_id,
                state_dir=args.state_dir,
                dry_run=args.dry_run,
            )
            if args.json:
                print(json.dumps(res, indent=2, ensure_ascii=False))
            else:
                print(
                    f"Opened {res['alias']} ({res['role']}) -> {res.get('selected_candidate', res.get('decision', {}).get('selected_candidate'))} "
                    f"| term={res.get('terminal_id', 'dry-run')} | tab={res.get('tab_id', 'n/a')} | wt={res['worktree']}"
                )
            return 0

        if args.command == "status":
            rows = seat_status(state_dir=args.state_dir)
            if not args.all:
                rows = [r for r in rows if r.get("status") == "open"]
            if args.json:
                print(json.dumps(rows, indent=2, ensure_ascii=False))
            else:
                if not rows:
                    print("No open KZA seats.")
                for r in rows:
                    print(
                        f"- {r['alias']:18s} | {r['role']:12s} | {r['status']:9s} | hc={str(r['live_hc_state']):8s} | "
                        f"{r['selected_candidate']} | report={'YES' if r['report_exists'] else 'NO'} ({r['report_bytes']}B)"
                    )
            return 0

        if args.command == "close":
            res = close_seat(
                alias=args.alias,
                merged=args.merged,
                merged_squash=args.merged_squash,
                done_readonly=args.done_readonly,
                abandon=args.abandon,
                allow_missing_report=args.allow_missing_report,
                rounds=args.rounds,
                approved=args.approved,
                outcome_notes=args.outcome_notes,
                record_seat_outcome=not args.no_outcome,
                base_ref=args.base_ref,
                state_dir=args.state_dir,
            )
            if args.json:
                print(json.dumps(res, indent=2, ensure_ascii=False))
            else:
                print(f"Closed {res['alias']} ({res['close_mode']}).")
            return 0

        if args.command == "sync-templates":
            res = sync_templates(target_dir=args.target_dir, force=args.force)
            if args.json:
                print(json.dumps(res, indent=2, ensure_ascii=False))
            else:
                print(
                    f"Synced KZA templates into {res['target_dir']}: "
                    f"linked={res['linked']}, unchanged={res['unchanged']}"
                )
            return 0
    except (RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        return 2

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
