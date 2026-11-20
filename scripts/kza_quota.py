#!/usr/bin/env python3
"""Real quota telemetry and weighting for KZA's 4 subscriptions.

Collects usage data for `claude`, `codex`, `muse`, and `agy` using zero-token
local probes/snapshots, applies progressive penalty from 70% usage, enforces
hard exclusions (>=92% 5h, >=95% 7d, or active throttle), and flags
`quota_emergency` fail-closed when no subscription is eligible.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

DEFAULT_STATE_DIR = Path.home() / ".local" / "state" / "kza-router"
DEFAULT_CLAUDE_TAP_DIR = Path.home() / ".local" / "state" / "thermouy-router"
DEFAULT_CODEX_SESSIONS_DIR = Path.home() / ".codex" / "sessions"

PENALTY_START_PCT = 70.0
HARD_EXCLUDE_5H_PCT = 92.0
HARD_EXCLUDE_7D_PCT = 95.0

AGY_BAND_WEIGHTS: dict[str, float] = {
    "high": 1.0,
    "medium": 0.85,
    "low": 0.50,
    "throttled": 0.0,
}

RUNTIMES: tuple[str, ...] = ("claude", "codex", "muse", "agy")


@dataclass
class SubscriptionQuota:
    """Normalized quota snapshot and routing weight for one subscription."""

    runtime: str
    status: str  # ok | degraded | excluded | throttled | probe_error | unknown
    weight: float
    excluded: bool
    exclude_reason: str | None = None
    five_hour_pct: float | None = None
    five_hour_resets_at: float | None = None
    seven_day_pct: float | None = None
    seven_day_resets_at: float | None = None
    throttled_until: float | None = None
    throttle_reason: str | None = None
    source: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize snapshot to JSON-compatible dictionary."""
        return asdict(self)


def _ensure_private_dir(path: Path) -> None:
    """Create directory with owner-only permissions (0700)."""
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Atomically write JSON with 0600 permissions."""
    _ensure_private_dir(path.parent)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        os.chmod(tmp_path, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, path)
        os.chmod(path, 0o600)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _parse_timestamp(value: Any) -> float | None:
    """Parse unix seconds, unix milliseconds, numeric strings, or ISO-8601 timestamps."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        ts = float(value)
        return ts / 1000.0 if ts > 1e12 else ts
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            ts = float(raw)
            return ts / 1000.0 if ts > 1e12 else ts
        except ValueError:
            pass
        try:
            iso = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
            return datetime.fromisoformat(iso).timestamp()
        except ValueError:
            return None
    return None


def _to_float(value: Any) -> float | None:
    """Convert numeric or numeric-string value to float safely."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _effective_pct(used_pct: float | None, resets_at: float | None, now: float) -> float | None:
    """Return 0.0 when a window has already reset; otherwise clamp to [0, 100]."""
    if used_pct is None:
        return None
    if resets_at is not None and now >= resets_at:
        return 0.0
    return max(0.0, min(100.0, used_pct))


def compute_window_weight(
    five_hour_pct: float | None,
    seven_day_pct: float | None,
) -> tuple[float, bool, str | None]:
    """Compute progressive penalty weight and hard exclusion from 5h/7d usage.

    Policy:
    - Hard exclusion at >=92% (5h) or >=95% (7d).
    - Progressive penalty from 70% up to the hard threshold (weight decays from 1.0 to 0.15).
    - Below 70% -> weight 1.0.
    """
    if five_hour_pct is not None and five_hour_pct >= HARD_EXCLUDE_5H_PCT:
        return 0.0, True, f"5h usage {five_hour_pct:.1f}% >= {HARD_EXCLUDE_5H_PCT:.0f}%"
    if seven_day_pct is not None and seven_day_pct >= HARD_EXCLUDE_7D_PCT:
        return 0.0, True, f"7d usage {seven_day_pct:.1f}% >= {HARD_EXCLUDE_7D_PCT:.0f}%"

    def _curve(pct: float | None, limit: float) -> float:
        if pct is None or pct < PENALTY_START_PCT:
            return 1.0
        span = max(1.0, limit - PENALTY_START_PCT)
        ratio = min(1.0, (pct - PENALTY_START_PCT) / span)
        # Smooth progressive decay from 1.0 at 70% down to 0.15 just below limit.
        return round(max(0.15, 1.0 - 0.85 * (ratio**1.25)), 4)

    w5 = _curve(five_hour_pct, HARD_EXCLUDE_5H_PCT)
    w7 = _curve(seven_day_pct, HARD_EXCLUDE_7D_PCT)
    return min(w5, w7), False, None


def _load_throttles(state_dir: Path, now: float) -> dict[str, dict[str, Any]]:
    """Load active (non-expired) throttle entries from state_dir/throttles.json."""
    path = state_dir / "throttles.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    active: dict[str, dict[str, Any]] = {}
    for runtime, entry in data.items():
        if not isinstance(entry, dict):
            continue
        expires_at = _parse_timestamp(entry.get("expires_at"))
        if expires_at is not None and expires_at > now:
            active[runtime] = {
                "expires_at": expires_at,
                "reason": str(entry.get("reason") or "manual/detected throttle"),
                "recorded_at": entry.get("recorded_at"),
            }
    return active


def record_throttle(
    runtime: str,
    ttl_minutes: float = 60.0,
    reason: str = "rate_limit",
    state_dir: Path | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Record a throttle TTL for a runtime in state_dir/throttles.json."""
    if runtime not in RUNTIMES:
        raise ValueError(f"Unknown runtime '{runtime}'. Expected one of {RUNTIMES}.")
    if ttl_minutes <= 0:
        raise ValueError("ttl_minutes must be positive.")
    sdir = state_dir or _default_state_dir()
    ts = now if now is not None else time.time()
    active = _load_throttles(sdir, ts)
    entry = {
        "recorded_at": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
        "expires_at": ts + ttl_minutes * 60.0,
        "reason": reason,
    }
    active[runtime] = entry
    _atomic_write_json(sdir / "throttles.json", active)
    return entry


def clear_throttle(runtime: str, state_dir: Path | None = None) -> bool:
    """Clear an active throttle for a runtime."""
    if runtime not in RUNTIMES:
        raise ValueError(f"Unknown runtime '{runtime}'. Expected one of {RUNTIMES}.")
    sdir = state_dir or _default_state_dir()
    active = _load_throttles(sdir, time.time())
    existed = runtime in active
    active.pop(runtime, None)
    _atomic_write_json(sdir / "throttles.json", active)
    return existed


def set_agy_band(band: str, state_dir: Path | None = None) -> dict[str, Any]:
    """Persist configured Agy quota band (high|medium|low|throttled)."""
    norm = band.strip().lower()
    if norm not in AGY_BAND_WEIGHTS:
        raise ValueError(f"Invalid agy band '{band}'. Expected one of {tuple(AGY_BAND_WEIGHTS)}.")
    sdir = state_dir or _default_state_dir()
    payload = {
        "band": norm,
        "updated_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    _atomic_write_json(sdir / "agy_band.json", payload)
    return payload


def _read_agy_band(state_dir: Path) -> str:
    """Read Agy band from env KZA_AGY_BAND or persisted state (default medium)."""
    env_band = os.environ.get("KZA_AGY_BAND", "").strip().lower()
    if env_band in AGY_BAND_WEIGHTS:
        return env_band
    path = state_dir / "agy_band.json"
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            band = str(data.get("band", "")).strip().lower()
            if band in AGY_BAND_WEIGHTS:
                return band
        except (OSError, json.JSONDecodeError):
            pass
    return "medium"


def _apply_throttle(quota: SubscriptionQuota, throttle: dict[str, Any] | None) -> SubscriptionQuota:
    """Override a SubscriptionQuota to excluded when a throttle TTL is active."""
    if not throttle:
        return quota
    quota.status = "throttled"
    quota.weight = 0.0
    quota.excluded = True
    quota.throttled_until = throttle["expires_at"]
    quota.throttle_reason = throttle["reason"]
    quota.exclude_reason = f"active throttle: {throttle['reason']}"
    return quota


def probe_claude(
    tap_dir: Path | None = None,
    throttle: dict[str, Any] | None = None,
    now: float | None = None,
) -> SubscriptionQuota:
    """Read atomic 0600 snapshots written by ~/.claude/statusline.sh (multi-account)."""
    ts = now if now is not None else time.time()
    tdir = tap_dir or Path(os.environ.get("KZA_CLAUDE_TAP_DIR", str(DEFAULT_CLAUDE_TAP_DIR)))
    files = sorted(tdir.glob("quota-*.json")) if tdir.is_dir() else []

    accounts: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    for fp in files:
        try:
            raw = json.loads(fp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(raw, dict):
            continue

        five_raw = raw.get("five_hour") if isinstance(raw.get("five_hour"), dict) else {}
        seven_raw = raw.get("seven_day") if isinstance(raw.get("seven_day"), dict) else {}

        r5 = _parse_timestamp(five_raw.get("resets_at"))
        r7 = _parse_timestamp(seven_raw.get("resets_at"))
        p5 = _effective_pct(_to_float(five_raw.get("used_percentage")), r5, ts)
        p7 = _effective_pct(_to_float(seven_raw.get("used_percentage")), r7, ts)
        weight, excluded, reason = compute_window_weight(p5, p7)

        acct_entry = {
            "account": str(raw.get("account") or fp.stem.removeprefix("quota-")),
            "file": str(fp),
            "timestamp": raw.get("timestamp"),
            "model": raw.get("model"),
            "five_hour_pct": p5,
            "five_hour_resets_at": r5,
            "seven_day_pct": p7,
            "seven_day_resets_at": r7,
            "weight": weight,
            "excluded": excluded,
            "exclude_reason": reason,
        }
        accounts.append(acct_entry)

        if best is None:
            best = acct_entry
        else:
            # Prefer non-excluded, then higher weight, then lower peak usage.
            best_key = (
                0 if best["excluded"] else 1,
                best["weight"],
                -max(best["five_hour_pct"] or 0.0, best["seven_day_pct"] or 0.0),
            )
            cur_key = (
                0 if acct_entry["excluded"] else 1,
                acct_entry["weight"],
                -max(acct_entry["five_hour_pct"] or 0.0, acct_entry["seven_day_pct"] or 0.0),
            )
            if cur_key > best_key:
                best = acct_entry

    if best is None:
        q = SubscriptionQuota(
            runtime="claude",
            status="unknown",
            weight=0.75,
            excluded=False,
            source=f"claude-tap-missing:{tdir}",
            details={"accounts": []},
        )
        return _apply_throttle(q, throttle)

    status = "excluded" if best["excluded"] else ("degraded" if best["weight"] < 1.0 else "ok")
    q = SubscriptionQuota(
        runtime="claude",
        status=status,
        weight=best["weight"],
        excluded=best["excluded"],
        exclude_reason=best["exclude_reason"],
        five_hour_pct=best["five_hour_pct"],
        five_hour_resets_at=best["five_hour_resets_at"],
        seven_day_pct=best["seven_day_pct"],
        seven_day_resets_at=best["seven_day_resets_at"],
        source=f"claude-statusline:{best['account']}",
        details={"selected_account": best["account"], "accounts": accounts},
    )
    return _apply_throttle(q, throttle)


def _extract_codex_windows(rl: dict[str, Any], now: float) -> tuple[float | None, float | None, float | None, float | None]:
    """Extract (5h_pct, 5h_reset, 7d_pct, 7d_reset) from a Codex rate_limits object."""
    p5: float | None = None
    r5: float | None = None
    p7: float | None = None
    r7: float | None = None

    for key in ("primary", "secondary"):
        win = rl.get(key)
        if not isinstance(win, dict):
            continue
        used = _to_float(win.get("used_percent"))
        if used is None:
            used = _to_float(win.get("used_percentage"))
        if used is None:
            continue
        resets_at = _parse_timestamp(win.get("resets_at"))
        eff = _effective_pct(used, resets_at, now)
        minutes = _to_float(win.get("window_minutes"))
        if minutes is not None and minutes <= 600:
            p5, r5 = eff, resets_at
        elif minutes is not None and minutes >= 1440:
            p7, r7 = eff, resets_at
        elif key == "primary":
            p5, r5 = eff, resets_at
        else:
            p7, r7 = eff, resets_at

    return p5, r5, p7, r7


def probe_codex(
    sessions_dir: Path | None = None,
    throttle: dict[str, Any] | None = None,
    now: float | None = None,
    max_files: int = 80,
) -> SubscriptionQuota:
    """Read latest non-null `rate_limits` event from ~/.codex/sessions/**/*.jsonl."""
    ts = now if now is not None else time.time()
    sdir = sessions_dir or Path(os.environ.get("KZA_CODEX_SESSIONS_DIR", str(DEFAULT_CODEX_SESSIONS_DIR)))

    latest_event: tuple[float, Path, dict[str, Any]] | None = None
    if sdir.is_dir():
        candidates = sorted(
            sdir.rglob("*.jsonl"),
            key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
            reverse=True,
        )[:max_files]

        for fp in candidates:
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if "rate_limits" not in text:
                continue
            file_best: tuple[float, dict[str, Any]] | None = None
            for line in text.splitlines():
                if "rate_limits" not in line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                payload = obj.get("payload") if isinstance(obj.get("payload"), dict) else obj
                rl = payload.get("rate_limits") if isinstance(payload, dict) else None
                if not isinstance(rl, dict):
                    continue
                if not isinstance(rl.get("primary"), dict) and not isinstance(rl.get("secondary"), dict):
                    continue
                ev_ts = _parse_timestamp(obj.get("timestamp")) or fp.stat().st_mtime
                if file_best is None or ev_ts >= file_best[0]:
                    file_best = (ev_ts, rl)
            if file_best is not None:
                if latest_event is None or file_best[0] > latest_event[0]:
                    latest_event = (file_best[0], fp, file_best[1])
                # Since files are sorted by mtime desc, once we find a populated rate_limits
                # event newer than older files' mtimes we can stop early.
                break

    if latest_event is None:
        q = SubscriptionQuota(
            runtime="codex",
            status="unknown",
            weight=0.80,
            excluded=False,
            source=f"codex-sessions-no-rate-limits:{sdir}",
        )
        return _apply_throttle(q, throttle)

    ev_ts, fp, rl = latest_event
    p5, r5, p7, r7 = _extract_codex_windows(rl, ts)
    weight, excluded, reason = compute_window_weight(p5, p7)
    status = "excluded" if excluded else ("degraded" if weight < 1.0 else "ok")
    q = SubscriptionQuota(
        runtime="codex",
        status=status,
        weight=weight,
        excluded=excluded,
        exclude_reason=reason,
        five_hour_pct=p5,
        five_hour_resets_at=r5,
        seven_day_pct=p7,
        seven_day_resets_at=r7,
        source=f"codex-session:{fp.name}",
        details={
            "event_timestamp": ev_ts,
            "limit_id": rl.get("limit_id"),
            "plan_type": rl.get("plan_type"),
        },
    )
    return _apply_throttle(q, throttle)


def _default_muse_runner() -> subprocess.CompletedProcess[str]:
    """Execute zero-token `muse usage --json` probe."""
    return subprocess.run(
        ["muse", "usage", "--json"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def probe_muse(
    runner: Callable[[], subprocess.CompletedProcess[str]] | None = None,
    throttle: dict[str, Any] | None = None,
    now: float | None = None,
) -> SubscriptionQuota:
    """Probe Muse via `muse usage --json` (supports fresh window & live mint camelCase)."""
    ts = now if now is not None else time.time()
    exec_fn = runner or _default_muse_runner
    try:
        proc = exec_fn()
    except (OSError, subprocess.SubprocessError) as exc:
        q = SubscriptionQuota(
            runtime="muse",
            status="probe_error",
            weight=0.0,
            excluded=True,
            exclude_reason=f"muse usage probe error: {exc}",
            source="muse-usage:error",
        )
        return _apply_throttle(q, throttle)

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()[:200]
        q = SubscriptionQuota(
            runtime="muse",
            status="probe_error",
            weight=0.0,
            excluded=True,
            exclude_reason=f"muse usage exited {proc.returncode}: {err}",
            source="muse-usage:nonzero",
        )
        return _apply_throttle(q, throttle)

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        q = SubscriptionQuota(
            runtime="muse",
            status="probe_error",
            weight=0.0,
            excluded=True,
            exclude_reason=f"muse usage invalid JSON: {exc}",
            source="muse-usage:invalid-json",
        )
        return _apply_throttle(q, throttle)

    if not isinstance(data, dict):
        q = SubscriptionQuota(
            runtime="muse",
            status="probe_error",
            weight=0.0,
            excluded=True,
            exclude_reason="muse usage returned non-object JSON",
            source="muse-usage:invalid-shape",
        )
        return _apply_throttle(q, throttle)

    reason = data.get("reason")
    allowance = data.get("allowance") if isinstance(data.get("allowance"), dict) else None

    if reason == "server_sent_no_usage" and allowance is None:
        q = SubscriptionQuota(
            runtime="muse",
            status="ok",
            weight=1.0,
            excluded=False,
            five_hour_pct=0.0,
            seven_day_pct=0.0,
            source="muse-usage:server_sent_no_usage",
            details={"reason": reason},
        )
        return _apply_throttle(q, throttle)

    if allowance is None:
        q = SubscriptionQuota(
            runtime="muse",
            status="probe_error",
            weight=0.0,
            excluded=True,
            exclude_reason=f"muse usage missing allowance (reason={reason!r})",
            source="muse-usage:no-allowance",
        )
        return _apply_throttle(q, throttle)

    win = allowance.get("window") if isinstance(allowance.get("window"), dict) else {}
    wk = allowance.get("weekly") if isinstance(allowance.get("weekly"), dict) else {}

    r5 = _parse_timestamp(win.get("resetsAtMs") if "resetsAtMs" in win else win.get("resets_at"))
    r7 = _parse_timestamp(wk.get("resetsAtMs") if "resetsAtMs" in wk else wk.get("resets_at"))
    raw_p5 = _to_float(win.get("usedPercent") if "usedPercent" in win else win.get("used_percent"))
    raw_p7 = _to_float(wk.get("usedPercent") if "usedPercent" in wk else wk.get("used_percent"))

    p5 = _effective_pct(raw_p5, r5, ts)
    p7 = _effective_pct(raw_p7, r7, ts)
    weight, excluded, ex_reason = compute_window_weight(p5, p7)
    status = "excluded" if excluded else ("degraded" if weight < 1.0 else "ok")

    q = SubscriptionQuota(
        runtime="muse",
        status=status,
        weight=weight,
        excluded=excluded,
        exclude_reason=ex_reason,
        five_hour_pct=p5,
        five_hour_resets_at=r5,
        seven_day_pct=p7,
        seven_day_resets_at=r7,
        source=f"muse-usage:{allowance.get('source', 'live')}",
        details={
            "tier": allowance.get("tier"),
            "asOfMs": allowance.get("asOfMs"),
        },
    )
    return _apply_throttle(q, throttle)


def probe_agy(
    state_dir: Path | None = None,
    throttle: dict[str, Any] | None = None,
) -> SubscriptionQuota:
    """Return honest null/unknown percentages for Agy plus configured band weight."""
    sdir = state_dir or _default_state_dir()
    band = _read_agy_band(sdir)
    weight = AGY_BAND_WEIGHTS[band]
    excluded = band == "throttled"
    status = "throttled" if excluded else ("degraded" if weight < 1.0 else "ok")
    q = SubscriptionQuota(
        runtime="agy",
        status=status,
        weight=weight,
        excluded=excluded,
        exclude_reason="agy band set to throttled" if excluded else None,
        five_hour_pct=None,
        seven_day_pct=None,
        source=f"agy-band:{band}",
        details={"band": band, "usage_known": False},
    )
    return _apply_throttle(q, throttle)


def _default_state_dir() -> Path:
    return Path(os.environ.get("KZA_ROUTER_STATE_DIR", str(DEFAULT_STATE_DIR)))


def collect_all_quotas(
    *,
    state_dir: Path | None = None,
    claude_tap_dir: Path | None = None,
    codex_sessions_dir: Path | None = None,
    muse_runner: Callable[[], subprocess.CompletedProcess[str]] | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Collect quota telemetry across all 4 subscriptions and evaluate quota_emergency."""
    ts = now if now is not None else time.time()
    sdir = state_dir or _default_state_dir()
    throttles = _load_throttles(sdir, ts)

    subs = {
        "claude": probe_claude(tap_dir=claude_tap_dir, throttle=throttles.get("claude"), now=ts),
        "codex": probe_codex(sessions_dir=codex_sessions_dir, throttle=throttles.get("codex"), now=ts),
        "muse": probe_muse(runner=muse_runner, throttle=throttles.get("muse"), now=ts),
        "agy": probe_agy(state_dir=sdir, throttle=throttles.get("agy")),
    }
    eligible = [name for name, q in subs.items() if not q.excluded and q.weight > 0.0]
    return {
        "timestamp": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
        "policy": {
            "penalty_start_pct": PENALTY_START_PCT,
            "hard_exclude_5h_pct": HARD_EXCLUDE_5H_PCT,
            "hard_exclude_7d_pct": HARD_EXCLUDE_7D_PCT,
        },
        "quota_emergency": len(eligible) == 0,
        "eligible_runtimes": eligible,
        "subscriptions": {name: q.to_dict() for name, q in subs.items()},
    }


def _format_human(snapshot: dict[str, Any]) -> str:
    """Format quota telemetry for terminal readability."""
    lines = [
        f"KZA Quota Snapshot ({snapshot['timestamp']})",
        f"Eligible runtimes: {', '.join(snapshot['eligible_runtimes']) or 'NONE (QUOTA_EMERGENCY)'}",
    ]
    for name in RUNTIMES:
        sub = snapshot["subscriptions"][name]
        p5 = "null" if sub["five_hour_pct"] is None else f"{sub['five_hour_pct']:.1f}%"
        p7 = "null" if sub["seven_day_pct"] is None else f"{sub['seven_day_pct']:.1f}%"
        extra = f" ({sub['exclude_reason']})" if sub.get("exclude_reason") else ""
        lines.append(
            f"  - {name:6s} | status={sub['status']:11s} | weight={sub['weight']:.2f} | 5h={p5:6s} | 7d={p7:6s} | src={sub['source']}{extra}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for KZA quota telemetry."""
    parser = argparse.ArgumentParser(description="KZA 4-subscription quota telemetry.")
    parser.add_argument("--json", action="store_true", help="Output JSON snapshot.")
    parser.add_argument("--state-dir", type=Path, default=None, help="Override state dir.")
    parser.add_argument("--claude-tap-dir", type=Path, default=None, help="Override Claude tap dir.")
    parser.add_argument("--codex-sessions-dir", type=Path, default=None, help="Override Codex sessions dir.")
    parser.add_argument(
        "--record-throttle",
        choices=RUNTIMES,
        default=None,
        help="Record an active throttle TTL for a runtime.",
    )
    parser.add_argument("--ttl-minutes", type=float, default=60.0, help="Throttle TTL in minutes.")
    parser.add_argument("--reason", type=str, default="rate_limit", help="Throttle reason.")
    parser.add_argument(
        "--clear-throttle",
        choices=RUNTIMES,
        default=None,
        help="Clear active throttle for a runtime.",
    )
    parser.add_argument(
        "--set-agy-band",
        choices=tuple(AGY_BAND_WEIGHTS.keys()),
        default=None,
        help="Set configured Agy quota band (high|medium|low|throttled).",
    )
    args = parser.parse_args(argv)

    if args.record_throttle:
        entry = record_throttle(
            args.record_throttle,
            ttl_minutes=args.ttl_minutes,
            reason=args.reason,
            state_dir=args.state_dir,
        )
        print(json.dumps({"recorded": args.record_throttle, **entry}, indent=2))
        return 0

    if args.clear_throttle:
        cleared = clear_throttle(args.clear_throttle, state_dir=args.state_dir)
        print(json.dumps({"cleared": args.clear_throttle, "existed": cleared}, indent=2))
        return 0

    if args.set_agy_band:
        saved = set_agy_band(args.set_agy_band, state_dir=args.state_dir)
        print(json.dumps(saved, indent=2))
        return 0

    snapshot = collect_all_quotas(
        state_dir=args.state_dir,
        claude_tap_dir=args.claude_tap_dir,
        codex_sessions_dir=args.codex_sessions_dir,
    )
    if args.json:
        print(json.dumps(snapshot, indent=2))
    else:
        print(_format_human(snapshot))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
