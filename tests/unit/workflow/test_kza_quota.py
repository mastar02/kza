"""Unit tests for scripts/kza_quota.py (real file/probe telemetry & weighting)."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import kza_quota  # noqa: E402


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_weight_policy_progressive_and_hard_exclusions() -> None:
    w_zero, ex_zero, _ = kza_quota.compute_window_weight(0.0, 0.0)
    assert w_zero == 1.0
    assert ex_zero is False

    w_50, ex_50, _ = kza_quota.compute_window_weight(50.0, 20.0)
    w_70, ex_70, _ = kza_quota.compute_window_weight(70.0, 40.0)
    w_85, ex_85, _ = kza_quota.compute_window_weight(85.0, 40.0)
    assert ex_50 is False and ex_70 is False and ex_85 is False
    assert 0.85 < w_50 < 1.0
    assert w_70 == 0.85
    assert 0.15 <= w_85 < w_70

    w_5h_cut, ex_5h, reason_5h = kza_quota.compute_window_weight(92.0, 10.0)
    assert w_5h_cut == 0.0 and ex_5h is True and "5h" in (reason_5h or "")

    w_7d_cut, ex_7d, reason_7d = kza_quota.compute_window_weight(10.0, 95.0)
    assert w_7d_cut == 0.0 and ex_7d is True and "7d" in (reason_7d or "")


def test_claude_default_uses_claude1_only_and_explicit_multi_account_auto_reset(tmp_path: Path) -> None:
    now = 1_800_000_000.0
    # Account 1: 97% 7d with future reset -> excluded.
    _write_json(
        tmp_path / "quota-claude-1.json",
        {
            "account": "claude-1",
            "five_hour": {"used_percentage": 40, "resets_at": str(int(now + 3600))},
            "seven_day": {"used_percentage": 97, "resets_at": str(int(now + 7200))},
        },
    )
    # Account 2: 96% 7d BUT resets_at is in the past -> auto-resets to 0%!
    _write_json(
        tmp_path / "quota-claude-2.json",
        {
            "account": "claude-2",
            "five_hour": {"used_percentage": 80, "resets_at": str(int(now - 60))},
            "seven_day": {"used_percentage": 96, "resets_at": str(int(now - 60))},
        },
    )

    # Default KZA policy uses only claude-1 and ignores claude-2.
    q_default = kza_quota.probe_claude(tap_dir=tmp_path, now=now)
    assert q_default.excluded is True
    assert q_default.details["selected_account"] == "claude-1"
    assert len(q_default.details["accounts"]) == 1

    # Explicit multi-account override still supports auto-reset & best-account selection.
    q = kza_quota.probe_claude(tap_dir=tmp_path, now=now, allowed_accounts=("claude-1", "claude-2"))
    assert q.excluded is False
    assert q.weight == 1.0
    assert q.five_hour_pct == 0.0
    assert q.seven_day_pct == 0.0
    assert q.details["selected_account"] == "claude-2"
    assert len(q.details["accounts"]) == 2


def test_codex_reads_latest_non_null_rate_limits_and_throttle_ttl(tmp_path: Path) -> None:
    now = 1_800_000_000.0
    older = tmp_path / "2026" / "11" / "18" / "rollout-older.jsonl"
    newer = tmp_path / "2026" / "11" / "19" / "rollout-newer.jsonl"
    older.parent.mkdir(parents=True, exist_ok=True)
    newer.parent.mkdir(parents=True, exist_ok=True)

    # Older file has real codex weekly window (10080 min = 7d) at 88%.
    older.write_text(
        json.dumps(
            {
                "timestamp": now - 120,
                "payload": {
                    "rate_limits": {
                        "limit_id": "codex",
                        "primary": {"used_percent": 88.0, "window_minutes": 10080, "resets_at": now + 3600},
                        "secondary": {"used_percent": 30.0, "window_minutes": 300, "resets_at": now + 1800},
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    # Newer file has null primary/secondary (like live `limit_id: "premium"`).
    newer.write_text(
        json.dumps(
            {
                "timestamp": now - 10,
                "payload": {
                    "rate_limits": {"limit_id": "premium", "primary": None, "secondary": None}
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    q = kza_quota.probe_codex(sessions_dir=tmp_path, now=now)
    assert q.excluded is False
    assert q.five_hour_pct == 30.0
    assert q.seven_day_pct == 88.0
    assert 0.15 < q.weight < 1.0  # Progressive penalty above 70%

    # Active throttle overrides to excluded.
    state_dir = tmp_path / "state"
    kza_quota.record_throttle("codex", ttl_minutes=30, reason="429 burst", state_dir=state_dir, now=now)
    snap = kza_quota.collect_all_quotas(
        state_dir=state_dir,
        claude_tap_dir=tmp_path / "empty_claude",
        codex_sessions_dir=tmp_path,
        muse_runner=lambda: subprocess.CompletedProcess(
            ["muse", "usage", "--json"], 0, json.dumps({"reason": "server_sent_no_usage", "allowance": None}), ""
        ),
        now=now + 60,
    )
    assert snap["subscriptions"]["codex"]["excluded"] is True
    assert snap["subscriptions"]["codex"]["status"] == "throttled"


def test_muse_probe_supports_fresh_window_mint_camel_case_and_nonzero_rc() -> None:
    now = 1_795_129_500.0

    # 1. Fresh window: reason == "server_sent_no_usage"
    q_fresh = kza_quota.probe_muse(
        runner=lambda: subprocess.CompletedProcess(
            ["muse", "usage", "--json"],
            0,
            json.dumps({"schemaVersion": 1, "allowance": None, "reason": "server_sent_no_usage"}),
            "",
        ),
        now=now,
    )
    assert q_fresh.excluded is False
    assert q_fresh.weight == 1.0
    assert q_fresh.five_hour_pct == 0.0
    assert q_fresh.seven_day_pct == 0.0

    # 2. Live mint camelCase schema with resetsAtMs in milliseconds
    mint_payload = {
        "schemaVersion": 1,
        "allowance": {
            "tier": "Muse Code Max Usage",
            "window": {"usedPercent": 3, "resetsAtMs": int((now + 5000) * 1000)},
            "weekly": {"usedPercent": 90, "resetsAtMs": int((now + 86400) * 1000)},
            "asOfMs": int(now * 1000),
            "source": "mint",
        },
        "reason": None,
    }
    q_mint = kza_quota.probe_muse(
        runner=lambda: subprocess.CompletedProcess(["muse", "usage", "--json"], 0, json.dumps(mint_payload), ""),
        now=now,
    )
    assert q_mint.excluded is False
    assert q_mint.five_hour_pct == 3.0
    assert q_mint.seven_day_pct == 90.0
    assert 0.15 <= q_mint.weight < 0.6

    # 3. Non-zero returncode must be rejected fail-closed (never invented as 0%)
    q_err = kza_quota.probe_muse(
        runner=lambda: subprocess.CompletedProcess(["muse", "usage", "--json"], 1, "", "auth expired"),
        now=now,
    )
    assert q_err.excluded is True
    assert q_err.status == "probe_error"
    assert q_err.weight == 0.0


def test_agy_honest_null_percentages_and_quota_emergency(tmp_path: Path) -> None:
    now = 1_800_000_000.0
    state_dir = tmp_path / "state"
    kza_quota.set_agy_band("low", state_dir=state_dir)

    q_agy = kza_quota.probe_agy(state_dir=state_dir)
    assert q_agy.five_hour_pct is None
    assert q_agy.seven_day_pct is None
    assert q_agy.details["usage_known"] is False
    assert q_agy.weight == 0.50

    # Put all 4 subscriptions into hard exclusion -> quota_emergency must be True.
    for rt in kza_quota.RUNTIMES:
        kza_quota.record_throttle(rt, ttl_minutes=30, reason="exhausted", state_dir=state_dir, now=now)

    snap = kza_quota.collect_all_quotas(
        state_dir=state_dir,
        claude_tap_dir=tmp_path / "none",
        codex_sessions_dir=tmp_path / "none2",
        muse_runner=lambda: subprocess.CompletedProcess(["muse"], 1, "", "err"),
        now=now + 10,
    )
    assert snap["quota_emergency"] is True
    assert snap["eligible_runtimes"] == []


def test_codex_multi_account_selects_healthier_codex2_and_per_account_throttle(tmp_path: Path) -> None:
    now = 1_800_000_000.0
    dir1 = tmp_path / "codex1" / "2026" / "11" / "24"
    dir2 = tmp_path / "codex2" / "2026" / "11" / "24"
    dir1.mkdir(parents=True)
    dir2.mkdir(parents=True)

    (dir1 / "r1.jsonl").write_text(
        json.dumps(
            {
                "timestamp": now - 30,
                "payload": {
                    "rate_limits": {
                        "limit_id": "codex",
                        "primary": {"used_percent": 65.0, "window_minutes": 10080, "resets_at": now + 3600},
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (dir2 / "r2.jsonl").write_text(
        json.dumps(
            {
                "timestamp": now - 60,
                "payload": {
                    "rate_limits": {
                        "limit_id": "codex",
                        "primary": {"used_percent": 0.0, "window_minutes": 10080, "resets_at": now + 7200},
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    q = kza_quota.probe_codex(sessions_dirs={"codex": tmp_path / "codex1", "codex-2": tmp_path / "codex2"}, now=now)
    assert q.excluded is False
    assert q.weight == 1.0
    assert q.seven_day_pct == 0.0
    assert q.details["selected_account"] == "codex-2"
    assert q.details["selected_cli"] == "codex-2"
    assert len(q.details["accounts"]) == 2

    # Throttling codex-2 individually fails over to primary codex.
    state_dir = tmp_path / "state"
    kza_quota.record_throttle("codex-2", ttl_minutes=30, reason="429 codex-2", state_dir=state_dir, now=now)
    snap = kza_quota.collect_all_quotas(
        state_dir=state_dir,
        claude_tap_dir=tmp_path / "none",
        codex_sessions_dirs={"codex": tmp_path / "codex1", "codex-2": tmp_path / "codex2"},
        muse_runner=lambda: subprocess.CompletedProcess(
            ["muse", "usage", "--json"], 0, json.dumps({"reason": "server_sent_no_usage", "allowance": None}), ""
        ),
        now=now + 10,
    )
    codex_sub = snap["subscriptions"]["codex"]
    assert codex_sub["excluded"] is False
    assert codex_sub["details"]["selected_account"] == "codex"
    assert codex_sub["details"]["selected_cli"] == "codex"
    assert 0.85 < codex_sub["weight"] < 0.90
