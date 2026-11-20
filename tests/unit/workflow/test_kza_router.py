"""Unit tests for scripts/kza_router.py (Decisions contract, cross-model, CLI flags, KPIs)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import kza_router  # noqa: E402


def _healthy_quota_snapshot() -> dict:
    return {
        "quota_emergency": False,
        "eligible_runtimes": ["claude", "codex", "muse", "agy"],
        "subscriptions": {
            "claude": {"weight": 0.9, "excluded": False, "status": "ok"},
            "codex": {"weight": 0.95, "excluded": False, "status": "ok"},
            "muse": {"weight": 0.85, "excluded": False, "status": "ok"},
            "agy": {"weight": 0.85, "excluded": False, "status": "ok"},
        },
    }


def test_decisions_payload_contract_and_response_normalization() -> None:
    cands = list(kza_router.DEFAULT_CATALOG[:3])
    payload = kza_router.build_decisions_payload(
        role="kza-qa",
        complexity="high",
        sanitized_summary="Review async pipeline",
        candidates=cands,
        quota_weights={"claude": 0.9, "codex": 0.95, "muse": 0.85, "agy": 0.85},
        excluded_families=["google"],
    )
    assert payload["model"] == "gpt-6-luna"
    assert "input" in payload and isinstance(payload["input"], str)
    choice_q = next(q for q in payload["questions"] if q["type"] == "choice")
    assert choice_q["name"] == "selected_candidate"
    assert "instructions" in choice_q
    assert all(isinstance(ch, dict) and "value" in ch and "description" in ch for ch in choice_q["choices"])

    score_q = next(q for q in payload["questions"] if q["type"] == "score")
    assert score_q["name"] == "task_rigor"
    assert "instructions" in score_q
    for lvl in score_q["levels"]:
        assert set(lvl.keys()) == {"label", "description"}
        assert "value" not in lvl

    # Verify live list probabilities AND 0-based score normalization (score=3.25 -> 4.25).
    live_shape = {
        "model": "gpt-6-luna",
        "answers": [
            {
                "name": "selected_candidate",
                "type": "choice",
                "choice": "codex:gpt-6.2-sol",
                "probabilities": [
                    {"value": "codex:gpt-6.2-sol", "probability": 0.72},
                    {"value": "claude:claude-opus-5-5", "probability": 0.28},
                ],
            },
            {"name": "task_rigor", "type": "score", "score": 3.25},
        ],
        "usage": {
            "input_tokens": 420,
            "input_tokens_details": {"cached_tokens": 120},
            "output_tokens": 0,
        },
    }
    choice, probs, rigor, usage = kza_router._extract_decisions_answers(live_shape)
    assert choice == "codex:gpt-6.2-sol"
    assert probs == {"codex:gpt-6.2-sol": 0.72, "claude:claude-opus-5-5": 0.28}
    assert rigor == 4.25
    assert usage == {"input_tokens": 420, "cached_tokens": 120, "output_tokens": 0}

    # Verify dict probabilities fixture shape also supported.
    assert kza_router.normalize_probabilities({"a": 0.6, "b": 0.4}) == {"a": 0.6, "b": 0.4}


def test_cross_model_exclusion_by_real_family_excludes_agy_claude(tmp_path: Path) -> None:
    decision = kza_router.route_task(
        role="kza-qa",
        summary="Cross-model QA check",
        exclude_families=["anthropic"],
        offline=True,
        state_dir=tmp_path,
        quota_snapshot=_healthy_quota_snapshot(),
    )
    assert decision["family"] != "anthropic"
    for cid in decision["eligible_candidates"]:
        cand = next(c for c in kza_router.DEFAULT_CATALOG if c.candidate_id == cid)
        assert cand.family != "anthropic"
    assert "agy:claude-opus-5-5-high" not in decision["eligible_candidates"]
    assert "claude:claude-opus-5-5" not in decision["eligible_candidates"]


def test_kza_ui_is_dynamically_routed_not_hardcoded(tmp_path: Path) -> None:
    # Even when openai is excluded, kza-ui routes dynamically to another capable family.
    dec = kza_router.route_task(
        role="kza-ui",
        summary="Verify NEXO browser privacy workflow",
        exclude_families=["openai"],
        offline=True,
        state_dir=tmp_path,
        quota_snapshot=_healthy_quota_snapshot(),
    )
    assert dec["role"] == "kza-ui"
    assert dec["family"] != "openai"
    assert len(dec["eligible_candidates"]) >= 3


def test_sanitize_summary_strips_secrets_emails_lan_ips_and_private_paths() -> None:
    raw = (
        "Check HA at 192.168.1.100 with token=supersecret12345 sk-proj-abc123456789012345 "
        "email user@example.com path /home/kza/app/src/main.py and data/users.json"
    )
    cleaned = kza_router.sanitize_summary(raw)
    assert "192.168.1.100" not in cleaned
    assert "supersecret12345" not in cleaned
    assert "sk-proj-" not in cleaned
    assert "user@example.com" not in cleaned
    assert "/home/kza/app" not in cleaned
    assert "data/users.json" not in cleaned


def test_safe_cli_flags_per_runtime_read_only_vs_write() -> None:
    codex_sol = next(c for c in kza_router.DEFAULT_CATALOG if c.candidate_id == "codex:gpt-6.2-sol")
    codex_terra = next(c for c in kza_router.DEFAULT_CATALOG if c.candidate_id == "codex:gpt-6.1-terra")
    codex_astra = next(c for c in kza_router.DEFAULT_CATALOG if c.candidate_id == "codex:gpt-6.2-astra")
    by_rt = {
        "codex": codex_sol,
        "muse": next(c for c in kza_router.DEFAULT_CATALOG if c.runtime == "muse"),
        "claude": next(c for c in kza_router.DEFAULT_CATALOG if c.runtime == "claude"),
        "agy": next(c for c in kza_router.DEFAULT_CATALOG if c.runtime == "agy" and c.family == "google"),
    }

    codex_ro = kza_router.build_cli_command(codex_sol, "kza-qa")
    assert "--no-daemon" in codex_ro and "-s" in codex_ro and "read-only" in codex_ro
    assert "service_tier=fast" in codex_ro
    codex_rw = kza_router.build_cli_command(codex_sol, "kza-dev")
    assert "--no-daemon" in codex_rw and "read-only" not in codex_rw
    assert "service_tier=fast" in codex_rw

    terra_ui = kza_router.build_cli_command(codex_terra, "kza-ui")
    assert "--no-daemon" in terra_ui and "service_tier=fast" in terra_ui and "read-only" in terra_ui

    astra_pre = kza_router.build_cli_command(codex_astra, "kza-pregate")
    assert "--no-daemon" in astra_pre and "service_tier=fast" not in astra_pre

    payload = kza_router.build_decisions_payload(
        role="kza-dev",
        complexity="high",
        sanitized_summary="Implement async handler",
        candidates=[codex_sol, codex_terra, codex_astra],
        quota_weights={"codex": 1.0},
        excluded_families=[],
    )
    assert "/fast on" in payload["input"] and "service_tier=fast" in payload["input"]
    choice_map = {ch["value"]: ch["description"] for ch in payload["questions"][0]["choices"]}
    assert "service_tier=fast" in choice_map["codex:gpt-6.2-sol"]
    assert "service_tier=fast" in choice_map["codex:gpt-6.1-terra"]
    assert "service_tier=fast" not in choice_map["codex:gpt-6.2-astra"]

    muse_ro = kza_router.build_cli_command(by_rt["muse"], "kza-qa")
    assert "--contributor-mode" in muse_ro and "off" in muse_ro
    assert "--disable-request-user-input" in muse_ro and "--disable-write" in muse_ro

    claude_ro = kza_router.build_cli_command(by_rt["claude"], "kza-analista")
    assert "--setting-sources" in claude_ro and "project" in claude_ro
    assert "--permission-mode" in claude_ro and "plan" in claude_ro

    agy_ro = kza_router.build_cli_command(by_rt["agy"], "kza-privacy")
    assert "--mode" in agy_ro and "plan" in agy_ro


def test_quota_emergency_fail_closed_unless_explicit_flag(tmp_path: Path) -> None:
    exhausted = {
        "quota_emergency": True,
        "eligible_runtimes": [],
        "subscriptions": {rt: {"weight": 0.0, "excluded": True, "status": "excluded"} for rt in ("claude", "codex", "muse", "agy")},
    }
    with pytest.raises(RuntimeError, match="quota_emergency"):
        kza_router.route_task(
            role="kza-qa",
            summary="Blocked when all quotas exhausted",
            offline=True,
            state_dir=tmp_path,
            quota_snapshot=exhausted,
        )

    dec = kza_router.route_task(
        role="kza-qa",
        summary="Allowed with explicit emergency flag",
        allow_quota_emergency=True,
        offline=True,
        state_dir=tmp_path,
        quota_snapshot=exhausted,
    )
    assert dec["quota_emergency"] is True


def test_api_routing_outcome_and_kpi_report(tmp_path: Path) -> None:
    def _fake_api(payload: dict, api_key: str) -> dict:
        assert api_key == "test-key"
        choices = payload["questions"][0]["choices"]
        picked = choices[0]["value"]
        return {
            "model": "gpt-6-luna",
            "answers": [
                {
                    "name": "selected_candidate",
                    "type": "choice",
                    "choice": picked,
                    "probabilities": [{"value": picked, "probability": 0.91}],
                },
                {"name": "task_rigor", "type": "score", "score": 2.0},
            ],
            "usage": {"input_tokens": 500, "input_tokens_details": {"cached_tokens": 100}, "output_tokens": 0},
        }

    dec = kza_router.route_task(
        role="kza-qa",
        summary="Audit PR",
        exclude_families=["anthropic"],
        state_dir=tmp_path,
        quota_snapshot=_healthy_quota_snapshot(),
        api_caller=_fake_api,
        api_key_override="test-key",
    )
    assert dec["source"] == "openai-decisions:gpt-6-luna"
    assert dec["task_rigor"] == 3.0
    assert dec["api_usage"]["input_tokens"] == 500
    assert dec["api_usage"]["output_tokens"] == 0

    kza_router.record_outcome(decision_id=dec["decision_id"], rounds=1, approved=True, state_dir=tmp_path)
    report = kza_router.generate_report(days=30, state_dir=tmp_path)
    assert report["total_decisions"] == 1
    assert report["api_decisions"] == 1
    assert report["round1_approval_rate_pct"] == 100.0
    assert report["round1_target_met"] is True
    assert report["estimated_cost_usd"] > 0
