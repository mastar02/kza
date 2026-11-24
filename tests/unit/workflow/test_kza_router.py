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

    for role in ("kza-qa", "kza-ui", "kza-pregate", "kza-analista", "kza-privacy", "kza-dev", "kza-ops"):
        codex_cmd = kza_router.build_cli_command(codex_sol, role)
        assert codex_cmd[0] == "codex"
        assert "--no-daemon" in codex_cmd
        assert "-a" in codex_cmd and "never" in codex_cmd
        assert "-s" in codex_cmd and "danger-full-access" in codex_cmd
        assert "--dangerously-bypass-hook-trust" in codex_cmd
        assert "read-only" not in codex_cmd
        assert "service_tier=fast" not in codex_cmd

        codex2_cmd = kza_router.build_cli_command(codex_sol, role, codex_binary="codex-2", effort_override="medium")
        assert codex2_cmd[0] == "codex-2"
        assert "model_reasoning_effort=medium" in codex2_cmd
        assert "service_tier=fast" not in codex2_cmd

        muse_cmd = kza_router.build_cli_command(by_rt["muse"], role)
        assert "--yolo" in muse_cmd
        assert "--contributor-mode" in muse_cmd and "off" in muse_cmd
        assert "--disable-request-user-input" in muse_cmd
        assert "--trust-workspace" in muse_cmd
        assert "--effort" in muse_cmd and by_rt["muse"].effort in muse_cmd
        assert "--disable-write" not in muse_cmd

        claude_cmd = kza_router.build_cli_command(by_rt["claude"], role)
        assert "--setting-sources" in claude_cmd and "user,project" in claude_cmd
        assert "--permission-mode" in claude_cmd and "bypassPermissions" in claude_cmd
        assert "plan" not in claude_cmd

        agy_cmd = kza_router.build_cli_command(by_rt["agy"], role)
        assert "--dangerously-skip-permissions" in agy_cmd
        assert "--mode" in agy_cmd and "auto" in agy_cmd
        assert "--effort" in agy_cmd and by_rt["agy"].effort in agy_cmd
        assert agy_cmd[-1] == "-i"
        assert "plan" not in agy_cmd

    terra_ui = kza_router.build_cli_command(codex_terra, "kza-ui")
    assert "--no-daemon" in terra_ui and "service_tier=fast" not in terra_ui and "danger-full-access" in terra_ui

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
    assert payload["input"].startswith(kza_router.STATIC_ROUTING_PREAMBLE)
    assert "service_tier=fast" not in payload["input"]
    choice_map = {ch["value"]: ch["description"] for ch in payload["questions"][0]["choices"]}
    assert "service_tier=fast" not in choice_map["codex:gpt-6.2-sol"]
    assert "service_tier=fast" not in choice_map["codex:gpt-6.1-terra"]
    assert "service_tier=fast" not in choice_map["codex:gpt-6.2-astra"]

    with pytest.raises(ValueError, match="Unsupported role"):
        kza_router.build_cli_command(codex_sol, "unknown-role")


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


def test_allowed_runtimes_policy_blocks_disallowed_even_under_emergency(tmp_path: Path) -> None:
    (tmp_path / "policy.json").write_text(json.dumps({"allowed_runtimes": ["claude", "codex"]}), encoding="utf-8")
    dec = kza_router.route_task(
        role="kza-qa",
        summary="QA restricted to claude and codex",
        exclude_families=["openai"],
        allow_quota_emergency=True,
        offline=True,
        state_dir=tmp_path,
        quota_snapshot={
            "quota_emergency": True,
            "eligible_runtimes": [],
            "subscriptions": {rt: {"weight": 0.0, "excluded": True, "status": "excluded"} for rt in ("claude", "codex", "muse", "agy")},
        },
    )
    assert dec["runtime"] == "claude"
    assert all(cid.startswith(("claude:", "codex:")) for cid in dec["eligible_candidates"])

    # When only claude+codex are allowed and both anthropic and openai are excluded, fail closed (never fall back to agy/muse).
    with pytest.raises(RuntimeError, match="No candidates remain"):
        kza_router.route_task(
            role="kza-qa",
            summary="No allowed runtime left",
            exclude_families=["anthropic", "openai"],
            allow_quota_emergency=True,
            offline=True,
            state_dir=tmp_path,
            quota_snapshot=_healthy_quota_snapshot(),
        )


def test_cacheable_preamble_codex2_account_propagation_and_rigor_effort(tmp_path: Path) -> None:
    assert len(kza_router.STATIC_ROUTING_PREAMBLE) >= 4800
    cands = list(kza_router.DEFAULT_CATALOG[:4])
    p1 = kza_router.build_decisions_payload(
        role="kza-dev",
        complexity="medium",
        sanitized_summary="Task A",
        candidates=cands,
        quota_weights={"claude": 0.9, "codex": 1.0},
        excluded_families=[],
    )
    p2 = kza_router.build_decisions_payload(
        role="kza-qa",
        complexity="high",
        sanitized_summary="Task B",
        candidates=cands,
        quota_weights={"claude": 0.8, "codex": 0.95},
        excluded_families=["anthropic"],
    )
    marker = "\n\n--- LIVE ROUTING REQUEST ---\n"
    assert p1["input"].split(marker)[0] == p2["input"].split(marker)[0]

    snap_codex2 = {
        "quota_emergency": False,
        "eligible_runtimes": ["claude", "codex", "muse", "agy"],
        "subscriptions": {
            "claude": {"weight": 0.9, "excluded": False, "status": "ok", "details": {"selected_account": "claude-1"}},
            "codex": {
                "weight": 1.0,
                "excluded": False,
                "status": "ok",
                "details": {"selected_account": "codex-2", "selected_cli": "codex-2"},
            },
            "muse": {"weight": 0.9, "excluded": False, "status": "ok"},
            "agy": {"weight": 0.85, "excluded": False, "status": "ok"},
        },
    }

    def _pick_astra_high_rigor(payload: dict, api_key: str) -> dict:
        return {
            "model": "gpt-6-luna",
            "answers": [
                {
                    "name": "selected_candidate",
                    "type": "choice",
                    "choice": "codex:gpt-6.2-astra",
                    "probabilities": [{"value": "codex:gpt-6.2-astra", "probability": 0.88}],
                },
                {"name": "task_rigor", "type": "score", "score": 3.6},  # 1-based = 4.6 >= 4.2
            ],
            "usage": {"input_tokens": 1300, "input_tokens_details": {"cached_tokens": 1024}, "output_tokens": 0},
        }

    dec_up = kza_router.route_task(
        role="kza-dev",
        summary="Tricky async concurrency fix",
        complexity="medium",
        state_dir=tmp_path,
        quota_snapshot=snap_codex2,
        api_caller=_pick_astra_high_rigor,
        api_key_override="test-key",
    )
    assert dec_up["selected_account"] == "codex-2"
    assert dec_up["cli_command"][0] == "codex-2"
    assert dec_up["effort"] == "medium"
    assert dec_up["effective_effort"] == "high"
    assert "model_reasoning_effort=high" in dec_up["cli_command"]
    assert "service_tier=fast" not in dec_up["cli_command"]

    def _pick_sol_low_rigor(payload: dict, api_key: str) -> dict:
        return {
            "model": "gpt-6-luna",
            "answers": [
                {
                    "name": "selected_candidate",
                    "type": "choice",
                    "choice": "codex:gpt-6.2-sol",
                    "probabilities": [{"value": "codex:gpt-6.2-sol", "probability": 0.85}],
                },
                {"name": "task_rigor", "type": "score", "score": 0.5},  # 1-based = 1.5 <= 2.2
            ],
            "usage": {"input_tokens": 1300, "input_tokens_details": {"cached_tokens": 1024}, "output_tokens": 0},
        }

    dec_down = kza_router.route_task(
        role="kza-dev",
        summary="Small typo fix in helper",
        complexity="low",
        state_dir=tmp_path,
        quota_snapshot=snap_codex2,
        api_caller=_pick_sol_low_rigor,
        api_key_override="test-key",
    )
    assert dec_down["effort"] == "high"
    assert dec_down["effective_effort"] == "medium"
    assert "model_reasoning_effort=medium" in dec_down["cli_command"]


def test_keychain_requires_explicit_opt_in_and_uses_kza_default_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert kza_router.DEFAULT_KEYCHAIN_SERVICE == "kza-openai-decisions"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("KZA_USE_KEYCHAIN", raising=False)
    monkeypatch.delenv("THERMOUY_USE_KEYCHAIN", raising=False)
    monkeypatch.delenv("KZA_KEYCHAIN_SERVICE", raising=False)

    called_services: list[str] = []

    def _fake_subproc_run(cmd, **kwargs):
        called_services.append(cmd[cmd.index("-s") + 1])
        return type("Proc", (), {"returncode": 0, "stdout": "sk-test-kc\n"})()

    monkeypatch.setattr(kza_router.subprocess, "run", _fake_subproc_run)

    # Without opt-in (no env, no policy.json opt-in), security is never invoked.
    key, src = kza_router.resolve_openai_api_key(state_dir=tmp_path)
    assert key is None and src == "none"
    assert called_services == []

    # With policy.json opt-in, configured service (or KZA default) is queried.
    (tmp_path / "policy.json").write_text(
        json.dumps({"use_keychain": True, "keychain_service": "kza-custom-kc"}),
        encoding="utf-8",
    )
    key2, src2 = kza_router.resolve_openai_api_key(state_dir=tmp_path)
    assert key2 == "sk-test-kc" and src2 == "keychain:kza-custom-kc"
    assert called_services == ["kza-custom-kc"]

    # Explicit boolean false leaves Keychain off without calling security.
    (tmp_path / "policy.json").write_text(json.dumps({"use_keychain": False}), encoding="utf-8")
    key3, src3 = kza_router.resolve_openai_api_key(state_dir=tmp_path)
    assert key3 is None and src3 == "none"
    assert called_services == ["kza-custom-kc"]

    # Truthy strings, non-bool ints, or malformed JSON fail closed before calling security.
    for bad_payload in (
        json.dumps({"use_keychain": "false"}),
        json.dumps({"use_keychain": "0"}),
        json.dumps({"use_keychain": 1}),
        "{broken-json",
    ):
        (tmp_path / "policy.json").write_text(bad_payload, encoding="utf-8")
        with pytest.raises(RuntimeError):
            kza_router.resolve_openai_api_key(state_dir=tmp_path)
    assert called_services == ["kza-custom-kc"]

    # Claude binary matches selected_account even when claude-2 is explicitly requested.
    snap_claude2 = {
        "quota_emergency": False,
        "eligible_runtimes": ["claude"],
        "subscriptions": {
            "claude": {"weight": 1.0, "excluded": False, "status": "ok", "details": {"selected_account": "claude-2"}},
            "codex": {"weight": 0.0, "excluded": True, "status": "excluded"},
            "muse": {"weight": 0.0, "excluded": True, "status": "excluded"},
            "agy": {"weight": 0.0, "excluded": True, "status": "excluded"},
        },
    }
    dec_c2 = kza_router.route_task(
        role="kza-privacy",
        summary="Verify account parity",
        offline=True,
        state_dir=tmp_path / "clean",
        quota_snapshot=snap_claude2,
    )
    assert dec_c2["selected_account"] == "claude-2"
    assert dec_c2["cli_command"][0] == "claude-2"
