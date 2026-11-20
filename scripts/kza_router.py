#!/usr/bin/env python3
"""Dynamic multi-console router for KZA via OpenAI Decisions API (`gpt-6-luna`).

Combines model capability/benchmark suitability with live 4-subscription quota
weights from `kza_quota.py`, enforces strict cross-model exclusion by real
model family (`anthropic` | `openai` | `meta` | `google`), sanitizes task
summaries before external calls, builds safe per-runtime CLI flags, falls back
deterministically when offline/unavailable, and tracks routing outcomes/KPIs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import kza_quota  # noqa: E402

DECISIONS_API_URL = "https://api.openai.com/v1/decisions"
DECISIONS_MODEL = "gpt-6-luna"
DEFAULT_KEYCHAIN_SERVICE = "thermouy-openai-decisions"
DEFAULT_STATE_DIR = kza_quota.DEFAULT_STATE_DIR

# Estimated gpt-6-luna pricing (USD per 1M input / cached input tokens; 0 output tokens).
LUNA_INPUT_USD_PER_M = 0.25
LUNA_CACHED_USD_PER_M = 0.025

FAMILIES: tuple[str, ...] = ("anthropic", "openai", "meta", "google")

READ_ONLY_ROLES: frozenset[str] = frozenset(
    {"kza-analista", "kza-pregate", "kza-qa", "kza-privacy", "kza-ui"}
)
WRITE_ROLES: frozenset[str] = frozenset({"kza-dev", "kza-ops"})
ALL_ROLES: frozenset[str] = READ_ONLY_ROLES | WRITE_ROLES


@dataclass(frozen=True)
class ModelCandidate:
    """Candidate runtime+model entry with real model family and benchmark profile."""

    candidate_id: str
    runtime: str  # claude | codex | muse | agy
    model: str
    family: str  # anthropic | openai | meta | google
    effort: str
    capability_notes: str
    role_suitability: dict[str, float] = field(default_factory=dict)
    fast_mode: bool = False


_CODEX_FAST_MODEL_RE = re.compile(r"(?:^|-)(?:sol|terra)(?:$|-)", re.IGNORECASE)


def uses_codex_fast_mode(candidate: ModelCandidate) -> bool:
    """Return True when Codex should launch with `/fast on` (`-c service_tier=fast`).

    Applies to Codex `sol` and `terra` models (or any Codex candidate with
    `fast_mode=True` in the catalog).
    """
    if candidate.runtime != "codex":
        return False
    return bool(candidate.fast_mode or _CODEX_FAST_MODEL_RE.search(candidate.model))


# Catalog derived from verified Mac CLI models and role benchmarks.
# Never hardcodes a single model for any role (including `kza-ui`); can be extended
# via ~/.local/state/kza-router/catalog_override.json or KZA_ROUTER_CATALOG.
DEFAULT_CATALOG: tuple[ModelCandidate, ...] = (
    ModelCandidate(
        candidate_id="claude:claude-opus-5-5",
        runtime="claude",
        model="claude-opus-5-5",
        family="anthropic",
        effort="high",
        capability_notes="Top architectural synthesis, deep code review, privacy/invariant auditing, multimodal UI analysis.",
        role_suitability={
            "kza-analista": 0.96,
            "kza-dev": 0.94,
            "kza-pregate": 0.86,
            "kza-qa": 0.97,
            "kza-privacy": 0.98,
            "kza-ui": 0.85,
            "kza-ops": 0.88,
        },
    ),
    ModelCandidate(
        candidate_id="claude:claude-sonnet-5-5",
        runtime="claude",
        model="claude-sonnet-5-5",
        family="anthropic",
        effort="medium",
        capability_notes="Fast, high-signal implementation, spec drafting, and pre-gate checks.",
        role_suitability={
            "kza-analista": 0.90,
            "kza-dev": 0.91,
            "kza-pregate": 0.93,
            "kza-qa": 0.88,
            "kza-privacy": 0.86,
            "kza-ui": 0.80,
            "kza-ops": 0.89,
        },
    ),
    ModelCandidate(
        candidate_id="codex:gpt-6.2-sol",
        runtime="codex",
        model="gpt-6.2-sol",
        family="openai",
        effort="high",
        capability_notes="Deep Python/async systems engineering, complex debugging, rigorous QA and invariant proofs; runs with Codex /fast on (service_tier=fast, ~2x speed, higher quota rate).",
        role_suitability={
            "kza-analista": 0.93,
            "kza-dev": 0.97,
            "kza-pregate": 0.88,
            "kza-qa": 0.96,
            "kza-privacy": 0.95,
            "kza-ui": 0.84,
            "kza-ops": 0.93,
        },
        fast_mode=True,
    ),
    ModelCandidate(
        candidate_id="codex:gpt-6.2-astra",
        runtime="codex",
        model="gpt-6.2-astra",
        family="openai",
        effort="medium",
        capability_notes="Fast, accurate coding, unit tests, pre-gate verification, and CI/ops tooling.",
        role_suitability={
            "kza-analista": 0.87,
            "kza-dev": 0.92,
            "kza-pregate": 0.95,
            "kza-qa": 0.87,
            "kza-privacy": 0.84,
            "kza-ui": 0.78,
            "kza-ops": 0.95,
        },
        fast_mode=False,
    ),
    ModelCandidate(
        candidate_id="codex:gpt-6.1-terra",
        runtime="codex",
        model="gpt-6.1-terra",
        family="openai",
        effort="medium",
        capability_notes="Specialized desktop/browser computer-use and visual E2E verification (NEXO UI, interactive PC checks); runs with Codex /fast on (service_tier=fast, ~1.5x speed).",
        role_suitability={
            "kza-analista": 0.72,
            "kza-dev": 0.75,
            "kza-pregate": 0.76,
            "kza-qa": 0.78,
            "kza-privacy": 0.74,
            "kza-ui": 0.98,
            "kza-ops": 0.78,
        },
        fast_mode=True,
    ),
    ModelCandidate(
        candidate_id="muse:muse-wonder-full-1.0",
        runtime="muse",
        model="muse-wonder-full-1.0",
        family="meta",
        effort="high",
        capability_notes="Strong independent Meta-family reviewer for cross-model QA, silent-failure hunting, specs, and privacy gates.",
        role_suitability={
            "kza-analista": 0.92,
            "kza-dev": 0.89,
            "kza-pregate": 0.88,
            "kza-qa": 0.94,
            "kza-privacy": 0.93,
            "kza-ui": 0.78,
            "kza-ops": 0.87,
        },
    ),
    ModelCandidate(
        candidate_id="muse:muse-wonder-1.0",
        runtime="muse",
        model="muse-wonder-1.0",
        family="meta",
        effort="medium",
        capability_notes="Fast Meta-family lane for pre-gates, focused reviews, and lightweight implementation.",
        role_suitability={
            "kza-analista": 0.85,
            "kza-dev": 0.84,
            "kza-pregate": 0.92,
            "kza-qa": 0.86,
            "kza-privacy": 0.83,
            "kza-ui": 0.72,
            "kza-ops": 0.86,
        },
    ),
    ModelCandidate(
        candidate_id="agy:gemini-4-neon-high",
        runtime="agy",
        model="gemini-4-neon-high",
        family="google",
        effort="high",
        capability_notes="Flagship Google-family reasoning, large-context synthesis, cross-model QA, and multimodal UI inspection.",
        role_suitability={
            "kza-analista": 0.95,
            "kza-dev": 0.93,
            "kza-pregate": 0.87,
            "kza-qa": 0.95,
            "kza-privacy": 0.94,
            "kza-ui": 0.93,
            "kza-ops": 0.90,
        },
    ),
    ModelCandidate(
        candidate_id="agy:gemini-4-carbon-medium",
        runtime="agy",
        model="gemini-4-carbon-medium",
        family="google",
        effort="medium",
        capability_notes="Balanced Google-family worker for fast dev, pre-gate checks, ops, and UI verification.",
        role_suitability={
            "kza-analista": 0.86,
            "kza-dev": 0.90,
            "kza-pregate": 0.93,
            "kza-qa": 0.86,
            "kza-privacy": 0.84,
            "kza-ui": 0.87,
            "kza-ops": 0.92,
        },
    ),
    ModelCandidate(
        candidate_id="agy:gemini-4-hydrogen-high",
        runtime="agy",
        model="gemini-4-hydrogen-high",
        family="google",
        effort="high",
        capability_notes="Deep Google-family analytical model for intricate privacy/authority checks and subtle concurrency bugs.",
        role_suitability={
            "kza-analista": 0.91,
            "kza-dev": 0.88,
            "kza-pregate": 0.84,
            "kza-qa": 0.93,
            "kza-privacy": 0.95,
            "kza-ui": 0.85,
            "kza-ops": 0.87,
        },
    ),
    ModelCandidate(
        candidate_id="agy:claude-opus-5-5-high",
        runtime="agy",
        model="claude-opus-5-5-high",
        family="anthropic",
        effort="high",
        capability_notes="Anthropic Opus served via Agy quota (counts as anthropic family for cross-model exclusion).",
        role_suitability={
            "kza-analista": 0.95,
            "kza-dev": 0.93,
            "kza-pregate": 0.85,
            "kza-qa": 0.96,
            "kza-privacy": 0.97,
            "kza-ui": 0.84,
            "kza-ops": 0.87,
        },
    ),
    ModelCandidate(
        candidate_id="agy:claude-sonnet-5-5-medium",
        runtime="agy",
        model="claude-sonnet-5-5-medium",
        family="anthropic",
        effort="medium",
        capability_notes="Anthropic Sonnet served via Agy quota (counts as anthropic family for cross-model exclusion).",
        role_suitability={
            "kza-analista": 0.89,
            "kza-dev": 0.90,
            "kza-pregate": 0.92,
            "kza-qa": 0.87,
            "kza-privacy": 0.85,
            "kza-ui": 0.79,
            "kza-ops": 0.88,
        },
    ),
)

# Defense-in-depth redaction patterns before sending any summary to OpenAI Decisions API.
_SANITIZE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S), "[REDACTED_KEY]"),
    (re.compile(r"\b(?:sk-ant-|sk-proj-|sk-)[A-Za-z0-9_-]{12,}\b"), "[REDACTED_API_KEY]"),
    (re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{16,}|github_pat_[A-Za-z0-9_]{16,}|xox[abprs]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{25,})\b"), "[REDACTED_TOKEN]"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"), "[REDACTED_JWT]"),
    (re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/=-]{8,}"), r"\1[REDACTED]"),
    (re.compile(r"(?i)\b((?:password|passwd|pwd|secret|token|api[_-]?key|client_secret)\s*[:=]\s*)[^\s\"',;]{4,}"), r"\1[REDACTED]"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),
    (re.compile(r"\b(?:192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b"), "[REDACTED_LAN_IP]"),
    (re.compile(r"/home/kza/[^\s\"']+"), "[REDACTED_PROD_PATH]"),
    (re.compile(r"(?:/Users/[^/\s]+/[^\s\"']*\.env\b|(?:\./)?data/[^\s\"']+)"), "[REDACTED_PRIVATE_PATH]"),
)


def sanitize_summary(text: str, max_len: int = 800) -> str:
    """Sanitize task summary removing secrets, emails, LAN IPs, and private paths."""
    cleaned = text or ""
    for pat, repl in _SANITIZE_PATTERNS:
        cleaned = pat.sub(repl, cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) > max_len:
        cleaned = cleaned[: max_len - 3].rstrip() + "..."
    return cleaned


def load_catalog(catalog_path: Path | None = None) -> list[ModelCandidate]:
    """Load default model catalog plus optional JSON override/extension."""
    by_id: dict[str, ModelCandidate] = {c.candidate_id: c for c in DEFAULT_CATALOG}
    override_path = catalog_path
    if override_path is None:
        env_path = os.environ.get("KZA_ROUTER_CATALOG", "").strip()
        if env_path:
            override_path = Path(env_path)
        else:
            candidate_file = _state_dir() / "catalog_override.json"
            if candidate_file.is_file():
                override_path = candidate_file

    if override_path and override_path.is_file():
        try:
            raw = json.loads(override_path.read_text(encoding="utf-8"))
            items = raw.get("candidates", raw) if isinstance(raw, dict) else raw
            if isinstance(items, list):
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    cid = str(item.get("candidate_id") or "")
                    if not cid:
                        continue
                    by_id[cid] = ModelCandidate(
                        candidate_id=cid,
                        runtime=str(item["runtime"]),
                        model=str(item["model"]),
                        family=str(item["family"]).lower(),
                        effort=str(item.get("effort", "medium")),
                        capability_notes=str(item.get("capability_notes", "")),
                        role_suitability={
                            str(k): float(v)
                            for k, v in (item.get("role_suitability") or {}).items()
                        },
                        fast_mode=bool(item.get("fast_mode", False)),
                    )
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
    return list(by_id.values())


def build_cli_command(candidate: ModelCandidate, role: str) -> list[str]:
    """Build mandatory unattended CLI invocation flags with safety-guard hooks."""
    if role not in ALL_ROLES:
        raise ValueError(f"Unsupported role '{role}'")

    if candidate.runtime == "codex":
        cmd = [
            "codex",
            "--no-daemon",
            "-m",
            candidate.model,
            "-c",
            f"model_reasoning_effort={candidate.effort}",
        ]
        if uses_codex_fast_mode(candidate):
            cmd.extend(["-c", "service_tier=fast"])
        cmd.extend(
            [
                "-a",
                "never",
                "-s",
                "danger-full-access",
                "--dangerously-bypass-hook-trust",
            ]
        )
        return cmd

    if candidate.runtime == "muse":
        return [
            "muse",
            "--model",
            candidate.model,
            "--effort",
            candidate.effort,
            "--yolo",
            "--contributor-mode",
            "off",
            "--disable-request-user-input",
            "--trust-workspace",
        ]

    if candidate.runtime == "claude":
        return [
            "claude",
            "--setting-sources",
            "project",
            "--model",
            candidate.model,
            "--effort",
            candidate.effort,
            "--permission-mode",
            "bypassPermissions",
        ]

    if candidate.runtime == "agy":
        return [
            "agy",
            "--model",
            candidate.model,
            "--effort",
            candidate.effort,
            "--dangerously-skip-permissions",
            "--mode",
            "auto",
            "-i",
        ]

    raise ValueError(f"Unsupported runtime '{candidate.runtime}'")


def resolve_openai_api_key() -> tuple[str | None, str]:
    """Resolve OpenAI API key from env or macOS Keychain without printing secrets."""
    env_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key, "env:OPENAI_API_KEY"

    use_kc = (
        os.environ.get("KZA_USE_KEYCHAIN", "").strip() == "1"
        or os.environ.get("THERMOUY_USE_KEYCHAIN", "").strip() == "1"
    )
    if not use_kc:
        return None, "none"

    service = os.environ.get("KZA_KEYCHAIN_SERVICE", "").strip() or DEFAULT_KEYCHAIN_SERVICE
    try:
        proc = subprocess.run(
            ["security", "find-generic-password", "-s", service, "-w"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip(), f"keychain:{service}"
    except (OSError, subprocess.SubprocessError):
        pass
    return None, f"keychain-missing:{service}"


def normalize_probabilities(raw_probs: Any) -> dict[str, float]:
    """Normalize choice probabilities from list `[{'value', 'probability'}]` or dict."""
    result: dict[str, float] = {}
    if isinstance(raw_probs, dict):
        for k, v in raw_probs.items():
            try:
                result[str(k)] = float(v)
            except (TypeError, ValueError):
                continue
        return result
    if isinstance(raw_probs, list):
        for item in raw_probs:
            if not isinstance(item, dict):
                continue
            val = item.get("value")
            prob = item.get("probability")
            if val is None or prob is None:
                continue
            try:
                result[str(val)] = float(prob)
            except (TypeError, ValueError):
                continue
    return result


def normalize_score_1_based(raw_score: Any, num_levels: int = 5) -> float | None:
    """Normalize 0-based Decisions API score (`0..N-1`) to 1-based (`1..N`)."""
    if raw_score is None or isinstance(raw_score, bool):
        return None
    try:
        val = float(raw_score)
    except (TypeError, ValueError):
        return None
    if 0.0 <= val <= float(num_levels - 1):
        return round(val + 1.0, 2)
    return round(max(1.0, min(float(num_levels), val)), 2)


def build_decisions_payload(
    *,
    role: str,
    complexity: str,
    sanitized_summary: str,
    candidates: list[ModelCandidate],
    quota_weights: dict[str, float],
    excluded_families: list[str],
) -> dict[str, Any]:
    """Construct the live `POST /v1/decisions` request payload for `gpt-6-luna`.

    Respects verified live contract details:
    - Top-level fields: `model`, `input`, `questions`.
    - Each question uses `name`, `type`, `instructions`.
    - `type: "choice"` uses `choices: [{"value", "description"}]`.
    - `type: "score"` levels carry ONLY `{"label", "description"}` (NO `"value"`).
    """
    cand_lines = []
    choice_items = []
    for c in candidates:
        qw = quota_weights.get(c.runtime, 0.0)
        suit = c.role_suitability.get(role, 0.80)
        fast_tag = ", codex_fast=/fast on (service_tier=fast)" if uses_codex_fast_mode(c) else ""
        desc = (
            f"runtime={c.runtime}, family={c.family}, effort={c.effort}{fast_tag}, "
            f"quota_weight={qw:.2f}, benchmark_prior={suit:.2f}; {c.capability_notes}"
        )
        cand_lines.append(f"- {c.candidate_id}: {desc}")
        choice_items.append({"value": c.candidate_id, "description": desc})

    input_text = (
        f"Project: KZA (local voice assistant for Home Assistant, Python 3.13 async, privacy/authority invariants).\n"
        f"Role: {role} ({'read-only' if role in READ_ONLY_ROLES else 'read-write'}).\n"
        f"Complexity: {complexity}.\n"
        f"Excluded model families (cross-model guard): {', '.join(excluded_families) or 'none'}.\n"
        f"Runtime policy: Codex sol and terra models run with `/fast on` (`-c service_tier=fast`: faster latency, higher quota consumption rate); factor this into speed vs quota trade-offs.\n"
        f"Task summary (sanitized): {sanitized_summary}\n"
        f"Eligible candidates with real subscription quota weights (higher = healthier quota) "
        f"and documented capability/benchmark priors:\n"
        + "\n".join(cand_lines)
        + "\nChoose the single candidate maximizing expected Round-1 quality per quota unit."
    )

    score_levels = [
        {"label": "1_minimal", "description": "Trivial/mechanical task"},
        {"label": "2_low", "description": "Simple scoped change or quick check"},
        {"label": "3_medium", "description": "Standard multi-file engineering or review task"},
        {"label": "4_high", "description": "Complex architecture, concurrency, privacy, or cross-module task"},
        {"label": "5_critical", "description": "High-risk release, authority, or multi-lens audit"},
    ]

    return {
        "model": DECISIONS_MODEL,
        "input": input_text,
        "questions": [
            {
                "name": "selected_candidate",
                "type": "choice",
                "instructions": "Select the best eligible candidate balancing capability/benchmarks for this role and real subscription quota weight.",
                "choices": choice_items,
            },
            {
                "name": "task_rigor",
                "type": "score",
                "instructions": "Rate the required reasoning rigor for this KZA task.",
                "levels": score_levels,
            },
        ],
    }


def _extract_decisions_answers(resp_json: dict[str, Any]) -> tuple[str | None, dict[str, float], float | None, dict[str, int]]:
    """Parse OpenAI Decisions API response into (choice, probs_dict, rigor_1_to_5, api_usage)."""
    usage_raw = resp_json.get("usage") if isinstance(resp_json.get("usage"), dict) else {}
    details = usage_raw.get("input_tokens_details") if isinstance(usage_raw.get("input_tokens_details"), dict) else {}
    api_usage = {
        "input_tokens": int(usage_raw.get("input_tokens") or usage_raw.get("prompt_tokens") or 0),
        "cached_tokens": int(usage_raw.get("cached_tokens") or details.get("cached_tokens") or 0),
        "output_tokens": int(usage_raw.get("output_tokens") or usage_raw.get("completion_tokens") or 0),
    }

    answers = resp_json.get("answers") or resp_json.get("results") or resp_json.get("decisions") or []
    if isinstance(answers, dict):
        answers = [{"name": k, **(v if isinstance(v, dict) else {"value": v})} for k, v in answers.items()]

    selected: str | None = None
    probs: dict[str, float] = {}
    rigor: float | None = None

    for item in answers:
        if not isinstance(item, dict):
            continue
        qname = str(item.get("name") or item.get("id") or item.get("question_id") or "")
        qtype = str(item.get("type") or "")
        if qname == "selected_candidate" or qtype == "choice":
            probs = normalize_probabilities(item.get("probabilities"))
            val = item.get("choice") or item.get("value") or item.get("answer")
            if isinstance(val, str) and val:
                selected = val
            elif probs:
                selected = max(probs.items(), key=lambda kv: kv[1])[0]
        elif qname == "task_rigor" or qtype == "score":
            raw_score = item.get("score")
            if raw_score is None:
                raw_score = item.get("value")
            rigor = normalize_score_1_based(raw_score, num_levels=5)

    return selected, probs, rigor, api_usage


def call_decisions_api(
    payload: dict[str, Any],
    api_key: str,
    timeout: float = 12.0,
) -> dict[str, Any]:
    """Execute HTTP POST to OpenAI Decisions API (`gpt-6-luna`)."""
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        DECISIONS_API_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _estimate_cost_usd(api_usage: dict[str, int]) -> float:
    """Estimate USD cost of a `gpt-6-luna` call (0 output tokens)."""
    inp = max(0, int(api_usage.get("input_tokens", 0)))
    cached = min(inp, max(0, int(api_usage.get("cached_tokens", 0))))
    uncached = inp - cached
    cost = (uncached * LUNA_INPUT_USD_PER_M + cached * LUNA_CACHED_USD_PER_M) / 1_000_000.0
    return round(cost, 8)


def _deterministic_scores(
    candidates: list[ModelCandidate],
    role: str,
    complexity: str,
    quota_weights: dict[str, float],
) -> dict[str, float]:
    """Compute transparent deterministic score per eligible candidate."""
    effort_bonus = {
        "low": {"low": 0.04, "medium": 0.03, "high": -0.02},
        "medium": {"low": -0.02, "medium": 0.03, "high": 0.02},
        "high": {"low": -0.05, "medium": 0.0, "high": 0.05},
    }.get(complexity, {})

    scores: dict[str, float] = {}
    for c in candidates:
        suit = c.role_suitability.get(role, 0.80)
        qw = quota_weights.get(c.runtime, 0.50)
        bonus = effort_bonus.get(c.effort, 0.0)
        # Blend capability (60%) and real quota health (40%) multiplicatively damped by severe scarcity.
        score = (0.60 * (suit + bonus) + 0.40 * qw) * (0.45 + 0.55 * qw)
        scores[c.candidate_id] = round(score, 6)
    return scores


def _state_dir(override: Path | None = None) -> Path:
    if override is not None:
        return override
    return Path(os.environ.get("KZA_ROUTER_STATE_DIR", str(DEFAULT_STATE_DIR)))


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append a JSON line to an owner-only state log."""
    kza_quota._ensure_private_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def route_task(
    *,
    role: str,
    summary: str,
    complexity: str = "medium",
    exclude_families: list[str] | None = None,
    allow_quota_emergency: bool = False,
    offline: bool = False,
    state_dir: Path | None = None,
    quota_snapshot: dict[str, Any] | None = None,
    catalog: list[ModelCandidate] | None = None,
    api_caller: Callable[[dict[str, Any], str], dict[str, Any]] | None = None,
    api_key_override: str | None = None,
    record_decision: bool = True,
) -> dict[str, Any]:
    """Select the optimal runtime+model candidate for a KZA role and task."""
    if role not in ALL_ROLES:
        raise ValueError(f"Unknown role '{role}'. Expected one of {sorted(ALL_ROLES)}.")

    excluded_fams = sorted({f.strip().lower() for f in (exclude_families or []) if f.strip()})
    for fam in excluded_fams:
        if fam not in FAMILIES:
            raise ValueError(f"Unknown family '{fam}'. Expected one of {FAMILIES}.")

    sdir = _state_dir(state_dir)
    qsnap = quota_snapshot if quota_snapshot is not None else kza_quota.collect_all_quotas(state_dir=sdir)
    subs = qsnap["subscriptions"]
    quota_weights = {rt: float(subs[rt]["weight"]) for rt in kza_quota.RUNTIMES if rt in subs}

    all_candidates = catalog if catalog is not None else load_catalog()
    # Stage 1 deterministic guardrails: cross-model exclusion by REAL model family.
    family_filtered = [c for c in all_candidates if c.family not in excluded_fams]
    if not family_filtered:
        raise RuntimeError(f"No candidates remain after excluding families {excluded_fams}.")

    eligible = [
        c
        for c in family_filtered
        if not subs.get(c.runtime, {}).get("excluded", False) and quota_weights.get(c.runtime, 0.0) > 0.0
    ]

    quota_emergency_used = False
    if not eligible:
        if not allow_quota_emergency:
            raise RuntimeError(
                "quota_emergency: all eligible subscriptions are excluded by quota/throttle policy. "
                "Pass --allow-quota-emergency to override explicitly."
            )
        quota_emergency_used = True
        eligible = family_filtered

    sanitized = sanitize_summary(summary)
    det_scores = _deterministic_scores(eligible, role, complexity, quota_weights)

    source = "fallback-deterministic"
    fallback_reason: str | None = None
    selected_id: str | None = None
    probabilities: dict[str, float] = {}
    task_rigor: int | None = None
    api_usage: dict[str, int] = {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0}

    if not offline:
        api_key = api_key_override
        key_src = "override"
        if api_key is None:
            api_key, key_src = resolve_openai_api_key()
        if not api_key:
            fallback_reason = f"no_api_key ({key_src})"
        else:
            payload = build_decisions_payload(
                role=role,
                complexity=complexity,
                sanitized_summary=sanitized,
                candidates=eligible,
                quota_weights=quota_weights,
                excluded_families=excluded_fams,
            )
            caller = api_caller or call_decisions_api
            try:
                resp = caller(payload, api_key)
                cand_choice, probs, rigor, usage = _extract_decisions_answers(resp)
                api_usage = usage
                probabilities = probs
                task_rigor = rigor
                eligible_ids = {c.candidate_id for c in eligible}
                if probs:
                    # Combine Decisions API probability with real quota weight for tie-breaking & safety.
                    weighted_probs = {
                        cid: prob * (0.35 + 0.65 * max(0.05, quota_weights.get(next(c.runtime for c in eligible if c.candidate_id == cid), 0.1)))
                        for cid, prob in probs.items()
                        if cid in eligible_ids
                    }
                    if weighted_probs:
                        selected_id = max(weighted_probs.items(), key=lambda kv: kv[1])[0]
                        source = f"openai-decisions:{DECISIONS_MODEL}"
                if selected_id is None and cand_choice in eligible_ids:
                    selected_id = cand_choice
                    source = f"openai-decisions:{DECISIONS_MODEL}"
                if selected_id is None:
                    fallback_reason = f"invalid_api_choice ({cand_choice!r})"
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, json.JSONDecodeError, ValueError) as exc:
                fallback_reason = f"api_error: {type(exc).__name__}: {exc}"
    else:
        fallback_reason = "offline_flag"

    if selected_id is None:
        selected_id = max(det_scores.items(), key=lambda kv: kv[1])[0]

    selected = next(c for c in eligible if c.candidate_id == selected_id)
    cli_command = build_cli_command(selected, role)
    decision_id = f"kza-dec-{uuid.uuid4().hex[:12]}"
    estimated_cost = _estimate_cost_usd(api_usage)

    record = {
        "type": "decision",
        "decision_id": decision_id,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "role": role,
        "read_only": role in READ_ONLY_ROLES,
        "complexity": complexity,
        "sanitized_summary": sanitized,
        "source": source,
        "fallback_reason": fallback_reason,
        "quota_emergency": quota_emergency_used,
        "excluded_families": excluded_fams,
        "selected_candidate": selected.candidate_id,
        "runtime": selected.runtime,
        "model": selected.model,
        "family": selected.family,
        "effort": selected.effort,
        "fast_mode": uses_codex_fast_mode(selected),
        "cli_command": cli_command,
        "quota_weights": quota_weights,
        "deterministic_scores": det_scores,
        "probabilities": probabilities,
        "task_rigor": task_rigor,
        "api_usage": api_usage,
        "estimated_cost_usd": estimated_cost,
        "eligible_candidates": [c.candidate_id for c in eligible],
    }

    if record_decision:
        _append_jsonl(sdir / "decisions.jsonl", record)
    return record


def record_outcome(
    *,
    decision_id: str,
    rounds: int,
    approved: bool,
    notes: str = "",
    state_dir: Path | None = None,
) -> dict[str, Any]:
    """Append an outcome event linked to a prior decision_id."""
    if rounds < 1:
        raise ValueError("rounds must be >= 1.")
    sdir = _state_dir(state_dir)
    event = {
        "type": "outcome",
        "decision_id": decision_id,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "rounds": rounds,
        "approved": approved,
        "round1_approved": bool(approved and rounds == 1),
        "notes": sanitize_summary(notes, max_len=300),
    }
    _append_jsonl(sdir / "decisions.jsonl", event)
    return event


def generate_report(*, days: int = 30, state_dir: Path | None = None) -> dict[str, Any]:
    """Compute KPI report from decisions.jsonl over the last N days."""
    sdir = _state_dir(state_dir)
    log_path = sdir / "decisions.jsonl"
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)

    decisions: dict[str, dict[str, Any]] = {}
    outcomes: dict[str, dict[str, Any]] = {}

    if log_path.is_file():
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            ts_val = kza_quota._parse_timestamp(obj.get("timestamp"))
            if ts_val is not None and datetime.fromtimestamp(ts_val, tz=timezone.utc) < cutoff:
                continue
            kind = obj.get("type", "decision")
            did = str(obj.get("decision_id") or "")
            if not did:
                continue
            if kind == "outcome":
                outcomes[did] = obj
            else:
                decisions[did] = obj

    total_decisions = len(decisions)
    runtime_counts = {rt: 0 for rt in kza_quota.RUNTIMES}
    api_calls = 0
    total_input = 0
    total_cached = 0
    total_output = 0
    total_cost = 0.0

    for d in decisions.values():
        rt = d.get("runtime")
        if rt in runtime_counts:
            runtime_counts[rt] += 1
        if str(d.get("source", "")).startswith("openai-decisions:"):
            api_calls += 1
        usage = d.get("api_usage") if isinstance(d.get("api_usage"), dict) else {}
        total_input += int(usage.get("input_tokens", 0))
        total_cached += int(usage.get("cached_tokens", 0))
        total_output += int(usage.get("output_tokens", 0))
        total_cost += float(d.get("estimated_cost_usd", 0.0))

    total_outcomes = len(outcomes)
    approved_count = sum(1 for o in outcomes.values() if o.get("approved"))
    round1_count = sum(1 for o in outcomes.values() if o.get("approved") and int(o.get("rounds", 99)) == 1)
    round1_rate = round((round1_count / total_outcomes) * 100.0, 2) if total_outcomes else None

    runtime_share = {
        rt: round((cnt / total_decisions) * 100.0, 2) if total_decisions else 0.0
        for rt, cnt in runtime_counts.items()
    }

    return {
        "window_days": days,
        "total_decisions": total_decisions,
        "api_decisions": api_calls,
        "fallback_decisions": total_decisions - api_calls,
        "total_outcomes": total_outcomes,
        "approved_outcomes": approved_count,
        "round1_approved_outcomes": round1_count,
        "round1_approval_rate_pct": round1_rate,
        "round1_target_pct": 85.0,
        "round1_target_met": (round1_rate >= 85.0) if round1_rate is not None else None,
        "subscription_counts": runtime_counts,
        "subscription_share_pct": runtime_share,
        "api_usage": {
            "input_tokens": total_input,
            "cached_tokens": total_cached,
            "output_tokens": total_output,
        },
        "estimated_cost_usd": round(total_cost, 6),
    }


def _parse_bool(raw: str) -> bool:
    norm = raw.strip().lower()
    if norm in ("true", "1", "yes", "y"):
        return True
    if norm in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Expected true|false, got '{raw}'.")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for KZA dynamic router."""
    parser = argparse.ArgumentParser(description="KZA dynamic router via OpenAI Decisions API (gpt-6-luna).")
    parser.add_argument("--state-dir", type=Path, default=None, help="Override state dir.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_route = sub.add_parser("route", help="Route a KZA role/task to the optimal runtime+model.")
    p_route.add_argument("--role", required=True, choices=sorted(ALL_ROLES))
    p_route.add_argument("--summary", required=True, help="Short task summary (sanitized before API call).")
    p_route.add_argument("--complexity", choices=("low", "medium", "high"), default="medium")
    p_route.add_argument(
        "--exclude-family",
        action="append",
        default=[],
        choices=FAMILIES,
        help="Exclude a real model family (repeatable) for strict cross-model routing.",
    )
    p_route.add_argument("--allow-quota-emergency", action="store_true")
    p_route.add_argument("--offline", action="store_true", help="Skip API and use deterministic fallback.")
    p_route.add_argument("--no-record", action="store_true", help="Do not append to decisions.jsonl.")
    p_route.add_argument("--json", action="store_true")

    p_out = sub.add_parser("outcome", help="Record outcome for a previous decision.")
    p_out.add_argument("decision_id")
    p_out.add_argument("--rounds", type=int, required=True)
    p_out.add_argument("--approved", type=_parse_bool, required=True)
    p_out.add_argument("--notes", default="")

    p_rep = sub.add_parser("report", help="Show routing KPIs (Round-1 approval rate, balance, USD cost).")
    p_rep.add_argument("--days", type=int, default=30)
    p_rep.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "route":
        try:
            decision = route_task(
                role=args.role,
                summary=args.summary,
                complexity=args.complexity,
                exclude_families=args.exclude_family,
                allow_quota_emergency=args.allow_quota_emergency,
                offline=args.offline,
                state_dir=args.state_dir,
                record_decision=not args.no_record,
            )
        except RuntimeError as exc:
            sys.stderr.write(f"ERROR: {exc}\n")
            return 2
        if args.json:
            print(json.dumps(decision, indent=2, ensure_ascii=False))
        else:
            print(
                f"{decision['decision_id']} -> {decision['selected_candidate']} "
                f"(family={decision['family']}, source={decision['source']})\n"
                f"  cmd: {' '.join(decision['cli_command'])}"
            )
            if decision["fallback_reason"]:
                print(f"  fallback_reason: {decision['fallback_reason']}")
        return 0

    if args.command == "outcome":
        ev = record_outcome(
            decision_id=args.decision_id,
            rounds=args.rounds,
            approved=args.approved,
            notes=args.notes,
            state_dir=args.state_dir,
        )
        print(json.dumps(ev, indent=2, ensure_ascii=False))
        return 0

    if args.command == "report":
        rep = generate_report(days=args.days, state_dir=args.state_dir)
        if args.json:
            print(json.dumps(rep, indent=2, ensure_ascii=False))
        else:
            r1 = "N/A" if rep["round1_approval_rate_pct"] is None else f"{rep['round1_approval_rate_pct']:.1f}%"
            print(
                f"KZA Router KPIs (last {rep['window_days']}d)\n"
                f"  Decisions: {rep['total_decisions']} (API={rep['api_decisions']}, fallback={rep['fallback_decisions']})\n"
                f"  Outcomes:  {rep['total_outcomes']} | Round-1 approval: {r1} (target >= {rep['round1_target_pct']:.0f}%)\n"
                f"  Balance:   {rep['subscription_share_pct']}\n"
                f"  Cost USD:  ${rep['estimated_cost_usd']:.6f} (in={rep['api_usage']['input_tokens']}, cached={rep['api_usage']['cached_tokens']}, out={rep['api_usage']['output_tokens']})"
            )
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
