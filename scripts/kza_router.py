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
DEFAULT_KEYCHAIN_SERVICE = "kza-openai-decisions"
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


def uses_codex_fast_mode(candidate: ModelCandidate) -> bool:
    """Return True only when a Codex candidate explicitly enables `fast_mode=True`."""
    if candidate.runtime != "codex":
        return False
    return bool(candidate.fast_mode)


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
        capability_notes="Deep Python/async systems engineering, complex debugging, rigorous QA and invariant proofs (standard Codex tier).",
        role_suitability={
            "kza-analista": 0.93,
            "kza-dev": 0.97,
            "kza-pregate": 0.88,
            "kza-qa": 0.96,
            "kza-privacy": 0.95,
            "kza-ui": 0.84,
            "kza-ops": 0.93,
        },
        fast_mode=False,
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
        capability_notes="Specialized desktop/browser computer-use and visual E2E verification (NEXO UI, interactive PC checks; standard Codex tier).",
        role_suitability={
            "kza-analista": 0.72,
            "kza-dev": 0.75,
            "kza-pregate": 0.76,
            "kza-qa": 0.78,
            "kza-privacy": 0.74,
            "kza-ui": 0.98,
            "kza-ops": 0.78,
        },
        fast_mode=False,
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


def resolve_effective_effort(
    candidate: ModelCandidate,
    role: str,
    complexity: str,
    task_rigor: float | None,
) -> str:
    """Calibrate runtime reasoning effort using `gpt-6-luna` task_rigor (1..5)."""
    if task_rigor is None:
        return candidate.effort
    if task_rigor >= 4.2 and candidate.effort == "medium":
        return "high"
    if (
        task_rigor <= 2.2
        and complexity == "low"
        and role in {"kza-dev", "kza-ops", "kza-pregate"}
        and candidate.effort == "high"
    ):
        return "medium"
    return candidate.effort


def build_cli_command(
    candidate: ModelCandidate,
    role: str,
    *,
    codex_binary: str = "codex",
    effort_override: str | None = None,
) -> list[str]:
    """Build mandatory unattended CLI invocation flags with safety-guard hooks."""
    if role not in ALL_ROLES:
        raise ValueError(f"Unsupported role '{role}'")

    eff = effort_override or candidate.effort

    if candidate.runtime == "codex":
        cmd = [
            codex_binary,
            "--no-daemon",
            "-m",
            candidate.model,
            "-c",
            f"model_reasoning_effort={eff}",
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
            eff,
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
            "user,project",
            "--model",
            candidate.model,
            "--effort",
            eff,
            "--permission-mode",
            "bypassPermissions",
        ]

    if candidate.runtime == "agy":
        return [
            "agy",
            "--model",
            candidate.model,
            "--effort",
            eff,
            "--dangerously-skip-permissions",
            "--mode",
            "auto",
            "-i",
        ]

    raise ValueError(f"Unsupported runtime '{candidate.runtime}'")


def _load_policy_dict(state_dir: Path | None = None) -> dict[str, Any]:
    """Load optional `<state_dir>/policy.json` object, or return empty dict if absent."""
    policy_path = _state_dir(state_dir) / "policy.json"
    if not policy_path.is_file():
        return {}
    try:
        data = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid router policy file {policy_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid router policy file {policy_path}: expected JSON object.")
    return data


def resolve_openai_api_key(state_dir: Path | None = None) -> tuple[str | None, str]:
    """Resolve OpenAI API key from env or explicit Keychain opt-in without printing secrets."""
    env_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key, "env:OPENAI_API_KEY"

    kza_kc = os.environ.get("KZA_USE_KEYCHAIN", "").strip()
    th_kc = os.environ.get("THERMOUY_USE_KEYCHAIN", "").strip()
    if kza_kc == "0" or th_kc == "0":
        return None, "disabled"

    policy: dict[str, Any] = {}
    try:
        policy = _load_policy_dict(state_dir)
    except RuntimeError:
        policy = {}

    policy_use_kc = bool(policy.get("use_keychain", False))
    policy_kc_service = str(policy.get("keychain_service") or "").strip()

    if not (kza_kc == "1" or th_kc == "1" or policy_use_kc):
        return None, "none"

    service = os.environ.get("KZA_KEYCHAIN_SERVICE", "").strip() or policy_kc_service or DEFAULT_KEYCHAIN_SERVICE
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


# Static cacheable prefix (>1,100 tokens) kept byte-identical across all `gpt-6-luna` calls.
STATIC_ROUTING_PREAMBLE = """You are the KZA Multi-Console Orchestrator Router.
Your objective is to select the single best runtime+model candidate and rate the required reasoning rigor (`task_rigor` from 1 to 5) for an isolated engineering or verification seat in the KZA project.

=== 1. KZA SYSTEM CONTEXT & HARD INVARIANTS ===
KZA is a local-first Spanish voice assistant integrated with Home Assistant, built on Python 3.13 asyncio pipelines, GPU-accelerated wakeword/STT/TTS/LLM services (cuda:0 / cuda:1), SQLite persistence with durable retirement/tombstone barriers, strict speaker/persona/account/consent separation, NEXO web UI, and bytecode-compiled CPython 3.13 production releases (`scripts/build_runtime.py`).
Critical invariants that all specialist seats must uphold:
- Async non-blocking I/O: never block the asyncio event loop with synchronous file, network, or heavy CPU calls.
- Dependency injection & config centralization: constructor DI, dataclasses/enums, centralized `config/settings.yaml`, no `print()` in `src/`.
- Privacy & authority separation: voice identity, persona profile, user account, admin roles, and consent withdrawal/erasure must remain strictly isolated; ACK only after real storage/SDK handle completion; zero leakage of audio, transcripts, embeddings, secrets, `.env`, or `/home/kza/*` paths.
- Worktree isolation & verification rigor: write seats operate on dedicated `seat/<alias>` branches; read-only gates run on detached `HEAD` with `dirty=0` and must cite real `file:line` evidence and executed test results without hiding regressions behind hollow mocks.
- Cross-model auditing: `kza-qa` and `kza-privacy` enforce strict exclusion by real model family (`anthropic`, `openai`, `meta`, `google`). Notice that Claude models served via Agy (`agy:claude-opus-5-5-high`, `agy:claude-sonnet-5-5-medium`) belong to the `anthropic` family, not `google`.

=== 2. KZA SPECIALIST ROLES ===
1. `kza-analista` (read-only, detached HEAD): Pre-Task Spec Gate. Investigates codebase, tests, and canonical architecture docs to emit a concise `<= 2 KB` specification covering exact scope, applicable invariants, concrete verification plan, and truly ambiguous open decisions.
2. `kza-dev` (read-write, `seat/<alias>` branch): Application implementation specialist. Writes clean Python 3.13 async code, JS/HTML NEXO UI code, and real unit/integration tests with atomic commits. High Round-1 QA pass rate is essential: avoid half-tested patches, missing edge-case guards, or unrequested scope creep.
3. `kza-ops` (read-write, `seat/<alias>` branch): CI/CD, GitHub Actions, bytecode packaging (`scripts/build_runtime.py`), local Mac orchestration/hooks tooling, and operational runbooks.
4. `kza-pregate` (read-only, detached HEAD): Fast hygiene pre-review before expensive cross-model QA. Checks spec conformance, diff cleanliness, obvious invariant violations, and affected tests.
5. `kza-qa` (read-only, detached HEAD): Independent cross-model QA gate with 4 integrated lenses: (1) correctness/design/types/async concurrency, (2) silent failures/swallowed exceptions/unsafe fallbacks, (3) test rigor/coverage/real execution without hiding bugs, (4) comment/doc/PR-claim consistency. Emits proportional `APROBADO` vs `BLOQUEA`.
6. `kza-privacy` (read-only, detached HEAD): Independent privacy, authority, and release gate. Audits voice/persona/account/consent boundaries, durable erasure/tombstone/lease barriers, self-contained bytecode releases, and zero PII/secret exposure.
7. `kza-ui` (read-only, detached HEAD): Real browser/desktop E2E verification specialist (NEXO UI, local HTTP flows, auth/session transitions, responsive layout, capture/privacy states, visual evidence).

=== 3. PROPORTIONALITY & QUOTA OPTIMIZATION POLICY ===
Four subscriptions are active: `claude` (single KZA account `claude-1`), `codex` (multi-account load-balanced across `codex` and `codex-2` on standard tier without `/fast`), `muse` (`muse-wonder-full-1.0`, `muse-wonder-1.0`), and `agy` (`gemini-4-*` and Agy-hosted Claude).
Follow strict proportionality to maximize Round-1 quality per quota unit across the full fleet:
- Reserve flagship heavyweights (`claude:claude-opus-5-5`, `codex:gpt-6.2-sol`, `muse:muse-wonder-full-1.0`, `agy:gemini-4-neon-high`, `agy:gemini-4-hydrogen-high`, `agy:claude-opus-5-5-high`) for `complexity=high`, multi-module architecture, subtle async/concurrency bugs, privacy/authority gates (`kza-privacy`), deep 4-lens `kza-qa`, or high-rigor specs.
- Prefer efficient medium-tier specialists (`claude:claude-sonnet-5-5`, `codex:gpt-6.2-astra`, `muse:muse-wonder-1.0`, `agy:gemini-4-carbon-medium`, `agy:claude-sonnet-5-5-medium`) for `kza-pregate`, `kza-ops` tooling, `complexity=low` or cleanly scoped `complexity=medium` tasks, and routine checks—unless the task summary reveals subtle concurrency, security, or privacy risk.
- For `kza-ui`, evaluate browser/computer-use strength alongside real quota weights: `codex:gpt-6.1-terra` excels at interactive desktop/browser E2E flows, while `agy:gemini-4-neon-high` and `agy:gemini-4-carbon-medium` offer strong multimodal UI inspection, and `claude:claude-opus-5-5` handles multimodal DOM/privacy checks when other families are excluded or depleted.
- Respect `quota_weight` seriously: when one subscription has a noticeably healthier `quota_weight` (for example `1.00` vs `0.75` or lower) and comparable role suitability, shift work toward the healthier subscription to keep all four fleets balanced and prevent hitting 5h/7d ceilings.
- Notice Codex operates on standard tier (`service_tier` default, `/fast` disabled) to preserve weekly quota capacity across `codex` and `codex-2`.

=== 4. TASK RIGOR SCORING RUBRIC (`task_rigor`) & ROUND-1 QUALITY SIGNALS ===
Score `task_rigor` accurately using the five calibrated levels rather than defaulting every task near 4.8:
- `1_minimal`: purely mechanical edits, typo fixes, formatting, trivial docstring/config updates with near-zero regression surface.
- `2_low`: small, single-file, well-bounded helper changes or focused unit-test additions with clear local contracts.
- `3_medium`: standard multi-file feature/bugfix work, pre-gates, CI/ops scripts, or scoped UI checks without deep concurrency or authority risk.
- `4_high`: complex async/concurrency flows, cross-module refactors, multi-lens `kza-qa` audits, nuanced `kza-analista` specs, or end-to-end NEXO session/privacy flows.
- `5_critical`: release-gating audits, privacy/authority/consent/erasure boundary changes (`kza-privacy`), SQLite durability/tombstone invariants, or production-safety changes.
Remember that Round-1 approval (`>= 85%` target) depends on matching specialist strengths to task risk: if a `kza-dev` task involves subtle asyncio lifecycle, cancellation, locking, or persistence barriers, prioritize high-precision candidates even at `complexity=medium`; if a task is clean, local, and well-specified, prefer medium-tier specialists to preserve flagship quota for high-rigor gates and complex engineering."""


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

    Places `STATIC_ROUTING_PREAMBLE` at the front of `input` so the request is
    eligible for >1,024-token prefix caching, and appends the variable request
    details and eligible candidates at the tail.
    """
    cand_lines = []
    choice_items = []
    for c in candidates:
        qw = quota_weights.get(c.runtime, 0.0)
        suit = c.role_suitability.get(role, 0.80)
        fast_tag = ", codex_fast=enabled" if uses_codex_fast_mode(c) else ""
        desc = (
            f"runtime={c.runtime}, family={c.family}, effort={c.effort}{fast_tag}, "
            f"quota_weight={qw:.2f}, benchmark_prior={suit:.2f}; {c.capability_notes}"
        )
        cand_lines.append(f"- {c.candidate_id}: {desc}")
        choice_items.append({"value": c.candidate_id, "description": desc})

    dynamic_section = (
        f"Role: {role} ({'read-only' if role in READ_ONLY_ROLES else 'read-write'}).\n"
        f"Complexity hint: {complexity}.\n"
        f"Excluded model families (cross-model guard): {', '.join(excluded_families) or 'none'}.\n"
        f"Task summary (sanitized): {sanitized_summary}\n"
        f"Eligible candidates with real subscription quota weights (higher = healthier quota) "
        f"and documented capability/benchmark priors:\n"
        + "\n".join(cand_lines)
        + "\nChoose the single candidate maximizing expected Round-1 quality per quota unit."
    )
    input_text = f"{STATIC_ROUTING_PREAMBLE}\n\n--- LIVE ROUTING REQUEST ---\n{dynamic_section}"

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


def load_allowed_runtimes(state_dir: Path | None = None) -> set[str] | None:
    """Load optional runtime allowlist from `KZA_ALLOWED_RUNTIMES` or `<state_dir>/policy.json`."""
    env_raw = os.environ.get("KZA_ALLOWED_RUNTIMES", "").strip()
    if env_raw:
        return {item.strip().lower() for item in env_raw.split(",") if item.strip()}

    policy_path = _state_dir(state_dir) / "policy.json"
    if not policy_path.is_file():
        return None
    try:
        data = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid router policy file {policy_path}: {exc}") from exc
    if not isinstance(data, dict) or "allowed_runtimes" not in data:
        return None
    raw_list = data["allowed_runtimes"]
    if not isinstance(raw_list, list):
        raise RuntimeError(f"Invalid 'allowed_runtimes' in {policy_path}: expected list.")
    return {str(item).strip().lower() for item in raw_list if str(item).strip()}


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
    allowed_runtimes = load_allowed_runtimes(sdir)
    if allowed_runtimes is not None:
        all_candidates = [c for c in all_candidates if c.runtime in allowed_runtimes]
        if not all_candidates:
            raise RuntimeError(f"No candidates remain under allowed_runtimes={sorted(allowed_runtimes)}.")

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
            api_key, key_src = resolve_openai_api_key(state_dir=sdir)
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
                    # Gentle tie-breaking by quota health without double-penalizing healthy subscriptions.
                    weighted_probs: dict[str, float] = {}
                    for cid, prob in probs.items():
                        if cid not in eligible_ids:
                            continue
                        rt = next(c.runtime for c in eligible if c.candidate_id == cid)
                        qw = max(0.05, quota_weights.get(rt, 0.1))
                        mult = (0.85 + 0.15 * qw) if qw >= 0.25 else (0.50 + 0.50 * qw)
                        weighted_probs[cid] = prob * mult
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
    effective_effort = resolve_effective_effort(selected, role, complexity, task_rigor)
    codex_details = (subs.get("codex", {}).get("details") or {}) if isinstance(subs.get("codex"), dict) else {}
    claude_details = (subs.get("claude", {}).get("details") or {}) if isinstance(subs.get("claude"), dict) else {}
    codex_binary = str(codex_details.get("selected_cli") or "codex")
    if selected.runtime == "codex":
        selected_account = str(codex_details.get("selected_account") or "codex")
    elif selected.runtime == "claude":
        selected_account = str(claude_details.get("selected_account") or "claude-1")
    else:
        selected_account = selected.runtime

    cli_command = build_cli_command(
        selected,
        role,
        codex_binary=codex_binary,
        effort_override=effective_effort,
    )
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
        "selected_account": selected_account,
        "runtime": selected.runtime,
        "model": selected.model,
        "family": selected.family,
        "effort": selected.effort,
        "effective_effort": effective_effort,
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


def has_recorded_outcome(decision_id: str, state_dir: Path | None = None) -> bool:
    """Return True if an outcome event already exists for `decision_id`."""
    log_path = _state_dir(state_dir) / "decisions.jsonl"
    if not log_path.is_file():
        return False
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if decision_id not in line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("type") == "outcome" and obj.get("decision_id") == decision_id:
            return True
    return False


def record_outcome(
    *,
    decision_id: str,
    rounds: int,
    approved: bool,
    notes: str = "",
    state_dir: Path | None = None,
    skip_if_exists: bool = False,
) -> dict[str, Any]:
    """Append an outcome event linked to a prior decision_id."""
    if rounds < 1:
        raise ValueError("rounds must be >= 1.")
    sdir = _state_dir(state_dir)
    if skip_if_exists and has_recorded_outcome(decision_id, state_dir=sdir):
        return {
            "type": "outcome",
            "decision_id": decision_id,
            "already_recorded": True,
        }
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
