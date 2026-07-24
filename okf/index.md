---
okf_version: "0.1"
title: KZA OKF Bundle
description: Open Knowledge Format bundle for the kza local voice-control system for Home Assistant.
timestamp: "2026-07-24T00:00:00Z"
---

# KZA OKF Bundle

KZA is a local voice-control system for Home Assistant: a wake word gates a
fast domotics path (<300ms), a parallel music path drives Spotify (~500ms),
and free-form questions are routed to a cloud LLM reasoner (slow path,
seconds). This bundle covers the real, currently-tracked top-level
components of the repository, grounded in `README.md`, `CLAUDE.md`,
`docs/architecture/*.md`, `docs/runbooks/CODE_INDEX.md`,
`config/settings.yaml`, and the source under `src/`, `config/`, `docker/`,
`systemd/`, `deploy/`, `scripts/`.

Start at [`system/architecture.md`](system/architecture.md) — it is the hub
concept and links out to everything else.

## Concepts

**System**
- [`system/architecture.md`](system/architecture.md) — entry point, DI root,
  the three execution paths.

**Pipeline** (the per-interaction chain)
- [`pipeline/wake-word.md`](pipeline/wake-word.md) — acoustic wake detection
  (Porcupine "nexa").
- [`pipeline/textual-wake-safety-net.md`](pipeline/textual-wake-safety-net.md)
  — ambient transcription + textual wake fallback.
- [`pipeline/voice-pipeline.md`](pipeline/voice-pipeline.md) — the per-request
  orchestrator (5 wired components).
- [`pipeline/stt.md`](pipeline/stt.md) — command-path speech-to-text.
- [`pipeline/nlu-command-gate.md`](pipeline/nlu-command-gate.md) — command
  acceptance gate + grammar/LLM classification.
- [`pipeline/llm-reasoning.md`](pipeline/llm-reasoning.md) — fast 7B router
  (`:8101`) and cloud reasoner (gateway `:8200`).
- [`pipeline/tts.md`](pipeline/tts.md) — multi-engine text-to-speech.

**Datastores**
- [`datastores/vectordb-ha-sync.md`](datastores/vectordb-ha-sync.md) —
  ChromaDB index of HA entities.

**Integrations**
- [`integrations/home-assistant-client.md`](integrations/home-assistant-client.md)
  — KZA → HA (REST/WebSocket).
- [`integrations/home-assistant-config-push.md`](integrations/home-assistant-config-push.md)
  — HA-side YAML exposing KZA back into Home Assistant.
- [`integrations/spotify-music-path.md`](integrations/spotify-music-path.md)
  — the music path.

**Orchestration**
- [`orchestration/multi-user-orchestrator.md`](orchestration/multi-user-orchestrator.md)
  — fast/slow routing, priority, per-user context, cancellation.
- [`orchestration/speaker-emotion.md`](orchestration/speaker-emotion.md) —
  speaker ID and emotion detection.

**Services**
- [`services/code-index-service.md`](services/code-index-service.md) —
  standalone semantic code-search service for coding agents (`:9515`).

**Deployment**
- [`deployment/kza-voice-systemd.md`](deployment/kza-voice-systemd.md) —
  tracked systemd unit, flagged as stale vs. documented production reality.
- [`deployment/llm-fast-router-external.md`](deployment/llm-fast-router-external.md)
  — local rollback LLM unit; fast-router service noted as untracked.
- [`deployment/docker-experimental.md`](deployment/docker-experimental.md) —
  explicitly non-production container setup.
- [`deployment/udev-xvf3800.md`](deployment/udev-xvf3800.md) — mic-array USB
  access rule.

**Config**
- [`config/settings.md`](config/settings.md) — the central `settings.yaml`.

## Scope and noise filtering

This is a ~24k-file repository; most of that is not represented here by
design. Treated as noise and excluded from bundling: `venv/`, `.venv/`
(~2.1GB of installed packages), `models/` (weight files, currently empty but
excluded regardless), `data/` (runtime state — DBs, JSON, logs, not source
components), `__pycache__/`, `.pytest_cache/`, `.worktrees/` /
`.claude/worktrees/`. Also out of scope as dedicated concepts (not noise,
just not "top-level components"): `tests/`, `benchmarks/`, `examples/`,
`tools/` (developer scripts, referenced from concepts above rather than
bundled individually), and historical planning docs under `docs/plans/`,
`docs/research/`, `docs/superpowers/`.

See individual concept files for explicit "uncertain" notes on: README.md /
`docs/architecture/HARDWARE.md` staleness vs. `CLAUDE.md`, the
`kza-voice.service` repo-vs-production drift, the untracked
`kza-llm-fast.service`, and GPU device-assignment ambiguity across sources.
