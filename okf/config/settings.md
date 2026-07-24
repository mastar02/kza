---
type: config
title: Central Settings
description: The single YAML configuration file wiring engines, endpoints, thresholds and feature flags for every component in the process.
resource: file:///Users/yo/Documents/kza/config/settings.yaml
tags: [kza, config, yaml, settings]
timestamp: "2026-07-24T00:00:00Z"
---

# Central Settings

`config/settings.yaml` is loaded once by `src/main.py`
([`../system/architecture.md`](../system/architecture.md)) and threaded
through the ~30 services constructed there. It has roughly 39 top-level
sections; the ones read in detail for this bundle map directly to the
components documented elsewhere here:

- `wake_word` → [`../pipeline/wake-word.md`](../pipeline/wake-word.md)
  (contains the stale `hey_jarvis` reference discussed there).
- `stt` → [`../pipeline/stt.md`](../pipeline/stt.md).
- `tts` → [`../pipeline/tts.md`](../pipeline/tts.md) (`engine: "dual"`).
- `embeddings` → [`../datastores/vectordb-ha-sync.md`](../datastores/vectordb-ha-sync.md)
  (BGE-M3 model config).
- `router` and `reasoner`, plus an `llm.failover` block →
  [`../pipeline/llm-reasoning.md`](../pipeline/llm-reasoning.md) (fast 7B
  router at `:8101`, cloud gateway at `:8200`, `cloud.consent` gate,
  commented-out local rollback).
- `nlu` (including `nlu.llm_router`) and `command_gate` (currently
  shadow-mode confidence enforcement) →
  [`../pipeline/nlu-command-gate.md`](../pipeline/nlu-command-gate.md).

Many other top-level sections exist (covering, e.g., users, alerts, timers,
intercom, presence, memory, training — inferred from module directory names
under `src/` such as `src/alerts/`, `src/memory/`, `src/presence/`,
`src/training/`) but were not individually read for this bundle; per the
brief's "quality over exhaustiveness" guidance, this bundle scopes to the
components that make up the documented fast/music/slow paths and their
direct dependencies, not every module in the tree. Their existence is noted
here rather than given dedicated concept files, to avoid asserting details
this recon did not actually verify.

This file itself is the resource for this concept, distinct from
[`../integrations/home-assistant-config-push.md`](../integrations/home-assistant-config-push.md),
which is YAML deployed into Home Assistant rather than read by KZA.
