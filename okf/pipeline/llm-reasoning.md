---
type: component
title: LLM Reasoning (Fast Router + Cloud Reasoner)
description: Two LLM clients sharing one module — a fast local 7B router for the <300ms path, and a cloud deep-reasoning model for complex queries.
resource: file:///Users/yo/Documents/kza/src/llm/
tags: [kza, llm, reasoning, router, cloud, failover]
timestamp: "2026-07-24T00:00:00Z"
---

# LLM Reasoning (Fast Router + Cloud Reasoner)

`src/llm/reasoner.py` defines two distinct clients that together implement
the "reasoning path" mentioned in `CLAUDE.md`'s architecture diagram:

- **`FastRouter`** — an HTTP client to a local `llama-server` at
  `http://127.0.0.1:8101/v1` (Qwen2.5-7B-Instruct Q4_K_M, per its own
  docstring, "kza-llm-fast.service"), used to classify/answer simple
  domotics-adjacent queries quickly, or emit a `[DEEP]` marker to hand off
  to the cloud reasoner. Config lives under `router:` in
  `config/settings.yaml` (`base_url`, model path, 30s timeout).
- **`LLMReasoner`** — an HTTP client configured (`reasoner:` block in
  `config/settings.yaml`) against a gateway at `http://192.168.1.2:8200/v1`
  ("gateway LiteLLM", decision dated 2026-05-30) forwarding to MiniMax cloud
  (`http_model: "MiniMax-M2.7-highspeed"`), gated by an explicit
  `cloud.consent` flag — the settings file documents this as a deliberate,
  reversible decision to send user transcripts/context to a third party.
  A commented-out "emergency rollback" block in the same file shows the
  alternative: a local GLM-Air model on the same port (see
  [`../deployment/llm-fast-router-external.md`](../deployment/llm-fast-router-external.md)
  for the systemd unit that can serve that local fallback).

Supporting files in the module: `router_factory.py` (constructs the right
router from config), `cloud_consent.py` (`is_cloud_endpoint` — the guard
behind the consent flag), `cooldown.py` + `error_classifier.py` (failover:
per `config/settings.yaml`'s `llm.failover` block, endpoints that fail enter
an exponential cooldown and are skipped, persisted to disk across restarts),
`idle_watchdog.py`, `buffered_streamer.py`, `metrics.py`.

`FastRouter` output feeds
[`../pipeline/nlu-command-gate.md`](../pipeline/nlu-command-gate.md)'s
`llm_router.py`; `LLMReasoner` is invoked from the slow path in
[`../orchestration/multi-user-orchestrator.md`](../orchestration/multi-user-orchestrator.md).
Neither the fast-router `llama-server` process (:8101) nor the cloud gateway
(:8200, owned by shared infra per `CLAUDE.md`) run inside this repository's
process — both are external services this code talks to over HTTP.
