---
type: deployment
title: Local LLM Services (Fast Router and Rollback Reasoner)
description: The systemd unit for a local rollback LLM, and the untracked service backing the fast 7B router that the fast path depends on.
resource: file:///Users/yo/Documents/kza/scripts/kza-llm-ik.service
tags: [kza, llm, systemd, deployment, gpu, rollback]
timestamp: "2026-07-24T00:00:00Z"
---

# Local LLM Services (Fast Router and Rollback Reasoner)

`scripts/kza-llm-ik.service` is a tracked systemd unit serving a local model
(Qwen3-30B-A3B, GLM-Air-equivalent per its own naming) via `ik_llama.cpp` on
port `:8200` — the same port
[`../pipeline/llm-reasoning.md`](../pipeline/llm-reasoning.md)'s
`LLMReasoner` normally reaches over the network at the cloud gateway. Per
`config/settings.yaml`'s commented-out "emergency rollback" block, this unit
is the way to fall back from the cloud reasoner to a local model without
code changes — swap the `reasoner.base_url` to `127.0.0.1:8200` and start
this service. As of this bundle it is **not** the active default; the cloud
gateway is.

Separately, `FastRouter`
([`../pipeline/llm-reasoning.md`](../pipeline/llm-reasoning.md)) depends on
a `kza-llm-fast.service` serving Qwen2.5-7B-Instruct on port `:8101` — no
systemd unit file for that service was found tracked in this repository
under `systemd/` or `scripts/`. It is referenced by name and port in
`src/llm/reasoner.py`'s docstring and `config/settings.yaml`, so the fast
path clearly depends on it existing, but its deployment definition appears
to live outside this repo (or under a name/path not discovered during
recon). Flagged as uncertain rather than invented.
