---
type: component
title: Multi-User Orchestrator
description: Routes requests between fast and slow paths and manages per-user conversational context, priority, and cancellation.
resource: file:///Users/yo/Documents/kza/src/orchestrator/
tags: [kza, orchestration, multi-user, priority-queue, cancellation]
timestamp: "2026-07-24T00:00:00Z"
---

# Multi-User Orchestrator

`src/orchestrator/__init__.py` documents this module's architecture directly
via an ASCII diagram, showing fast-path commands (domotics, resolved via
vector search and returning to TTS in <300ms) and slow-path questions
(routed to the cloud reasoner) both flowing through a shared dispatcher.
The module's exported (`__all__`) surface includes:

- `MultiUserOrchestrator` — top-level coordinator.
- `RequestDispatcher` — decides fast vs. slow path per incoming request and
  invokes [`../pipeline/llm-reasoning.md`](../pipeline/llm-reasoning.md)'s
  `LLMReasoner` for the slow path.
- `PriorityRequestQueue` — orders concurrent requests, presumably so that,
  e.g., a domotics command from one user isn't starved by a long-running
  cloud reasoning request from another (inferred from the name; not
  confirmed by reading the queueing logic itself).
- `ContextManager` — per-user conversational context/history.
- `CancellationToken` — lets a newer request cancel/supersede an
  in-flight one.

This is the cross-request coordination layer that `request_router.py` (in
[`../pipeline/voice-pipeline.md`](../pipeline/voice-pipeline.md)) hands off
to once the NLU gate has accepted a command; it is what makes the "multiple
simultaneous users/rooms" property of KZA (implied by the multi-room audio
loop and per-room ambient workers in
[`../pipeline/textual-wake-safety-net.md`](../pipeline/textual-wake-safety-net.md))
actually coherent rather than racy.
