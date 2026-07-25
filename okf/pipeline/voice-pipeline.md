---
type: pipeline
title: Voice Pipeline
description: The thin orchestrator that sequences audio capture, command processing, routing and response for a single voice interaction.
resource: file:///Users/yo/Documents/kza/src/pipeline/
tags: [kza, pipeline, voice, orchestration, latency]
timestamp: "2026-07-24T00:00:00Z"
---

# Voice Pipeline

`src/pipeline/voice_pipeline.py` (`VoicePipeline`) is, in its own words, "a
thin orchestrator that wires together 5 extracted components":

1. `audio_loop.py` (`AudioLoop`, plus `multi_room_audio_loop.py` for
   multi-room) — audio capture, wake word (see
   [`../pipeline/wake-word.md`](../pipeline/wake-word.md)), echo suppression
   and ambient detection hookup.
2. `command_processor.py` (`CommandProcessor`) — audio → text + speaker ID +
   emotion, delegating to [`../pipeline/stt.md`](../pipeline/stt.md) and
   [`../orchestration/speaker-emotion.md`](../orchestration/speaker-emotion.md).
3. `request_router.py` (`RequestRouter`) — routes a recognized command
   through the orchestrated or legacy path, handing off to
   [`../orchestration/multi-user-orchestrator.md`](../orchestration/multi-user-orchestrator.md).
4. `response_handler.py` (`ResponseHandler`) — text → audio, with streaming
   and zone routing, via [`../pipeline/tts.md`](../pipeline/tts.md).
5. `feature_manager.py` (`FeatureManager`) — timers, intercom,
   notifications, alerts, Home Assistant integration side-effects.

Supporting modules in the same directory: `command_event.py` (shared
`CommandEvent` DTO, also used by the ambient/textual-wake path),
`ambient_guard.py`, `earcon_gate.py` (audio-cue gating) and
`model_manager.py`.

This is the concept that realizes the "fast path" and "music path" described
in [`../system/architecture.md`](../system/architecture.md): it is the
per-interaction sequencer, while
[`../orchestration/multi-user-orchestrator.md`](../orchestration/multi-user-orchestrator.md)
handles cross-request concerns (multiple users, priority, cancellation) once
`RequestRouter` hands a request off.
