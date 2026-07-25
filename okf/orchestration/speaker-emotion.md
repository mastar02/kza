---
type: component
title: Speaker Identification and Emotion
description: Per-utterance speaker ID and emotion/tone detection, feeding per-user context in the orchestrator.
resource: file:///Users/yo/Documents/kza/src/users/
tags: [kza, speaker-id, emotion, users, personalization]
timestamp: "2026-07-24T00:00:00Z"
---

# Speaker Identification and Emotion

`src/users/` (per `CLAUDE.md`'s architecture diagram, which shows
"SpeakerID/Emotion" branching off the same step as STT) provides per-user
identification and emotion/tone signal extraction from voice, run alongside
transcription in `command_processor.py`
(part of [`../pipeline/voice-pipeline.md`](../pipeline/voice-pipeline.md)).

This bundle did not read the module's source files individually — its
existence and role are grounded in `CLAUDE.md`'s architecture diagram and
the top-level `src/` directory listing rather than in file-level docstrings.
Treat the specifics (which models, how "emotion" is represented) as
unconfirmed.

Output is expected to feed
[`../orchestration/multi-user-orchestrator.md`](../orchestration/multi-user-orchestrator.md)'s
`ContextManager`, so that per-user conversational history and personalization
are keyed by recognized speaker rather than by device/session alone.
