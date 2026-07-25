---
type: system
title: KZA System Architecture
description: Entry point and dependency-injection root that wires the whole local voice assistant for Home Assistant into one process.
resource: file:///Users/yo/Documents/kza/src/main.py
tags: [kza, architecture, entry-point, voice-assistant, home-assistant]
timestamp: "2026-07-24T00:00:00Z"
---

# KZA System Architecture

KZA is a local voice-control system for Home Assistant. `src/main.py` is the
single entry point (`python -m src.main`): it loads `config/settings.yaml`
(see [`../config/settings.md`](../config/settings.md)), then constructs and
wires ~30 services by hand (constructor injection, no framework) — STT, TTS,
vector sync, HA client, reasoner, orchestrator, users, audio zones, alerts,
training, etc.

Per `CLAUDE.md`, the system has three execution paths that share the same
process:

- **Fast path** (target <300ms): mic → wake word → STT → vector search →
  Home Assistant action → TTS, for domotics commands ("prende la luz").
- **Music path** (~500ms): Spotify command → mood mapper → zone controller →
  TTS.
- **Slow path** (seconds): free-form questions routed to the cloud reasoner
  → long-term memory → TTS.

High-level chain (from `CLAUDE.md`):

```
Mic → WakeWord(CPU) → STT(GPU) → Router 7B(GPU, :8101) → TTS(GPU) → Speaker
                         ↕                  ↕
                   SpeakerID/Emotion   Reasoner cloud (gateway :8200)
                   BGE-M3 embeddings   ChromaDB / Home Assistant
```

This concept is the hub of the bundle; the pieces of that chain are each
their own concept:

- [`../pipeline/wake-word.md`](../pipeline/wake-word.md) — always-on wake
  detection that gates the rest of the fast path.
- [`../pipeline/textual-wake-safety-net.md`](../pipeline/textual-wake-safety-net.md)
  — a second, textual wake channel over an always-on ambient transcript.
- [`../pipeline/voice-pipeline.md`](../pipeline/voice-pipeline.md) — the
  orchestrator that actually sequences audio capture → command processing →
  response for a single interaction.
- [`../pipeline/stt.md`](../pipeline/stt.md) and
  [`../pipeline/tts.md`](../pipeline/tts.md) — the two ends of the audio
  path.
- [`../pipeline/nlu-command-gate.md`](../pipeline/nlu-command-gate.md) —
  decides whether a transcript is a real command before it reaches routing.
- [`../pipeline/llm-reasoning.md`](../pipeline/llm-reasoning.md) — both the
  fast 7B router (:8101) and the cloud deep reasoner (gateway :8200) live
  here.
- [`../datastores/vectordb-ha-sync.md`](../datastores/vectordb-ha-sync.md)
  and [`../integrations/home-assistant-client.md`](../integrations/home-assistant-client.md)
  — how domotics commands are resolved and executed against Home Assistant.
- [`../orchestration/multi-user-orchestrator.md`](../orchestration/multi-user-orchestrator.md)
  — request dispatch (fast/slow path), per-user context and cancellation.
- [`../orchestration/speaker-emotion.md`](../orchestration/speaker-emotion.md)
  — who is speaking and how, feeding into the orchestrator's per-user
  context.
- [`../integrations/spotify-music-path.md`](../integrations/spotify-music-path.md)
  — the parallel music path mentioned above.
- How this process is actually run on the server is described in
  [`../deployment/kza-voice-systemd.md`](../deployment/kza-voice-systemd.md).

**Uncertain / stale sources:** `README.md` at the repo root describes an
older/aspirational design (4x RTX 3070, `openwakeword "hey_jarvis"`, local
70B LLM reasoner) that contradicts the more recently dated `CLAUDE.md` and
`docs/architecture/DEPLOYMENT.md` (2 GPUs today, Porcupine "nexa" wake word,
cloud reasoner via a gateway). This bundle follows `CLAUDE.md` and
`docs/architecture/DEPLOYMENT.md` as the fresher sources and flags README.md
as partially outdated rather than treating it as authoritative. Exact
per-component GPU (`cuda:N`) assignment is not asserted here — `CLAUDE.md`,
`docs/architecture/HARDWARE.md` and inline comments in
`config/settings.yaml` disagree on details (2 vs 4 GPUs, which GPU hosts
STT) and appear to be at different points in an ongoing hardware migration.
