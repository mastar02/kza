---
type: component
title: Text-to-Speech
description: Multi-engine TTS (Piper, XTTS, Kokoro, Qwen3-TTS) with streaming playback and a response cache; current config runs Kokoro only.
resource: file:///Users/yo/Documents/kza/src/tts/
tags: [kza, tts, audio, kokoro, piper, streaming]
timestamp: "2026-07-24T00:00:00Z"
---

# Text-to-Speech

`src/tts/piper_tts.py` — despite its filename — implements several TTS
engines as separate classes: `PiperTTS`, `XTTS`, `KokoroTTS`, `Qwen3TTS`, and
two composites, `HybridTTS` and `DualTTS` (route by text length: short →
fast engine, long → higher-quality/conversational engine). `StreamingAudioPlayer`
in the same file handles buffered, low-latency playback.

`config/settings.yaml`'s `tts:` block sets `engine: "dual"`, with a comment
that on current 2-GPU hardware `DualTTS` runs **Kokoro only** — Qwen3-TTS is
disabled because a second GPU isn't available for it. `response_cache.py`
caches synthesized responses to skip re-synthesis for repeated phrases
(referenced in `docs/superpowers/plans/2026-04-24-s2-tts-response-cache.md`).

Invoked from `response_handler.py`, part of
[`../pipeline/voice-pipeline.md`](../pipeline/voice-pipeline.md), as the
final step of both the fast and slow paths described in
[`../system/architecture.md`](../system/architecture.md).
