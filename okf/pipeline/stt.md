---
type: component
title: Speech-to-Text (Command Path)
description: GPU speech-to-text for the <300ms domotics command path, with a filler/hallucination acceptance gate.
resource: file:///Users/yo/Documents/kza/src/stt/
tags: [kza, stt, asr, whisper, parakeet, latency]
timestamp: "2026-07-24T00:00:00Z"
---

# Speech-to-Text (Command Path)

`src/stt/` implements the transcription step of the fast/command path (as
opposed to the always-on ambient path in
[`../pipeline/textual-wake-safety-net.md`](../pipeline/textual-wake-safety-net.md),
which has its own transcriber):

- `whisper_fast.py` — the default engine, described in
  `config/settings.yaml` as "faster-whisper turbo GPU".
- `parakeet_stt.py` — an alternative engine (Parakeet-TDT, ONNX, CPU, 0
  VRAM), explicitly called out in `config/settings.yaml` as not to be
  switched to for the command path "without shadow-mode benchmark data" —
  the fast path has a hard <300ms budget. `config/settings.yaml` also
  documents an in-progress `shadow_engine` mode that runs a second STT in
  parallel purely to log/compare.
- `streaming_stt.py` — streaming variant.

Output confidence feeds `src/nlu/command_gate.py`
([`../pipeline/nlu-command-gate.md`](../pipeline/nlu-command-gate.md)),
which decides whether a transcript is a real command or noise/TV
audio/hallucination before it is acted on.

This module is invoked by `command_processor.py`
(part of [`../pipeline/voice-pipeline.md`](../pipeline/voice-pipeline.md))
after [`../pipeline/wake-word.md`](../pipeline/wake-word.md) fires.

**Uncertain:** exact GPU device assignment for this engine is not asserted
here — see the note in [`../system/architecture.md`](../system/architecture.md).
