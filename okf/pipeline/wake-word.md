---
type: component
title: Wake Word Detection
description: Always-on acoustic detector that gates the fast path by listening for the "nexa" wake word.
resource: file:///Users/yo/Documents/kza/src/wakeword/
tags: [kza, wake-word, audio, porcupine, cpu]
timestamp: "2026-07-24T00:00:00Z"
---

# Wake Word Detection

`src/wakeword/` holds the acoustic wake-word detectors. The module exports
(`__init__.py`) `WakeWordDetector`, `WakeWordTrainer`, `WakeWordRecorder`
(`detector.py`, `trainer.py`, `recorder.py`) — originally built around
pre-trained OpenWakeWord models (e.g. `hey_jarvis`) and custom-trained
models.

The currently-referenced production wake word is different: `porcupine_wake.py`
implements `PorcupineWakeDetector`, a Picovoice Porcupine-based detector for
a dedicated Spanish wake word **"Nexa"**, trained in the Picovoice console.
Its docstring explicitly says it replaces "the stopgap 'Whisper-as-wake-word'
that hallucinated over silence", and that it runs on CPU (no GPU cost). There
is also `streaming_whisper_wake.py` / `whisper_wake.py` as
Whisper-based alternatives, and `wake_clip_writer.py` for capturing audio
clips around a detection.

**Uncertain:** `config/settings.yaml`'s `wake_word:` block still names
`model: "hey_jarvis"` and marks its own `threshold` field as legacy/dead
("el umbral activo es rooms.wake_word.threshold"), while the actual active
detector appears to be Porcupine/"nexa" per the module docstrings and
`docs/architecture/WAKE_WORD_TRAINING.md`. Which detector is wired into
production at any given moment is not fully resolved from static reading —
treat `PorcupineWakeDetector` as the current design intent, not a confirmed
fact about the running server.

Detected wake events feed `src/pipeline/audio_loop.py`, part of
[`../pipeline/voice-pipeline.md`](../pipeline/voice-pipeline.md), which then
hands the following audio to [`../pipeline/stt.md`](../pipeline/stt.md).
It is complemented, not replaced, by
[`../pipeline/textual-wake-safety-net.md`](../pipeline/textual-wake-safety-net.md),
a second, purely textual wake channel meant to catch cases where the
acoustic detector misses far-field speech.
