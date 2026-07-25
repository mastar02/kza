---
type: component
title: Ambient Path and Textual Wake Safety Net
description: Always-on ambient transcription stream, with a fuzzy-text "nexa" detector as a fallback for missed acoustic wake events.
resource: file:///Users/yo/Documents/kza/src/ambient/
tags: [kza, wake-word, ambient, asr, safety-net]
timestamp: "2026-07-24T00:00:00Z"
---

# Ambient Path and Textual Wake Safety Net

`src/ambient/` runs a separate, always-on transcription path (distinct from
the command-path STT): `tap.py` taps the mic stream, `segmenter.py` cuts it
into segments, `transcriber.py` (`AmbientTranscriber`) runs one async worker
per room that transcribes each segment and classifies its source
(`source_classifier.py`, `doa.py` for direction-of-arrival,
`speaker_tagger.py`). Per its docstring, transcription here also runs on
`cuda:0`. Transcripts are kept only as text, on a TTL, in
`data/ambient.db` (`store.py`) — "destilar y descartar": raw text is purged
after a retention window, and only distilled facts are meant to persist in
long-term memory. `distiller.py` and `language_quality.py` support that
distillation/filtering.

`textual_wake.py` (`TextualWakeDetector`) adds a **safety net on top of that
stream**: the acoustic wake detector in
[`../pipeline/wake-word.md`](../pipeline/wake-word.md) can miss far-field
speech ("compuerta acústica", per the module's own reference to a 2026-07-05
spec). If an already-transcribed ambient utterance — filtered to exclude TV
audio and the assistant's own TTS echo — contains "nexa" (or the calibrated
near variants "next up", fuzzy-matched at edit-distance ≤1, with an explicit
denylist for common false-positive words like "next"/"nena"/"nexo") it is
re-dispatched as a command using the text already transcribed, skipping a
second STT pass. The module's own comments document a deliberately accepted
false positive ("anexa") and the calibration data behind the thresholds
(`ambient.db` audit, 2026-07-05).

This is a secondary detection path, not the primary one — it exists
specifically to catch what
[`../pipeline/wake-word.md`](../pipeline/wake-word.md) misses, and feeds
detected commands into the same downstream flow described in
[`../pipeline/voice-pipeline.md`](../pipeline/voice-pipeline.md).
