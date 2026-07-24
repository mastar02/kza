---
type: component
title: NLU Command Acceptance Gate
description: Decides whether a post-wake transcript is a real user command before it is routed, filtering noise, TV audio, and STT hallucination.
resource: file:///Users/yo/Documents/kza/src/nlu/
tags: [kza, nlu, command-gate, grammar, hallucination-filtering]
timestamp: "2026-07-24T00:00:00Z"
---

# NLU Command Acceptance Gate

`src/nlu/command_gate.py` (`CommandAcceptanceGate`) is, per its own
docstring, a consolidation of noise/echo heuristics that used to live
scattered in `request_router.py`, now combined with STT confidence. A
rejection is a silent discard upstream — it never reaches routing. It
matches against a curated list of non-command phrases (YouTube/TV
boilerplate, filler) and — per `config/settings.yaml`'s `command_gate:`
block — has a confidence-based rule set that, as of this bundle, runs only
in **shadow mode** ("NO flippear enforce_confidence"): a 2026-06-03 report
found the STT's own confidence signals (`no_speech_prob`, `avg_logprob`)
unreliable/inverted on the `whisper turbo` engine.

Other files in the module:

- `command_grammar.py` — a deterministic grammar engine so that recognized
  domotics phrasings can bypass the LLM classifier entirely (referenced in
  `config/settings.yaml`'s NLU comments as "la domótica ya no toca el LLM").
- `slot_extractor.py` — pulls structured slots (device, room, value) out of
  accepted text.
- `llm_router.py`, `llm_gate.py` — the LLM-backed classification path for
  text the grammar engine doesn't cover; per `config/settings.yaml`'s
  `nlu.llm_router` block, this calls the fast 7B model described in
  [`../pipeline/llm-reasoning.md`](../pipeline/llm-reasoning.md) to decide
  if text is a real command vs. ambient noise, with a configured
  `min_command_confidence` and timeout.
- `sensitive_actions.py` — presumably flags actions needing extra
  confirmation; not read in detail for this bundle (uncertain).

This gate sits between [`../pipeline/stt.md`](../pipeline/stt.md) output and
routing in [`../pipeline/voice-pipeline.md`](../pipeline/voice-pipeline.md) /
[`../orchestration/multi-user-orchestrator.md`](../orchestration/multi-user-orchestrator.md).
