---
type: component
title: Spotify Music Path
description: The parallel "music path" (~500ms target) that maps voice commands to Spotify playback across zones/rooms.
resource: file:///Users/yo/Documents/kza/src/spotify/
tags: [kza, spotify, music, zones, mood]
timestamp: "2026-07-24T00:00:00Z"
---

# Spotify Music Path

`src/spotify/music_dispatcher.py` defines `MusicIntent` (a `StrEnum` of
recognized music actions — play/pause/skip/volume/mood-based requests, per
the file's opening definitions) and dispatches them against Spotify. Per
`CLAUDE.md`, this is the second of the three execution paths sharing the
KZA process, targeting ~500ms rather than the <300ms domotics budget or the
multi-second slow path.

The directory (per top-level listing) also includes a mood-to-playlist
mapper and a zone/room controller so a command like "pon música relajante en
la cocina" resolves to both a Spotify action and a target output zone —
not examined file-by-file for this bundle beyond `music_dispatcher.py`.

Reached the same way as domotics commands — via
[`../pipeline/nlu-command-gate.md`](../pipeline/nlu-command-gate.md) and
`request_router.py` in [`../pipeline/voice-pipeline.md`](../pipeline/voice-pipeline.md)
— but resolved against Spotify rather than
[`../integrations/home-assistant-client.md`](../integrations/home-assistant-client.md).
