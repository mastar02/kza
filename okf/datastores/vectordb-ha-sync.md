---
type: datastore
title: Vector Store and Home Assistant Sync
description: ChromaDB vector index of Home Assistant entities, kept in sync so the fast path can resolve domotics commands without an LLM call.
resource: file:///Users/yo/Documents/kza/src/vectordb/
tags: [kza, chromadb, vectordb, embeddings, home-assistant, latency]
timestamp: "2026-07-24T00:00:00Z"
---

# Vector Store and Home Assistant Sync

`src/vectordb/chroma_sync.py` (`ChromaSync`) maintains a ChromaDB collection
of Home Assistant entities (lights, switches, climate, etc.), embedded with
BGE-M3 (per `CLAUDE.md`'s architecture diagram and `config/settings.yaml`'s
`embeddings:` block), so that a transcribed command like "prende la luz del
living" can be resolved to the right `entity_id` by nearest-neighbor search
instead of a full LLM call — this is what keeps the domotics path inside its
<300ms budget.

The module's own comments document real production tuning history: a
`PREFER_AREA_BOOST` constant that biases matches toward the entity's declared
area, raised from `0.15` to `0.35` on 2026-06-04 after a 2026-05-03 incident
where cross-room matches won on raw embedding similarity.

`src/vectordb/` has no other source files: just `chroma_sync.py` (642 lines)
and an empty `__init__.py`. There is no separate embedding-client wrapper and
no separate sync-trigger file.

The sync mechanism itself is not a reactive listener on Home Assistant
registry/websocket events — `chroma_sync.py` and
`src/home_assistant/ha_client.py` were grepped for
`registry|entity_registry_updated|listen|subscribe|websocket` and nothing
ties a websocket/registry event to a ChromaDB resync (`ha_client.py`'s own
`subscribe_events`/`state_changed` machinery, around lines 697-855, feeds its
own internal HA state cache, not `ChromaSync`). Instead, `ChromaSync.sync_commands()`
(`chroma_sync.py:145`) is a **manual, voice-command-triggered full rebuild**:
it deletes every existing record in the commands collection
(`chroma_sync.py:172-176`) and regenerates all phrases from scratch via the
LLM reasoner (`chroma_sync.py:178-213`). It is invoked from
`src/pipeline/request_router.py:932-936` once `_is_sync_command(text)`
(`request_router.py:1371-1378`, matching phrases like "sincroniza" /
"actualiza los comandos") recognizes the utterance as a sync request.

`chroma.search_command` / `chroma.asearch_command` (the read path used to
resolve a command to an entity) are called from `src/pipeline/request_router.py:1050,1083`
(part of [`../pipeline/voice-pipeline.md`](../pipeline/voice-pipeline.md),
which explicitly covers `request_router.py`) and from `RequestDispatcher` in
`src/orchestrator/dispatcher.py:730` (part of
[`../orchestration/multi-user-orchestrator.md`](../orchestration/multi-user-orchestrator.md)).
There is no such call in `src/nlu/` — the NLU command gate
([`../pipeline/nlu-command-gate.md`](../pipeline/nlu-command-gate.md)) decides
whether a transcript is a real command at all, upstream of this vector
resolution step, but does not itself call into `ChromaSync`.

**Uncertain / scope note:** this is the voice-pipeline's own ChromaDB
instance. It is unrelated to the separate ChromaDB instance used by
[`../services/code-index-service.md`](../services/code-index-service.md),
which explicitly does not touch this one.
