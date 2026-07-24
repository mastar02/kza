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

Other files in the directory (not read in full for this bundle, listed for
completeness): embedding client wrapper and sync-trigger logic that reacts to
Home Assistant entity registry changes pushed over the connection described
in [`../integrations/home-assistant-client.md`](../integrations/home-assistant-client.md).

Consumed by the NLU/routing layer in
[`../pipeline/nlu-command-gate.md`](../pipeline/nlu-command-gate.md) and
`request_router.py` (part of
[`../pipeline/voice-pipeline.md`](../pipeline/voice-pipeline.md)) to resolve
slots extracted from a command into concrete HA entities before dispatching
an action via [`../integrations/home-assistant-client.md`](../integrations/home-assistant-client.md).

**Uncertain / scope note:** this is the voice-pipeline's own ChromaDB
instance. It is unrelated to the separate ChromaDB instance used by
[`../services/code-index-service.md`](../services/code-index-service.md),
which explicitly does not touch this one.
