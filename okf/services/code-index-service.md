---
type: service
title: Code Index Service
description: A separate systemd service exposing semantic code search over this repository for coding agents, with its own ChromaDB instance.
resource: file:///Users/yo/Documents/kza/src/code_index/
tags: [kza, code-index, chromadb, service, developer-tooling]
timestamp: "2026-07-24T00:00:00Z"
---

# Code Index Service

`src/code_index/` implements a semantic code-search index for the KZA
repository itself, meant for coding agents rather than the voice assistant's
runtime users. Per `docs/runbooks/CODE_INDEX.md`, it runs as its own
systemd **`--user`** unit, `kza-code-index.service`
(see [`../deployment/kza-voice-systemd.md`](../deployment/kza-voice-systemd.md)
for how that compares to the voice service's own deployment story), listening
on port `:9515`, backed by an independent ChromaDB instance at
`/home/kza/code-index/chroma/`. The runbook is explicit that this instance
is separate from — and never touches — the voice pipeline's own ChromaDB
described in
[`../datastores/vectordb-ha-sync.md`](../datastores/vectordb-ha-sync.md).

`tools/code_search.py` is the client used to query this service (referenced
in `docs/runbooks/CODE_INDEX.md`; not read in detail for this bundle).

This is a developer-tooling service, not part of the voice-interaction fast
or slow paths described in
[`../system/architecture.md`](../system/architecture.md) — it is included in
this bundle because it is a real, independently-deployed top-level
component of the repository, not because it participates in the runtime
voice pipeline.
