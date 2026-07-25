---
type: deployment
title: kza-voice systemd Unit
description: The tracked systemd unit for the main voice-pipeline process — a system-level template that appears stale relative to documented production reality.
resource: file:///Users/yo/Documents/kza/systemd/kza-voice.service
tags: [kza, systemd, deployment, gpu, drift]
timestamp: "2026-07-24T00:00:00Z"
---

# kza-voice systemd Unit

`systemd/kza-voice.service` is a tracked, **system-level** systemd unit
(`User=kza`, `Group=kza`, `WorkingDirectory=/opt/kza`,
`CUDA_VISIBLE_DEVICES=0,1,2,3` for 4 GPUs, plus several security-hardening
directives) that runs `python3.13 -m src.main`
(see [`../system/architecture.md`](../system/architecture.md)).

**This appears to be a stale template, not what actually runs in
production.** `docs/architecture/DEPLOYMENT.md` self-declares itself the
source of truth for deployment ("si otro doc contradice esto, este gana") and
states that production instead runs `kza-voice.service` as a systemd
**`--user`** unit at `/home/kza/.config/systemd/user/`, with system-level
units explicitly documented as disallowed ("prohibido por contrato") —
justified there by an R10 exception: containerization was judged infeasible
because of direct USB access to the ReSpeaker mic array, an MA1260 serial
device, and the <300ms latency budget.

This is a deliberately flagged, unresolved discrepancy between two files in
the same repository, not something this bundle resolves — see also
[`../deployment/docker-experimental.md`](../deployment/docker-experimental.md)
for the same "documented deployment story differs from what's actually
deployed" pattern applied to containerization. GPU count (4 here vs. 2 in
`CLAUDE.md`) is part of the same drift and is not asserted as current fact —
see the note in [`../system/architecture.md`](../system/architecture.md).
