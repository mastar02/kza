---
type: deployment
title: Docker Compose Setup (Experimental)
description: A tracked but explicitly non-production, experimental containerization of KZA services, without parity to the real deployment.
resource: file:///Users/yo/Documents/kza/docker/
tags: [kza, docker, experimental, deployment, non-production]
timestamp: "2026-07-24T00:00:00Z"
---

# Docker Compose Setup (Experimental)

`docker/` and the root `docker-compose.yml` define a containerized version of
some KZA services (including a `chromadb` service). This is explicitly and
repeatedly self-documented as **not production-ready**: `docker-compose.yml`'s
own header banner, `docker/README.md` (which lists what works vs. what's
missing, plus a future plan referenced as "BL-013"), and
`docs/architecture/DEPLOYMENT.md` (the deployment source of truth) all agree
that this setup lacks parity with the real deployment and must not be treated
as how KZA is actually run.

The real deployment is the systemd `--user` unit described (with its own
repo-vs-production drift caveat) in
[`../deployment/kza-voice-systemd.md`](../deployment/kza-voice-systemd.md) —
per `docs/architecture/DEPLOYMENT.md`, containers were explicitly rejected
for production due to direct USB device access requirements (ReSpeaker mic
array) and the <300ms latency budget.

Included in this bundle because it is a real, non-trivial tracked directory
representing a genuine (if experimental) alternate deployment path — not
because it should be relied on.
