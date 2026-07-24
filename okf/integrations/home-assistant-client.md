---
type: component
title: Home Assistant Client
description: Outbound REST + WebSocket client from KZA to Home Assistant, with graceful degradation when HA is unreachable.
resource: file:///Users/yo/Documents/kza/src/home_assistant/
tags: [kza, home-assistant, integration, rest, websocket]
timestamp: "2026-07-24T00:00:00Z"
---

# Home Assistant Client

`src/home_assistant/ha_client.py` is KZA's outbound integration with Home
Assistant: it calls HA's REST API to execute actions (turn on a light, set a
thermostat) and keeps a WebSocket connection for state/entity-registry
updates. Its docstring explicitly emphasizes graceful degradation — defined
timeout constants and an `HAConnectionState` enum suggest the client is
designed to keep the voice assistant partially functional even when Home
Assistant itself is down or unreachable, rather than failing the whole
pipeline.

This is the "outbound" half of the KZA↔HA relationship; the "inbound" half —
where Home Assistant is configured to expose KZA back to itself as sensors,
services and automations — is
[`../integrations/home-assistant-config-push.md`](../integrations/home-assistant-config-push.md).

Entity data pulled/pushed here is what
[`../datastores/vectordb-ha-sync.md`](../datastores/vectordb-ha-sync.md)
indexes for fast command resolution; actions resolved on the fast path are
ultimately executed by calling into this client.
