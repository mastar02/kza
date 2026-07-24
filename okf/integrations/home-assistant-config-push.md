---
type: deployment
title: KZA-side Home Assistant Configuration
description: YAML pushed into Home Assistant's own config so KZA appears there as sensors, services, scripts and automations.
resource: file:///Users/yo/Documents/kza/config/homeassistant/
tags: [kza, home-assistant, yaml, configuration, integration]
timestamp: "2026-07-24T00:00:00Z"
---

# KZA-side Home Assistant Configuration

`config/homeassistant/` holds YAML meant to be installed inside Home
Assistant's own configuration (per `config/homeassistant/README.md`'s
instructions), not consumed by the KZA Python process itself. It is what
makes KZA show up as a first-class citizen inside HA: sensors reporting KZA
state, services HA automations can call to trigger KZA behavior, helper
scripts, and HA-side automations that react to KZA events.

This is the "inbound" counterpart to
[`../integrations/home-assistant-client.md`](../integrations/home-assistant-client.md),
which is the KZA-side code that talks to HA over REST/WebSocket. Together
they form the bidirectional integration summarized in
[`../system/architecture.md`](../system/architecture.md).

Typed as `deployment` rather than `component` because, unlike the rest of
this bundle's `config/homeassistant` sibling
([`../config/settings.md`](../config/settings.md)), this directory's content
is not loaded by KZA at runtime — it is deployed into a different system
(Home Assistant) as a one-time or periodic install step.
