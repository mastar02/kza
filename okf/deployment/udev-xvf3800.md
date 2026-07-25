---
type: deployment
title: XVF3800 udev Rule
description: udev rule granting the kza user pyusb access to the XVF3800 mic-array USB device.
resource: file:///Users/yo/Documents/kza/deploy/udev/99-xvf3800.rules
tags: [kza, udev, usb, hardware, microphone]
timestamp: "2026-07-24T00:00:00Z"
---

# XVF3800 udev Rule

`deploy/udev/99-xvf3800.rules` is a udev rule that grants the `kza` system
user direct (non-root) USB access, via `pyusb`, to the XVF3800 microphone
array hardware — the same class of direct-USB-device requirement that
`docs/architecture/DEPLOYMENT.md` cites as the reason containerized
deployment was rejected for production (see
[`../deployment/docker-experimental.md`](../deployment/docker-experimental.md)
and [`../deployment/kza-voice-systemd.md`](../deployment/kza-voice-systemd.md)).

This rule is a small but concrete piece of evidence for that architectural
constraint: the audio input hardware needs to be reachable as a raw USB
device by the process running
[`../pipeline/wake-word.md`](../pipeline/wake-word.md) and
[`../pipeline/voice-pipeline.md`](../pipeline/voice-pipeline.md)'s
`AudioLoop`.
