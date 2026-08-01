#!/usr/bin/env python3
"""Poller externo: avisa por Home Assistant si un micrófono quedó sordo.

Corre FUERA de kza-voice a propósito. Los dos incidentes de sordera (27h y 7h)
tuvieron el servicio en `active` todo el tiempo, así que ni systemd ni una task
interna podían detectarlos.

Uso:
    python tools/audio_watchdog_alert.py --once     # un chequeo, para cron
    python tools/audio_watchdog_alert.py            # bucle
"""
import argparse
import json
import os
import sys
import time

import requests

DEFAULT_HEALTH = "/home/kza/app/data/audio_health.json"


def notify_ha(base_url: str, token: str, title: str, message: str) -> bool:
    """Crear una notificación persistente en HA. Devuelve si HA la aceptó."""
    resp = requests.post(
        f"{base_url}/api/services/persistent_notification/create",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "title": title,
            "message": message,
            "notification_id": "kza_audio_deaf",
        },
        timeout=10,
    )
    return resp.status_code == 200


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--health-path", default=os.environ.get("KZA_AUDIO_HEALTH", DEFAULT_HEALTH))
    ap.add_argument("--deaf-after-s", type=float, default=300.0)
    ap.add_argument("--interval-s", type=float, default=60.0)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    base_url = os.environ.get("HOME_ASSISTANT_URL", "http://localhost:8123").rstrip("/")
    token = os.environ.get("HOME_ASSISTANT_TOKEN", "")
    if not token:
        print("HOME_ASSISTANT_TOKEN no está seteado", file=sys.stderr)
        return 2

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.monitoring.audio_health import evaluate_health

    while True:
        try:
            with open(args.health_path) as fh:
                snapshot = json.load(fh)
            deaf = evaluate_health(snapshot, time.time(), args.deaf_after_s)
        except FileNotFoundError:
            # El archivo no existe: kza-voice nunca arrancó o no llegó a
            # escribirlo. Eso también es una anomalía que hay que reportar.
            deaf = ["(sin snapshot de audio)"]
        except Exception as e:
            print(f"error leyendo {args.health_path}: {e}", file=sys.stderr)
            deaf = []

        if deaf:
            msg = "Sin audio de: " + ", ".join(deaf)
            print(msg)
            notify_ha(base_url, token, "KZA quedó sordo", msg)

        if args.once:
            return 1 if deaf else 0
        time.sleep(args.interval_s)


if __name__ == "__main__":
    raise SystemExit(main())
