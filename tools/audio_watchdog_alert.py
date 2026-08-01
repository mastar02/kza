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
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def notify_ha(base_url: str, token: str, title: str, message: str) -> bool:
    """Crear una notificación persistente en HA. Devuelve si HA la aceptó.

    Nunca propaga: este poller existe para avisar cuando OTRO proceso falló
    en silencio, así que si él mismo se muere porque HA está caído o
    inalcanzable, sería el mismo patrón de fallo silencioso — nadie se
    entera, y encima ya no queda ni el reintento del próximo ciclo. Ante
    cualquier problema de red/timeout se loguea a stderr (journald lo
    captura si esto corre como servicio/timer) y se devuelve False; la
    firma ya promete un bool, "no pude entregar" es un valor de retorno,
    no una excepción.
    """
    try:
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
    except requests.exceptions.RequestException as e:
        print(f"no pude avisar a HA ({base_url}): {e}", file=sys.stderr)
        return False


def check_once(health_path: str, deaf_after_s: float, base_url: str, token: str) -> list[str]:
    """Ejecutar un chequeo: leer el snapshot, evaluar, avisar si hace falta.

    Devuelve la lista de rooms sordas (vacía = todo sano). Importa
    `evaluate_health` en cada llamada, no al nivel de módulo, para no exigir
    que `src/` esté en el path solo por importar este archivo. El insert
    está guardado por membership: sin el chequeo, cada vuelta del bucle
    agregaba una entrada nueva a `sys.path` sin límite (medido: 200 vueltas
    → 201 entradas duplicadas), y cada import subsiguiente escanea esa
    lista linealmente.
    """
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    from src.monitoring.audio_health import evaluate_health

    try:
        with open(health_path) as fh:
            snapshot = json.load(fh)
        deaf = evaluate_health(snapshot, time.time(), deaf_after_s)
    except FileNotFoundError:
        # El archivo no existe: kza-voice nunca arrancó o no llegó a
        # escribirlo. Eso también es una anomalía que hay que reportar.
        deaf = ["(sin snapshot de audio)"]
    except Exception as e:
        # Cualquier otra falla de lectura/parseo (JSON corrupto, forma
        # incorrecta como una lista en vez de un dict, PermissionError si
        # el poller corre bajo otro usuario, lo que sea) es, en un
        # vigilante, una anomalía — nunca "todo bien". Antes esta rama
        # devolvía deaf=[], que apagaba la alarma en silencio para siempre
        # si el snapshot quedaba ilegible: el mismo patrón de fallo
        # silencioso que este poller existe para eliminar. El principio:
        # cualquier cosa que no se pueda confirmar como sana es sorda.
        print(f"snapshot ilegible en {health_path}: {e}", file=sys.stderr)
        deaf = ["(snapshot ilegible)"]

    if deaf:
        msg = "Sin audio de: " + ", ".join(deaf)
        print(msg)
        notify_ha(base_url, token, "KZA quedó sordo", msg)

    return deaf


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--health-path", default=os.environ.get("KZA_AUDIO_HEALTH", DEFAULT_HEALTH))
    ap.add_argument("--deaf-after-s", type=float, default=300.0)
    ap.add_argument("--interval-s", type=float, default=60.0)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args(argv)

    base_url = os.environ.get("HOME_ASSISTANT_URL", "http://localhost:8123").rstrip("/")
    token = os.environ.get("HOME_ASSISTANT_TOKEN", "")
    if not token:
        print("HOME_ASSISTANT_TOKEN no está seteado", file=sys.stderr)
        return 2

    while True:
        try:
            deaf = check_once(args.health_path, args.deaf_after_s, base_url, token)
        except Exception as e:
            # Ninguna excepción de una vuelta puede matar el bucle: un
            # watchdog que se cae cuando lo que vigila no responde es peor
            # que no tener watchdog (su silencio se lee como "todo bien").
            # Se loguea y se sigue durmiendo hasta la próxima vuelta. En
            # --once, un error nunca reporta "sano": no hay evidencia de
            # que no haya sordera.
            print(f"error inesperado en el ciclo del poller: {e}", file=sys.stderr)
            if args.once:
                return 1
            time.sleep(args.interval_s)
            continue

        if args.once:
            return 1 if deaf else 0
        time.sleep(args.interval_s)


if __name__ == "__main__":
    raise SystemExit(main())
