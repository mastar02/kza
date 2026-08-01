"""Publica el heartbeat de audio para que un proceso externo pueda leerlo.

El watchdog interno (multi_room_audio_loop._stream_watchdog) sabe si un mic
entrega audio, pero ese dato nunca sale del proceso: durante 27h de sordera
systemd mostró `active` y nadie tuvo cómo enterarse. Este módulo escribe un
snapshot que un poller externo consume — externo a propósito, porque el modo
de falla observado es "el proceso vive y no entrega audio".
"""
import contextlib
import json
import logging
import os
import tempfile

logger = logging.getLogger(__name__)


def write_audio_health(
    path: str,
    rooms: dict[str, tuple[float, float]],
    now_wall: float,
    now_mono: float,
) -> None:
    """Escribir el snapshot de salud de audio de forma atómica.

    Args:
        path: destino del JSON.
        rooms: room_id -> (last_frame_ts, opened_ts), ambos monotónicos.
            last_frame_ts == 0.0 significa "todavía no entregó ningún frame".
        now_wall: time.time() — referencia para que el lector, que corre en
            otro proceso, pueda detectar un snapshot viejo.
        now_mono: time.monotonic() — para convertir los ts a edades.
    """
    payload = {
        "wall": now_wall,
        "rooms": {
            room_id: {
                "age_s": (now_mono - last_ts) if last_ts > 0.0 else (now_mono - opened_ts),
                "ever": last_ts > 0.0,
            }
            for room_id, (last_ts, opened_ts) in rooms.items()
        },
    }
    directory = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        # mkstemp crea el temporal en 0600 (solo el dueño puede leerlo), pero
        # el poller externo (tools/audio_watchdog_alert.py) está pensado para
        # correr bajo OTRO usuario que este proceso — ese es justamente el
        # punto de que sea externo. Sin este chmod, el poller recibía
        # PermissionError al leer y lo trataba como "todo bien" para
        # siempre (ver check_once en audio_watchdog_alert.py). os.replace
        # es un rename POSIX: preserva los permisos del inodo del
        # temporal, así que hay que fijarlos ANTES del replace.
        os.fchmod(fd, 0o644)
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh)
        os.replace(tmp, path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def evaluate_health(
    snapshot: dict,
    now_wall: float,
    deaf_after_s: float,
    first_frame_grace_s: float = 180.0,
    snapshot_stale_after_s: float = 120.0,
) -> list[str]:
    """Devolver las rooms consideradas sordas.

    Un snapshot viejo implica que el escritor dejó de escribir — el proceso
    está trabado — y eso cuenta como sordera de TODAS las rooms: el silencio
    del vigilante no puede significar "todo bien".

    La gracia de primer frame es 180s porque los arranques medidos van de
    1.5-2s (lo normal) a 135s (el peor caso observado).
    """
    rooms = snapshot.get("rooms", {})
    snapshot_age = now_wall - snapshot.get("wall", 0.0)
    if snapshot_age > snapshot_stale_after_s:
        return sorted(rooms)

    deaf = []
    for room_id, info in rooms.items():
        threshold = deaf_after_s if info.get("ever") else first_frame_grace_s
        if info.get("age_s", 0.0) > threshold:
            deaf.append(room_id)
    return sorted(deaf)
