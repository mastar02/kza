#!/usr/bin/env python3
"""Smoke test en DRY-RUN de la cadena de comandos de voz.

Verifica que un comando canónico llegaría a ejecutarse — sin ejecutarlo:

    frase → vector search (Chroma) → entidad viva en HA → payload que HA acepta

No llama a ningún servicio, así que es seguro correrlo en producción y tantas
veces como haga falta.

Por qué existe (2026-07-30): tres fallos llegaron a producción sin que nada
avisara, y los tres empezaron con el usuario diciendo "no funciona":

  * un slot `entity` del LLM hacía que HA rechazara el payload entero;
  * un `--wipe` de Chroma sin API key dejó el índice con solo escenas;
  * entidades `unavailable` aceptan la llamada y no hacen nada.

Los tres son detectables sin tocar el hogar.

Uso:
    python tools/smoke_test.py
    python tools/smoke_test.py --config config/settings.yaml -v

Exit code 0 si todo pasa, 1 si algo falla (sirve para encadenar en scripts).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from src.monitoring.smoke_check import (  # noqa: E402
    STAGE_OK,
    check_phrase,
    entity_problem,
    indexed_entity_ids,
)

# Frases canónicas: cubren los caminos que más se rompieron. Deliberadamente
# escritas como las diría el usuario, no como las normaliza el pipeline.
FRASES = [
    "prendé la luz del living",
    "apagá la luz del living",
    "prendé la luz de la cocina",
    "prendé la luz del escritorio",
    "poné la luz del living al 50 por ciento",
]

VERDE = "\033[32m"
ROJO = "\033[31m"
GRIS = "\033[90m"
RESET = "\033[0m"


def _load_env_file(path: Path) -> None:
    """Cargar KEY=VALUE de un .env al entorno, sin pisar lo ya definido."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _fetch_states(url: str, token: str, timeout: float = 10.0) -> dict[str, str]:
    """GET /api/states → {entity_id: state}. Solo lectura."""
    req = urllib.request.Request(
        f"{url.rstrip('/')}/api/states",
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    return {e["entity_id"]: e["state"] for e in data}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/settings.yaml")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="mostrar el payload y la similitud de cada frase")
    args = ap.parse_args()

    for env_path in (Path("/home/kza/secrets/.env"), Path(".env")):
        _load_env_file(env_path)

    config = yaml.safe_load(Path(args.config).read_text())

    ha_url = os.environ.get("HOME_ASSISTANT_URL") or (
        config.get("home_assistant", {}) or {}
    ).get("url")
    ha_token = os.environ.get("HOME_ASSISTANT_TOKEN")
    if not ha_url or not ha_token:
        print(f"{ROJO}✗ falta HOME_ASSISTANT_URL/TOKEN en el entorno o .env{RESET}")
        return 1

    print(f"{GRIS}Home Assistant: {ha_url}{RESET}")
    try:
        ha_states = _fetch_states(ha_url, ha_token)
    except (urllib.error.URLError, OSError, ValueError) as e:
        print(f"{ROJO}✗ no pude leer los estados de HA: {e}{RESET}")
        return 1
    print(f"{GRIS}  {len(ha_states)} entidades{RESET}\n")

    # Chroma + embedder (BGE-M3 en CPU según settings → no toca la GPU).
    from src.vectordb.chroma_sync import ChromaSync

    vectordb_config = config.get("vectordb", {}) or {}
    embeddings_config = config.get("embeddings", {}) or {}
    chroma = ChromaSync(
        chroma_path=vectordb_config.get("path", "./data/chroma_db"),
        embedder_model=embeddings_config["model"],
        embedder_device=embeddings_config["device"],
        excluded_entities=vectordb_config.get("exclude_entities"),
        excluded_patterns=vectordb_config.get("exclude_patterns"),
    )
    threshold = vectordb_config.get("threshold", 0.65)

    fallos = 0

    print("Comandos canónicos (dry-run, no se ejecuta nada):")
    for frase in FRASES:
        try:
            resolved = chroma.search_command(frase, threshold=threshold)
        except Exception as e:  # noqa: BLE001 — un tool de diagnóstico no debe explotar
            print(f"  {ROJO}✗{RESET} {frase!r}\n      vector search falló: {e}")
            fallos += 1
            continue

        r = check_phrase(frase, resolved, ha_states)
        if r.ok:
            extra = ""
            if args.verbose:
                sim = f" sim={r.similarity:.2f}" if r.similarity is not None else ""
                extra = f"{GRIS}  data={r.service_data}{sim}{RESET}"
            print(f"  {VERDE}✓{RESET} {frase!r} → {r.detail}{extra}")
        else:
            fallos += 1
            print(f"  {ROJO}✗{RESET} {frase!r}")
            print(f"      [{r.stage}] {r.detail}")

    # La cobertura la define el sistema (qué hay indexado en Chroma, o sea
    # qué se puede pedir por voz), no una lista a mano: el `default_light`
    # de cada room dejaba fuera entidades vivas del índice (2026-07-31:
    # cuarto/balcón/escalera no son `default_light` de ninguna room y
    # resolvían por voz con similitud 0.92-1.00 sin que el smoke test las
    # viera). Unimos el índice con los `default_light` del config en vez de
    # reemplazar uno por otro: así seguimos detectando un `default_light`
    # que quedara fuera del índice (excluido o con drift de config) — algo
    # que hoy SÍ se chequea y que perderíamos si solo mirásemos Chroma.
    print("\nEntidades direccionables por voz (índice + rooms):")
    try:
        entity_ids = set(indexed_entity_ids(chroma.commands))
    except Exception as e:  # noqa: BLE001 — un tool de diagnóstico no debe explotar
        print(f"  {ROJO}✗ no pude leer el índice de Chroma: {e}{RESET}")
        fallos += 1
        entity_ids = set()

    for room_cfg in (config.get("rooms") or {}).values():
        if isinstance(room_cfg, dict) and room_cfg.get("default_light"):
            entity_ids.add(room_cfg["default_light"])

    for entity_id in sorted(entity_ids):
        problem = entity_problem(entity_id, ha_states)
        if problem is None:
            estado = ha_states[entity_id]
            print(f"  {VERDE}✓{RESET} {entity_id} ({estado})")
        else:
            fallos += 1
            print(f"  {ROJO}✗{RESET} {problem}")

    print()
    if fallos:
        print(f"{ROJO}✗ {fallos} problema(s){RESET}")
        return 1
    print(f"{VERDE}✓ todo en orden{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
