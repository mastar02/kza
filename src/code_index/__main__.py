"""Entry point: python -m src.code_index (servicio en el server, systemd --user)."""

import logging
import os
from pathlib import Path

import yaml
from aiohttp import web

from src.code_index.service import build_indexer, create_app
from src.core.settings_schema import resolve_env_vars


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    config_path = os.environ.get("CONFIG_PATH", "config/settings.yaml")
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    # Este servicio es independiente de src.main.load_config (ver comentario
    # en config/settings.yaml sobre code_index) y hasta acá cargaba la config
    # con yaml.safe_load puro, sin resolver ${VAR} — cards.base_url llegaba
    # sin resolver a CardGenerator si faltaba el .env. resolve_env_vars()
    # replica acá el mismo mecanismo que usa main.py.
    cfg = resolve_env_vars(raw)["code_index"]

    indexer = build_indexer(cfg, repo_root=Path.cwd())
    app = create_app(indexer)
    web.run_app(app, host=cfg.get("host", "0.0.0.0"), port=cfg.get("port", 9510))  # nosec B104 -- bind a LAN deliberado: el servicio code-index se consume desde la red (tools/code_search.py y agentes remotos), no solo localhost


if __name__ == "__main__":
    main()
