"""Entry point: python -m src.code_index (servicio en el server, systemd --user)."""

import logging
import os
from pathlib import Path

import yaml
from aiohttp import web

from src.code_index.service import build_indexer, create_app


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    config_path = os.environ.get("CONFIG_PATH", "config/settings.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)["code_index"]

    indexer = build_indexer(cfg, repo_root=Path.cwd())
    app = create_app(indexer)
    web.run_app(app, host=cfg.get("host", "0.0.0.0"), port=cfg.get("port", 9510))


if __name__ == "__main__":
    main()
