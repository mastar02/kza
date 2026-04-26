"""
Factory que construye un LLMRouter desde un dict de config.

Espera formato:
    config["llm"]["failover"] = {
        "cooldown_persistence_path": "./data/llm_cooldowns.json",
        "endpoints": [
            {"id": "fast", "kind": "fast_router", "priority": 0,
             "max_tokens_default": 128, "idle_timeout_s": null},
            {"id": "deep", "kind": "http_reasoner", "priority": 1,
             "max_tokens_default": 512, "idle_timeout_s": 30.0},
        ],
    }

`clients` es un dict {endpoint_id: ClienteAdapter} que el caller pasa.
Ya armados (FastRouterAdapter / HttpReasonerAdapter envolviendo los clientes
nativos). Separamos esto del factory para que main.py mantenga la
responsabilidad de cargar/conectar a los servicios reales.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.llm.cooldown import CooldownManager
from src.llm.router import LLMRouter
from src.llm.types import EndpointKind, LLMEndpoint

logger = logging.getLogger(__name__)


def build_llm_router_from_config(
    config: dict,
    clients: dict[str, Any],
) -> LLMRouter:
    """Construir LLMRouter desde config + dict de clientes ya creados."""
    failover_cfg = config.get("llm", {}).get("failover", {})

    persistence_path = Path(
        failover_cfg.get("cooldown_persistence_path", "./data/llm_cooldowns.json")
    )
    cd_manager = CooldownManager(persistence_path=persistence_path)

    endpoints: list[LLMEndpoint] = []
    for ep_cfg in failover_cfg.get("endpoints", []):
        ep_id = ep_cfg["id"]
        client = clients.get(ep_id)
        if client is None:
            logger.warning(
                f"[RouterFactory] endpoint '{ep_id}' configured but no client provided — skipping"
            )
            continue

        kind_raw = ep_cfg["kind"]
        try:
            kind = EndpointKind(kind_raw)
        except ValueError as e:
            raise ValueError(f"unknown endpoint kind '{kind_raw}' for id '{ep_id}'") from e

        endpoints.append(LLMEndpoint(
            id=ep_id,
            kind=kind,
            client=client,
            priority=ep_cfg.get("priority", 0),
            timeout_s=ep_cfg.get("timeout_s", 30.0),
            idle_timeout_s=ep_cfg.get("idle_timeout_s"),
            max_tokens_default=ep_cfg.get("max_tokens_default", 256),
        ))

    if not endpoints:
        raise ValueError("at least one endpoint must be configured with a valid client")

    return LLMRouter(endpoints, cd_manager)
