"""Factory para construir LLMRouter desde dict de configuración.

Lee `config["llm"]["failover"]` (settings.yaml) y un dict de clientes ya
inicializados, y devuelve un LLMRouter listo para inyectar en el dispatcher.

Los clientes (FastRouter, HttpReasoner) se construyen en `main.py` con sus
respectivas configs (URL, modelo, etc.). Esta factory solo arma el wrapper
de routing — no decide cómo se conectan los clientes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.llm.cooldown import CooldownManager
from src.llm.router import LLMRouter
from src.llm.types import EndpointKind, LLMEndpoint

logger = logging.getLogger(__name__)


DEFAULT_COOLDOWN_PATH = "./data/llm_cooldowns.json"


def build_llm_router(
    failover_config: dict,
    clients: dict[str, Any],
    metrics_tracker=None,
) -> LLMRouter:
    """Construir LLMRouter desde config + clientes ya inicializados.

    Args:
        failover_config: Sub-dict `llm.failover` del settings.yaml. Estructura:
            ```yaml
            endpoints:
              - id: fast_router_7b
                kind: fast_router      # mapea a EndpointKind
                priority: 1
                timeout_s: 5.0
                idle_timeout_s: null   # 7B no necesita watchdog
              - id: reasoner_72b
                kind: http_reasoner
                priority: 2
                timeout_s: 30.0
                idle_timeout_s: 8.0    # 72B en CPU se cuelga
            cooldowns:
              persist_path: ./data/llm_cooldowns.json
            ```
        clients: Mapa `{endpoint_id: client_obj}`. El client_obj debe exponer
            `async def complete(prompt: str, max_tokens: int, **kwargs) -> str`.

    Returns:
        LLMRouter inicializado con los endpoints y un CooldownManager.

    Raises:
        ValueError: Si falta un cliente referenciado por la config, o si la
            config no tiene endpoints.
        KeyError: Si un endpoint no tiene `id`, `kind` o `priority`.
    """
    endpoints_cfg = failover_config.get("endpoints") or []
    if not endpoints_cfg:
        raise ValueError("llm.failover.endpoints está vacío")

    endpoints = []
    skipped: list[str] = []
    for ep_cfg in endpoints_cfg:
        ep_id = ep_cfg["id"]
        client = clients.get(ep_id)
        if client is None:
            # Endpoint listado en config pero sin cliente disponible (ej:
            # 72B caído al startup). Lo skippeamos en lugar de fallar — el
            # router queda con menos candidatos pero sigue funcionando.
            logger.warning(
                f"[LLMRouter] skip endpoint {ep_id!r} — sin cliente. "
                f"Clientes disponibles: {list(clients)}"
            )
            skipped.append(ep_id)
            continue
        endpoints.append(LLMEndpoint(
            id=ep_id,
            kind=EndpointKind(ep_cfg["kind"]),
            client=client,
            priority=int(ep_cfg["priority"]),
            timeout_s=float(ep_cfg.get("timeout_s", 30.0)),
            idle_timeout_s=ep_cfg.get("idle_timeout_s"),
            max_tokens_default=int(ep_cfg.get("max_tokens_default", 256)),
        ))

    if not endpoints:
        raise ValueError(
            f"Ningún endpoint disponible — todos skippeados: {skipped}"
        )

    cooldowns_cfg = failover_config.get("cooldowns") or {}
    persist_path = Path(cooldowns_cfg.get("persist_path", DEFAULT_COOLDOWN_PATH))
    persist_path.parent.mkdir(parents=True, exist_ok=True)
    cd_manager = CooldownManager(persistence_path=persist_path)

    logger.info(
        f"[LLMRouter] inicializado con {len(endpoints)} endpoints: "
        + ", ".join(f"{e.id}(prio={e.priority})" for e in endpoints)
    )
    return LLMRouter(endpoints=endpoints, cooldown_manager=cd_manager,
                     metrics_tracker=metrics_tracker)
