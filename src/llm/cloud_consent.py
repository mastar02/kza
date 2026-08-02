"""Gate de consentimiento para reasoners cloud (privacidad).

El reasoner cloud manda datos del usuario (transcripción, historial, estado del
hogar) a un tercero — rompe la premisa 100%-local. Requiere consent explícito
en config para activarse. Endpoints localhost no requieren consent.
"""

from __future__ import annotations

import logging
from urllib.parse import urlparse

from src.core.settings_schema import is_unresolved_placeholder

logger = logging.getLogger(__name__)

_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}  # nosec B104 -- no es un bind: es el set de hosts locales del gate de privacidad (is_cloud_endpoint compara contra esto, no abre socket)


def is_cloud_endpoint(base_url: str) -> bool:
    """True si base_url no es localhost (sale de la máquina)."""
    host = urlparse(base_url).hostname or ""
    return host not in _LOCAL_HOSTS


def cloud_reasoner_allowed(reasoner_config: dict) -> bool:
    """¿Está permitido instanciar este reasoner?

    - Endpoint local → siempre permitido.
    - Endpoint cloud → solo si reasoner.cloud.consent es True.
    """
    base_url = reasoner_config.get("http_base_url", "")
    if not is_cloud_endpoint(base_url):
        return True
    consent = bool(reasoner_config.get("cloud", {}).get("consent", False))
    if not consent:
        logger.warning(
            "Reasoner cloud %s NO instanciado: reasoner.cloud.consent=false. "
            "Activar consent para enviar datos del usuario al tercero.",
            base_url,
        )
    return consent


def resolve_http_reasoner_base_url(
    reasoner_config: dict, default_local_url: str
) -> tuple[bool, str]:
    """Evalúa el gate de consent y RECIÉN DESPUÉS resuelve el placeholder.

    Invariante de privacidad (Critical cerrado 2026-08-02, Task 5 "CI a
    verde"): el gate tiene que ver ``http_base_url`` TAL COMO ESTÁ ESCRITO en
    la config, placeholder incluido. ``is_cloud_endpoint`` clasifica un
    ``"${VAR}"`` sin resolver como cloud (host vacío ∉ ``_LOCAL_HOSTS``) —
    igual que clasificaría la URL real una vez resuelta (LAN/gateway, nunca
    localhost). Si el fallback local pisara el placeholder ANTES de este
    check, ``is_cloud_endpoint`` vería ``"127.0.0.1"`` y el gate pasaría de
    fail-closed a fail-open: alcanzaría con que falte ``LLM_GATEWAY_URL`` en
    el ``.env`` para saltear ``consent=false`` y arrancar igual el reasoner
    cloud.

    Esta función es la ÚNICA fuente de verdad para ese orden — no lo
    reimplementes inline en ``main()``. Invertir las dos líneas de abajo
    reabre el Critical en silencio; por eso vive acá, chica y testeada, en
    vez de inline en una función gigante y no-testeable (ver
    ``tests/unit/llm/test_cloud_consent.py::test_gate_evaluated_before_fallback_is_applied``,
    que falla si se invierte el orden).

    No cubre el bypass del compactor (`src/main.py`, reusa este mismo dict
    sin pasar por este gate — preexistente, fuera de alcance, ver informe de
    Task 5 §8).

    Args:
        reasoner_config: dict de config del reasoner. Se muta in-place si
            ``http_base_url`` es un placeholder sin resolver (mismo dict que
            reusa el compactor más abajo en ``main()``).
        default_local_url: fallback cuando el placeholder no se resolvió.

    Returns:
        Tupla ``(gate_allowed, resolved_base_url)``.
    """
    gate_allowed = cloud_reasoner_allowed(reasoner_config)

    if is_unresolved_placeholder(reasoner_config.get("http_base_url")):
        logger.warning(
            "reasoner.http_base_url sin resolver (¿falta LLM_GATEWAY_URL en "
            f".env?) — usando fallback local {default_local_url}"
        )
        reasoner_config["http_base_url"] = default_local_url

    return gate_allowed, reasoner_config.get("http_base_url", default_local_url)
