"""Gate de consentimiento para reasoners cloud (privacidad).

El reasoner cloud manda datos del usuario (transcripción, historial, estado del
hogar) a un tercero — rompe la premisa 100%-local. Requiere consent explícito
en config para activarse. Endpoints localhost no requieren consent.
"""

from __future__ import annotations

import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}


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
