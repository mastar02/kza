"""Gate de consentimiento para reasoners cloud (privacidad).

El reasoner cloud manda datos del usuario (transcripción, historial, estado del
hogar) a un tercero — rompe la premisa 100%-local. Requiere consent explícito
en config para activarse. Endpoints localhost no requieren consent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from urllib.parse import urlparse

from src.core.settings_schema import DEFAULT_LOCAL_LLM_GATEWAY, is_unresolved_placeholder

logger = logging.getLogger(__name__)

_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}  # nosec B104 -- no es un bind: es el set de hosts locales del gate de privacidad (is_cloud_endpoint compara contra esto, no abre socket)

DEFAULT_COMPACTION_LOCAL_URL = "http://127.0.0.1:8101/v1"
"""Fallback local del compactor cuando el gate bloquea el reasoner cloud.

Mismo endpoint que ya usa el ambient distiller (``src/ambient/distiller.py``)
para una tarea más difícil (extracción de hechos en JSON) sobre la misma
clase de dato (conversación del hogar) — precedente escrito y desplegado,
ver diseño en ``.superpowers/sdd/compactor-consent/diseno.md``.
"""

DEFAULT_COMPACTION_LOCAL_MODEL = "local"
"""Model id por defecto para el fallback local del compactor (igual que el distiller)."""


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

    El bypass del compactor (`src/main.py`, que reusaba este mismo dict sin
    pasar por este gate) YA NO existe: ver `resolve_compaction_endpoint` más
    abajo, que sí lo cubre desde el PR de `cloud.consent` para el compactor
    (`.superpowers/sdd/compactor-consent/`).

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


@dataclass(frozen=True, slots=True)
class CompactionEndpoint:
    """Endpoint HTTP resuelto para el compactor de contexto (Plan #2 OpenClaw).

    Devuelto por ``resolve_compaction_endpoint`` — nunca construido a mano en
    ``main()``, para que el orden de resolución quede en un solo lugar
    testeado (mismo patrón que ``resolve_http_reasoner_base_url``).
    """

    base_url: str
    model: str | None
    api_key_env: str | None
    api_style: str


def resolve_compaction_endpoint(
    compaction_cfg: dict, reasoner_config: dict, gate_allowed: bool
) -> CompactionEndpoint:
    """Resuelve a qué LLM manda el compactor sus turnos, respetando el gate.

    El compactor (``src/orchestrator/compactor.py``) manda al LLM la
    transcripción literal de la conversación (``_build_prompt``) — es la
    misma clase de dato que el reasoner principal, así que tiene que pasar
    por el mismo gate de privacidad. Antes de esta función, ``main.py``
    heredaba ``reasoner_config`` (base_url/model/api_key_env) sin mirar
    ``gate_allowed`` — con ``cloud.consent=false`` el reasoner principal se
    apagaba pero el compactor seguía mandando datos a MiniMax igual. Ver
    ``.superpowers/sdd/compactor-consent/diseno.md``.

    Orden de resolución (comportamiento con consent:true queda
    byte-idéntico al que había antes de este gate):

    a. ``compaction.base_url`` explícito Y (no es cloud O ``gate_allowed``)
       → usar ese ``base_url``. Si ``gate_allowed`` es True, hereda
       model/api_key_env/api_style de ``reasoner_config`` tal cual (camino
       de producción actual, sin cambios). Si ``gate_allowed`` es False,
       el override de ``base_url`` se respeta pero model/api_key_env/
       api_style NO se heredan de ``reasoner_config`` — mismo pineo que la
       rama (c): ``api_key_env=None`` siempre. Sin esto (Important #1 y #2
       del review 2026-08-02), un override *local* documentado en este
       mismo archivo (``local_base_url: "http://127.0.0.1:8101/v1"``,
       arriba) o el loopback ``:8200`` (que ``is_cloud_endpoint`` clasifica
       como no-cloud por el fail-open ya conocido, bug aparte y fuera de
       alcance de este PR) igual heredaban ``api_key_env=MINIMAX_API_KEY``:
       o revienta ``_resolve_api_key`` contra un endpoint sin esa var (la
       misma degradación silenciosa que este gate vino a eliminar) o, peor,
       el :8200 reenvía con la key real a MiniMax — egreso cloud real con
       ``consent=false``.
    b. ``gate_allowed`` → heredar ``reasoner_config`` tal cual (camino de
       producción actual, sin cambios).
    c. Gate bloqueado (y sin override local) → degradar a LLM local:
       ``compaction.local_base_url`` (default
       ``DEFAULT_COMPACTION_LOCAL_URL``, el mismo :8101 que ya usa el
       ambient distiller para conversación del hogar), model
       ``compaction.local_model`` (default ``DEFAULT_COMPACTION_LOCAL_MODEL``),
       ``api_key_env=None`` SIEMPRE (nunca heredar la key del reasoner
       cloud: si el compactor hereda ``MINIMAX_API_KEY`` para un endpoint
       local, ``_resolve_api_key`` revienta con ``RuntimeError`` en
       ``load()``, el try/except de ``main.py`` lo traga y
       ``compactor=None`` — exactamente la degradación silenciosa que este
       gate viene a eliminar), ``api_style="chat"`` pineado (no heredar: el
       bloque de rollback de emergencia en settings.yaml pondría
       "completions").
    d. Si (c) falla en ``load()`` → el try/except ya existente en
       ``main.py`` deja ``compactor=None`` (equivale a no compactar).
       Degradación explícita, no requiere manejo nuevo acá.

    Args:
        compaction_cfg: ``orchestrator.context.compaction`` de settings.yaml.
        reasoner_config: config del reasoner principal (mismo dict que ya
            pasó por ``resolve_http_reasoner_base_url``, o sin tocar si
            ``reasoner.mode == "local"``).
        gate_allowed: resultado de ``cloud_reasoner_allowed`` /
            ``resolve_http_reasoner_base_url`` para el reasoner principal —
            el compactor NO evalúa el gate de nuevo, reusa la misma
            decisión (mismo consent, mismo dato).

    Returns:
        ``CompactionEndpoint`` listo para pasarle a ``HttpReasoner(**...)``.
    """
    explicit_base_url = compaction_cfg.get("base_url")
    if explicit_base_url and (not is_cloud_endpoint(explicit_base_url) or gate_allowed):
        if gate_allowed:
            model = compaction_cfg.get("model") or reasoner_config.get("http_model")
            api_key_env = reasoner_config.get("api_key_env")
            api_style = reasoner_config.get("api_style", "completions")
        else:
            # Important #1/#2 (review 2026-08-02): el override de base_url se
            # respeta (por eso llegamos acá con gate_allowed=False — la
            # condición de arriba ya filtró los cloud "de verdad"), pero las
            # credenciales/modelo NO se heredan de reasoner_config. Mismo
            # pineo que la rama (c) de abajo, mismo motivo: si el override
            # apunta a un endpoint sin MINIMAX_API_KEY seteada (p.ej. el
            # :8101 documentado más arriba), _resolve_api_key revienta y el
            # try/except de main.py lo traga → compactor=None, la misma
            # degradación silenciosa que este gate existe para eliminar; si
            # apunta al loopback :8200 (fail-open conocido de
            # is_cloud_endpoint, bug aparte), heredar la key mandaría la
            # conversación a MiniMax igual con consent=false.
            logger.warning(
                "compaction.base_url=%s con cloud.consent=false: se respeta "
                "el override de base_url pero NO se heredan credenciales/"
                "modelo del reasoner cloud.",
                explicit_base_url,
            )
            model = compaction_cfg.get("model") or DEFAULT_COMPACTION_LOCAL_MODEL
            api_key_env = None
            api_style = "chat"
        return CompactionEndpoint(
            base_url=explicit_base_url,
            model=model,
            api_key_env=api_key_env,
            api_style=api_style,
        )

    if gate_allowed:
        return CompactionEndpoint(
            base_url=reasoner_config.get("http_base_url", DEFAULT_LOCAL_LLM_GATEWAY),
            model=compaction_cfg.get("model") or reasoner_config.get("http_model"),
            api_key_env=reasoner_config.get("api_key_env"),
            api_style=reasoner_config.get("api_style", "completions"),
        )

    local_base_url = compaction_cfg.get("local_base_url") or DEFAULT_COMPACTION_LOCAL_URL
    local_model = compaction_cfg.get("local_model") or DEFAULT_COMPACTION_LOCAL_MODEL
    logger.warning(
        "compaction degradada a LLM local (%s) por cloud.consent=false — "
        "el resumen de contexto ya no sale a MiniMax.",
        local_base_url,
    )
    return CompactionEndpoint(
        base_url=local_base_url,
        model=local_model,
        api_key_env=None,
        api_style="chat",
    )
