"""Tests for the cloud consent gate (privacy)."""

from src.core.settings_schema import DEFAULT_LOCAL_LLM_GATEWAY
from src.llm.cloud_consent import (
    DEFAULT_COMPACTION_LOCAL_URL,
    cloud_reasoner_allowed,
    is_cloud_endpoint,
    resolve_compaction_endpoint,
    resolve_http_reasoner_base_url,
)


def test_cloud_blocked_without_consent():
    cfg = {"http_base_url": "https://api.minimax.io/v1", "cloud": {"consent": False}}
    assert cloud_reasoner_allowed(cfg) is False


def test_cloud_allowed_with_consent():
    cfg = {"http_base_url": "https://api.minimax.io/v1", "cloud": {"consent": True}}
    assert cloud_reasoner_allowed(cfg) is True


def test_localhost_always_allowed():
    cfg = {"http_base_url": "http://127.0.0.1:8200/v1", "cloud": {"consent": False}}
    assert cloud_reasoner_allowed(cfg) is True


def test_is_cloud_endpoint():
    assert is_cloud_endpoint("https://api.minimax.io/v1") is True
    assert is_cloud_endpoint("http://127.0.0.1:8200/v1") is False
    assert is_cloud_endpoint("http://localhost:8101/v1") is False
    # IPv6 loopback: urlparse("http://[::1]:8200/v1").hostname == "::1"
    assert is_cloud_endpoint("http://[::1]:8200/v1") is False
    # Bind-all address also treated as local
    assert is_cloud_endpoint("http://0.0.0.0:8200/v1") is False


def test_unresolved_placeholder_is_classified_as_cloud():
    """Un "${VAR}" sin resolver tiene hostname vacío → clasificado cloud.

    Es justo lo que hace que el orden gate→fallback importe: si el gate
    viera el placeholder DESPUÉS de que el fallback local lo pisa, vería
    "127.0.0.1" (local) en vez de esto (cloud) y pasaría fail-open.
    """
    assert is_cloud_endpoint("${LLM_GATEWAY_URL}") is True


def test_gate_evaluated_before_fallback_is_applied():
    """Regresión del Critical cerrado 2026-08-02 (Task 5, "CI a verde").

    resolve_http_reasoner_base_url es la única fuente de verdad para el
    orden gate→fallback (ver su docstring en src/llm/cloud_consent.py).
    Si alguien invierte esas dos líneas — resuelve el placeholder ANTES
    de evaluar el gate — is_cloud_endpoint ve "127.0.0.1" en vez del
    placeholder y gate_allowed pasa a True aunque consent sea False. Este
    test tiene que quedar rojo si eso pasa (verificado moviendo el
    fallback antes del gate en una copia vía `git archive`, no en el
    repo real — ver informe de Task 5).
    """
    # consent=False + placeholder sin resolver → el reasoner NO se habilita,
    # aunque el fallback SÍ se aplique (para que el literal opaco no llegue
    # a ningún cliente HTTP — ver docstring de la función).
    cfg_blocked = {
        "http_base_url": "${LLM_GATEWAY_URL}",
        "cloud": {"consent": False},
    }
    gate_allowed, resolved = resolve_http_reasoner_base_url(
        cfg_blocked, DEFAULT_LOCAL_LLM_GATEWAY
    )
    assert gate_allowed is False
    assert resolved == DEFAULT_LOCAL_LLM_GATEWAY
    assert cfg_blocked["http_base_url"] == DEFAULT_LOCAL_LLM_GATEWAY

    # consent=True + placeholder sin resolver → SÍ se habilita, contra el
    # fallback local (loopback, nunca el literal opaco).
    cfg_allowed = {
        "http_base_url": "${LLM_GATEWAY_URL}",
        "cloud": {"consent": True},
    }
    gate_allowed, resolved = resolve_http_reasoner_base_url(
        cfg_allowed, DEFAULT_LOCAL_LLM_GATEWAY
    )
    assert gate_allowed is True
    assert resolved == DEFAULT_LOCAL_LLM_GATEWAY


def test_gate_blocks_cloud_even_when_url_already_resolved():
    """Caso control: sin placeholder de por medio, consent=False bloquea."""
    cfg = {
        "http_base_url": "https://api.minimax.io/v1",
        "cloud": {"consent": False},
    }
    gate_allowed, resolved = resolve_http_reasoner_base_url(
        cfg, DEFAULT_LOCAL_LLM_GATEWAY
    )
    assert gate_allowed is False
    # No había placeholder — el fallback no debe tocar la URL real.
    assert resolved == "https://api.minimax.io/v1"


# --- resolve_compaction_endpoint: el compactor deja de bypassear el gate ---
#
# Antes de estos tests, main.py construía el HttpReasoner del compactor
# reusando reasoner_config (base_url/model/api_key_env) sin mirar
# gate_allowed — con cloud.consent=false el reasoner principal se apagaba
# pero el compactor (Compactor._build_prompt manda la transcripción literal)
# seguía mandando la conversación a MiniMax igual. Ver
# .superpowers/sdd/compactor-consent/diseno.md.


def test_compaction_inherits_cloud_endpoint_when_gate_allows():
    """consent:true → el compactor hereda el mismo endpoint que el reasoner hoy.

    Guard de "no cambiar el comportamiento actual": este es el camino de
    producción vigente (gate permitido) y tiene que quedar byte-idéntico a
    la construcción inline que reemplaza este resolver.
    """
    reasoner_config = {
        "http_base_url": "https://api.minimax.io/v1",
        "http_model": "MiniMax-M2.7-highspeed",
        "api_key_env": "MINIMAX_API_KEY",
        "api_style": "chat",
        "cloud": {"consent": True},
    }
    compaction_cfg = {"enabled": True}
    gate_allowed = cloud_reasoner_allowed(reasoner_config)
    assert gate_allowed is True

    endpoint = resolve_compaction_endpoint(compaction_cfg, reasoner_config, gate_allowed)

    assert endpoint.base_url == reasoner_config["http_base_url"]
    assert endpoint.model == reasoner_config["http_model"]
    assert endpoint.api_key_env == reasoner_config["api_key_env"]
    assert endpoint.api_style == reasoner_config["api_style"]


def test_compaction_falls_back_to_local_when_gate_blocks():
    """consent:false → el compactor NO hereda el reasoner cloud (bug cerrado).

    Sin este fix, `is_cloud_endpoint(endpoint.base_url)` daría True acá —
    exactamente el leak que este PR elimina.
    """
    reasoner_config = {
        "http_base_url": "https://api.minimax.io/v1",
        "http_model": "MiniMax-M2.7-highspeed",
        "api_key_env": "MINIMAX_API_KEY",
        "api_style": "chat",
        "cloud": {"consent": False},
    }
    compaction_cfg = {"enabled": True}
    gate_allowed = cloud_reasoner_allowed(reasoner_config)
    assert gate_allowed is False

    endpoint = resolve_compaction_endpoint(compaction_cfg, reasoner_config, gate_allowed)

    assert is_cloud_endpoint(endpoint.base_url) is False
    assert endpoint.api_key_env is None
    assert endpoint.model != reasoner_config["http_model"]


def test_explicit_cloud_override_does_not_bypass_gate():
    """compaction.base_url apuntando a cloud + consent:false → NO se honra.

    Si el override explícito ganara contra el gate, alcanzaría con setear
    orchestrator.context.compaction.base_url a mano (sin tocar consent) para
    reabrir el leak — el gate tiene que ganarle al override, no al revés.
    """
    reasoner_config = {
        "http_base_url": "https://api.minimax.io/v1",
        "http_model": "MiniMax-M2.7-highspeed",
        "api_key_env": "MINIMAX_API_KEY",
        "api_style": "chat",
        "cloud": {"consent": False},
    }
    compaction_cfg = {
        "enabled": True,
        "base_url": "https://api.minimax.io/v1",
        "model": "MiniMax-M2.7-highspeed",
    }
    gate_allowed = cloud_reasoner_allowed(reasoner_config)
    assert gate_allowed is False

    endpoint = resolve_compaction_endpoint(compaction_cfg, reasoner_config, gate_allowed)

    assert endpoint.base_url != compaction_cfg["base_url"]
    assert is_cloud_endpoint(endpoint.base_url) is False
    assert endpoint.api_key_env is None


def test_gate_blocked_never_yields_cloud_endpoint():
    """Matriz: con el gate bloqueado, ningún vector de bypass entrega cloud.

    ``gate_allowed`` se pasa YA resuelto (False) en los 4 escenarios —
    ``resolve_compaction_endpoint`` no re-evalúa el gate (ver su docstring),
    así que lo que hay que fijar acá es que la función en sí misma nunca deja
    pasar un endpoint cloud pase lo que pase en compaction_cfg/reasoner_config,
    dado gate_allowed=False. Cubre los 4 vectores encontrados al inventariar
    el bypass original: override explícito a cloud, herencia ciega de
    reasoner_config cloud, placeholder "${...}" sin resolver, y el loopback
    :8200 (gateway LiteLLM que reenvía a MiniMax pero que is_cloud_endpoint
    NO detecta como cloud — ese es un bug aparte, fuera de alcance de este PR;
    lo que este test fija es que, YA CON gate_allowed=False, ninguna forma de
    http_base_url cambia el resultado: siempre el fallback local pineado).

    Este es el test que atrapa al PRÓXIMO consumidor: si alguien reescribe
    las condiciones de las ramas (a)/(b) de resolve_compaction_endpoint de
    forma que alguno de estos 4 vectores se cuele con el gate bloqueado, este
    test queda rojo.
    """
    gate_allowed = False
    base_reasoner_config = {
        "http_model": "MiniMax-M2.7-highspeed",
        "api_key_env": "MINIMAX_API_KEY",
        "api_style": "chat",
        "cloud": {"consent": False},
    }
    scenarios = {
        "override_cloud": (
            {"enabled": True, "base_url": "https://api.minimax.io/v1"},
            {**base_reasoner_config, "http_base_url": "https://api.minimax.io/v1"},
        ),
        "herencia_cloud": (
            {"enabled": True},
            {**base_reasoner_config, "http_base_url": "https://api.minimax.io/v1"},
        ),
        "placeholder_sin_resolver": (
            {"enabled": True},
            {**base_reasoner_config, "http_base_url": "${LLM_GATEWAY_URL}"},
        ),
        "loopback_8200": (
            {"enabled": True},
            {**base_reasoner_config, "http_base_url": "http://127.0.0.1:8200/v1"},
        ),
    }

    for label, (compaction_cfg, reasoner_config) in scenarios.items():
        endpoint = resolve_compaction_endpoint(compaction_cfg, reasoner_config, gate_allowed)

        assert is_cloud_endpoint(endpoint.base_url) is False, f"[{label}] {endpoint=}"
        assert endpoint.base_url == DEFAULT_COMPACTION_LOCAL_URL, f"[{label}] {endpoint=}"
        assert endpoint.api_key_env is None, f"[{label}] {endpoint=}"
        assert endpoint.model != "MiniMax-M2.7-highspeed", f"[{label}] {endpoint=}"
