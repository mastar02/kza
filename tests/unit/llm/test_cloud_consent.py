"""Tests for the cloud consent gate (privacy)."""

from src.core.settings_schema import DEFAULT_LOCAL_LLM_GATEWAY
from src.llm.cloud_consent import (
    cloud_reasoner_allowed,
    is_cloud_endpoint,
    resolve_compaction_endpoint,
    resolve_http_reasoner_base_url,
    resolve_reasoner_gate,
)


def test_cloud_blocked_without_consent():
    cfg = {"http_base_url": "https://api.minimax.io/v1", "cloud": {"consent": False}}
    assert cloud_reasoner_allowed(cfg) is False


def test_cloud_allowed_with_consent():
    cfg = {"http_base_url": "https://api.minimax.io/v1", "cloud": {"consent": True}}
    assert cloud_reasoner_allowed(cfg) is True


def test_loopback_gateway_requires_cloud_consent():
    cfg = {"http_base_url": "http://127.0.0.1:8200/v1", "cloud": {"consent": False}}
    assert cloud_reasoner_allowed(cfg) is False


def test_is_cloud_endpoint():
    assert is_cloud_endpoint("https://api.minimax.io/v1") is True
    assert is_cloud_endpoint("http://127.0.0.1:8200/v1") is True
    assert is_cloud_endpoint("http://localhost:8101/v1") is True
    # IPv6 loopback: urlparse("http://[::1]:8200/v1").hostname == "::1"
    assert is_cloud_endpoint("http://[::1]:8200/v1") is True
    # Unknown destination also applies to a bind-all address
    assert is_cloud_endpoint("http://0.0.0.0:8200/v1") is True


def test_unresolved_placeholder_is_classified_as_cloud():
    """An unresolved destination never proves local processing."""
    assert is_cloud_endpoint("${LLM_GATEWAY_URL}") is True


def test_gate_evaluated_before_fallback_is_applied():
    """Resolving a gateway placeholder preserves the cloud consent decision."""
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
    # gateway resuelto (sigue siendo cloud, nunca el literal opaco).
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
    """Consented compaction inherits the configured cloud endpoint and credentials."""
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
    """Denied cloud compaction selects the declared local 7B without cloud keys."""
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

    assert is_cloud_endpoint(endpoint.base_url, destination=endpoint.destination) is False
    assert endpoint.api_key_env is None
    assert endpoint.model != reasoner_config["http_model"]
    # Pin the known local backend: comparing against its own constant would
    # miss an accidental retargeting to the cloud gateway.
    assert endpoint.base_url == "http://127.0.0.1:8101/v1"


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
    assert is_cloud_endpoint(endpoint.base_url, destination=endpoint.destination) is False
    assert endpoint.api_key_env is None


def test_explicit_override_inherits_reasoner_credentials_when_gate_allows():
    """A consented cloud override inherits model, API style and credentials."""
    reasoner_config = {
        "http_base_url": "https://api.minimax.io/v1",
        "http_model": "MiniMax-M2.7-highspeed",
        "api_key_env": "MINIMAX_API_KEY",
        "api_style": "chat",
        "cloud": {"consent": True},
    }
    compaction_cfg = {
        "enabled": True,
        "base_url": "http://192.168.1.50:9000/v1",
    }
    gate_allowed = cloud_reasoner_allowed(reasoner_config)
    assert gate_allowed is True

    endpoint = resolve_compaction_endpoint(compaction_cfg, reasoner_config, gate_allowed)

    assert endpoint.base_url == compaction_cfg["base_url"]
    assert endpoint.model == reasoner_config["http_model"]
    assert endpoint.api_key_env == reasoner_config["api_key_env"]
    assert endpoint.api_style == reasoner_config["api_style"]


def test_gate_blocked_never_yields_cloud_endpoint():
    """Unknown overrides, including loopback, cannot bypass denied cloud consent."""
    gate_allowed = False
    reasoner_config = {
        "http_base_url": "https://api.minimax.io/v1",
        "http_model": "MiniMax-M2.7-highspeed",
        "api_key_env": "MINIMAX_API_KEY",
        "api_style": "chat",
        "cloud": {"consent": False},
    }
    base_url_scenarios = {
        "ausente": None,
        "cloud": "https://api.minimax.io/v1",
        "loopback_8200": "http://127.0.0.1:8200/v1",
        "local_8101": "http://127.0.0.1:8101/v1",
    }

    for label, explicit_base_url in base_url_scenarios.items():
        compaction_cfg = {"enabled": True}
        if explicit_base_url is not None:
            compaction_cfg["base_url"] = explicit_base_url

        endpoint = resolve_compaction_endpoint(compaction_cfg, reasoner_config, gate_allowed)

        assert is_cloud_endpoint(endpoint.base_url, destination=endpoint.destination) is False, f"[{label}] {endpoint=}"
        assert endpoint.api_key_env is None, f"[{label}] {endpoint=}"
        assert endpoint.model != "MiniMax-M2.7-highspeed", f"[{label}] {endpoint=}"


# --- Agujeros encontrados en el review fresco 2026-08-02 (ronda 3) ---


def test_local_base_url_override_cannot_be_cloud():
    """A custom fallback without a local declaration cannot receive private text."""
    reasoner_config = {
        "http_base_url": "https://api.minimax.io/v1",
        "http_model": "MiniMax-M2.7-highspeed",
        "api_key_env": "MINIMAX_API_KEY",
        "api_style": "chat",
        "cloud": {"consent": False},
    }
    compaction_cfg = {"enabled": True, "local_base_url": "https://api.minimax.io/v1"}

    endpoint = resolve_compaction_endpoint(compaction_cfg, reasoner_config, gate_allowed=False)

    assert is_cloud_endpoint(endpoint.base_url, destination=endpoint.destination) is False
    assert endpoint.api_key_env is None
    # Literal a propósito (mismo motivo que test_compaction_falls_back_to_local_
    # when_gate_blocks): comparar contra la constante que la mutación mueve es
    # tautológico.
    assert endpoint.base_url == "http://127.0.0.1:8101/v1"


def test_local_base_url_override_is_honored_when_actually_local():
    """An explicitly declared local fallback remains usable without cloud consent."""
    reasoner_config = {"http_base_url": "https://api.minimax.io/v1", "cloud": {"consent": False}}
    compaction_cfg = {"enabled": True, "local_destination": "local", "local_base_url": "http://127.0.0.1:9000/v1"}

    endpoint = resolve_compaction_endpoint(compaction_cfg, reasoner_config, gate_allowed=False)

    assert endpoint.base_url == "http://127.0.0.1:9000/v1"


def test_gate_is_evaluated_when_reasoner_mode_is_not_http():
    """``mode != "http"`` NO implica "no hay nada que bloquear".

    ``main.py`` hardcodeaba ``gate_allowed = True`` para esa rama con el
    argumento de que sin reasoner cloud no hay egreso que gatear. Falso: el
    compactor sigue leyendo ``reasoner_config["http_base_url"]`` (que apunta
    al gateway → MiniMax) por la rama (b). Con ``mode: "local"`` —o un typo
    en ``mode``, porque la condición es ``== "http"``— ``consent:false``
    quedaba en no-op y la conversación del hogar salía igual, con la
    ``MINIMAX_API_KEY`` real.
    """
    reasoner_config = {
        "http_base_url": "http://192.168.1.2:8200/v1",  # LLM_GATEWAY_URL real
        "http_model": "MiniMax-M2.7-highspeed",
        "api_key_env": "MINIMAX_API_KEY",
        "api_style": "chat",
        "cloud": {"consent": False},
    }

    for mode in ("local", "htpp", "", "disabled"):
        gate_allowed, _ = resolve_reasoner_gate(
            reasoner_config, mode, DEFAULT_LOCAL_LLM_GATEWAY
        )
        assert gate_allowed is False, f"[mode={mode!r}] el gate no debe abrirse"

        endpoint = resolve_compaction_endpoint({"enabled": True}, reasoner_config, gate_allowed)
        assert is_cloud_endpoint(endpoint.base_url, destination=endpoint.destination) is False, f"[mode={mode!r}] {endpoint=}"
        assert endpoint.api_key_env is None, f"[mode={mode!r}] {endpoint=}"


def test_gate_allows_non_http_mode_when_consent_is_true():
    """Contrapeso: con consent:true, ``mode != "http"`` no bloquea nada.

    Sin esto, "devolver siempre False fuera de http" pasaría el test de
    arriba y apagaría el compactor cloud de quien tiene consent dado.
    """
    reasoner_config = {
        "http_base_url": "http://192.168.1.2:8200/v1",
        "cloud": {"consent": True},
    }

    gate_allowed, _ = resolve_reasoner_gate(
        reasoner_config, "local", DEFAULT_LOCAL_LLM_GATEWAY
    )

    assert gate_allowed is True


def test_gate_for_http_mode_still_resolves_placeholder():
    """HTTP resolution replaces the placeholder without granting consent."""
    reasoner_config = {
        "http_base_url": "${LLM_GATEWAY_URL}",
        "cloud": {"consent": False},
    }

    gate_allowed, resolved = resolve_reasoner_gate(
        reasoner_config, "http", DEFAULT_LOCAL_LLM_GATEWAY
    )

    assert gate_allowed is False
    assert resolved == DEFAULT_LOCAL_LLM_GATEWAY


# --- resolve_reasoner_gate: branch nuevo para mode="hermes_cli" ---
#
# Hermes corre como subproceso local (no HTTP client): no hay base_url que
# is_cloud_endpoint pueda evaluar. Igual sale de la máquina siempre (a la
# cuenta ChatGPT del usuario), así que se trata como cloud incondicional,
# gateado SOLO por reasoner.cloud.consent.


def test_hermes_cli_mode_blocked_without_consent():
    """hermes_cli es cloud incondicional — sin URL que evaluar, solo consent."""
    cfg = {"cloud": {"consent": False}}
    gate_allowed, resolved = resolve_reasoner_gate(cfg, "hermes_cli", DEFAULT_LOCAL_LLM_GATEWAY)
    assert gate_allowed is False


def test_hermes_cli_mode_allowed_with_consent():
    cfg = {"cloud": {"consent": True}}
    gate_allowed, resolved = resolve_reasoner_gate(cfg, "hermes_cli", DEFAULT_LOCAL_LLM_GATEWAY)
    assert gate_allowed is True


def test_hermes_cli_mode_returns_no_url():
    """No hay http_base_url que resolver para este modo — a diferencia de mode='http'."""
    cfg = {"cloud": {"consent": True}}
    _, resolved = resolve_reasoner_gate(cfg, "hermes_cli", DEFAULT_LOCAL_LLM_GATEWAY)
    assert resolved is None


def test_hermes_cli_mode_ignores_leftover_http_base_url():
    """CLI processing is cloud regardless of any leftover HTTP endpoint."""
    cfg = {
        "http_base_url": "http://127.0.0.1:8200/v1",  # no debería importar acá
        "cloud": {"consent": False},
    }
    gate_allowed, resolved = resolve_reasoner_gate(cfg, "hermes_cli", DEFAULT_LOCAL_LLM_GATEWAY)
    assert gate_allowed is False
    assert resolved is None


def test_hermes_cli_mode_defaults_to_blocked_when_consent_key_missing():
    """Fail-closed: sin la key cloud.consent, hermes_cli no se habilita solo."""
    cfg = {}
    gate_allowed, _ = resolve_reasoner_gate(cfg, "hermes_cli", DEFAULT_LOCAL_LLM_GATEWAY)
    assert gate_allowed is False
