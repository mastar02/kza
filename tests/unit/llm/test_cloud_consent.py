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
    # Literal, NO la constante DEFAULT_COMPACTION_LOCAL_URL (Important
    # diferido, review 2026-08-02 ronda 2 / Mutación B): is_cloud_endpoint
    # clasifica CUALQUIER loopback como no-cloud, así que si alguien repunta
    # la constante de :8101 a :8200 (u otro puerto loopback), el assert de
    # arriba sigue en verde — comparar contra la constante misma que la
    # mutación mueve es tautológico y no lo detecta. El literal es la única
    # forma de que este test grite si el fallback local deja de ser :8101.
    # No "simplificar" esto de vuelta a la constante sin agregar una
    # aserción equivalente que fije el valor real.
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
    assert is_cloud_endpoint(endpoint.base_url) is False
    assert endpoint.api_key_env is None


def test_explicit_override_inherits_reasoner_credentials_when_gate_allows():
    """Rama (a) + ``gate_allowed=True``: código nuevo de este PR, sin test propio.

    (Important diferido, review 2026-08-02 ronda 2 / Mutación C.)
    ``test_compaction_inherits_cloud_endpoint_when_gate_allows`` cubre la
    rama (b) (sin ``compaction.base_url``); ``test_explicit_cloud_override_
    does_not_bypass_gate`` y la matriz de arriba cubren la rama (a) con
    ``gate_allowed=False``. Pero la rama (a) con ``gate_allowed=True`` —el
    sub-camino que el diff de este PR agrega (antes de esto, cualquier
    ``base_url`` explícito heredaba credenciales incondicionalmente, sin
    if/else)— no tenía ningún test que la ejercitara. Una regresión ahí
    (p.ej. dejar de heredar model/api_key_env/api_style al "simplificar" el
    if/else) rompería el override documentado de ``compaction.base_url`` con
    ``consent:true`` EN SILENCIO: la suite completa seguía en verde porque
    ningún otro test cruza override explícito × gate permitido.

    Usa un ``base_url`` de override que NO coincide con
    ``reasoner_config["http_base_url"]`` a propósito, para distinguir esta
    rama (a) de la (b): si el resolver tomara la (b) por error, el
    ``base_url`` del endpoint sería el del reasoner, no el del override, y
    el primer assert ya fallaría.
    """
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
    """Matriz: con el gate bloqueado, ningún vector de bypass entrega cloud.

    Cruza los DOS ejes que juntos deciden qué rama de
    ``resolve_compaction_endpoint`` corre y qué credenciales devuelve:

    - ``compaction_cfg["base_url"]`` (el override explícito que dispara la
      rama (a)) ∈ {ausente, cloud, loopback :8200, local :8101}.
    - ``gate_allowed`` fijo en False en los 4 casos.

    Versión anterior de este test (Important #3, review 2026-08-02): variaba
    ``reasoner_config["http_base_url"]`` en vez de ``compaction_cfg["base_url"]``
    en 3 de los 4 escenarios — con ``compaction_cfg["base_url"]`` ausente, esos
    3 caían siempre en la rama (c) (que ni lee ``reasoner_config``), así que
    en la práctica ejercitaban el mismo único camino tres veces y jamás
    tocaban la rama (a) con un override no-cloud. Por eso no atraparon los
    Important #1 (override local ``:8101`` heredando la key cloud) y #2
    (override loopback ``:8200`` heredando la key cloud) — ambos viven
    exclusivamente en la rama (a).

    Este test SÍ varía ``compaction_cfg["base_url"]`` directamente, así que
    cubre las 4 combinaciones reales: "ausente" y "cloud" ya pasaban antes de
    este fix (control — la rama (a) exige ``not is_cloud_endpoint(...) or
    gate_allowed``, así que un override cloud con el gate bloqueado nunca
    entra a la rama (a)); "loopback_8200" y "local_8101" SÍ fallaban contra
    el código sin fix, porque ``is_cloud_endpoint`` los clasifica como
    no-cloud (correctamente para :8101, por el fail-open ya conocido y fuera
    de alcance para :8200) y la rama (a) heredaba model/api_key_env de
    ``reasoner_config`` sin mirar ``gate_allowed``.

    Este es el test que atrapa al PRÓXIMO consumidor: si alguien reescribe
    las condiciones o el cuerpo de la rama (a) de forma que alguno de estos 4
    vectores se cuele con el gate bloqueado, este test queda rojo.
    """
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

        assert is_cloud_endpoint(endpoint.base_url) is False, f"[{label}] {endpoint=}"
        assert endpoint.api_key_env is None, f"[{label}] {endpoint=}"
        assert endpoint.model != "MiniMax-M2.7-highspeed", f"[{label}] {endpoint=}"


# --- Agujeros encontrados en el review fresco 2026-08-02 (ronda 3) ---


def test_local_base_url_override_cannot_be_cloud():
    """El fallback local NO es una puerta trasera al cloud.

    ``compaction.local_base_url`` lo introduce este mismo PR y lo documenta
    en settings.yaml, pero la rama (c) lo usaba SIN validar: bastaba
    apuntarlo a MiniMax para que ``consent:false`` egresara igual — y encima
    el logger anunciaba "degradada a LLM local ... ya no sale a MiniMax"
    mientras devolvía la URL de MiniMax (log mentiroso, el patrón raíz de
    los fallos silenciosos de KZA).

    La matriz ``test_gate_blocked_never_yields_cloud_endpoint`` no lo
    atrapaba porque varía ``compaction_cfg["base_url"]``, nunca
    ``local_base_url`` — otro eje del mismo parámetro, no un eje nuevo.
    """
    reasoner_config = {
        "http_base_url": "https://api.minimax.io/v1",
        "http_model": "MiniMax-M2.7-highspeed",
        "api_key_env": "MINIMAX_API_KEY",
        "api_style": "chat",
        "cloud": {"consent": False},
    }
    compaction_cfg = {"enabled": True, "local_base_url": "https://api.minimax.io/v1"}

    endpoint = resolve_compaction_endpoint(compaction_cfg, reasoner_config, gate_allowed=False)

    assert is_cloud_endpoint(endpoint.base_url) is False
    assert endpoint.api_key_env is None
    # Literal a propósito (mismo motivo que test_compaction_falls_back_to_local_
    # when_gate_blocks): comparar contra la constante que la mutación mueve es
    # tautológico.
    assert endpoint.base_url == "http://127.0.0.1:8101/v1"


def test_local_base_url_override_is_honored_when_actually_local():
    """Contrapeso del test de arriba: un override local legítimo SÍ se respeta.

    Sin este test, "ignorar siempre ``local_base_url``" pasaría el test
    anterior y rompería la key documentada en settings.yaml en silencio.
    """
    reasoner_config = {"http_base_url": "https://api.minimax.io/v1", "cloud": {"consent": False}}
    compaction_cfg = {"enabled": True, "local_base_url": "http://127.0.0.1:9000/v1"}

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
        assert is_cloud_endpoint(endpoint.base_url) is False, f"[mode={mode!r}] {endpoint=}"
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
    """``resolve_reasoner_gate`` con mode="http" delega tal cual al resolver viejo.

    Guard de que la extracción no cambió el camino de producción: gate
    evaluado ANTES del fallback (el Critical de Task 5) y placeholder
    resuelto después.
    """
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
    """Si reasoner_config todavía tiene http_base_url (config vieja sin limpiar),
    el gate de hermes_cli no lo usa para nada — is_cloud_endpoint nunca se llama.
    """
    cfg = {
        "http_base_url": "http://127.0.0.1:8200/v1",  # no debería importar acá
        "cloud": {"consent": False},
    }
    gate_allowed, resolved = resolve_reasoner_gate(cfg, "hermes_cli", DEFAULT_LOCAL_LLM_GATEWAY)
    assert gate_allowed is False  # si is_cloud_endpoint se colara, esto daría True (fail-open)
    assert resolved is None


def test_hermes_cli_mode_defaults_to_blocked_when_consent_key_missing():
    """Fail-closed: sin la key cloud.consent, hermes_cli no se habilita solo."""
    cfg = {}
    gate_allowed, _ = resolve_reasoner_gate(cfg, "hermes_cli", DEFAULT_LOCAL_LLM_GATEWAY)
    assert gate_allowed is False
