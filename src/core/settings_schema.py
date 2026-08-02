"""Schema Pydantic de config/settings.yaml — validación al boot.

Valida la estructura mínima que el pipeline necesita para arrancar:
presencia de las secciones núcleo y los campos que src/main.py exige
(embeddings siempre; speaker_id/emotion model+device solo cuando el
subsistema está habilitado, igual que main.py; HA url).

El schema es deliberadamente permisivo (extra="allow" en todos los
niveles): settings.yaml crece con cada feature y el objetivo es atrapar
configs rotas al boot con un error claro, no modelar las ~1500 líneas.
"""

import logging
import os
from typing import Optional

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

logger = logging.getLogger(__name__)

DEFAULT_LOCAL_LLM_GATEWAY = "http://127.0.0.1:8200/v1"
"""Fallback cuando ``${LLM_GATEWAY_URL}`` no se resuelve (falta .env).

El gateway LiteLLM (:8200) corre en el mismo host físico que kza-voice y
kza-code-index, así que loopback es una dirección que de hecho funciona
(no un placeholder inerte) — ver el bloque de rollback comentado en
config/settings.yaml junto a ``reasoner.http_base_url``. Usarlo como
default en el punto de consumo evita que el literal sin resolver
``"${LLM_GATEWAY_URL}"`` llegue a un cliente HTTP (HttpReasoner /
openai.AsyncOpenAI), que fallaría recién en el primer uso del slow path
con un error opaco de conexión, lejos del boot.
"""


def is_unresolved_placeholder(value: object) -> bool:
    """True si value es un string ``${VAR}`` que replace_env_vars no resolvió."""
    return isinstance(value, str) and value.startswith("${") and value.endswith("}")


def resolve_env_vars(obj):
    """Reemplazar recursivamente placeholders ``${VAR}`` por su valor de entorno.

    Un placeholder cuya env var no está seteada queda como el literal
    ``${VAR}`` (comportamiento de ``os.getenv(var, default=obj)``) — el
    caller decide si eso es fatal (``check_unresolved_env_vars``, solo
    para paths críticos) o si hay un fallback seguro en el punto de
    consumo (p. ej. ``DEFAULT_LOCAL_LLM_GATEWAY``).

    Args:
        obj: Valor de config (dict/list/str/lo que sea) recién leído del YAML.

    Returns:
        El mismo obj con cada ``${VAR}`` reemplazado por su valor de entorno,
        recursivamente.
    """
    if is_unresolved_placeholder(obj):
        var_name = obj[2:-1]
        return os.getenv(var_name, obj)
    if isinstance(obj, dict):
        return {k: resolve_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_env_vars(item) for item in obj]
    return obj


class _Section(BaseModel):
    """Base permisiva: cualquier clave extra es válida."""

    model_config = ConfigDict(extra="allow")


class HomeAssistantSettings(_Section):
    url: str


class EmbeddingsSettings(_Section):
    """main.py exige embeddings.model y embeddings.device siempre."""

    model: str
    device: str


class ModelDeviceSettings(_Section):
    """Secciones con modelo en GPU/CPU.

    main.py exige model y device solo cuando la sección está habilitada
    (enabled default True, como ``config.get("enabled", True)`` en main);
    este schema replica esa condición para no rechazar configs válidas
    con el subsistema apagado.
    """

    enabled: bool = True
    model: Optional[str] = None
    device: Optional[str] = None

    @model_validator(mode="after")
    def _require_model_device_when_enabled(self) -> "ModelDeviceSettings":
        if self.enabled and (self.model is None or self.device is None):
            raise ValueError("model y device son requeridos cuando enabled=true")
        return self


class SettingsSchema(_Section):
    """Secciones que una config sana siempre declara.

    Su ausencia casi siempre indica un settings.yaml roto/truncado; las
    estrictamente boot-blocking en main.py son home_assistant.url y
    embeddings.model/device.
    """

    home_assistant: HomeAssistantSettings
    audio: dict
    wake_word: dict
    stt: dict
    tts: dict
    router: dict
    reasoner: dict
    vectordb: dict
    embeddings: EmbeddingsSettings
    speaker_id: ModelDeviceSettings
    emotion: ModelDeviceSettings


def check_unresolved_env_vars(
    config: dict, critical_prefixes: tuple = ("home_assistant.",)
) -> None:
    """Detectar placeholders ``${VAR}`` que quedaron sin resolver.

    Un placeholder sin resolver significa que falta el .env o que el
    EnvironmentFile del service no cargó — sin este chequeo el error
    aflora aguas abajo como "no se puede conectar" o un auth opaco de
    Spotify, lejos de la causa real.

    Args:
        config: Configuración post reemplazo de env vars.
        critical_prefixes: Paths (con punto final implícito en sección)
            cuyos placeholders abortan el boot. El resto solo loguea
            WARNING (p. ej. credenciales de Spotify: el pipeline puede
            arrancar sin música).

    Raises:
        ValueError: Si hay placeholders sin resolver bajo un path crítico.
    """

    def _walk(obj, path=""):
        if is_unresolved_placeholder(obj):
            return [(path, obj)]
        if isinstance(obj, dict):
            return [
                hit
                for k, v in obj.items()
                for hit in _walk(v, f"{path}.{k}" if path else str(k))
            ]
        if isinstance(obj, list):
            return [hit for i, v in enumerate(obj) for hit in _walk(v, f"{path}[{i}]")]
        return []

    unresolved = _walk(config)
    if not unresolved:
        return
    detail = ", ".join(f"{path}={value}" for path, value in unresolved)
    critical = [p for p, _ in unresolved if p.startswith(critical_prefixes)]
    if critical:
        raise ValueError(
            f"Variables de entorno sin resolver en settings (¿falta .env / "
            f"EnvironmentFile del service?): {detail}"
        )
    logger.warning(f"[SettingsSchema] Env vars sin resolver (¿falta .env?): {detail}")


def validate_settings(config: dict) -> dict:
    """Validar el dict de settings contra el schema.

    Args:
        config: Configuración ya cargada (post reemplazo de env vars).

    Returns:
        El mismo dict (pass-through), para encadenar en load_config.

    Raises:
        ValueError: Si la config no es un dict o no cumple el schema.
            El mensaje lista cada campo inválido con su ubicación.
    """
    if not isinstance(config, dict):
        raise ValueError(
            f"settings.yaml inválido: se esperaba un mapping, llegó {type(config).__name__}"
        )
    try:
        SettingsSchema.model_validate(config)
    except ValidationError as e:
        detail = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()
        )
        logger.error(f"[SettingsSchema] settings.yaml no pasa validación: {detail}")
        raise ValueError(f"settings.yaml no pasa validación: {detail}") from e
    return config
