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
from typing import Optional

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

logger = logging.getLogger(__name__)


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
        if isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
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
