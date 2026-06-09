"""Schema Pydantic de config/settings.yaml — validación al boot.

Valida la estructura mínima que el pipeline necesita para arrancar:
presencia de las secciones núcleo y los campos que src/main.py exige
explícitamente (embeddings/speaker_id/emotion model+device, HA url).

El schema es deliberadamente permisivo (extra="allow" en todos los
niveles): settings.yaml crece con cada feature y el objetivo es atrapar
configs rotas al boot con un error claro, no modelar las ~1500 líneas.
"""

import logging

from pydantic import BaseModel, ConfigDict, ValidationError

logger = logging.getLogger(__name__)


class _Section(BaseModel):
    """Base permisiva: cualquier clave extra es válida."""

    model_config = ConfigDict(extra="allow")


class HomeAssistantSettings(_Section):
    url: str


class ModelDeviceSettings(_Section):
    """Secciones con modelo en GPU/CPU (main.py exige model y device)."""

    model: str
    device: str


class SettingsSchema(_Section):
    """Secciones núcleo sin las cuales el pipeline no puede armarse."""

    home_assistant: HomeAssistantSettings
    audio: dict
    wake_word: dict
    stt: dict
    tts: dict
    router: dict
    reasoner: dict
    vectordb: dict
    embeddings: ModelDeviceSettings
    speaker_id: ModelDeviceSettings
    emotion: ModelDeviceSettings


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
