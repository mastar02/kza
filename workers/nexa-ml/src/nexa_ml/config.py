"""Typed ``[ml]`` configuration loaded from ``/etc/nexa/nexa.toml`` (stdlib ``tomllib``).

Every key is optional so that new worker-only knobs never require editing the
shared ``nexa.toml``; unknown keys are logged and ignored (forward compatibility
with newer deploy subsets).
"""

import dataclasses
import logging
import pathlib
import re
import tomllib
import types
import typing
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = pathlib.Path("/etc/nexa/nexa.toml")
_DEVICE_RE = re.compile(r"^(cpu|cuda:\d{1,2})$")
_LANG_RE = re.compile(r"^[a-z]{2,3}$")


class ConfigError(ValueError):
    """Raised when the ``[ml]`` section is malformed."""


@dataclass(frozen=True)
class MlConfig:
    """Worker configuration (the ``[ml]`` table of ``nexa.toml``).

    Attributes:
        command_stt_model: Whisper alias/path for the low-latency COMMAND slot.
        command_stt_device: Torch/CTranslate2 device for COMMAND (``cuda:N`` or ``cpu``).
        command_stt_compute_type: CTranslate2 compute type for COMMAND.
        command_vad_filter: faster-whisper ``vad_filter`` for COMMAND (prod: off).
        command_beam_size: Default beam when the request leaves ``beam_size`` at 0.
        ambient_stt_model: Whisper alias/path for the AMBIENT pool.
        ambient_stt_device: Device for every AMBIENT pool member.
        ambient_stt_compute_type: CTranslate2 compute type for AMBIENT.
        ambient_stt_workers: Number of independent AMBIENT model instances.
        ambient_vad_filter: faster-whisper ``vad_filter`` for AMBIENT (on).
        ambient_beam_size: Default AMBIENT beam when the request leaves it at 0.
        stt_language: Default ISO language when the request leaves it empty.
        max_audio_seconds: Upper bound for any PCM payload.
        tts_model: Kokoro Hugging Face repo id.
        tts_voice: Default Kokoro voice.
        tts_lang_code: Default Kokoro ``KPipeline`` language code.
        tts_device: Kokoro device.
        speaker_encoder_model: SpeechBrain ECAPA-TDNN source.
        speaker_encoder_device: ECAPA device (default ``cuda:0``).
        embeddings_model: Sentence-transformers model id (BGE-M3).
        embeddings_device: Embedding device (CPU by policy).
        embed_max_texts: Maximum texts per ``EmbedText`` call.
        models_dir: Optional local model root (download/cache dir); ``None`` uses library defaults.
        offline: Force Hugging Face offline mode (the unit enforces ``IPAddressDeny=any``).
        max_pending: Maximum queued+running requests per slot before ``RESOURCE_EXHAUSTED``.
        startup_timeout_s: Budget for initial model loads before ``READY=1``.
    """

    command_stt_model: str = "whisper-v3-turbo"
    command_stt_device: str = "cuda:1"
    command_stt_compute_type: str = "int8_float16"
    command_vad_filter: bool = False
    command_beam_size: int = 1
    ambient_stt_model: str = "whisper-large-v3"
    ambient_stt_device: str = "cuda:0"
    ambient_stt_compute_type: str = "int8_float16"
    ambient_stt_workers: int = 2
    ambient_vad_filter: bool = True
    ambient_beam_size: int = 5
    stt_language: str = "es"
    max_audio_seconds: float = 120.0
    tts_model: str = "hexgrad/Kokoro-82M"
    tts_voice: str = "ef_dora"
    tts_lang_code: str = "e"
    tts_device: str = "cuda:1"
    speaker_encoder_model: str = "speechbrain/spkrec-ecapa-voxceleb"
    speaker_encoder_device: str = "cuda:0"
    embeddings_model: str = "BAAI/bge-m3"
    embeddings_device: str = "cpu"
    embed_max_texts: int = 64
    models_dir: str | None = None
    offline: bool = True
    max_pending: int = 8
    startup_timeout_s: float = 110.0

    def __post_init__(self) -> None:
        """Validate ranges and device strings."""
        for name in (
            "command_stt_device",
            "ambient_stt_device",
            "tts_device",
            "speaker_encoder_device",
            "embeddings_device",
        ):
            if not _DEVICE_RE.match(getattr(self, name)):
                raise ConfigError(f"ml.{name} must be 'cpu' or 'cuda:N'")
        if not 1 <= self.ambient_stt_workers <= 4:
            raise ConfigError("ml.ambient_stt_workers must be within 1..4")
        for name in ("command_beam_size", "ambient_beam_size"):
            if not 1 <= getattr(self, name) <= 10:
                raise ConfigError(f"ml.{name} must be within 1..10")
        if not _LANG_RE.match(self.stt_language):
            raise ConfigError("ml.stt_language must be an ISO 639 code")
        if not 1.0 <= self.max_audio_seconds <= 600.0:
            raise ConfigError("ml.max_audio_seconds must be within 1..600")
        if not 1 <= self.embed_max_texts <= 1024:
            raise ConfigError("ml.embed_max_texts must be within 1..1024")
        if not 1 <= self.max_pending <= 256:
            raise ConfigError("ml.max_pending must be within 1..256")
        if not 1.0 <= self.startup_timeout_s <= 600.0:
            raise ConfigError("ml.startup_timeout_s must be within 1..600")


def _coerce(name: str, annotation: object, value: object) -> object:
    """Strictly type-check one TOML value against its dataclass annotation."""
    if isinstance(annotation, types.UnionType):
        if value is None:
            return None
        annotation = next(a for a in typing.get_args(annotation) if a is not type(None))
    if annotation is bool:
        if type(value) is not bool:
            raise ConfigError(f"ml.{name} must be a boolean")
        return value
    if annotation is int:
        if type(value) is not int:
            raise ConfigError(f"ml.{name} must be an integer")
        return value
    if annotation is float:
        if type(value) not in (int, float):
            raise ConfigError(f"ml.{name} must be a number")
        return float(value)
    if annotation is str:
        if type(value) is not str or not value:
            raise ConfigError(f"ml.{name} must be a non-empty string")
        return value
    raise ConfigError(f"ml.{name} has an unsupported type")  # pragma: no cover


def parse_ml_section(document: dict[str, object]) -> MlConfig:
    """Build an :class:`MlConfig` from a parsed TOML document.

    Args:
        document: Full ``nexa.toml`` document as returned by ``tomllib``.

    Returns:
        The validated configuration; defaults fill any missing key.

    Raises:
        ConfigError: If ``[ml]`` is not a table or a value has the wrong type/range.
    """
    section = document.get("ml", {})
    if not isinstance(section, dict):
        raise ConfigError("[ml] must be a table")
    hints = typing.get_type_hints(MlConfig)
    known = {f.name for f in dataclasses.fields(MlConfig)}
    values: dict[str, object] = {}
    for key, value in section.items():
        if key not in known:
            logger.warning("Ignoring unknown ml config key %s", key)
            continue
        values[key] = _coerce(key, hints[key], value)
    return MlConfig(**values)


def load_config(path: pathlib.Path = DEFAULT_CONFIG_PATH) -> MlConfig:
    """Read and validate the ``[ml]`` section of a ``nexa.toml`` file.

    Args:
        path: Path to the TOML file.

    Returns:
        The validated :class:`MlConfig`.

    Raises:
        ConfigError: If the file is missing, unreadable, not TOML or invalid.
    """
    try:
        with path.open("rb") as handle:
            document = tomllib.load(handle)
    except OSError as exc:
        raise ConfigError(f"cannot read config {path}: {exc.strerror}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from exc
    return parse_ml_section(document)
