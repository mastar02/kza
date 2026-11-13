"""Resolve model and training-output paths from the central configuration.

U3b-0e (design v3, 2b; reviews F2 and N10): a managed child runs with
cwd = the immutable slot runtime, which has no ``models`` entry. A
cwd-relative ``./models`` default would write into that runtime (or fail),
so every model path in code goes through :func:`models_path`, rooted at
``paths.models_root`` in ``config/settings.yaml``.

Rules:
    - Without an admission ticket (development), a missing
      ``paths.models_root`` falls back to ``./models``.
    - Under a ticket (a managed child), a missing root refuses with
      ``runtime_admission_models_root_missing``, and a relative one with
      ``runtime_admission_models_root_relative``: a relative root would
      resolve inside the read-only runtime.
    - ``speaker_id.savedir`` follows the same rules through
      :func:`speaker_savedir`.
"""
from __future__ import annotations

from collections.abc import Mapping
import copy
import functools
import os
from pathlib import PurePosixPath

from src.core.runtime_admission import TICKET_ENV

DEVELOPMENT_MODELS_ROOT = "./models"
# Training outputs live in the state (runtime/data links to state/data), never
# under the shared models directory (design v3 2b).
TRAINING_OUTPUT_ROOT = "./data/training_outputs"
TRAINING_OUTPUT_LORA = TRAINING_OUTPUT_ROOT + "/lora"
TRAINING_OUTPUT_NIGHTLY = TRAINING_OUTPUT_ROOT + "/nightly"
TRAINING_OUTPUT_WHISPER = TRAINING_OUTPUT_ROOT + "/whisper"


def development_settings(*, environ: Mapping[str, str] | None = None) -> dict:
    """The central configuration's paths for constructor defaults (review F10).

    Without a ticket, defaults of classes built without configuration
    (WakeWordDetector, WakeWordTrainer, SpeakerIdentifier, ModelManager) and
    of the development scripts follow ``paths.models_root`` and
    ``speaker_id.savedir`` from ``CONFIG_PATH`` (default
    config/settings.yaml). Under a ticket this returns {}, so those defaults
    refuse; the serve graph always passes explicit values. An unreadable or
    absent file gives {} (the ./models fallback).
    """
    environ = os.environ if environ is None else environ
    if _managed(environ):
        return {}
    path = os.path.abspath(environ.get("CONFIG_PATH", "config/settings.yaml"))
    try:
        info = os.stat(path)
    except OSError:
        return {}
    return copy.deepcopy(_development_settings(path, info.st_mtime_ns, info.st_size))


@functools.lru_cache(maxsize=8)
def _development_settings(path: str, _mtime_ns: int, _size: int) -> dict:
    import yaml
    try:
        with open(path, encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except (OSError, ValueError, yaml.YAMLError):
        return {}
    if not isinstance(loaded, dict):
        return {}
    keep = {}
    for name in ("paths", "speaker_id"):
        if isinstance(loaded.get(name), dict):
            keep[name] = dict(loaded[name])
    return keep


def _managed(environ: Mapping[str, str] | None) -> bool:
    return TICKET_ENV in (os.environ if environ is None else environ)


def _section(config: Mapping | None, name: str) -> Mapping:
    value = (config or {}).get(name) if isinstance(config, Mapping) else None
    return value if isinstance(value, Mapping) else {}


def _checked_parts(parts: tuple[str, ...]) -> tuple[str, ...]:
    for part in parts:
        if not isinstance(part, str) or not part:
            raise ValueError("models_path_part_invalid")
        pure = PurePosixPath(part)
        if pure.is_absolute() or ".." in pure.parts:
            raise ValueError("models_path_part_invalid")
    return parts


def join_models_root(root: str | os.PathLike | None, *parts: str,
                     environ: Mapping[str, str] | None = None) -> str:
    """Join ``parts`` under an explicit models root with the ticket rules.

    Args:
        root: The configured ``paths.models_root`` value (None or '' if absent).
        *parts: Relative components under the root (no '..', not absolute).
        environ: Environment to read the admission ticket from (os.environ).

    Returns:
        The joined path as text.

    Raises:
        RuntimeError: Under a ticket, when the root is missing or relative.
        ValueError: When a part is absolute, empty or contains '..'.
    """
    _checked_parts(parts)
    managed = _managed(environ)
    if root is None or (isinstance(root, str) and not root):
        if managed:
            raise RuntimeError("runtime_admission_models_root_missing")
        root = DEVELOPMENT_MODELS_ROOT
    elif not isinstance(root, (str, os.PathLike)):
        raise RuntimeError("runtime_admission_models_root_invalid")
    elif managed and not os.path.isabs(os.fspath(root)):
        raise RuntimeError("runtime_admission_models_root_relative")
    text = os.fspath(root)
    return os.path.join(text, *parts) if parts else text


def models_root(config: Mapping | None, *, environ: Mapping[str, str] | None = None) -> str:
    """Return ``paths.models_root`` with the development fallback and ticket rules."""
    return join_models_root(_section(config, "paths").get("models_root"), environ=environ)


def models_path(config: Mapping | None, *parts: str, environ: Mapping[str, str] | None = None) -> str:
    """Return ``<paths.models_root>/<parts...>``.

    Args:
        config: The central configuration (the whole settings dictionary).
        *parts: Relative components under the models root.
        environ: Environment to read the admission ticket from (os.environ).

    Returns:
        The model path as text.
    """
    return join_models_root(_section(config, "paths").get("models_root"), *parts, environ=environ)


def speaker_savedir(config: Mapping | None, model_name: str, *,
                    environ: Mapping[str, str] | None = None) -> str:
    """Return the SpeechBrain cache directory for ``model_name``.

    ``speaker_id.savedir`` wins when set; otherwise ``<models_root>/speaker``.
    The model name becomes one directory (``/`` replaced by ``_``), as the
    legacy ``./models/speaker/<id>`` layout did.

    Raises:
        RuntimeError: Under a ticket, when the savedir is relative, or when it
            is absent and the models root is missing or relative.
    """
    leaf = model_name.replace("/", "_")
    configured = _section(config, "speaker_id").get("savedir")
    if configured is None or (isinstance(configured, str) and not configured):
        return models_path(config, "speaker", leaf, environ=environ)
    if not isinstance(configured, (str, os.PathLike)):
        raise RuntimeError("runtime_admission_speaker_savedir_invalid")
    if _managed(environ) and not os.path.isabs(os.fspath(configured)):
        raise RuntimeError("runtime_admission_speaker_savedir_relative")
    return os.path.join(os.fspath(configured), *_checked_parts((leaf,)))


PIPER_DEFAULT_MODEL = "es_ES-davefx-medium.onnx"
PIPER_ENGINES = frozenset({"piper", "hybrid"})


def model_file(config: Mapping | None, value: str, *subdir: str,
               environ: Mapping[str, str] | None = None) -> str:
    """Resolve a model file value: a bare file name goes under ``<models_root>/<subdir>``.

    A value with a directory part or an absolute path is returned unchanged
    (the preflight judges those). Review F3: a bare name such as
    'es_ES-davefx-medium.onnx' would otherwise be read relative to the cwd,
    which under management is the read-only runtime.
    """
    if not isinstance(value, str) or not value or os.path.isabs(value) or "/" in value or value in {".", ".."}:
        return value
    return models_path(config, *subdir, value, environ=environ)


def tts_settings(config: Mapping | None, *, environ: Mapping[str, str] | None = None) -> dict:
    """The ``tts`` section with tts.piper.model resolved, when the engine uses Piper.

    Other engines (Kokoro/Qwen3 'dual', XTTS) are returned unchanged, so a
    dormant Piper value never requires the models root.
    """
    tts = dict(_section(config, "tts"))
    if tts.get("engine", "piper") in PIPER_ENGINES:
        piper = dict(tts.get("piper") or {})
        piper["model"] = model_file(config, piper.get("model") or PIPER_DEFAULT_MODEL, "piper", environ=environ)
        tts["piper"] = piper
    return tts
