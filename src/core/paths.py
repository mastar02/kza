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
