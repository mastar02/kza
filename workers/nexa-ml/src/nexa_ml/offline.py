"""Fail-closed offline enforcement for every real model loader (zero network).

The systemd unit adds ``IPAddressDeny=any``, but the worker must not rely on the
sandbox alone: with ``offline=true`` the Hugging Face flags are *forced* (not
``setdefault``), remote sources are rejected and each loader re-checks that the
effective ``huggingface_hub`` setting is offline before touching a model.
"""

import os
import re
import sys

_URI_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
OFFLINE_ENV = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")


class OfflineViolationError(RuntimeError):
    """A loader would be able to reach the network while offline mode is required."""


def enforce_offline_env() -> None:
    """Force (overwrite) the Hugging Face offline flags in this process."""
    for name in OFFLINE_ENV:
        os.environ[name] = "1"


def assert_offline_effective() -> None:
    """Verify offline mode is in effect for the Hugging Face client.

    ``huggingface_hub`` snapshots ``HF_HUB_OFFLINE`` at import time, so an import
    that happened before :func:`enforce_offline_env` would still be online.

    Raises:
        OfflineViolationError: Flags not forced, or ``huggingface_hub`` already
            imported in online mode.
    """
    for name in OFFLINE_ENV:
        if os.environ.get(name) != "1":
            raise OfflineViolationError(f"{name} must be 1 in offline mode")
    constants = sys.modules.get("huggingface_hub.constants")
    if constants is not None and not getattr(constants, "HF_HUB_OFFLINE", False):
        raise OfflineViolationError("huggingface_hub was imported before offline mode was forced")


def reject_remote_source(source: str) -> None:
    """Reject URI sources (``http://``, ``https://``, ``s3://``...) in offline mode.

    Raises:
        OfflineViolationError: ``source`` carries a URI scheme.
    """
    if _URI_RE.match(source):
        raise OfflineViolationError("remote model sources are not allowed in offline mode")
