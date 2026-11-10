"""Shared browser and evidence location helpers for the manual browser harnesses.

Linux keeps the historical ``chromium`` binary and its snap-confined profile
root. macOS uses Google Chrome.app, whose profile may live in the default
temporary directory. Evidence defaults to a fresh run-scoped directory so a new
run never writes into retained historical evidence.
"""

import os
from pathlib import Path
import shutil
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
MAC_CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'


def browser_binary(override_variable: str = 'KZA_BROWSER_BINARY',
                   missing: str = 'browser_binary_missing') -> str:
    """Select an owned headless browser without changing the Linux default.

    Args:
        override_variable: Environment variable holding an absolute browser path.
        missing: RuntimeError message raised when no usable browser is found.

    Returns:
        ``chromium`` on Linux without override, otherwise an absolute executable path.
    """
    override = os.environ.get(override_variable)
    try:
        if override is None and sys.platform != 'darwin':
            if shutil.which('chromium') is not None:
                return 'chromium'
        else:
            selected = Path(override if override is not None else MAC_CHROME)
            if selected.is_absolute() and selected.is_file() and os.access(selected, os.X_OK):
                return str(selected)
    except (OSError, ValueError):
        pass
    raise RuntimeError(missing)


def browser_profile_root(mode: int = 0o777) -> Path | None:
    """Return the temporary-profile parent the selected browser can access.

    Snap Chromium can only read profiles under ``~/snap/chromium/common``; on
    macOS ``None`` keeps ``tempfile``'s default location.
    """
    if sys.platform == 'darwin':
        return None
    root = Path.home() / 'snap/chromium/common'
    root.mkdir(mode=mode, parents=True, exist_ok=True)
    return root


def evidence_directory(variable: str, default_parent: str) -> Path:
    """Resolve the evidence directory for one browser run.

    Args:
        variable: Environment variable that names an explicit evidence directory.
        default_parent: Repository-relative parent for the default run directory.

    Returns:
        The explicit directory, or a new ``run-<ns>`` directory under ``default_parent``.
    """
    explicit = os.environ.get(variable)
    if explicit:
        return Path(explicit)
    return ROOT / default_parent / f'run-{time.time_ns()}'
