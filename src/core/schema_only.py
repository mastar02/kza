"""The adoption's schema-only entry point (U3b-0e S1, G1 capability).

Run only as :func:`src.core.entry_guard.schema_only_command` with
``KZA_ADOPTION_SCHEMA_ONLY=1`` and no admission ticket: '-I -B', no SDK, no
network, no secrets. It runs the registered schema initializers against an
explicit adoption copy (``--state``) and never loads the serving
configuration, so it opens no serving store.

``SCHEMA_INITIALIZERS`` is filled by the adoption library (U3b-0e2). Each
initializer is a callable taking the state directory (a ``Path``) that runs
only open-time DDL on that copy.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable
import json
import logging
import os
from pathlib import Path
import sys

from src.core.entry_guard import (
    CAPABILITY_CODE, SCHEMA_ONLY_ENV, SCHEMA_ONLY_VALUE, TICKET_ENV, schema_only_invocation,
)

logger = logging.getLogger(__name__)

SCHEMA_INITIALIZERS: tuple[Callable[[Path], None], ...] = ()


def main(argv: list[str] | None = None) -> int:
    """Run the schema initializers on ``--state``; print a content-free summary.

    Returns:
        0 on success; 78 when the capability or invocation is not exact.
    """
    if (os.environ.get(SCHEMA_ONLY_ENV) != SCHEMA_ONLY_VALUE or TICKET_ENV in os.environ
            or not schema_only_invocation(list(getattr(sys, "orig_argv", sys.argv)))):
        logger.critical("[SchemaOnly] refused: %s", CAPABILITY_CODE)
        return 78
    parser = argparse.ArgumentParser(prog="schema_only")
    parser.add_argument("--state", required=True)
    options = parser.parse_args(argv)
    state = Path(options.state)
    if state.is_symlink() or not state.is_dir():
        logger.critical("[SchemaOnly] refused: schema_only_state_invalid")
        return 78
    for initializer in SCHEMA_INITIALIZERS:
        initializer(state)
    sys.stdout.write(json.dumps({"schema_only": True, "initializers": len(SCHEMA_INITIALIZERS)}) + "\n")
    return 0
