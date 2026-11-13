"""The adoption's schema-only entry point (U3b-0e S1, G1 capability).

Run only as :func:`src.core.entry_guard.schema_only_command` with
``KZA_ADOPTION_SCHEMA_ONLY=1`` and no admission ticket: '-I -B', no SDK, no
network, no secrets. It runs the registered schema initializers against an
explicit adoption copy (``--state``) and never loads the serving
configuration. :func:`state_decision` refuses any slot state under a managed
root and any bound state, and requires the adoption-copy marker or the P4
tmpfs location, so it opens no serving store (independent review F1).

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
import stat
import sys

from src.core.entry_guard import (
    CAPABILITY_CODE, SCHEMA_ONLY_ENV, SCHEMA_ONLY_VALUE, TICKET_ENV, managed_root_of, schema_only_invocation,
)

logger = logging.getLogger(__name__)

SCHEMA_INITIALIZERS: tuple[Callable[[Path], None], ...] = ()
# The adoption copy carries this marker (written by the adoption library,
# U3b-0e2), naming the plan it was made for.
COPY_MARKER = ".kza-adoption-copy.json"
BINDING_MARKER = ".kza-state-binding.json"  # runtime_admission.MARKER: a bound serving state


def default_tmpfs_roots() -> tuple[str, ...]:
    """The P4 scratch location: the user's tmpfs runtime directory."""
    return (f"/run/user/{os.getuid()}",)


def _marker_ok(state: Path) -> bool:
    path = state / COPY_MARKER
    try:
        if not stat.S_ISREG(path.lstat().st_mode):
            return False
        value = json.loads(path.read_text())
    except (OSError, ValueError, UnicodeError):
        return False
    if type(value) is not dict:
        return False
    digest = value.get("plan_sha256")
    return (type(value.get("schema_version")) is int and value["schema_version"] == 1
            and type(digest) is str and len(digest) == 64 and set(digest) <= set("0123456789abcdef"))


def state_decision(state: str | os.PathLike, *, tmpfs_roots: tuple[str, ...] | None = None) -> str | None:
    """None if ``state`` is an adoption copy the schema-only child may open (review F1).

    Refuses a symlink or non-directory, anything under a managed root's
    slots/*/state (lexically or resolved, so 'current' is covered), and any
    state bound by the admission marker. The copy must then carry
    ``COPY_MARKER`` or lie in the P4 tmpfs scratch location.
    """
    lexical = Path(os.path.abspath(os.fspath(state)))
    if lexical.is_symlink() or not lexical.is_dir():
        return "schema_only_state_invalid"
    resolved = lexical.resolve()
    for candidate in (lexical, resolved):
        root = managed_root_of([candidate])
        if root is not None:
            for base in {root, root.resolve()}:
                if candidate.is_relative_to(base):
                    parts = candidate.relative_to(base).parts
                    if len(parts) >= 3 and parts[0] == "slots" and parts[2] == "state":
                        return "schema_only_state_serving"
        for parent in (candidate, *candidate.parents):
            if os.path.lexists(parent / BINDING_MARKER):
                return "schema_only_state_bound"
    if _marker_ok(resolved):
        return None
    roots = default_tmpfs_roots() if tmpfs_roots is None else tmpfs_roots
    if any(resolved.is_relative_to(Path(item).resolve()) and resolved != Path(item).resolve() for item in roots):
        return None
    return "schema_only_state_unmarked"


def _refuse(code: str) -> int:
    logger.critical("[SchemaOnly] refused: %s", code)
    return 78


def main(argv: list[str] | None = None) -> int:
    """Run the schema initializers on ``--state``; print a content-free summary.

    Returns:
        0 on success; 78 when the capability, invocation, runtime or state is not exact.
    """
    process_argv = list(getattr(sys, "orig_argv", sys.argv))
    if (os.environ.get(SCHEMA_ONLY_ENV) != SCHEMA_ONLY_VALUE or TICKET_ENV in os.environ
            or not schema_only_invocation(process_argv)):
        return _refuse(CAPABILITY_CODE)
    # The bootstrap's runtime argument must be the runtime holding this code.
    own_runtime = Path(__file__).resolve().parents[2]
    if len(sys.argv) < 2 or Path(sys.argv[1]).resolve() != own_runtime:
        return _refuse("schema_only_runtime_mismatch")
    parser = argparse.ArgumentParser(prog="schema_only")
    parser.add_argument("--state", required=True)
    options = parser.parse_args(argv)
    code = state_decision(options.state)
    if code is not None:
        return _refuse(code)
    state = Path(options.state).resolve()
    for initializer in SCHEMA_INITIALIZERS:
        initializer(state)
    sys.stdout.write(json.dumps({"schema_only": True, "initializers": len(SCHEMA_INITIALIZERS)}) + "\n")
    return 0
