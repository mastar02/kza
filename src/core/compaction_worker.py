"""The owner-compaction worker entry point (U3b-0c, G1 capability).

Run only as :func:`src.core.entry_guard.compaction_worker_command` with
``KZA_COMPACTION_WORKER=1`` and no admission ticket: '-I -B', from the
controller's own runtime tree. It performs the phases of
scripts/compaction.py that need application code (the closed-state snapshot
of H and the identity clone into K, owner-compaction design v2 D1-D5, v4 F3)
under its own admission lease, with gate ``compaction:<id>``, and prints one
content-free JSON line. Refusals exit 78 with the code on stderr.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time

from src.core.entry_guard import (
    CAPABILITY_CODE, COMPACTION_WORKER_ENV, COMPACTION_WORKER_VALUE, TICKET_ENV, compaction_worker_invocation,
)

logger = logging.getLogger(__name__)

REFUSED_EXIT = 78
PHASES = ("materialize",)


def code_location_refusal(root: Path) -> str | None:
    """P7 (design v2/v3): the worker's code never comes from a slot under the root."""
    own = Path(__file__).resolve()
    slots = Path(root).resolve() / "slots"
    if own.is_relative_to(slots):
        return "runtime_admission_compaction_controller_location"
    return None


def main(argv: list[str]) -> int:
    """Parse, check the capability and the code location, run one phase, print its JSON."""
    if (os.environ.get(COMPACTION_WORKER_ENV) != COMPACTION_WORKER_VALUE or TICKET_ENV in os.environ
            or not compaction_worker_invocation(list(getattr(sys, "orig_argv", sys.argv)))):
        print(CAPABILITY_CODE, file=sys.stderr)
        return REFUSED_EXIT
    parser = argparse.ArgumentParser(prog="compaction-worker")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--compaction", required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    args = parser.parse_args(argv)
    refusal = code_location_refusal(args.root)
    if refusal is not None:
        print(refusal, file=sys.stderr)
        return REFUSED_EXIT
    from src.memory.owner_compaction import run_materialization
    try:
        result = run_materialization(args.root, args.compaction, wall=time.time)
    except ValueError as error:
        logger.error("[Compaction] worker refused: %s", str(error)[:160])
        print(str(error)[:300], file=sys.stderr)
        return REFUSED_EXIT
    print(json.dumps(result, sort_keys=True))
    return 0
