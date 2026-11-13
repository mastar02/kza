"""Register the pre-2026-10-28 erasure target with 30da279's own code (U3b-0e2 review F2 fixture).

Run as ``python -I -B adoption_legacy_child_30da279.py <30da279 tree> <legacy checkout>``.
The working directory becomes the legacy checkout, as for the service that
ran from /home/kza/app -> /home/kza/kza before the 10-28 slot copy. Only the
30da279 tree is importable: its privacy services open data/memory_privacy.db
and its register_policy_index_target registers the semantic index with the
resolve()d checkout path, the descriptor shape production may still hold
(erasure targets are never deleted). The parent then copies data/ into the
b0866e8 slot state, as the 10-28 prepare_slot copy did.
"""
from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
import json
import os
from pathlib import Path
import sys

TREE, CHECKOUT = sys.argv[1], Path(sys.argv[2])
sys.path.insert(0, TREE)


async def build() -> dict:
    from src.memory.index_erasure import register_policy_index_target
    from src.memory.privacy_services import build_privacy_services

    config = {"memory": {"privacy_db_path": "./data/memory_privacy.db", "chroma_path": "./data/memory_db"}}
    async with AsyncExitStack() as shutdown:
        privacy = await build_privacy_services(config, shutdown=shutdown, clock=lambda: 1_700_000_000.0)
        descriptor = await register_policy_index_target(config, privacy)
    return {"descriptor": descriptor}


def main() -> int:
    os.chdir(CHECKOUT)
    Path("data").mkdir(exist_ok=True)
    sys.stdout.write(json.dumps(asyncio.run(build())) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
