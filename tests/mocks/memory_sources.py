"""Synthetic provenance for a real, bounded test capture."""

import time
from uuid import uuid4

from src.memory.fact_ledger import MemoryOrigin, MemorySource


def memory_source(*, now=None, ttl=48*3600, conversation_id=None):
    now = time.time() if now is None else now
    return MemorySource(str(uuid4()), conversation_id or str(uuid4()), now, now+ttl, MemoryOrigin.VOICE)
