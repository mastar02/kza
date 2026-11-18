"""SC4C-Q3: the Chroma files kza-voice writes hold no earlier heap contents.

Mechanism (Chroma 1.5.5, chroma-core/hnswlib; measured on the server with
glibc 2.39): a persistent HNSW index is initialized with ``malloc``'d buffers
(``data_level0`` = capacity x element size, ``length`` = capacity floats, the
capacity 100 at first) and ``initPersistentIndex`` writes both whole, before
any element exists. A later save writes only the slots of elements, so the
unused capacity keeps whatever that memory held; a deletion only marks a slot.
The local writer initializes the index again, in the process that opens the
collection (a ``count()`` is enough), whenever the segment has no persisted
non-empty id map: fewer than ``sync_threshold`` (1000) operations since its
last persist, or no live records. Production's stores are in that state
(SC4C-Q2: 337 and 130 records), so every open by kza-voice rewrote
``data_level0.bin`` and ``length.bin`` from its own heap: texts the process
had in memory, a deleted fact's included, and heap addresses.

Fix: the managed child runs with glibc's ``perturb`` tunable and without the
per-thread cache (``tcache_count=0``; a cached chunk is returned without the
perturb fill). Then ``malloc`` never returns earlier contents: the requested
bytes are the perturb pattern (or zero, a fresh mapping); the new part of a
``realloc`` grown in place can hold the free fill and free-chunk link words
(heap addresses), never process data. So an initialized index writes no
residue. The vector of a deleted record stays in ``data_level0.bin`` until
compaction (a deletion only marks its slot); this does not change that. The tunables are read by the dynamic loader at exec,
so the parent sets them: the managed launcher (``child_environment``, the same
literal as :data:`HEAP_HYGIENE_TUNABLES`, a test pins both) and the
owner-compaction clean child. The managed entry refuses (exit 78) before it
arms when :func:`heap_hygiene_active` is False (a launcher that lost the
variable, or an interposed allocator that ignores it). Outside glibc (the Mac
development machine) the property is not available and nothing is refused.

Stdlib only; imported by ``src.main`` before the managed entry arms.
"""
from __future__ import annotations

from collections.abc import Mapping
import ctypes
import logging
import os

logger = logging.getLogger(__name__)

GLIBC_TUNABLES_ENV = "GLIBC_TUNABLES"
# perturb=165 (0xA5): malloc fills with 0x5A, free with 0xA5. The launcher holds the same literal.
HEAP_HYGIENE_TUNABLES = (("glibc.malloc.perturb", "165"), ("glibc.malloc.tcache_count", "0"))
_ALLOCATED_BYTE = 165 ^ 0xFF
_PROBE_SIZES = (24, 400, 1000, 4096, 65536)  # fastbin, the cached sizes (<= 1032) and larger, under mmap
TICKET_ENV = "KZA_RUNTIME_ADMISSION"  # same literal as src.core.entry_guard.TICKET_ENV (a test pins it)
REFUSAL_CODE = "runtime_admission_heap_hygiene_inactive"


def with_heap_hygiene(value: str | None) -> str:
    """A ``GLIBC_TUNABLES`` value with the hygiene tunables last.

    Args:
        value: The inherited value, or None.

    Returns:
        The other tunables in their order, then the hygiene ones; an inherited
        entry with the same name as a hygiene tunable is dropped.
    """
    names = {name for name, _ in HEAP_HYGIENE_TUNABLES}
    kept = [item for item in (value or "").split(":") if item and item.split("=", 1)[0] not in names]
    return ":".join([*kept, *(f"{name}={setting}" for name, setting in HEAP_HYGIENE_TUNABLES)])


def _glibc() -> bool:
    try:
        return bool(os.confstr("CS_GNU_LIBC_VERSION"))
    except (ValueError, OSError):
        return False


def heap_hygiene_active() -> bool | None:
    """Whether this process's ``malloc`` returns only the perturb pattern.

    Each probe size is allocated, checked and freed twice: the second
    allocation reuses the freed chunk, which the per-thread cache would hand
    back without the fill (its link and key words differ from the pattern).

    Returns:
        None outside glibc; otherwise True only when every probed byte is the
        allocation pattern of :data:`HEAP_HYGIENE_TUNABLES`.
    """
    if not _glibc():
        return None
    libc = ctypes.CDLL(None)
    malloc, free = libc.malloc, libc.free
    malloc.restype, malloc.argtypes = ctypes.c_void_p, [ctypes.c_size_t]
    free.restype, free.argtypes = None, [ctypes.c_void_p]
    for size in _PROBE_SIZES:
        for _ in range(2):
            pointer = malloc(size)
            if not pointer:
                return False
            try:
                filled = ctypes.string_at(pointer, size) == bytes([_ALLOCATED_BYTE]) * size
            finally:
                free(pointer)
            if not filled:
                return False
    return True


class HeapHygieneRefused(SystemExit):
    """Exit 78 (the managed unit's RestartPreventExitStatus) with a content-free code."""

    def __init__(self) -> None:
        super().__init__(78)
        self.refusal = REFUSAL_CODE

    def __str__(self) -> str:
        return self.refusal


def refuse_unprotected_managed_entry(environ: Mapping[str, str] | None = None) -> None:
    """Refuse the managed child (it carries the ticket) when glibc's malloc can return old contents.

    Args:
        environ: The process environment (default ``os.environ``).

    Raises:
        HeapHygieneRefused: Under a ticket, on glibc, without the tunables in effect.
    """
    environ = os.environ if environ is None else environ
    if TICKET_ENV not in environ:
        return
    if heap_hygiene_active() is False:
        logger.critical("[SC4C-Q3] refused: %s", REFUSAL_CODE)
        raise HeapHygieneRefused()
