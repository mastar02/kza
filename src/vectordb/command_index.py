"""Active command-index resolution shared by the runtime and the offline tools.

U3b-0d2: the runtime rebuilds the command index into a staging collection and
makes it active with one write to a pointer record (the metadata of the
``<base>__pointer`` collection, key ``active``). Everything that reads or
writes the command index (ChromaSync, scripts/sync_ha_to_chroma.py,
tools/clean_contaminated_chroma_docs.py, scripts/test_voice_command.py) must
resolve the collection through this module, never by the fixed base name.

Rules:
    * No pointer, or a pointer naming the base: the base (legacy) collection
      is active. This is every index written before 0d2.
    * A pointer naming another collection: that collection is opened with
      ``get_collection``. It is never created implicitly. If it is missing,
      the error is logged, the base collection is used and the result is
      marked degraded so health reports it.
    * The base collection is never dropped by a sync while rollback to
      pre-0d2 code is possible (pre-0d2 code opens only the base name).

The chromadb client is passed in; this module imports no SDK.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

LEGACY_COMMANDS_COLLECTION = "home_assistant_commands"
COSINE = {"hnsw:space": "cosine"}
POINTER_MISSING_TARGET = "pointer_target_missing"


def pointer_name(base: str = LEGACY_COMMANDS_COLLECTION) -> str:
    """Name of the collection whose metadata records the active index."""
    return f"{base}__pointer"


def _missing(error: Exception) -> bool:
    try:
        from chromadb.errors import NotFoundError
    except ImportError:  # pragma: no cover - chromadb is a runtime dependency
        NotFoundError = ()  # noqa: N806
    return isinstance(error, NotFoundError) or "does not exist" in str(error)


def pointer_target(client, base: str = LEGACY_COMMANDS_COLLECTION) -> str | None:
    """Collection name the persisted pointer records, without creating anything.

    Args:
        client: chromadb client.
        base: Base (legacy) command collection name.

    Returns:
        The recorded name, or None when there is no pointer or it names nothing.
    """
    try:
        pointer = client.get_collection(name=pointer_name(base))
    except Exception as error:
        if _missing(error):
            return None
        raise
    return (pointer.metadata or {}).get("active") or None


@dataclass(frozen=True)
class ActiveCommandIndex:
    """The command collection that is active, and how it was resolved.

    Attributes:
        name: Name of the opened collection.
        pointer_target: Name the persisted pointer records (None: no pointer).
        degraded: Reason when the pointer target could not be opened.
    """

    name: str
    pointer_target: str | None = None
    degraded: str | None = None


def open_active_commands(client, base: str = LEGACY_COMMANDS_COLLECTION, *,
                         create_base: bool = True):
    """Open the active command collection.

    Args:
        client: chromadb client.
        base: Base (legacy) command collection name.
        create_base: Create the base collection when it is the active one and
            does not exist yet (a fresh install). Read-only tools pass False.

    Returns:
        ``(collection, ActiveCommandIndex)``.

    Raises:
        Exception: The SDK's not-found error when ``create_base`` is False and
            the base collection is missing.
    """
    target = pointer_target(client, base)
    degraded = None
    if target is not None and target != base:
        try:
            return client.get_collection(name=target), ActiveCommandIndex(target, target)
        except Exception as error:
            if not _missing(error):
                raise
            degraded = POINTER_MISSING_TARGET
            logger.error("[CommandIndex] the pointer names a missing collection; "
                         "falling back to the base collection (index degraded)")
    if create_base:
        collection = client.get_or_create_collection(name=base, metadata=COSINE)
    else:
        collection = client.get_collection(name=base)
    return collection, ActiveCommandIndex(base, target, degraded)
