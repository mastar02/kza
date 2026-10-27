"""A real Chroma SQLite source with explicitly selected synthetic legacy IDs."""

from dataclasses import dataclass
import json
from pathlib import Path

from tests.mocks.memory_graph import MemoryClient


@dataclass
class LegacyMemory:
    client: MemoryClient
    source: Path
    preview: Path
    output: Path
    selected: tuple[str, ...]


def legacy_memory(root: Path) -> LegacyMemory:
    """Seed actual document, typed metadata and FLOAT32 queue storage."""
    client = MemoryClient(root/'source')
    collection = client.get_or_create_collection('legacy_memories').collection
    identifiers = ('synthetic-old-a', 'synthetic-old-b', 'synthetic-new-c')
    collection.upsert(ids=list(identifiers), embeddings=[[1., 2., float(i)] for i in range(3)],
        documents=['private legacy marker '+identifier for identifier in identifiers],
        metadatas=[{'origin':'ambient','created_at':1.,'access_count':i} for i in range(3)])
    source = root/'source/chroma.sqlite3'
    preview = root/'preview.json'
    selected = identifiers[:2]
    preview.write_text(json.dumps({'source':{'path':str(source),'collection':'legacy_memories'},
        'record_metadata':[{'id':identifier} for identifier in selected]}))
    return LegacyMemory(client, source, preview, root/'copy', selected)
