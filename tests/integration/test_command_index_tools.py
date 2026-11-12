"""U3b-0d2 R3/R5: the shared command-index resolver and the offline tools that use it.

The runtime's staging syncs move the active command index off the base name
('home_assistant_commands') to a pointer-named collection. The offline sync
script (today's real sync path), the contamination cleaner and the voice test
must all read and write that same collection.
"""

import json
import sys
import types

import chromadb
from chromadb.config import Settings
import numpy as np
import pytest

from src.vectordb.command_index import (
    LEGACY_COMMANDS_COLLECTION, POINTER_MISSING_TARGET, open_active_commands, pointer_name, pointer_target,
)

ACTIVE = "home_assistant_commands__20261112000000abcdef"


def _client(path):
    return chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))


def _pointed_index(path):
    """A legacy collection (rollback copy) plus a pointer-named active one."""
    client = _client(path)
    legacy = client.get_or_create_collection(LEGACY_COMMANDS_COLLECTION, metadata={"hnsw:space": "cosine"})
    legacy.add(ids=["old"], embeddings=[[0.1, 0.2, 0.3]], documents=["frase pre-0d2"])
    active = client.create_collection(ACTIVE, metadata={"hnsw:space": "cosine"})
    active.add(ids=["cmd_0"], embeddings=[[0.1, 0.2, 0.3]], documents=["prende el living"],
               metadatas=[{"entity_id": "light.grupo_living", "area": "living"}])
    client.get_or_create_collection(pointer_name()).modify(metadata={"active": ACTIVE})
    return client


def test_without_a_pointer_the_base_collection_is_active(tmp_path):
    client = _client(tmp_path)
    assert pointer_target(client) is None
    collection, active = open_active_commands(client)
    assert collection.name == active.name == LEGACY_COMMANDS_COLLECTION and active.degraded is None
    assert pointer_name() not in [c.name for c in client.list_collections()]  # resolving creates no pointer


def test_read_only_resolution_never_creates_the_base(tmp_path):
    client = _client(tmp_path)
    with pytest.raises(Exception):
        open_active_commands(client, create_base=False)
    assert [c.name for c in client.list_collections()] == []


def test_the_pointer_names_the_active_collection(tmp_path):
    client = _pointed_index(tmp_path)
    collection, active = open_active_commands(client)
    assert collection.name == ACTIVE and active.pointer_target == ACTIVE and active.degraded is None


def test_a_missing_pointer_target_is_degraded_and_never_created(tmp_path, caplog):
    client = _pointed_index(tmp_path)
    client.delete_collection(ACTIVE)
    collection, active = open_active_commands(client)
    assert collection.name == LEGACY_COMMANDS_COLLECTION and active.degraded == POINTER_MISSING_TARGET
    assert ACTIVE not in [c.name for c in client.list_collections()]
    assert "the pointer names a missing collection" in caplog.text


def test_clean_contaminated_reads_the_active_collection(tmp_path, monkeypatch, capsys):
    from tools import clean_contaminated_chroma_docs as tool

    _pointed_index(tmp_path / "chroma")
    monkeypatch.setattr(sys, "argv", ["clean", "--chroma-path", str(tmp_path / "chroma")])
    assert tool.main() == 0
    assert f"Colección: {ACTIVE}" in capsys.readouterr().out


class _Embedder:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, phrase):
        return np.array([0.1, 0.2, 0.3])


class _LLM:
    def __init__(self, *args, **kwargs):
        pass

    def complete(self, prompt, max_tokens=300, temperature=0.5):
        return json.dumps(["prende el grupo del living", "encendé el living"])


@pytest.mark.parametrize("wipe", [False, True])
def test_the_offline_sync_writes_the_collection_the_runtime_reads(tmp_path, monkeypatch, wipe):
    """R5: scripts/sync_ha_to_chroma.py adds to the pointer-named collection."""
    import scripts.sync_ha_to_chroma as script

    (tmp_path / "data").mkdir()
    _pointed_index(tmp_path / "data/chroma_db")
    config = tmp_path / "settings.yaml"
    config.write_text("{}\n")
    monkeypatch.setattr(script, "ROOT", tmp_path)
    monkeypatch.setattr(script, "ha_get", lambda path: [{
        "entity_id": "light.grupo_living", "state": "on",
        "attributes": {"friendly_name": "Living", "supported_color_modes": ["onoff"]}}])
    monkeypatch.setattr(script, "resolve_areas", lambda ids: {i: "living" for i in ids})
    monkeypatch.setattr(script, "VLLMClient", _LLM)
    monkeypatch.setitem(sys.modules, "sentence_transformers",
                        types.SimpleNamespace(SentenceTransformer=_Embedder))
    monkeypatch.setattr(sys, "argv", ["sync", "--embedder-device", "cpu", "--config", str(config),
                                      *(["--wipe"] if wipe else [])])
    script.main()

    client = _client(tmp_path / "data/chroma_db")
    collection, active = open_active_commands(client)
    assert active.name == ACTIVE and active.degraded is None
    documents = collection.get()["documents"]
    assert "prende el grupo del living" in documents
    assert ("prende el living" in documents) is not wipe  # --wipe recreated the same active name
    # The legacy rollback copy is untouched either way.
    assert client.get_collection(LEGACY_COMMANDS_COLLECTION).get()["documents"] == ["frase pre-0d2"]
