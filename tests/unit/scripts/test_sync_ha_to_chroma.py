"""Tests de las funciones puras de scripts/sync_ha_to_chroma.py (sin acceso a HA)."""
import importlib


def test_module_imports_without_env(monkeypatch):
    # Sin HOME_ASSISTANT_* y sin .env, el import no debe crashear.
    monkeypatch.delenv("HOME_ASSISTANT_URL", raising=False)
    monkeypatch.delenv("HOME_ASSISTANT_TOKEN", raising=False)
    mod = importlib.import_module("scripts.sync_ha_to_chroma")
    assert hasattr(mod, "is_group_entity")
    assert hasattr(mod, "main")


from scripts.sync_ha_to_chroma import is_group_entity


def test_is_group_entity_prefix():
    assert is_group_entity("light.grupo_cocina", "Cocina") is True
    assert is_group_entity("light.grupo_escritorio", "Escritorio") is True
    # Modelo viejo Hue y Z2M one-off NO son grupos del modelo nuevo:
    assert is_group_entity("light.cocina", "Cocina") is False
    assert is_group_entity("light.escritorio_2", "Escritorio 4") is False
    assert is_group_entity("light.hogar", "Hogar") is False
    # Bombita individual:
    assert is_group_entity("light.l1", "L1") is False


from scripts.sync_ha_to_chroma import build_scene_specs

_ROOM_WORDS = {"cocina", "living", "escritorio", "baño", "bano", "hall",
               "cuarto", "balcón", "balcon", "escalera", "pasillo"}


def test_build_scene_specs_shape():
    specs = build_scene_specs()
    ids = {s.entity_id for s in specs}
    assert ids == {"scene.cine", "scene.lectura", "scene.calida",
                   "scene.fria", "scene.relax"}
    for s in specs:
        assert len(s.phrases) >= 3
        assert s.value_label and "." not in s.value_label


def test_scene_phrases_no_room_collision():
    # Las frases de escena NO deben mencionar un cuarto (evita pisar el
    # color_temp por-cuarto: 'poné la cocina fría' → grupo_cocina, no scene.fria).
    for s in build_scene_specs():
        for phrase in s.phrases:
            low = phrase.lower()
            assert not any(room in low.split() for room in _ROOM_WORDS), \
                f"frase de escena con cuarto: {phrase!r}"


import json as _json
from scripts.sync_ha_to_chroma import build_scene_documents


def test_build_scene_documents_metadata():
    docs = build_scene_documents(build_scene_specs())
    assert len(docs) >= 15  # 5 escenas × ≥3 frases
    by_eid = {}
    for doc_id, phrase, meta in docs:
        assert meta["domain"] == "scene"
        assert meta["service"] == "turn_on"
        assert meta["entity_id"].startswith("scene.")
        assert meta["capability"] == "scene"
        assert _json.loads(meta["service_data"]) == {}
        assert meta["is_group"] is False
        assert isinstance(phrase, str) and phrase
        assert doc_id[-1] in "0123456789"
        by_eid[meta["entity_id"]] = by_eid.get(meta["entity_id"], 0) + 1
    assert by_eid["scene.cine"] >= 3
    # cache_key estable entre llamadas (idempotencia incremental):
    docs2 = build_scene_documents(build_scene_specs())
    assert [d[0] for d in docs] == [d[0] for d in docs2]


def test_scene_indexing_block_persists():
    """build_scene_documents → collection.add con domain=scene/service=turn_on."""
    added = []

    class FakeColl:
        def add(self, ids, embeddings, documents, metadatas):
            added.append((ids[0], metadatas[0]["domain"], metadatas[0]["service"]))

    class FakeEmb:
        def encode(self, p):
            class _A:
                def tolist(self_): return [0.0, 0.1, 0.2]
            return _A()

    coll, emb = FakeColl(), FakeEmb()
    for doc_id, phrase, meta in build_scene_documents(build_scene_specs()):
        coll.add(ids=[doc_id], embeddings=[emb.encode(phrase).tolist()],
                 documents=[phrase], metadatas=[meta])
    assert added and all(d == "scene" and s == "turn_on" for _, d, s in added)
