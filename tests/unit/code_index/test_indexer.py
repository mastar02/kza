"""Tests del CodeIndexer con Chroma/embedder/cards fake."""

from pathlib import Path

import pytest

from src.code_index.indexer import CodeIndexer
from src.code_index.manifest import IndexManifest
from tests.mocks.mock_chroma import MockChromaCollection


class FakeEmbedder:
    """encode() determinístico: vector según largo del texto."""

    def encode(self, texts):
        import numpy as np

        return np.array([[float(len(t) % 7 + 1), 1.0, 0.0] for t in texts])


class FakeCardGenerator:
    def __init__(self, fail_paths=None):
        self.fail_paths = set(fail_paths or [])
        self.calls: list[str] = []

    async def generate(self, path: str, source: str) -> str:
        self.calls.append(path)
        if path in self.fail_paths:
            raise RuntimeError("gateway caído")
        return f"## Propósito\nCard de {path}"


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("def fa():\n    return 1\n")
    (tmp_path / "src" / "b.py").write_text("def fb():\n    return 2\n")
    return tmp_path


def make_indexer(repo, card_gen=None):
    chunks = MockChromaCollection("code_chunks")
    cards = MockChromaCollection("code_cards")
    manifest = IndexManifest(repo / "manifest.json")
    manifest.load()
    idx = CodeIndexer(
        repo_root=repo,
        chunks_collection=chunks,
        cards_collection=cards,
        embedder=FakeEmbedder(),
        card_generator=card_gen or FakeCardGenerator(),
        manifest=manifest,
    )
    return idx, chunks, cards, manifest


async def test_initial_reindex_indexes_everything(repo):
    idx, chunks, cards, manifest = make_indexer(repo)
    stats = await idx.reindex()
    assert stats == {"indexed": 2, "deleted": 0, "cards_failed": 0, "errors": 0}
    assert chunks.count() == 2  # fa y fb
    assert cards.count() == 2
    assert set(manifest.files) == {"src/a.py", "src/b.py"}
    assert all(e["card_done"] for e in manifest.files.values())


async def test_incremental_skips_unchanged(repo):
    card_gen = FakeCardGenerator()
    idx, chunks, cards, manifest = make_indexer(repo, card_gen)
    await idx.reindex()
    card_gen.calls.clear()

    (repo / "src" / "a.py").write_text("def fa():\n    return 99\n")
    stats = await idx.reindex()

    assert stats["indexed"] == 1
    assert card_gen.calls == ["src/a.py"]
    assert "return 99" in chunks.get(ids=["src/a.py::fa"])["documents"][0]


async def test_deleted_file_is_purged(repo):
    idx, chunks, cards, manifest = make_indexer(repo)
    await idx.reindex()

    (repo / "src" / "b.py").unlink()
    stats = await idx.reindex()

    assert stats["deleted"] == 1
    assert "src/b.py" not in manifest.files
    assert chunks.get(ids=["src/b.py::fb"])["ids"] == []
    assert cards.get(ids=["src/b.py"])["ids"] == []


async def test_card_failure_marks_pending_and_retries(repo):
    card_gen = FakeCardGenerator(fail_paths={"src/a.py"})
    idx, chunks, cards, manifest = make_indexer(repo, card_gen)

    stats = await idx.reindex()
    assert stats["cards_failed"] == 1
    assert manifest.files["src/a.py"]["card_done"] is False
    assert chunks.count() == 2  # los chunks se indexan igual

    # gateway "vuelve": el próximo reindex reintenta solo la card pendiente
    card_gen.fail_paths.clear()
    card_gen.calls.clear()
    stats = await idx.reindex()
    assert "src/a.py" in card_gen.calls
    assert manifest.files["src/a.py"]["card_done"] is True


async def test_full_mode_reindexes_all(repo):
    card_gen = FakeCardGenerator()
    idx, chunks, cards, manifest = make_indexer(repo, card_gen)
    await idx.reindex()
    card_gen.calls.clear()

    stats = await idx.reindex(mode="full")
    assert stats["indexed"] == 2
    assert sorted(card_gen.calls) == ["src/a.py", "src/b.py"]


async def test_full_mode_purges_deleted_files(repo):
    idx, chunks, cards, manifest = make_indexer(repo)
    await idx.reindex()

    (repo / "src" / "b.py").unlink()
    stats = await idx.reindex(mode="full")

    assert stats["deleted"] == 1
    assert "src/b.py" not in manifest.files
    assert chunks.get(ids=["src/b.py::fb"])["ids"] == []
    assert cards.get(ids=["src/b.py"])["ids"] == []


async def test_property_setter_file_indexes_without_duplicate_ids(repo):
    (repo / "src" / "conn.py").write_text(
        "class Conn:\n"
        "    @property\n"
        "    def state(self):\n"
        "        return self._s\n"
        "\n"
        "    @state.setter\n"
        "    def state(self, v):\n"
        "        self._s = v\n"
    )
    idx, chunks, cards, manifest = make_indexer(repo)
    stats = await idx.reindex()
    assert stats["errors"] == 0
    assert "src/conn.py" in manifest.files


async def test_empty_file_skips_card_generation(repo):
    (repo / "src" / "__init__.py").write_text("")
    card_gen = FakeCardGenerator()
    idx, chunks, cards, manifest = make_indexer(repo, card_gen)
    await idx.reindex()
    assert "src/__init__.py" not in card_gen.calls
    assert manifest.files["src/__init__.py"]["card_done"] is True


async def test_unreadable_file_does_not_wedge_reindex(repo):
    # archivo no-UTF8: debe contarse como error y NO impedir indexar el resto
    (repo / "src" / "bad.py").write_bytes(b"\xff\xfe invalid \xff")
    idx, chunks, cards, manifest = make_indexer(repo)

    stats = await idx.reindex()

    assert stats["errors"] == 1
    assert stats["indexed"] == 2          # a.py y b.py se indexaron igual
    assert "src/bad.py" not in manifest.files
    # el archivo malo no bloquea reindexes futuros de los demás
    (repo / "src" / "a.py").write_text("def fa():\n    return 3\n")
    stats2 = await idx.reindex()
    assert stats2["indexed"] == 1
    assert stats2["errors"] == 1
