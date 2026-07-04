"""Tests de CodeIndexer.search."""

from tests.unit.code_index.test_indexer import FakeCardGenerator, make_indexer, repo  # noqa: F401


async def test_search_returns_ranked_results_with_cards(repo):  # noqa: F811
    idx, chunks, cards, manifest = make_indexer(repo)
    await idx.reindex()

    results = await idx.search("funcion a", top_k=2)

    assert len(results) == 2
    r = results[0]
    assert set(r) == {
        "path", "name", "kind", "start_line", "end_line",
        "blob_hash", "score", "snippet", "card",
    }
    assert r["path"].startswith("src/")
    assert r["card"].startswith("## Propósito")
    assert isinstance(r["score"], float)


async def test_search_card_none_when_pending(repo):  # noqa: F811
    card_gen = FakeCardGenerator(fail_paths={"src/a.py", "src/b.py"})
    idx, chunks, cards, manifest = make_indexer(repo, card_gen)
    await idx.reindex()

    results = await idx.search("cualquier cosa", top_k=2)
    assert all(r["card"] is None for r in results)


async def test_search_empty_index_returns_empty(repo):  # noqa: F811
    idx, *_ = make_indexer(repo)
    assert await idx.search("algo") == []


async def test_search_truncates_snippet(repo):  # noqa: F811
    (repo / "src" / "big.py").write_text(
        "def big():\n" + "    x = 1\n" * 500
    )
    idx, *_ = make_indexer(repo)
    await idx.reindex()

    results = await idx.search("big", top_k=5)
    assert all(len(r["snippet"]) <= 1500 for r in results)
