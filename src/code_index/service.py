"""Servicio HTTP del code-index (aiohttp): /health, /search, /reindex."""

import asyncio
import logging
from pathlib import Path

from aiohttp import web

from src.code_index.cards import CardGenerator
from src.code_index.indexer import CodeIndexer
from src.code_index.manifest import IndexManifest

logger = logging.getLogger(__name__)


async def handle_health(request: web.Request) -> web.Response:
    idx: CodeIndexer = request.app["indexer"]
    return web.json_response(
        {
            "status": "ok",
            "indexed_sha": idx.manifest.head_sha,
            "files": len(idx.manifest.files),
            "reindex_running": idx.running,
            "last_stats": idx.last_stats,
        }
    )


async def handle_search(request: web.Request) -> web.Response:
    body = await request.json()
    query = (body.get("query") or "").strip()
    if not query:
        return web.json_response({"error": "query vacía"}, status=400)
    top_k = int(body.get("top_k", 8))
    idx: CodeIndexer = request.app["indexer"]
    results = await idx.search(query, top_k=top_k)
    return web.json_response({"results": results})


async def handle_reindex(request: web.Request) -> web.Response:
    idx: CodeIndexer = request.app["indexer"]
    if idx.running:
        return web.json_response({"status": "already_running"}, status=409)
    mode = "incremental"
    if request.can_read_body:
        try:
            body = await request.json()
            mode = body.get("mode", "incremental")
        except ValueError:
            pass
    # referencia guardada en app para que el task no sea recolectado
    request.app["reindex_task"] = asyncio.get_running_loop().create_task(
        idx.reindex(mode=mode)
    )
    return web.json_response({"status": "started", "mode": mode}, status=202)


def create_app(indexer: CodeIndexer) -> web.Application:
    """Armar la app aiohttp con el indexer inyectado."""
    app = web.Application()
    app["indexer"] = indexer
    app.add_routes(
        [
            web.get("/health", handle_health),
            web.post("/search", handle_search),
            web.post("/reindex", handle_reindex),
        ]
    )
    return app


def build_indexer(cfg: dict, repo_root: Path) -> CodeIndexer:
    """Factory con dependencias reales (Chroma persistente, BGE-M3 CPU, MiniMax)."""
    import chromadb
    from sentence_transformers import SentenceTransformer

    client = chromadb.PersistentClient(path=cfg["chroma_path"])
    chunks = client.get_or_create_collection(
        "code_chunks", metadata={"hnsw:space": "cosine"}
    )
    cards = client.get_or_create_collection(
        "code_cards", metadata={"hnsw:space": "cosine"}
    )

    emb_cfg = cfg.get("embedder", {})
    device = emb_cfg.get("device", "cpu")
    logger.info(f"[CodeIndex] Cargando embedder {emb_cfg.get('model')} en {device}")
    embedder = SentenceTransformer(emb_cfg.get("model", "BAAI/bge-m3"), device=device)

    cards_cfg = cfg["cards"]
    card_gen = CardGenerator(
        base_url=cards_cfg["base_url"],
        model=cards_cfg["model"],
        api_key_env=cards_cfg.get("api_key_env", "MINIMAX_API_KEY"),
        timeout=cards_cfg.get("timeout", 120.0),
    )

    manifest = IndexManifest(Path(cfg["manifest_path"]))
    manifest.load()

    return CodeIndexer(
        repo_root=repo_root,
        chunks_collection=chunks,
        cards_collection=cards,
        embedder=embedder,
        card_generator=card_gen,
        manifest=manifest,
        include_globs=cfg.get("include_globs"),
    )
