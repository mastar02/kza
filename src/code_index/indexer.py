"""Orquestador del índice: scan → diff → chunk → embed → Chroma (+cards)."""

import asyncio
import logging
from pathlib import Path

from src.code_index.chunker import extract_chunks
from src.code_index.manifest import IndexManifest, ManifestDiff, git_blob_hash

logger = logging.getLogger(__name__)

DEFAULT_GLOBS = ["src/**/*.py"]


class CodeIndexer:
    """Indexa el codebase en dos colecciones Chroma: code_chunks y code_cards."""

    def __init__(
        self,
        repo_root: Path,
        chunks_collection,
        cards_collection,
        embedder,
        card_generator,
        manifest: IndexManifest,
        include_globs: list[str] | None = None,
    ):
        self.repo_root = Path(repo_root)
        self.chunks = chunks_collection
        self.cards = cards_collection
        self.embedder = embedder
        self.card_generator = card_generator
        self.manifest = manifest
        self.include_globs = include_globs or DEFAULT_GLOBS
        self._lock = asyncio.Lock()
        self.last_stats: dict | None = None

    @property
    def running(self) -> bool:
        """True si hay un reindex en curso."""
        return self._lock.locked()

    async def reindex(self, mode: str = "incremental") -> dict:
        """Reindexar el repo.

        mode="full" reprocesa todos los archivos actuales y purga los
        borrados (detectados contra el manifest previo). mode="incremental"
        (default) solo procesa lo agregado/cambiado desde el último reindex.

        Idempotente ante interrupciones: el manifest se persiste por archivo.
        """
        async with self._lock:
            current = await asyncio.to_thread(self._scan)
            diff = self.manifest.diff(current)
            if mode == "full":
                diff = ManifestDiff(added=sorted(current), deleted=diff.deleted)
            stats = {"indexed": 0, "deleted": 0, "cards_failed": 0, "errors": 0}
            logger.info(
                f"[CodeIndexer] reindex mode={mode}: +{len(diff.added)} "
                f"~{len(diff.changed)} -{len(diff.deleted)}"
            )

            for path in diff.deleted:
                await asyncio.to_thread(self._purge_path, path)
                await asyncio.to_thread(self.manifest.remove_file, path)
                stats["deleted"] += 1

            for path in diff.added + diff.changed:
                try:
                    await self._index_file(path, current[path], stats)
                except Exception as e:
                    logger.error(f"[CodeIndexer] Error indexando {path}: {e}")
                    stats["errors"] += 1

            self.manifest.head_sha = await self._git_head()
            await asyncio.to_thread(self.manifest.save)
            self.last_stats = stats
            logger.info(f"[CodeIndexer] reindex done: {stats}")
            return stats

    def _scan(self) -> dict[str, str]:
        """Hashear el árbol actual: {path relativo: git blob hash}."""
        current: dict[str, str] = {}
        for pattern in self.include_globs:
            for f in sorted(self.repo_root.glob(pattern)):
                rel = f.relative_to(self.repo_root).as_posix()
                try:
                    current[rel] = git_blob_hash(f.read_bytes())
                except OSError as e:
                    logger.warning(f"[CodeIndexer] No se pudo leer {rel}: {e}")
        return current

    def _purge_path(self, path: str) -> None:
        self.chunks.delete(where={"path": path})
        self.cards.delete(ids=[path])

    async def _index_file(self, path: str, blob_hash: str, stats: dict) -> None:
        source = await asyncio.to_thread(
            (self.repo_root / path).read_text, "utf-8"
        )
        chunks = extract_chunks(source, path)
        await asyncio.to_thread(self._purge_path, path)

        if chunks:
            embeddings = await asyncio.to_thread(
                self.embedder.encode, [c.text for c in chunks]
            )
            await asyncio.to_thread(
                self.chunks.add,
                ids=[c.chunk_id for c in chunks],
                embeddings=[list(map(float, e)) for e in embeddings],
                documents=[c.text for c in chunks],
                metadatas=[
                    {
                        "path": c.path,
                        "name": c.name,
                        "kind": c.kind,
                        "start_line": c.start_line,
                        "end_line": c.end_line,
                        "blob_hash": blob_hash,
                    }
                    for c in chunks
                ],
            )

        if not source.strip():
            # Archivo vacío (ej. __init__.py): no hay nada que resumir, no
            # gastar una llamada al gateway MiniMax en una card sin contenido.
            card_done = True
        else:
            card_done = False
            try:
                card = await self.card_generator.generate(path, source)
                card_emb = await asyncio.to_thread(self.embedder.encode, [card])
                await asyncio.to_thread(
                    self.cards.add,
                    ids=[path],
                    embeddings=[list(map(float, card_emb[0]))],
                    documents=[card],
                    metadatas=[{"path": path, "blob_hash": blob_hash}],
                )
                card_done = True
            except Exception as e:
                logger.warning(f"[CodeIndexer] Card falló para {path}: {e}")
                stats["cards_failed"] += 1

        await asyncio.to_thread(self.manifest.update_file, path, blob_hash, card_done)
        stats["indexed"] += 1

    async def _git_head(self) -> str | None:
        """SHA de HEAD del árbol indexado (None si no es repo git)."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "rev-parse", "HEAD",
                cwd=str(self.repo_root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            out, _ = await proc.communicate()
            if proc.returncode == 0:
                return out.decode().strip()
        except OSError:
            pass
        return None

    SNIPPET_MAX_CHARS = 1500

    async def search(self, query: str, top_k: int = 8) -> list[dict]:
        """Búsqueda semántica sobre code_chunks, enriquecida con cards.

        Returns:
            Lista rankeada de dicts con path, name, kind, líneas, blob_hash,
            score (1 - distancia coseno), snippet y card del archivo (o None).
        """
        emb = await asyncio.to_thread(self.embedder.encode, [query])
        q = list(map(float, emb[0]))
        res = await asyncio.to_thread(
            lambda: self.chunks.query(
                query_embeddings=[q],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        )
        if not res["ids"] or not res["ids"][0]:
            return []

        paths = list({m["path"] for m in res["metadatas"][0]})
        cards_res = await asyncio.to_thread(
            lambda: self.cards.get(ids=paths, include=["documents"])
        )
        cards_by_path = dict(zip(cards_res["ids"], cards_res["documents"]))

        results = []
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            results.append(
                {
                    "path": meta["path"],
                    "name": meta["name"],
                    "kind": meta["kind"],
                    "start_line": meta["start_line"],
                    "end_line": meta["end_line"],
                    "blob_hash": meta["blob_hash"],
                    "score": round(1.0 - dist, 4),
                    "snippet": doc[: self.SNIPPET_MAX_CHARS],
                    "card": cards_by_path.get(meta["path"]),
                }
            )
        return results
