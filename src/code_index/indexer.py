"""Orquestador del índice: scan → diff → chunk → embed → Chroma (+cards)."""

import asyncio
import logging
from pathlib import Path

from src.code_index.chunker import extract_chunks
from src.code_index.manifest import IndexManifest, git_blob_hash

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
        """Reindexar el repo. mode="full" resetea el manifest primero.

        Idempotente ante interrupciones: el manifest se persiste por archivo.
        """
        async with self._lock:
            if mode == "full":
                self.manifest.files = {}
            current = await asyncio.to_thread(self._scan)
            diff = self.manifest.diff(current)
            stats = {"indexed": 0, "deleted": 0, "cards_failed": 0}
            logger.info(
                f"[CodeIndexer] reindex mode={mode}: +{len(diff.added)} "
                f"~{len(diff.changed)} -{len(diff.deleted)}"
            )

            for path in diff.deleted:
                await asyncio.to_thread(self._purge_path, path)
                self.manifest.remove_file(path)
                stats["deleted"] += 1

            for path in diff.added + diff.changed:
                await self._index_file(path, current[path], stats)

            self.manifest.head_sha = await self._git_head()
            self.manifest.save()
            self.last_stats = stats
            logger.info(f"[CodeIndexer] reindex done: {stats}")
            return stats

    def _scan(self) -> dict[str, str]:
        """Hashear el árbol actual: {path relativo: git blob hash}."""
        current: dict[str, str] = {}
        for pattern in self.include_globs:
            for f in sorted(self.repo_root.glob(pattern)):
                rel = f.relative_to(self.repo_root).as_posix()
                current[rel] = git_blob_hash(f.read_bytes())
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
            self.chunks.add(
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

        card_done = False
        try:
            card = await self.card_generator.generate(path, source)
            card_emb = await asyncio.to_thread(self.embedder.encode, [card])
            self.cards.add(
                ids=[path],
                embeddings=[list(map(float, card_emb[0]))],
                documents=[card],
                metadatas=[{"path": path, "blob_hash": blob_hash}],
            )
            card_done = True
        except Exception as e:
            logger.warning(f"[CodeIndexer] Card falló para {path}: {e}")
            stats["cards_failed"] += 1

        self.manifest.update_file(path, blob_hash, card_done)
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
