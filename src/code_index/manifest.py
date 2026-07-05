"""Manifest incremental: hash por archivo + SHA de HEAD indexado.

La escritura es por-archivo (update_file/remove_file persisten al toque),
así un reindex interrumpido retoma donde quedó sin corromper el estado.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


def git_blob_hash(data: bytes) -> str:
    """SHA1 estilo `git hash-object` (blob) — comparable con el repo."""
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()


@dataclass
class ManifestDiff:
    """Resultado de comparar el manifest contra el árbol actual."""

    added: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not (self.added or self.changed or self.deleted)


class IndexManifest:
    """Estado persistente del índice de código."""

    def __init__(self, path: Path):
        self._path = Path(path)
        self.files: dict[str, dict] = {}
        self.head_sha: str | None = None

    def load(self) -> None:
        """Cargar desde disco; si no existe, queda vacío."""
        if self._path.exists():
            data = json.loads(self._path.read_text())
            self.files = data.get("files", {})
            self.head_sha = data.get("head_sha")

    def save(self) -> None:
        """Persistir atómicamente (write tmp + replace)."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"files": self.files, "head_sha": self.head_sha}, indent=1)
        )
        tmp.replace(self._path)

    def diff(self, current: dict[str, str]) -> ManifestDiff:
        """Comparar contra hashes actuales {path: blob_hash}.

        Un archivo con card pendiente (card_done=False) cuenta como changed
        aunque el hash no haya cambiado — así las cards fallidas se reintentan.
        """
        d = ManifestDiff()
        for path, blob_hash in current.items():
            entry = self.files.get(path)
            if entry is None:
                d.added.append(path)
            elif entry["hash"] != blob_hash or not entry.get("card_done", False):
                d.changed.append(path)
        d.deleted = [p for p in self.files if p not in current]
        return d

    def update_file(self, path: str, blob_hash: str, card_done: bool) -> None:
        self.files[path] = {"hash": blob_hash, "card_done": card_done}
        self.save()

    def remove_file(self, path: str) -> None:
        self.files.pop(path, None)
        self.save()
