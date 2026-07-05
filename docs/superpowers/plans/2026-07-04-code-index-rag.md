# Code-Index RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Servicio `code-index` en el server (:9515) que indexa el codebase (chunks AST + cards MiniMax) en un Chroma propio y lo expone a los agentes vía HTTP + CLI, con reindex incremental disparado por deploy.

**Architecture:** Servicio aiohttp standalone (`python -m src.code_index`) bajo systemd --user `kza`, con Chroma persistente propio en `/home/kza/code-index/`, embeddings BGE-M3 **en CPU** (cero VRAM) y cards por archivo generadas con MiniMax vía gateway :8200. Reindex incremental por `git hash-object` disparado por hook `post-merge` del server. Los agentes consultan con `tools/code_search.py`, que detecta drift laptop↔server por hash.

**Tech Stack:** Python 3.13, aiohttp (ya en requirements), chromadb `PersistentClient`, sentence-transformers (BGE-M3, `device="cpu"`), openai `AsyncOpenAI` contra gateway :8200, `ast` stdlib, pytest + pytest-asyncio (`asyncio_mode = auto`).

**Spec:** `docs/superpowers/specs/2026-07-04-code-index-rag-design.md`

## Global Constraints

- Python del venv SIEMPRE: `/Users/yo/Documents/kza/.venv/bin/python -m pytest ...` (el python3 del sistema es 3.9 y rompe).
- Imports absolutos: `from src.code_index.chunker import ...` — nunca relativos.
- `async/await` para todo I/O; trabajo síncrono pesado (encode, Chroma, disco) via `asyncio.to_thread`.
- `@dataclass` para DTOs; `logger = logging.getLogger(__name__)`; docstrings Google-style en API pública; type hints.
- Config SOLO en `config/settings.yaml` (sección `code_index:` nueva) — ningún archivo de config nuevo.
- Embedder `BAAI/bge-m3` con `device="cpu"` — NUNCA GPU (cuda:0 al límite; regla de proyecto).
- Cards: gateway `http://192.168.1.2:8200/v1`, model `MiniMax-M2.7-highspeed`, `api_key_env=MINIMAX_API_KEY`, api chat. MiniMax emite `<think>...</think>` → SIEMPRE strippear.
- Puerto del servicio: **9515** (sub-rango KZA 9500-9599; :9500 ocupado por obs).
- Este servicio NO toca kza-voice, ni el Chroma del pipeline, ni GPUs.
- Tests nuevos en `tests/unit/code_index/`; mocks reutilizables en `tests/mocks/`.
- Commits frecuentes, mensajes `feat(code-index): ...` / `test(code-index): ...`.
- No indexar `tests/`, `docs/`, `scripts/` en v1 — solo `src/**/*.py`.

---

### Task 1: Chunker AST

**Files:**
- Create: `src/code_index/__init__.py` (vacío)
- Create: `src/code_index/chunker.py`
- Create: `tests/unit/code_index/__init__.py` (vacío)
- Test: `tests/unit/code_index/test_chunker.py`

**Interfaces:**
- Consumes: nada (stdlib `ast`).
- Produces: `@dataclass CodeChunk(chunk_id: str, path: str, name: str, kind: str, start_line: int, end_line: int, text: str)` y `extract_chunks(source: str, path: str) -> list[CodeChunk]`. `kind ∈ {"function", "method", "class"}`. `chunk_id = f"{path}::{name}"`, `name` calificado (`Clase.metodo`). Líneas 1-indexed inclusivas. El chunk de una clase cubre desde su decorador/`class` hasta la línea previa al primer método (o toda la clase si no tiene métodos).

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/code_index/test_chunker.py
"""Tests del chunker AST del code-index."""

from src.code_index.chunker import CodeChunk, extract_chunks

SAMPLE = '''\
import os


def top_level(a: int) -> int:
    """Suma uno."""
    return a + 1


@property_like
def decorated():
    return 2


class Foo:
    """Docstring de Foo."""

    ATTR = 1

    def method_a(self):
        return self.ATTR

    async def method_b(self):
        return 2


class Empty:
    """Sin métodos."""

    X = 5
'''


def test_top_level_function_chunk():
    chunks = extract_chunks(SAMPLE, "src/foo.py")
    fn = next(c for c in chunks if c.name == "top_level")
    assert fn.kind == "function"
    assert fn.chunk_id == "src/foo.py::top_level"
    assert fn.start_line == 4
    assert fn.end_line == 6
    assert '"""Suma uno."""' in fn.text
    assert fn.text.startswith("def top_level")


def test_decorator_included_in_chunk():
    chunks = extract_chunks(SAMPLE, "src/foo.py")
    fn = next(c for c in chunks if c.name == "decorated")
    assert fn.text.startswith("@property_like")
    assert fn.start_line == 9


def test_methods_have_qualified_name():
    chunks = extract_chunks(SAMPLE, "src/foo.py")
    names = {c.name for c in chunks}
    assert "Foo.method_a" in names
    assert "Foo.method_b" in names
    mb = next(c for c in chunks if c.name == "Foo.method_b")
    assert mb.kind == "method"
    assert mb.text.startswith("async def method_b")


def test_class_chunk_is_header_without_method_bodies():
    chunks = extract_chunks(SAMPLE, "src/foo.py")
    cls = next(c for c in chunks if c.name == "Foo")
    assert cls.kind == "class"
    assert '"""Docstring de Foo."""' in cls.text
    assert "ATTR = 1" in cls.text
    assert "def method_a" not in cls.text


def test_class_without_methods_is_full_chunk():
    chunks = extract_chunks(SAMPLE, "src/foo.py")
    cls = next(c for c in chunks if c.name == "Empty")
    assert "X = 5" in cls.text


def test_syntax_error_returns_empty():
    assert extract_chunks("def broken(:\n  pass", "src/bad.py") == []


def test_empty_source_returns_empty():
    assert extract_chunks("", "src/empty.py") == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/code_index/test_chunker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.code_index'`

- [ ] **Step 3: Write the implementation**

```python
# src/code_index/chunker.py
"""Chunker AST: parte un archivo Python en chunks por función/método/clase."""

import ast
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CodeChunk:
    """Un fragmento indexable del codebase."""

    chunk_id: str    # "<path>::<nombre calificado>"
    path: str        # relativo al repo, ej. "src/llm/reasoner.py"
    name: str        # "HttpReasoner.load" | "main" | "HttpReasoner"
    kind: str        # "function" | "method" | "class"
    start_line: int  # 1-indexed, inclusivo
    end_line: int    # 1-indexed, inclusivo
    text: str


def extract_chunks(source: str, path: str) -> list[CodeChunk]:
    """Extraer chunks de un archivo Python.

    Args:
        source: Contenido del archivo.
        path: Ruta relativa al repo (para chunk_id y metadata).

    Returns:
        Lista de CodeChunk. Vacía si el archivo no parsea (se loguea warning).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        logger.warning(f"[Chunker] Syntax error en {path}: {e}")
        return []

    lines = source.splitlines()
    chunks: list[CodeChunk] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunks.append(_make_chunk(node, node.name, "function", path, lines))
        elif isinstance(node, ast.ClassDef):
            chunks.extend(_class_chunks(node, path, lines))
    return chunks


def _node_start(node: ast.AST) -> int:
    """Línea de inicio incluyendo decoradores."""
    decorators = getattr(node, "decorator_list", [])
    if decorators:
        return min(d.lineno for d in decorators)
    return node.lineno


def _make_chunk(
    node: ast.AST,
    qualname: str,
    kind: str,
    path: str,
    lines: list[str],
    end_line: int | None = None,
) -> CodeChunk:
    start = _node_start(node)
    end = end_line if end_line is not None else node.end_lineno
    return CodeChunk(
        chunk_id=f"{path}::{qualname}",
        path=path,
        name=qualname,
        kind=kind,
        start_line=start,
        end_line=end,
        text="\n".join(lines[start - 1 : end]),
    )


def _class_chunks(node: ast.ClassDef, path: str, lines: list[str]) -> list[CodeChunk]:
    """Una clase produce: 1 chunk de header (hasta el primer método) + 1 por método."""
    methods = [
        n for n in node.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    header_end = _node_start(methods[0]) - 1 if methods else node.end_lineno
    chunks = [_make_chunk(node, node.name, "class", path, lines, end_line=header_end)]
    for m in methods:
        chunks.append(_make_chunk(m, f"{node.name}.{m.name}", "method", path, lines))
    return chunks
```

También crear vacíos: `src/code_index/__init__.py` y `tests/unit/code_index/__init__.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/code_index/test_chunker.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/code_index/ tests/unit/code_index/
git commit -m "feat(code-index): AST chunker — chunks por función/método/clase"
```

---

### Task 2: Manifest incremental

**Files:**
- Create: `src/code_index/manifest.py`
- Test: `tests/unit/code_index/test_manifest.py`

**Interfaces:**
- Consumes: nada.
- Produces:
  - `git_blob_hash(data: bytes) -> str` — SHA1 estilo `git hash-object` (comparable con hashes del repo).
  - `@dataclass ManifestDiff(added: list[str], changed: list[str], deleted: list[str])` con property `empty: bool`.
  - `IndexManifest(path: Path)` con: attrs `files: dict[str, dict]` (path → `{"hash": str, "card_done": bool}`) y `head_sha: str | None`; métodos `load()`, `save()`, `diff(current: dict[str, str]) -> ManifestDiff`, `update_file(path: str, blob_hash: str, card_done: bool)` (persiste inmediatamente), `remove_file(path: str)` (persiste inmediatamente). Un archivo con `card_done=False` aparece en `changed` aunque el hash no cambie (retry de cards).

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/code_index/test_manifest.py
"""Tests del manifest incremental del code-index."""

import json

from src.code_index.manifest import IndexManifest, git_blob_hash


def test_git_blob_hash_matches_git():
    # `echo "hello" | git hash-object --stdin` → ce01362...
    assert git_blob_hash(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"


def test_diff_detects_added_changed_deleted(tmp_path):
    m = IndexManifest(tmp_path / "manifest.json")
    m.files = {
        "src/a.py": {"hash": "h1", "card_done": True},
        "src/b.py": {"hash": "h2", "card_done": True},
    }
    diff = m.diff({"src/a.py": "h1-nuevo", "src/c.py": "h3"})
    assert diff.changed == ["src/a.py"]
    assert diff.added == ["src/c.py"]
    assert diff.deleted == ["src/b.py"]
    assert not diff.empty


def test_diff_retries_pending_cards():
    m = IndexManifest.__new__(IndexManifest)
    m.files = {"src/a.py": {"hash": "h1", "card_done": False}}
    m.head_sha = None
    diff = m.diff({"src/a.py": "h1"})  # hash igual pero card pendiente
    assert diff.changed == ["src/a.py"]


def test_diff_empty_when_no_changes(tmp_path):
    m = IndexManifest(tmp_path / "manifest.json")
    m.files = {"src/a.py": {"hash": "h1", "card_done": True}}
    assert m.diff({"src/a.py": "h1"}).empty


def test_save_load_roundtrip(tmp_path):
    path = tmp_path / "sub" / "manifest.json"
    m = IndexManifest(path)
    m.files = {"src/a.py": {"hash": "h1", "card_done": True}}
    m.head_sha = "abc123"
    m.save()

    m2 = IndexManifest(path)
    m2.load()
    assert m2.files == m.files
    assert m2.head_sha == "abc123"


def test_load_missing_file_is_noop(tmp_path):
    m = IndexManifest(tmp_path / "nope.json")
    m.load()
    assert m.files == {}
    assert m.head_sha is None


def test_update_and_remove_persist_immediately(tmp_path):
    path = tmp_path / "manifest.json"
    m = IndexManifest(path)
    m.update_file("src/a.py", "h1", card_done=False)
    on_disk = json.loads(path.read_text())
    assert on_disk["files"]["src/a.py"] == {"hash": "h1", "card_done": False}

    m.remove_file("src/a.py")
    on_disk = json.loads(path.read_text())
    assert on_disk["files"] == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/code_index/test_manifest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.code_index.manifest'`

- [ ] **Step 3: Write the implementation**

```python
# src/code_index/manifest.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/code_index/test_manifest.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/code_index/manifest.py tests/unit/code_index/test_manifest.py
git commit -m "feat(code-index): manifest incremental con git blob hash y retry de cards"
```

---

### Task 3: CardGenerator (MiniMax vía gateway)

**Files:**
- Create: `src/code_index/cards.py`
- Test: `tests/unit/code_index/test_cards.py`

**Interfaces:**
- Consumes: nada del proyecto (cliente `openai.AsyncOpenAI` lazy).
- Produces: `CardGenerator(base_url: str, model: str, api_key_env: str = "MINIMAX_API_KEY", timeout: float = 120.0, max_source_chars: int = 48000)` con `async generate(path: str, source: str) -> str` (markdown de la card, con `<think>` strippeado; raises si el gateway falla) y helper `_strip_think(text: str) -> str`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/code_index/test_cards.py
"""Tests del generador de cards (MiniMax vía gateway)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

from src.code_index.cards import CardGenerator, _strip_think


def _fake_response(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def test_strip_think_removes_block():
    text = "<think>\nrazonando...\n</think>\n## Propósito\nAlgo."
    assert _strip_think(text).strip().startswith("## Propósito")


def test_strip_think_noop_without_block():
    assert _strip_think("## Propósito\nX") == "## Propósito\nX"


async def test_generate_returns_stripped_card():
    gen = CardGenerator(base_url="http://fake:8200/v1", model="test-model")
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(
                    return_value=_fake_response("<think>x</think>## Propósito\nCard.")
                )
            )
        )
    )
    gen._client = fake_client

    card = await gen.generate("src/foo.py", "def f(): pass")

    assert card.startswith("## Propósito")
    call = fake_client.chat.completions.create.call_args
    assert call.kwargs["model"] == "test-model"
    prompt = call.kwargs["messages"][0]["content"]
    assert "src/foo.py" in prompt
    assert "def f(): pass" in prompt


async def test_generate_truncates_long_source():
    gen = CardGenerator(
        base_url="http://fake:8200/v1", model="m", max_source_chars=10
    )
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(return_value=_fake_response("card"))
            )
        )
    )
    gen._client = fake_client

    await gen.generate("src/foo.py", "x" * 100)

    prompt = fake_client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "x" * 11 not in prompt
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/code_index/test_cards.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.code_index.cards'`

- [ ] **Step 3: Write the implementation**

```python
# src/code_index/cards.py
"""Generador de "cards" por archivo vía gateway :8200 (MiniMax, OpenAI-compat)."""

import logging
import os
import re

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

CARD_PROMPT = """Sos un ingeniero senior documentando un codebase Python de un asistente de voz local (KZA).
Generá una "card" en markdown para el archivo `{path}`. Secciones exactas:

## Propósito
(2-3 frases: qué resuelve este archivo dentro del sistema)

## API pública
(clases/funciones públicas con firma y una línea de descripción cada una)

## Dependencias
(qué usa: otros módulos src.*, servicios externos, hardware)

## Invariantes y gotchas
(supuestos, órdenes de llamada requeridos, edge cases no obvios; si no hay, "—")

Máximo ~300 palabras. Sin preámbulo ni cierre: solo la card.

```python
{source}
```"""


def _strip_think(text: str) -> str:
    """MiniMax emite bloques <think> — sacarlos siempre."""
    return _THINK_RE.sub("", text)


class CardGenerator:
    """Genera resúmenes ("cards") de archivo con MiniMax vía el gateway LiteLLM."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key_env: str = "MINIMAX_API_KEY",
        timeout: float = 120.0,
        max_source_chars: int = 48_000,
    ):
        self.base_url = base_url
        self.model = model
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.max_source_chars = max_source_chars
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=os.environ.get(self.api_key_env, "dummy"),
                timeout=self.timeout,
            )
        return self._client

    async def generate(self, path: str, source: str) -> str:
        """Generar la card markdown de un archivo.

        Raises:
            Exception: si el gateway falla (el caller decide el retry —
            el manifest marca card_done=False y se reintenta en el próximo
            reindex).
        """
        prompt = CARD_PROMPT.format(path=path, source=source[: self.max_source_chars])
        client = self._get_client()
        resp = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content or ""
        return _strip_think(text).strip()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/code_index/test_cards.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/code_index/cards.py tests/unit/code_index/test_cards.py
git commit -m "feat(code-index): CardGenerator — cards por archivo con MiniMax vía gateway"
```

---

### Task 4: CodeIndexer (reindex incremental)

**Files:**
- Create: `src/code_index/indexer.py`
- Create: `tests/mocks/mock_chroma.py`
- Test: `tests/unit/code_index/test_indexer.py`

**Interfaces:**
- Consumes: `extract_chunks` (Task 1), `IndexManifest`/`git_blob_hash`/`ManifestDiff` (Task 2), `CardGenerator.generate` (Task 3).
- Produces: `CodeIndexer(repo_root: Path, chunks_collection, cards_collection, embedder, card_generator, manifest: IndexManifest, include_globs: list[str] | None = None)` con:
  - `async reindex(mode: str = "incremental") -> dict` — stats `{"indexed": int, "deleted": int, "cards_failed": int}`. `mode="full"` resetea el manifest antes.
  - property `running: bool` (lock tomado) y attr `last_stats: dict | None`.
  - `include_globs` default `["src/**/*.py"]`.
  - Metadata de chunk en Chroma: `{"path", "name", "kind", "start_line", "end_line", "blob_hash"}`; id = `chunk_id`. Cards: id = `path`, metadata `{"path", "blob_hash"}`.
- El mock de Chroma (`tests/mocks/mock_chroma.py`) expone `MockChromaCollection` con `add/delete/query/get` compatibles con la API real usada.

- [ ] **Step 1: Write the mock de Chroma**

```python
# tests/mocks/mock_chroma.py
"""Mock in-memory de una colección Chroma (API mínima usada por code_index)."""

import math


class MockChromaCollection:
    """Colección Chroma fake: almacenamiento dict + cosine por fuerza bruta."""

    def __init__(self, name: str = "mock"):
        self.name = name
        self._store: dict[str, dict] = {}  # id -> {embedding, document, metadata}

    def add(self, ids, embeddings, documents, metadatas):
        for i, id_ in enumerate(ids):
            self._store[id_] = {
                "embedding": list(embeddings[i]),
                "document": documents[i],
                "metadata": metadatas[i],
            }

    def delete(self, ids=None, where=None):
        if ids is not None:
            for id_ in ids:
                self._store.pop(id_, None)
        if where is not None:
            matches = [
                id_ for id_, e in self._store.items()
                if all(e["metadata"].get(k) == v for k, v in where.items())
            ]
            for id_ in matches:
                self._store.pop(id_)

    def get(self, ids=None, include=None):
        found = [i for i in (ids or list(self._store)) if i in self._store]
        return {
            "ids": found,
            "documents": [self._store[i]["document"] for i in found],
            "metadatas": [self._store[i]["metadata"] for i in found],
        }

    def query(self, query_embeddings, n_results=8, include=None):
        q = query_embeddings[0]

        def dist(id_):
            e = self._store[id_]["embedding"]
            dot = sum(a * b for a, b in zip(q, e))
            nq = math.sqrt(sum(a * a for a in q)) or 1.0
            ne = math.sqrt(sum(a * a for a in e)) or 1.0
            return 1.0 - dot / (nq * ne)

        ranked = sorted(self._store, key=dist)[:n_results]
        return {
            "ids": [ranked],
            "documents": [[self._store[i]["document"] for i in ranked]],
            "metadatas": [[self._store[i]["metadata"] for i in ranked]],
            "distances": [[dist(i) for i in ranked]],
        }

    def count(self):
        return len(self._store)
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/unit/code_index/test_indexer.py
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
    assert stats == {"indexed": 2, "deleted": 0, "cards_failed": 0}
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/code_index/test_indexer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.code_index.indexer'`

- [ ] **Step 4: Write the implementation**

```python
# src/code_index/indexer.py
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/code_index/test_indexer.py -v`
Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add src/code_index/indexer.py tests/mocks/mock_chroma.py tests/unit/code_index/test_indexer.py
git commit -m "feat(code-index): CodeIndexer — reindex incremental con purge y retry de cards"
```

---

### Task 5: Búsqueda semántica (CodeIndexer.search)

**Files:**
- Modify: `src/code_index/indexer.py` (agregar método `search` a `CodeIndexer`)
- Test: `tests/unit/code_index/test_search.py`

**Interfaces:**
- Consumes: `CodeIndexer` (Task 4) con sus colecciones.
- Produces: `async CodeIndexer.search(query: str, top_k: int = 8) -> list[dict]`. Cada dict: `{"path", "name", "kind", "start_line", "end_line", "blob_hash", "score" (1-distancia, redondeado a 4), "snippet" (documento truncado a 1500 chars), "card" (str | None — card del archivo si existe)}`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/code_index/test_search.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/code_index/test_search.py -v`
Expected: FAIL — `AttributeError: 'CodeIndexer' object has no attribute 'search'`

- [ ] **Step 3: Write the implementation** (agregar al final de la clase `CodeIndexer`)

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/code_index/ -v`
Expected: todos los tests del módulo passed (chunker + manifest + cards + indexer + search)

- [ ] **Step 5: Commit**

```bash
git add src/code_index/indexer.py tests/unit/code_index/test_search.py
git commit -m "feat(code-index): búsqueda semántica con cards y score coseno"
```

---

### Task 6: Servicio HTTP aiohttp + entry point

**Files:**
- Create: `src/code_index/service.py`
- Create: `src/code_index/__main__.py`
- Test: `tests/unit/code_index/test_service.py`

**Interfaces:**
- Consumes: `CodeIndexer` (Tasks 4-5): `search()`, `reindex()`, `running`, `last_stats`, `manifest.head_sha`, `manifest.files`.
- Produces:
  - `create_app(indexer) -> aiohttp.web.Application` con rutas: `GET /health`, `POST /search` (`{"query": str, "top_k"?: int}` → `{"results": [...]}`, 400 si query vacía), `POST /reindex` (`{"mode"?: "incremental"|"full"}` → 202 `{"status": "started"}`, 409 si ya corre).
  - `build_indexer(cfg: dict, repo_root: Path) -> CodeIndexer` — factory con Chroma/embedder/cards reales.
  - `python -m src.code_index` levanta el servicio leyendo `config/settings.yaml` sección `code_index`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/code_index/test_service.py
"""Tests del servicio HTTP del code-index."""

import asyncio

import pytest
from aiohttp.test_utils import TestClient, TestServer

from src.code_index.service import create_app
from tests.unit.code_index.test_indexer import make_indexer, repo  # noqa: F401


@pytest.fixture
async def client(repo):  # noqa: F811
    idx, *_ = make_indexer(repo)
    app = create_app(idx)
    c = TestClient(TestServer(app))
    await c.start_server()
    yield c
    await c.close()


async def test_health(client):
    resp = await client.post("/reindex")
    assert resp.status == 202

    # esperar a que el reindex en background termine
    for _ in range(50):
        h = await (await client.get("/health")).json()
        if not h["reindex_running"] and h["files"] == 2:
            break
        await asyncio.sleep(0.05)
    assert h["status"] == "ok"
    assert h["files"] == 2
    assert h["last_stats"]["indexed"] == 2


async def test_search_endpoint(client):
    await client.post("/reindex")
    for _ in range(50):
        h = await (await client.get("/health")).json()
        if h["files"] == 2:
            break
        await asyncio.sleep(0.05)

    resp = await client.post("/search", json={"query": "funcion a", "top_k": 1})
    assert resp.status == 200
    data = await resp.json()
    assert len(data["results"]) == 1
    assert "path" in data["results"][0]


async def test_search_empty_query_is_400(client):
    resp = await client.post("/search", json={"query": "  "})
    assert resp.status == 400


async def test_reindex_returns_409_when_running(client, repo):  # noqa: F811
    # bloquear el lock del indexer manualmente para simular reindex en curso
    idx = client.server.app["indexer"]
    await idx._lock.acquire()
    try:
        resp = await client.post("/reindex")
        assert resp.status == 409
    finally:
        idx._lock.release()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/code_index/test_service.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.code_index.service'`

- [ ] **Step 3: Write the implementation**

```python
# src/code_index/service.py
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
```

```python
# src/code_index/__main__.py
"""Entry point: python -m src.code_index (servicio en el server, systemd --user)."""

import logging
import os
from pathlib import Path

import yaml
from aiohttp import web

from src.code_index.service import build_indexer, create_app


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    config_path = os.environ.get("CONFIG_PATH", "config/settings.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)["code_index"]

    indexer = build_indexer(cfg, repo_root=Path.cwd())
    app = create_app(indexer)
    web.run_app(app, host=cfg.get("host", "0.0.0.0"), port=cfg.get("port", 9515))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/code_index/test_service.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/code_index/service.py src/code_index/__main__.py tests/unit/code_index/test_service.py
git commit -m "feat(code-index): servicio HTTP aiohttp (/health /search /reindex) + entry point"
```

---

### Task 7: Config, unit file y hook post-merge

**Files:**
- Modify: `config/settings.yaml` (agregar sección `code_index:` al final)
- Create: `scripts/kza-code-index.service`
- Create: `scripts/install_code_index_hook.sh`

**Interfaces:**
- Consumes: claves que lee `build_indexer` (Task 6): `chroma_path`, `manifest_path`, `include_globs`, `embedder.{model,device}`, `cards.{base_url,model,api_key_env,timeout}`, `host`, `port`.
- Produces: config canónica en settings.yaml; unit file para el server; script que instala el hook `post-merge` (deploy = `git pull` en el server → dispara reindex incremental).

- [ ] **Step 1: Agregar la sección a `config/settings.yaml`** (al final del archivo)

```yaml

# Code-Index — búsqueda semántica del codebase para agentes (spec 2026-07-04).
# Servicio SEPARADO (kza-code-index.service, :9515); el pipeline de voz NO lo
# consume. Embedder SIEMPRE en CPU: cuda:0 está al límite (regla de proyecto).
code_index:
  enabled: true
  host: "0.0.0.0"
  port: 9515                    # sub-rango KZA 9500-9599 (:9500 = obs)
  chroma_path: "/home/kza/code-index/chroma"
  manifest_path: "/home/kza/code-index/manifest.json"
  include_globs:
    - "src/**/*.py"             # v1: solo código productivo (sin tests/docs)
  embedder:
    model: "BAAI/bge-m3"
    device: "cpu"               # NUNCA GPU
  cards:
    base_url: "http://192.168.1.2:8200/v1"   # gateway LiteLLM → MiniMax
    model: "MiniMax-M2.7-highspeed"
    api_key_env: "MINIMAX_API_KEY"            # virtual key del gateway
    timeout: 120
```

- [ ] **Step 2: Crear el unit file**

```ini
# scripts/kza-code-index.service
# Instalar en el server: cp a ~/.config/systemd/user/ + daemon-reload + enable --now
[Unit]
Description=KZA code-index — búsqueda semántica del codebase (port 9515)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/home/kza/app
EnvironmentFile=/home/kza/secrets/.env
ExecStart=/home/kza/app/.venv/bin/python -m src.code_index
Restart=on-failure
RestartSec=10s
StartLimitBurst=3
StartLimitIntervalSec=300

[Install]
WantedBy=default.target
```

- [ ] **Step 3: Crear el script de instalación del hook**

```bash
#!/usr/bin/env bash
# scripts/install_code_index_hook.sh
#
# Instala el hook post-merge que dispara el reindex incremental del
# code-index después de cada deploy (git pull) EN EL SERVER.
#
# Uso (en el server): bash scripts/install_code_index_hook.sh
set -euo pipefail

HOOK="$(git rev-parse --git-dir)/hooks/post-merge"

cat > "$HOOK" <<'EOF'
#!/usr/bin/env bash
# Dispara reindex incremental del code-index tras cada git pull (deploy).
# No bloquea el deploy si el servicio está caído.
curl -fsS -X POST -m 5 http://127.0.0.1:9515/reindex \
  -H 'Content-Type: application/json' -d '{"mode":"incremental"}' \
  || echo "[post-merge] code-index no disponible (reindex omitido)"
EOF

chmod +x "$HOOK"
echo "✓ hook post-merge instalado en $HOOK"
```

- [ ] **Step 4: Validar sintaxis**

Run:
```bash
/Users/yo/Documents/kza/.venv/bin/python -c "import yaml; cfg = yaml.safe_load(open('config/settings.yaml'))['code_index']; assert cfg['port'] == 9515 and cfg['embedder']['device'] == 'cpu'; print('yaml OK')"
bash -n scripts/install_code_index_hook.sh && echo "hook script OK"
```
Expected: `yaml OK` y `hook script OK`

- [ ] **Step 5: Commit**

```bash
git add config/settings.yaml scripts/kza-code-index.service scripts/install_code_index_hook.sh
git commit -m "feat(code-index): config settings.yaml + unit file + hook post-merge de deploy"
```

---

### Task 8: CLI `tools/code_search.py` (cliente para agentes)

**Files:**
- Create: `tools/code_search.py`
- Test: `tests/unit/code_index/test_code_search_cli.py`

**Interfaces:**
- Consumes: endpoint `POST /search` (Task 6). **Standalone a propósito**: solo stdlib (urllib), sin imports de `src.*` — duplica `git_blob_hash` para poder correr sin venv/PYTHONPATH.
- Produces: CLI `python tools/code_search.py "<query>" [--top-k N] [--url URL]`. Funciones testeables: `git_blob_hash(data: bytes) -> str`, `format_result(res: dict, repo_root: Path, seen_cards: set) -> str` (marca `⚠ STALE` si el hash local difiere del indexado, `⚠ LOCAL MISSING` si el archivo no existe local). Exit codes: 0 ok, 1 servicio no disponible, 2 argumentos inválidos.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/code_index/test_code_search_cli.py
"""Tests de la lógica de formato/drift del CLI code_search."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "code_search",
    Path(__file__).resolve().parents[3] / "tools" / "code_search.py",
)
code_search = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(code_search)


def _result(path="src/a.py", blob_hash="X"):
    return {
        "path": path,
        "name": "fa",
        "kind": "function",
        "start_line": 1,
        "end_line": 2,
        "blob_hash": blob_hash,
        "score": 0.9,
        "snippet": "def fa():\n    return 1",
        "card": "## Propósito\nCard de a.",
    }


def test_blob_hash_matches_git():
    assert (
        code_search.git_blob_hash(b"hello\n")
        == "ce013625030ba8dba906f756967f9e9ca394464a"
    )


def test_format_fresh_result_no_stale(tmp_path):
    f = tmp_path / "src" / "a.py"
    f.parent.mkdir()
    f.write_text("def fa():\n    return 1\n")
    res = _result(blob_hash=code_search.git_blob_hash(f.read_bytes()))

    out = code_search.format_result(res, tmp_path, seen_cards=set())

    assert "STALE" not in out
    assert "src/a.py:1-2" in out
    assert "## Propósito" in out


def test_format_stale_when_local_differs(tmp_path):
    f = tmp_path / "src" / "a.py"
    f.parent.mkdir()
    f.write_text("def fa():\n    return 999\n")
    res = _result(blob_hash="hash-viejo-del-indice")

    out = code_search.format_result(res, tmp_path, seen_cards=set())

    assert "⚠ STALE" in out


def test_format_missing_local_file(tmp_path):
    out = code_search.format_result(_result(), tmp_path, seen_cards=set())
    assert "⚠ LOCAL MISSING" in out


def test_card_printed_once_per_path(tmp_path):
    seen: set = set()
    out1 = code_search.format_result(_result(), tmp_path, seen_cards=seen)
    out2 = code_search.format_result(_result(), tmp_path, seen_cards=seen)
    assert "## Propósito" in out1
    assert "## Propósito" not in out2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/code_index/test_code_search_cli.py -v`
Expected: FAIL — `FileNotFoundError` (tools/code_search.py no existe)

- [ ] **Step 3: Write the implementation**

```python
#!/usr/bin/env python3
# tools/code_search.py
"""Búsqueda semántica en el code-index del server (para agentes).

Uso:
    python tools/code_search.py "cómo se maneja el timeout de HA al boot"
    python tools/code_search.py "reconexión websocket" --top-k 5

Standalone a propósito (solo stdlib): corre sin venv ni PYTHONPATH.
Los resultados ⚠ STALE difieren del índice (rama local sin deployar):
leé el archivo real en vez de confiar en el snippet.
Si el servicio no responde: fallback a Grep/Glob de siempre (exit 1).
"""

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_URL = "http://192.168.1.2:9515"
SNIPPET_PRINT_CHARS = 1200


def git_blob_hash(data: bytes) -> str:
    """SHA1 estilo `git hash-object` (duplicado de src a propósito: standalone)."""
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()


def format_result(res: dict, repo_root: Path, seen_cards: set) -> str:
    """Formatear un resultado con marca de drift y card (una vez por path)."""
    local = repo_root / res["path"]
    if not local.exists():
        drift = "  ⚠ LOCAL MISSING"
    elif git_blob_hash(local.read_bytes()) != res["blob_hash"]:
        drift = "  ⚠ STALE (difiere del índice — leer el archivo real)"
    else:
        drift = ""

    lines = [
        f"== {res['path']}:{res['start_line']}-{res['end_line']}  "
        f"{res['name']} [{res['kind']}] score={res['score']:.3f}{drift}"
    ]
    if res.get("card") and res["path"] not in seen_cards:
        seen_cards.add(res["path"])
        lines += ["--- card ---", res["card"], "------------"]
    lines.append(res["snippet"][:SNIPPET_PRINT_CHARS])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--url", default=DEFAULT_URL)
    args = parser.parse_args()

    payload = json.dumps({"query": args.query, "top_k": args.top_k}).encode()
    req = urllib.request.Request(
        f"{args.url}/search",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        print(
            f"code-index no disponible ({e}). Fallback: usar Grep/Glob.",
            file=sys.stderr,
        )
        return 1

    repo_root = Path(__file__).resolve().parent.parent
    seen_cards: set = set()
    for res in data.get("results", []):
        print(format_result(res, repo_root, seen_cards))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/code_index/test_code_search_cli.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add tools/code_search.py tests/unit/code_index/test_code_search_cli.py
git commit -m "feat(code-index): CLI code_search.py con detección de drift por blob hash"
```

---

### Task 9: Documentación + suite completa

**Files:**
- Modify: `CLAUDE.md` (fila nueva en "Mapa de Archivos Clave" + comando en "Comandos de Desarrollo")
- Create: `docs/runbooks/CODE_INDEX.md`

**Interfaces:**
- Consumes: todo lo anterior (documenta el sistema terminado).
- Produces: documentación operativa del deploy y uso.

- [ ] **Step 1: Agregar a CLAUDE.md**

En la tabla "Mapa de Archivos Clave", después de la fila de `src/vectordb/chroma_sync.py`:

```markdown
| `src/code_index/` | Servicio índice semántico del codebase (:9515) | Cambios en búsqueda de código para agentes |
```

En "Comandos de Desarrollo", después del bloque "Benchmark":

```markdown
# Búsqueda semántica del codebase (requiere kza-code-index en el server)
python tools/code_search.py "cómo se maneja el timeout de HA al boot"
```

- [ ] **Step 2: Crear el runbook**

```markdown
# docs/runbooks/CODE_INDEX.md
# Code-Index — Runbook

Servicio de búsqueda semántica del codebase para agentes (spec
`docs/superpowers/specs/2026-07-04-code-index-rag-design.md`).

## Qué es

- `kza-code-index.service` (systemd --user `kza`) en `:9515`.
- Chroma persistente propio: `/home/kza/code-index/chroma/` (colecciones
  `code_chunks` y `code_cards`) + manifest `/home/kza/code-index/manifest.json`.
- Embeddings BGE-M3 **en CPU** (cero VRAM). Cards por archivo con MiniMax vía
  gateway :8200 (`MINIMAX_API_KEY` del `.env` = virtual key del gateway).
- NO toca kza-voice ni el Chroma del pipeline.

## Deploy inicial (en el server)

```bash
ssh kza
cd /home/kza/app && git pull
mkdir -p /home/kza/code-index
cp scripts/kza-code-index.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now kza-code-index
bash scripts/install_code_index_hook.sh   # hook post-merge → reindex por deploy

# primer indexado (full): ~300 archivos, cards MiniMax — tarda varios minutos
curl -X POST localhost:9515/reindex -H 'Content-Type: application/json' -d '{"mode":"full"}'
watch -n 5 'curl -s localhost:9515/health'
```

## Uso (desde la laptop)

```bash
python tools/code_search.py "dónde se reintenta la conexión al gateway"
```

- `⚠ STALE` = el archivo local difiere del indexado (rama sin deployar) →
  leer el archivo real.
- Servicio caído → exit 1 con mensaje; fallback a Grep/Glob.

## Operación

| Acción | Comando |
|--------|---------|
| Estado | `curl -s localhost:9515/health` |
| Reindex incremental manual | `curl -X POST localhost:9515/reindex` |
| Reindex full (reconstruir) | `curl -X POST localhost:9515/reindex -d '{"mode":"full"}' -H 'Content-Type: application/json'` |
| Logs | `journalctl --user -u kza-code-index -f` |
| Reset total | parar servicio, borrar `/home/kza/code-index/`, arrancar, reindex full |

## Notas

- El reindex automático corre en el hook `post-merge` del repo del server
  (cada `git pull` de deploy). Si el servicio está caído el deploy NO se
  bloquea (el hook solo avisa).
- Cards fallidas (gateway caído) quedan `card_done: false` en el manifest y
  se reintentan solas en el próximo reindex.
```

- [ ] **Step 3: Correr la suite completa del módulo + smoke de imports**

Run:
```bash
/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/code_index/ -v
/Users/yo/Documents/kza/.venv/bin/python -c "import src.code_index.service, src.code_index.__main__; print('imports OK')"
```
Expected: todos passed + `imports OK`

- [ ] **Step 4: Correr la suite global para verificar cero regresiones**

Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/ -q 2>&1 | tail -5`
Expected: sin fallas NUEVAS (baseline conocida: 5 fallas pre-existentes documentadas en la sesión PLAN_MEJORAS 2026-06-09 — verificar que las fallas, si las hay, sean esas).

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md docs/runbooks/CODE_INDEX.md
git commit -m "docs(code-index): runbook de deploy/operación + mapa en CLAUDE.md"
```

---

## Post-plan (manual, fuera del alcance de los tasks)

1. Merge de `feat/code-index` a `main` + push (flujo normal laptop→GitHub).
2. Deploy en el server según `docs/runbooks/CODE_INDEX.md` (⚠ server = producción: coordinar, no reiniciar nada de voz).
3. Primer reindex full y verificación de calidad de cards con 2-3 queries reales.
4. Probar el flujo del agente: `python tools/code_search.py "..."` desde la laptop.
