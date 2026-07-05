"""Chunker AST: parte un archivo Python en chunks por función/método/clase."""

import ast
import logging
import textwrap
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

    Nota sobre chunk_id: dos nodos con el mismo nombre calificado (ej. un
    `@property` y su `@x.setter`, ambos "Clase.metodo") producirían el mismo
    "<path>::<nombre>" y romperían la unicidad que exige Chroma en `add()`.
    Por eso, si el nombre calificado se repite dentro del archivo, TODOS los
    chunks con ese nombre (incluido el primero) llevan sufijo `#<start_line>`
    en su chunk_id — así ningún id queda ambiguo entre sí.
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
    _dedupe_chunk_ids(chunks)
    return chunks


def _dedupe_chunk_ids(chunks: list[CodeChunk]) -> None:
    """Desambiguar chunk_ids repetidos (ej. @property + @x.setter) in-place.

    Todos los chunks cuyo nombre calificado colisiona con otro en el mismo
    archivo reciben el sufijo `#<start_line>` en su chunk_id, incluido el
    primero — ninguno queda "sin marcar" y ambiguo respecto de los demás.
    """
    by_name: dict[str, list[CodeChunk]] = {}
    for c in chunks:
        by_name.setdefault(c.name, []).append(c)
    for group in by_name.values():
        if len(group) > 1:
            for c in group:
                c.chunk_id = f"{c.chunk_id}#{c.start_line}"


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
    text = "\n".join(lines[start - 1 : end])
    # Dedent to remove common leading indentation
    text = textwrap.dedent(text)
    return CodeChunk(
        chunk_id=f"{path}::{qualname}",
        path=path,
        name=qualname,
        kind=kind,
        start_line=start,
        end_line=end,
        text=text,
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
