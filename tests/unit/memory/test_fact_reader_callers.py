"""Pin the Memoria view's claim that stored memories are not used in answers.

The NEXO Memoria copy (src/dashboard/frontend/home/nexo/memory.js) states that
in this version KZA does not use stored memories to answer. That holds only
while no production module outside ``src/memory`` reads facts through the
memory readers. This source scan fails as soon as one does, so the copy and
the code cannot drift apart silently.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / 'src'
MEMORY = SRC / 'memory'
# Readers that return stored fact content (or ids) for prompts or answers.
READERS = frozenset({
    'prompt_context', 'aprompt_context', 'build_context', 'search_relevant', 'asearch_relevant',
    'get_by_category', 'eligible_ids', 'acall_prompt',
})
# ``src.memory.prompt_context`` is also a helper module (SourcesInvalidated, run_private,
# merge_sources): importing or naming the module is not a fact read.
MODULE = ('src', 'memory', 'prompt_context')


def _dotted(node: ast.AST) -> tuple[str, ...] | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return tuple(reversed(parts))
    return None


def reader_uses(source: str, filename: str = '<scan>') -> list[str]:
    """Return ``line:name`` for every reference to a fact reader in one module's source."""
    found = []
    for node in ast.walk(ast.parse(source, filename)):
        if isinstance(node, ast.Attribute) and node.attr in READERS:
            dotted = _dotted(node)
            if dotted is not None and dotted[:len(MODULE)] == MODULE:
                continue
            found.append(f'{node.lineno}:{node.attr}')
        elif isinstance(node, ast.Name) and node.id in READERS:
            found.append(f'{node.lineno}:{node.id}')
        elif isinstance(node, ast.Constant) and node.value in READERS:
            # getattr(memory, 'search_relevant') and similar indirections.
            found.append(f'{node.lineno}:{node.value!r}')
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('src.memory'):
            found += [f'{node.lineno}:import {alias.name}' for alias in node.names
                      if alias.name in READERS and not (node.module == 'src.memory' and alias.name == 'prompt_context')]
    return found


def test_no_module_outside_src_memory_reads_stored_facts():
    offenders = {}
    for path in sorted(SRC.rglob('*.py')):
        if MEMORY in path.parents:
            continue
        uses = reader_uses(path.read_text(encoding='utf-8'), str(path))
        if uses:
            offenders[str(path.relative_to(ROOT))] = uses
    assert offenders == {}, (
        'A module outside src/memory reads stored facts. The NEXO Memoria copy says KZA does not use '
        'memories to answer in this version: update that copy (memory.js) and this pin together. '
        f'Offenders: {offenders}')


def test_the_scan_detects_each_reader_and_ignores_the_helper_module():
    for name in sorted(READERS):
        assert reader_uses(f'async def answer(memory):\n    return await memory.{name}("q")\n')
        assert reader_uses(f'def answer(memory):\n    return getattr(memory, {name!r})("q")\n')
    assert reader_uses('async def answer(memory):\n    return await memory.long_term.acall(memory.long_term.search_relevant, "q")\n')
    assert reader_uses('from src.memory.memory_manager import build_context\n')
    helper = ('from src.memory.prompt_context import SourcesInvalidated, merge_sources\n'
              'import src.memory.prompt_context\n'
              'x = src.memory.prompt_context.SourcesInvalidated\n')
    assert reader_uses(helper) == []
