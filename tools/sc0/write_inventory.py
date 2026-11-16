"""SC-0 (1): static inventory of SQL write statements on protected tables.

Walks every module under src/ with the AST and records each call to
``execute``/``executemany``/``executescript`` whose SQL text (constants,
f-strings and ``+`` concatenations, with module-level string constants
resolved) holds an INSERT, UPDATE, DELETE or REPLACE whose target is a table
of ``scripts.stage_runtime._PROTECTED_TABLES``, or whose target cannot be
resolved statically (``{?}``). Output: JSON list of
{file, line, function, verb, table, dynamic}. Reachability is classified
separately (tools/sc0/classify.py) from this list and the dynamic trace.

Usage: python tools/sc0/write_inventory.py [--repo PATH] > inventory.json
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

WRITE = re.compile(
    r"\b(INSERT(?:\s+OR\s+\w+)?\s+INTO|REPLACE\s+INTO|UPDATE|DELETE\s+FROM)\s+[\"'`]?([A-Za-z_{?}][\w{?}]*)",
    re.IGNORECASE)
CALLS = {"execute", "executemany", "executescript"}


def _protected(repo: Path) -> frozenset:
    sys.path.insert(0, str(repo))
    from scripts.stage_runtime import _PROTECTED_TABLES  # noqa: PLC0415
    return _PROTECTED_TABLES


def _constants(tree: ast.Module) -> dict:
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    found[target.id] = node.value.value
    return found


def _text(node, constants) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(_text(part, constants) if isinstance(part, ast.Constant) else "{?}"
                       for part in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mod)):
        return _text(node.left, constants) + _text(node.right, constants)
    if isinstance(node, ast.Name) and node.id in constants:
        return constants[node.id]
    return "{?}"


class Visitor(ast.NodeVisitor):
    def __init__(self, path, constants, protected, rows):
        self.path, self.constants, self.protected, self.rows = path, constants, protected, rows
        self.stack: list[str] = []

    def _scope(self, node):
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _scope

    def visit_Call(self, node):
        name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
        if name in CALLS and node.args:
            sql = _text(node.args[0], self.constants)
            for match in WRITE.finditer(sql):
                verb = re.sub(r"\s+", " ", match.group(1).upper())
                table = match.group(2)
                dynamic = "{?}" in table
                if verb == "UPDATE" and table.upper() in {"SET"}:
                    continue
                if dynamic or table in self.protected:
                    self.rows.append({"file": self.path, "line": node.lineno,
                                      "function": ".".join(self.stack) or "<module>",
                                      "call": name, "verb": verb, "table": table,
                                      "dynamic": dynamic})
        self.generic_visit(node)


def scan(repo: Path) -> list[dict]:
    protected = _protected(repo)
    rows: list[dict] = []
    for path in sorted((repo / "src").rglob("*.py")):
        tree = ast.parse(path.read_text(), str(path))
        Visitor(path.relative_to(repo).as_posix(), _constants(tree), protected, rows).visit(tree)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    rows = scan(Path(parser.parse_args().repo))
    json.dump(rows, sys.stdout, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
