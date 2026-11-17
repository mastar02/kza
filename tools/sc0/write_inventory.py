"""SC-0 (1): static inventory of SQL write statements on projected tables.

"Projected" is every table whose rows can reach
``scripts.stage_runtime.capture_state_authority`` (R5V-4's scope):

* ``full``: ``_PROTECTED_TABLES`` (every row hashed);
* ``partial``: the tables capture_state_authority and ``_mixed_projections``
  hash in part (facts of owned/confirmed/deleted/quarantined state and their
  sources and dependencies, ``policy_index_deletions`` as
  deleted_fact_revisions, manual voice groups, voice fingerprints (fences),
  suggestions (non-pending rows and fences), event text fences, wake clips
  (fences and manual_clip_closure), stt corrections, conversation feedback,
  utterances (distilled rows and fences), ``sqlite_sequence`` of
  live_deliveries);
* ``schema_marker``: the single-row ``_SCHEMA_MARKERS`` tables hashed by
  ``schema_contract_v1``.

The partial set is extracted from the source of capture_state_authority and
_mixed_projections (``_row_projection(db, "t"``, ``_deadline_projection(db,
"t"``, ``"t" in names``, ``for table in ("t", ...)``, ``{"t", ...} <= names``)
so a new projected table cannot be missed silently.

Pass 1 walks every module under src/ with the AST and records each call to
``execute``/``executemany``/``executescript`` whose SQL text (constants,
f-strings and ``+`` concatenations, with module-level string constants
resolved) holds an INSERT, UPDATE, DELETE, REPLACE or ALTER TABLE (an
upgrade migration changes schema_contract_v1 and every row hash of the table)
whose target is a projected table, or whose target cannot be resolved
statically (``{?}``).
Pass 2 finds the remaining literal write statements on projected tables that
sit outside any scanned call (DDL constants run through a helper, such as the
schema-marker INSERTs and the ALTER TABLE migrations of ambient/store.py's
``_MIGRATIONS``); they are listed with ``call: "ddl_constant"``.
Output: JSON list of {file, line, function, call, verb, table, projection,
dynamic, sql}; ``sql`` is the statement's own text (whitespace-normalized, from
its verb to the next ';'), which classify.py uses only to tell apart two
statements of one function with the same verb and table (SCR-P-1: rules are
keyed by function, verb and table, never by line). Reachability is
classified separately (tools/sc0/classify.py).

Usage: python tools/sc0/write_inventory.py [--repo PATH] > inventory.json
"""
from __future__ import annotations

import argparse
import ast
import inspect
import json
import re
import sys
from pathlib import Path

WRITE = re.compile(
    r"\b(INSERT(?:\s+OR\s+\w+)?\s+INTO|REPLACE\s+INTO|UPDATE|DELETE\s+FROM|ALTER\s+TABLE)\s+[\"'`]?([A-Za-z_{?}][\w{?}]*)",
    re.IGNORECASE)
CALLS = {"execute", "executemany", "executescript"}
_SOURCE_PATTERNS = (
    re.compile(r"_row_projection\(\s*db,\s*\"(\w+)\""),
    re.compile(r"_deadline_projection\(\s*db,\s*\"(\w+)\""),
    re.compile(r"\"(\w+)\" in names"),
)
_TUPLE_PATTERNS = (
    re.compile(r"for table in \(([^)]*)\)"),
    re.compile(r"\{([^{}]*)\} <= names"),
    re.compile(r"names & \{([^{}]*)\}"),
)


def projected_tables(repo: Path) -> dict[str, str]:
    """Map every table capture_state_authority can hash to full/partial/schema_marker."""
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from scripts import stage_runtime  # noqa: PLC0415
    # The capture's body moved into _capture_state_authority in SC-0b (capture_state_authority is
    # its memo wrapper), and its selection into _authority_selection / _authority_table_selection in
    # SC-4b (shared with W1); all are read so every layout yields the same table set.
    source = "".join(inspect.getsource(function) for function in (
        stage_runtime.capture_state_authority, getattr(stage_runtime, "_capture_state_authority", None),
        getattr(stage_runtime, "_authority_selection", None),
        getattr(stage_runtime, "_authority_table_selection", None),
        stage_runtime._mixed_projections) if function is not None)
    partial: set[str] = set()
    for pattern in _SOURCE_PATTERNS:
        partial.update(pattern.findall(source))
    for pattern in _TUPLE_PATTERNS:
        for group in pattern.findall(source):
            partial.update(re.findall(r"\"(\w+)\"", group))
    kinds = {name: "partial" for name in partial}
    kinds.update({name: "schema_marker" for name in stage_runtime._SCHEMA_MARKERS})
    kinds.update({name: "full" for name in stage_runtime._PROTECTED_TABLES})
    return kinds


# capture_state_authority's derived projection names -> the source tables they read.
PROJECTION_SOURCES = {
    "deleted_fact_revisions": ("policy_index_deletions",),
    "wake_deadline_fences": ("wake_clips_v1",),
    "manual_clip_closure": ("wake_clips_v1", "stt_corrections_v1", "conversation_turns_v1"),
    "conversation_feedback": ("conversation_turns_v1",),
    "distillation_receipts": ("utterances",),
    "ambient_deadline_fences": ("utterances",),
    "voice_deadline_fences": ("voice_fingerprints",),
    "manual_group_fingerprints": ("voice_fingerprints", "voice_groups", "voice_group_members"),
    "suggestion_deadline_fences": ("suggestions_policy_v1",),
    "event_text_deadline_fences": ("events_policy_v1",),
    "fact_deadline_fences": ("facts", "fact_sources"),
    "facts": ("facts", "fact_owners", "fact_owner_revisions"),
    "fact_sources": ("fact_sources", "facts", "fact_owners", "fact_owner_revisions"),
    "fact_dependencies": ("fact_dependencies", "facts", "fact_owners", "fact_owner_revisions"),
    "delivery_sequence_reservation": ("sqlite_sequence", "live_deliveries"),
    "schema_contract_v1": ("voice_group_schema", "voice_person_schema", "voice_capture_review_schema",
                           "fact_owner_schema", "voice_attribution_schema", "auth_meta"),
}


def source_tables(projection_name: str) -> tuple[str, ...]:
    """Tables whose writes can change one projection entry."""
    return PROJECTION_SOURCES.get(projection_name, (projection_name,))


def _constants(tree: ast.Module) -> dict:
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    found[target.id] = node.value.value
    return found


def _statement(sql: str, start: int) -> str:
    """The statement that starts at ``start``: whitespace-normalized, up to the next ';' (bounded)."""
    end = sql.find(";", start)
    return re.sub(r"\s+", " ", sql[start:end if end >= 0 else len(sql)]).strip()[:300]


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
    def __init__(self, path, constants, projected, rows, spans):
        self.path, self.constants, self.projected, self.rows = path, constants, projected, rows
        self.spans = spans
        self.stack: list[str] = []

    def _scope(self, node):
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _scope

    def visit_Call(self, node):
        name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
        if name in CALLS and node.args:
            self.spans.append((node.lineno, node.end_lineno))
            sql = _text(node.args[0], self.constants)
            for match in WRITE.finditer(sql):
                verb = re.sub(r"\s+", " ", match.group(1).upper())
                table = match.group(2)
                dynamic = "{?}" in table
                if verb == "UPDATE" and table.upper() in {"SET"}:
                    continue
                if dynamic or table in self.projected:
                    self.rows.append({"file": self.path, "line": node.lineno,
                                      "function": ".".join(self.stack) or "<module>",
                                      "call": name, "verb": verb, "table": table,
                                      "projection": self.projected.get(table, "unresolved"),
                                      "dynamic": dynamic, "sql": _statement(sql, match.start())})
        self.generic_visit(node)


def _enclosing(tree: ast.Module, line: int) -> str:
    """Name of the module-level assignment or function holding a line."""
    for node in tree.body:
        if node.lineno <= line <= (node.end_lineno or node.lineno):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                return "<constant " + ",".join(t.id for t in targets if isinstance(t, ast.Name)) + ">"
            return getattr(node, "name", "<module>")
    return "<module>"


def scan(repo: Path) -> list[dict]:
    projected = projected_tables(repo)
    rows: list[dict] = []
    for path in sorted((repo / "src").rglob("*.py")):
        text = path.read_text()
        tree = ast.parse(text, str(path))
        relative = path.relative_to(repo).as_posix()
        spans: list[tuple[int, int]] = []
        Visitor(relative, _constants(tree), projected, rows, spans).visit(tree)
        for number, line in enumerate(text.splitlines(), 1):
            if any(start <= number <= end for start, end in spans):
                continue
            for match in WRITE.finditer(line):
                table = match.group(2)
                if table in projected and match.group(1).split()[0].isupper():
                    rows.append({"file": relative, "line": number, "function": _enclosing(tree, number),
                                 "call": "ddl_constant",
                                 "verb": re.sub(r"\s+", " ", match.group(1).upper()),
                                 "table": table, "projection": projected[table], "dynamic": False,
                                 "sql": _statement(line, match.start())})
    return sorted(rows, key=lambda row: (row["file"], row["line"], row["table"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--tables", action="store_true", help="print the projected table set instead")
    args = parser.parse_args()
    if args.tables:
        json.dump(projected_tables(Path(args.repo)), sys.stdout, indent=1, sort_keys=True)
    else:
        json.dump(scan(Path(args.repo)), sys.stdout, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
