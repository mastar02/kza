"""AST lint: no cwd-relative models/logs default in src/ (U3b-0e 2b, review N10, review F3).

A managed child runs with cwd = the read-only slot runtime, so a relative
models/logs default would write into the runtime or fail. Any string literal
that IS such a path ('models', 'logs', './models', '../logs/x', 'models/w',
...), and any f-string that starts with one, is flagged wherever it appears
(a default, a call argument, a Path()/join()/% operand, an alias call). Dict
keys, subscripts and docstrings are names, not paths. Such values must go
through src.core.paths, or be allow-listed here with a reason.
"""

import ast
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[3]
PATH_LITERAL = re.compile(r"(\.{1,2}/)*(models|logs)(/.*)?", re.S)
PATH_HEAD = re.compile(r"(\.{1,2}/)*(models|logs)(/|$)")

# (relative file, flagged text, stripped source line) -> reason. Every entry must still be found.
ALLOW_LIST = {
    ("src/core/paths.py", "./models", 'DEVELOPMENT_MODELS_ROOT = "./models"'):
        "the development fallback itself (refused under a ticket)",
    ("src/llm/http_generation.py", "models",
     'self._models_request = self._client.build_request("GET", self.base_url+"models", headers=self._headers)'):
        "an HTTP route of the model gateway (GET /v1/models), not a file path",
    ("src/memory/runtime_closure.py", "models",
     '"pipeline", "privacy", "ledger", "models", "stt_executors", "tts_models",'):
        "a graph attribute name in the closure inventory, not a file path",
    ("src/pipeline/voice_pipeline.py", "models", 'self.lifecycle.stop.checkpoint("models")'):
        "a stop-checkpoint label, not a file path",
}


def violations(source: str, filename: str = "<memory>") -> list[tuple[int, str]]:
    """Return (line, text) for each relative models/logs literal or f-string head."""
    found = []
    names = set()
    tree = ast.parse(source, filename)
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            names.update(id(key) for key in node.keys if key is not None)
        elif isinstance(node, ast.Subscript):
            names.add(id(node.slice))
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            names.add(id(node.value))  # docstrings
        elif isinstance(node, ast.JoinedStr):
            names.update(id(part) for part in node.values)
            head = node.values[0] if node.values else None
            if isinstance(head, ast.Constant) and isinstance(head.value, str) and PATH_HEAD.match(head.value):
                found.append((node.lineno, ast.unparse(node)))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in names
                and PATH_LITERAL.fullmatch(node.value)):
            found.append((node.lineno, node.value))
    return sorted(set(found))


def _scan() -> dict[tuple[str, str, str], int]:
    result = {}
    for path in sorted((ROOT / "src").rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix()
        source = path.read_text()
        lines = source.splitlines()
        for line, text in violations(source, relative):
            result[(relative, text, lines[line - 1].strip())] = line
    return result


def test_src_has_no_unlisted_models_or_logs_default():
    found = _scan()
    unlisted = {key: line for key, line in found.items() if key not in ALLOW_LIST}
    assert unlisted == {}, "route through src.core.paths or allow-list with a reason"
    assert set(ALLOW_LIST) <= set(found), "stale allow-list entry"
    assert all(reason.strip() for reason in ALLOW_LIST.values())


@pytest.mark.parametrize("snippet", [
    'X = "./models/whisper"',
    'X = "models/wakeword/nexa.ppn"',
    'X = "./logs/a.log"',
    'X = "logs/a.log"',
    'X = "./models"',
    'X = f"./models/speaker/{name}"',
    'X = f"models/{name}.onnx"',
    'X = f"logs/{name}"',
    'X = Path("models")',
    'X = Path("models", "x")',
    'X = pathlib.Path("logs") / "x"',
    'X = os.path.join("models", name)',
    'X = os.path.join("logs", "a.log")',
    'X = join("./models", name)',
    'X = PurePosixPath("models/") / name',
    'def f(output_dir: str = "./models/lora_adapters"):\n    pass',
    'X = config.get("model", "./models/whisper-v3-turbo")',
    # independent review F3: the nine evasions
    'def f(models_dir="models"):\n    pass',
    'X = Path(cfg.get("dir", "models")) / "w"',
    'X = os.path.join(".", "models", n)',
    'X = Path(".") / "models"',
    'X = Path.cwd() / "models"',
    'from pathlib import Path as P\nX = P("models")',
    'X = "%s/w" % "models"',
    'X = "../models/w"',
    'X = "logs"',
])
def test_each_form_is_flagged(snippet):
    assert violations(snippet)


@pytest.mark.parametrize("snippet", [
    'X = "./data/x.db"',
    'X = "my_models/x"',
    'X = "hf/models/x"',
    'X = Path(root, "modelsx")',
    'X = os.path.join(root, "logsdir")',
    'X = f"{root}/models"',
    'X = "speechbrain/spkrec-ecapa-voxceleb"',
    'X = status["models"]',
    'X = {"models": []}',
    'def f():\n    """models"""',
])
def test_unrelated_values_pass(snippet):
    assert violations(snippet) == []


@pytest.mark.parametrize("snippet", [
    'DEFAULT = "./models/wakeword"',
    'X = f"./models/piper/{voice}.onnx"',
    'X = Path("models") / "speaker"',
    'X = os.path.join("logs", "voice.log")',
])
def test_a_mutant_adding_a_default_to_src_fails_the_lint(snippet):
    """Acceptance: a mutant adding one of the forms to real source is caught."""
    path = ROOT / "src" / "pipeline" / "model_manager.py"
    source = path.read_text()
    assert violations(source, "model_manager.py") == []
    assert violations(source + "\n" + snippet + "\n", "model_manager.py")
