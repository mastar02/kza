"""AST lint: no cwd-relative models/logs default in src/ (U3b-0e 2b, review N10).

A managed child runs with cwd = the read-only slot runtime, so a literal
'./models', 'models/', './logs' or 'logs/' default (or Path('models'),
os.path.join('logs', ...)) would write into the runtime or fail. Such values
must go through src.core.paths, or be allow-listed here with a reason.
"""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PREFIXES = ("./models", "models/", "./logs", "logs/")
HEADS = {"models", "logs", "./models", "./logs", "models/", "logs/", "./models/", "./logs/"}
PATH_CALLS = {"Path", "PurePath", "PosixPath", "PurePosixPath", "join"}

# (relative file, flagged text) -> reason. Every entry must still be found.
ALLOW_LIST = {
    ("src/core/paths.py", "./models"): "the development fallback itself (refused under a ticket)",
}


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def violations(source: str, filename: str = "<memory>") -> list[tuple[int, str]]:
    """Return (line, text) for each Constant, JoinedStr, Path() or join() default."""
    found = []
    joined_parts = set()
    tree = ast.parse(source, filename)
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            head = node.values[0] if node.values else None
            if isinstance(head, ast.Constant) and isinstance(head.value, str):
                joined_parts.add(id(head))
                if head.value.startswith(PREFIXES):
                    found.append((node.lineno, ast.unparse(node)))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in joined_parts
                and node.value.startswith(PREFIXES)):
            found.append((node.lineno, node.value))
        elif isinstance(node, ast.Call) and _call_name(node) in PATH_CALLS and node.args:
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str) and first.value in HEADS:
                found.append((node.lineno, ast.unparse(node)))
    return sorted(set(found))


def _scan() -> dict[tuple[str, str], int]:
    result = {}
    for path in sorted((ROOT / "src").rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix()
        for line, text in violations(path.read_text(), relative):
            result[(relative, text)] = line
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
])
def test_each_form_is_flagged(snippet):
    assert violations(snippet)


@pytest.mark.parametrize("snippet", [
    'X = "./data/x.db"',
    'X = "my_models/x"',
    'X = "hf/models/x"',
    'X = lifecycle.checkpoint("models")',
    'X = Path(root, "models")',
    'X = os.path.join(root, "logs")',
    'X = f"{root}/models"',
    'X = "speechbrain/spkrec-ecapa-voxceleb"',
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
