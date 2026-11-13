"""G1: unmanaged R1 entries refuse under a managed root (U3b-0e 2b, F1/N8).

A managed root holds admission/controller.json. The fixture copies the code
into <root>/slots/s/runtime (as prepare_slot does), points <root>/current at
it and runs entries the way the legacy unit and its timers would: no ticket,
PYTHONPATH=<root>/current, WorkingDirectory = the slot state.
"""

import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from src.core import entry_guard
from src.core.entry_guard import (
    CAPABILITY_CODE, SCHEMA_ONLY_ENV, TICKET_ENV, UNMANAGED_CODE, entry_decision, managed_root_of,
    schema_only_command,
)

REPOSITORY = Path(__file__).resolve().parents[3]
TIMEOUT = 120


def _copy_code(runtime: Path) -> None:
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for name in ("src", "scripts", "tools"):
        shutil.copytree(REPOSITORY / name, runtime / name, ignore=ignore)


@pytest.fixture(scope="module")
def code(tmp_path_factory):
    runtime = tmp_path_factory.mktemp("code") / "runtime"
    _copy_code(runtime)
    return runtime


def _layout(base: Path, code: Path, *, controller: bool = True) -> dict:
    root = base / "releases"
    runtime = root / "slots" / "s" / "runtime"
    state = root / "slots" / "s" / "state"
    shutil.copytree(code, runtime, symlinks=True)
    (state / "config").mkdir(parents=True)
    (state / "data").mkdir()
    (state / "config" / "settings.yaml").write_text("memory:\n  privacy_db_path: ./data/memory_privacy.db\n")
    (state / "data" / "notes.json").write_text("{}")
    for name in ("config", "data"):
        (runtime / name).symlink_to(Path("..") / "state" / name)
    (root / "current").symlink_to(Path("slots") / "s" / "runtime")
    if controller:
        (root / "admission").mkdir()
        (root / "admission" / "controller.json").write_text(json.dumps({"controller_epoch": 1}))
    return {"root": root, "runtime": runtime, "state": state, "current": root / "current"}


def _tree(path: Path) -> dict:
    return {str(item.relative_to(path)): (item.lstat().st_size, item.lstat().st_mtime_ns)
            for item in sorted(path.rglob("*"))}


def _environment(**extra) -> dict:
    environment = {key: value for key, value in os.environ.items()
                   if key not in {TICKET_ENV, SCHEMA_ONLY_ENV, "PYTHONPATH", "KZA_AUDIT_DB"}
                   and not key.startswith("COV_")}
    environment.update(extra)
    return environment


def _run(command, *, cwd, env) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, timeout=TIMEOUT,
                          stdin=subprocess.DEVNULL)


@pytest.fixture
def managed(tmp_path, code):
    return _layout(tmp_path, code)


# The entries a unit, a timer or an operator can start from 'current'.
ENTRIES = [
    ["-B", "-P", "-m", "src.main"],                 # kza-voice (legacy ExecStart)
    ["-B", "-m", "src.health.health_check"],
    ["-B", "-m", "src.training.nightly_trainer"],
    ["-B", "-m", "src.code_index"],                  # kza-code-index if retargeted
    ["-B", "-m", "src.rooms.room_context", "--detect"],
    ["-B", "{current}/tools/audio_watchdog_alert.py", "--once", "--health-path", "{state}/data/h.json"],
    ["-B", "{current}/scripts/sync_ha_to_chroma.py"],
    ["-B", "{current}/scripts/train_custom_wake.py", "generate"],
    ["-B", "-c", "import src"],
]


@pytest.mark.parametrize("argv", ENTRIES, ids=lambda argv: " ".join(argv[:4]))
def test_unmanaged_entries_refuse_before_any_store(managed, argv):
    before = _tree(managed["root"])
    command = [sys.executable, *[part.format(current=managed["current"], state=managed["state"]) for part in argv]]
    result = _run(command, cwd=managed["state"], env=_environment(
        PYTHONPATH=str(managed["current"]), HOME_ASSISTANT_TOKEN="t", HOME_ASSISTANT_URL="http://ha.invalid"))
    assert result.returncode == 78, result.stderr[-2000:]
    assert UNMANAGED_CODE in result.stderr
    assert _tree(managed["root"]) == before


def test_refused_from_the_resolved_slot_even_with_cwd_elsewhere(managed, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    result = _run([sys.executable, "-B", "-c", "import src"], cwd=elsewhere,
                  env=_environment(PYTHONPATH=str(managed["runtime"])))
    assert result.returncode == 78 and UNMANAGED_CODE in result.stderr
    assert list(elsewhere.iterdir()) == []


def test_without_a_controller_record_nothing_is_refused(tmp_path, code):
    layout = _layout(tmp_path, code, controller=False)
    result = _run([sys.executable, "-B", "-c", "import src"], cwd=layout["state"],
                  env=_environment(PYTHONPATH=str(layout["current"])))
    assert result.returncode == 0, result.stderr[-2000:]


def test_a_ticket_defers_to_the_runtime_admission(managed):
    result = _run([sys.executable, "-B", "-c", "import src"], cwd=managed["runtime"],
                  env=_environment(PYTHONPATH=str(managed["current"]), **{TICKET_ENV: "{}"}))
    assert result.returncode == 0, result.stderr[-2000:]


# --- the adoption's schema-only capability -------------------------------------------------------

def test_schema_only_child_is_not_refused_and_opens_no_serving_store(managed, tmp_path):
    copy = tmp_path / "adoption-copy"
    shutil.copytree(managed["state"], copy)
    before, copy_before = _tree(managed["root"]), _tree(copy)
    result = _run(schema_only_command(sys.executable, managed["current"], "--state", str(copy)),
                  cwd=managed["root"], env=_environment(**{SCHEMA_ONLY_ENV: "1"}))
    assert result.returncode == 0, result.stderr[-2000:]
    assert json.loads(result.stdout) == {"schema_only": True, "initializers": 0}
    assert _tree(managed["root"]) == before
    assert _tree(copy) == copy_before


@pytest.mark.parametrize("variant", ["other_entry", "serve", "ticket", "value", "flags", "no_runtime_arg"])
def test_the_capability_is_accepted_only_by_its_bootstrap(managed, variant):
    environment = _environment(**{SCHEMA_ONLY_ENV: "1"})
    command = schema_only_command(sys.executable, managed["current"], "--state", str(managed["state"]))
    if variant == "other_entry":
        command = [sys.executable, "-I", "-B", "-c",
                   "import sys; sys.path.insert(0, sys.argv[1]); import src", str(managed["current"])]
    elif variant == "serve":
        command = [sys.executable, "-B", "-P", "-m", "src.main"]
        environment["PYTHONPATH"] = str(managed["current"])
    elif variant == "ticket":
        environment[TICKET_ENV] = "{}"
    elif variant == "value":
        environment[SCHEMA_ONLY_ENV] = "yes"
    elif variant == "flags":
        command = [command[0], "-B", *command[3:]]
    elif variant == "no_runtime_arg":
        command = command[:5]
        environment["PYTHONPATH"] = str(managed["current"])
    before = _tree(managed["root"])
    result = _run(command, cwd=managed["state"], env=environment)
    assert result.returncode != 0
    if variant != "no_runtime_arg":  # that one fails closed before importing src
        assert result.returncode == 78 and CAPABILITY_CODE in result.stderr
    assert _tree(managed["root"]) == before


def test_the_capability_outside_a_managed_root_still_needs_its_bootstrap(tmp_path, code):
    layout = _layout(tmp_path, code, controller=False)
    result = _run([sys.executable, "-B", "-c", "import src"], cwd=layout["state"],
                  env=_environment(PYTHONPATH=str(layout["current"]), **{SCHEMA_ONLY_ENV: "1"}))
    assert result.returncode == 78 and CAPABILITY_CODE in result.stderr


# --- the serve entry (runtime_admission) ------------------------------------------------------------

def test_serve_entry_refuses_a_ticketless_start_under_a_managed_root(managed, monkeypatch):
    from src.core.runtime_admission import verify_runtime_admission
    monkeypatch.delenv(TICKET_ENV, raising=False)
    monkeypatch.delenv(SCHEMA_ONLY_ENV, raising=False)
    monkeypatch.chdir(managed["state"])
    before = _tree(managed["root"])
    config = {"memory": {"privacy_db_path": str(managed["state"] / "data" / "memory_privacy.db")}}
    with pytest.raises(RuntimeError, match=UNMANAGED_CODE):
        verify_runtime_admission(config)
    monkeypatch.chdir(REPOSITORY)
    with pytest.raises(RuntimeError, match=UNMANAGED_CODE):
        verify_runtime_admission(config)  # the state it would open is under the root
    with pytest.raises(RuntimeError, match=UNMANAGED_CODE):
        verify_runtime_admission({}, runtime_root=managed["runtime"])
    assert _tree(managed["root"]) == before
    assert verify_runtime_admission({"memory": {"privacy_db_path": ":memory:"}}) is None


def test_serve_entry_never_accepts_the_capability(monkeypatch):
    from src.core.runtime_admission import verify_runtime_admission
    monkeypatch.setenv(SCHEMA_ONLY_ENV, "1")
    with pytest.raises(RuntimeError, match=CAPABILITY_CODE):
        verify_runtime_admission({"memory": {"privacy_db_path": ":memory:"}})


def test_serve_entry_refuses_the_audit_env_under_a_ticket(monkeypatch, tmp_path):
    from src.core.runtime_admission import verify_runtime_admission
    monkeypatch.setenv(TICKET_ENV, "{}")
    monkeypatch.setenv("KZA_AUDIT_DB", str(tmp_path / "audit.db"))
    with pytest.raises(RuntimeError, match="runtime_admission_audit_env_refused"):
        verify_runtime_admission({"memory": {"privacy_db_path": str(tmp_path / "p.db")}})
    assert list(tmp_path.iterdir()) == []


# --- decisions --------------------------------------------------------------------------------------

def test_ticket_literal_matches_runtime_admission():
    from src.core import runtime_admission
    assert entry_guard.TICKET_ENV == runtime_admission.TICKET_ENV


def test_managed_root_of_lexical_resolved_and_damaged(tmp_path):
    root = tmp_path / "releases"
    (root / "slots" / "s" / "runtime").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "link").symlink_to(root / "slots" / "s" / "runtime")
    assert managed_root_of([root / "slots" / "s" / "runtime"]) is None
    (root / "admission").mkdir()
    (root / "admission" / "controller.json").symlink_to(tmp_path / "missing")  # damaged still counts
    assert managed_root_of([root / "slots" / "s" / "runtime" / "src"]) == root
    assert managed_root_of([outside / "link" / "src"]) == root
    assert managed_root_of([outside]) is None


def test_refused_from_a_managed_cwd_even_with_code_elsewhere(managed, code, tmp_path):
    """The legacy unit's WorkingDirectory is the container under the root (F1)."""
    elsewhere = tmp_path / "elsewhere-code"
    shutil.copytree(code, elsewhere, symlinks=True)
    before = _tree(managed["root"])
    result = _run([sys.executable, "-B", "-c", "import src"], cwd=managed["state"],
                  env=_environment(PYTHONPATH=str(elsewhere)))
    assert result.returncode == 78 and UNMANAGED_CODE in result.stderr
    assert _tree(managed["root"]) == before


def test_serve_entry_refuses_from_a_managed_cwd(managed, monkeypatch):
    from src.core.runtime_admission import verify_runtime_admission
    monkeypatch.delenv(TICKET_ENV, raising=False)
    monkeypatch.chdir(managed["state"])
    with pytest.raises(RuntimeError, match=UNMANAGED_CODE):
        verify_runtime_admission({"memory": {"privacy_db_path": ":memory:"}})


def test_schema_only_capability_decision(tmp_path):
    root = tmp_path / "r"
    (root / "admission").mkdir(parents=True)
    (root / "admission" / "controller.json").write_text("{}")
    exact = schema_only_command("/py", root, "--state", "x")
    isolated = type("Flags", (), {"isolated": 1, "dont_write_bytecode": True})()
    plain = type("Flags", (), {"isolated": 0, "dont_write_bytecode": True})()
    capability = {SCHEMA_ONLY_ENV: "1"}
    assert entry_decision(environ=capability, paths=[root], argv=exact, flags=isolated) is None
    assert entry_decision(environ={**capability, TICKET_ENV: "{}"}, paths=[root], argv=exact,
                          flags=isolated) == CAPABILITY_CODE
    assert entry_decision(environ=capability, paths=[root], argv=exact, flags=plain) == CAPABILITY_CODE
    assert entry_decision(environ={SCHEMA_ONLY_ENV: "2"}, paths=[root], argv=exact, flags=isolated) == CAPABILITY_CODE
    assert entry_decision(environ=capability, paths=[root], argv=[*exact[:4], exact[4] + " ", *exact[5:]],
                          flags=isolated) == CAPABILITY_CODE
    assert entry_decision(environ=capability, paths=[root], argv=exact[:5], flags=isolated) == CAPABILITY_CODE
    assert entry_decision(environ=capability, paths=[root], argv=None, flags=isolated) == CAPABILITY_CODE


def test_entry_decision_table(tmp_path):
    root = tmp_path / "r"
    (root / "admission").mkdir(parents=True)
    (root / "admission" / "controller.json").write_text("{}")
    inside = [root / "slots"]
    assert entry_decision(environ={}, paths=inside) == UNMANAGED_CODE
    assert entry_decision(environ={}, paths=[tmp_path / "other"]) is None
    assert entry_decision(environ={TICKET_ENV: "{}"}, paths=inside) is None
    assert entry_decision(environ={SCHEMA_ONLY_ENV: "1"}, paths=inside, argv=["py", "-c", "x"]) == CAPABILITY_CODE
    assert entry_decision(environ={SCHEMA_ONLY_ENV: "1"}, paths=[], argv=None) == CAPABILITY_CODE


# --- coverage of every entry point ------------------------------------------------------------------

# Entries that never import src, with the reason G1 need not refuse them.
NO_SRC_ENTRIES = {
    "scripts/stage_runtime.py": "the controller itself (ticketless by design; never imports src)",
    "scripts/build_runtime.py": "builds artifacts from a Git checkout; opens no KZA store",
    "scripts/inventory_home.py": "reads Home Assistant into an explicit private output; opens no KZA store",
    "scripts/validate_hardware.py": "hardware probe; opens no KZA store",
    "tools/code_search.py": "HTTP client of the code-index service; opens no KZA store",
    "tools/make_earcon.py": "regenerates a committed WAV; opens no KZA store",
    "tools/publish_voice_order.py": "legacy publisher; refuses once managed.conf exists (U3b-0d P7)",
    "tools/publish_voice_loading.py": "legacy publisher through publish_voice_order (same refusal)",
    "tools/diagnose_room_microphones.py": "read-only USB/ALSA and log diagnosis",
    "tools/analyze_dashboard_audio.py": "reads through the authenticated dashboard API only",
    "tools/analyze_audio_quality.py": "reads an explicit WAV inventory; writes only its output",
    "tools/runtime_diagnostics.py": "fixed read-only snapshot for a restricted SSH key",
    "tools/roadmap_status.py": "read-only roadmap checks",
    "tools/diagnose_voice_loading.py": "HTTPS timing probe; opens no KZA store",
    "tools/bench_llm.py": "LLM HTTP benchmark; opens no KZA store",
}


def _entry_files():
    for top in ("src", "scripts", "tools"):
        for path in sorted((REPOSITORY / top).rglob("*.py")):
            source = path.read_text()
            if "__main__" in source:
                yield path.relative_to(REPOSITORY).as_posix(), ast.parse(source)


def _is_src_import(node) -> bool:
    if isinstance(node, ast.Import):
        return any(alias.name == "src" or alias.name.startswith("src.") for alias in node.names)
    return (isinstance(node, ast.ImportFrom) and node.level == 0 and node.module is not None
            and (node.module == "src" or node.module.startswith("src.")))


def _path_setup(node) -> bool:
    return "sys.path" in ast.unparse(node) and isinstance(node, (ast.If, ast.Expr))


def _guarded(tree) -> bool:
    """src is imported at module level, or first thing in main() (after sys.path setup)."""
    if any(_is_src_import(node) for node in tree.body):
        return True
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            for statement in node.body:
                if _is_src_import(statement):
                    return True
                docstring = isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant)
                if not (docstring or _path_setup(statement)):
                    return False
    return False


def test_every_entry_refuses_before_any_store_or_is_listed_with_a_reason():
    """N8: the guard runs at 'import src'; scripts/tools must reach it first."""
    unguarded, stale = [], set(NO_SRC_ENTRIES)
    for relative, tree in _entry_files():
        if relative.startswith("src/"):
            continue  # inside the package: the guard runs at its own import
        if relative in NO_SRC_ENTRIES:
            stale.discard(relative)
            assert not any(_is_src_import(node) for node in ast.walk(tree)), relative
        elif not _guarded(tree):
            unguarded.append(relative)
    assert unguarded == []
    assert stale == set()
