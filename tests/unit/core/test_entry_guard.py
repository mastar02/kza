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
    CAPABILITY_CODE, SCHEMA_ONLY_ENV, TICKET_CODE, TICKET_ENV, UNMANAGED_CODE, entry_decision,
    managed_root_of, schema_only_command,
)
from src.core.schema_only import BINDING_MARKER, COPY_MARKER

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


def _ticket(root: Path, **changes) -> str:
    """A ticket of the controller's shape whose binding is recorded under ``root``."""
    binding = "b" * 64
    bindings = root / "admission" / "bindings"
    bindings.mkdir(parents=True, exist_ok=True)
    (bindings / (binding + ".json")).write_text("{}")
    ticket = {"schema_version": 1, "admission_root": str(root), "binding_id": binding, "run_id": "c" * 32,
              "generation": 1, "controller_pid": 1, "lease_fd": 3,
              "lease_identity": {"device": 1, "inode": 2}}
    ticket.update(changes)
    return json.dumps(ticket)


def _marked_copy(source: Path, destination: Path) -> Path:
    shutil.copytree(source, destination)
    (destination / COPY_MARKER).write_text(json.dumps({"schema_version": 1, "plan_sha256": "d" * 64}))
    return destination


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
    ticket = _ticket(managed["root"])
    result = _run([sys.executable, "-B", "-c", "import src"], cwd=managed["runtime"],
                  env=_environment(PYTHONPATH=str(managed["current"]), **{TICKET_ENV: ticket}))
    assert result.returncode == 0, result.stderr[-2000:]


@pytest.mark.parametrize("ticket", ["{}", "bogus", "", "STALE_OTHER_ROOT", "UNRECORDED_BINDING", "BAD_RUN_ID"])
def test_a_ticket_of_the_wrong_shape_refuses_under_a_managed_root(managed, tmp_path, ticket):
    """Review F4: G1 checks the ticket's shape, root and recorded binding, not only its presence."""
    if ticket == "STALE_OTHER_ROOT":
        ticket = _ticket(tmp_path / "other-root")
    elif ticket == "UNRECORDED_BINDING":
        ticket = json.dumps(dict(json.loads(_ticket(managed["root"])), binding_id="e" * 64))
    elif ticket == "BAD_RUN_ID":
        ticket = _ticket(managed["root"], run_id="X" * 32)
    before = _tree(managed["root"])
    result = _run([sys.executable, "-B", "-c", "import src"], cwd=managed["state"],
                  env=_environment(PYTHONPATH=str(managed["current"]), **{TICKET_ENV: ticket}))
    assert result.returncode == 78 and TICKET_CODE in result.stderr
    assert _tree(managed["root"]) == before


# --- the adoption's schema-only capability -------------------------------------------------------

def test_schema_only_child_is_not_refused_and_opens_no_serving_store(managed, tmp_path):
    copy = _marked_copy(managed["state"], tmp_path / "adoption-copy")
    before, copy_before = _tree(managed["root"]), _tree(copy)
    result = _run(schema_only_command(sys.executable, managed["current"], "--state", str(copy)),
                  cwd=managed["root"], env=_environment(**{SCHEMA_ONLY_ENV: "1"}))
    assert result.returncode == 0, result.stderr[-2000:]
    assert json.loads(result.stdout) == {"schema_only": True, "initializers": 0}
    assert _tree(managed["root"]) == before
    assert _tree(copy) == copy_before


@pytest.mark.parametrize("variant", ["other_entry", "serve", "ticket", "value", "flags", "x_dev", "no_runtime_arg"])
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
        environment[TICKET_ENV] = _ticket(managed["root"])
    elif variant == "value":
        environment[SCHEMA_ONLY_ENV] = "yes"
    elif variant == "flags":
        command = [command[0], "-B", *command[3:]]
    elif variant == "x_dev":  # review RM3: an extra option between the exact flags
        command = [command[0], "-X", "dev", *command[1:]]
    elif variant == "no_runtime_arg":
        command = command[:5]
        environment["PYTHONPATH"] = str(managed["current"])
    before = _tree(managed["root"])
    result = _run(command, cwd=managed["state"], env=environment)
    assert result.returncode != 0
    if variant != "no_runtime_arg":  # that one fails closed before importing src
        assert result.returncode == 78 and CAPABILITY_CODE in result.stderr
    assert _tree(managed["root"]) == before


@pytest.mark.parametrize("target", ["slot_state", "aliased_state", "bound", "unmarked", "symlink"])
def test_schema_only_refuses_a_serving_or_unmarked_state(managed, tmp_path, target):
    """Review F1/F8 (RM2): only a marked adoption copy (or the P4 tmpfs) is opened."""
    marker = json.dumps({"schema_version": 1, "plan_sha256": "d" * 64})
    if target == "slot_state":  # even a marker does not make a slot's state a copy
        state, code_name = managed["state"], "schema_only_state_serving"
        (state / COPY_MARKER).write_text(marker)
    elif target == "aliased_state":  # the slot's state reached through an alias of the root
        (tmp_path / "alias").symlink_to(managed["root"], target_is_directory=True)
        state, code_name = tmp_path / "alias" / "slots" / "s" / "state", "schema_only_state_serving"
    elif target == "bound":
        state = _marked_copy(managed["state"], tmp_path / "bound-copy")
        (state / BINDING_MARKER).write_text("{}")
        code_name = "schema_only_state_bound"
    elif target == "unmarked":
        state = tmp_path / "unmarked-copy"
        shutil.copytree(managed["state"], state)
        code_name = "schema_only_state_unmarked"
    else:
        real = _marked_copy(managed["state"], tmp_path / "real-copy")
        state = tmp_path / "link-copy"
        state.symlink_to(real, target_is_directory=True)
        code_name = "schema_only_state_invalid"
    before = _tree(managed["root"])
    result = _run(schema_only_command(sys.executable, managed["current"], "--state", str(state)),
                  cwd=tmp_path, env=_environment(**{SCHEMA_ONLY_ENV: "1"}))
    assert result.returncode == 78 and code_name in result.stderr, result.stderr[-2000:]
    assert _tree(managed["root"]) == before


def test_schema_only_refuses_a_runtime_argument_that_is_not_its_own(monkeypatch, tmp_path, caplog):
    from src.core import schema_only
    monkeypatch.setenv(SCHEMA_ONLY_ENV, "1")
    monkeypatch.delenv(TICKET_ENV, raising=False)
    monkeypatch.setattr(schema_only, "schema_only_invocation", lambda argv: True)
    monkeypatch.setattr(sys, "argv", ["-c", str(tmp_path), "--state", str(tmp_path)])
    assert schema_only.main(["--state", str(tmp_path)]) == 78
    assert "schema_only_runtime_mismatch" in caplog.text
    assert list(tmp_path.iterdir()) == []


def test_schema_only_state_decision_tmpfs_and_marker(tmp_path):
    from src.core import schema_only
    scratch = tmp_path / "run-user" / "adoption"
    scratch.mkdir(parents=True)
    assert schema_only.state_decision(scratch, tmpfs_roots=(str(tmp_path / "run-user"),)) is None
    assert schema_only.state_decision(scratch, tmpfs_roots=(str(tmp_path / "elsewhere"),)) == "schema_only_state_unmarked"
    assert schema_only.state_decision(tmp_path / "run-user", tmpfs_roots=(str(tmp_path / "run-user"),)) == \
        "schema_only_state_unmarked"  # the tmpfs root itself is not a scratch copy
    for bad in ({"schema_version": 1, "plan_sha256": "x"}, {"schema_version": 2, "plan_sha256": "d" * 64}, []):
        (scratch / COPY_MARKER).write_text(json.dumps(bad))
        assert schema_only.state_decision(scratch, tmpfs_roots=()) == "schema_only_state_unmarked"
    (scratch / COPY_MARKER).write_text(json.dumps({"schema_version": 1, "plan_sha256": "d" * 64}))
    assert schema_only.state_decision(scratch, tmpfs_roots=()) is None
    assert schema_only.default_tmpfs_roots() == (f"/run/user/{os.getuid()}",)


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


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a 0000 directory")
def test_an_unreadable_controller_record_still_counts_as_managed(tmp_path):
    """Review F8 (RM1): a PermissionError while looking for the record fails closed."""
    root = tmp_path / "releases"
    (root / "slots" / "s" / "runtime").mkdir(parents=True)
    (root / "admission").mkdir()
    (root / "admission" / "controller.json").write_text("{}")
    (root / "admission").chmod(0)
    try:
        assert managed_root_of([root / "slots" / "s" / "runtime"]) == root
        assert entry_decision(environ={}, paths=[root / "slots"]) == UNMANAGED_CODE
    finally:
        (root / "admission").chmod(0o755)


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
    assert entry_decision(environ={TICKET_ENV: "{}"}, paths=inside) == TICKET_CODE
    assert entry_decision(environ={TICKET_ENV: _ticket(root)}, paths=inside) is None
    assert entry_decision(environ={TICKET_ENV: _ticket(root, generation=True)}, paths=inside) == TICKET_CODE
    assert entry_decision(environ={TICKET_ENV: _ticket(root, extra=1)}, paths=inside) == TICKET_CODE
    assert entry_decision(environ={TICKET_ENV: _ticket(root, lease_fd=-1)}, paths=inside) == TICKET_CODE
    assert entry_decision(environ={TICKET_ENV: _ticket(root, binding_id="../" + "b" * 61)},
                          paths=inside) == TICKET_CODE
    other = tmp_path / "other-root"
    other.mkdir()
    assert entry_decision(environ={TICKET_ENV: _ticket(root, admission_root=str(other))},
                          paths=inside) == TICKET_CODE  # binding recorded here, ticket names another root
    binding = root / "admission" / "bindings" / ("b" * 64 + ".json")
    binding.unlink()
    binding.mkdir()
    assert entry_decision(environ={TICKET_ENV: json.dumps(dict(json.loads(_ticket(other)), admission_root=str(root)))},
                          paths=inside) == TICKET_CODE  # a directory is not a recorded binding
    binding.rmdir()
    binding.symlink_to(other / "admission" / "bindings" / ("b" * 64 + ".json"))
    assert entry_decision(environ={TICKET_ENV: json.dumps(dict(json.loads(_ticket(other)), admission_root=str(root)))},
                          paths=inside) == TICKET_CODE  # nor is a link to one elsewhere
    assert entry_decision(environ={TICKET_ENV: "{}"}, paths=[tmp_path / "other"]) is None
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


def _optional_src_import(node) -> bool:
    """``try: import src`` whose only handler is ModuleNotFoundError re-raised unless name is 'src'."""
    return (isinstance(node, ast.Try) and len(node.body) == 1 and _is_src_import(node.body[0])
            and len(node.handlers) == 1 and ast.unparse(node.handlers[0].type) == "ModuleNotFoundError"
            and "!= 'src'" in ast.unparse(node.handlers[0]) and "raise" in ast.unparse(node.handlers[0]))


def _guarded(tree) -> bool:
    """src is imported at module level, or first thing in main() (after sys.path setup).

    Standalone tools may import it optionally (review F2): the guard still runs
    whenever the package is importable, which is the only case it can protect.
    """
    if any(_is_src_import(node) or _optional_src_import(node) for node in tree.body):
        return True
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            for statement in node.body:
                if _is_src_import(statement) or _optional_src_import(statement):
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


# --- standalone tools (review F2) -------------------------------------------------------------------

@pytest.mark.parametrize("tool, extra, argv", [
    ("tools/xvf_noise_trial.py", ["src/audio/xvf_controller.py"], ["--help"]),
    ("tools/harvest_hallucinations.py", [], ["--file", os.devnull]),
    ("tools/benchmark_ambient.py", [], ["--help"]),
    ("tools/audio_watchdog_alert.py", [], ["--help"]),
    ("scripts/train_custom_wake.py", [], ["--help"]),
])
def test_standalone_tools_still_run_without_src(tmp_path, tool, extra, argv):
    """A tool copied alone (no src package anywhere) keeps working; G1 only guards src."""
    for relative in (tool, *extra):
        shutil.copy2(REPOSITORY / relative, tmp_path / Path(relative).name)
    result = _run([sys.executable, "-B", str(tmp_path / Path(tool).name), *argv], cwd=tmp_path,
                  env=_environment())
    assert result.returncode == 0, result.stderr[-2000:]
    assert "ModuleNotFoundError" not in result.stderr


def test_a_broken_src_is_not_mistaken_for_a_standalone_run(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "__init__.py").write_text("import kza_missing_dependency\n")
    shutil.copy2(REPOSITORY / "tools" / "harvest_hallucinations.py", tmp_path / "harvest.py")
    result = _run([sys.executable, "-B", str(tmp_path / "harvest.py"), "--file", os.devnull], cwd=tmp_path,
                  env=_environment(PYTHONPATH=str(tmp_path)))
    assert result.returncode != 0 and "kza_missing_dependency" in result.stderr
