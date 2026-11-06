"""Archive bytes must not depend on another interpreter's import/intern state."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import py_compile
import re
import signal
import subprocess
import sys
import time
import warnings

import pytest


INTERN_PATHS = (
    "scripts/train_assistant.py", "src/core/adapter_lifecycle.py",
    "src/home_assistant/control_policy.py", "src/users/speaker_identifier.py",
    "src/wakeword/trainer.py", "tools/roadmap_status.py",
)
WORKER = r'''
import ast, hashlib, importlib.util, json, os, platform, subprocess, sys
from pathlib import Path
control = json.loads(os.environ['KZA_BUILD_REPRO_CONTROL'])
repo, script, revision = control['repo'], control['script'], control['revision']
output, context = control['output'], control['context']
assert sys.version_info[:2] == (3, 13)
assert sys.flags.no_site and sys.flags.safe_path and sys.dont_write_bytecode
assert sys.flags.ignore_environment == 0
assert sys.flags.hash_randomization == int(control['seed'] != '0')
assert os.environ['PYTHONHASHSEED'] == control['seed']
assert sys.executable == control['executable']
assert hashlib.sha256(Path(script).read_bytes()).hexdigest() == control['builder_sha256']
assert not any(n == 'src' or n.startswith('src.') for n in sys.modules)
interned = []
if context == 'intern':
    for name in control['intern_paths']:
        raw = subprocess.check_output(['git', 'show', revision + ':' + name], cwd=repo)
        tree = ast.parse(raw)
        interned.extend(sys.intern(n.value) for n in ast.walk(tree)
                        if isinstance(n, ast.Constant) and type(n.value) is str)
else:
    assert context == 'clean'
spec = importlib.util.spec_from_file_location('controlled_build_runtime', script)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
manifest = builder.build_runtime(Path(repo), Path(output), revision=revision)
assert manifest['git_commit'] == revision
assert not any(n == 'src' or n.startswith('src.') for n in sys.modules)
result = {'python_version': list(sys.version_info[:3]), 'executable': sys.executable,
    'architecture': platform.machine(), 'magic': importlib.util.MAGIC_NUMBER.hex(),
    'seed': os.environ['PYTHONHASHSEED'], 'context': context,
    'interned_string_count': len(interned), 'source_commit': manifest['git_commit'],
    'archive_sha256': hashlib.sha256(Path(output).read_bytes()).hexdigest(),
    'files': manifest['files'], 'no_src_imported': True,
    'flags': {'no_site': sys.flags.no_site, 'safe_path': sys.flags.safe_path,
              'ignore_environment': sys.flags.ignore_environment,
              'hash_randomization': sys.flags.hash_randomization}}
Path(control['metadata']).write_text(json.dumps(result, indent=2) + '\n')
'''


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def _wait(proc, seconds, receipt):
    item = {"budget_s": seconds, "returned": False}
    receipt["waits"].append(item)
    try:
        code = proc.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        item["timed_out"] = True
        raise
    item.update(returned=True, exit_code=code)
    receipt["actual_wait_returned"] = True
    return code


def _contain(proc, receipt):
    receipt["containment_only"] = True
    if proc.poll() is not None:
        _wait(proc, .1, receipt)
        return
    for number, seconds in ((signal.SIGTERM, 2), (signal.SIGKILL, 10)):
        event = {"pgid": proc.pid, "signal": int(number), "sent": False}
        receipt["signals"].append(event)
        try:
            os.killpg(proc.pid, number)
            event["sent"] = True
        except ProcessLookupError:
            event["process_lookup_error"] = True
        try:
            _wait(proc, seconds, receipt)
            return
        except subprocess.TimeoutExpired:
            continue
    raise AssertionError("build_worker_exit_unconfirmed")


def _build_in_worker(repo, script, revision, directory, seed, context, deadline):
    budget = min(45, deadline - time.monotonic() - 15)
    assert budget > 0, "parent deadline exhausted before build worker"
    directory.mkdir(mode=0o700)
    output, metadata = directory / "runtime.tar.gz", directory / "worker-result.json"
    control = {"repo": str(repo), "script": str(script), "revision": revision,
        "output": str(output), "context": context, "seed": seed,
        "executable": os.path.abspath(sys.executable), "builder_sha256": _sha(script),
        "intern_paths": INTERN_PATHS, "metadata": str(metadata)}
    environment = {key: os.environ[key] for key in ("PATH", "TMPDIR", "LANG", "LC_ALL", "LC_CTYPE")
        if key in os.environ}
    environment.update(PYTHONHASHSEED=seed, KZA_BUILD_REPRO_CONTROL=json.dumps(control))
    command = [control["executable"], "-P", "-S", "-B", "-"]
    receipt = {"command": command, "control": control, "environment_names": sorted(environment),
        "budget_s": budget, "waits": [], "signals": [], "actual_wait_returned": False,
        "containment_only": False}
    proc, primary = None, None
    cleanup_errors = []
    with (directory / "stdout.log").open("x") as out, (directory / "stderr.log").open("x") as err:
        try:
            proc = subprocess.Popen(command, cwd=directory, env=environment, stdin=subprocess.PIPE,
                stdout=out, stderr=err, start_new_session=True)
            receipt.update(pid=proc.pid, owned_pgid=proc.pid)
            proc.stdin.write(WORKER.encode())
            proc.stdin.close()
            receipt["exit_code"] = _wait(proc, budget, receipt)
            assert receipt["exit_code"] == 0, f"worker setup/build failed: {directory}"
        except BaseException as error:
            primary = error
        finally:
            if proc is not None:
                if proc.stdin is not None and not proc.stdin.closed:
                    try:
                        proc.stdin.close()
                    except BaseException as error:
                        cleanup_errors.append(error)
                if not receipt["actual_wait_returned"]:
                    try:
                        _contain(proc, receipt)
                    except BaseException as error:
                        cleanup_errors.append(error)
            receipt["error_kind"] = None if primary is None else type(primary).__name__
            receipt["cleanup_error_kinds"] = [type(error).__name__ for error in cleanup_errors]
            try:
                _save(directory / "receipt.json", receipt)
            except BaseException as error:
                cleanup_errors.append(error)
    if primary is not None and cleanup_errors:
        raise BaseExceptionGroup("build worker and cleanup", [primary, *cleanup_errors])
    if primary is not None:
        raise primary
    if cleanup_errors:
        raise BaseExceptionGroup("build worker cleanup", cleanup_errors)
    result = json.loads(metadata.read_text())
    assert result["seed"] == seed and result["context"] == context
    assert result["source_commit"] == revision and result["no_src_imported"] is True
    assert result["python_version"] == list(sys.version_info[:3])
    assert result["executable"] == control["executable"]
    assert result["magic"] == importlib.util.MAGIC_NUMBER.hex()
    assert result["flags"] == {"no_site": 1, "safe_path": 1, "ignore_environment": 0,
        "hash_randomization": int(seed != "0")}
    assert (result["interned_string_count"] > 0) is (context == "intern")
    assert _sha(output) == result["archive_sha256"]
    assert _sha(script) == control["builder_sha256"]
    return output, result


def test_archive_bytes_reproduce_across_hashseeds_and_intern_contexts(tmp_path):
    """Interned strings may change context, never the promised archive bytes."""
    started = time.monotonic()
    deadline = started + 240
    assert sys.implementation.name == "cpython" and sys.version_info[:2] == (3, 13)
    repo = Path(__file__).resolve().parents[3]
    working = tmp_path.resolve()
    assert not working.is_relative_to(repo) and not repo.is_relative_to(working)
    script = repo / "scripts/build_runtime.py"
    before = _sha(script)
    requested = os.environ.get("KZA_BUILD_REPRO_REVISION")
    assert requested is None or re.fullmatch(r"[0-9a-f]{40}", requested)
    revision = subprocess.check_output(["git", "rev-parse", "--verify",
        (requested if requested is not None else "HEAD") + "^{commit}"], cwd=repo, text=True).strip()
    assert re.fullmatch(r"[0-9a-f]{40}", revision)
    results = []
    for seed, context in (("0", "clean"), ("0", "intern"), ("1", "clean"), ("1", "intern")):
        directory = working / ("seed-" + seed + "-" + context)
        output, result = _build_in_worker(repo, script, revision, directory, seed, context, deadline)
        results.append((output, result))
    assert _sha(script) == before
    assert time.monotonic() <= deadline
    reference, first = results[0]
    comparisons = [{"seed": result["seed"], "context": result["context"],
        "archive_sha256": result["archive_sha256"],
        "changed_members": [name for name, digest in result["files"].items()
            if first["files"].get(name) != digest]} for _, result in results]
    _save(working / "comparison.json", {"source_commit": revision, "builder_sha256": before,
        "elapsed_s": time.monotonic() - started, "all_actual_waits_returned_zero": True,
        "worker_source_sha256": hashlib.sha256(WORKER.encode()).hexdigest(),
        "oracle": "Exact raw complete archive byte equality; no normalization", "comparisons": comparisons})
    for output, result in results:
        assert result["architecture"] == first["architecture"]
        assert set(result["files"]) == set(first["files"])
        assert output.read_bytes() == reference.read_bytes(), (
            f"archive bytes depend on seed={result['seed']}/context={result['context']}; "
            f"raw artifacts and comparison retained at {working}")


def _api_builder():
    path = Path(__file__).resolve().parents[3] / "scripts/build_runtime.py"
    spec = importlib.util.spec_from_file_location("public_build_runtime_api", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_repository(tmp_path, source):
    repo = tmp_path / "synthetic-repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src/__init__.py").write_text("")
    (repo / "src/main.py").write_text(source)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "add", "src"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=Synthetic build fixture", "-c",
        "user.email=fixture@example.invalid", "-c", "commit.gpgsign=false", "-c",
        "core.hooksPath=/dev/null", "commit", "-qm", "Synthetic compile input"], cwd=repo, check=True)
    return repo


def _native_compile(repo, tmp_path):
    return py_compile.compile(str(repo / "src/main.py"), cfile=str(tmp_path / "native-control.pyc"),
        dfile="src/main.py", doraise=True, optimize=0,
        invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH)


def _compile_error_details(error):
    value = error.exc_value
    return {"file": error.file, "msg": error.msg, "exc_type_name": error.exc_type_name,
        "native_type": type(value), "args": value.args,
        "native_details": {name: getattr(value, name, None) for name in (
            "msg", "filename", "lineno", "offset", "text", "end_lineno", "end_offset")}}


def _syntax_warnings(records):
    return [(item.category, str(item.message), item.filename, item.lineno)
        for item in records if issubclass(item.category, SyntaxWarning)]


def test_invalid_source_preserves_native_pycompile_error_without_publishing(tmp_path, monkeypatch):
    repo = _source_repository(tmp_path, "def broken(:\n    pass\n")
    archive = tmp_path / "invalid.tar.gz"
    # Relative diagnostic filenames must resolve to the same real source text
    # for both native control and public API, never checkout/src/main.py.
    with monkeypatch.context() as context:
        context.chdir(repo)
        with pytest.raises(py_compile.PyCompileError) as expected:
            _native_compile(repo, tmp_path)
        with pytest.raises(py_compile.PyCompileError) as observed:
            _api_builder().build_runtime(repo, archive)
    assert type(observed.value.exc_value) is SyntaxError
    assert observed.value.file == "src/main.py"
    assert _compile_error_details(observed.value) == _compile_error_details(expected.value)
    assert not archive.exists()


def test_native_syntax_warning_reaches_caller_without_suppression(tmp_path, monkeypatch):
    repo = _source_repository(tmp_path, "value = 1\nif value is 1:\n    pass\n")
    archive = tmp_path / "warning.tar.gz"
    with monkeypatch.context() as context:
        context.chdir(repo)
        with warnings.catch_warnings(record=True) as expected:
            warnings.simplefilter("always", SyntaxWarning)
            _native_compile(repo, tmp_path)
        expected_warnings = _syntax_warnings(expected)
        assert expected_warnings, "native control must actually emit SyntaxWarning"
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always", SyntaxWarning)
            manifest = _api_builder().build_runtime(repo, archive)
    assert _syntax_warnings(observed) == expected_warnings
    assert archive.is_file() and "src/main.pyc" in manifest["files"]


def test_error_filter_preserves_native_syntax_error_without_publishing(tmp_path, monkeypatch):
    repo = _source_repository(tmp_path, "value = 1\nif value is 1:\n    pass\n")
    archive = tmp_path / "fatal-warning.tar.gz"
    with monkeypatch.context() as context:
        context.chdir(repo)
        with warnings.catch_warnings():
            warnings.simplefilter("error", SyntaxWarning)
            with pytest.raises(py_compile.PyCompileError) as expected:
                _native_compile(repo, tmp_path)
            with pytest.raises(py_compile.PyCompileError) as observed:
                _api_builder().build_runtime(repo, archive)
    assert type(expected.value.exc_value) is SyntaxError
    assert type(observed.value.exc_value) is SyntaxError
    assert _compile_error_details(observed.value) == _compile_error_details(expected.value)
    assert not archive.exists()
