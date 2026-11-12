#!/usr/bin/env python3
"""Build a committed KZA revision into a source-free CPython 3.13 archive."""

import argparse
import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import py_compile
import re
import subprocess
import sys
import tarfile
import tempfile
import warnings


ASSET_SUFFIXES = {".html", ".css", ".js", ".jsx", ".json", ".csv", ".svg", ".txt",
                  ".png", ".jpg", ".ico", ".woff", ".woff2", ".ttf", ".wav"}

# U3b-0d (design v3 N7, review 0d3 K2): integers the managed launcher reads
# from release.json. They are copied from the built revision's own
# scripts/stage_runtime.py text (``NAME = <int>`` at column 0), never from this
# builder and never by importing it, so an older revision built today gets no
# key (read as 0) and cannot claim a contract it does not implement.
RELEASE_INTEGERS = {"launcher_contract": "LAUNCHER_CONTRACT", "format_epoch": "FORMAT_EPOCH",
                    "receipt_state": "RECEIPT_STATE_SCHEMA", "controller_epoch": "CONTROLLER_EPOCH"}

_WARNING_TYPES = {"SyntaxWarning": SyntaxWarning, "DeprecationWarning": DeprecationWarning}
_SYNTAX_FIELDS = ("msg", "filename", "lineno", "offset", "text", "end_lineno",
                  "end_offset", "print_file_and_line")

# A fixed bootstrap gets a fresh interpreter and intern table for each build.
_COMPILER_WORKER = r'''
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import py_compile
import sys
import warnings

WARNING_TYPES = {"SyntaxWarning": SyntaxWarning, "DeprecationWarning": DeprecationWarning}
SYNTAX_TYPES = {"SyntaxError": SyntaxError, "IndentationError": IndentationError, "TabError": TabError}
SYNTAX_FIELDS = ("msg", "filename", "lineno", "offset", "text", "end_lineno", "end_offset", "print_file_and_line")

def require(condition, message):
    if not condition:
        raise ValueError(message)

def abi():
    return {"implementation": sys.implementation.name, "version": list(sys.version_info[:3]),
            "cache_tag": sys.implementation.cache_tag, "magic": importlib.util.MAGIC_NUMBER.hex()}

def compile_error(exc):
    value = exc.exc_value
    require(type(value) in SYNTAX_TYPES.values(), "unsupported_compiler_exception")
    args = value.args
    require(len(args) in (1, 2) and type(args[0]) is str, "unsupported_syntax_arguments")
    if len(args) == 2:
        require(type(args[1]) is tuple and len(args[1]) in (4, 6), "unsupported_syntax_details")
    return {"file": exc.file, "msg": exc.msg, "exc_type_name": exc.exc_type_name,
            "args": args, "attributes": {name: getattr(value, name) for name in SYNTAX_FIELDS}}

def warning_record(message, category, filename, lineno, ordinal):
    require(WARNING_TYPES.get(category.__name__) is category, "unsupported_compiler_warning")
    return {"message": str(message), "category": category.__name__, "filename": filename,
            "lineno": lineno, "ordinal": ordinal}

def main():
    request = json.loads(sys.stdin.buffer.read())
    require(type(request) is dict and set(request) == {"schema_version", "abi", "executable", "modules", "diagnostic"}
            and request["schema_version"] == 1, "invalid_compiler_request")
    require(abi() == request["abi"] and sys.implementation.name == "cpython"
            and sys.version_info[:2] == (3, 13), "compiler_abi_mismatch")
    require(sys.executable == request["executable"], "compiler_executable_mismatch")
    require(sys.flags.safe_path and sys.flags.no_site and sys.dont_write_bytecode
            and not sys.flags.ignore_environment and not sys.flags.isolated
            and not sys.flags.optimize and not sys.flags.hash_randomization
            and os.environ.get("PYTHONHASHSEED") == "0", "compiler_startup_mismatch")
    root = Path.cwd().resolve()
    modules = request["modules"]
    require(type(modules) is list and modules, "invalid_compiler_modules")
    names = []
    for module in modules:
        require(type(module) is dict and set(module) == {"path", "sha256"}, "invalid_compiler_module")
        name = module["path"]
        require(type(name) is str, "invalid_compiler_path")
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and ".." not in relative.parts and relative.suffix == ".py"
                and relative.parts[0] in {"src", "tools", "scripts"}, "invalid_compiler_path")
        source = root / name
        require(source.resolve() == source and source.is_file(), "redirected_compiler_source")
        require(hashlib.sha256(source.read_bytes()).hexdigest() == module["sha256"], "compiler_source_changed")
        names.append(name)
    require(names == sorted(set(names)), "unordered_compiler_modules")
    diagnostic = request["diagnostic"]
    selected = names
    matched = False
    if diagnostic is not None:
        require(type(diagnostic) is dict and set(diagnostic) == {"message", "category", "filename", "lineno", "ordinal"}
                and diagnostic["filename"] in names, "invalid_diagnostic_target")
        selected = [diagnostic["filename"]]
    caught_error = None
    records = []
    outputs = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if diagnostic is not None:
            ordinal = 0
            def fail_selected(message, category, filename, lineno, file=None, line=None):
                nonlocal ordinal, matched
                record = warning_record(message, category, filename, lineno, ordinal)
                ordinal += 1
                if record["ordinal"] == diagnostic["ordinal"]:
                    require(record == diagnostic, "diagnostic_warning_changed")
                    matched = True
                    # Native compiler/parser converts this to its own precise
                    # SyntaxError at the active source location.
                    raise category(str(message))
            warnings.showwarning = fail_selected
        try:
            for name in selected:
                source = root / name
                compiled = source.with_suffix(".pyc")
                if diagnostic is not None:
                    compiled = root / ".diagnostic" / PurePosixPath(name).with_suffix(".pyc")
                py_compile.compile(str(source), cfile=str(compiled), dfile=name,
                    doraise=True, optimize=0,
                    invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH)
                raw = compiled.read_bytes()
                outputs.append({"path": str(PurePosixPath(name).with_suffix(".pyc")),
                                "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
        except py_compile.PyCompileError as exc:
            caught_error = compile_error(exc)
        ordinals = {}
        for warning in caught:
            ordinal = ordinals.get(warning.filename, 0)
            records.append(warning_record(warning.message, warning.category,
                                          warning.filename, warning.lineno, ordinal))
            ordinals[warning.filename] = ordinal + 1
    if diagnostic is not None:
        require(matched and caught_error is not None, "diagnostic_warning_not_fatal")
    return {"schema_version": 1, "abi": abi(), "status": "compile_error" if caught_error else "ok",
            "error": caught_error, "warnings": records, "outputs": [] if caught_error else outputs,
            "diagnostic_matched": matched}

try:
    result = main()
except Exception as exc:
    print(json.dumps({"schema_version": 1, "status": "worker_error",
                      "type": type(exc).__name__, "message": str(exc)}, ensure_ascii=True))
    raise SystemExit(3)
print(json.dumps(result, ensure_ascii=True, allow_nan=False))
raise SystemExit(2 if result["error"] else 0)
'''


class _CompilerWorkerError(RuntimeError):
    """Retain raw isolated-compiler diagnostics on protocol failure."""

    def __init__(self, message: str, stdout: bytes, stderr: bytes):
        super().__init__(message)
        self.stdout = stdout
        self.stderr = stderr


def _compiler_abi() -> dict:
    return {"implementation": sys.implementation.name, "version": list(sys.version_info[:3]),
            "cache_tag": sys.implementation.cache_tag, "magic": importlib.util.MAGIC_NUMBER.hex()}


def _run_compiler(staging: Path, modules: list[dict], diagnostic: dict | None = None) -> dict:
    """Wait for a fresh seeded compiler without inheriting caller Python flags."""
    executable = os.path.abspath(sys.executable)
    request = {"schema_version": 1, "abi": _compiler_abi(), "executable": executable,
               "modules": modules, "diagnostic": diagnostic}
    env = {key: os.environ[key] for key in ("PATH", "TMPDIR", "TEMP", "TMP", "SYSTEMROOT", "WINDIR")
           if key in os.environ}
    env.update(PYTHONHASHSEED="0", PYTHONUTF8="1", LC_ALL="C")
    command = [executable, "-P", "-S", "-B", "-c", _COMPILER_WORKER]
    try:
        # Inherit the caller's group so an outer harness owns both processes.
        # run() drains both pipes and kills/reaps the worker on timeout.
        completed = subprocess.run(command, cwd=staging, env=env,
            input=json.dumps(request, ensure_ascii=True).encode(), capture_output=True,
            check=False, timeout=60)
    except subprocess.TimeoutExpired as exc:
        if exc.stderr:
            sys.stderr.write(exc.stderr.decode("utf-8", errors="backslashreplace"))
        raise
    if completed.stderr:
        sys.stderr.write(completed.stderr.decode("utf-8", errors="backslashreplace"))

    def reject(message: str) -> None:
        raise _CompilerWorkerError(message, completed.stdout, completed.stderr)

    try:
        result = json.loads(completed.stdout)
    except (ValueError, UnicodeError):
        reject("Invalid compiler worker response")
    expected = {"schema_version", "abi", "status", "error", "warnings", "outputs", "diagnostic_matched"}
    if (type(result) is not dict or set(result) != expected or result["schema_version"] != 1
            or result["abi"] != request["abi"] or type(result["diagnostic_matched"]) is not bool
            or result["diagnostic_matched"] != (diagnostic is not None)):
        reject("Compiler worker protocol or ABI mismatch")
    if ((result["status"] == "ok" and (completed.returncode != 0 or result["error"] is not None))
            or (result["status"] == "compile_error" and (completed.returncode != 2 or type(result["error"]) is not dict))
            or result["status"] not in {"ok", "compile_error"}):
        reject("Compiler worker did not complete normally")
    if type(result["warnings"]) is not list or type(result["outputs"]) is not list:
        reject("Invalid compiler worker records")
    names = {module["path"] for module in modules}
    ordinals = {}
    for record in result["warnings"]:
        if (type(record) is not dict or set(record) != {"message", "category", "filename", "lineno", "ordinal"}
                or type(record["message"]) is not str or record["category"] not in _WARNING_TYPES
                or record["filename"] not in names or type(record["lineno"]) is not int
                or type(record["ordinal"]) is not int
                or record["ordinal"] != ordinals.get(record["filename"], 0)):
            reject("Invalid compiler warning record")
        ordinals[record["filename"]] = record["ordinal"] + 1
    if result["status"] == "compile_error" and result["outputs"]:
        reject("Failed compiler returned accepted outputs")
    return result


def _compile_error(record: dict, names: set[str]) -> py_compile.PyCompileError:
    """Rebuild only native parser error classes from explicit JSON data."""
    classes = {"SyntaxError": SyntaxError, "IndentationError": IndentationError, "TabError": TabError}
    if (set(record) != {"file", "msg", "exc_type_name", "args", "attributes"}
            or record["file"] not in names or type(record["msg"]) is not str
            or record["exc_type_name"] not in classes or type(record["args"]) is not list
            or len(record["args"]) not in (1, 2) or type(record["args"][0]) is not str
            or type(record["attributes"]) is not dict or set(record["attributes"]) != set(_SYNTAX_FIELDS)):
        raise ValueError("Invalid compiler error record")
    args = record["args"]
    if len(args) == 2:
        details = args[1]
        if type(details) is not list or len(details) not in (4, 6):
            raise ValueError("Invalid compiler error details")
        for index, item in enumerate(details):
            expected = str if index in (0, 3) else int
            if item is not None and type(item) is not expected:
                raise ValueError("Invalid compiler error detail type")
    attributes = record["attributes"]
    for name, item in attributes.items():
        expected = (str,) if name in {"msg", "filename", "text"} else (int,)
        if name == "print_file_and_line":
            expected = (int, bool)
        if item is not None and type(item) not in expected:
            raise ValueError("Invalid compiler error attribute")
    kind = classes[record["exc_type_name"]]
    value = kind(args[0]) if len(args) == 1 else kind(args[0], tuple(args[1]))
    for name in _SYNTAX_FIELDS:
        setattr(value, name, attributes[name])
    return py_compile.PyCompileError(kind, value, record["file"], msg=record["msg"])


def _compile_modules(staging: Path, modules: list[dict]) -> dict[str, bytes]:
    """Compile sources, replay caller warning policy, then accept original bytes."""
    modules.sort(key=lambda module: module["path"])
    result = _run_compiler(staging, modules)
    names = {module["path"] for module in modules}
    for record in result["warnings"]:
        category = _WARNING_TYPES[record["category"]]
        try:
            warnings.warn_explicit(record["message"], category, record["filename"], record["lineno"],
                                   registry=None)
        except Exception as exc:
            if isinstance(exc, category):
                diagnostic = _run_compiler(staging, modules, diagnostic=record)
                if diagnostic["status"] != "compile_error":
                    raise RuntimeError("Compiler warning diagnostic did not fail") from exc
                raise _compile_error(diagnostic["error"], names) from None
            # py_compile also wraps a caller's failing warning display hook.
            raise py_compile.PyCompileError(type(exc), exc, record["filename"]) from exc
    if result["error"] is not None:
        raise _compile_error(result["error"], names)
    expected = {str(PurePosixPath(name).with_suffix(".pyc")) for name in names}
    accepted = {}
    for output in result["outputs"]:
        if (type(output) is not dict or set(output) != {"path", "size", "sha256"}
                or output["path"] not in expected or output["path"] in accepted
                or type(output["size"]) is not int or type(output["sha256"]) is not str):
            raise ValueError("Invalid compiler output inventory")
        path = staging / output["path"]
        if path.resolve() != path or not path.is_file():
            raise ValueError("Invalid compiler output path")
        raw = path.read_bytes()
        source = path.with_suffix(".py").read_bytes()
        if (len(raw) != output["size"] or hashlib.sha256(raw).hexdigest() != output["sha256"]
                or raw[:4] != importlib.util.MAGIC_NUMBER or raw[4:8] != b"\x01\x00\x00\x00"
                or raw[8:16] != importlib.util.source_hash(source)):
            raise ValueError("Compiler output integrity mismatch")
        accepted[output["path"]] = raw
    if set(accepted) != expected:
        raise ValueError("Compiler output inventory incomplete")
    return accepted


def release_integers(source: bytes) -> dict:
    """The launcher integers a stage_runtime.py source declares (absent ones omitted)."""
    values = {}
    for key, name in RELEASE_INTEGERS.items():
        found = re.findall(rb"^" + name.encode() + rb" = ([0-9]{1,6})[ \t]*(?:#.*)?$", source, re.MULTILINE)
        if len(found) > 1:
            raise ValueError(f"Ambiguous release integer {name}")
        if found:
            values[key] = int(found[0])
    return values


def build_runtime(repo: Path, output: Path, revision: str = "HEAD") -> dict:
    """Compile tracked runtime modules and bundle assets without runtime state.

    Only the selected Git revision is read. Configuration, environment files,
    dependencies, models and data remain external to the release archive.
    Existing output files are never replaced.
    """
    if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 13):
        raise ValueError("Build with CPython 3.13, matching the production runtime")
    repo = repo.resolve()
    output = output.resolve()
    if output == repo or repo.is_relative_to(output):
        raise ValueError("Output must be an archive file outside repository parents")
    if output.exists():
        raise FileExistsError(output)
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", revision + "^{commit}"], cwd=repo, text=True,
    ).strip()
    timestamp = int(subprocess.check_output(
        ["git", "show", "-s", "--format=%ct", commit], cwd=repo, text=True,
    ).strip())
    tracked = subprocess.check_output(
        ["git", "ls-tree", "-rz", commit, "--", "src", "tools", "scripts"], cwd=repo,
    )
    files = {}
    declared = {}
    with tempfile.TemporaryDirectory(prefix="kza-runtime-") as temporary:
        staging = Path(temporary).resolve()
        modules = []
        for entry in tracked.split(b"\0"):
            if not entry:
                continue
            metadata, raw_name = entry.split(b"\t", 1)
            mode, kind, object_id = metadata.split()
            path = PurePosixPath(raw_name.decode())
            if kind != b"blob":
                continue
            if path == PurePosixPath("scripts/build_runtime.py"):
                continue
            python_module = path.suffix == ".py"
            frontend_asset = (
                path.is_relative_to("src/dashboard/frontend")
                and path.suffix in ASSET_SUFFIXES
            )
            package_asset = path.parts[0] == "src" and path.suffix == ".csv"
            if not (python_module or frontend_asset or package_asset):
                continue
            if mode == b"120000":
                raise ValueError(f"Runtime symlinks require explicit review: {path}")
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Invalid runtime path: {path}")
            data = subprocess.check_output(["git", "cat-file", "blob", object_id], cwd=repo)
            if path == PurePosixPath("scripts/stage_runtime.py"):
                declared = release_integers(data)
            if python_module:
                source = staging / str(path)
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_bytes(data)
                modules.append({"path": str(path), "sha256": hashlib.sha256(data).hexdigest()})
            else:
                files[str(path)] = data
        if modules:
            files.update(_compile_modules(staging, modules))

    if "src/main.pyc" not in files or "src/__init__.pyc" not in files:
        raise ValueError("Revision does not contain the KZA application")
    manifest = {
        "format": "kza-cpython-bytecode-v1", "git_commit": commit,
        "python": "3.13", "python_magic": importlib.util.MAGIC_NUMBER.hex(),
        "entrypoint": ["python", "-m", "src.main"],
        "external_paths": ["config", "data", "models", "logs"],
        "files": {name: hashlib.sha256(data).hexdigest()
                  for name, data in sorted(files.items())},
        **declared,
    }
    files["release.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".kza-release-", dir=output.parent)
    temporary_output = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as raw:
            with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=timestamp) as zipped:
                with tarfile.open(fileobj=zipped, mode="w") as archive:
                    for name, data in sorted(files.items()):
                        info = tarfile.TarInfo(name)
                        info.size = len(data)
                        info.mode = 0o644
                        info.mtime = timestamp
                        archive.addfile(info, io.BytesIO(data))
        # Atomic publication that refuses a concurrent overwrite.
        os.link(temporary_output, output)
    finally:
        temporary_output.unlink(missing_ok=True)
    return manifest


def main() -> None:
    """Build an archive from the requested committed revision."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--revision", default="HEAD")
    args = parser.parse_args()
    manifest = build_runtime(Path(__file__).resolve().parents[1], args.output, args.revision)
    print(json.dumps({"archive": str(args.output), "commit": manifest["git_commit"],
                      "compiled_modules": sum(p.endswith(".pyc") for p in manifest["files"]),
                      "files": len(manifest["files"])}, indent=2))


if __name__ == "__main__":
    main()
