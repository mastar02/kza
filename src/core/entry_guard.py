"""G1: refuse unmanaged R1 entries under a managed release root.

U3b-0e (design v3 2b, reviews F1 and N8). A managed root is a directory
holding ``admission/controller.json`` (written by install-controller). Under
such a root, R1 code may run only:

- as the managed child, which carries the admission ticket
  (``KZA_RUNTIME_ADMISSION``) and passes the full runtime admission; or
- as the adoption's schema-only child, whose capability
  (``KZA_ADOPTION_SCHEMA_ONLY=1``) is accepted only when the process command is
  exactly :func:`schema_only_command` ('-I -B -c' with the fixed bootstrap).

Everything else is refused before any store opens. The package guard runs
when ``src`` is imported, so it covers every entry that can reach a KZA store
(``python -m src.main``, ``src.training.nightly_trainer``,
``src.health.health_check``, ``src.code_index``, and the scripts/tools that
import ``src``), including a legacy unit or timer that starts R1 code through
``current``. The serve entry repeats the check in
``runtime_admission.verify_runtime_admission`` with the state path it would open.

This module is stdlib-only and must stay importable from ``src/__init__``.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
import logging
import os
from pathlib import Path
import sys

logger = logging.getLogger(__name__)

# Same literal as src.core.runtime_admission.TICKET_ENV (a test pins it).
TICKET_ENV = "KZA_RUNTIME_ADMISSION"
SCHEMA_ONLY_ENV = "KZA_ADOPTION_SCHEMA_ONLY"
SCHEMA_ONLY_VALUE = "1"
CONTROLLER_RECORD = ("admission", "controller.json")
REFUSED_EXIT = 78  # EX_CONFIG: the managed unit's RestartPreventExitStatus
UNMANAGED_CODE = "runtime_admission_unmanaged_under_managed_root"
CAPABILITY_CODE = "runtime_admission_schema_only_capability_refused"

# The only command form that may carry the schema-only capability. argv after
# the source: <runtime root> then the entry's own arguments.
SCHEMA_ONLY_BOOTSTRAP = "\n".join([
    "import sys",
    "sys.path.insert(0, sys.argv[1])",
    "from src.core.schema_only import main",
    "raise SystemExit(main(sys.argv[2:]))",
])


class UnmanagedEntryRefused(SystemExit):
    """A process-level refusal with exit code 78 and a content-free code."""

    def __init__(self, code: str) -> None:
        super().__init__(REFUSED_EXIT)
        self.refusal = code

    def __str__(self) -> str:
        return self.refusal


def schema_only_command(python: str, runtime_root: str | os.PathLike, *arguments: str) -> list[str]:
    """argv of the adoption's schema-only child (run with SCHEMA_ONLY_ENV=1, no ticket)."""
    return [python, "-I", "-B", "-c", SCHEMA_ONLY_BOOTSTRAP, os.fspath(runtime_root), *arguments]


def managed_root_of(paths: Iterable[str | os.PathLike]) -> Path | None:
    """The nearest ancestor (lexical or resolved) holding admission/controller.json.

    Any entry at that name counts (a symlink or a damaged record too): the
    guard errs towards refusing.
    """
    for raw in paths:
        if raw is None:
            continue
        path = Path(os.path.abspath(os.fspath(raw)))
        candidates = [path]
        try:
            candidates.append(path.resolve(strict=False))
        except (OSError, RuntimeError):
            pass
        for candidate in candidates:
            for parent in (candidate, *candidate.parents):
                try:
                    os.lstat(parent.joinpath(*CONTROLLER_RECORD))
                except FileNotFoundError:
                    continue
                except NotADirectoryError:
                    continue
                except OSError:
                    return parent  # unreadable: treat as managed (refuse)
                return parent
    return None


def schema_only_invocation(argv: list[str] | None) -> bool:
    """True iff this process is exactly the schema-only bootstrap under -I -B."""
    if argv is None or not sys.flags.isolated or not sys.flags.dont_write_bytecode:
        return False
    try:
        index = argv.index("-c")
    except ValueError:
        return False
    options = argv[1:index]
    return (set(options) <= {"-I", "-B"} and {"-I", "-B"} <= set(options)
            and index + 2 < len(argv) and argv[index + 1] == SCHEMA_ONLY_BOOTSTRAP)


def entry_decision(*, environ: Mapping[str, str] | None = None,
                   paths: Iterable[str | os.PathLike] = (),
                   argv: list[str] | None = None) -> str | None:
    """Return None to allow, or the refusal code, for one process entry.

    Args:
        environ: Process environment (os.environ).
        paths: Locations this entry runs from or would open (code, cwd, state).
        argv: The original interpreter argv (sys.orig_argv).

    Returns:
        None, ``UNMANAGED_CODE`` or ``CAPABILITY_CODE``.
    """
    environ = os.environ if environ is None else environ
    capability = environ.get(SCHEMA_ONLY_ENV)
    if capability is not None:
        # Accepted only by the schema-only bootstrap, never with a ticket.
        if (capability != SCHEMA_ONLY_VALUE or TICKET_ENV in environ
                or not schema_only_invocation(argv)):
            return CAPABILITY_CODE
        return None
    if TICKET_ENV in environ:
        return None  # the managed child: the full runtime admission decides
    return UNMANAGED_CODE if managed_root_of(paths) is not None else None


def _default_paths() -> list[str]:
    paths = [os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
    try:
        paths.append(os.getcwd())
    except OSError:
        pass
    return paths


def guard_package_entry() -> None:
    """Refuse at ``import src`` (exit 78), before any module opens a store."""
    code = entry_decision(paths=_default_paths(), argv=list(getattr(sys, "orig_argv", sys.argv)))
    if code is not None:
        logger.critical("[G1] refused: %s", code)
        raise UnmanagedEntryRefused(code)


def check_serve_entry(state_paths: Iterable[str | os.PathLike] = (), *,
                      runtime_root: str | os.PathLike | None = None) -> None:
    """The serve entry's G1 (runtime_admission): RuntimeError instead of exit.

    The schema-only capability is never accepted here.
    """
    if SCHEMA_ONLY_ENV in os.environ:
        raise RuntimeError(CAPABILITY_CODE)
    paths = [*_default_paths(), *state_paths]
    if runtime_root is not None:
        paths.append(runtime_root)
    code = entry_decision(paths=paths, argv=None)
    if code is not None:
        raise RuntimeError(code)
