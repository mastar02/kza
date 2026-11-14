"""Owner-state compaction P14 (U3b-0c): user units and logrotate configurations that name a member.

Owner-compaction design v4 F2 (P14), v5 G1 (path resolution) and the v5
review findings E1, E2 and E4. Before compact start and again before retire,
every user unit kza's manager can run is read through ``systemctl --user``
and every logrotate configuration those units (or the launcher install
record) reference is read as text. The scan refuses when any path they name
reaches a retirement member:

* ``runtime_admission_compaction_unit_scan_unavailable``: any systemctl call
  failed (non-zero exit, timeout, no user bus). Never read as an empty list.
* ``runtime_admission_compaction_unit_unparsed:<unit>:<directive>``: a
  path-bearing value still holds an unresolved ``$VAR``/``${VAR}`` or a
  ``%`` specifier, an argv cannot be read, or a required EnvironmentFile is
  missing; ``...:logrotate:<basename>`` for a configuration that is missing,
  unreadable or does not parse.
* ``runtime_admission_compaction_unit_path:<unit>``: a path is equal to, under
  or (strictly below the admission root) above a member root, by its literal
  form, its resolved form or the device/inode of any existing component.
  Logrotate findings name ``logrotate:<basename>`` as the unit.

Unit text and logrotate configurations are outside members and hold no
personal data. Nothing under a member root is opened or listed: a path is
compared before the file it names is read, and a glob is expanded one
directory at a time, never into a member. The report is content-free.
"""
from __future__ import annotations

import fnmatch
import glob
import logging
import os
from pathlib import Path
import re
import shlex
import stat
from typing import Callable

logger = logging.getLogger(__name__)

launcher = None  # scripts/managed_launcher
commands = None  # scripts/adoption_commands


def bind(launcher_module, commands_module) -> None:
    """Use these implementations (the compiled controller loads this module by path and binds it)."""
    global launcher, commands
    launcher, commands = launcher_module, commands_module


if __package__ == "scripts":  # in-process users (tests)
    from scripts import adoption_commands as _commands_module  # noqa: E402
    from scripts import managed_launcher as _launcher_module  # noqa: E402

    bind(_launcher_module, _commands_module)


SCAN_UNAVAILABLE = "runtime_admission_compaction_unit_scan_unavailable"
UNPARSED = "runtime_admission_compaction_unit_unparsed:"
PATH_HIT = "runtime_admission_compaction_unit_path:"
MEMBERS_INVALID = "runtime_admission_compaction_unit_members"
SYSTEMCTL_TIMEOUT = 30.0  # list-unit-files on a loaded manager can exceed ServiceManager's 2 s default

# E2: the union of these two listings. Every unit file state is listed, not only E2's enabled,
# generated, static and linked: a disabled, inactive unit is neither loaded (list-units) nor in
# those states, yet the operator can start it (and the runbook disables kza-voice during a
# compaction). Found by the real-systemd rehearsal; vendor units are still dropped by origin.
LIST_UNIT_FILES = ("list-unit-files", "--no-legend", "--plain", "--no-pager")
LIST_UNITS = ("list-units", "--all", "--type=service,timer,socket,path", "--no-legend", "--plain", "--no-pager")
# Review F9: the manager's own environment block (set-environment, the user session's import) is
# what systemd substitutes into $VAR / ${VAR} besides the unit's Environment= and EnvironmentFile=.
SHOW_ENVIRONMENT = ("show-environment",)
# Variables systemd itself sets per process whose values are never paths (pids, ids, results,
# counters, unit names). A reference to one of them resolves to a non-path placeholder; every other
# unresolved reference (the *_DIRECTORY and TRIGGER_PATH ones included) refuses.
SYSTEMD_PROVIDED = frozenset({
    "MAINPID", "MANAGERPID", "INVOCATION_ID", "SERVICE_RESULT", "EXIT_CODE", "EXIT_STATUS",
    "MONITOR_SERVICE_RESULT", "MONITOR_EXIT_CODE", "MONITOR_EXIT_STATUS", "MONITOR_INVOCATION_ID",
    "MONITOR_UNIT", "TRIGGER_UNIT", "TRIGGER_TIMER_REALTIME_USEC", "TRIGGER_TIMER_MONOTONIC_USEC",
    "LISTEN_FDS", "LISTEN_PID", "LISTEN_FDNAMES", "WATCHDOG_PID", "WATCHDOG_USEC", "JOURNAL_STREAM",
    "USER", "LOGNAME"})
_PROVIDED_PLACEHOLDER = "0"

ORIGIN_PROPERTIES = ("LoadState", "FragmentPath", "SourcePath", "DropInPaths")
EXEC_PROPERTIES = ("ExecCondition", "ExecStartPre", "ExecStart", "ExecStartPost", "ExecReload", "ExecStop",
                   "ExecStopPost")
LIST_PROPERTIES = ("ReadWritePaths", "ReadOnlyPaths")
BIND_PROPERTIES = ("BindPaths", "BindReadOnlyPaths")
STREAM_PROPERTIES = ("StandardOutput", "StandardError")
TRIGGER_PROPERTIES = ("Triggers", "Unit")
SHOWN = (ORIGIN_PROPERTIES + ("Environment", "EnvironmentFiles", "WorkingDirectory", "RootDirectory")
         + EXEC_PROPERTIES + LIST_PROPERTIES + BIND_PROPERTIES + ("StateDirectory", "LogsDirectory")
         + STREAM_PROPERTIES + TRIGGER_PROPERTIES)
TRIGGERING_TYPES = (".timer", ".socket", ".path")
SKIPPED_LOAD_STATES = frozenset({"not-found", "masked"})  # nothing systemd could run
FILE_OUTPUTS = ("file:", "append:", "truncate:")

# E1: a token is path-bearing when absolute, when it contains '/', or when it follows one of these.
PATH_OPTIONS = frozenset({"--root", "--state", "--config", "--log-file", "--logrotate"})
LOGROTATE = "logrotate"
LOGROTATE_PATH_OPTIONS = frozenset({"-s", "--state", "-l", "--log"})
LOGROTATE_VALUE_OPTIONS = LOGROTATE_PATH_OPTIONS | {"-m", "--mail"}
LOGROTATE_SCRIPTS = frozenset({"postrotate", "prerotate", "firstaction", "lastaction", "preremove"})
LOGROTATE_FILE_LIMIT = 1 << 20
LOGROTATE_DEPTH = 16
GLOB_ENTRY_LIMIT = 20000


class UnitScanRefusal(ValueError):
    """A content-free refusal of the P14 scan; ``str(error)`` is the code."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _refuse(code: str):
    raise UnitScanRefusal(code)


def strict_systemctl(manager) -> Callable[..., str]:
    """``run(*args) -> stdout`` of ``systemctl --user <args>`` through ``manager`` (a ServiceManager).

    Exit 0 only (``strict=True``): ServiceManager's default accepts exit 1,
    which is also how systemctl reports 'Failed to connect to bus'. Any
    failure raises the scan_unavailable refusal (E2).
    """
    def run(*arguments: str) -> str:
        output = manager._run("--user", *arguments, timeout=SYSTEMCTL_TIMEOUT, strict=True)
        if output is None:
            _refuse(SCAN_UNAVAILABLE)
        return output
    return run


def _within(path: str, parent: str) -> bool:
    return path == parent or path.startswith(parent.rstrip("/") + "/")


def _show_values(output: str, names: tuple[str, ...]) -> dict[str, list[str]]:
    """``systemctl show -p ...`` output as name -> every value line (Exec* repeat one line per command)."""
    values: dict[str, list[str]] = {name: [] for name in names}
    for line in output.splitlines():
        key, separator, value = line.partition("=")
        if separator and key in values:
            values[key].append(value)
    return values


def _listed_names(output: str) -> set[str]:
    names = set()
    for line in output.splitlines():  # list-units marks failed or not-found units with a leading bullet
        words = [word for word in line.split() if word not in ("●", "*")]
        if words:
            names.add(words[0])
    return names


def _template(unit: str) -> bool:
    return re.search(r"@\.[a-z]+$", unit) is not None


def _template_of(unit: str) -> str | None:
    """``name@instance.type`` -> ``name@.type`` (None for a unit that is not an instance)."""
    match = re.fullmatch(r"(.+@)[^@]+(\.[a-z]+)", unit)
    return match.group(1) + match.group(2) if match and not _template(unit) else None


def _listed_states(output: str) -> dict[str, str]:
    """list-unit-files output as name -> unit file state (linked, static, enabled, ...)."""
    states = {}
    for line in output.splitlines():
        words = [word for word in line.split() if word not in ("●", "*")]
        if len(words) >= 2:
            states[words[0]] = words[1]
    return states


# Review r2 N4: a template unit in a user unit directory is read as text (systemctl show refuses a
# template name; an instance is listed only once loaded). These keys' values are path-bearing.
TEMPLATE_KEYS = frozenset(EXEC_PROPERTIES) | frozenset(LIST_PROPERTIES) | frozenset(BIND_PROPERTIES) | {
    "WorkingDirectory", "RootDirectory", "Environment", "EnvironmentFile", "StandardOutput", "StandardError",
    "StateDirectory", "LogsDirectory"}
LINKED_STATES = frozenset({"linked", "linked-runtime"})


def _exec_argv(unit: str, directive: str, value: str) -> list[list[str]]:
    """The argv readings of one ``show`` Exec* value, plus its ``path=``.

    ``systemctl show`` joins argv with spaces and no quoting, so the shell
    reading (when it parses) and the plain whitespace reading are both
    returned and both checked: stricter than either alone. The argv runs up
    to the last ``; ignore_errors=`` so a ``;`` inside an argument is kept.
    """
    match = re.search(r"argv\[\]=(.*) ; ignore_errors=", value)
    readings: list[list[str]] = []
    if match is None:
        try:
            readings.append(launcher.exec_start_argv(value))
        except ValueError:
            _refuse(UNPARSED + unit + ":" + directive)
        raw = re.search(r"argv\[\]=(.*?) ;", value).group(1)
    else:
        raw = match.group(1)
        try:
            readings.append(shlex.split(raw))
        except ValueError:
            pass
    readings.append([word.strip("'\"") for word in raw.split()])
    program = re.search(r"\{ path=(.*?) ; argv\[\]=", value)
    if program is not None and program.group(1):
        readings.append([program.group(1)])
    return readings


def _stream_assignments(text: str) -> dict[str, str | None]:
    """The effective StandardOutput=/StandardError= of ``systemctl cat`` text (last wins, empty resets)."""
    values: dict[str, str | None] = {name: None for name in STREAM_PROPERTIES}
    for raw in text.replace("\\\n", " ").splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ";", "[")):
            continue
        key, separator, value = line.partition("=")
        if separator and key.strip() in values:
            values[key.strip()] = value.strip() or None
    return values


class _Scan:
    """One P14 pass: the member forms, the unit walk and the logrotate parse."""

    def __init__(self, systemctl: Callable[..., str], root: Path, members: list[dict], home: Path, uid: int,
                 environ: dict, named: tuple[str, ...] = ()) -> None:
        self.systemctl = systemctl
        self.home = os.path.normpath(os.fspath(home))
        self.environ = environ
        self.roots = {os.path.normpath(os.fspath(root)), os.path.realpath(root)}
        self.members: list[str] = []
        self.identities: set[tuple[int, int]] = set()
        for member in members:
            try:
                path, identity = member["path"], member["identity"]
                device, inode = identity["device"], identity["inode"]
            except (KeyError, TypeError):
                _refuse(MEMBERS_INVALID)
            if (not isinstance(path, str) or not os.path.isabs(path) or not isinstance(device, int)
                    or not isinstance(inode, int)):
                _refuse(MEMBERS_INVALID)
            self.members += sorted({os.path.normpath(path), os.path.realpath(path)})
            self.identities.add((device, inode))
        self.uid = uid
        self.unit_directories = self._unit_directories(uid)
        self._manager_environment: dict | None = None
        # U3b-0e4 rule (review F9 follow-up): an unresolved reference is tolerated only in a unit nobody
        # named whose text and own environment mention no kza path, no member and no root.
        self.named = frozenset({launcher.UNIT, commands.LLM_FAST_UNIT, commands.CODE_INDEX_UNIT, *named})
        self.operator_named = frozenset(named)
        self.states: dict[str, str] = {}
        self.markers = commands.kza_markers([os.fspath(root), os.path.join(self.home, "kza"), *self.members])
        self._texts: dict[str, str] = {}
        self.tolerated: set[str] = set()
        self.logrotate_configs: list[tuple[str, str]] = []
        self.logrotate_seen: set[str] = set()
        self.glob_entries = 0
        self.templates: list[str] = []

    # --- matching (rule 5) ----------------------------------------------------------------------

    def _unit_directories(self, uid: int) -> tuple[str, ...]:
        config = self.environ.get("XDG_CONFIG_HOME") or os.path.join(self.home, ".config")
        data = self.environ.get("XDG_DATA_HOME") or os.path.join(self.home, ".local", "share")
        runtimes = {f"/run/user/{uid}"}
        if self.environ.get("XDG_RUNTIME_DIR"):
            runtimes.add(self.environ["XDG_RUNTIME_DIR"])
        found = set()
        for base in {config, os.path.join(self.home, ".config")}:
            found |= {os.path.join(base, "systemd", "user"), os.path.join(base, "systemd", "user.control"),
                      os.path.join(base, "containers", "systemd")}
        found.add(os.path.join(data, "systemd", "user"))
        for runtime in runtimes:
            for name in ("generator", "generator.early", "generator.late", "user", "user.control", "transient"):
                found.add(os.path.join(runtime, "systemd", name))
        return tuple(sorted({os.path.normpath(path) for path in found}
                            | {os.path.realpath(path) for path in found}))

    def _under_root(self, path: str) -> bool:
        """Strictly below the admission root: an ancestor of a member here is a reference to it."""
        return any(path != root and _within(path, root) for root in self.roots)

    def _identity_hit(self, path: str) -> bool:
        current = "/"
        for part in path.split("/"):
            if not part:
                continue
            current = os.path.join(current, part)
            try:
                info = os.stat(current)
            except (OSError, ValueError):
                return False
            if (info.st_dev, info.st_ino) in self.identities:
                return True
        return False

    def hit(self, path: str, ancestors: bool = True) -> bool:
        """Whether an absolute path reaches a member (literal or resolved form, or a component's identity).

        An ancestor of a member counts only strictly below the admission
        root (``<root>/slots``): the root itself and everything above it
        (the home directory, '/') contain every slot by construction, and
        the launcher's own unit names the root (WorkingDirectory, --root).
        ``ancestors`` False (a directory a glob walks through) drops that rule.
        """
        literal = os.path.normpath(path)
        resolved = os.path.realpath(literal)
        for form in {literal, resolved}:
            for member in self.members:
                if _within(form, member) or (ancestors and _within(member, form) and self._under_root(form)):
                    return True
        return self._identity_hit(literal) or (resolved != literal and self._identity_hit(resolved))

    def check(self, unit: str, path: str, ancestors: bool = True) -> None:
        if self.hit(path, ancestors):
            _refuse(PATH_HIT + unit)

    def check_value(self, unit: str, text: str, base: str, rooted: str | None = None) -> None:
        """Every path reading of one path-bearing value: whole, after '=', each ':' piece and word."""
        pieces = {text}
        if "=" in text:
            pieces.add(text.split("=", 1)[1])
        for piece in list(pieces):
            pieces.update(commands._words(piece))
        for piece in list(pieces):
            if ":" in piece:
                pieces.update(part for part in piece.split(":") if "/" in part)
        for piece in pieces:
            if not piece:
                continue
            if piece == "~" or piece.startswith("~/"):
                piece = self.home + piece[1:]
            path = piece if os.path.isabs(piece) else os.path.join(base, piece)
            self.check(unit, path)
            if rooted and os.path.isabs(piece):
                self.check(unit, os.path.join(rooted, piece.lstrip("/")))

    # --- systemctl ------------------------------------------------------------------------------

    def call(self, *arguments: str) -> str:
        try:
            output = self.systemctl(*arguments)
        except UnitScanRefusal:
            raise
        except Exception:  # noqa: BLE001 - any failure of the reader is fail-closed (E2)
            logger.warning("[CompactionUnits] systemctl %s failed", arguments[0])
            _refuse(SCAN_UNAVAILABLE)
        if not isinstance(output, str):
            _refuse(SCAN_UNAVAILABLE)
        return output

    def show(self, unit: str) -> dict[str, list[str]]:
        arguments = ["show", unit]
        for name in SHOWN:
            arguments += ["-p", name]
        values = _show_values(self.call(*arguments), SHOWN)
        if not values["LoadState"]:
            _refuse(SCAN_UNAVAILABLE)  # an answer without the unit's load state is not an answer
        return values

    def _in_unit_directories(self, origins: list[str]) -> bool:
        return any(_within(os.path.normpath(path), directory) or _within(os.path.realpath(path), directory)
                   for path in origins if os.path.isabs(path) for directory in self.unit_directories)

    def _installed_here(self, unit: str) -> bool:
        """A unit (or an instance's template) whose name is in a user unit directory or is linked there.

        Review r2 N4: ``systemctl --user link`` puts a symlink in the user
        unit directory; whatever FragmentPath a manager then reports (systemd
        255 reports the symlink, others the target), the unit is the account's.
        """
        names = {unit} | ({_template_of(unit)} - {None})
        return any(self.states.get(name) in LINKED_STATES for name in names) or any(
            os.path.lexists(os.path.join(directory, name)) for directory in self.unit_directories for name in names)

    def _template_ours(self, unit: str) -> bool:
        """A (vendor) template with a drop-in directory in a user unit directory: its text includes it."""
        return any(os.path.lexists(os.path.join(directory, unit + ".d")) for directory in self.unit_directories)

    def _ours(self, unit: str, shown: dict[str, list[str]]) -> bool:
        if unit in self.named or self._installed_here(unit):
            return True  # review r2 N4: a named unit is scanned wherever its file lives
        origins = [value for name in ("FragmentPath", "SourcePath") for value in shown[name] if value]
        origins += [word for value in shown["DropInPaths"] for word in value.split()]
        return self._in_unit_directories(origins)

    def _origin_check(self, unit: str, shown: dict[str, list[str]]) -> None:
        """A unit file or drop-in that lives in a member (a linked unit) is a reference to it."""
        for name in ("FragmentPath", "SourcePath", "DropInPaths"):
            for value in shown[name]:
                for word in value.split():
                    if os.path.isabs(word):
                        self.check(unit, word)

    # --- one unit (rules 3 and 4) ---------------------------------------------------------------

    def _substitute(self, token: str, environment: dict) -> tuple[str, bool]:
        """systemd's $VAR/${VAR} substitution (manager, then unit environment); ``$$`` is a literal dollar."""
        unresolved = False

        def replace(match):
            nonlocal unresolved
            if match.group(0) == "$$":
                return "$"
            name = match.group(1) or match.group(2)
            if name in environment:
                return environment[name]
            if name in SYSTEMD_PROVIDED:
                return _PROVIDED_PLACEHOLDER
            unresolved = True
            return match.group(0)
        return commands._ENV_REFERENCE.sub(replace, token), unresolved

    def manager_environment(self) -> dict:
        """``systemctl --user show-environment`` as name -> value (read once per scan).

        systemd prints a value with special characters as ``$'...'`` (C-style
        escapes). Such a value is not decoded here: a reference to it stays
        unresolved, which refuses (fail closed).
        """
        if self._manager_environment is None:
            environment: dict = {}
            for line in self.call(*SHOW_ENVIRONMENT).splitlines():
                name, separator, value = line.partition("=")
                if not separator or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None:
                    _refuse(SCAN_UNAVAILABLE)  # not the manager's environment block
                if not value.startswith("$'"):
                    environment[name] = value
            self._manager_environment = environment
        return self._manager_environment

    def _environment(self, unit: str, shown: dict[str, list[str]]) -> dict:
        environment: dict[str, str] = {}
        for value in shown["Environment"]:
            try:
                assignments = shlex.split(value)
            except ValueError:
                _refuse(UNPARSED + unit + ":Environment")
            for assignment in assignments:
                name, _separator, item = assignment.partition("=")
                environment[name] = item
        for specification in shown["EnvironmentFiles"]:
            if not specification.strip():
                continue
            path = specification.rsplit(" (ignore_errors=", 1)[0].strip().lstrip("-")
            if not os.path.isabs(path):
                _refuse(UNPARSED + unit + ":EnvironmentFiles")
            self.check(unit, path)  # before it is opened: nothing in a member is read
            try:
                values, _keys = commands._env_file_values(unit, specification)
            except commands.CommandRefusal:
                _refuse(UNPARSED + unit + ":EnvironmentFiles")
            environment.update(values)
        return environment

    def _working_directory(self, unit: str, shown: dict[str, list[str]]) -> str:
        directory = (shown["WorkingDirectory"] or [""])[-1].lstrip("-!")
        if not directory or directory == "~":
            return self.home
        if directory.startswith("~/"):
            directory = self.home + directory[1:]
        if not os.path.isabs(directory):
            _refuse(UNPARSED + unit + ":WorkingDirectory")
        self.check(unit, directory)
        return directory

    def _tolerates_unresolved(self, unit: str, own: dict) -> bool:
        """The U3b-0e4 leniency (adoption_commands.mentions_markers): not named, and not kza's by its text.

        systemd substitutes an unset ``${NAME}`` with nothing and leaves a
        ``$NAME`` inside a word to the program (stock IBus, unity and podman
        units do this). A named unit, or one whose unit text or own
        environment names a kza path, a member or the root, still refuses.
        """
        if unit in self.named:
            return False
        if unit not in self._texts:
            self._texts[unit] = self.call("cat", unit)
        if commands.mentions_markers(self._texts[unit], own, self.markers):
            return False
        self.tolerated.add(unit)
        return True

    def _exec(self, unit: str, directive: str, value: str, environment: dict, base: str, rooted: str | None,
              own: dict | None = None) -> None:
        for argv in _exec_argv(unit, directive, value):
            texts = [(token, *self._substitute(token, environment)) for token in argv]
            rotate = next((index for index, (_token, text, _unresolved) in enumerate(texts)
                           if os.path.basename(text) == LOGROTATE), None)
            takes_value = False
            for index, (token, text, unresolved) in enumerate(texts):
                after_rotate = rotate is not None and index > rotate
                options = PATH_OPTIONS | (LOGROTATE_PATH_OPTIONS if after_rotate else frozenset())
                option = token.split("=", 1)[0] if token.startswith("-") and "=" in token else None
                bearing = ((index > 0 and texts[index - 1][0] in options) or option in options
                           or "/" in text or "/" in token or text.startswith("~"))
                config = False
                if after_rotate:
                    if takes_value:
                        takes_value = False
                    elif token.startswith("-"):
                        takes_value = token in LOGROTATE_VALUE_OPTIONS
                    else:
                        config = bearing = True
                if unresolved and not self._tolerates_unresolved(unit, own or {}):
                    # Review F9: a bare $VAR / ${VAR} is refused too; its value could name a member.
                    _refuse(UNPARSED + unit + ":" + directive)
                if not bearing:
                    continue
                if "%" in text:
                    _refuse(UNPARSED + unit + ":" + directive)
                self.check_value(unit, text, base, rooted)
                if config:
                    path = text if os.path.isabs(text) else os.path.join(base, text)
                    self.logrotate_configs.append((os.path.normpath(path), unit))

    def _streams(self, unit: str, shown: dict[str, list[str]]) -> None:
        missing = []
        for name in STREAM_PROPERTIES:
            value = (shown[name] or [""])[-1]
            prefix = next((item for item in FILE_OUTPUTS if value.startswith(item)), None)
            if prefix is not None and os.path.isabs(value[len(prefix):]):
                self.check(unit, value[len(prefix):])
            else:
                missing.append(name)
        if not missing:
            return
        assignments = _stream_assignments(self.call("cat", unit))
        for name in missing:
            value = assignments[name]
            prefix = next((item for item in FILE_OUTPUTS if value and value.startswith(item)), None)
            if prefix is None:
                continue
            path = value[len(prefix):]
            if "%" in path or "$" in path or not os.path.isabs(path):
                _refuse(UNPARSED + unit + ":" + name)
            self.check(unit, path)

    def scan_unit(self, unit: str, shown: dict[str, list[str]]) -> None:
        """Every path one unit names (rule 3), after its environment (rule 4)."""
        own = self._environment(unit, shown)
        environment = {**self.manager_environment(), **own}
        base = self._working_directory(unit, shown)
        rooted = None
        for value in shown["RootDirectory"]:
            if value:
                if not os.path.isabs(value):
                    _refuse(UNPARSED + unit + ":RootDirectory")
                self.check(unit, value)
                rooted = value
        for value in own.values():
            text, _unresolved = self._substitute(value, environment)  # systemd leaves these to the program
            if "/" in text or text.startswith("~"):
                self.check_value(unit, text, base, rooted)
        for name in LIST_PROPERTIES + BIND_PROPERTIES:
            for value in shown[name]:
                for item in value.split():
                    for piece in item.split(":"):
                        piece = piece.lstrip("-+")
                        if piece.startswith("/"):
                            self.check(unit, piece)
        state_home = self.environ.get("XDG_STATE_HOME") or os.path.join(self.home, ".local", "state")
        for name, base_directory in (("StateDirectory", state_home), ("LogsDirectory", os.path.join(state_home, "log"))):
            for value in shown[name]:
                for item in value.split():
                    for piece in item.split(":"):
                        if piece:
                            self.check(unit, os.path.join(base_directory, piece))
        for directive in EXEC_PROPERTIES:
            for value in shown[directive]:
                if value.strip():
                    self._exec(unit, directive, value, environment, base, rooted, own)
        self._streams(unit, shown)

    def _specifiers(self, word: str) -> str:
        """The account-fixed systemd specifiers of a user unit; the others (%i, %n, ...) stay unexpanded."""
        config = self.environ.get("XDG_CONFIG_HOME") or os.path.join(self.home, ".config")
        state = self.environ.get("XDG_STATE_HOME") or os.path.join(self.home, ".local", "state")
        cache = self.environ.get("XDG_CACHE_HOME") or os.path.join(self.home, ".cache")
        runtime = self.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{self.uid}"
        fixed = {"h": self.home, "t": runtime, "E": config, "S": state, "C": cache, "L": os.path.join(state, "log"),
                 "T": "/tmp", "V": "/var/tmp", "U": str(self.uid), "%": "\0"}
        return re.sub(r"%(.)", lambda match: fixed.get(match.group(1), match.group(0)), word)

    def scan_template(self, unit: str) -> None:
        """Review r2 N4: a template's path-bearing values, from its text, up to the first unexpanded part.

        An instance can be started at any time with any instance name, so the
        literal part of each word is what the template fixes: it refuses when
        it reaches a member, or when the part after it (``%i``, ``$VAR``) could
        land under the admission root (the literal directory is the root, under
        it, or above it).
        """
        text = self.call("cat", unit)
        for line in text.splitlines():
            line = line.strip()
            if not line or line[0] in "#;[":
                continue
            key, separator, value = line.partition("=")
            key = key.strip()
            if not separator or key not in TEMPLATE_KEYS:
                continue
            for word in value.split():
                word = word.strip("\"'")
                if re.match(r"-*[A-Za-z_][\w.-]*=", word):
                    word = word.split("=", 1)[1]  # NAME=value, --option=value
                word = word.lstrip("-@:+!")  # Exec* prefixes, ignore-missing '-'
                for prefix in FILE_OUTPUTS:
                    if word.startswith(prefix):
                        word = word[len(prefix):]
                expanded = self._specifiers(word).replace("$$", "\0")
                literal = re.split(r"[%$]", expanded, maxsplit=1)[0].replace("\0", "%")
                if literal == "~" or literal.startswith("~/"):
                    literal = self.home + literal[1:]
                if not literal.startswith("/"):
                    continue
                self.check_value(unit, literal, self.home)
                if literal != expanded.replace("\0", "%"):
                    directory = os.path.normpath(literal if literal.endswith("/") else os.path.dirname(literal))
                    if any(_within(directory, root) or _within(root, directory) for root in self.roots):
                        _refuse(PATH_HIT + unit)

    def check_manager_environment(self) -> None:
        """Review r2 N5: a manager environment value is held to the unit Environment= rule.

        Every process of the manager inherits it, so a program (or a shell
        reading ``$$NAME``) can use a value no unit text shows.
        """
        for value in self.manager_environment().values():
            if "/" in value or value.startswith("~"):
                self.check_value("show-environment", value, self.home)

    # --- the unit walk (E2) ---------------------------------------------------------------------

    def units(self) -> list[str]:
        self.check_manager_environment()
        files = self.call(*LIST_UNIT_FILES)
        self.states = _listed_states(files)
        listed = _listed_names(files) | _listed_names(self.call(*LIST_UNITS))
        shown: dict[str, dict[str, list[str]]] = {}
        queue = []
        for unit in sorted(listed | self.named):
            if _template(unit):
                # An instance is listed (list-units --all) once it is loaded; before that, the
                # template in a user unit directory is read as text (review r2 N4).
                if self._installed_here(unit) or self._template_ours(unit):
                    self.scan_template(unit)
                    self.templates.append(unit)
                continue
            shown[unit] = self.show(unit)
            if unit not in listed and shown[unit]["LoadState"][-1] in SKIPPED_LOAD_STATES:
                if unit in self.operator_named:
                    _refuse(UNPARSED + unit + ":LoadState")  # a unit the operator named must exist
                continue
            self._origin_check(unit, shown[unit])
            if self._ours(unit, shown[unit]):
                queue.append(unit)
        scanned: list[str] = []
        done: set[str] = set()
        while queue:
            unit = queue.pop(0)
            if unit in done:
                continue
            done.add(unit)
            if unit not in shown:
                shown[unit] = self.show(unit)
                self._origin_check(unit, shown[unit])
            values = shown[unit]
            if values["LoadState"][-1] in SKIPPED_LOAD_STATES:
                continue
            self.scan_unit(unit, values)
            scanned.append(unit)
            if unit.endswith(TRIGGERING_TYPES):
                for name in TRIGGER_PROPERTIES:
                    for value in values[name]:
                        queue += [word for word in value.split() if not _template(word) and word not in done]
        return sorted(scanned)

    # --- logrotate (rule 6, E4) -----------------------------------------------------------------

    def _read(self, label: str, path: str) -> str:
        """The text of one configuration (outside members: compared first), O_NOFOLLOW and bounded."""
        self.check(label, path)
        real = os.path.realpath(path)
        try:
            descriptor = os.open(real, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except OSError:
            _refuse(UNPARSED + label)
        with os.fdopen(descriptor, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                _refuse(UNPARSED + label)
            raw = stream.read(LOGROTATE_FILE_LIMIT + 1)
        if len(raw) > LOGROTATE_FILE_LIMIT:
            _refuse(UNPARSED + label)
        try:
            return raw.decode()
        except UnicodeDecodeError:
            _refuse(UNPARSED + label)

    def _absolute(self, label: str, value: str, directory: str) -> str:
        if value == "~" or value.startswith("~/"):
            value = self.home + value[1:]
        return os.path.normpath(value if os.path.isabs(value) else os.path.join(directory, value))

    def _glob(self, label: str, pattern: str) -> None:
        """A glob target: its literal prefix, what the pattern can name, and (under the root) its expansion."""
        parts = os.path.normpath(pattern).split("/")[1:]
        first = next(index for index, part in enumerate(parts) if glob.has_magic(part))
        prefix = "/" + "/".join(parts[:first])
        self.check(label, prefix)
        resolved = tuple(part for part in os.path.realpath(prefix).split("/") if part)
        for form in {tuple(parts), resolved + tuple(parts[first:])}:
            for member in self.members:
                names = member.split("/")[1:]
                if not all(fnmatch.fnmatchcase(name, part) for name, part in zip(names, form)):
                    continue
                if len(names) <= len(form) or self._under_root("/" + "/".join(names[:len(form)])):
                    _refuse(PATH_HIT + label)
        if any(_within(form, root) for form in {prefix, os.path.realpath(prefix)} for root in self.roots):
            self._expand(label, prefix, parts[first:])

    def _expand(self, label: str, prefix: str, parts: list[str]) -> None:
        """glob.glob one directory at a time; each entry is compared (lstat/stat) before it is descended."""
        current = [prefix]
        for index, part in enumerate(parts):
            last = index == len(parts) - 1
            following = []
            for directory in current:
                if glob.has_magic(part):
                    names = sorted(glob.glob(part, root_dir=directory))
                else:
                    names = [part] if os.path.lexists(os.path.join(directory, part)) else []
                for name in names:
                    self.glob_entries += 1
                    if self.glob_entries > GLOB_ENTRY_LIMIT:
                        _refuse(UNPARSED + label)
                    path = os.path.join(directory, name)
                    self.check(label, path, ancestors=last)  # walked through, not named: <root>/slots
                    if not last and os.path.isdir(path):
                        following.append(path)
            current = following

    def _target(self, label: str, value: str) -> None:
        if value == "~" or value.startswith("~/"):
            value = self.home + value[1:]
        if not os.path.isabs(value):
            _refuse(UNPARSED + label)
        if glob.has_magic(value):
            self._glob(label, value)
        else:
            self.check(label, value)

    def _words(self, label: str, text: str) -> list[str]:
        try:
            return shlex.split(text)
        except ValueError:
            _refuse(UNPARSED + label)

    def _include(self, label: str, value: str, directory: str, state: dict, depth: int) -> None:
        path = self._absolute(label, value, directory)
        self.check(label, path)
        if os.path.isdir(path):
            try:
                with os.scandir(path) as entries:
                    files = sorted(entry.path for entry in entries if entry.is_file())
            except OSError:
                _refuse(UNPARSED + "logrotate:" + os.path.basename(path))
            for item in files:
                self.parse_logrotate(item, state, depth + 1)
        else:
            self.parse_logrotate(path, state, depth + 1)

    def parse_logrotate(self, path: str, state: dict, depth: int = 0) -> None:
        """One configuration and everything it includes: targets, olddir, include, script paths."""
        label = "logrotate:" + os.path.basename(path)
        if depth > LOGROTATE_DEPTH:
            _refuse(UNPARSED + label)
        self.check(label, path)
        if not os.path.lexists(path):
            _refuse(UNPARSED + label)
        real = os.path.realpath(path)
        if real in self.logrotate_seen:
            return
        self.logrotate_seen.add(real)
        text = self._read(label, path)
        directory = os.path.dirname(os.path.normpath(path))
        pending: list[str] = []
        block: dict | None = None
        script = False
        for raw in text.splitlines():
            line = raw.strip()
            if script:
                if line.split()[:1] == ["endscript"]:
                    script = False
                else:
                    for word in re.split(r"[\s;|&()<>]+", line):
                        word = word.strip("'\"")
                        if word.startswith("/") and not re.search(r"[$`*?\[]", word):
                            self.check(label, word)
                continue
            line = re.sub(r"(^|\s)#.*$", "", line).strip()
            for segment in re.split(r"([{}])", line):
                segment = segment.strip()
                if not segment:
                    continue
                if segment == "{":
                    if block is not None or not pending:
                        _refuse(UNPARSED + label)
                    block, pending = {"targets": pending, "olddir": state["olddir"]}, []
                elif segment == "}":
                    if block is None:
                        _refuse(UNPARSED + label)
                    self._finish(label, block)
                    block = None
                elif script:
                    continue  # the rest of a script keyword's line
                else:
                    words = self._words(label, segment)
                    if not words:
                        continue
                    if block is None and ("/" in words[0] or words[0].startswith("~")):
                        pending += words
                        continue
                    if pending:
                        _refuse(UNPARSED + label)  # a directive between the log names and '{'
                    script = self._directive(label, words, block, state, directory, depth)
        if block is not None or pending or script:
            _refuse(UNPARSED + label)

    def _directive(self, label: str, words: list[str], block: dict | None, state: dict, directory: str,
                   depth: int) -> bool:
        keyword = words[0]
        if keyword in LOGROTATE_SCRIPTS:
            return True
        if keyword == "include":
            if len(words) != 2:
                _refuse(UNPARSED + label)
            self._include(label, words[1], directory, state, depth)
        elif keyword == "olddir":
            if len(words) != 2:
                _refuse(UNPARSED + label)
            (block if block is not None else state)["olddir"] = words[1]
        elif keyword == "noolddir":
            (block if block is not None else state)["olddir"] = None
        return False  # any other directive is ignored

    def _finish(self, label: str, block: dict) -> None:
        olddir = block["olddir"]
        for target in block["targets"]:
            self._target(label, target)
            if olddir:
                if olddir == "~" or olddir.startswith("~/") or os.path.isabs(olddir):
                    self._target(label, olddir)
                else:  # relative to each log's directory
                    self._target(label, os.path.join(os.path.dirname(target), olddir))


def scan_units(systemctl: Callable[..., str], *, root: Path, members: list[dict], home: Path | None = None,
               uid: int | None = None, extra_logrotate: tuple[Path, ...] = (), environ: dict | None = None,
               named: tuple[str, ...] = ()) -> dict:
    """P14: refuse when any user unit or logrotate configuration names a retirement member.

    Args:
        systemctl: ``run(*args) -> stdout`` of ``systemctl --user <args>``
            (strict_systemctl in production); a failure raises or returns
            a non-string, and the scan refuses.
        root: The admission root (members are its slots).
        members: ``{"path": abs root of the member, "identity": {"device", "inode"}}`` each.
        home: The account's home (default ``~``).
        uid: The account's uid for ``/run/user/<uid>`` (default the caller's).
        extra_logrotate: Configurations to parse besides those the units
            reference (the launcher install record's ``logrotate_conf``).
        environ: XDG_* lookups (default ``os.environ``).
        named: Units the operator lists besides kza-voice, kza-llm-fast and
            kza-code-index; a named unit never gets the unresolved-reference
            leniency (U3b-0e4 rule).

    Returns:
        ``{"units_scanned", "unit_names", "logrotate_files", "tolerated_unresolved", "templates_scanned"}``
        (content-free).

    Raises:
        UnitScanRefusal: scan_unavailable, unit_unparsed or unit_path.
    """
    scan = _Scan(systemctl, Path(root), members, Path(home if home is not None else os.path.expanduser("~")),
                 os.getuid() if uid is None else uid, dict(os.environ if environ is None else environ),
                 tuple(named))
    names = scan.units()
    for path, _unit in scan.logrotate_configs:  # each logrotate run starts from fresh global options
        scan.parse_logrotate(path, {"olddir": None})
    for path in extra_logrotate:
        scan.parse_logrotate(os.path.normpath(os.fspath(path)), {"olddir": None})
    report = {"units_scanned": len(names), "unit_names": names, "logrotate_files": len(scan.logrotate_seen),
              "tolerated_unresolved": sorted(scan.tolerated), "templates_scanned": sorted(scan.templates)}
    logger.info("[CompactionUnits] scanned %d units, %d logrotate files", report["units_scanned"],
                report["logrotate_files"])
    return report
