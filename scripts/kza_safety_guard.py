#!/usr/bin/env python3
"""Cross-runtime PreToolUse safety guard for KZA (Claude, Codex, Muse).

Evaluates shell commands and file write/edit targets against KZA hard
invariants and git safety rules before execution. Exits 0 on allow, or emits
a structured deny payload on stdout and exits 2 on block.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
from dataclasses import dataclass
from typing import Any

SHELL_TOOL_NAMES = frozenset(
    {
        "bash",
        "shell",
        "shell_command",
        "run_shell",
        "run_command",
        "exec_command",
        "terminal",
        "computer_bash",
        "muse_shell",
        "local_shell",
    }
)

WRITE_TOOL_NAMES = frozenset(
    {
        "write",
        "edit",
        "multiedit",
        "write_file",
        "edit_file",
        "create_file",
        "replace_in_files",
        "apply_patch",
        "notebookedit",
        "str_replace_editor",
    }
)

# Paths that must never be edited or written by KZA development agents.
FORBIDDEN_WRITE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"(^|/)home/kza(/|$)"),
        "Prohibido editar o escribir bajo la cuenta/despliegue de producción /home/kza.",
    ),
    (
        re.compile(r"(^|/)\.config/herdr-core/templates/(tu|gsm)-"),
        "Prohibido modificar templates de otros proyectos (tu-*, gsm-*).",
    ),
    (
        re.compile(r"(^|/)\.herdr/worktrees/(thermouy|gsm|game-set-match)", re.IGNORECASE),
        "Prohibido tocar worktrees de otros proyectos.",
    ),
    (
        re.compile(r"(^|/)Documents/Proyectos/(ThermoUY|Game-Set-Match-uy|homelab)(/|$)"),
        "Prohibido modificar archivos fuera de KZA en proyectos vecinos.",
    ),
    (
        re.compile(r"(^|/)\.env(?:\.(?!example$|docker\.example$)[A-Za-z0-9_.-]+)?$"),
        "Prohibido escribir secretos directamente en archivos .env.",
    ),
)

# Command-level KZA hard prohibitions (checked across full command text and subshells).
FORBIDDEN_CMD_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"/home/kza(?:/|\b)"),
        "Prohibido operar sobre la cuenta/despliegue de producción /home/kza.",
    ),
    (
        re.compile(r"\b(?:sudo\s+-u\s+kza|su\s+(?:-\s+)?kza)\b"),
        "Prohibido escalar o cambiar a la cuenta administrativa/runtime kza en tareas de desarrollo.",
    ),
    (
        re.compile(r"\bsystemctl\b[^\n;&|]*(?:restart|stop|start|reload|enable|disable)\b[^\n;&|]*\bkza\b"),
        "Prohibido reiniciar o alterar servicios de producción kza sin GO de release.",
    ),
    (
        re.compile(r"\b(?:rm|mv|cp|ln|chmod|chown|sed|tee)\b[^\n;&|]*\.config/herdr-core/templates/(?:tu|gsm)-"),
        "Prohibido tocar templates de otros proyectos (tu-*, gsm-*).",
    ),
    (
        re.compile(r"\b(?:rm|mv|git\s+worktree\s+remove)\b[^\n;&|]*\.herdr/worktrees/(?:thermouy|gsm|game-set-match)", re.IGNORECASE),
        "Prohibido tocar worktrees de otros proyectos.",
    ),
    (
        re.compile(r"\bsecurity\s+add-generic-password\b(?![^\n]*\s-w\s+\S)"),
        "Prohibido usar `security add-generic-password -w` interactivo (BSD getpass trunca a 128 bytes); usar `read -rs`.",
    ),
)

_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$")
_SHELL_NAMES = frozenset({"sh", "bash", "zsh", "dash", "ksh", "fish"})


@dataclass(frozen=True)
class GuardResult:
    """Result of evaluating a tool invocation."""

    allowed: bool
    reason: str = ""
    rule_id: str = ""


def _split_shell_segments(command: str) -> list[str]:
    """Split a possibly multiline shell script into logical command segments."""
    segments: list[str] = []
    buf: list[str] = []
    in_single = False
    in_double = False
    escaped = False
    i = 0
    n = len(command)
    while i < n:
        ch = command[i]
        if escaped:
            buf.append(ch)
            escaped = False
            i += 1
            continue
        if ch == "\\" and not in_single:
            if i + 1 < n and command[i + 1] == "\n":
                # Line continuation collapses into whitespace.
                buf.append(" ")
                i += 2
                continue
            buf.append(ch)
            escaped = True
            i += 1
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            buf.append(ch)
            i += 1
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            buf.append(ch)
            i += 1
            continue
        if not in_single and not in_double:
            if ch in ("\n", ";"):
                seg = "".join(buf).strip()
                if seg:
                    segments.append(seg)
                buf = []
                i += 1
                continue
            if ch in ("&", "|"):
                seg = "".join(buf).strip()
                if seg:
                    segments.append(seg)
                buf = []
                if i + 1 < n and command[i + 1] == ch:
                    i += 2
                else:
                    i += 1
                continue
        buf.append(ch)
        i += 1
    tail = "".join(buf).strip()
    if tail:
        segments.append(tail)
    return segments


def _tokenize(segment: str) -> list[str]:
    """Tokenize a shell segment tolerating unmatched quotes."""
    try:
        return shlex.split(segment, posix=True)
    except ValueError:
        return segment.split()


def _unwrap_prefixes(tokens: list[str]) -> list[str]:
    """Strip env assignments and wrapper builtins (`command`, `env`, `builtin`, `exec`, `nohup`, `time`)."""
    idx = 0
    n = len(tokens)
    while idx < n:
        tok = tokens[idx]
        if _ENV_ASSIGN_RE.match(tok):
            idx += 1
            continue
        base = os.path.basename(tok)
        if base in ("command", "builtin", "exec", "nohup", "time"):
            idx += 1
            while idx < n and tokens[idx].startswith("-"):
                idx += 1
            continue
        if base == "env":
            idx += 1
            while idx < n and (tokens[idx].startswith("-") or _ENV_ASSIGN_RE.match(tokens[idx])):
                idx += 1
            continue
        break
    return tokens[idx:]


def _extract_nested_commands(tokens: list[str]) -> list[str]:
    """Extract subshell strings passed via `sh/bash/zsh -c ...` or `eval ...`."""
    nested: list[str] = []
    if not tokens:
        return nested
    cmd = os.path.basename(tokens[0])
    if cmd in _SHELL_NAMES:
        i = 1
        while i < len(tokens):
            tok = tokens[i]
            if tok == "-c" or (tok.startswith("-") and not tok.startswith("--") and "c" in tok[1:]):
                if i + 1 < len(tokens):
                    nested.append(tokens[i + 1])
                break
            i += 1
    elif cmd == "eval" and len(tokens) > 1:
        nested.append(" ".join(tokens[1:]))
    return nested


def _current_branch_is_main(cwd: str | None = None) -> bool:
    """Fast local check (no subprocess) whether the current git HEAD points to `main`."""
    cur = os.path.abspath(cwd or os.getcwd())
    while True:
        dot_git = os.path.join(cur, ".git")
        head_path = ""
        if os.path.isdir(dot_git):
            head_path = os.path.join(dot_git, "HEAD")
        elif os.path.isfile(dot_git):
            try:
                with open(dot_git, encoding="utf-8") as fh:
                    first = fh.readline().strip()
                if first.startswith("gitdir:"):
                    gdir = first.split(":", 1)[1].strip()
                    if not os.path.isabs(gdir):
                        gdir = os.path.join(cur, gdir)
                    head_path = os.path.join(gdir, "HEAD")
            except OSError:
                return False
        if head_path:
            try:
                with open(head_path, encoding="utf-8") as fh:
                    return fh.readline().strip() == "ref: refs/heads/main"
            except OSError:
                return False
        parent = os.path.dirname(cur)
        if parent == cur:
            return False
        cur = parent


def _check_git_tokens(tokens: list[str]) -> GuardResult:
    """Validate git subcommand invocations."""
    # Parse global git options like -C <dir>, -c k=v, --git-dir=...
    i = 1
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        if tok in ("-C", "-c", "--git-dir", "--work-tree", "--namespace", "--super-prefix"):
            i += 2
            continue
        if tok.startswith("-"):
            i += 1
            continue
        break
    if i >= n:
        return GuardResult(True)

    sub = tokens[i]
    args = tokens[i + 1 :]

    if sub == "stash":
        return GuardResult(
            False,
            "Bloqueado: `git stash` está prohibido en KZA (riesgo de ocultar/perder trabajo entre worktrees).",
            "git-stash",
        )

    if sub == "add":
        for a in args:
            if a in ("-A", "--all", ".", "./") or (
                a.startswith("-") and not a.startswith("--") and "A" in a[1:]
            ):
                return GuardResult(
                    False,
                    "Bloqueado: `git add -A` / `git add .` está prohibido; stagear archivos explícitos.",
                    "git-add-all",
                )
            norm = a[2:] if a.startswith("./") else a
            is_env_secret = (
                norm == ".env"
                or (
                    norm.startswith(".env.")
                    and not norm.endswith(".example")
                )
            )
            if is_env_secret or norm.startswith(("data/", "models/")):
                if norm != "models/.gitkeep":
                    return GuardResult(
                        False,
                        f"Bloqueado: prohibido stagear datos/secretos/modelos (`{a}`).",
                        "git-add-sensitive",
                    )

    if sub == "push":
        positionals = [a for a in args if not a.startswith("-")]
        is_force = any(
            a in ("-f", "--force", "--force-with-lease") or a.startswith("--force-with-lease=")
            for a in args
        )
        for p in positionals:
            ref_target = p.split(":")[-1]
            if ref_target in ("main", "refs/heads/main"):
                return GuardResult(
                    False,
                    "Bloqueado: prohibido hacer `git push` directo a `main`.",
                    "git-push-main",
                )
            if is_force and ref_target in (
                "recovery/production-2026-10-25",
                "refs/heads/recovery/production-2026-10-25",
            ):
                return GuardResult(
                    False,
                    "Bloqueado: prohibido `git push --force` sobre `recovery/production-2026-10-25`.",
                    "git-force-push-protected",
                )
        if len(positionals) <= 1 and _current_branch_is_main():
            return GuardResult(
                False,
                "Bloqueado: `git push` implícito desde rama `main` está prohibido.",
                "git-push-main-implicit",
            )
        if not positionals and is_force:
            return GuardResult(
                False,
                "Bloqueado: `git push --force` implícito sin ref explícita.",
                "git-force-push-implicit",
            )

    if sub == "reset" and any(a == "--hard" for a in args):
        return GuardResult(
            False,
            "Bloqueado: `git reset --hard` es destructivo; requiere autorización explícita.",
            "git-reset-hard",
        )

    if sub == "clean":
        has_force = any(
            a == "--force" or (a.startswith("-") and not a.startswith("--") and "f" in a[1:])
            for a in args
        )
        if has_force and not any(a in ("-n", "--dry-run") for a in args):
            return GuardResult(
                False,
                "Bloqueado: `git clean -f` destructivo está prohibido.",
                "git-clean-force",
            )

    return GuardResult(True)


def check_shell_command(command: str, _depth: int = 0) -> GuardResult:
    """Recursively check a shell command (including multiline and `-c` subshells)."""
    if not command or not command.strip():
        return GuardResult(True)
    if _depth > 4:
        return GuardResult(
            False,
            "Bloqueado: anidamiento excesivo de subshells (-c/eval).",
            "subshell-depth",
        )

    for pattern, reason in FORBIDDEN_CMD_PATTERNS:
        if pattern.search(command):
            return GuardResult(False, reason, "kza-invariant-cmd")

    for segment in _split_shell_segments(command):
        # Also inspect $(...) or `...` command substitutions inside the segment.
        for sub_match in re.finditer(r"\$\(([^()]+)\)|`([^`]+)`", segment):
            inner = sub_match.group(1) or sub_match.group(2) or ""
            res = check_shell_command(inner, _depth + 1)
            if not res.allowed:
                return res

        tokens = _unwrap_prefixes(_tokenize(segment))
        if not tokens:
            continue

        # Handle `xargs git ...`
        if os.path.basename(tokens[0]) == "xargs":
            rest = [t for t in tokens[1:] if not t.startswith("-")]
            if rest and os.path.basename(rest[0]) == "git":
                res = _check_git_tokens(rest)
                if not res.allowed:
                    return res

        if os.path.basename(tokens[0]) == "git":
            res = _check_git_tokens(tokens)
            if not res.allowed:
                return res

        for nested in _extract_nested_commands(tokens):
            res = check_shell_command(nested, _depth + 1)
            if not res.allowed:
                return res

    return GuardResult(True)


def check_write_path(path: str) -> GuardResult:
    """Check whether a file path is forbidden for write/edit operations."""
    if not path:
        return GuardResult(True)
    normalized = path.replace("\\", "/")
    for pattern, reason in FORBIDDEN_WRITE_PATTERNS:
        if pattern.search(normalized):
            return GuardResult(False, reason, "kza-invariant-path")
    return GuardResult(True)


def _extract_command_from_input(tool_input: Any) -> str:
    """Extract shell command text from Claude, Codex, or Muse tool input dicts."""
    if isinstance(tool_input, str):
        return tool_input
    if not isinstance(tool_input, dict):
        return ""
    for key in ("command", "CommandLine", "cmd", "script", "input", "code"):
        val = tool_input.get(key)
        if isinstance(val, str) and val.strip():
            return val
        if isinstance(val, list) and all(isinstance(x, str) for x in val):
            return " ".join(shlex.quote(x) for x in val)
    return ""


def _extract_paths_from_input(tool_input: Any) -> list[str]:
    """Extract target file paths from write/edit tool inputs."""
    if not isinstance(tool_input, dict):
        return []
    paths: list[str] = []
    for key in ("file_path", "path", "Path", "target_file", "filename"):
        val = tool_input.get(key)
        if isinstance(val, str) and val.strip():
            paths.append(val)
    edits = tool_input.get("edits")
    if isinstance(edits, list):
        for item in edits:
            if isinstance(item, dict):
                for key in ("file_path", "path"):
                    val = item.get(key)
                    if isinstance(val, str) and val.strip():
                        paths.append(val)
    patch = tool_input.get("patch") or tool_input.get("input")
    if isinstance(patch, str) and "*** Update File:" in patch:
        for m in re.finditer(r"^\*\*\*\s+(?:Update|Add|Delete)\s+File:\s+(.+)$", patch, re.M):
            paths.append(m.group(1).strip())
    return paths


def evaluate_hook_payload(payload: dict[str, Any]) -> GuardResult:
    """Evaluate a PreToolUse JSON payload from Claude, Codex, or Muse."""
    tool_name = str(payload.get("tool_name") or payload.get("tool") or "").strip()
    tool_input = payload.get("tool_input")
    if tool_input is None:
        tool_input = payload.get("input") or payload.get("arguments") or {}

    lowered = tool_name.lower()
    if lowered in SHELL_TOOL_NAMES or "bash" in lowered or "shell" in lowered or "command" in lowered:
        cmd = _extract_command_from_input(tool_input)
        return check_shell_command(cmd)

    if lowered in WRITE_TOOL_NAMES or any(k in lowered for k in ("write", "edit", "patch")):
        for path in _extract_paths_from_input(tool_input):
            res = check_write_path(path)
            if not res.allowed:
                return res
        return GuardResult(True)

    return GuardResult(True)


def main() -> int:
    """Read PreToolUse payload from stdin and enforce KZA guardrails."""
    raw = sys.stdin.read()
    if not raw.strip():
        return 0
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        # Fail-open on malformed non-JSON stdin to avoid false locks on unknown hook envelopes.
        return 0
    if not isinstance(payload, dict):
        return 0

    result = evaluate_hook_payload(payload)
    if result.allowed:
        return 0

    response = {
        "decision": "block",
        "reason": result.reason,
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": f"[kza-safety-guard:{result.rule_id}] {result.reason}",
        },
    }
    sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
    sys.stderr.write(f"[kza-safety-guard:{result.rule_id}] {result.reason}\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
