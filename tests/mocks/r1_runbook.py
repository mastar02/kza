"""The R1 runbook's pinned-archive check, read from the runbook itself and run as written (SC-6a).

sc-release-design v3 epochs.r1_rebuild (PS-13): before W1 the operator runs
``sha256sum "$R1_ARCHIVE"`` against the sha256 pinned in
docs/runbooks/2026-11-13-r1-adoption-window.md and stops on any difference.
The tests run that fenced block with bash, never a copy of it, so a change
of the runbook's check (for instance comparing the commit instead of the
archive bytes, mutant M-6a.2) changes what the rehearsal runs.

Review SC6A-R2: right after ``R1 fijado: OK`` the runbook extracts that same
archive into a new ``mktemp -d`` directory, checks its ``release.json``
commit and only then defines ``$ST``; ``run_check_and_extract`` runs both
blocks as written, in one shell, as the operator does.

Re-review RR-2: the extraction block defines ``PY`` itself, so a new shell
that runs only the two blocks (the runbook's recovery after a lost session)
works. The helper adds nothing to the blocks: it only replaces the value of
that ``PY=`` line with the rehearsal's interpreter, as it replaces
``R1_ARCHIVE``.

Re-review RR-1: day 14 runs in a new session from the installed controller;
``day14_block`` is the block that defines its ``PY``, ``C`` and ``ST``.
"""
from __future__ import annotations

import gzip
import os
from pathlib import Path
import re
import shlex
import subprocess

REPO = Path(__file__).resolve().parents[2]
RUNBOOK = REPO / "docs" / "runbooks" / "2026-11-13-r1-adoption-window.md"
HEADING = "### Archivo R1 fijado (paso obligatorio antes de W1)"
OK = "R1 fijado: OK"
STOP = "R1 distinto del fijado: DETENER"
EXTRACT_HEADING = "### Extraer R1 verificado (justo después de `R1 fijado: OK`)"
EXTRACTED = "R1 extraído: OK"
EXTRACT_STOP = "R1 extraído distinto del fijado: DETENER"
DAY14_HEADING = "## Día 14: retiro Q5 del estado de b0866e8 y sus copias del 2026-10-29"
RUNBOOK_PY = "PY=/home/kza/kza/.venv/bin/python"
_ASSIGNMENT = re.compile(r"^(R1_ARCHIVE|R1_SHA256|R1_COMMIT)=(.*)$")


def runbook_text() -> str:
    return RUNBOOK.read_text()


def _section_block(heading: str, text: str | None) -> str:
    text = runbook_text() if text is None else text
    if text.count(heading) != 1:
        raise AssertionError("the runbook has no single section " + heading)
    section = text.split(heading, 1)[1].split("\n## ", 1)[0].split("\n### ", 1)[0]
    blocks = re.findall(r"```bash\n(.*?)```", section, flags=re.S)
    if len(blocks) != 1:
        raise AssertionError("the section needs exactly one bash block: " + heading)
    return blocks[0]


def check_block(text: str | None = None) -> str:
    """The fenced bash block of the pinned-archive section (exactly one)."""
    return _section_block(HEADING, text)


def extract_block(text: str | None = None) -> str:
    """The fenced bash block of the extraction section (exactly one)."""
    return _section_block(EXTRACT_HEADING, text)


def day14_block(text: str | None = None) -> str:
    """The day-14 section's first bash block: it defines PY, C and ST before any step."""
    text = runbook_text() if text is None else text
    if text.count(DAY14_HEADING) != 1:
        raise AssertionError("the runbook has no single day-14 section")
    section = text.split(DAY14_HEADING, 1)[1].split("\n## ", 1)[0]
    blocks = re.findall(r"```bash\n(.*?)```", section, flags=re.S)
    if not blocks:
        raise AssertionError("the day-14 section has no bash block")
    return blocks[0]


def pinned(text: str | None = None) -> dict:
    """{'R1_ARCHIVE': ..., 'R1_SHA256': ..., 'R1_COMMIT': ...} as the block assigns them."""
    found = {}
    for line in check_block(text).splitlines():
        match = _ASSIGNMENT.match(line)
        if match:
            found[match.group(1)] = match.group(2)
    return found


def _check_lines(archive: Path, sha256: str | None, commit: str | None, text: str | None) -> list[str]:
    lines = []
    for line in check_block(text).splitlines():
        match = _ASSIGNMENT.match(line)
        if match and match.group(1) == "R1_ARCHIVE":
            line = "R1_ARCHIVE=" + shlex.quote(str(archive))
        elif match and match.group(1) == "R1_SHA256" and sha256 is not None:
            line = "R1_SHA256=" + sha256
        elif match and match.group(1) == "R1_COMMIT" and commit is not None:
            line = "R1_COMMIT=" + commit
        lines.append(line)
    return lines


def run_check(archive: Path, *, sha256: str | None = None, commit: str | None = None,
              text: str | None = None) -> subprocess.CompletedProcess:
    """Run the runbook's block with R1_ARCHIVE set to ``archive``.

    ``sha256``/``commit`` replace the pinned values (a rehearsal archive is never
    the pinned one); None keeps the runbook's own pin."""
    lines = _check_lines(archive, sha256, commit, text)
    return subprocess.run(["bash", "-c", "\n".join(lines)], capture_output=True, text=True, timeout=120,
                          stdin=subprocess.DEVNULL)


def run_check_and_extract(archive: Path, *, python: str, tmpdir: Path, sha256: str | None = None,
                          commit: str | None = None, environment: dict | None = None,
                          text: str | None = None) -> tuple[subprocess.CompletedProcess, dict]:
    """Run the check block and then the extraction block, as written, in one new bash.

    ``python`` replaces the value of the extraction block's own ``PY=`` line
    (the server's interpreter); nothing defines ``PY`` before the blocks, as in
    a new operator session. ``tmpdir`` is the ``TMPDIR`` that ``mktemp -d``
    uses; ``environment`` adds variables the shell already had (for instance a
    stale ``R1`` and ``ST``). Stops with 3 when the check does not pass, as the
    runbook does. Returns the process and the ``R1``/``ST`` the shell ends with."""
    extract = extract_block(text).splitlines()
    if extract.count(RUNBOOK_PY) != 1:
        raise AssertionError("the extraction block must define " + RUNBOOK_PY + " exactly once")
    extract = ["PY=" + shlex.quote(python) if line == RUNBOOK_PY else line for line in extract]
    lines = [*_check_lines(archive, sha256, commit, text),
             '[ "$?" -eq 0 ] || exit 3', *extract,
             "status=$?", 'printf \'\\nR1=%s\\nST=%s\\n\' "$R1" "$ST"', 'exit "$status"']
    env = dict(os.environ, TMPDIR=str(tmpdir), **(environment or {}))
    env.pop("PY", None)
    result = subprocess.run(["bash", "-c", "\n".join(lines)], capture_output=True, text=True, timeout=120,
                            stdin=subprocess.DEVNULL, env=env)
    tail = dict(line.split("=", 1) for line in result.stdout.splitlines()[-2:] if "=" in line)
    return result, tail


def regzipped(source: Path, destination: Path) -> Path:
    """A rebuilt-but-different archive of the same commit: identical tar bytes, another gzip header.

    Its release.json (git_commit, every file digest) is unchanged, so only a
    comparison of the archive's own sha256 tells it from the source."""
    tar = gzip.decompress(source.read_bytes())
    destination.write_bytes(gzip.compress(tar, mtime=1))
    assert destination.read_bytes() != source.read_bytes()
    return destination
