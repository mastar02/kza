"""A production-shaped legacy layout written by b0866e8's own code (U3b-0e2 tests).

``b0866e8_tree`` extracts src/ and config/ of b0866e8 from this repository
(``git archive``); ``build_legacy`` lays out the 63ea8d3 container, the
b0866e8 slot state, the legacy checkout's models and the legacy metrics
directory under one root (every link relative), then runs
tests/mocks/adoption_legacy_child.py with b0866e8's code only. Afterwards it
adds what production has and b0866e8's API never writes on demand: a hot
rollback journal (lists.db), a .env, lora adapters under the models root and
a pre-policy metrics file.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys

from scripts.adoption import SourceLayout

REPOSITORY = Path(__file__).resolve().parents[2]
CHILD = Path(__file__).with_name("adoption_legacy_child.py")
LEGACY = "b0866e8"
OLD_EVENTS = "8558ff1^"  # the last commit whose EventLogger created the pre-policy events table
B_SLOT = "release_20261028T145442Z_b0866e8"
C_SLOT = "release_20261029T174830709997Z_63ea8d3"
UNCOMMITTED = "KZA-UNCOMMITTED-LIST"


def available() -> bool:
    """Whether this clone holds the legacy commits."""
    return all(subprocess.run(["git", "cat-file", "-e", commit + "^{commit}"], cwd=REPOSITORY,
                              capture_output=True).returncode == 0 for commit in (LEGACY, "8558ff1"))


def b0866e8_tree(destination: Path) -> Path:
    """Extract b0866e8's src/ and config/ into ``destination``."""
    destination.mkdir(parents=True)
    archive = subprocess.run(["git", "archive", "--format=tar", LEGACY, "src", "config"], cwd=REPOSITORY,
                             check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", str(destination)], input=archive, check=True)
    return destination


def legacy_events_ddl() -> str:
    source = subprocess.run(["git", "show", OLD_EVENTS + ":src/analytics/event_logger.py"], cwd=REPOSITORY,
                            check=True, capture_output=True, text=True).stdout
    return re.search(r"CREATE TABLE IF NOT EXISTS events \(.*?\n\s*\)", source, re.S).group(0)


def layout_of(root: Path) -> SourceLayout:
    """The SourceLayout of a fixture root."""
    root = Path(root)
    return SourceLayout(container=root / "releases/slots" / C_SLOT / "state",
                        legacy_state=root / "releases/slots" / B_SLOT / "state",
                        models_root=root / "kza/models", legacy_checkout=root / "kza",
                        metrics_root=root / "logs", code_index=root / "code-index")


def _hot_journal(database: Path) -> None:
    """Leave ``database`` as a SIGKILL would: uncommitted pages on disk plus its hot -journal."""
    work = database.with_name("work-" + database.name)
    shutil.copyfile(database, work)
    db = sqlite3.connect(work, isolation_level=None)
    db.execute("PRAGMA journal_mode=DELETE")
    db.execute("PRAGMA cache_size=1")
    db.execute("BEGIN IMMEDIATE")
    db.executemany("INSERT INTO lists(id,name,owner_type,owner_id,created_at,updated_at) VALUES (?,?,?,?,?,?)",
                   [(f"list-{index}", UNCOMMITTED + "x" * 400, "user", "user_a", 1.0, 1.0) for index in range(400)])
    shutil.copyfile(work, database)
    shutil.copyfile(str(work) + "-journal", str(database) + "-journal")
    db.execute("ROLLBACK")
    db.close()
    work.unlink()


def build_legacy(root: Path, tree: Path, *, adapters: bool = False) -> SourceLayout:
    """Write the legacy layout under ``root`` with b0866e8's code from ``tree``."""
    layout = layout_of(root)
    state = layout.legacy_state
    for directory in (state / "config", state / "data", state / "logs", layout.container, layout.models_root,
                      layout.metrics_root):
        directory.mkdir(parents=True)
    shutil.copyfile(tree / "config" / "settings.yaml", state / "config" / "settings.yaml")
    (state / ".env").write_text("HOME_ASSISTANT_URL=http://192.168.1.100:8123\nCONFIG_PATH=config/settings.yaml\n")
    (state / "logs" / "voice_assistant.log").write_text("legacy log\n")
    for name in ("config", "data", ".env", "logs"):
        os.symlink(Path("..") / ".." / B_SLOT / "state" / name, layout.container / name)
    os.symlink(Path("..") / ".." / ".." / ".." / "kza" / "models", layout.container / "models")
    (layout.models_root / "whisper-v3-turbo").mkdir()
    (layout.models_root / "whisper-v3-turbo" / "model.bin").write_bytes(b"model")
    result = subprocess.run([sys.executable, "-I", "-B", str(CHILD), str(tree), str(layout.container),
                             str(layout.metrics_root), legacy_events_ddl()],
                            capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-4000:])
    report = json.loads(result.stdout.strip().splitlines()[-1])
    (layout.metrics_root / "kza-metrics.jsonl").write_text('{"pre_policy": 1}\n')
    _hot_journal(state / "data" / "lists.db")
    if adapters:
        adapter = layout.models_root / "lora_adapters" / "custom_adapter"
        adapter.mkdir(parents=True)
        (adapter / "adapter_model.safetensors").write_bytes(b"lora")
    (root / "report.json").write_text(json.dumps(report))
    return layout


def report_of(root: Path) -> dict:
    return json.loads((Path(root) / "report.json").read_text())


def tree_bytes(root: Path) -> dict:
    """Every entry under ``root`` (links as their text) with its bytes' sha256, for byte-identity checks."""
    import hashlib

    result = {}
    for path in sorted(Path(root).rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            result[relative] = "link:" + os.readlink(path)
        elif path.is_file():
            result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            result[relative] = "dir"
    return result
