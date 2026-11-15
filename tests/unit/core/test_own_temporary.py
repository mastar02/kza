"""Own mkstemp temporaries: cleanup unlinks only the own pinned inode (review F10).

Each publisher below used to unlink its temporary *name* in a ``finally``, so a
file substituted at that name was deleted. The sites now share
``src.core.own_temporary`` (the by-path scripts keep a stdlib copy); every
site test plants a foreign file at the own name after the real
``os.replace``/``os.link`` and requires it to survive cleanup.
"""
from __future__ import annotations

import errno
import json
import os
from pathlib import Path

import pytest

from scripts import managed_launcher as launcher
from scripts import stage_runtime as stage
from src.core import own_temporary as module
from src.core.own_temporary import own_temporary, remove_own_temporary

FOREIGN = b"a foreign file substituted at the own temporary name"


def _plant(name: str) -> None:
    descriptor = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(FOREIGN)


@pytest.fixture
def substitute_after(monkeypatch):
    """After the real publishing call, a foreign file takes the own temporary's name.

    ``os.replace`` frees the name, ``os.link`` leaves the own inode there (it
    is then unlinked by the substituting party). Returns the planted names.
    """
    planted: list[str] = []

    def install(call: str, prefix: str) -> list[str]:
        real = getattr(os, call)

        def substituting(source, destination, *args, **kwargs):
            result = real(source, destination, *args, **kwargs)
            name = os.fspath(source)
            if os.path.basename(name).startswith(prefix):
                if call == "link":
                    os.unlink(name)
                _plant(name)
                planted.append(name)
            return result

        monkeypatch.setattr(os, call, substituting)
        return planted

    return install


def _survived(planted: list[str]) -> None:
    assert len(planted) >= 1
    for name in planted:
        assert Path(name).read_bytes() == FOREIGN


# ----------------------------------------------------------------------
# The helper itself
# ----------------------------------------------------------------------

def test_a_substitute_after_the_own_inode_was_unlinked_is_retained(tmp_path):
    """The real reuse window: the own name is unlinked and recreated while pinned.

    Without the pin, ext4 hands the freed number to the next allocation at
    once, so the substitute would carry the own (st_dev, st_ino). While the
    pin is held it cannot (asserted on the real filesystem), and cleanup
    keeps the substitute.
    """
    with own_temporary(tmp_path, prefix=".own-") as (descriptor, name):
        os.close(descriptor)
        own = os.lstat(name)
        os.unlink(name)
        _plant(name)
        substitute = os.lstat(name)
        assert (substitute.st_dev, substitute.st_ino) != (own.st_dev, own.st_ino)
    assert Path(name).read_bytes() == FOREIGN


def test_the_own_temporary_is_removed_when_the_body_fails(tmp_path):
    with pytest.raises(RuntimeError, match="body failed"):
        with own_temporary(tmp_path, prefix=".own-") as (descriptor, name):
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(b"partial")
            raise RuntimeError("body failed")
    assert not os.path.lexists(name)
    assert list(tmp_path.iterdir()) == []


def test_the_pin_is_closed_after_cleanup(tmp_path, monkeypatch):
    duplicated = []
    real_dup = os.dup

    def recording(descriptor):
        result = real_dup(descriptor)
        duplicated.append(result)
        return result

    monkeypatch.setattr(module.os, "dup", recording)
    with own_temporary(tmp_path, prefix=".own-") as (descriptor, name):
        os.close(descriptor)
        os.unlink(name)
    assert len(duplicated) == 1
    with pytest.raises(OSError) as error:
        os.fstat(duplicated[0])
    assert error.value.errno == errno.EBADF


@pytest.fixture
def failing_dup(monkeypatch, tmp_path):
    """os.dup of an own ``.own-`` temporary fails (EMFILE); records the closes of those descriptors."""
    closed = []
    real_close, real_dup = os.close, os.dup

    def own(descriptor):
        try:
            held = os.fstat(descriptor)
        except OSError:
            return False
        return any(os.lstat(name).st_ino == held.st_ino for name in tmp_path.glob(".own-*"))

    refused = set()

    def failing(descriptor):
        if own(descriptor):
            refused.add(descriptor)
            raise OSError(errno.EMFILE, "synthetic descriptor exhaustion")
        return real_dup(descriptor)

    def recording_close(descriptor):
        if descriptor in refused:
            closed.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(os, "dup", failing)
    monkeypatch.setattr(os, "close", recording_close)
    return closed


def test_a_failed_pin_removes_the_own_temporary_through_its_mkstemp_descriptor(tmp_path, failing_dup):
    """Review R7: the open mkstemp descriptor is itself a valid pin; nothing is left behind."""
    with pytest.raises(OSError, match="synthetic descriptor exhaustion"):
        with own_temporary(tmp_path, prefix=".own-"):
            raise AssertionError("the body must not run unpinned")
    assert len(failing_dup) == 1
    assert list(tmp_path.iterdir()) == []


def test_the_stage_copy_removes_its_own_temporary_when_the_pin_fails(tmp_path, failing_dup):
    with pytest.raises(OSError, match="synthetic descriptor exhaustion"):
        with stage.own_temporary(tmp_path, prefix=".own-"):
            raise AssertionError("the body must not run unpinned")
    assert len(failing_dup) == 1
    assert list(tmp_path.iterdir()) == []


def test_diagnostic_projection_removes_its_stage_when_the_pin_fails(tmp_path, monkeypatch):
    from contextlib import closing
    from src.obs import diagnostic_store
    from src.obs.diagnostic_store import DiagnosticStore
    from src.obs.metrics_emitter import project_diagnostic
    path = tmp_path / "diagnostics.jsonl"
    real_dup = os.dup

    def failing(descriptor):
        held = os.fstat(descriptor)
        if any(os.lstat(own).st_ino == held.st_ino for own in tmp_path.glob(".diag-*.tmp")):
            raise OSError(errno.EMFILE, "synthetic descriptor exhaustion")
        return real_dup(descriptor)

    monkeypatch.setattr(diagnostic_store.os, "dup", failing)
    with closing(DiagnosticStore(path, clock=lambda: 2_000_000_000.0, retention_days=1, rotate_bytes=10**6,
                                 project=project_diagnostic)) as store:
        with pytest.raises(OSError, match="synthetic descriptor exhaustion"):
            store.prune()
        monkeypatch.setattr(diagnostic_store.os, "dup", real_dup)
    assert not list(tmp_path.glob(".diag-*.tmp"))


def test_an_unpinned_name_is_never_unlinked(tmp_path):
    name = tmp_path / ".own-unpinned"
    _plant(str(name))
    assert remove_own_temporary(name, None) is False
    assert name.read_bytes() == FOREIGN


def test_a_symlink_at_the_own_name_is_retained(tmp_path):
    with own_temporary(tmp_path, prefix=".own-") as (descriptor, name):
        os.close(descriptor)
        os.unlink(name)
        os.symlink(tmp_path / "elsewhere", name)
    assert os.path.islink(name)


# ----------------------------------------------------------------------
# Every former path-only cleanup site
# ----------------------------------------------------------------------

def test_stage_runtime_atomic_record_keeps_a_substitute(tmp_path, substitute_after):
    planted = substitute_after("replace", ".admission-")
    stage._atomic_record(tmp_path / "head.json", {"revision": 1})
    assert json.loads((tmp_path / "head.json").read_text()) == {"revision": 1}
    _survived(planted)


def test_stage_runtime_immutable_record_keeps_a_substitute(tmp_path, substitute_after):
    planted = substitute_after("link", ".admission-")
    stage._atomic_record(tmp_path / "record.json", {"revision": 1}, immutable=True)
    assert json.loads((tmp_path / "record.json").read_text()) == {"revision": 1}
    _survived(planted)


def test_stage_runtime_helper_matches_the_shared_one(tmp_path):
    with stage.own_temporary(tmp_path, prefix=".own-") as (descriptor, name):
        os.close(descriptor)
        own = os.lstat(name)
        os.unlink(name)
        _plant(name)
        assert (os.lstat(name).st_dev, os.lstat(name).st_ino) != (own.st_dev, own.st_ino)
    assert Path(name).read_bytes() == FOREIGN
    with pytest.raises(RuntimeError):
        with stage.own_temporary(tmp_path, prefix=".gone-") as (descriptor, gone):
            os.close(descriptor)
            raise RuntimeError("failed")
    assert not os.path.lexists(gone)


def test_managed_launcher_text_keeps_a_substitute(tmp_path, substitute_after):
    planted = substitute_after("replace", ".launcher-")
    target = tmp_path / "unit.service"
    launcher._write_text_atomic(target, "[Service]\n", mode=0o640)
    assert target.read_text() == "[Service]\n" and target.stat().st_mode & 0o777 == 0o640
    _survived(planted)


def test_copy_catalog_head_keeps_a_substitute(tmp_path, substitute_after, monkeypatch):
    from src.memory import copy_catalog
    # Record validation is not under test; only the head's temporary cleanup.
    monkeypatch.setattr(copy_catalog, "_record_shape", lambda record: None)
    monkeypatch.setattr(copy_catalog, "_initial", lambda record: None)
    planted = substitute_after("replace", ".head-")
    head = copy_catalog._append(tmp_path, {"copy_id": "0" * 32}, None)
    assert head["revision"] == 1
    assert json.loads((tmp_path / "head.json").read_text()) == head
    _survived(planted)


def test_copy_observation_head_keeps_a_substitute(tmp_path, substitute_after, monkeypatch):
    from src.memory import copy_catalog
    from src.memory import copy_observation
    intent = {"observation_id": "1" * 32}
    copy_catalog._write_new(tmp_path / "intent.json", copy_catalog._json(intent) + b"\n")
    # Intent validation is not under test; only the head's temporary cleanup.
    monkeypatch.setattr(copy_observation, "_shape_intent", lambda value: None)
    planted = substitute_after("replace", ".head-")
    head = copy_observation._publish(tmp_path, intent, None, None)
    assert head["revision"] == 1
    _survived(planted)


def test_payload_derivation_file_keeps_a_substitute(tmp_path, substitute_after):
    from src.memory import payload_derivation
    planted = substitute_after("replace", ".payload-")
    target = tmp_path.resolve() / "payload.json"
    payload_derivation._atomic_file(target, b"{}\n")
    assert target.read_bytes() == b"{}\n"
    _survived(planted)


def test_cooldown_snapshot_keeps_a_substitute(tmp_path, substitute_after):
    from src.llm.cooldown import CooldownManager
    manager = CooldownManager(persistence_path=tmp_path / "cooldown.json")
    planted = substitute_after("replace", ".cooldown-")
    manager._save_to_disk(strict=True)
    assert json.loads((tmp_path / "cooldown.json").read_text()) == {"states": []}
    _survived(planted)


def test_context_snapshot_keeps_a_substitute(tmp_path, substitute_after):
    from src.orchestrator.context_manager import UserContext
    from src.orchestrator.context_persister import ContextPersister
    from tests.mocks.memory_sources import memory_source
    persister = ContextPersister(base_path=tmp_path / "contexts")
    planted = substitute_after("replace", ".snapshot-")
    persister.save(UserContext(user_id="u1", user_name="Juan", compacted_summary="Resumen.",
                               summary_sources=(memory_source(),)))
    assert persister.load("u1")["user_id"] == "u1"
    _survived(planted)


def test_audio_health_snapshot_keeps_a_substitute(tmp_path, substitute_after):
    from src.monitoring.audio_health import write_audio_health
    planted = substitute_after("replace", "tmp")
    target = tmp_path / "audio_health.json"
    write_audio_health(str(target), rooms={"cocina": (980.0, True)}, now_wall=1000.0)
    assert json.loads(target.read_text())["rooms"]["cocina"]["ever"] is True
    assert target.stat().st_mode & 0o777 == 0o644
    _survived(planted)


def test_diagnostic_projection_keeps_a_substitute_and_drops_its_staged_row(tmp_path, substitute_after):
    from contextlib import closing
    from src.obs.diagnostic_store import DiagnosticStore
    from src.obs.metrics_emitter import project_diagnostic
    now = [2_000_000_000.0]
    path = tmp_path / "diagnostics.jsonl"
    planted = substitute_after("replace", ".diag-")
    with closing(DiagnosticStore(path, clock=lambda: now[0], retention_days=1, rotate_bytes=10**6,
                                 project=project_diagnostic)) as store:
        store.prune()
    _survived(planted)
    # The rows are gone, so no later recovery unlinks the foreign names by path.
    import sqlite3
    with closing(sqlite3.connect(path.with_suffix(path.suffix + ".sqlite3"))) as db:
        assert db.execute("SELECT COUNT(*) FROM staged").fetchone()[0] == 0
