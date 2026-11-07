"""Public planning over real, individually byte/schema-bound SQLite images.

The existing codec fixture supplies no controller ancestry or execution grant.
Real materialization is covered by the unchanged compiled O/N/R/F node, whose
additional assertions inspect its genuine closed snapshots and final binding.
"""
from contextlib import closing
from dataclasses import replace
import hashlib
from pathlib import Path
import sqlite3

import pytest

from scripts.stage_runtime import AdmissionController
from src.memory import payload_derivation as payload
from src.memory import state_snapshot as snapshots
from src.memory.fact_ledger import MemoryReviewer
from tests.mocks.memory_sources import memory_source
from tests.unit.memory.test_payload_derivation import (
    DIAGNOSTICS, NOW, PRIVACY, _image, _layout, _snapshot_codec,
)


INDEX_NAME = "idx_erasure_target_participant_id"
DROP_INDEX = "DROP INDEX " + INDEX_NAME
CREATE_INDEX = "CREATE INDEX " + INDEX_NAME + " ON erasure_targets(participant, id)"


def _statements(*values: str) -> tuple:
    return tuple((value, ()) for value in values)


def _bytes(*images) -> dict:
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for image in images
        for root in (image.snapshot_root, Path(image.source_binding["state_root"]))
        for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("old_index,new_index", [
    pytest.param(False, False, id="same-legacy"),
    pytest.param(True, True, id="same-current"),
    pytest.param(False, True, id="known-addition"),
])
def test_public_plan_accepts_only_the_declared_compatible_history(tmp_path, monkeypatch, old_index, new_index):
    ordinary = _image(tmp_path, "O", statements=() if old_index else _statements(DROP_INDEX))
    latest = _image(tmp_path, "N", statements=() if new_index else _statements(DROP_INDEX))
    destination = tmp_path / "R"
    before = _bytes(ordinary, latest)
    monkeypatch.setattr(payload.importlib.metadata, "version", lambda name: pytest.fail("SDK lookup during plan"))
    first = payload.plan_payload_cut(ordinary, latest, now=NOW, layout=_layout(destination))
    second = payload.plan_payload_cut(ordinary, latest, now=NOW, layout=_layout(destination))
    assert first == second and first.plan_sha256 == second.plan_sha256
    assert first.coverage == tuple(sorted(payload.FAMILIES))
    assert first.source_manifest_hashes == (ordinary.manifest_sha256, latest.manifest_sha256)
    assert snapshots.verify_snapshot(ordinary) and snapshots.verify_snapshot(latest)
    assert _bytes(ordinary, latest) == before and not destination.exists()


@pytest.mark.parametrize("old_changes,new_changes", [
    pytest.param((), (DROP_INDEX,), id="reverse-removal"),
    pytest.param((DROP_INDEX,), (DROP_INDEX,
        "CREATE INDEX unrecognized_lookup ON erasure_targets(participant, id)"), id="unknown-index"),
    pytest.param((DROP_INDEX,), (DROP_INDEX,
        "CREATE UNIQUE INDEX " + INDEX_NAME + " ON erasure_targets(participant, id)"), id="unique"),
    pytest.param((DROP_INDEX,), (DROP_INDEX,
        CREATE_INDEX + " WHERE version=1"), id="partial"),
    pytest.param((DROP_INDEX,), (DROP_INDEX,
        "CREATE INDEX " + INDEX_NAME + " ON erasure_targets(participant COLLATE NOCASE, id)"), id="collation"),
    pytest.param((DROP_INDEX,), (DROP_INDEX,
        "CREATE INDEX " + INDEX_NAME + " ON erasure_targets(id, participant)"), id="column-order"),
    pytest.param((DROP_INDEX,), (DROP_INDEX,
        "CREATE INDEX " + INDEX_NAME + " ON facts(id)"), id="different-table"),
    pytest.param((DROP_INDEX,), (DROP_INDEX,
        "CREATE INDEX " + INDEX_NAME + " ON erasure_targets(participant, id ASC)"), id="noncanonical-sql"),
    pytest.param((DROP_INDEX,), ("CREATE INDEX unrelated_extra ON facts(id)",), id="canonical-plus-extra"),
    pytest.param((DROP_INDEX, "CREATE UNIQUE INDEX " + INDEX_NAME + " ON erasure_targets(participant, id)"),
        (), id="replace-existing-definition"),
    pytest.param((DROP_INDEX,), ("PRAGMA application_id=17",), id="application-id"),
    pytest.param((DROP_INDEX,), ("PRAGMA user_version=1",), id="user-version"),
])
def test_public_plan_rejects_other_schema_transitions_before_destination(tmp_path, old_changes, new_changes):
    ordinary = _image(tmp_path, "O", statements=_statements(*old_changes))
    latest = _image(tmp_path, "N", statements=_statements(*new_changes))
    destination = tmp_path / "R"
    before = _bytes(ordinary, latest)
    # Each image binds its own real schema; this tests the compatibility gate,
    # not a manufactured corrupt snapshot that fails before reaching it.
    assert snapshots.verify_snapshot(ordinary) and snapshots.verify_snapshot(latest)
    with pytest.raises(payload.PayloadDerivationError, match="^payload_database_schema_changed$"):
        payload.plan_payload_cut(ordinary, latest, now=NOW, layout=_layout(destination))
    assert not destination.exists() and _bytes(ordinary, latest) == before


def test_public_plan_uses_declared_privacy_role_with_a_different_locator(tmp_path):
    renamed = "data/graph/authority-state.sqlite3"
    images = []
    for name, changes in (("O", _statements(DROP_INDEX)), ("N", ())):
        image = _image(tmp_path, name, statements=changes)
        source = Path(image.source_binding["state_root"])
        (source / PRIVACY).rename(source / renamed)
        images.append(_snapshot_codec(source, tmp_path / (name + "-renamed"), (renamed, DIAGNOSTICS)))
    destination = tmp_path / "R"
    original = _layout(destination)
    layout = replace(original, privacy_relative=renamed, database_relocations=tuple(
        payload.DatabaseRelocation(renamed, renamed, renamed) if item.latest_relative == PRIVACY else item
        for item in original.database_relocations))
    before = _bytes(*images)
    plan = payload.plan_payload_cut(*images, now=NOW, layout=layout)
    assert plan.layout.privacy_relative == renamed
    assert _bytes(*images) == before and not destination.exists()


def test_public_plan_refuses_the_same_addition_outside_declared_privacy(tmp_path):
    images = []
    for name in ("O", "N"):
        image = _image(tmp_path, name)
        source = Path(image.source_binding["state_root"])
        with closing(sqlite3.connect(source / DIAGNOSTICS)) as database, database:
            database.execute("CREATE TABLE erasure_targets (id TEXT PRIMARY KEY, participant TEXT, version INTEGER, descriptor TEXT)")
            if name == "N":
                database.execute(CREATE_INDEX)
        images.append(_snapshot_codec(source, tmp_path / (name + "-other-store"), (PRIVACY, DIAGNOSTICS)))
    destination = tmp_path / "R"
    before = _bytes(*images)
    with pytest.raises(payload.PayloadDerivationError, match="^payload_database_schema_changed$"):
        payload.plan_payload_cut(*images, now=NOW, layout=_layout(destination))
    assert not destination.exists() and _bytes(*images) == before


@pytest.mark.parametrize("tampered", ["ordinary", "latest"])
def test_known_addition_does_not_bypass_either_source_binding(tmp_path, tampered):
    ordinary = _image(tmp_path, "O", statements=_statements(DROP_INDEX))
    latest = _image(tmp_path, "N")
    chosen = ordinary if tampered == "ordinary" else latest
    before = _bytes(ordinary, latest)
    chosen.source_binding["schemas"][PRIVACY] = "0" * 64
    destination = tmp_path / "R"
    with pytest.raises(ValueError):
        payload.plan_payload_cut(ordinary, latest, now=NOW, layout=_layout(destination))
    assert not destination.exists() and _bytes(ordinary, latest) == before


def test_known_addition_keeps_latest_revocation_in_public_plan(tmp_path):
    source = memory_source(now=NOW - 1)

    def seed(ledger):
        ledger.store("Synthetic ordinary fact", "fact", .8, (source,))

    def revoke(ledger):
        seed(ledger)
        ledger.delete_conversation(source.conversation_id, MemoryReviewer("synthetic-reviewer"))

    ordinary = _image(tmp_path, "O", statements=_statements(DROP_INDEX), seed=seed)
    latest = _image(tmp_path, "N", seed=revoke)
    destination = tmp_path / "R"
    before = _bytes(ordinary, latest)
    plan = payload.plan_payload_cut(ordinary, latest, now=NOW, layout=_layout(destination))
    facts = next(cut for cut in plan.tables if cut.database == PRIVACY and cut.table == "facts")
    deleted, = facts.selected_rows
    assert deleted.source == "latest" and deleted.values["state"] == "deleted"
    assert deleted.values["content"] is None
    assert _bytes(ordinary, latest) == before and not destination.exists()


def test_known_addition_never_grants_materialization_to_codec_images(tmp_path):
    ordinary = _image(tmp_path, "O", statements=_statements(DROP_INDEX))
    latest = _image(tmp_path, "N")
    destination = tmp_path / "R"
    plan = payload.plan_payload_cut(ordinary, latest, now=NOW, layout=_layout(destination))
    with AdmissionController(tmp_path / "stage") as controller:
        with pytest.raises((ValueError, FileNotFoundError)):
            payload.materialize_derived_state(plan, destination, controller=controller)
    assert not destination.exists()
