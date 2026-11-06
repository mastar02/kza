"""Projection codecs over actual collectors, without admission capabilities."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from src.memory import manual_components as manual
from src.memory.erasure_journal import ErasureParticipant
from src.training.stt_correction_collector import STTCorrectionCollector
from src.wakeword.clip_store import CollectorKind
from tests.mocks.conversations import graph


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def identity(path):
    info = path.stat()
    return {"device": info.st_dev, "inode": info.st_ino}


@pytest.fixture
async def corpus(tmp_path):
    root = tmp_path / "N"
    async with graph(root) as g:
        original = await g.add(user_input="Private manual input", assistant_response="Private original output")
        views = await g.collector.get_turns()
        try:
            view = next(item for item in views if item.id == original)
            assert await g.collector.mark_response(view, "corrected", "Private latest correction",
                                                   sources=(g.base.source(),))
        finally:
            for view in views:
                view.clear()
        views = await g.collector.get_turns()
        try:
            marked, = (view.id for view in views if view.quality.value == "corrected")
        finally:
            for view in views:
                view.clear()
        marked_cursor = max(row["sequence"] for row in await g.base.rows())
        coparent = await g.add(user_input="Private co-parent input", assistant_response="Private co-parent output")
        exported = await g.collector.export_for_training(str(root / "exports"), only_marked=False)
        assert exported is not None
        try:
            mixed_export = exported.id
        finally:
            exported.clear()
        linked_parent = await g.add(user_input="Private linked input", assistant_response="Private linked output")
        exported = await g.collector.export_for_training(str(root / "exports"), only_marked=False,
                                                         after_sequence=marked_cursor)
        assert exported is not None
        try:
            linked_export = exported.id
        finally:
            exported.clear()
        ordinary_cursor = max(row["sequence"] for row in await g.base.rows())
        ordinary = await g.add(user_input="Independent ordinary input", assistant_response="Independent ordinary output")
        exported = await g.collector.export_for_training(str(root / "exports"), only_marked=False,
                                                         after_sequence=ordinary_cursor)
        assert exported is not None
        try:
            ordinary_export = exported.id
        finally:
            exported.clear()
        target = await g.store._work.run(g.store.register, root / "corrections", CollectorKind.STT_CORRECTIONS)
        correction = STTCorrectionCollector(target.root, store=g.store, target=target, close_timeout_s=.03)
        stt = await correction.add_correction(np.full(160, .25, dtype=np.float32),
            "Private original STT", "Private corrected STT", sources=(g.base.source(),),
            correction_sources=(g.base.source(),))
        assert stt is not None

        def read():
            with g.privacy.ledger._transaction() as db:
                tables = {name: [dict(row) for row in db.execute("SELECT * FROM " + name)]
                          for name in manual.TABLES}
            return tables, g.privacy.ledger.erasures.targets(ErasureParticipant.COLLECTORS)

        tables, targets = await g.privacy.acall(read)
    assert g.privacy.resource_status()["close_phase"] == "complete"
    assert not g.privacy.resource_status()["writer_lease"]
    # Opaque codec correlation only; it is not a controller-issued binding.
    tag = digest({"unit": "manual projection only"})
    owners = {item["id"]: manual.CollectorOwner(item["id"], item["descriptor"]["kind"],
        Path(item["descriptor"]["root"]), root, identity(root), tag) for item in targets}
    return {"root": root, "tables": tables, "owners": owners, "marked": marked, "coparent": coparent,
        "linked_parent": linked_parent, "mixed_export": mixed_export, "linked_export": linked_export,
        "ordinary": ordinary, "ordinary_export": ordinary_export, "stt": stt}


def capture(corpus):
    return manual.capture_manual_components(corpus["root"], {"privacy.db": corpus["tables"]}, corpus["owners"])


def protected(corpus):
    return {corpus[key] for key in ("marked", "coparent", "linked_parent", "mixed_export", "linked_export", "stt")}


def relocated(corpus, tmp_path):
    """A projection codec input, not a productive restore or admission claim."""
    root = tmp_path / "R"
    shutil.copytree(corpus["root"], root)
    owners, mapping = {}, {}
    for old, owner in corpus["owners"].items():
        new_root = root / owner.root.relative_to(corpus["root"])
        descriptor = {"kind": owner.kind, "root": str(new_root)}
        encoded = json.dumps(descriptor, sort_keys=True, separators=(",", ":"), allow_nan=False)
        new = hashlib.sha256(json.dumps(["collectors", 1, encoded]).encode()).hexdigest()
        owners[new] = replace(owner, target_id=new, root=new_root, state_root=root,
                              state_identity=identity(root), binding_id=digest({"projection": "R"}))
        mapping[old] = new
    tables = deepcopy(corpus["tables"])
    for row in tables["wake_clips_v1"]:
        row["target_id"] = mapping[row["target_id"]]
    candidate = manual.capture_manual_components(root, {"privacy.db": tables}, owners)
    used = {row["target_id"] for row in corpus["tables"]["wake_clips_v1"] if row["id"] in protected(corpus)}
    return candidate, {old: mapping[old] for old in used}


async def test_actual_manual_export_closure_includes_shared_export_and_all_coparents(corpus):
    from src.memory.payload_derivation import _protected_collector_records

    expected = protected(corpus)
    assert manual.protected_collector_records(corpus["tables"]) == expected
    snapshot_tables = {name: {"rows": rows} for name, rows in corpus["tables"].items()}
    assert _protected_collector_records(snapshot_tables) == expected
    assert {corpus["ordinary"], corpus["ordinary_export"]}.isdisjoint(expected)


async def test_projection_hashes_exact_actual_rows_edges_and_bytes_without_payload(corpus):
    result = capture(corpus)
    store = result["stores"]["privacy.db"]
    assert set(store["clips"]) == protected(corpus)
    for row in corpus["tables"]["wake_clips_v1"]:
        if row["id"] in protected(corpus):
            projected = store["clips"][row["id"]]
            assert projected["row_sha256"] == digest(row)
            assert projected["relocatable_row_sha256"] == digest({k: v for k, v in row.items() if k != "target_id"})
            path = Path(projected["locator"]["root"]) / projected["file"]["relative_file"]
            assert projected["file"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
            assert projected["file"]["size"] == path.stat().st_size
    for edge in store["edges"]:
        assert edge["row_sha256"] == digest({k: edge[k] for k in ("export_id", "parent_id")})
    assert b"Private" not in canonical(result)
    manual.verify_manual_components(result, deepcopy(result))


async def test_only_an_explicit_typed_target_relocation_can_compare_copied_components(corpus, tmp_path):
    latest = capture(corpus)
    candidate, mapping = relocated(corpus, tmp_path)
    assert manual.equivalent_manual_components(latest, candidate, target_relocations=mapping)
    assert not manual.equivalent_manual_components(latest, candidate, target_relocations={})
    wrong = dict(mapping)
    wrong[next(iter(wrong))] = "f" * 64
    assert not manual.equivalent_manual_components(latest, candidate, target_relocations=wrong)
    with pytest.raises(manual.ManualComponentError, match="closed_state_changed"):
        manual.verify_manual_components(candidate, latest)


async def test_explicit_qualified_consolidation_does_not_require_a_target_bijection(corpus, tmp_path):
    candidate, mapping = relocated(corpus, tmp_path)
    delegated_state = tmp_path / "O"
    shutil.copytree(corpus["root"], delegated_state)
    row = next(row for row in corpus["tables"]["wake_clips_v1"] if row["id"] == corpus["coparent"])
    original = corpus["owners"][row["target_id"]]
    delegated_root = delegated_state / original.root.relative_to(corpus["root"])
    descriptor = {"kind": original.kind, "root": str(delegated_root)}
    encoded = json.dumps(descriptor, sort_keys=True, separators=(",", ":"), allow_nan=False)
    delegated_id = hashlib.sha256(json.dumps(["collectors", 1, encoded]).encode()).hexdigest()
    corpus["owners"][delegated_id] = replace(original, target_id=delegated_id, root=delegated_root,
        state_root=delegated_state, state_identity=identity(delegated_state), binding_id=digest({"projection": "O"}))
    mapping[delegated_id] = mapping[row["target_id"]]
    row["target_id"] = delegated_id
    # Both physical sources and the destination are known fixture data. The
    # mapping is a comparison input, not the controller's delegation capability.
    assert manual.equivalent_manual_components(capture(corpus), candidate, target_relocations=mapping)


async def test_same_target_cannot_hide_a_replaced_file_behind_identical_bytes(corpus):
    before = capture(corpus)
    clip = before["stores"]["privacy.db"]["clips"][corpus["mixed_export"]]
    path = Path(clip["locator"]["root"]) / clip["file"]["relative_file"]
    replacement = path.with_suffix(".replacement")
    replacement.write_bytes(path.read_bytes())
    replacement.replace(path)
    after = capture(corpus)
    actual = after["stores"]["privacy.db"]["clips"][corpus["mixed_export"]]
    assert actual["file"]["sha256"] == clip["file"]["sha256"]
    assert actual["file"]["identity"] != clip["file"]["identity"]
    assert not manual.equivalent_manual_components(before, after, target_relocations={})
    with pytest.raises(manual.ManualComponentError, match="closed_state_changed"):
        manual.verify_manual_components(after, before)


@pytest.mark.parametrize("field", ["quality", "corrected_text", "used", "sources_json", "expires_at", "edge"])
async def test_latest_manual_decisions_sources_deadlines_and_edges_are_exact(corpus, field):
    before = capture(corpus)
    tables = corpus["tables"]
    if field in {"corrected_text", "used"}:
        row = next(row for row in tables["stt_corrections_v1"] if row["id"] == corpus["stt"])
        row[field] = "Changed correction" if field == "corrected_text" else 1
    elif field == "quality":
        next(row for row in tables["conversation_turns_v1"] if row["id"] == corpus["marked"])[field] = "good"
    elif field == "edge":
        tables["conversation_export_parents_v1"] = [row for row in tables["conversation_export_parents_v1"]
            if not (row["export_id"] == corpus["mixed_export"] and row["parent_id"] == corpus["coparent"])]
    else:
        row = next(row for row in tables["wake_clips_v1"] if row["id"] == corpus["marked"])
        row[field] = "[]" if field == "sources_json" else row[field] - 1
    assert not manual.equivalent_manual_components(before, capture(corpus), target_relocations={})


@pytest.mark.parametrize("change", ["contents", "absent"])
async def test_actual_file_bytes_or_absence_cannot_borrow_prior_witness(corpus, change):
    before = capture(corpus)
    clip = before["stores"]["privacy.db"]["clips"][corpus["mixed_export"]]
    path = Path(clip["locator"]["root"]) / clip["file"]["relative_file"]
    if change == "contents":
        path.write_bytes(path.read_bytes() + b"\n")
    else:
        path.unlink()
    after = capture(corpus)
    assert not manual.equivalent_manual_components(before, after, target_relocations={})
    if change == "absent":
        actual = after["stores"]["privacy.db"]["clips"][corpus["mixed_export"]]["file"]
        assert actual == {"relative_file": clip["file"]["relative_file"], "status": "absent",
                          "sha256": None, "size": None, "identity": None}


@pytest.mark.parametrize("invalid", ["unknown_owner", "wrong_kind", "off_root", "symlink", "hardlink", "unknown_column", "orphan"])
async def test_unknown_ambiguous_or_redirected_component_fails_closed(corpus, tmp_path, invalid):
    row = next(row for row in corpus["tables"]["wake_clips_v1"] if row["id"] == corpus["mixed_export"])
    owner = corpus["owners"][row["target_id"]]
    path = owner.root / "policy_v1" / (row["id"] + ".jsonl")
    if invalid == "unknown_owner":
        corpus["owners"].pop(row["target_id"])
    elif invalid == "wrong_kind":
        corpus["owners"][row["target_id"]] = replace(owner, kind="conversations")
    elif invalid == "off_root":
        corpus["owners"][row["target_id"]] = replace(owner, root=tmp_path / "unowned")
    elif invalid == "unknown_column":
        row["uninterpreted_decision"] = "unknown"
    elif invalid == "orphan":
        corpus["tables"]["conversation_turns_v1"] = [row for row in corpus["tables"]["conversation_turns_v1"]
                                                     if row["id"] != corpus["coparent"]]
    elif invalid == "symlink":
        moved = tmp_path / "unowned-file"
        path.rename(moved)
        path.symlink_to(moved)
    else:
        (tmp_path / "second-link").hardlink_to(path)
    with pytest.raises(manual.ManualComponentError):
        capture(corpus)


@pytest.mark.parametrize("malformed", [{}, {"schema_version": True}, {"schema_version": 2}])
def test_unknown_projection_is_never_an_equal_empty_ack(malformed):
    with pytest.raises(manual.ManualComponentError):
        manual.verify_manual_components(malformed, deepcopy(malformed))
