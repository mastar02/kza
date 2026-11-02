"""Bookkeeping must not weaken frozen acceptance or dependency gates."""

import copy
import json
from pathlib import Path

import pytest

from tools.roadmap_status import contract_hashes, ready_tasks, validate_manifest


@pytest.fixture
def manifest():
    return json.loads(Path("docs/plans/2026-11-02-autonomous-roadmap.json").read_text())


def test_committed_manifest_preserves_confirmed_contract(manifest):
    assert validate_manifest(manifest) == []
    assert contract_hashes(manifest) == (
        "3ccb8e7c84f9e1ae0c3f74ac515484c16aeaf80c9fe9e31c8c940ac15d33366d",
        "65065c19b6795fecade28c9dd580d2d3e4d90c5a42fd2c0410ba31bf0d0d02ef",
    )


@pytest.mark.parametrize("section,key", [
    ("criteria", "target"), ("criteria", "minimum_evidence"),
    ("confirmed_decisions", "decision"), ("integration_portfolio", "name"),
])
def test_contract_edit_is_detected(manifest, section, key):
    manifest[section][0][key] = "changed requirement"
    assert any("contract mismatch" in error for error in validate_manifest(manifest))


def test_tracking_does_not_change_frozen_targets(manifest):
    before = contract_hashes(manifest)
    manifest["criteria"][0].update(status="accepted", evidence=["private evidence"])
    assert contract_hashes(manifest) == before


def test_phase_cannot_skip_required_work(manifest):
    manifest["phases"][0].update(status="verified_local", evidence=["report"])
    manifest["tasks"][0].update(status="verified_local", evidence=[])
    manifest["tasks"][1].update(status="planned", evidence=[])
    assert "T001: verified state needs evidence" in validate_manifest(manifest)
    assert "P0: required task not locally verified" in validate_manifest(manifest)


def test_wait_must_preserve_next_action(manifest):
    manifest["tasks"][0].update(status="waiting_external", next_action=None)
    assert "T001: external wait needs a concrete next action" in validate_manifest(manifest)


def test_dependencies_require_local_verification(manifest):
    for phase in manifest["phases"]:
        phase["status"] = "planned"
    for task in manifest["tasks"]:
        task["status"] = "planned"
    assert ready_tasks(manifest) == ["T001", "T002", "T003", "T004"]
    manifest["phases"][0]["status"] = "implemented"
    assert "T010" not in ready_tasks(manifest)
    manifest["phases"][0]["status"] = "verified_local"
    manifest["phases"][0]["evidence"] = ["local report"]
    for task in manifest["tasks"]:
        if task["phase"] == "P0":
            task.update(status="verified_local", evidence=["local report"])
    assert "T010" in ready_tasks(manifest)


def test_field_is_not_ready_without_concrete_authorization(manifest):
    for phase in manifest["phases"]:
        phase["status"] = "verified_local"
    for task in manifest["tasks"]:
        task["status"] = "planned"
    assert "T070" not in ready_tasks(manifest)
    assert "T080" not in ready_tasks(manifest)


def test_duplicate_and_dangling_references_rejected(manifest):
    manifest["tasks"].append(copy.deepcopy(manifest["tasks"][0]))
    manifest["tasks"][0]["criteria"] = ["C99"]
    errors = validate_manifest(manifest)
    assert "tasks: duplicate identifiers" in errors
    assert "T001: unknown criterion" in errors


def test_completion_cannot_be_inferred_from_task_counts(manifest):
    manifest["status"] = "accepted"
    assert "completion: criterion not accepted" in validate_manifest(manifest)
    assert "completion: area not accepted" in validate_manifest(manifest)


def test_field_acceptance_requires_human_results_before_p8(manifest):
    manifest["field_authorization_evidence"] = "private concrete window authorization"
    for phase in manifest["phases"]:
        if phase["id"] != "P8":
            phase.update(status="verified_local", evidence=["report"])
    for task in manifest["tasks"]:
        if task["phase"] != "P8":
            task.update(status="verified_local", evidence=["report"])
        else:
            task["status"] = "planned"
    phase7 = next(phase for phase in manifest["phases"] if phase["id"] == "P7")
    phase7["status"] = "accepted"
    assert "P7: human-gated task not accepted" in validate_manifest(manifest)
    assert ready_tasks(manifest) == []
    for task in manifest["tasks"]:
        if task["phase"] == "P7":
            task["status"] = "accepted"
    assert validate_manifest(manifest) == []
    assert any(task["phase"] == "P8" and task["id"] in ready_tasks(manifest)
               for task in manifest["tasks"])
