"""Read-only checks of frozen roadmap contracts and dependency readiness."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


TASK_STATES = frozenset({
    "planned", "in_progress", "implemented", "verified_local",
    "waiting_external", "accepted", "deferred_by_user",
})
LOCAL_STATES = frozenset({"verified_local", "accepted"})
DEFAULT_MANIFEST = Path("docs/plans/2026-11-02-autonomous-roadmap.json")


def contract_hashes(manifest: dict) -> tuple[str, str]:
    """Return the original Q15 target and Q14/Q15/Q19 profile digests."""
    def digest(value: dict) -> str:
        encoded = json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha256(encoded).hexdigest()

    targets = digest({
        "criteria": [
            {key: value for key, value in item.items() if key not in {"status", "evidence"}}
            for item in manifest["criteria"]
        ],
        "stratification": manifest["stratification"],
    })
    profile = digest({
        "targets": targets,
        "decisions": manifest["confirmed_decisions"],
        "integrations": manifest["integration_portfolio"],
    })
    return targets, profile


def validate_manifest(manifest: dict) -> list[str]:
    """Check hashes, references and required evidence; not evidence coverage.

    A reference still requires independent review. This structural check cannot
    prove a command's coverage, a real source or physical acceptance.
    """
    errors = []
    targets, profile = contract_hashes(manifest)
    for key, actual in [("acceptance_targets_sha256", targets), ("profile_sha256", profile)]:
        if manifest[key] != actual:
            errors.append(f"{key}: frozen contract mismatch")
    groups = {key: {item["id"]: item for item in manifest[key]}
              for key in ("tasks", "criteria", "phases")}
    for key, indexed in groups.items():
        if len(indexed) != len(manifest[key]):
            errors.append(f"{key}: duplicate identifiers")
    for task in manifest["tasks"]:
        label = task["id"]
        if task["phase"] not in groups["phases"]:
            errors.append(f"{label}: unknown phase")
        if set(task["criteria"]) - groups["criteria"].keys():
            errors.append(f"{label}: unknown criterion")
        if task["status"] not in TASK_STATES:
            errors.append(f"{label}: unknown task state")
        if task["status"] in LOCAL_STATES and not task["evidence"]:
            errors.append(f"{label}: verified state needs evidence")
        if task["status"] == "waiting_external" and not task["next_action"]:
            errors.append(f"{label}: external wait needs a concrete next action")
    for phase in manifest["phases"]:
        label = phase["id"]
        if phase["status"] not in TASK_STATES:
            errors.append(f"{label}: unknown phase state")
        if set(phase["depends_on"]) - groups["phases"].keys():
            errors.append(f"{label}: unknown dependency")
        if phase["status"] not in LOCAL_STATES:
            continue
        if not phase["evidence"]:
            errors.append(f"{label}: verified state needs evidence")
        required = [task for task in manifest["tasks"]
                    if task["phase"] == label and task["inclusion"] == "required"]
        if any(task["status"] not in LOCAL_STATES for task in required):
            errors.append(f"{label}: required task not locally verified")
        if phase["status"] == "accepted" and any(
            task["human_gate"] and task["status"] != "accepted" for task in required
        ):
            errors.append(f"{label}: human-gated task not accepted")
        states = {"accepted"} if phase["dependency_level"] == "field_accepted" else LOCAL_STATES
        if any(groups["phases"].get(dep, {}).get("status") not in states
               for dep in phase["depends_on"]):
            errors.append(f"{label}: dependency not verified at required level")
        if label in {"P7", "P8"} and not manifest.get("field_authorization_evidence"):
            errors.append(f"{label}: concrete field authorization evidence missing")
    for criterion in manifest["criteria"]:
        if criterion["status"] == "accepted" and not criterion["evidence"]:
            errors.append(f"{criterion['id']}: accepted criterion needs evidence")
    if manifest["status"] == "accepted":
        if any(item["status"] != "accepted" for item in manifest["criteria"]):
            errors.append("completion: criterion not accepted")
        if any(item["status"] != "accepted" for item in manifest["areas"]):
            errors.append("completion: area not accepted")
        if groups["phases"].get("P8", {}).get("status") != "accepted":
            errors.append("completion: final handover not accepted")
    return errors


def ready_tasks(manifest: dict) -> list[str]:
    """List pending required tasks whose phase dependencies are verified."""
    if validate_manifest(manifest):
        return []
    phases = {phase["id"]: phase for phase in manifest["phases"]}
    ready = []
    for task in manifest["tasks"]:
        phase = phases[task["phase"]]
        states = {"accepted"} if phase["dependency_level"] == "field_accepted" else LOCAL_STATES
        if task["status"] not in {"planned", "in_progress"} or task["inclusion"] != "required":
            continue
        if phase["id"] in {"P7", "P8"} and not manifest.get("field_authorization_evidence"):
            continue
        if all(phases[dep]["status"] in states for dep in phase["depends_on"]):
            ready.append(task["id"])
    return ready


def main() -> int:
    """Print a bounded summary without private data and return check status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    errors = validate_manifest(manifest)
    print(json.dumps({
        "profile": manifest["profile_id"], "errors": errors,
        "ready_tasks": ready_tasks(manifest) if not errors else [],
        "criteria_accepted": sum(c["status"] == "accepted" for c in manifest["criteria"]),
        "criteria_total": len(manifest["criteria"]),
    }, ensure_ascii=False, indent=2))
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
