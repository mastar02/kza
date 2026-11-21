"""Audit the frozen KZA profile and the reviewed P6/P7 evidence offline."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from hashlib import file_digest, sha256
from pathlib import Path

from src.evaluation.freeze_manifest import canonical_sha256
from tools.p7_field_protocol import (
    P6_DOCUMENT,
    P7_CRITERIA,
    PROTOCOL,
    ROOMS,
    Document,
    EvidenceError,
    array,
    calibration_clips,
    candidate,
    count,
    exact_keys,
    flags,
    hash_value,
    identifiers,
    indexed,
    instant,
    mapping,
    number,
    private_path,
    read_json,
    require,
    text_value,
    validate_field,
    validate_release_review,
)
from tools.roadmap_status import DEFAULT_MANIFEST, contract_hashes, validate_manifest

TARGETS_SHA256 = "3ccb8e7c84f9e1ae0c3f74ac515484c16aeaf80c9fe9e31c8c940ac15d33366d"
PROFILE_SHA256 = "65065c19b6795fecade28c9dd580d2d3e4d90c5a42fd2c0410ba31bf0d0d02ef"
# State and evidence may advance. Deleting requirements or changing their scope cannot create a pass.
SCOPE_SHA256 = "066941241d2e2bf24e9b2537f19e4220c340379f42e03e6bf07c3b29490bcf4f"
VERIFIED = {"verified_local", "accepted"}
P6_TASKS = ("T060", "T061", "T062", "T063", "T064")
HITOS = tuple(f"H{i}" for i in range(7))
CRITERIA = tuple(f"C{i:02}" for i in range(1, 22))
REQUIRED_SOURCES = {
    "C01": ("T060",),
    "C02": ("T060", "T063"),
    "C03": ("T060", "T061"),
    "C04": ("T063", "T071", "T075"),
    "C05": ("T060", "T063", "T064", "T075"),
    "C06": ("T080",),
    "C07": ("T073",),
    "C08": ("T073",),
    "C09": ("T073",),
    "C10": ("T073",),
    "C11": ("T073",),
    "C12": ("T062", "T071", "T072"),
    "C13": ("T062", "T072"),
    "C14": ("T062", "T072"),
    "C15": ("T062", "T072"),
    "C16": ("T062", "T072"),
    "C17": ("T062", "T072"),
    "C18": ("T062", "T071", "T072"),
    "C19": ("T061", "T070", "T074"),
    "C20": ("T061", "T070", "T074"),
    "C21": ("T070", "T075"),
}
INPUT_ERRORS = (ValueError, TypeError, KeyError, OverflowError, OSError)


def _scope(roadmap: Document) -> tuple[dict[str, Document], ...]:
    require(
        contract_hashes(roadmap)
        == (roadmap["acceptance_targets_sha256"], roadmap["profile_sha256"])
        == (TARGETS_SHA256, PROFILE_SHA256),
        "frozen acceptance_targets_sha256 or profile_sha256 mismatch",
    )
    tasks = indexed(roadmap.get("tasks"), "roadmap tasks")
    phases = indexed(roadmap.get("phases"), "roadmap phases")
    findings = indexed(roadmap.get("finding_trace"), "roadmap findings")
    areas = indexed(roadmap.get("areas"), "roadmap areas", "name")
    criteria = indexed(roadmap.get("criteria"), "roadmap criteria")
    exact_keys(criteria, CRITERIA, "roadmap criteria")
    scope = {
        "tasks": {
            key: [
                row[field]
                for field in (
                    "phase",
                    "criteria",
                    "inclusion",
                    "human_gate",
                    "optional",
                )
            ]
            for key, row in tasks.items()
        },
        "phases": {
            key: [row[field] for field in ("maps_to", "depends_on", "dependency_level")]
            for key, row in phases.items()
        },
        "findings": {key: row["tasks"] for key, row in findings.items()},
        "areas": {key: [row["criteria"], row["target"]] for key, row in areas.items()},
        "completion_policy": roadmap["completion_policy"],
    }
    require(canonical_sha256(scope) == SCOPE_SHA256, "roadmap acceptance scope changed")
    return tasks, phases, findings, areas, criteria


class ArtifactStore:
    """Verify evidence receipts in a private content-addressed directory."""

    def __init__(self, directory: Path) -> None:
        self.directory = private_path(directory)
        self.verified: set[str] = set()

    def verify(self, value: object, label: str) -> str:
        """Check that a nonempty regular artifact has the referenced SHA-256."""
        digest = hash_value(value, label)
        if digest not in self.verified:
            path = private_path(self.directory / digest)
            require(
                path.parent == self.directory and path.is_file(),
                f"{label}: missing evidence artifact",
            )
            with path.open("rb") as contents:
                require(
                    path.stat().st_size > 0
                    and file_digest(contents, "sha256").hexdigest() == digest,
                    f"{label}: evidence artifact hash mismatch",
                )
            self.verified.add(digest)
        return digest

    def record(self, value: object, expected: Document, label: str) -> None:
        """Join a decision receipt to the candidate and approved content."""
        digest = self.verify(value, label)
        record, actual = read_json(self.directory / digest)
        require(
            actual == digest
            and all(record.get(key) == item for key, item in expected.items()),
            f"{label}: recorded decision does not match the audited content",
        )


def _local(
    task: str, data: Document, identity: Document, artifacts: ArtifactStore
) -> None:
    require(
        data.get("task") == task
        and data.get("kind") == "local"
        and candidate(data.get("candidate")) == identity,
        f"{task}: candidate-bound local evidence required",
    )
    artifacts.verify(data.get("evidence_sha256"), task)
    if task == "T060":
        families = indexed(
            data.get("factory_families"), "C01 factory families", "family"
        )
        exact_keys(
            families,
            identifiers(data.get("included_families"), "C01 included families"),
            "C01 factory scope",
        )
        require(bool(families), "C01 missing factory coverage")
        for row in families.values():
            tested = count(row.get("mandatory"), "C01 mandatory cases", 1)
            require(
                count(row.get("passed"), "C01 passed") == tested,
                "C01 real factory contract failed",
            )
            flags(
                row,
                "C01",
                "real_factory",
                "actor_policy_persistence_effect_response",
                "restart_and_faults_checked",
            )
        count(data.get("negative_cases"), "C02 negative cases", 200)
        require(
            count(data.get("invariant_violations"), "C02 invariant violations") == 0,
            "C02 invariant violation",
        )
        flags(
            data,
            "T060",
            "required_suite_passed",
            "no_required_xfail",
            "tier2_passed",
            "compiled_solo_passed",
            "new_warnings_reviewed",
        )
        h2 = mapping(data.get("h2"), "C03 H2")
        count(h2.get("repetitions"), "C03 repetitions", 10)
        require(
            number(h2.get("heartbeat_ms"), "C03 heartbeat") == 5,
            "C03 heartbeat must be 5ms",
        )
        for key, limit in {
            "delay_p95_ms": 25,
            "delay_max_ms": 75,
            "cancel_ms": 100,
            "simulated_stop_ms": 200,
            "cleanup_ms": 1000,
        }.items():
            number(h2.get(key), f"C03 {key}", limit)
        flags(
            h2,
            "C03",
            "pcm_exact",
            "bounded_queues",
            "long_responses",
            "retained_native_work",
            "producers_200ms",
        )
    elif task == "T061":
        duration = count(data.get("soak_seconds"), "T061 local soak", 72 * 3600)
        series = mapping(data.get("useful_seconds"), "T061 soak series")
        exact_keys(series, ("KZA", *ROOMS), "T061 soak series")
        for key, value in series.items():
            seconds = count(value, f"T061 useful {key}")
            require(
                seconds <= duration and seconds * 1000 >= duration * 999,
                f"T061 {key}: local availability",
            )
        flags(
            data,
            "T061",
            "candidate_frozen",
            "privacy_checks_passed",
            "mute_durable_no_voice_reactivation",
            "late_producer_races_checked",
            "h2_preserved",
            "faults_outside_soak",
            "healthy_inputs_kept_heartbeat",
        )
        faults = mapping(data.get("recovery_seconds"), "T061 recovery seconds")
        exact_keys(faults, ("ha", "ws", "mic"), "T061 recovery types")
        for key, limit in {"ha": 60, "ws": 60, "mic": 180}.items():
            number(faults[key], f"T061 {key} recovery", limit)
    elif task == "T062":
        require(
            data.get("protocol") == PROTOCOL
            and data.get("protocol_sha256")
            == sha256(P6_DOCUMENT.read_bytes()).hexdigest(),
            "T062 p6-v1 protocol hash",
        )
        flags(data, "T062", "evaluator_tests_passed", "independent_reserve_plan_sealed")
    elif task == "T063":
        flags(
            data,
            "T063",
            "builds_identical",
            "dependencies_reproducible",
            "runtime_outside_sources",
            "startup_config_verified",
            "training_disabled_or_accepted",
            "compatible_data_restored",
            "runtime_rollback_verified",
            "no_resurrection",
            "no_action_replay",
        )
        require(
            data.get("rebuilt_archive_sha256") == identity["archive_sha256"],
            "T063 rebuild digest",
        )
    elif task == "T064":
        digest = artifacts.verify(
            data.get("release_review_sha256"), "T064 release-review.json"
        )
        review, actual = read_json(artifacts.directory / digest)
        require(actual == digest, "T064 release review bytes changed")
        validate_release_review(review, identity)
        flags(data, "T064", "release_review_verify_passed", "rollback_script_reviewed")


def _cross_checks(
    rows: dict[str, Document], artifacts: ArtifactStore, roadmap: Document
) -> None:
    field = [rows[task] for task in P7_CRITERIA]
    request = field[0]["request"]
    require(
        all(
            row["request"] == request
            and row["q17_approval"] == field[0]["q17_approval"]
            for row in field
        ),
        "P7 must share a single Q17 decision and request",
    )
    q16 = artifacts.verify(
        field[0]["q16_approval"]["receipt_sha256"], "Q16 human decision"
    )
    q17 = artifacts.verify(
        field[0]["q17_approval"]["receipt_sha256"], "Q17 human decision"
    )
    for question, digest in (("q16", q16), ("q17", q17)):
        decision = {
            key: value
            for key, value in field[0][f"{question}_approval"].items()
            if key != "receipt_sha256"
        }
        artifacts.record(digest, decision, f"{question.upper()} human decision")
    authorization = array(
        roadmap.get("field_authorization_evidence"), "roadmap field authorization"
    )
    require(
        q16 != q17 and {q16, q17} <= set(authorization),
        "distinct Q16/Q17 receipts required in roadmap",
    )
    artifacts.verify(request["q16"]["inventory_sha256"], "Q16 reviewed inventory")
    require(
        request["q17"]["release_review_sha256"]
        == rows["T064"]["release_review_sha256"],
        "P6/P7 release-review.json mismatch",
    )
    calibration, reserve = rows["T071"], rows["T072"]
    for field_name in (
        "acoustic_sha256",
        "frozen_at",
        "reserve_opened_at",
        "evaluation_close",
    ):
        require(
            calibration[field_name] == reserve[field_name],
            f"T071/T072 {field_name} mismatch",
        )
    calibration_rows = indexed(calibration.get("clips"), "T071 calibration", "clip_id")
    combined = indexed(reserve.get("clips"), "T072 inventory", "clip_id")
    same_clips = all(
        key in combined
        and all(combined[key].get(field) == value for field, value in row.items())
        for key, row in calibration_rows.items()
    )
    require(
        same_clips
        and set(calibration_clips(calibration))
        == {key for key, clip in combined.items() if clip["split"] == "calibration"},
        "T071/T072 calibration inventory mismatch",
    )
    adoption, promoted = rows["T075"]["adoption"], rows["T075"]
    adopted_at = instant(adoption["adopted_at"], "T075 adopted_at")
    promotion_at = instant(promoted["observed_at"], "T075 promotion time")
    require(
        all(
            adopted_at <= instant(row["observed_at"], "P7 observation") <= promotion_at
            for row in field
        ),
        "P7 observations must follow adoption and precede promotion",
    )
    require(
        adopted_at <= instant(rows["T074"]["soak"]["start"], "T074 soak start"),
        "T074 soak began before candidate adoption",
    )
    require(
        all(
            adopted_at <= instant(period["start"], "T072 useful interval")
            for period in reserve["wake"]["useful_intervals"]
        ),
        "T072 house clock began before candidate adoption",
    )
    for key, row in rows["T070"]["integrations"].items():
        artifacts.verify(row["observation_sha256"], f"{key} real integration")
    for key, row in promoted["integration_recheck"].items():
        artifacts.verify(row["observation_sha256"], f"{key} real integration recheck")
    browser = rows["T073"]
    artifacts.verify(
        browser["clients"]["iphone"]["recording_sha256"],
        "C10 physical iPhone recording",
    )
    for name in (
        "preregistered_scope_sha256",
        "automated_report_sha256",
        "manual_report_sha256",
    ):
        artifacts.verify(browser["accessibility"][name], f"C07 {name}")
    for fault in rows["T074"]["faults"]:
        artifacts.verify(fault["trace_sha256"], "C20 physical recovery trace")
    artifacts.verify(adoption["receipt_sha256"], "T075 adoption")
    artifacts.verify(promoted["promotion_decision_sha256"], "T075 promotion decision")
    artifacts.verify(promoted["census"]["report_sha256"], "T075 V4 census")
    artifacts.verify(promoted["restore"]["report_sha256"], "T075 compatible restore")
    artifacts.verify(
        promoted["rollback"]["clean_close_receipt_sha256"], "T075 W9 clean close"
    )
    artifacts.verify(promoted["rollback"]["rehearsal_sha256"], "T075 L4 rehearsal")


def _observability(value: object) -> None:
    obs = mapping(value, "C06 observability")
    components = indexed(obs.get("components"), "C06 operational components")
    exact_keys(
        components,
        identifiers(obs.get("included_components"), "C06 operational scope"),
        "C06 component coverage",
    )
    require(bool(components), "C06 operational inventory empty")
    for row in components.values():
        flags(
            row,
            "C06",
            "real_source",
            "actual_timestamp_and_expiry",
            "backend_effect_verified",
        )
    scenarios = identifiers(obs.get("verified_scenarios"), "C06 operational scenarios")
    require(
        {
            "401",
            "403",
            "409",
            "503",
            "timeout",
            "disconnect",
            "reconnect",
            "stale",
            "partial",
            "uncertain",
        }
        <= scenarios,
        "C06 missing operational scenarios",
    )


def audit(roadmap: Document, bundle: Document, directory: Path) -> Document:
    """Return checked areas, H0–H6 and blockers without mutating either input."""
    tasks, phases, findings, areas, criteria = _scope(roadmap)
    artifacts = ArtifactStore(directory)
    blockers = validate_manifest(roadmap)

    def attempt(label: str, action: Callable[[], None]) -> bool:
        try:
            action()
        except EvidenceError as exc:
            blockers.append(f"{label}: {exc}")
        except INPUT_ERRORS as exc:
            blockers.append(
                f"{label}: missing or invalid evidence ({type(exc).__name__})"
            )
        else:
            return True
        return False

    required = {
        key: row
        for key, row in tasks.items()
        if row["inclusion"].startswith("required")
    }
    task_states = {}
    for key, row in required.items():
        eligible = {"accepted"} if row["human_gate"] else VERIFIED
        task_states[key] = row["status"] in eligible
        if not task_states[key]:
            blockers.append(f"{key}: required state not verified")
    open_findings = sorted(
        key
        for key, row in findings.items()
        if set(row["tasks"]) & required.keys() and row["status"] != "closed"
    )
    blockers.extend(f"{key}: finding still open" for key in open_findings)

    identity = candidate(bundle.get("candidate"))
    rows = mapping(bundle.get("tasks"), "bundle task evidence")
    expected = set(P6_TASKS) | P7_CRITERIA.keys()
    require(not rows.keys() - expected, "unexpected P6/P7 task evidence")
    for task in (*P6_TASKS, *P7_CRITERIA):

        def validate_task(task: str = task) -> None:
            data = mapping(rows[task], task)
            if task in P6_TASKS:
                _local(task, data, identity, artifacts)
            else:
                validate_field(task, data)
                require(
                    data["candidate"] == identity, f"{task} evidence candidate mismatch"
                )
                artifacts.verify(data.get("evidence_sha256"), f"{task} field evidence")
            require(
                data["evidence_sha256"]
                in array(tasks[task]["evidence"], f"{task} roadmap evidence"),
                f"{task}: evidence receipt missing from roadmap",
            )

        valid = attempt(task, validate_task)
        task_states[task] = valid and task_states[task]
    field_tasks = (*P7_CRITERIA, "T064")
    connected = all(task_states[task] for task in field_tasks) and attempt(
        "P7 chain",
        lambda: _cross_checks(rows, artifacts, roadmap),
    )
    reviews = mapping(bundle.get("criterion_reviews"), "independent criterion reviews")
    require(not reviews.keys() - criteria.keys(), "unknown criterion review")
    reviewers = set()

    def verify_criterion(key: str) -> None:
        row = mapping(reviews[key], key)
        require(
            criteria[key]["status"] == row.get("verdict") == "accepted",
            "criterion is not accepted",
        )
        flags(row, key, "independent", "original_observations_reviewed")
        reviewer = text_value(row.get("reviewer"), f"{key} reviewer")
        if any(task in P7_CRITERIA for task in REQUIRED_SOURCES[key]):
            require(connected, "P7 evidence/authorization chain incomplete")
            require(
                reviewer != rows["T070"]["request"]["q17"]["operator"],
                "reviewer is the release operator",
            )
        reviewers.add(reviewer)
        review = artifacts.verify(row.get("review_sha256"), key)
        refs = mapping(row.get("sources"), "review sources")
        exact_keys(refs, REQUIRED_SOURCES[key], "criterion source coverage")
        for task, digest in refs.items():
            require(task_states[task], f"{task} unresolved")
            if task != "T080":
                require(
                    digest == rows[task]["evidence_sha256"],
                    f"{task} source receipt mismatch",
                )
            require(
                artifacts.verify(digest, f"{key} {task}") != review,
                "independent review cannot be its own source",
            )
        require(
            review in array(criteria[key].get("evidence"), f"{key} roadmap evidence"),
            "criterion review receipt missing from roadmap",
        )
        if key == "C06":
            _observability(row.get("observability"))

    criterion_states = {
        key: attempt(key, lambda key=key: verify_criterion(key)) for key in CRITERIA
    }
    area_states = {}
    for name, row in areas.items():
        missing = [key for key in row["criteria"] if not criterion_states[key]]
        accepted = not missing and row["status"] == "accepted"
        if not accepted:
            blockers.append(f"{name}: area not accepted")
        area_states[name] = {
            "status": "accepted" if accepted else "blocked",
            "missing_criteria": missing,
        }

    def handover() -> None:
        record = mapping(bundle.get("handover"), "T081 handover")
        require(
            text_value(record.get("reviewer"), "T080/T081 reviewer") in reviewers,
            "T080/T081 handover requires a criterion reviewer",
        )
        flags(
            record,
            "T081",
            "independent_review_complete",
            "home_guide_updated",
            "runbooks_updated",
            "matrix_updated",
            "remaining_limits_documented",
        )
        flags(record, "T082", "human_final_acceptance")
        for field in (
            "independent_review_sha256",
            "handover_sha256",
            "human_acceptance_sha256",
        ):
            artifacts.verify(record.get(field), field)
        require(
            len(
                {
                    record[field]
                    for field in (
                        "independent_review_sha256",
                        "handover_sha256",
                        "human_acceptance_sha256",
                    )
                }
            )
            == 3,
            "T080–T082 require distinct records",
        )
        for task, field in (
            ("T080", "independent_review_sha256"),
            ("T081", "handover_sha256"),
            ("T082", "human_acceptance_sha256"),
        ):
            require(
                record[field]
                in array(tasks[task]["evidence"], f"{task} roadmap evidence"),
                f"{task}: handover receipt missing from roadmap",
            )
        artifacts.record(
            record["human_acceptance_sha256"],
            {
                "decision": "accepted",
                "candidate": identity,
                "profile_sha256": PROFILE_SHA256,
                "handover_sha256": record["handover_sha256"],
            },
            "T082 final human acceptance",
        )

    handover_valid = attempt("T080–T082", handover)
    for task in ("T080", "T081", "T082"):
        task_states[task] = task_states[task] and handover_valid
    milestones = {}
    for key in HITOS:
        related = {phase for phase, row in phases.items() if key in row["maps_to"]}
        pending = [
            task
            for task, row in required.items()
            if row["phase"] in related and not task_states[task]
        ]
        pending_findings = [
            finding
            for finding in open_findings
            if any(
                tasks[task]["phase"] in related for task in findings[finding]["tasks"]
            )
        ]
        related_criteria = {
            criterion
            for row in required.values()
            if row["phase"] in related
            for criterion in row["criteria"]
        }
        pending_criteria = sorted(
            criterion
            for criterion in related_criteria
            if not criterion_states[criterion]
        )
        eligible = not (pending or pending_findings or pending_criteria) and (
            key != "H6" or not blockers
        )
        milestones[key] = {
            "status": "evidence_complete" if eligible else "blocked",
            "phases": sorted(related),
            "pending_tasks": pending,
            "pending_criteria": pending_criteria,
            "open_findings": pending_findings,
        }
    return {
        "candidate": identity,
        "T082": "ready_for_closure" if not blockers else "blocked",
        "areas": area_states,
        "criteria": {
            key: "accepted" if valid else "blocked"
            for key, valid in criterion_states.items()
        },
        "milestones": milestones,
        "open_findings": open_findings,
        "blockers": blockers,
    }


def main(argv: list[str] | None = None) -> int:
    """Check closure eligibility or print the read-only handover report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "report"))
    parser.add_argument("--roadmap", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        roadmap, _ = read_json(args.roadmap)
        bundle, _ = read_json(args.bundle)
        result = audit(roadmap, bundle, args.artifacts)
    except EvidenceError as exc:
        sys.stderr.write(f"T082 blocked: {exc}\n")
        return 1
    except INPUT_ERRORS as exc:
        sys.stderr.write(
            f"T082 blocked: invalid or unreadable input ({type(exc).__name__})\n"
        )
        return 1
    sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return int(args.mode == "check" and result["T082"] != "ready_for_closure")


if __name__ == "__main__":
    sys.exit(main())
