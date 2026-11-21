"""Render approval requests and validate private P7 evidence offline."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any

from src.evaluation.freeze_manifest import (
    ConsentReceipt,
    EvaluationClip,
    Partition,
    VoiceAssignment,
    WakeCounts,
    canonical_sha256,
    clarity_median,
    nearest_rank,
    seeded_order,
    validate_inventory,
    voice_pair_counts,
)
from tools.release_manifest_review import (
    FROZEN_SLOT,
    L4_PHRASE,
    LEGACY_COMMIT,
    LEGACY_SLOT,
    PRODUCTION_ROOT,
    SCHEMA_DIGEST,
    SCHEMA_EPOCH,
    rollback_plan,
)

ROOMS = ("Living", "Hall", "Escritorio", "Cocina")
DISTANCES = (1, 3)
CONDITIONS = ("clear", "noise", "tv")
INTEGRATIONS = ("I01", "I02", "I03", "I04", "I05")
CLIENTS = ("iphone", "terra_chrome")
WIDTHS = (320, 390, 768, 1280, 1440)
L4_SECONDS = 14 * 86400
PROTOCOL = "p6-v1"
P6_DOCUMENT = Path(__file__).resolve().parents[1] / "docs/plans/p6-evaluation.md"
P7_CRITERIA = {
    "T070": ("C19", "C20", "C21"),
    "T071": ("C04", "C12", "C18"),
    "T072": ("C12", "C13", "C14", "C15", "C16", "C17", "C18"),
    "T073": ("C07", "C08", "C09", "C10", "C11"),
    "T074": ("C19", "C20"),
    "T075": ("C04", "C05", "C21"),
}

type Document = dict[str, Any]


class EvidenceError(ValueError):
    """Indicate missing, inconsistent or failed evidence."""


def require(condition: bool, label: str) -> None:
    """Reject an unmet evidence obligation."""
    if not condition:
        raise EvidenceError(label)


def mapping(value: object, label: str) -> Document:
    """Require a JSON object."""
    require(type(value) is dict, label)
    return value


def array(value: object, label: str) -> list[Any]:
    """Require a JSON array."""
    require(type(value) is list, label)
    return value


def text_value(value: object, label: str) -> str:
    """Require a nonempty single-line identifier or label."""
    require(
        type(value) is str
        and bool(value)
        and value.strip() == value
        and value.isprintable(),
        label,
    )
    return value


def hash_value(value: object, label: str, *, length: int = 64) -> str:
    """Require a complete lowercase digest."""
    result = text_value(value, label)
    require(re.fullmatch(rf"[a-f0-9]{{{length}}}", result) is not None, label)
    return result


def count(value: object, label: str, minimum: int = 0) -> int:
    """Require a nonnegative integer with an optional minimum."""
    require(type(value) is int and value >= minimum, label)
    return value


def number(value: object, label: str, maximum: float = math.inf) -> float:
    """Require a finite nonnegative measurement at or below its limit."""
    require(
        type(value) in (int, float) and 0 <= value <= maximum and math.isfinite(value),
        label,
    )
    return value


def instant(value: object, label: str) -> datetime:
    """Require a timestamp with an explicit offset and normalize it to UTC."""
    result = datetime.fromisoformat(text_value(value, label))
    require(result.utcoffset() is not None, label)
    return result.astimezone(UTC)


def interval(source: Document, label: str) -> tuple[datetime, datetime]:
    """Read a positive half-open interval from start and end fields."""
    start = instant(source.get("start"), label)
    end = instant(source.get("end"), label)
    require(start < end, label)
    return start, end


def flags(source: Document, label: str, *fields: str) -> None:
    """Require explicit positive observations for the named fields."""
    for field in fields:
        require(source.get(field) is True, f"{label}.{field}")


def indexed(value: object, label: str, key: str = "id") -> dict[str, Document]:
    """Index unique records by a mandatory identifier."""
    rows = {}
    for item in array(value, label):
        row = mapping(item, label)
        item_id = text_value(row.get(key), label)
        require(item_id not in rows, f"{label}: duplicate {key}")
        rows[item_id] = row
    return rows


def identifiers(value: object, label: str) -> set[str]:
    """Read an array of distinct identifiers."""
    rows = [text_value(item, label) for item in array(value, label)]
    require(len(set(rows)) == len(rows), f"{label}: duplicate identifier")
    return set(rows)


def exact_keys(source: Document, expected: Iterable[str], label: str) -> None:
    """Require every expected key and reject unexpected alternatives."""
    require(set(source) == set(expected), f"{label}: incomplete or unexpected keys")


def private_path(path: Path) -> Path:
    """Resolve an input path outside the production tree."""
    require(
        not path.absolute().is_relative_to(PRODUCTION_ROOT), "production path forbidden"
    )
    path = path.resolve()
    require(not path.is_relative_to(PRODUCTION_ROOT), "production path forbidden")
    return path


def read_json(path: Path) -> tuple[Document, str]:
    """Read a strict JSON object and return its original byte digest."""

    def unique(pairs: list[tuple[str, Any]]) -> Document:
        result: Document = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def invalid_constant(_: str) -> None:
        raise EvidenceError("non-finite JSON number")

    def finite_float(raw: str) -> float:
        result = float(raw)
        require(math.isfinite(result), "non-finite JSON number")
        return result

    contents = private_path(path).read_bytes()
    value = json.loads(
        contents,
        object_pairs_hook=unique,
        parse_constant=invalid_constant,
        parse_float=finite_float,
    )
    return mapping(value, "JSON root"), sha256(contents).hexdigest()


def candidate(value: object) -> Document:
    """Validate the identity shared by approvals and all field evidence."""
    data = mapping(value, "candidate")
    exact_keys(
        data, ("git_commit", "archive_sha256", "schema_contract_epoch"), "candidate"
    )
    require(
        hash_value(data["git_commit"], "candidate.git_commit", length=40)
        != LEGACY_COMMIT,
        "candidate must differ from the legacy release",
    )
    hash_value(data["archive_sha256"], "candidate.archive_sha256")
    require(
        count(data["schema_contract_epoch"], "candidate epoch") == SCHEMA_EPOCH,
        "Q17 requires SCHEMA_CONTRACT_EPOCH = 2",
    )
    return data


def _entity(value: object, domain: str) -> None:
    entity = text_value(value, "HA entity")
    require(
        re.fullmatch(rf"{domain}\.[a-z0-9_]+", entity) is not None, "invalid HA entity"
    )


def _selection(value: object) -> None:
    data = mapping(value, "Q16 selection")
    exact_keys(data, INTEGRATIONS, "Q16 selection")
    energy = mapping(data["I01"], "I01")
    exact_keys(energy, ("energy", "power"), "I01")
    groups = [identifiers(energy[key], f"I01.{key}") for key in ("energy", "power")]
    require(
        all(groups) and not groups[0] & groups[1],
        "I01 energy/power must be distinct and nonempty",
    )
    for sensor in groups[0] | groups[1]:
        _entity(sensor, "sensor")
    _entity(data["I02"], "todo")
    calendar = mapping(data["I03"], "I03")
    exact_keys(calendar, ("kind", "source"), "I03")
    if calendar["kind"] == "ha":
        _entity(calendar["source"], "calendar")
    else:
        source = text_value(calendar["source"], "I03 .ics reference")
        require(
            calendar["kind"] == "ics"
            and re.fullmatch(r"[a-zA-Z0-9_.-]+\.ics", source) is not None,
            "I03 requires a private .ics reference or a calendar entity",
        )
    media = mapping(data["I04"], "I04")
    exact_keys(media, ("entity", "controls"), "I04")
    _entity(media["entity"], "media_player")
    controls = identifiers(media["controls"], "I04 controls")
    require(
        bool(controls) and controls <= {"play", "pause", "volume"},
        "I04 selected controls",
    )
    cameras = mapping(data["I05"], "I05")
    exact_keys(cameras, ("entities", "video"), "I05")
    entities = identifiers(cameras["entities"], "I05 technical entities")
    require(
        bool(entities) and cameras["video"] is False,
        "I05 requires technical entities without video",
    )
    for entity in entities:
        _entity(entity, "(?:camera|sensor|binary_sensor)")


def validate_release_review(review: Document, identity: Document) -> None:
    """Check the Q4 identity and preflight fields of the P6 review."""
    require(
        count(review.get("manifest_version"), "release review version") == 1,
        "release review version",
    )
    require(
        review.get("git_commit") == identity["git_commit"],
        "release review candidate mismatch",
    )
    artifact = mapping(review.get("artifact"), "release review artifact")
    require(
        artifact.get("sha256") == identity["archive_sha256"]
        and artifact.get("python_abi") == "cp313"
        and count(artifact.get("python_sources"), "release source count") == 0,
        "release review artifact mismatch",
    )
    schema = mapping(review.get("schema_contract"), "release review schema")
    require(
        count(schema.get("epoch"), "release review epoch") == SCHEMA_EPOCH
        and schema.get("sha256") == SCHEMA_DIGEST,
        "release review requires pinned Epoch 2",
    )
    hash_value(review.get("external_runtime_lock_sha256"), "review runtime lock")
    require(
        review.get("rollback") == rollback_plan(), "release review Q4 L4 plan mismatch"
    )
    preflight = mapping(
        review.get("synthetic_adoption_preflight"), "release review preflight"
    )
    require(
        preflight.get("schema_only") == "passed", "release review schema-only preflight"
    )
    flags(
        preflight,
        "release review preflight",
        "legacy_current_verified",
        "frozen_database_unchanged",
    )


ADOPTION = {
    "mode": "direct_once",
    "legacy_slot": LEGACY_SLOT,
    "legacy_commit": LEGACY_COMMIT,
    "legacy_reminder_columns": 11,
    "scope": "private",
    "delivery_room": None,
}
ROLLBACK = {
    "method": "L4",
    "managed": False,
    "legacy_slot": LEGACY_SLOT,
    "frozen_state_slot": FROZEN_SLOT,
    "seconds_from_first_clean_close": L4_SECONDS,
    "phrase": L4_PHRASE,
}


def render_request(
    proposal: Document, review: Document, review_sha256: str
) -> Document:
    """Build an unsigned Q16/Q17 request bound to the P6 review."""
    exact_keys(proposal, ("candidate", "q16", "q17"), "proposal")
    identity = candidate(proposal["candidate"])
    validate_release_review(review, identity)
    q17 = mapping(proposal["q17"], "Q17 proposal")
    exact_keys(q17, ("operator", "window", "interventions"), "Q17 proposal")
    package = {
        "status": "pending_human_approval",
        "candidate": identity,
        "q16": proposal["q16"],
        "q17": {
            **q17,
            "release_review_sha256": review_sha256,
            "adoption": dict(ADOPTION),
            "rollback": dict(ROLLBACK),
        },
    }
    validate_request(package)
    return package


def validate_request(package: Document) -> None:
    """Verify the unsigned request and the exact operational limits."""
    exact_keys(package, ("status", "candidate", "q16", "q17"), "approval request")
    require(
        package["status"] == "pending_human_approval", "request must remain unsigned"
    )
    candidate(package["candidate"])
    q16 = mapping(package["q16"], "Q16")
    exact_keys(q16, ("inventory_sha256", "selection", "differences"), "Q16")
    hash_value(q16["inventory_sha256"], "Q16 inventory")
    _selection(q16["selection"])
    text_value(q16["differences"], "Q16 inventory differences")
    q17 = mapping(package["q17"], "Q17")
    exact_keys(
        q17,
        (
            "operator",
            "window",
            "interventions",
            "release_review_sha256",
            "adoption",
            "rollback",
        ),
        "Q17",
    )
    text_value(q17["operator"], "Q17 operator")
    start, end = interval(mapping(q17["window"], "Q17 window"), "Q17 window")
    require(end - start <= timedelta(days=7), "Q17 field window exceeds seven days")
    require(
        bool(identifiers(q17["interventions"], "Q17 interventions")),
        "Q17 intervention scope missing",
    )
    hash_value(q17["release_review_sha256"], "release-review.json SHA-256")
    require(
        canonical_sha256(q17["adoption"]) == canonical_sha256(ADOPTION),
        "Q17 direct adoption",
    )
    require(
        canonical_sha256(q17["rollback"]) == canonical_sha256(ROLLBACK), "Q17 L4 plan"
    )


def approval(
    value: object, question: str, package: Document, before: datetime
) -> Document:
    """Bind a recorded human decision to the exact requested candidate and selection."""
    decision = mapping(value, question)
    require(
        decision.get("question") == question and decision.get("decision") == "approved",
        f"{question}: explicit human approval required",
    )
    require(
        decision.get("package_sha256") == canonical_sha256(package),
        f"{question}: request hash mismatch",
    )
    hash_value(decision.get("receipt_sha256"), f"{question}: decision receipt")
    require(
        instant(decision.get("approved_at"), question) <= before,
        f"{question}: late authorization",
    )
    return decision


def _field_request(data: Document) -> tuple[Document, datetime, datetime]:
    package = mapping(data.get("request"), "field approval request")
    validate_request(package)
    require(data["candidate"] == package["candidate"], "approved candidate mismatch")
    start, end = interval(package["q17"]["window"], "Q17 window")
    approval(data.get("q17_approval"), "Q17", package, start)
    require(
        start <= instant(data["observed_at"], "observed_at") <= end,
        "observation outside Q17 window",
    )
    return package, start, end


def validate_t070(data: Document) -> None:
    """Validate physical channel mapping and all five real integrations."""
    package, start, _ = _field_request(data)
    approval(data.get("q16_approval"), "Q16", package, start)
    microphones = mapping(data.get("microphones"), "T070 microphones")
    exact_keys(microphones, ROOMS, "T070 microphones")
    channels = set()
    ports = set()
    for room, mic in microphones.items():
        mic = mapping(mic, "T070 microphone")
        channel = count(mic.get("channel"), "T070 channel")
        port = text_value(mic.get("usb_port"), "T070 USB port")
        flags(
            mic,
            f"T070 {room}",
            "physical_probe",
            "present",
            "hw_ptr_advanced",
            "channel_match",
        )
        heartbeat = mapping(mic.get("heartbeat"), f"T070 {room} heartbeat")
        flags(heartbeat, f"T070 {room} heartbeat", "ever", "coverage_complete")
        count(heartbeat.get("sampled_seconds"), f"T070 {room} heartbeat coverage", 300)
        number(heartbeat.get("max_age_s"), f"T070 {room} heartbeat age", 8)
        number(heartbeat.get("max_staleness_s"), f"T070 {room} heartbeat staleness", 10)
        require(
            mic.get("capture_state") == "running"
            and mic.get("stimulus_room") == room
            and mic.get("observed_room") == room,
            f"T070 {room} channel mismatch",
        )
        require(
            channel not in channels and port not in ports,
            "T070 duplicate physical microphone",
        )
        channels.add(channel)
        ports.add(port)
        if room == "Cocina":
            require(port == "7-5.3", "T070 Cocina XVF3800 physical port")
    selection = package["q16"]["selection"]
    results = mapping(data.get("integrations"), "T070 integration results")
    exact_keys(results, INTEGRATIONS, "T070 integration results")
    for key in INTEGRATIONS:
        row = mapping(results[key], key)
        require(
            canonical_sha256(row.get("selection")) == canonical_sha256(selection[key]),
            f"{key}: Q16 selection mismatch",
        )
        require(
            row.get("source_kind") == "real" and row.get("simulated") is False,
            f"{key}: real source required",
        )
        flags(
            row,
            key,
            "operated",
            "contract_passed",
            "rejection_passed",
            "expiry_passed",
            "permissions_passed",
        )
        hash_value(row.get("observation_sha256"), f"{key}: real observation")
    flags(
        results["I02"],
        "I02",
        "idempotency_passed",
        "reconciliation_passed",
        "personal_lists_local",
    )
    flags(results["I03"], "I03", "read_only")
    flags(results["I04"], "I04", "selected_controls_verified", "idempotency_passed")
    flags(results["I05"], "I05", "technical_only")
    require(results["I05"].get("video_accessed") is False, "I05 video forbidden")


def _clips(value: object) -> dict[str, EvaluationClip]:
    return {
        key: EvaluationClip(
            key,
            hash_value(row.get("audio_sha256"), "PCM hash"),
            hash_value(row.get("source_sha256"), "source PCM hash"),
            text_value(row.get("session_id"), "session"),
            Partition(row.get("split")),
            instant(row.get("start"), "capture start"),
            instant(row.get("end"), "capture end"),
            frozenset(identifiers(row.get("speaker_refs"), "clip speakers")),
            text_value(row.get("consent_ref"), "consent reference"),
        )
        for key, row in indexed(value, "clip inventory", "clip_id").items()
    }


def _receipts(value: object, close: datetime) -> dict[str, ConsentReceipt]:
    result = {}
    for key, row in indexed(value, "consent receipts").items():
        expires = min(instant(row.get("expires_at"), "consent expiry"), close)
        revoked = (
            instant(row["revoked_at"], "consent revocation")
            if row["revoked_at"] is not None
            else None
        )
        flags(row, "consent", "granted")
        hash_value(row.get("receipt_sha256"), "consent record")
        result[key] = ConsentReceipt(
            text_value(row.get("session_id"), "consent session"),
            text_value(row.get("purpose"), "consent purpose"),
            True,
            frozenset(identifiers(row.get("speaker_refs"), "consent speakers")),
            instant(row.get("granted_at"), "consent grant"),
            expires,
            revoked,
        )
    return result


def calibration_clips(data: Document) -> dict[str, EvaluationClip]:
    """Validate the consent and sample minimum of the T071 calibration set."""
    frozen, _, close = _freeze(data)
    observed = instant(data["observed_at"], "T071 observed_at")
    clips = _clips(data.get("clips"))
    rows = indexed(data.get("clips"), "calibration clips", "clip_id")
    receipts = _receipts(data.get("receipts"), close)
    speakers: Counter[str] = Counter()
    for clip in clips.values():
        require(
            clip.split is Partition.CALIBRATION and len(clip.speaker_refs) == 1,
            "T071 calibration clip split/voice",
        )
        flags(rows[clip.clip_id], "T071 calibration", "clear", "deletion_checked")
        receipt = receipts.get(clip.consent_ref)
        require(
            receipt is not None
            and receipt.session_id == clip.session_id
            and receipt.purpose == "consented_calibration_and_evaluation"
            and clip.speaker_refs <= receipt.speaker_refs,
            "T071 consent does not cover capture",
        )
        deadline = min(
            clip.end + timedelta(days=30),
            receipt.expires_at,
            receipt.revoked_at or close,
        )
        require(
            receipt.granted_at
            <= clip.start
            < clip.end
            <= frozen
            <= observed
            < deadline,
            "T071 consent expired or capture invalid",
        )
        speakers.update(clip.speaker_refs)
    require(
        len({clip.audio_sha256 for clip in clips.values()}) == len(clips),
        "T071 duplicate PCM",
    )
    require(
        len(speakers) == 3 and min(speakers.values()) >= 10,
        "T071 needs three voices x 10 clear clips",
    )
    return clips


def _freeze(data: Document) -> tuple[datetime, datetime, datetime]:
    sealed = instant(data.get("frozen_at"), "acoustic freeze")
    opened = instant(data.get("reserve_opened_at"), "reserve opening")
    close = instant(data.get("evaluation_close"), "evaluation closing")
    require(
        sealed < opened < close,
        "acoustic candidate must be frozen before opening reserve",
    )
    fields = (
        "runtime",
        "models",
        "dsp",
        "configuration",
        "corpus_contract",
        "exclusions",
        "instrumentation",
        "reserve_inventory",
        "strata_plan",
    )
    seals = mapping(data.get("seals"), "acoustic seals")
    exact_keys(seals, fields, "acoustic seals")
    for field in fields:
        hash_value(seals[field], f"acoustic seal {field}")
    require(
        data.get("acoustic_sha256") == canonical_sha256(seals), "acoustic seal mismatch"
    )
    return sealed, opened, close


def validate_t071(data: Document) -> None:
    """Check three consented calibration voices and the acoustic freeze."""
    _, start, end = _field_request(data)
    frozen, opened, close = _freeze(data)
    observed = instant(data["observed_at"], "T071 observed_at")
    require(
        start <= frozen <= observed < opened < close <= end,
        "T071 freeze/opening outside Q17 window",
    )
    require(
        all(start <= clip.start for clip in calibration_clips(data).values()),
        "T071 capture before authorization",
    )


def _planned_rows(data: Document, plan: Document, section: str) -> list[Document]:
    rows = indexed(data.get(section), section)
    schedule = mapping(plan["scheduled"][section], f"scheduled {section}")
    exact_keys(rows, schedule, f"{section}: every preregistered attempt must remain")
    extra_fields = {
        "requests": ("family",),
        "simple": ("family", "cache", "rapid_ha"),
        "listening_pairs": ("length",),
    }
    fields = (
        "room",
        "distance_m",
        "participant",
        "condition",
        *extra_fields.get(section, ()),
    )
    for key, row in rows.items():
        expected = mapping(schedule[key], "planned stratum")
        exact_keys(expected, fields, "planned stratum")
        require(
            canonical_sha256({field: row[field] for field in fields})
            == canonical_sha256(expected),
            f"{section}: stratum differs from sealed plan",
        )
        require(
            row["room"] in ROOMS
            and type(row["distance_m"]) is int
            and row["distance_m"] in DISTANCES
            and row["participant"] in plan["panel"]
            and row["condition"] in plan["conditions"],
            f"{section}: invalid stratum",
        )
    return list(rows.values())


def _strata(
    rows: list[Document], plan: Document, label: str
) -> Iterable[tuple[str, list[Document]]]:
    yield f"{label} global", rows
    for field, keys in (
        ("room", ROOMS),
        ("distance_m", DISTANCES),
        ("participant", plan["panel"]),
        ("condition", plan["conditions"]),
    ):
        for key in keys:
            group = [row for row in rows if row[field] == key]
            name = f"{label} {field}={key}"
            require(len(group) >= 10, f"{name}: fewer than ten eligible cases")
            yield name, group


def _ratio(numerator: int, denominator: int, minimum_percent: int, label: str) -> None:
    require(
        0 <= numerator <= denominator
        and denominator > 0
        and numerator * 100 >= minimum_percent * denominator,
        label,
    )


def _recognition(rows: list[Document], clear: bool, label: str) -> None:
    words = errors = matched = 0
    for row in rows:
        score = row["recognition"]
        if score is not None:
            score = mapping(score, "C12 recognition")
            flags(score, "C12", "human_aligned")
            words += count(score.get("words"), "C12 reference words", 1)
            edits = [
                count(score.get(field), f"C12 {field}")
                for field in ("substitutions", "insertions", "deletions")
            ]
            require(
                edits[0] + edits[2] <= score["words"],
                "C12 impossible substitutions/deletions",
            )
            errors += sum(edits)
            count(score.get("characters"), "C12 reference characters", score["words"])
            count(score.get("character_edits"), "C12 character edits")
            matched += 1
    _ratio(matched, len(rows), 95, f"C12 {label}: coverage <95%")
    require(
        words > 0 and errors * 100 <= (10 if clear else 20) * words,
        f"C12 {label}: WER exceeded",
    )


def _task_effect(rows: list[Document], clear: bool, label: str) -> None:
    completed = 0
    for row in rows:
        result = row.get("effect_verified")
        require(type(result) is bool, "C13 requires adjudicated task effect")
        if result:
            hash_value(row.get("effect_sha256"), "C13 independent effect observation")
            completed += 1
    _ratio(
        completed, len(rows), 98 if clear else 95, f"C13 {label}: verified task success"
    )


def _families(
    rows: list[Document], expected: object, label: str
) -> Iterable[tuple[str, list[Document]]]:
    groups = defaultdict(list)
    for row in rows:
        groups[text_value(row.get("family"), label)].append(row)
    exact_keys(groups, identifiers(expected, label), label)
    for key, group in groups.items():
        require(len(group) >= 10, f"{label} {key}: fewer than ten cases")
        yield key, group


def _check_words(data: Document, plan: Document) -> None:
    rows = _planned_rows(data, plan, "requests")
    conditions = Counter(row["condition"] for row in rows)
    require(
        conditions["clear"] >= 200 and len(rows) - conditions["clear"] >= 100,
        "C12 clear/adverse minima",
    )
    require(
        all(conditions[key] >= 20 for key in plan["critical_conditions"]),
        "C12 critical condition minimum",
    )
    for label, group in _strata(rows, plan, "C12/C13"):
        for condition in plan["conditions"]:
            cases = [row for row in group if row["condition"] == condition]
            if cases:
                _recognition(cases, condition == "clear", label)
                _task_effect(cases, condition == "clear", label)
    for label, group in _families(rows, plan.get("task_families"), "C13 families"):
        for condition in plan["conditions"]:
            cases = [row for row in group if row["condition"] == condition]
            if cases:
                _task_effect(cases, condition == "clear", label)


def _house_seconds(wake: Document, window: tuple[datetime, datetime]) -> int:
    intervals = []
    for row in array(wake.get("useful_intervals"), "C14 useful intervals"):
        row = mapping(row, "C14 useful interval")
        require(
            identifiers(row.get("useful_inputs"), "C14 useful inputs") == set(ROOMS),
            "C14 requires all four simultaneous useful inputs",
        )
        intervals.append(interval(row, "C14 useful interval"))
    seconds = 0
    previous = window[0]
    for start, end in sorted(intervals):
        require(
            previous <= start < end <= window[1],
            "C14 overlapping or out-of-window house clock intervals",
        )
        duration = (end - start).total_seconds()
        require(duration.is_integer(), "C14 house interval requires whole seconds")
        seconds += int(duration)
        previous = end
    return seconds


def _check_wake(
    data: Document, plan: Document, window: tuple[datetime, datetime]
) -> None:
    wake = mapping(data.get("wake"), "C14 wake")
    require(wake.get("clock") == "house_clock", "C14 requires house-clock hours")
    seconds = _house_seconds(wake, window)
    require(
        seconds >= 48 * 3600
        and count(wake.get("house_seconds"), "C14 reported clock") == seconds,
        "C14 needs >=48h of simultaneous house time without summing microphone-hours",
    )
    rows = _planned_rows(data, plan, "wakes")
    require(len(rows) >= 100, "C14 needs at least 100 intentional wakes")
    flags(
        wake,
        "C14",
        "emissions_inventory_complete",
        "deduplication_physically_corroborated",
    )
    raw = mapping(wake.get("raw_by_room"), "C14 original microphone counts")
    exact_keys(raw, ROOMS, "C14 original microphone counts")
    emissions = indexed(wake.get("emissions"), "C14 emissions")
    matched = {}
    for row in emissions.values():
        rooms = identifiers(row.get("rooms"), "C14 emitting inputs")
        require(bool(rooms) and rooms <= set(ROOMS), "C14 emitting rooms")
        if len(rooms) > 1:
            hash_value(
                row.get("corroboration_sha256"), "C14 deduplicated physical event"
            )
        require(
            row.get("outcome") in {"true", "false", "uncertain"},
            "C14 emission adjudication",
        )
        if row["outcome"] == "true":
            attempt = text_value(
                row.get("attempt_id"), "C14 matched intentional attempt"
            )
            require(attempt not in matched, "C14 duplicate matched wake")
            hash_value(row.get("correlation_sha256"), "C14 physical wake correlation")
            matched[attempt] = rooms
    accepted = {
        row["id"]: row["room"] for row in rows if row.get("outcome") == "accepted"
    }
    require(
        matched.keys() == accepted.keys()
        and all(room in matched[key] for key, room in accepted.items()),
        "C14 wake correlation",
    )
    for label, group in _strata(rows, plan, "C14"):
        outcomes = Counter(row.get("outcome") for row in group)
        require(
            set(outcomes) <= {"accepted", "rejected", "uncertain"},
            f"{label}: intentional wake adjudication",
        )
        counts = WakeCounts(
            outcomes["accepted"],
            outcomes["rejected"],
            outcomes["uncertain"],
            0,
            0,
            0,
            seconds,
        )
        require(counts.rejection_upper <= 0.05, f"{label}: conservative rejection rate")
    for room in (None, *ROOMS):
        group = [
            row for row in emissions.values() if room is None or room in row["rooms"]
        ]
        if room is not None:
            require(
                count(raw[room], "C14 original count") == len(group),
                "C14 emission tally mismatch",
            )
        outcomes = Counter(row["outcome"] for row in group)
        counts = WakeCounts(
            0,
            0,
            0,
            outcomes["false"],
            outcomes["uncertain"],
            outcomes["true"] + outcomes["false"],
            seconds,
        )
        require(
            not group or counts.review_coverage >= 0.95, "C14 emission review coverage"
        )
        require(counts.false_per_hour_upper <= 0.1, "C14 conservative false wake rate")


def _latencies(rows: list[Document], field: str) -> list[float | None]:
    return [None if row[field] is None else number(row[field], field) for row in rows]


def _check_simple(data: Document, plan: Document) -> None:
    rows = _planned_rows(data, plan, "simple")
    require(len(rows) >= 100, "C15 needs 100 simple runs")
    require(
        any(row["rapid_ha"] is True for row in rows),
        "C15 requires rapid HA action measurements",
    )
    for row in rows:
        flags(row, "C15", "earcon_excluded", "clock_bound_applied")
        require(
            row["cache"] in {"cold", "hot"} and type(row["rapid_ha"]) is bool,
            "C15 cache/HA applicability",
        )
    families = _families(rows, plan.get("simple_families"), "C15 families")
    cache = [
        (f"C15 {key}", [row for row in rows if row["cache"] == key])
        for key in ("cold", "hot")
    ]
    for label, group in (*_strata(rows, plan, "C15"), *families, *cache):
        voice = _latencies(group, "voice_ms")
        require(
            bool(voice)
            and nearest_rank(voice, 50) <= 800
            and nearest_rank(voice, 95) <= 1500,
            f"{label}: voice latency",
        )
        rapid = [row for row in group if row["rapid_ha"]]
        require(
            not rapid or nearest_rank(_latencies(rapid, "ha_ms"), 95) <= 2000,
            f"{label}: HA latency",
        )


def _check_delivery(data: Document, plan: Document) -> None:
    rows = _planned_rows(data, plan, "deliveries")
    require(len(rows) >= 100, "C16 needs 100 due deliveries")
    for row in rows:
        require(
            row.get("outcome")
            in {"complete", "failed", "partial", "timeout", "blocked"},
            "C16 delivery outcome",
        )
        flags(row, "C16", "state_matches_observation", "queue_and_pcm_recorded")
        if row["outcome"] == "complete":
            flags(row, "C16", "audible_end_verified")
    for label, group in _strata(rows, plan, "C16"):
        _ratio(
            sum(row["outcome"] == "complete" for row in group),
            len(group),
            99,
            f"{label}: delivery rate",
        )
    cancellations = indexed(
        data.get("voluntary_cancellations"), "C16 voluntary cancellations"
    )
    require(
        not cancellations.keys() & {row["id"] for row in rows},
        "C16 cancellation also counted as due",
    )
    for row in cancellations.values():
        flags(
            row, "C16 cancellation", "voluntary_confirmed", "state_matches_observation"
        )
    interrupts = _planned_rows(data, plan, "interruptions")
    require(len(interrupts) >= 20, "C16 needs 20 physical interruptions")
    for row in interrupts:
        flags(
            row,
            "C16 interruption",
            "physical",
            "barge_in",
            "concurrent_capture",
            "clock_bound_applied",
        )
    require(
        any(row.get("long_uncached") is True for row in interrupts),
        "C16 needs a long uncached interruption",
    )
    for label, group in _strata(interrupts, plan, "C16 interruption"):
        require(
            nearest_rank(_latencies(group, "stop_ms"), 95) <= 250,
            f"{label}: acoustic interrupt latency",
        )


def _check_listening(data: Document, plan: Document) -> None:
    listening = mapping(data.get("listening"), "C17 listening")
    flags(
        listening, "C17", "blind_until_locked", "comparable_paired_audio", "consented"
    )
    rows = _planned_rows(data, plan, "listening_pairs")
    require(
        len(rows) >= 30 and len(rows) % 2 == 0,
        "C17 requires at least 30 paired answers and an even lot",
    )
    panel = sorted(plan["panel"])
    ordered = seeded_order([row["id"] for row in rows], purpose="listening", seed=0)
    by_id = {row["id"]: row for row in rows}
    require(
        {row["length"] for row in rows} == {"short", "long"},
        "C17 needs short and long answers",
    )
    for i, key in enumerate(ordered):
        ratings = mapping(by_id[key].get("ratings"), "C17 ratings")
        exact_keys(ratings, panel, "C17 listeners")
        for j, listener in enumerate(panel):
            vote = mapping(ratings[listener], "C17 vote")
            require(
                count(vote.get("first_variant"), "C17 variant order") == (i + j) % 2,
                "C17 unbalanced presentation",
            )
            require(vote.get("severe") is False, "C17 severe defect")
            clarity_median([vote.get("reference"), vote.get("candidate")])
    for label, group in _strata(rows, plan, "C17"):
        for listener in panel:
            require(
                clarity_median([row["ratings"][listener]["candidate"] for row in group])
                >= 4,
                f"{label}: candidate clarity median <4 for a participant",
            )


def _check_h3(
    data: Document, reserved: dict[str, EvaluationClip], panel: set[str]
) -> None:
    h3 = mapping(data.get("h3"), "C18 H3")
    clips = indexed(h3.get("clips"), "C18 H3 clips")
    require(clips.keys() <= reserved.keys(), "C18 H3 clip outside consented reserve")
    clears: Counter[str] = Counter()
    conditions: Counter[str] = Counter()
    rooms = set()
    distances = set()
    for key, row in clips.items():
        kind = text_value(row.get("condition"), "C18 condition")
        require(
            kind in {"clear", "short", "tv", "echo", "overlap"},
            "C18 unknown acoustic condition",
        )
        require(
            row.get("room") in ROOMS
            and type(row.get("distance_m")) is int
            and row["distance_m"] in DISTANCES,
            "C18 invalid room/distance",
        )
        rooms.add(row["room"])
        distances.add(row["distance_m"])
        conditions[kind] += 1
        voices = reserved[key].speaker_refs
        if kind == "clear":
            require(
                len(voices) == 1 and voices <= panel,
                "C18 clear clip must have one panel voice",
            )
            clears.update(voices)
        require(
            len(voices) >= 2 if kind == "overlap" else len(voices) == 1,
            "C18 invalid voice adjudication",
        )
        if kind == "short":
            require(
                (reserved[key].end - reserved[key].start).total_seconds() < 2,
                "C18 short clip duration",
            )
    require(
        set(clears) == panel
        and min(clears.values()) >= 30
        and len(rooms) >= 2
        and len(distances) == 2
        and all(conditions[key] >= 10 for key in ("short", "tv", "echo", "overlap")),
        "C18 reserve minima",
    )
    order = sorted(clips, key=lambda key: reserved[key].start)
    require(
        len({reserved[key].start for key in order}) == len(order),
        "C18 ambiguous original capture order",
    )
    runs = indexed(h3.get("runs"), "C18 H3 runs")
    exact_keys(runs, ("original", *(str(seed) for seed in range(10))), "C18 H3 runs")
    for run_id, run in runs.items():
        flags(run, "C18", "started_empty")
        expected = (
            order
            if run_id == "original"
            else seeded_order(order, purpose="h3", seed=int(run_id))
        )
        assignments = indexed(run.get("assignments"), "C18 assignments")
        require(
            list(assignments) == list(expected),
            f"C18 {run_id}: execution order or inventory mismatch",
        )
        rows = []
        for key, row in assignments.items():
            kind = clips[key]["condition"]
            voices = reserved[key].speaker_refs
            speaker = next(iter(voices)) if len(voices) == 1 else None
            group = row["group"]
            if group is not None:
                require(
                    text_value(group, "C18 group").lower() != "unknown",
                    "C18 unknown must be null",
                )
            rows.append(
                VoiceAssignment(
                    key, speaker, group, kind == "clear", kind in {"short", "overlap"}
                )
            )
        counts = voice_pair_counts(rows)
        require(
            counts.mixed_pairs == 0
            and counts.abstention_violations == 0
            and counts.tp > 0
            and counts.precision >= 0.99
            and counts.recall >= 0.80
            and counts.coverage >= 0.70,
            f"C18 {run_id}: pair precision/recall/coverage, mixed voices or abstention",
        )


def validate_t072(data: Document) -> None:
    """Validate p6-v1 inventory, stratified speech, wake, listening and H3."""
    _, start, end = _field_request(data)
    require(data.get("protocol") == PROTOCOL, "T072 requires p6-v1")
    require(
        data.get("protocol_sha256") == sha256(P6_DOCUMENT.read_bytes()).hexdigest(),
        "T072 protocol hash",
    )
    frozen, opened, close = _freeze(data)
    observed = instant(data["observed_at"], "T072 observed_at")
    require(
        start <= frozen < opened <= observed < close <= end,
        "T072 evidence outside consented evaluation",
    )
    flags(
        data,
        "T072",
        "blind_adjudication",
        "deletion_checked",
        "original_result_retained",
    )
    inventory = array(data.get("clips"), "T072 sealed inventory")
    require(
        data["seals"]["reserve_inventory"] == canonical_sha256(inventory),
        "T072 inventory seal mismatch",
    )
    clips = _clips(inventory)
    plan = mapping(data.get("plan"), "T072 strata plan")
    require(
        data["seals"]["strata_plan"] == canonical_sha256(plan),
        "T072 strata seal mismatch",
    )
    require(
        instant(plan.get("registered_at"), "T072 plan timing") <= frozen,
        "T072 plan registered after freeze",
    )
    panel = identifiers(plan.get("panel"), "T072 panel")
    require(len(panel) == 3, "T072 needs exactly three participants")
    conditions = identifiers(plan.get("conditions"), "strata conditions")
    critical = identifiers(
        plan.get("critical_conditions"), "strata critical conditions"
    )
    require(
        set(CONDITIONS) <= conditions and {"noise", "tv"} <= critical <= conditions,
        "strata must include clear, noise, TV and their critical minima",
    )
    exact_keys(
        mapping(plan.get("scheduled"), "T072 scheduled inventory"),
        (
            "requests",
            "wakes",
            "simple",
            "deliveries",
            "interruptions",
            "listening_pairs",
        ),
        "T072 scheduled inventory",
    )
    receipts = _receipts(data.get("receipts"), close)
    validate_inventory(
        clips.values(), receipts, now=observed, shared_panel=frozenset(panel)
    )
    reserved = {
        key: clip for key, clip in clips.items() if clip.split is Partition.RESERVED
    }
    require(
        all(
            start <= clip.start and clip.end <= frozen
            for clip in clips.values()
            if clip.split is Partition.CALIBRATION
        ),
        "T072 calibration outside authorized freeze window",
    )
    require(
        all(start <= clip.start and clip.end <= frozen for clip in reserved.values()),
        "T072 reserve inventory outside authorized sealed window",
    )
    require(
        canonical_sha256(data["h3"]["clips"]) == plan.get("h3_inventory_sha256"),
        "C18 H3 inventory seal mismatch",
    )
    for row in data["requests"]:
        clip = reserved.get(row["id"])
        require(
            clip is not None and row["participant"] in clip.speaker_refs,
            "C12 request outside identified reserve",
        )
    _check_words(data, plan)
    _check_wake(data, plan, (start, observed))
    _check_simple(data, plan)
    _check_delivery(data, plan)
    _check_listening(data, plan)
    _check_h3(data, reserved, panel)


def _p75(values: list[float]) -> float:
    return sorted(values)[(75 * len(values) + 99) // 100 - 1]


def _iphone_ms(value: object, fps: int, label: str) -> float:
    frames = array(value, label)
    require(len(frames) == 2, f"{label}: two independent frame counts required")
    readings = [count(item, label) for item in frames]
    return max(readings) * 1000 / fps


def _timing(
    data: Document, schedule: Document, client: str, phase: str, fps: int
) -> None:
    label = f"{client} {phase}"
    rows = indexed(data.get(phase), label)
    exact_keys(
        rows,
        identifiers(schedule.get(phase), label),
        f"{label}: missing preregistered timing attempts",
    )
    require(len(rows) >= 40, f"{label}: fewer than 40 timing attempts")
    content = []
    interaction = []
    shifts = []
    feedback = []
    preview = []
    for row in rows.values():
        require(
            row.get("status") in {"ok", "error", "timeout"},
            f"C11 {label}: unknown outcome",
        )
        if row["status"] != "ok":
            feedback.append(None)
            preview.append(None)
            continue
        flags(
            row,
            f"C10/C11 {label}",
            "useful_paint_observed",
            "audio_playable_by_native_gesture",
        )
        number(row.get("preview_duration_s"), "C11 preview duration", 6)
        require(
            row.get("touch_target_jump") is False, f"C10 {label}: touch target moved"
        )
        if fps:
            content.append(_iphone_ms(row.get("content_frames"), fps, "C10 content"))
            interaction.append(
                _iphone_ms(row.get("interaction_frames"), fps, "C10 interaction")
            )
            feedback.append(_iphone_ms(row.get("feedback_frames"), fps, "C11 feedback"))
            preview.append(_iphone_ms(row.get("preview_frames"), fps, "C11 preview"))
        else:
            content.append(number(row.get("lcp_ms"), "C10 LCP"))
            interaction.append(number(row.get("inp_ms"), "C10 INP"))
            feedback.append(number(row.get("feedback_ms"), "C11 feedback"))
            preview.append(number(row.get("preview_ms"), "C11 preview"))
        if not fps or data["cls_api_available"]:
            shifts.append(number(row.get("cls"), "C10 CLS"))
    require(
        nearest_rank(feedback, 95) <= 100 and nearest_rank(preview, 95) <= 1500,
        f"C11 {label}: timing threshold",
    )
    require(None not in feedback, f"C11 {label}: unresolved error or timeout retained")
    require(
        _p75(content) <= 2500
        and _p75(interaction) <= 200
        and (not shifts or _p75(shifts) <= 0.1),
        f"C10 {label}: p75 threshold",
    )


def validate_t073(data: Document) -> None:
    """Validate accessibility, user tasks and physical browser timing."""
    _, start, _ = _field_request(data)
    a11y = mapping(data.get("accessibility"), "C07 accessibility")
    require(a11y.get("standard") == "WCAG 2.2 AA", "C07 standard")
    flags(
        a11y,
        "C07",
        "automated_passed",
        "manual_passed",
        "keyboard_focus_passed",
        "screen_reader_passed",
        "errors_understandable",
    )
    for field in (
        "preregistered_scope_sha256",
        "automated_report_sha256",
        "manual_report_sha256",
    ):
        hash_value(a11y.get(field), f"C07 {field}")
    plan = mapping(data.get("plan"), "T073 plan")
    require(data.get("plan_sha256") == canonical_sha256(plan), "T073 protocol seal")
    require(
        instant(plan.get("registered_at"), "T073 plan") <= start,
        "T073 plan registered after start",
    )
    users = indexed(data.get("participants"), "C09 participants")
    user_plan = mapping(plan.get("user_tasks"), "C09 plan")
    exact_keys(users, user_plan, "C09 registered participants")
    require(len(users) == 3, "C09 needs three participants")
    unassisted = 0
    for key, user in users.items():
        tasks = indexed(user.get("tasks"), "C09 participant tasks")
        exact_keys(
            tasks,
            identifiers(user_plan[key], "C09 scheduled tasks"),
            "C09 preregistered tasks",
        )
        require(len(tasks) == 10, "C09 needs 10 tasks per participant")
        for task in tasks.values():
            require(
                task.get("critical_confusion") is False
                and task.get("critical_blocked") is False,
                "C09 critical confusion or blocked participant",
            )
            require(
                task.get("outcome")
                in {"unassisted_complete", "assisted", "incomplete", "timeout"},
                "C09 outcome",
            )
            unassisted += task["outcome"] == "unassisted_complete"
    _ratio(unassisted, 30, 95, "C09 unassisted task completion")
    clients = mapping(data.get("clients"), "C08 clients")
    schedules = mapping(plan.get("timing"), "C10 timing plan")
    exact_keys(clients, CLIENTS, "C08 clients")
    exact_keys(schedules, CLIENTS, "C10 timing plan")
    for client, row in clients.items():
        row = mapping(row, f"C08 {client}")
        flags(row, client, "physical_device", "lan", "trace_cross_checked")
        for field in (
            "model",
            "os_version",
            "browser",
            "browser_version",
            "orientation",
        ):
            text_value(row.get(field), f"C08 {client} {field}")
        if client == "iphone":
            require(
                row.get("platform") == "iOS"
                and row["browser"] == plan.get("iphone_browser"),
                "C08 chosen iPhone browser",
            )
            fps = count(row.get("fps"), "C10 iPhone fps")
            require(fps in {120, 240}, "C10 requires physical 120/240 fps recording")
            counters = identifiers(
                row.get("frame_observers"), "C10 iPhone frame observers"
            )
            require(len(counters) == 2, "C10 needs independent iPhone frame counters")
            hash_value(row.get("recording_sha256"), "C10 physical recording")
            require(
                type(row.get("cls_api_available")) is bool,
                "C10 iPhone CLS applicability",
            )
        else:
            require(
                row.get("platform") == "Terra" and row["browser"] == "Chrome",
                "C08 Terra Chrome",
            )
            fps = 0
        widths = mapping(row.get("widths"), "C08 reflow matrix")
        exact_keys(widths, map(str, WIDTHS), "C08 required reflow widths")
        for width, result in widths.items():
            flags(
                mapping(result, "C08 width"),
                f"C08 {client} {width}",
                "usable_reflow",
                "text_enlarged",
                "keyboard_checked",
                "session_checked",
                "playback_checked",
            )
        schedule = mapping(schedules[client], "C10 client schedule")
        exact_keys(schedule, ("cold", "hot"), "C10 cold/hot schedule")
        require(
            not set(schedule["cold"]) & set(schedule["hot"]),
            "C10 cold/hot attempts must be distinct",
        )
        for phase in ("cold", "hot"):
            _timing(row, schedule, client, phase, fps)


def validate_t074(data: Document) -> None:
    """Validate the fixed soak and each preregistered physical recovery."""
    package, start, end = _field_request(data)
    block = mapping(data.get("soak"), "C19 soak")
    first, last = interval(block, "C19 soak")
    observed = instant(data["observed_at"], "C19 observed_at")
    require(
        start <= first < last <= observed <= end
        and last - first >= timedelta(hours=72),
        "C19 needs a fixed >=72h block within seven days",
    )
    require(
        first.microsecond == last.microsecond == 0,
        "C19 block must use whole clock seconds",
    )
    require(
        instant(block.get("registered_at"), "C19 preregistration") < first,
        "C19 preregistration after start",
    )
    require(
        instant(block.get("sampler_start"), "C19 sampler start")
        <= first - timedelta(minutes=5)
        and last + timedelta(minutes=5)
        <= instant(block.get("sampler_end"), "C19 sampler end")
        <= observed,
        "C19 sampler margins missing",
    )
    flags(
        block,
        "C19",
        "physical",
        "candidate_frozen",
        "no_injections_in_block",
        "pause_policy_applied",
    )
    history = mapping(block.get("useful_seconds"), "C19 use by series")
    exact_keys(history, ("KZA", *ROOMS), "C19 KZA and all four microphones")
    duration = int((last - first).total_seconds())
    gaps = count(block.get("sampler_gap_seconds"), "C19 sampler gaps")
    array(block.get("spontaneous_failures"), "C19 spontaneous failures")
    array(block.get("authenticated_pauses"), "C19 authenticated pauses")
    for room, value in history.items():
        seconds = count(value, f"C19 {room} useful seconds")
        require(
            seconds <= duration - gaps and seconds * 1000 >= duration * 999,
            f"C19 {room}: availability below 99.9%",
        )
    prereg = mapping(data.get("fault_plan"), "C20 fault plan")
    hash_value(prereg.get("sha256"), "C20 preregistered fault plan document")
    require(
        instant(prereg.get("registered_at"), "C20 fault plan") <= start,
        "C20 fault plan registered late",
    )
    kinds = {"HA": "ha", "WS": "ws", "F1": "mic", "F2": "mic", "F3": "mic", "F4": "mic"}
    faults = indexed(data.get("faults"), "C20 faults")
    require(
        set(kinds) <= faults.keys(),
        "C20 missing HA, WS or preregistered Cocina F1–F4 evidence",
    )
    exact_keys(
        faults,
        identifiers(prereg.get("ids"), "C20 planned faults"),
        "C20 fault coverage",
    )
    for key, row in faults.items():
        kind = row.get("kind")
        require(
            kind in {"ha", "ws", "mic"} and kind == kinds.get(key, kind),
            "C20 fault kind",
        )
        require(
            key in package["q17"]["interventions"],
            f"C20 {key}: intervention outside Q17",
        )
        flags(
            row,
            f"C20 {key}",
            "physical",
            "authorized",
            "healthy_preconditions",
            "retries_bounded",
            "offline_h2_intact",
        )
        if key == "F3":
            flags(row, "C20 F3", "user_present", "separate_window")
        inject, stop = interval(row, "C20 physical fault")
        ready = instant(row.get("dependency_healthy_at"), "C20 dependency healthy")
        recovered = instant(row.get("recovered_at"), "C20 recovered")
        trace_start = instant(row.get("sampler_start"), "C20 sampler start")
        trace_end = instant(row.get("sampler_end"), "C20 sampler end")
        require(
            start <= inject <= ready <= recovered <= stop <= observed
            and (stop <= first or last <= inject),
            f"C20 {key}: fault overlaps soak or has inconsistent timing",
        )
        require(
            start <= trace_start <= inject - timedelta(minutes=5)
            and max(recovered, ready + timedelta(seconds=180)) + timedelta(minutes=5)
            <= trace_end
            <= observed,
            f"C20 {key}: sampler margins missing",
        )
        if key in {"F1", "F2", "F3"}:
            require(
                ready - inject >= timedelta(seconds=600 if key == "F2" else 30),
                f"C20 {key}: physical disconnection too short",
            )
        if key in {"F1", "F2"}:
            flags(row, f"C20 {key}", "same_usb_port")
        if key == "F4":
            flags(row, "C20 F4", "service_restarted_while_absent")
        mic = kind == "mic"
        require(
            (recovered - ready).total_seconds() <= (180 if mic else 60),
            f"C20 {key}: recovery deadline",
        )
        affected = row.get("room")
        if mic:
            require(
                affected in ROOMS and (key not in kinds or affected == "Cocina"),
                f"C20 {key}: microphone room",
            )
            flags(row, f"C20 {key}", "reenumerated", "new_capture_observed")
        else:
            require(affected is None, "C20 HA/WS healthy microphones are all four")
        heartbeat = mapping(row.get("healthy_inputs"), "C20 healthy-input evidence")
        exact_keys(heartbeat, set(ROOMS) - {affected}, "C20 healthy inputs")
        for sample in heartbeat.values():
            sample = mapping(sample, "C20 heartbeat")
            flags(sample, f"C20 {key}", "coverage_complete", "ever")
            number(sample.get("max_age_s"), f"C20 {key}: healthy heartbeat age", 8)
            number(sample.get("max_staleness_s"), f"C20 {key}: sampler staleness", 10)
        hash_value(row.get("trace_sha256"), f"C20 {key}: trace")
    cocina_order = [
        interval(faults[key], "C20 Cocina fault") for key in ("F1", "F2", "F3", "F4")
    ]
    require(
        all(
            previous[1] <= current[0]
            for previous, current in zip(cocina_order, cocina_order[1:])
        ),
        "C20 Cocina faults must follow the preregistered order",
    )


def validate_t075(data: Document) -> None:
    """Validate Epoch 2 promotion, complete V4 census and rollback readiness."""
    package, start, _ = _field_request(data)
    observed = instant(data["observed_at"], "T075 observed_at")
    adoption = mapping(data.get("adoption"), "T075 adoption")
    require(data.get("outcome") == "promoted", "T075 candidate promotion required")
    require(
        {"adopt", "promote"} <= set(package["q17"]["interventions"]),
        "T075 adoption/promotion outside Q17",
    )
    require(
        canonical_sha256(adoption.get("contract")) == canonical_sha256(ADOPTION),
        "T075 Q4 adoption mismatch",
    )
    require(
        count(adoption.get("adoption_count"), "T075 adoption count") == 1,
        "T075 direct unique adoption",
    )
    require(
        adoption.get("operator") == package["q17"]["operator"],
        "T075 authorized release operator",
    )
    require(
        adoption.get("release_review_sha256")
        == package["q17"]["release_review_sha256"],
        "T075 release review mismatch",
    )
    flags(
        adoption,
        "T075",
        "archive_verified",
        "reminder_rows_preserved",
        "legacy_database_unchanged",
        "ready_observed",
    )
    hash_value(adoption.get("receipt_sha256"), "T075 adoption receipt")
    adopted = instant(adoption.get("adopted_at"), "T075 adoption time")
    require(start <= adopted <= observed, "T075 adoption outside Q17 window")
    census = mapping(data.get("census"), "T075 V4 census")
    require(
        census.get("version") == "v4"
        and count(census.get("registered_producers"), "T075 producers") == 32,
        "T075 V4 needs exactly 32 producers",
    )
    flags(
        census,
        "T075 census",
        "global_inventory_complete",
        "producer_coverage_complete",
        "destination_enumeration_complete",
        "representation_classification_complete",
        "subject_erasure_complete",
    )
    require(
        census.get("uncovered_producers") == []
        and census.get("inventory_reasons") == [],
        "T075 census gaps",
    )
    hash_value(census.get("report_sha256"), "T075 census report")
    restore = mapping(data.get("restore"), "T075 compatible restore")
    flags(
        restore,
        "T075 restore",
        "compatible",
        "runtime_and_data_verified",
        "tombstones_preserved",
        "no_resurrection",
        "no_action_replay",
    )
    hash_value(restore.get("report_sha256"), "T075 restore report")
    rollback = mapping(data.get("rollback"), "T075 L4 evidence")
    require(
        canonical_sha256(rollback.get("contract")) == canonical_sha256(ROLLBACK),
        "T075 L4 plan mismatch",
    )
    flags(
        rollback,
        "T075 L4",
        "first_clean_close",
        "legacy_integrity_verified",
        "links_verified",
        "before_retirement_intent",
        "unmanaged_rehearsal_passed",
    )
    for field in ("clean_close_receipt_sha256", "rehearsal_sha256"):
        hash_value(rollback.get(field), f"T075 {field}")
    day_zero = instant(rollback.get("day_zero"), "T075 day zero")
    deadline = instant(rollback.get("deadline"), "T075 L4 deadline")
    require(
        adopted
        <= day_zero
        <= observed
        < deadline
        == day_zero + timedelta(seconds=L4_SECONDS),
        "T075 L4 window must end 14 days after first clean close",
    )
    hash_value(data.get("promotion_decision_sha256"), "T075 promotion decision")
    checks = mapping(data.get("integration_recheck"), "T075 C21 recheck")
    exact_keys(checks, INTEGRATIONS, "T075 C21 recheck")
    selection = package["q16"]["selection"]
    for key, row in checks.items():
        row = mapping(row, "T075 integration")
        require(
            canonical_sha256(row.get("selection")) == canonical_sha256(selection[key])
            and row.get("source_kind") == "real"
            and row.get("simulated") is False,
            f"T075 {key}: real Q16 source required",
        )
        flags(row, f"T075 {key}", "operated")
        hash_value(row.get("observation_sha256"), f"T075 {key}: real recheck")


VALIDATORS = {
    "T070": validate_t070,
    "T071": validate_t071,
    "T072": validate_t072,
    "T073": validate_t073,
    "T074": validate_t074,
    "T075": validate_t075,
}


def validate_field(task: str, value: object) -> Document:
    """Run a task validator and translate malformed evidence into a closed failure."""
    try:
        data = mapping(value, task)
        require(
            data.get("task") == task and data.get("kind") == "field",
            f"{task}: field evidence required",
        )
        candidate(data.get("candidate"))
        hash_value(data.get("evidence_sha256"), f"{task}: observation record")
        VALIDATORS[task](data)
    except EvidenceError:
        raise
    except (KeyError, ValueError, TypeError, OverflowError) as exc:
        raise EvidenceError(
            f"{task}: invalid or missing evidence ({type(exc).__name__})"
        ) from exc
    return data


def _markdown(package: Document, digest: str) -> str:
    encoded = json.dumps(package, ensure_ascii=False, indent=2, sort_keys=True)
    fence = "`" * max(
        3, 1 + max((len(run) for run in re.findall(r"`+", encoded)), default=0)
    )
    return (
        f"# Q16 y Q17 · Solicitud de aprobación\n\n"
        f"Estado pendiente de decisión humana. SHA-256 canónico `{digest}`.\n\n"
        "Q16 solicita las fuentes y los controles exactos. Q17 solicita el operador, la ventana "
        "y las intervenciones para el candidato sellado. Una ejecución de L4 exige su propia decisión "
        "con la frase literal «revivir datos borrados».\n\n"
        f"{fence}json\n{encoded}\n{fence}\n"
    )


def main(argv: list[str] | None = None) -> int:
    """Read private evidence and emit a package or a minimal verdict."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    render = commands.add_parser("render-q16-q17")
    render.add_argument("proposal", type=Path)
    render.add_argument("--release-review", type=Path, required=True)
    render.add_argument("--format", choices=("json", "markdown"), default="json")
    for task in VALIDATORS:
        commands.add_parser(f"validate-{task.lower()}").add_argument(
            "evidence", type=Path
        )
    args = parser.parse_args(argv)
    try:
        if args.command == "render-q16-q17":
            proposal, _ = read_json(args.proposal)
            review, review_sha256 = read_json(args.release_review)
            package = render_request(proposal, review, review_sha256)
            digest = canonical_sha256(package)
            if args.format == "markdown":
                sys.stdout.write(_markdown(package, digest))
            else:
                sys.stdout.write(
                    json.dumps(
                        {"package": package, "package_sha256": digest},
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n"
                )
        else:
            task = args.command.removeprefix("validate-").upper()
            evidence, _ = read_json(args.evidence)
            validate_field(task, evidence)
            sys.stdout.write(
                json.dumps({"task": task, "status": "evidence_valid"}) + "\n"
            )
    except EvidenceError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    except (ValueError, TypeError, KeyError, OverflowError, OSError) as exc:
        sys.stderr.write(f"Invalid or unreadable input ({type(exc).__name__})\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
