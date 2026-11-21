"""P7/P8 protocol fixtures and fail-closed regressions."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path

import pytest

from src.evaluation.freeze_manifest import canonical_sha256, seeded_order
from tools import p7_field_protocol as p7
from tools import p8_acceptance_audit as p8
from tools.release_manifest_review import rollback_plan

START = datetime(2026, 11, 23, tzinfo=UTC)
PANEL = ["voice-a", "voice-b", "voice-c"]


def at(*, days: int = 0, hours: int = 0, seconds: int = 0) -> str:
    return (START + timedelta(days=days, hours=hours, seconds=seconds)).isoformat()


def good(*names: str) -> dict:
    return dict.fromkeys(names, True)


def write_blob(directory: Path, content: bytes) -> str:
    digest = sha256(content).hexdigest()
    (directory / digest).write_bytes(content)
    return digest


def json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False).encode()


@pytest.fixture
def field(tmp_path):
    def proof(label):
        return write_blob(tmp_path, label.encode())

    identity = dict(
        git_commit="a" * 40, archive_sha256="b" * 64, schema_contract_epoch=2
    )
    review = dict(
        manifest_version=1,
        git_commit=identity["git_commit"],
        artifact=dict(
            sha256=identity["archive_sha256"], python_abi="cp313", python_sources=0
        ),
        schema_contract=dict(epoch=2, sha256=p7.SCHEMA_DIGEST),
        external_runtime_lock_sha256=proof("lock"),
        rollback=rollback_plan(),
        synthetic_adoption_preflight=dict(
            schema_only="passed",
            legacy_current_verified=True,
            frozen_database_unchanged=True,
        ),
    )
    review_digest = write_blob(tmp_path, json_bytes(review))
    selection = {
        "I01": {"energy": ["sensor.energy"], "power": ["sensor.power"]},
        "I02": "todo.home",
        "I03": {"kind": "ics", "source": "home.ics"},
        "I04": {"entity": "media_player.test", "controls": ["play", "pause", "volume"]},
        "I05": {"entities": ["binary_sensor.camera_online"], "video": False},
    }
    proposal = dict(
        candidate=identity,
        q16=dict(
            inventory_sha256=proof("Q16 inventory"),
            selection=selection,
            differences="Inventory reviewed",
        ),
        q17=dict(
            operator="operator",
            window=dict(start=at(), end=at(days=7)),
            interventions=["adopt", "promote", "HA", "WS", "F1", "F2", "F3", "F4"],
        ),
    )
    request = p7.render_request(proposal, review, review_digest)

    def approval(question):
        decision = dict(
            question=question,
            decision="approved",
            package_sha256=canonical_sha256(request),
            approved_at=at(seconds=-3600),
        )
        return dict(
            **decision, receipt_sha256=write_blob(tmp_path, json_bytes(decision))
        )

    q16, q17 = approval("Q16"), approval("Q17")
    rows = {
        key: dict(
            task=key,
            kind="field",
            candidate=deepcopy(identity),
            request=deepcopy(request),
            q17_approval=deepcopy(q17),
            observed_at=at(days=5),
            evidence_sha256=proof(key),
        )
        for key in p7.P7_CRITERIA
    }
    microphones = {
        room: dict(
            channel=i,
            usb_port="7-5.3" if room == "Cocina" else f"7-4.{i}",
            stimulus_room=room,
            observed_room=room,
            capture_state="running",
            heartbeat=dict(
                ever=True,
                coverage_complete=True,
                sampled_seconds=300,
                max_age_s=8,
                max_staleness_s=10,
            ),
            **good("physical_probe", "present", "hw_ptr_advanced", "channel_match"),
        )
        for i, room in enumerate(p7.ROOMS)
    }
    integrations = {
        key: dict(
            selection=deepcopy(value),
            source_kind="real",
            simulated=False,
            observation_sha256=proof(key),
            **good(
                "operated",
                "contract_passed",
                "rejection_passed",
                "expiry_passed",
                "permissions_passed",
            ),
        )
        for key, value in selection.items()
    }
    integrations["I02"].update(
        good("idempotency_passed", "reconciliation_passed", "personal_lists_local")
    )
    integrations["I03"]["read_only"] = True
    integrations["I04"]["selected_controls_verified"] = True
    integrations["I05"].update(technical_only=True, video_accessed=False)
    rows["T070"].update(
        observed_at=at(hours=2),
        q16_approval=q16,
        microphones=microphones,
        integrations=integrations,
    )
    acoustic(rows["T071"], rows["T072"], proof)
    browser(rows["T073"], proof)
    soak(rows["T074"], proof)
    rows["T075"].update(
        observed_at=at(days=6),
        outcome="promoted",
        adoption=dict(
            contract=deepcopy(p7.ADOPTION),
            adoption_count=1,
            operator="operator",
            adopted_at=at(hours=1),
            release_review_sha256=review_digest,
            receipt_sha256=proof("adoption"),
            **good(
                "archive_verified",
                "reminder_rows_preserved",
                "legacy_database_unchanged",
                "ready_observed",
            ),
        ),
        census=dict(
            version="v4",
            registered_producers=32,
            uncovered_producers=[],
            inventory_reasons=[],
            report_sha256=proof("census"),
            **good(
                "global_inventory_complete",
                "producer_coverage_complete",
                "destination_enumeration_complete",
                "representation_classification_complete",
                "subject_erasure_complete",
            ),
        ),
        restore=dict(
            report_sha256=proof("restore"),
            **good(
                "compatible",
                "runtime_and_data_verified",
                "tombstones_preserved",
                "no_resurrection",
                "no_action_replay",
            ),
        ),
        rollback=dict(
            contract=deepcopy(p7.ROLLBACK),
            day_zero=at(days=2),
            deadline=at(days=16),
            clean_close_receipt_sha256=proof("W9"),
            rehearsal_sha256=proof("L4"),
            **good(
                "first_clean_close",
                "legacy_integrity_verified",
                "links_verified",
                "before_retirement_intent",
                "unmanaged_rehearsal_passed",
            ),
        ),
        promotion_decision_sha256=proof("promote"),
        integration_recheck={
            key: dict(
                selection=value,
                operated=True,
                source_kind="real",
                observation_sha256=proof(f"{key} recheck"),
            )
            for key, value in selection.items()
        },
    )
    return dict(
        rows=rows,
        identity=identity,
        proposal=proposal,
        review=review,
        review_digest=review_digest,
        q16=q16,
        q17=q17,
        proof=proof,
    )


def acoustic(calibration, reserve, proof):
    def clip(key, split, day, second, speakers):
        return dict(
            clip_id=key,
            split=split,
            audio_sha256=canonical_sha256(["pcm", key]),
            source_sha256=canonical_sha256(["source", key]),
            start=at(days=day, seconds=second),
            end=at(days=day, seconds=second + 1),
            speaker_refs=speakers,
            session_id=split,
            consent_ref=split,
            **good("clear", "deletion_checked"),
        )

    def consent(split):
        return dict(
            id=split,
            session_id=split,
            purpose="consented_calibration_and_evaluation",
            granted=True,
            speaker_refs=PANEL,
            granted_at=at(seconds=-60),
            expires_at=at(days=20),
            revoked_at=None,
            receipt_sha256=proof(f"{split} receipt"),
        )

    clips = [
        clip(f"cal{i}", "calibration", 1, i * 2, [PANEL[i % 3]]) for i in range(30)
    ]
    inventory = deepcopy(clips)
    calibration.update(
        clips=clips, receipts=[consent("calibration")], observed_at=at(days=2, hours=13)
    )
    plan = dict(
        registered_at=at(),
        panel=PANEL,
        conditions=list(p7.CONDITIONS),
        critical_conditions=["noise", "tv"],
        task_families=["home"],
        simple_families=["home"],
        scheduled={},
    )

    def schedule(section, n, make_result, *, extra_fields=()):
        rows = []
        for i in range(n):
            condition = (
                ("clear" if i < 200 else "noise" if i < 250 else "tv")
                if section == "requests"
                else p7.CONDITIONS[i % 3]
            )
            strata = dict(
                room=p7.ROOMS[i % 4],
                distance_m=1 + 2 * ((i // 3) % 2),
                participant=PANEL[(i // 4) % 3],
                condition=condition,
            )
            key = f"{section}-{i}"
            row = dict(id=key, **strata, **make_result(i))
            for extra in extra_fields:
                strata[extra] = row[extra]
            rows.append(row)
            if section == "requests":
                inventory.append(
                    clip(key, "reserved", 2, 1000 + i * 2, [strata["participant"]])
                )
            plan["scheduled"].setdefault(section, {})[key] = strata
        reserve[section] = rows

    score = dict(
        words=10,
        substitutions=0,
        insertions=0,
        deletions=0,
        characters=50,
        character_edits=0,
        human_aligned=True,
    )
    schedule(
        "requests",
        300,
        lambda _: dict(
            family="home",
            recognition=dict(score),
            effect_verified=True,
            effect_sha256=proof("effects"),
        ),
        extra_fields=("family",),
    )
    schedule("wakes", 120, lambda _: dict(outcome="accepted"))
    schedule(
        "simple",
        120,
        lambda i: dict(
            family="home",
            cache="hot" if i % 2 else "cold",
            rapid_ha=True,
            voice_ms=800,
            ha_ms=2000,
            earcon_excluded=True,
            clock_bound_applied=True,
        ),
        extra_fields=("family", "cache", "rapid_ha"),
    )
    schedule(
        "deliveries",
        120,
        lambda _: dict(
            outcome="complete",
            **good(
                "state_matches_observation",
                "queue_and_pcm_recorded",
                "audible_end_verified",
            ),
        ),
    )
    schedule(
        "interruptions",
        48,
        lambda _: dict(
            stop_ms=250,
            **good(
                "physical",
                "barge_in",
                "concurrent_capture",
                "clock_bound_applied",
                "long_uncached",
            ),
        ),
    )
    schedule(
        "listening_pairs",
        48,
        lambda i: dict(length="long" if i % 2 else "short", ratings={}),
        extra_fields=("length",),
    )
    listen = {row["id"]: row for row in reserve["listening_pairs"]}
    for i, key in enumerate(seeded_order(listen, purpose="listening", seed=0)):
        listen[key]["ratings"] = {
            v: dict(first_variant=(i + j) % 2, severe=False, candidate=5, reference=5)
            for j, v in enumerate(PANEL)
        }
    reserve["wake"] = dict(
        clock="house_clock",
        house_seconds=172800,
        useful_intervals=[
            dict(
                start=at(hours=2), end=at(days=2, hours=2), useful_inputs=list(p7.ROOMS)
            )
        ],
        raw_by_room=dict.fromkeys(p7.ROOMS, 30),
        emissions=[
            dict(
                id=f"emission-{i}",
                rooms=[row["room"]],
                outcome="true",
                attempt_id=row["id"],
                correlation_sha256=proof("wake correlation interval"),
            )
            for i, row in enumerate(reserve["wakes"])
        ],
        **good("emissions_inventory_complete", "deduplication_physically_corroborated"),
    )
    h3 = {}
    for i, kind in enumerate(
        ["clear"] * 90
        + [k for k in ("short", "tv", "echo", "overlap") for _ in range(10)]
    ):
        key = f"h3-{i}"
        voices = PANEL[:2] if kind == "overlap" else [PANEL[i % 3]]
        inventory.append(clip(key, "reserved", 2, 2000 + i * 2, voices))
        h3[key] = (
            dict(
                id=key,
                condition=kind,
                room=p7.ROOMS[i % 4],
                distance_m=p7.DISTANCES[i % 2],
            ),
            None if kind in {"short", "overlap"} else voices[0],
        )
    reserve["h3"] = dict(
        clips=[row for row, _ in h3.values()],
        runs=[
            dict(
                id="original" if seed is None else str(seed),
                started_empty=True,
                assignments=[
                    dict(id=key, group=h3[key][1])
                    for key in (
                        h3
                        if seed is None
                        else seeded_order(h3, purpose="h3", seed=seed)
                    )
                ],
            )
            for seed in (None, *range(10))
        ],
    )
    plan["h3_inventory_sha256"] = canonical_sha256(reserve["h3"]["clips"])
    seals = {
        key: proof(key)
        for key in (
            "runtime",
            "models",
            "dsp",
            "configuration",
            "corpus_contract",
            "exclusions",
            "instrumentation",
        )
    }
    seals.update(
        reserve_inventory=canonical_sha256(inventory),
        strata_plan=canonical_sha256(plan),
    )
    for row in (calibration, reserve):
        row.update(
            frozen_at=at(days=2, hours=12),
            reserve_opened_at=at(days=3),
            evaluation_close=at(days=6),
            seals=deepcopy(seals),
            acoustic_sha256=canonical_sha256(seals),
        )
    reserve.update(
        protocol=p7.PROTOCOL,
        protocol_sha256=sha256(p7.P6_DOCUMENT.read_bytes()).hexdigest(),
        clips=inventory,
        receipts=[consent("calibration"), consent("reserved")],
        plan=plan,
        listening=good("blind_until_locked", "comparable_paired_audio", "consented"),
        voluntary_cancellations=[],
        **good("blind_adjudication", "deletion_checked", "original_result_retained"),
    )


def browser(row, proof):
    tasks = [f"task-{i}" for i in range(10)]
    plan = dict(
        registered_at=at(),
        iphone_browser="Selected browser",
        user_tasks=dict.fromkeys(PANEL, tasks),
        timing={},
    )
    row["participants"] = [
        dict(
            id=key,
            tasks=[
                dict(
                    id=task,
                    critical_confusion=False,
                    critical_blocked=False,
                    outcome="unassisted_complete",
                )
                for task in tasks
            ],
        )
        for key in PANEL
    ]
    row["accessibility"] = dict(
        standard="WCAG 2.2 AA",
        preregistered_scope_sha256=proof("a11y scope"),
        automated_report_sha256=proof("a11y automated"),
        manual_report_sha256=proof("a11y manual"),
        **good(
            "automated_passed",
            "manual_passed",
            "keyboard_focus_passed",
            "screen_reader_passed",
            "errors_understandable",
        ),
    )
    row["clients"] = {}
    for client in p7.CLIENTS:
        iphone = client == "iphone"
        info = dict(
            platform="iOS" if iphone else "Terra",
            model="physical model",
            os_version="1",
            browser_version="1",
            orientation="portrait and landscape",
            browser=plan["iphone_browser"] if iphone else "Chrome",
            widths={
                str(width): good(
                    "usable_reflow",
                    "text_enlarged",
                    "keyboard_checked",
                    "session_checked",
                    "playback_checked",
                )
                for width in p7.WIDTHS
            },
            **good("physical_device", "lan", "trace_cross_checked"),
        )
        if iphone:
            info.update(
                fps=120,
                frame_observers=["a", "b"],
                recording_sha256=proof("iphone video"),
                cls_api_available=False,
            )
        for phase in ("cold", "hot"):
            ids = [f"{phase}-{i}" for i in range(40)]
            plan["timing"].setdefault(client, {})[phase] = ids
            metrics = (
                dict(
                    content_frames=[299, 300],
                    interaction_frames=[24, 24],
                    feedback_frames=[12, 12],
                    preview_frames=[180, 180],
                )
                if iphone
                else dict(
                    lcp_ms=2500, inp_ms=200, cls=0.1, feedback_ms=100, preview_ms=1500
                )
            )
            info[phase] = [
                dict(
                    id=key,
                    status="ok",
                    preview_duration_s=6,
                    touch_target_jump=False,
                    **deepcopy(metrics),
                    **good("useful_paint_observed", "audio_playable_by_native_gesture"),
                )
                for key in ids
            ]
        row["clients"][client] = info
    row.update(plan=plan, plan_sha256=canonical_sha256(plan))


def soak(row, proof):
    row["soak"] = dict(
        start=at(days=1),
        end=at(days=4),
        registered_at=at(),
        sampler_start=at(days=1, seconds=-300),
        sampler_end=at(days=4, seconds=300),
        useful_seconds=dict.fromkeys(("KZA", *p7.ROOMS), 259200 - 259),
        sampler_gap_seconds=0,
        spontaneous_failures=[],
        authenticated_pauses=[],
        **good(
            "physical",
            "candidate_frozen",
            "no_injections_in_block",
            "pause_policy_applied",
        ),
    )
    ids = ("HA", "WS", "F1", "F2", "F3", "F4")
    row["fault_plan"] = dict(
        sha256=proof("C20 plan"), registered_at=at(), ids=list(ids)
    )
    row["faults"] = []
    for i, key in enumerate(ids):
        mic = key.startswith("F")
        start = 3600 * (i + 3)
        up = start + (600 if key == "F2" else 30)
        recovery = up + (180 if mic else 60)
        row["faults"].append(
            dict(
                id=key,
                kind="mic" if mic else key.lower(),
                room="Cocina" if mic else None,
                start=at(seconds=start),
                end=at(seconds=recovery + 1),
                dependency_healthy_at=at(seconds=up),
                recovered_at=at(seconds=recovery),
                sampler_start=at(seconds=start - 300),
                sampler_end=at(seconds=up + 480),
                trace_sha256=proof(f"{key} trace"),
                healthy_inputs={
                    room: dict(
                        coverage_complete=True,
                        ever=True,
                        max_age_s=8,
                        max_staleness_s=10,
                    )
                    for room in p7.ROOMS
                    if not (mic and room == "Cocina")
                },
                **good(
                    "physical",
                    "authorized",
                    "healthy_preconditions",
                    "retries_bounded",
                    "offline_h2_intact",
                    "reenumerated",
                    "new_capture_observed",
                    "user_present",
                    "separate_window",
                    "same_usb_port",
                    "service_restarted_while_absent",
                ),
            )
        )


@pytest.mark.parametrize("task", p7.P7_CRITERIA)
def test_p7_valid(task, field):
    assert p7.validate_field(task, field["rows"][task])["task"] == task


@pytest.mark.parametrize(
    ("task", "path", "value", "message"),
    [
        (
            "T070",
            ("microphones", "Cocina", "observed_room"),
            "Hall",
            "channel mismatch",
        ),
        (
            "T070",
            ("microphones", "Cocina", "heartbeat", "ever"),
            False,
            "heartbeat.ever",
        ),
        ("T070", ("integrations", "I03", "simulated"), True, "real source required"),
        ("T070", ("integrations", "I04", "operated"), False, "operated"),
        ("T071", ("receipts", 0, "revoked_at"), at(days=2), "consent expired"),
        ("T071", ("receipts", 0, "expires_at"), at(days=2), "consent expired"),
        ("T071", ("frozen_at",), at(days=3), "frozen before"),
        ("T072", ("wake", "house_seconds"), 691200, "microphone-hours"),
        ("T072", ("wake", "clock"), "microphone_hours", "house-clock"),
        (
            "T072",
            ("wake", "emissions", 0, "attempt_id"),
            "unknown-wake",
            "wake correlation",
        ),
        ("T072", ("simple", 0, "earcon_excluded"), False, "earcon_excluded"),
        ("T072", ("interruptions", 0, "physical"), False, "physical"),
        ("T072", ("h3", "runs", 10, "started_empty"), False, "started_empty"),
        ("T073", ("accessibility", "screen_reader_passed"), False, "screen_reader"),
        (
            "T073",
            ("clients", "iphone", "cold", 0, "status"),
            "timeout",
            "timeout retained",
        ),
        ("T073", ("clients", "iphone", "fps"), 60, "120/240 fps"),
        (
            "T073",
            ("participants", 0, "tasks", 0, "critical_blocked"),
            True,
            "blocked participant",
        ),
        ("T074", ("soak", "useful_seconds", "Cocina"), 258940, "Cocina: availability"),
        (
            "T074",
            ("faults", 0, "recovered_at"),
            at(seconds=3 * 3600 + 91),
            "recovery deadline",
        ),
        (
            "T074",
            ("faults", 2, "healthy_inputs", "Hall", "max_age_s"),
            8.01,
            "heartbeat age",
        ),
        (
            "T075",
            ("census", "global_inventory_complete"),
            False,
            "global_inventory_complete",
        ),
        ("T075", ("restore", "no_resurrection"), False, "no_resurrection"),
        ("T075", ("rollback", "deadline"), at(days=17), "14 days"),
        ("T075", ("candidate", "schema_contract_epoch"), 3, "EPOCH = 2"),
        ("T075", ("integration_recheck", "I03", "operated"), False, "operated"),
    ],
)
def test_bad_field_evidence_is_rejected(task, path, value, message, field):
    row = field["rows"][task]
    target = row
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(p7.EvidenceError, match=message):
        p7.validate_field(task, row)


def test_failed_room_even_when_global_wer_passes(field):
    row = field["rows"]["T072"]
    failed = next(
        case
        for case in row["requests"]
        if case["room"] == "Cocina" and case["condition"] == "clear"
    )
    failed["recognition"]["insertions"] = 60
    assert 60 / (200 * 10) < 0.1
    with pytest.raises(p7.EvidenceError, match="Cocina.*WER"):
        p7.validate_field("T072", row)


def test_failed_adverse_condition_in_one_room_is_not_blended(field):
    row = field["rows"]["T072"]
    failed = next(
        case
        for case in row["requests"]
        if case["room"] == "Cocina" and case["condition"] == "tv"
    )
    failed["recognition"]["insertions"] = 30
    assert 30 / (50 * 10) < 0.2
    with pytest.raises(p7.EvidenceError, match="Cocina.*WER"):
        p7.validate_field("T072", row)


def test_failed_listener_median_cannot_be_averaged(field):
    row = field["rows"]["T072"]
    for i, pair in enumerate(row["listening_pairs"]):
        pair["ratings"][PANEL[0]]["candidate"] = 3 if i < 24 else 4
    with pytest.raises(p7.EvidenceError, match="median <4"):
        p7.validate_field("T072", row)


def test_excluding_timeout_does_not_silently_improve_p95(field):
    row = field["rows"]["T073"]
    attempts = row["clients"]["iphone"]["cold"]
    attempts.append(dict(id="timeout-extra", status="timeout"))
    row["plan"]["timing"]["iphone"]["cold"].append("timeout-extra")
    row["plan_sha256"] = canonical_sha256(row["plan"])
    attempts.pop()  # Forty recorded successes still meet the numerical minimum.
    with pytest.raises(p7.EvidenceError, match="missing preregistered timing attempts"):
        p7.validate_field("T073", row)


def test_uncertain_wake_counts_against_false_rate(field):
    row = field["rows"]["T072"]
    for i in range(5):
        room = p7.ROOMS[i % 4]
        row["wake"]["emissions"].append(
            dict(id=f"uncertain-{i}", rooms=[room], outcome="uncertain")
        )
        row["wake"]["raw_by_room"][room] += 1
    with pytest.raises(p7.EvidenceError, match="conservative false wake rate"):
        p7.validate_field("T072", row)


def test_h3_mixed_pair_fails_even_if_other_seeds_pass(field):
    row = field["rows"]["T072"]
    for assignment in row["h3"]["runs"][2]["assignments"]:
        if assignment["id"] == "h3-0":
            assignment["group"] = PANEL[1]
    with pytest.raises(p7.EvidenceError, match="mixed voices"):
        p7.validate_field("T072", row)


def test_q16_q17_cli_formats_and_epoch(field, tmp_path, capsys):
    proposal = tmp_path / "proposal.json"
    proposal.write_bytes(json_bytes(field["proposal"]))
    flags = [str(proposal), "--release-review", str(tmp_path / field["review_digest"])]
    assert p7.main(["render-q16-q17", *flags]) == 0
    rendered = json.loads(capsys.readouterr().out)
    assert rendered["package_sha256"] == canonical_sha256(rendered["package"])
    assert rendered["package"]["q17"]["adoption"] == p7.ADOPTION
    assert p7.main(["render-q16-q17", *flags, "--format", "markdown"]) == 0
    assert "revivir datos borrados" in capsys.readouterr().out
    field["proposal"]["candidate"]["schema_contract_epoch"] = 1
    proposal.write_bytes(json_bytes(field["proposal"]))
    assert p7.main(["render-q16-q17", *flags]) == 1
    assert "EPOCH = 2" in capsys.readouterr().err


@pytest.fixture
def acceptance(field, tmp_path):
    proof = field["proof"]
    identity = field["identity"]
    roadmap, _ = p7.read_json(p8.DEFAULT_MANIFEST)
    for task in roadmap["tasks"]:
        task["status"] = "accepted" if task["human_gate"] else "verified_local"
        task["evidence"] = [proof(task["id"])]
    for entry in roadmap["finding_trace"]:
        entry["status"] = "closed"
    for entry in roadmap["areas"]:
        entry["status"] = "accepted"
    roadmap["field_authorization_evidence"] = [
        field["q16"]["receipt_sha256"],
        field["q17"]["receipt_sha256"],
    ]
    local = {
        task: dict(
            task=task, kind="local", candidate=identity, evidence_sha256=proof(task)
        )
        for task in p8.P6_TASKS
    }
    local["T060"].update(
        included_families=["home"],
        factory_families=[
            dict(
                family="home",
                mandatory=10,
                passed=10,
                **good(
                    "real_factory",
                    "actor_policy_persistence_effect_response",
                    "restart_and_faults_checked",
                ),
            )
        ],
        negative_cases=200,
        invariant_violations=0,
        h2=dict(
            repetitions=10,
            heartbeat_ms=5,
            delay_p95_ms=25,
            delay_max_ms=75,
            cancel_ms=100,
            simulated_stop_ms=200,
            cleanup_ms=1000,
            **good(
                "pcm_exact",
                "bounded_queues",
                "long_responses",
                "retained_native_work",
                "producers_200ms",
            ),
        ),
        **good(
            "required_suite_passed",
            "no_required_xfail",
            "tier2_passed",
            "compiled_solo_passed",
            "new_warnings_reviewed",
        ),
    )
    local["T061"].update(
        soak_seconds=259200,
        useful_seconds=dict.fromkeys(("KZA", *p7.ROOMS), 259200),
        recovery_seconds=dict(ha=60, ws=60, mic=180),
        **good(
            "candidate_frozen",
            "privacy_checks_passed",
            "mute_durable_no_voice_reactivation",
            "late_producer_races_checked",
            "h2_preserved",
            "faults_outside_soak",
            "healthy_inputs_kept_heartbeat",
        ),
    )
    local["T062"].update(
        protocol=p7.PROTOCOL,
        protocol_sha256=field["rows"]["T072"]["protocol_sha256"],
        **good("evaluator_tests_passed", "independent_reserve_plan_sealed"),
    )
    local["T063"].update(
        rebuilt_archive_sha256=identity["archive_sha256"],
        **good(
            "builds_identical",
            "dependencies_reproducible",
            "runtime_outside_sources",
            "startup_config_verified",
            "training_disabled_or_accepted",
            "compatible_data_restored",
            "runtime_rollback_verified",
            "no_resurrection",
            "no_action_replay",
        ),
    )
    local["T064"].update(
        release_review_sha256=field["review_digest"],
        **good("release_review_verify_passed", "rollback_script_reviewed"),
    )
    rows = local | field["rows"]
    reviews = {}
    for criterion in roadmap["criteria"]:
        key = criterion["id"]
        review = proof(f"{key} reviewed")
        criterion.update(status="accepted", evidence=[review])
        reviews[key] = dict(
            verdict="accepted",
            independent=True,
            original_observations_reviewed=True,
            reviewer="reviewer",
            review_sha256=review,
            sources={
                task: rows[task]["evidence_sha256"] if task in rows else proof(task)
                for task in p8.REQUIRED_SOURCES[key]
            },
        )
    reviews["C06"]["observability"] = dict(
        included_components=["panel"],
        components=[
            dict(
                id="panel",
                **good(
                    "real_source",
                    "actual_timestamp_and_expiry",
                    "backend_effect_verified",
                ),
            )
        ],
        verified_scenarios=[
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
        ],
    )
    handover = dict(
        reviewer="reviewer",
        independent_review_sha256=proof("independent review"),
        handover_sha256=proof("handover"),
        **good(
            "independent_review_complete",
            "home_guide_updated",
            "runbooks_updated",
            "matrix_updated",
            "remaining_limits_documented",
            "human_final_acceptance",
        ),
    )
    human = dict(
        decision="accepted",
        candidate=identity,
        profile_sha256=p8.PROFILE_SHA256,
        handover_sha256=handover["handover_sha256"],
    )
    handover["human_acceptance_sha256"] = write_blob(tmp_path, json_bytes(human))
    for task, key in (
        ("T080", "independent_review_sha256"),
        ("T081", "handover_sha256"),
        ("T082", "human_acceptance_sha256"),
    ):
        next(row for row in roadmap["tasks"] if row["id"] == task)["evidence"].append(
            handover[key]
        )
    return roadmap, dict(
        candidate=identity, tasks=rows, criterion_reviews=reviews, handover=handover
    )


def test_complete_audit_and_modes_are_read_only(acceptance, tmp_path, capsys):
    roadmap, bundle = acceptance
    result = p8.audit(roadmap, bundle, tmp_path)
    assert result["blockers"] == []
    assert result["T082"] == "ready_for_closure"
    assert len(result["areas"]) == 7
    assert all(value["status"] == "accepted" for value in result["areas"].values())
    assert set(result["milestones"]) == set(p8.HITOS)
    assert len(result["criteria"]) == 21
    roadmap_path, bundle_path = tmp_path / "roadmap.json", tmp_path / "bundle.json"
    original = json_bytes(roadmap)
    roadmap_path.write_bytes(original)
    bundle_path.write_bytes(json_bytes(bundle))
    args = [
        "--roadmap",
        str(roadmap_path),
        "--bundle",
        str(bundle_path),
        "--artifacts",
        str(tmp_path),
    ]
    assert p8.main(["check", *args]) == 0
    assert json.loads(capsys.readouterr().out)["T082"] == "ready_for_closure"
    assert roadmap_path.read_bytes() == original
    roadmap["criteria"][0]["status"] = "not_evaluated"
    blocked = json_bytes(roadmap)
    roadmap_path.write_bytes(blocked)
    assert p8.main(["check", *args]) == 1
    assert json.loads(capsys.readouterr().out)["T082"] == "blocked"
    assert p8.main(["report", *args]) == 0
    assert json.loads(capsys.readouterr().out)["T082"] == "blocked"
    assert roadmap_path.read_bytes() == blocked


@pytest.mark.parametrize(
    "mutation",
    [
        "criterion unevaluated",
        "open finding",
        "required task pending",
        "physical task local",
        "missing field",
        "simulated calendar",
        "offline media",
        "failed stratum",
        "missing artifact",
        "no human acceptance",
        "self review",
        "release mismatched",
        "different request",
    ],
)
def test_acceptance_blocks_any_incomplete_condition(mutation, acceptance, tmp_path):
    roadmap, bundle = acceptance
    rows = bundle["tasks"]
    if mutation == "criterion unevaluated":
        roadmap["criteria"][6]["status"] = "not_evaluated"
    elif mutation == "open finding":
        roadmap["finding_trace"][0]["status"] = "open"
    elif mutation == "required task pending":
        next(task for task in roadmap["tasks"] if task["id"] == "T054")["status"] = (
            "waiting_external"
        )
    elif mutation == "physical task local":
        next(task for task in roadmap["tasks"] if task["id"] == "T070")["status"] = (
            "verified_local"
        )
    elif mutation == "missing field":
        del rows["T073"]
    elif mutation == "simulated calendar":
        rows["T070"]["integrations"]["I03"]["source_kind"] = "simulator"
    elif mutation == "offline media":
        rows["T070"]["integrations"]["I04"]["operated"] = False
    elif mutation == "failed stratum":
        rows["T072"]["requests"][3]["recognition"]["insertions"] = 60
    elif mutation == "missing artifact":
        (tmp_path / rows["T074"]["evidence_sha256"]).unlink()
    elif mutation == "no human acceptance":
        bundle["handover"]["human_final_acceptance"] = False
    elif mutation == "self review":
        bundle["criterion_reviews"]["C09"]["reviewer"] = rows["T070"]["request"]["q17"][
            "operator"
        ]
    elif mutation == "release mismatched":
        rows["T064"]["release_review_sha256"] = rows["T061"]["evidence_sha256"]
    elif mutation == "different request":
        row = rows["T073"]
        row["request"]["q17"]["operator"] = "different operator"
        row["q17_approval"]["package_sha256"] = canonical_sha256(row["request"])
    result = p8.audit(roadmap, bundle, tmp_path)
    assert result["T082"] == "blocked"
    assert result["milestones"]["H6"]["status"] == "blocked"
    assert result["blockers"]


@pytest.mark.parametrize("field_name", ("acceptance_targets_sha256", "profile_sha256"))
def test_changed_pinned_roadmap_hash_blocks(field_name, acceptance, tmp_path):
    roadmap, bundle = acceptance
    roadmap[field_name] = "0" * 64
    with pytest.raises(p7.EvidenceError, match="frozen .* mismatch"):
        p8.audit(roadmap, bundle, tmp_path)


def test_rehashing_relaxed_threshold_cannot_bypass_frozen_pins(acceptance, tmp_path):
    roadmap, bundle = acceptance
    roadmap["criteria"][11]["target"] = "WER unlimited"
    roadmap["acceptance_targets_sha256"], roadmap["profile_sha256"] = (
        p8.contract_hashes(roadmap)
    )
    with pytest.raises(p7.EvidenceError, match="frozen .* mismatch"):
        p8.audit(roadmap, bundle, tmp_path)


def test_removing_finding_cannot_bypass_roadmap_scope(acceptance, tmp_path):
    roadmap, bundle = acceptance
    roadmap["finding_trace"].pop()
    with pytest.raises(p7.EvidenceError, match="scope changed"):
        p8.audit(roadmap, bundle, tmp_path)


def test_strict_json_blocks_duplicates(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"epoch": 3, "epoch": 2}')
    with pytest.raises(p7.EvidenceError, match="duplicate JSON key"):
        p7.read_json(path)


def test_decision_receipt_must_match_the_package(acceptance, tmp_path):
    roadmap, bundle = acceptance
    original = bundle["tasks"]["T070"]["q16_approval"]
    different = {
        **original,
        "receipt_sha256": write_blob(tmp_path, b'{"decision":"approved"}'),
    }
    roadmap["field_authorization_evidence"][0] = different["receipt_sha256"]
    bundle["tasks"]["T070"]["q16_approval"] = different
    result = p8.audit(roadmap, bundle, tmp_path)
    assert result["T082"] == "blocked"
    assert any(
        "recorded decision does not match" in item for item in result["blockers"]
    )


def test_field_cli_keeps_failed_result_nonzero(field, tmp_path, capsys):
    path = tmp_path / "t070.json"
    row = field["rows"]["T070"]
    path.write_bytes(json_bytes(row))
    assert p7.main(["validate-t070", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "evidence_valid"
    row["integrations"]["I04"]["operated"] = False
    path.write_bytes(json_bytes(row))
    assert p7.main(["validate-t070", str(path)]) == 1
    assert "I04.operated" in capsys.readouterr().err
