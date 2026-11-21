"""T024 feature publisher metadata and real V4 journal round-trips."""

from contextlib import AsyncExitStack
from copy import deepcopy
from hashlib import sha256
from pathlib import PurePath
from uuid import uuid4

import pytest

from src.core import runtime_admission
from src.dashboard.live_erasure import LIVE_TARGET, REST_TARGET, VOICE_REVIEW_TARGET
from src.memory import output_registration as base
from src.memory import output_registration_v2 as v2
from src.memory import output_registration_v4 as v4
from src.memory.output_registration_features import (
    FEATURES_PRODUCERS,
    GRAMMAR_SPECS,
    dashboard_transport_plan,
    features_aggregate,
    home_integrations_plan,
    household_features_plan,
    latency_plan,
)
from tests.unit.memory.test_output_publisher_seam import (  # noqa: F401
    _compose_legacy_fixture,
    decoder,
    delegated_decoder_world,
    local_grammar_table,
)

ROLE_IDS = (
    ("feature_manager", "timers_runtime", "smart_notifications", "alerts_runtime", "intercom_delivery"),
    ("ha_client", "home_actions", "home_capabilities", "energy_monitor", "camera_health",
     "calendar_reader", "media_control", "ha_todo_sync"),
    ("dashboard_http", "live_event_bus", "live_receipt_store", "browser_delivery"),
    ("latency_sqlite", "health_aggregator", "llm_metrics"),
)
BUILDERS = (household_features_plan, home_integrations_plan, dashboard_transport_plan, latency_plan)
PLAN_OPTIONS = ({}, {"ha_url": "https://ha.example.invalid"}, {}, {"db_path": PurePath("/synthetic/latency.db")})


def _event(spec, phase, plan, **extra):
    return {"kind": spec.event_kind, "phase": phase, "plan": plan, **extra}


@pytest.mark.parametrize("index", range(4))
def test_nominal_roles_and_closed_lifecycle(index):
    producer, spec = FEATURES_PRODUCERS[index], GRAMMAR_SPECS[index]
    roles, builder = ROLE_IDS[index], BUILDERS[index]
    assert producer == spec.producer and spec.grammar_id == f"{producer}_publisher/1"
    plan = builder(selected_roles=roles, **PLAN_OPTIONS[index])
    assert plan["roles"] == sorted(roles) and set(plan["owners"]) == set(roles)
    header = {"selected": {producer: True}}
    prior = None
    for phase in ("intent", "attached", "opening"):
        prior = spec.decode_event(_event(spec, phase, plan), prior, True, header)
    aggregate = features_aggregate(producer, drained_roles=roles)
    assert spec.decode_event(_event(spec, "drained", plan, aggregate=aggregate), prior, True, header) == {
        "producer": producer, "phase": "drained", "plan": plan, "aggregate": aggregate,
    }

    disabled = builder(selected_roles=())
    assert spec.decode_event(_event(spec, "disabled", disabled), None, False,
                             {"selected": {producer: False}})["plan"] == disabled
    with pytest.raises(RuntimeError, match="features_selection"):
        spec.decode_event(_event(spec, "intent", disabled), None, True, header)


def test_plans_reference_original_boundaries_and_redact_destinations():
    household, integrations, dashboard, latency = [
        builder(selected_roles=roles, **options)
        for builder, roles, options in zip(BUILDERS, ROLE_IDS, PLAN_OPTIONS, strict=True)
    ]
    assert {k: v["durable_boundary"] for k, v in household["owners"].items() if "durable_boundary" in v} == {
        "timers_runtime": "reminders", "alerts_runtime": "reminders",
    }
    assert all(v["kind"] == "resource_only" for v in household["owners"].values())
    home = integrations["owners"]
    assert {role: home[role]["kind"] for role in ROLE_IDS[1][3:]} == {
        "energy_monitor": "readonly_input", "camera_health": "readonly_input",
        "calendar_reader": "readonly_input", "media_control": "external_effect", "ha_todo_sync": "external_effect",
    }
    assert home["ha_todo_sync"]["local_boundary"] == "lists" and "lists" not in FEATURES_PRODUCERS
    plan = home_integrations_plan(
        selected_roles=ROLE_IDS[1], selection_configured=True,
        ha_url="https://user:secret@HA.example.invalid:443/private?auth=synthetic#ignored",
    )
    assert plan == dict(integrations, selection_configured=True)
    assert plan["origin_sha256"] == sha256(b"https|ha.example.invalid|443").hexdigest()
    assert plan["selection"] == "home_assistant_control_selection_path"
    assert dashboard["receipt_targets"] == {
        "live_browser": dict(LIVE_TARGET, version=1),
        "conversation": dict(REST_TARGET, version=3),
        "voice_review": dict(VOICE_REVIEW_TARGET, version=4),
    }
    assert dashboard["sent_remote_bytes_locally_erasable"] is False
    assert latency["kind"] == "sqlite" and latency["database"] == "monitoring_db_path"
    assert latency["database_sha256"] == sha256(b"/synthetic/latency.db").hexdigest()
    assert latency["scope"] == "technical_metrics"


@pytest.mark.parametrize("index", range(4))
@pytest.mark.parametrize("injection", (
    {"unreviewed": "unapproved"}, {"headers": {"auth": "synthetic"}}, {"token": "synthetic"},
    {"transcript": "synthetic"}, {"embeddings": {"synthetic": 1}}, {"audio": "synthetic"},
))
def test_additional_metadata_rejected_in_plan_and_aggregate(index, injection):
    spec, builder, roles = GRAMMAR_SPECS[index], BUILDERS[index], ROLE_IDS[index]
    plan = builder(selected_roles=roles, **PLAN_OPTIONS[index])
    header = {"selected": {spec.producer: True}}
    with pytest.raises(RuntimeError, match="publisher_metadata|features_plan"):
        spec.decode_event(_event(spec, "intent", dict(plan, **injection)), None, True, header)
    aggregate = features_aggregate(spec.producer, drained_roles=roles)
    with pytest.raises(RuntimeError, match="publisher_metadata|features_aggregate"):
        spec.decode_event(_event(spec, "drained", plan, aggregate=dict(aggregate, **injection)),
                          {"phase": "opening", "plan": plan}, True, header)


@pytest.mark.parametrize("value", ("<object at 0x123>", "https://user:pass@ha.invalid/?auth=synthetic", 1, None))
def test_existing_fields_reject_unreviewed_values_and_numeric_boolean_aliases(value):
    spec = GRAMMAR_SPECS[1]
    header = {"selected": {spec.producer: True}}
    plan = home_integrations_plan(selected_roles=("ha_client",), ha_url="http://ha.example.invalid")
    for field in ("origin_sha256", "selection", "selection_configured"):
        invalid = dict(plan, **{field: value})
        with pytest.raises(RuntimeError, match="features_plan"):
            spec.decode_event(_event(spec, "intent", invalid), None, True, header)
    invalid = deepcopy(plan)
    invalid["owners"]["ha_client"]["kind"] = value
    with pytest.raises(RuntimeError, match="features_plan"):
        spec.decode_event(_event(spec, "intent", invalid), None, True, header)

    for aggregate in (
        {"drained": dict(plan["enabled"], ha_client=value)},
        features_aggregate(spec.producer, drained_roles=()),
        features_aggregate(spec.producer, drained_roles=ROLE_IDS[1]),
    ):
        with pytest.raises(RuntimeError, match="features_aggregate"):
            spec.decode_event(_event(spec, "drained", plan, aggregate=aggregate),
                              {"phase": "opening", "plan": plan}, True, header)


async def test_feature_grammars_round_trip_through_the_coordinator(monkeypatch, decoder, local_grammar_table):  # noqa: F811
    config = {"dashboard": {"enabled": False}, "analytics": {"enabled": False}}
    ticket = dict(decoder["header"], admission_root=str(decoder["root"]), run_id=uuid4().hex, generation=3)
    admission = runtime_admission.RuntimeAdmission(ticket, {"state_root": str(decoder["state"])},
        str(decoder["state"] / "privacy.db"), runtime_admission._config_digest(config))
    monkeypatch.setattr(runtime_admission, "_verified_admissions", {admission: False})
    monkeypatch.setattr(admission, "_revalidate_live", lambda snapshot=None: None)
    async with AsyncExitStack() as stack:
        admission.install_on(stack)
        writer = v2.OutputRegistrationV2(admission, config, key=base._KEY, journal_format=v4.V4_FORMAT,
            extra_grammars=GRAMMAR_SPECS, publisher_selection={"dashboard_transport": False})
        admission._output_registration = writer
        await writer._run(writer._start)
        try:
            _compose_legacy_fixture(writer)
            for producer, builder, roles, options in zip(
                FEATURES_PRODUCERS, BUILDERS, ROLE_IDS, PLAN_OPTIONS, strict=True,
            ):
                if producer == "dashboard_transport":
                    await writer.publisher_disabled(producer, builder(selected_roles=()))
                    continue
                await writer.publisher_intent(producer, builder(selected_roles=roles, **options))
                owner = await writer.attach_publisher(producer)
                await owner.opening()
                await owner.drained(features_aggregate(producer, drained_roles=roles))
            await writer.composed()
            header, _, states, _ = v4.chain(writer._directory)
            assert {key: header["grammars"][key] for key in FEATURES_PRODUCERS} == {
                spec.producer: spec.grammar_id for spec in GRAMMAR_SPECS
            }
            assert {key: states[key]["phase"] for key in FEATURES_PRODUCERS} == {
                "household_features": "drained", "home_integrations": "drained",
                "dashboard_transport": "disabled", "latency": "drained",
            }
        finally:
            await writer._disk.aclose()
