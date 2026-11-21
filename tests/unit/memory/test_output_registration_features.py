"""T024 feature publisher metadata and real V4 journal round-trips."""

from contextlib import AsyncExitStack
from copy import deepcopy
from hashlib import sha256
from pathlib import PurePath
from uuid import uuid4

import pytest

from src.core import runtime_admission
from src.dashboard.live_erasure import LIVE_TARGET, REST_TARGET, VOICE_REVIEW_TARGET
from src.home_assistant.ha_client import HomeAssistantClient
from src.llm.metrics import LLMMetricsTracker
from src.memory import output_registration as base
from src.memory import output_registration_v2 as v2
from src.memory import output_registration_v4 as v4
from src.memory.output_registration_features import (
    FEATURES_GRAMMAR_SPECS,
    FEATURES_PRODUCERS,
    GRAMMAR_SPECS,
    bind_features_family_publishers,
    dashboard_transport_plan,
    features_aggregate,
    features_family_selection,
    home_integrations_plan,
    household_features_plan,
    latency_plan,
)
from src.monitoring.health_aggregator import HealthAggregator
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


def _aggregate(producer, roles):
    return features_aggregate(producer, instance_sha256={role: sha256(role.encode()).hexdigest() for role in roles})


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
    aggregate = _aggregate(producer, roles)
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
    aggregate = _aggregate(spec.producer, roles)
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

    invalid_digest = _aggregate(spec.producer, ("ha_client",))
    invalid_digest["owners"]["ha_client"]["instance_sha256"] = value
    invalid_observation = _aggregate(spec.producer, ("ha_client",))
    invalid_observation["owners"]["ha_client"]["observation"] = value
    for aggregate in (invalid_digest, invalid_observation, _aggregate(spec.producer, ()),
                      _aggregate(spec.producer, ROLE_IDS[1])):
        with pytest.raises(RuntimeError, match="features_aggregate"):
            spec.decode_event(_event(spec, "drained", plan, aggregate=aggregate),
                              {"phase": "opening", "plan": plan}, True, header)


def test_terminal_codec_rejects_a_shared_instance_digest_for_two_owners():
    spec = GRAMMAR_SPECS[3]
    roles = ("health_aggregator", "llm_metrics")
    plan = latency_plan(selected_roles=roles)
    aggregate = _aggregate(spec.producer, roles)
    aggregate["owners"]["llm_metrics"]["instance_sha256"] = aggregate["owners"]["health_aggregator"]["instance_sha256"]
    with pytest.raises(RuntimeError, match="features_aggregate"):
        spec.decode_event(_event(spec, "drained", plan, aggregate=aggregate),
                          {"phase": "opening", "plan": plan}, True, {"selected": {spec.producer: True}})


@pytest.mark.parametrize("scenario", ["success", "admission_rejection", "selection_mismatch"])
async def test_feature_grammars_round_trip_through_the_coordinator(
    monkeypatch, decoder, local_grammar_table, scenario,  # noqa: F811
):
    config = {"dashboard": {"enabled": False}, "analytics": {"enabled": False}}
    ticket = dict(decoder["header"], admission_root=str(decoder["root"]), run_id=uuid4().hex, generation=3)
    admission = runtime_admission.RuntimeAdmission(ticket, {"state_root": str(decoder["state"])},
        str(decoder["state"] / "privacy.db"), runtime_admission._config_digest(config))
    monkeypatch.setattr(runtime_admission, "_verified_admissions", {admission: False})
    monkeypatch.setattr(admission, "_revalidate_live", lambda snapshot=None: None)
    url = "https://ha.example.invalid"
    plans = {
        "household_features": household_features_plan(selected_roles=()),
        "home_integrations": home_integrations_plan(selected_roles=("ha_client",), ha_url=url),
        "dashboard_transport": dashboard_transport_plan(selected_roles=()),
        "latency": latency_plan(selected_roles=("health_aggregator", "llm_metrics")),
    }
    assert FEATURES_GRAMMAR_SPECS is GRAMMAR_SPECS
    revalidations = []

    def revalidate():
        revalidations.append(True)
        if scenario == "admission_rejection":
            raise RuntimeError("admission_revalidation_rejected")

    monkeypatch.setattr(admission, "revalidate", revalidate)
    selection = features_family_selection(plans)
    if scenario == "selection_mismatch":
        selection["latency"] = False
    async with AsyncExitStack() as stack:
        admission.install_on(stack)
        writer = v2.OutputRegistrationV2(admission, config, key=base._KEY, journal_format=v4.V4_FORMAT,
            extra_grammars=FEATURES_GRAMMAR_SPECS, publisher_selection=selection)
        admission._output_registration = writer
        await writer._run(writer._start)
        try:
            _compose_legacy_fixture(writer)
            before = v4.chain(writer._directory)
            if scenario == "selection_mismatch":
                with pytest.raises(RuntimeError, match="^output_registration_features_selection$"):
                    await bind_features_family_publishers(writer, plans)
                assert v4.chain(writer._directory) == before and not revalidations and not writer._failed
                corrected = dict(plans, latency=latency_plan(selected_roles=()))
                assert set(await bind_features_family_publishers(writer, corrected)) == {"home_integrations"}
                return
            invalid = dict(plans, dashboard_transport=dict(plans["dashboard_transport"], unreviewed="invalid"))
            with pytest.raises(RuntimeError, match="features_plan"):
                await bind_features_family_publishers(writer, invalid)
            assert v4.chain(writer._directory) == before and not revalidations
            if scenario == "admission_rejection":
                with pytest.raises(RuntimeError, match="admission_revalidation_rejected"):
                    await bind_features_family_publishers(writer, plans)
                assert v4.chain(writer._directory) == before and len(revalidations) == 1
                return
            bindings = await bind_features_family_publishers(writer, plans)
            assert set(bindings) == {"home_integrations", "latency"} and len(revalidations) == 1
            owners = {
                "home_integrations": {"ha_client": HomeAssistantClient(url, "synthetic-only")},
                "latency": {"health_aggregator": HealthAggregator(), "llm_metrics": LLMMetricsTracker()},
            }
            stack.push_async_callback(owners["home_integrations"]["ha_client"].close)
            latency = bindings["latency"]

            class HealthReplacement(HealthAggregator):
                pass

            with pytest.raises(RuntimeError, match="features_owner"):
                await latency.attach("health_aggregator", HealthReplacement())
            for producer, binding in bindings.items():
                for role, owner in owners[producer].items():
                    await binding.attach(role, owner)
                    await binding.opening(role, owner)
                    await binding.opened(role, owner)
                assert binding.matches(owners[producer])
            await writer.composed()
            observed = v4.chain(writer._directory)
            with pytest.raises(RuntimeError, match="features_owner"):
                await latency.drained(dict(owners["latency"], health_aggregator=HealthAggregator()))
            with pytest.raises(RuntimeError, match="features_owner"):
                await bindings["home_integrations"].drained(owners["home_integrations"])
            assert v4.chain(writer._directory) == observed
            await owners["home_integrations"]["ha_client"].close()
            for producer, binding in bindings.items():
                await binding.drained(owners[producer])
            header, _, states, _ = v4.chain(writer._directory)
            assert {key: header["grammars"][key] for key in FEATURES_PRODUCERS} == {
                spec.producer: spec.grammar_id for spec in GRAMMAR_SPECS
            }
            assert {key: states[key]["phase"] for key in FEATURES_PRODUCERS} == {
                "household_features": "disabled", "home_integrations": "drained",
                "dashboard_transport": "disabled", "latency": "drained",
            }
            seen = set()
            for producer, graph in owners.items():
                evidence = states[producer]["aggregate"]["owners"]
                identities = {role: evidence[role]["instance_sha256"] for role in graph}
                assert seen.isdisjoint(identities.values())
                assert states[producer]["aggregate"] == features_aggregate(producer, instance_sha256=identities)
                seen.update(identities.values())
            assert len(seen) == 3
        finally:
            await writer._disk.aclose()
