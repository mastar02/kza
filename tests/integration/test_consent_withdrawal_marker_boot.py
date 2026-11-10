"""OQ-8: the durable consent-withdrawal marker across boots, and the boot gate.

Boots reuse the managed configuration-publisher seams (synthetic admission
validation and verified-instance registry); DashboardAuthority, PrivacyServices,
ConfigurationChanges, the V4 journal and the owned AsyncExitStack are real.
Each boot is a new run of the same admission root with a higher generation.
"""
from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
import copy
import hashlib
import logging
from types import SimpleNamespace
from uuid import uuid4

import pytest
import yaml

from src.core import runtime_admission
from src.core.egress import CloudConsentGate, ConsentWithdrawalSource, EgressPolicy, ProcessingDestination
from src.core.state_configuration import ConfigurationChanges, MarkerRecordState
from src.dashboard.authority import DashboardAuthority
from src.memory import output_registration as base
from src.memory import output_registration_v2 as registration
from src.memory.privacy_services import build_privacy_services
from tests.integration.test_configuration_publisher_registration import (
    TOKEN, application_graph, config_error, g_observer, messages, phases, shutdown,
)

RAW_TRUE = b'reasoner:\n  cloud:\n    consent: true\n  api_key_env: "${SYNTHETIC_KEY}"\nother: untouched\n'
RAW_FALSE = b'reasoner:\n  cloud:\n    consent: false\nother: untouched\n'


def v4():
    from src.memory import output_registration_v4
    return output_registration_v4


@asynccontextmanager
async def boot(tmp_path, monkeypatch, generation, *, raw=None, dashboard=True, privacy=True, crash=None):
    """One managed run of the shared admission root; raw=None keeps the current settings file."""
    root, state = tmp_path / "admission-root", tmp_path / "state"
    (root / "admission").mkdir(parents=True, exist_ok=True)
    data = state / "data"
    data.mkdir(parents=True, exist_ok=True)
    settings = state / "config" / "settings.yaml"
    settings.parent.mkdir(exist_ok=True)
    if raw is not None:
        settings.write_bytes(raw)
    current = settings.read_bytes()
    config = {
        "memory": {"privacy_db_path": str(data / "privacy.db")},
        "dashboard": {"enabled": dashboard, "auth": {"db_path": str(data / "authority.db")}},
        "routines": {"db_path": str(data / "routines.db")},
        "analytics": {"events_db": str(data / "events.db"), "suggestions_path": str(data / "suggestions.json")},
        "speaker_id": {"users_path": str(data / "users.json")},
        "reasoner": copy.deepcopy(yaml.safe_load(current)["reasoner"]),
        **{name: {"db_path": str(data / (name + ".db"))} for name in ("lists", "reminders")},
    }
    ticket = {"admission_root": str(root), "run_id": uuid4().hex, "binding_id": "a" * 64, "generation": generation}
    binding = {"state_root": str(state),
               "authority": {"projection": {"configuration": {"config/settings.yaml":
                                                               hashlib.sha256(current).hexdigest()}}}}
    admission = runtime_admission.RuntimeAdmission(ticket, binding, config["memory"]["privacy_db_path"],
                                                   runtime_admission._config_digest(config))
    monkeypatch.setattr(admission, "revalidate", lambda: None)
    admission._validation_snapshot = object()
    monkeypatch.setattr(admission, "_revalidate_live", lambda snapshot=None: None)
    monkeypatch.setattr(runtime_admission, "_verified_runs", {(ticket["run_id"], ticket["binding_id"])})
    monkeypatch.setattr(runtime_admission, "_verified_admissions", {admission: False})
    monkeypatch.setattr(base, "_ACTIVE", set())
    stack = AsyncExitStack()
    admission.install_on(stack)
    g = SimpleNamespace(stack=stack, config=config, state=state, settings=settings, admission=admission,
                        closed=False, error=None, changes=None, observed=[], authority=None, actor=None,
                        privacy=None, api=None, owner=None, gate=CloudConsentGate(), run_id=ticket["run_id"],
                        boot_error=None)
    try:
        try:
            if crash is not None:
                crash(monkeypatch)
            g.owner = await admission.begin_destination_registration(config)
            if dashboard:
                g.authority = DashboardAuthority(config["dashboard"]["auth"]["db_path"], expected_token=lambda: TOKEN)
                stack.push_async_callback(g.authority.aclose)
                await registration.attach_output(g.authority, "authority_sessions")
                stack.push_async_exit(g_observer(g))
                g.changes = ConfigurationChanges(g.authority, admission, g.gate)
                g.changes.install_on(stack)
                await g.owner.attach_configuration(g.changes, g.authority)
                await g.authority.initialize()
                g.actor = g.authority.resolve("Bearer " + TOKEN)
            else:
                await g.owner.configuration_disabled()
            if privacy:
                g.privacy = await build_privacy_services(config, shutdown=stack)
        except Exception as error:  # noqa: BLE001 - a crashed boot is the oracle
            g.boot_error = error
        yield g
    finally:
        await shutdown(g)


def run_dir(g):
    return g.owner._directory


def configuration_events(g):
    return [event for event in (base._read(p)[0] for p in sorted(run_dir(g).glob("[0-9]*.json")))
            if event["kind"] == "configuration_publisher"]


def ready(g):
    g.changes.attach_graph(application_graph(g))


async def withdraw(g):
    outcome = await g.changes.withdraw_cloud_consent(g.actor)
    assert outcome.gate_closed is True and g.gate.withdrawn is True
    assert outcome.marker is MarkerRecordState.DURABLE and outcome.failure is None
    return outcome


def poison_like_a_crash(g):
    """A killed process appends nothing more: mark the epoch failed before its exits."""
    g.owner._failed = True


# ----------------------------------------------------------------------
# B1: the decision is on the FIRST configuration record of every run.
# ----------------------------------------------------------------------

async def test_decision_is_on_intent_and_attached(tmp_path, monkeypatch):
    async with boot(tmp_path, monkeypatch, 1, raw=RAW_TRUE) as g1:
        assert g1.boot_error is None and g1.owner.effective_decision is None
        await withdraw(g1)
    async with boot(tmp_path, monkeypatch, 2) as g2:
        decision = g2.owner.effective_decision
        assert decision["basis"] == "actor" and decision["origin"] == "inherited"
        intent, attached = configuration_events(g2)[:2]
        assert (intent["phase"], attached["phase"]) == ("intent", "attached")
        assert intent["decision"] == attached["decision"] == decision
        assert decision["source"]["run_id"] == g1.run_id and decision["source"]["generation"] == 1
        assert decision["pinned_config_sha256"] == hashlib.sha256(RAW_TRUE).hexdigest()


async def test_kill_between_epoch_and_head_keeps_the_prior_marker_for_two_boots(tmp_path, monkeypatch):
    async with boot(tmp_path, monkeypatch, 1, raw=RAW_TRUE) as g1:
        await withdraw(g1)

    def crash_before_head(patch):
        def head(directory, previous, value):
            raise OSError("synthetic crash before HEAD")
        patch.setattr(base, "_head", head)
    async with boot(tmp_path, monkeypatch, 2, crash=crash_before_head) as g2:
        assert isinstance(g2.boot_error, OSError) or g2.owner is None
    monkeypatch.undo()
    async with boot(tmp_path, monkeypatch, 3) as g3:
        assert g3.boot_error is None
        assert g3.owner.effective_decision["basis"] == "actor"
        assert g3.owner.effective_decision["source"]["run_id"] == g1.run_id
    async with boot(tmp_path, monkeypatch, 4) as g4:
        assert g4.owner.effective_decision["basis"] == "actor"
        assert g4.owner.effective_decision["source"]["run_id"] == g3.run_id


async def test_kill_between_intent_and_attached_is_inherited_from_intent(tmp_path, monkeypatch):
    async with boot(tmp_path, monkeypatch, 1, raw=RAW_TRUE) as g1:
        await withdraw(g1)
    real = registration.OutputRegistrationV2._append_configuration

    def crash_at_attached(patch):
        def append(self, phase, plan, aggregate=None):
            if phase == "attached":
                raise OSError("synthetic crash before attached")
            return real(self, phase, plan, aggregate)
        patch.setattr(registration.OutputRegistrationV2, "_append_configuration", append)
    async with boot(tmp_path, monkeypatch, 2, crash=crash_at_attached) as g2:
        assert g2.boot_error is not None
        assert [event["phase"] for event in configuration_events(g2)] == ["intent"]
    monkeypatch.setattr(registration.OutputRegistrationV2, "_append_configuration", real)
    async with boot(tmp_path, monkeypatch, 3) as g3:
        decision = g3.owner.effective_decision
        assert decision["basis"] == "actor" and decision["source"]["run_id"] == g2.run_id


async def test_kill_after_withdrawn_before_shutdown_closes_the_next_boot(tmp_path, monkeypatch):
    async with boot(tmp_path, monkeypatch, 1, raw=RAW_TRUE) as g1:
        ready(g1)
        await withdraw(g1)
        poison_like_a_crash(g1)
    assert phases(g1.owner)[-1] == "withdrawn"
    async with boot(tmp_path, monkeypatch, 2) as g2:
        assert g2.owner.effective_decision["basis"] == "actor"
        assert g2.owner.effective_decision["source"]["run_id"] == g1.run_id


async def test_disabled_dashboard_records_the_inherited_decision(tmp_path, monkeypatch):
    async with boot(tmp_path, monkeypatch, 1, raw=RAW_TRUE) as g1:
        await withdraw(g1)
    async with boot(tmp_path, monkeypatch, 2, dashboard=False) as g2:
        event, = configuration_events(g2)
        assert event["phase"] == "disabled" and event["decision"]["basis"] == "actor"
    async with boot(tmp_path, monkeypatch, 3) as g3:
        assert g3.owner.effective_decision["source"]["run_id"] == g2.run_id


# ----------------------------------------------------------------------
# B2: unverifiable is recorded and carried until a settling boot.
# ----------------------------------------------------------------------

async def test_torn_newest_run_is_recorded_unverifiable_and_carried(tmp_path, monkeypatch):
    async with boot(tmp_path, monkeypatch, 1, raw=RAW_TRUE) as g1:
        pass
    raw = base._encoded({"kind": "configuration_publisher", "torn": True})
    (run_dir(g1) / f"{99:012d}-{hashlib.sha256(raw).hexdigest()}.json").write_bytes(raw)
    async with boot(tmp_path, monkeypatch, 2) as g2:
        decision = g2.owner.effective_decision
        assert decision["basis"] == "unverifiable"
        assert decision["source"] == {"run_id": g1.run_id, "generation": 1, "record_sha256": None,
                                      "reason": "chain_invalid"}
        assert configuration_events(g2)[0]["decision"] == decision
    async with boot(tmp_path, monkeypatch, 3) as g3:
        decision = g3.owner.effective_decision
        assert decision["basis"] == "unverifiable" and decision["source"]["run_id"] == g2.run_id
        assert decision["source"]["reason"] is None and decision["source"]["record_sha256"] is not None


async def test_settling_boot_clears_then_a_grant_opens(tmp_path, monkeypatch, caplog):
    async with boot(tmp_path, monkeypatch, 1, raw=RAW_TRUE) as g1:
        await withdraw(g1)
    with caplog.at_level(logging.WARNING):
        async with boot(tmp_path, monkeypatch, 2, raw=RAW_FALSE) as g2:
            assert g2.owner.effective_decision is None
            assert configuration_events(g2)[0]["decision"] is None
            assert g2.owner.settled_decision["basis"] == "actor"
    assert g1.run_id in caplog.text and "settled" in caplog.text.lower()
    async with boot(tmp_path, monkeypatch, 3, raw=RAW_TRUE) as g3:
        assert g3.owner.effective_decision is None and g3.owner.settled_decision is None


async def test_committed_withdrawal_settles_on_the_next_boot(tmp_path, monkeypatch):
    async with boot(tmp_path, monkeypatch, 1, raw=RAW_TRUE) as g1:
        ready(g1)
        await withdraw(g1)
    assert g1.error is None and phases(g1.owner) == ["intent", "attached", "withdrawn", "opening", "drained"]
    assert yaml.safe_load(g1.settings.read_bytes())["reasoner"]["cloud"]["consent"] is False
    async with boot(tmp_path, monkeypatch, 2) as g2:
        assert g2.owner.effective_decision is None
        assert g2.owner.settled_decision["source"]["run_id"] == g1.run_id


@pytest.mark.parametrize("value", ['"true"', "1", "null", '"yes"'])
async def test_settling_predicate_is_not_true(tmp_path, monkeypatch, value):
    async with boot(tmp_path, monkeypatch, 1, raw=RAW_TRUE) as g1:
        await withdraw(g1)
    raw = f"reasoner:\n  cloud:\n    consent: {value}\nother: untouched\n".encode()
    async with boot(tmp_path, monkeypatch, 2, raw=raw) as g2:
        # A non-boolean value is no grant: it settles (cloud stays off by config).
        assert g2.owner.effective_decision is None


# ----------------------------------------------------------------------
# F1/B3 at the journal: cancellation never yields a clean epoch without marker.
# ----------------------------------------------------------------------

async def test_cancelled_observer_still_records_the_marker(tmp_path, monkeypatch):
    async with boot(tmp_path, monkeypatch, 1, raw=RAW_TRUE) as g1:
        # No graph attached: the queued write is refused, so the marker alone carries the withdrawal.
        entered, release = asyncio.Event(), asyncio.Event()
        real = registration.configuration_withdrawn

        async def blocked(owner):
            entered.set()
            await release.wait()
            return await real(owner)
        monkeypatch.setattr(registration, "configuration_withdrawn", blocked)
        request = asyncio.create_task(g1.changes.withdraw_cloud_consent(g1.actor))
        await asyncio.wait_for(entered.wait(), 5)
        assert g1.gate.withdrawn is True
        request.cancel()
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await request
        assert phases(g1.owner)[-1] == "withdrawn" and not g1.owner._failed
    assert phases(g1.owner) == ["intent", "attached", "withdrawn", "drained"]
    assert yaml.safe_load(g1.settings.read_bytes())["reasoner"]["cloud"]["consent"] is True
    async with boot(tmp_path, monkeypatch, 2) as g2:
        assert g2.owner.effective_decision["basis"] == "actor"


async def test_cancelled_withdrawal_between_gate_and_marker_poisons_the_epoch(tmp_path, monkeypatch):
    receipts = []
    async with boot(tmp_path, monkeypatch, 1, raw=RAW_TRUE) as g1:
        ready(g1)
        entered = asyncio.Event()

        async def never(owner):
            entered.set()
            await asyncio.Event().wait()
        monkeypatch.setattr(registration, "configuration_withdrawn", never)
        request = asyncio.create_task(g1.changes.withdraw_cloud_consent(g1.actor))
        await asyncio.wait_for(entered.wait(), 5)
        assert g1.gate.withdrawn is True
        g1.changes._withdrawing.cancel()  # The retained task itself (e.g. loop teardown).
        outcome = await request
        assert outcome.gate_closed is True and outcome.marker is MarkerRecordState.REGISTRATION_FAILED
        assert g1.owner._failed is True
        monkeypatch.setattr(runtime_admission, "_stage",
                            lambda: SimpleNamespace(write_admission_receipt=lambda *a: receipts.append(a)))
    assert "configuration_withdrawal_unconfirmed" in messages(config_error(g1))
    assert "drained" not in phases(g1.owner) and receipts == []
    assert str(g1.owner._closing.exception()) in {
        "output_registration_epoch_incomplete", "output_registration_registry_unavailable"}


async def test_main_gate_follows_the_recorded_decision(tmp_path, monkeypatch):
    import src.main as main_module
    async with boot(tmp_path, monkeypatch, 1, raw=RAW_TRUE) as g1:
        await withdraw(g1)
    async with boot(tmp_path, monkeypatch, 2) as g2:
        gate = CloudConsentGate()
        main_module.apply_boot_consent_decision(gate, g2.owner)
        assert gate.withdrawn and gate.source is ConsentWithdrawalSource.DURABLE_MARKER
        gate = CloudConsentGate()
        main_module.apply_boot_consent_decision(gate, None)
        assert gate.withdrawn is False
        g2.owner._effective_decision = dict(g2.owner._effective_decision, basis="unverifiable")
        gate = CloudConsentGate()
        main_module.apply_boot_consent_decision(gate, g2.owner)
        assert gate.source is ConsentWithdrawalSource.MARKER_UNVERIFIABLE


# ----------------------------------------------------------------------
# boot_main and F6: the real factory, every cloud policy shares one gate.
# ----------------------------------------------------------------------

from tests.integration.test_application_graph import leaves  # noqa: E402,F401
from tests.integration.test_application_http_reasoner_retirement import graph_http, http_leaves  # noqa: E402,F401
from tests.mocks.application_graph import offline_config, offline_home  # noqa: E402


def _cloud_config(tmp_path, url, *, destination="cloud", mode="http"):
    config = offline_config(tmp_path, url)
    config["reasoner"] = {"mode": mode, "http_base_url": "http://127.0.0.1:8200/v1", "http_model": "synthetic-main",
                          "destination": destination, "cloud": {"consent": True}}
    config["router"] = {"enabled": True, "base_url": "http://127.0.0.1:8101/v1", "destination": "cloud"}
    config["orchestrator"] = {"enabled": True, "context": {"compaction": {"enabled": True, "timeout_s": 30}}}
    return config


def _policies(root, *, limit=200000):
    """Every EgressPolicy reachable from the graph (bounded object walk)."""
    seen, found, pending = set(), [], [root]
    while pending and len(seen) < limit:
        value = pending.pop()
        if id(value) in seen or isinstance(value, (str, bytes, int, float, bool, type(None), type)):
            continue
        seen.add(id(value))
        if isinstance(value, EgressPolicy):
            found.append(value)
            continue
        if isinstance(value, dict):
            pending.extend(value.values())
        elif isinstance(value, (list, tuple, set, frozenset)):
            pending.extend(value)
        else:
            for name in getattr(type(value), "__slots__", ()):
                pending.append(getattr(value, name, None))
            try:
                pending.extend(vars(value).values())
            except TypeError:
                pass
    return found


def _instances(root, kind, *, limit=200000):
    """Every distinct instance of ``kind`` reachable from the graph (bounded object walk)."""
    seen, found, pending = set(), [], [root]
    while pending and len(seen) < limit:
        value = pending.pop()
        if id(value) in seen or isinstance(value, (str, bytes, int, float, bool, type(None), type)):
            continue
        seen.add(id(value))
        if isinstance(value, kind):
            found.append(value)
        if isinstance(value, dict):
            pending.extend(value.values())
        elif isinstance(value, (list, tuple, set, frozenset)):
            pending.extend(value)
        else:
            for name in getattr(type(value), "__slots__", ()):
                pending.append(getattr(value, name, None))
            try:
                pending.extend(vars(value).values())
            except TypeError:
                pass
    return found


def _clients(graph):
    from src.llm.hermes_reasoner import HermesCliReasoner
    from src.llm.reasoner import FastRouter, HttpReasoner
    reasoners = list(graph.http_reasoners._models.values())
    hermes = list(graph.hermes_reasoners._models.values())
    assert {id(r) for r in _instances(graph, HttpReasoner)} == {id(r) for r in reasoners}
    assert {id(r) for r in _instances(graph, HermesCliReasoner)} == {id(r) for r in hermes}
    return reasoners, hermes, _instances(graph, FastRouter)


async def test_every_cloud_policy_shares_the_graph_gate_http(graph_http, leaves, tmp_path):  # noqa: F811
    async with offline_home() as (url, observations):
        async with AsyncExitStack() as shutdown_stack:
            graph = await build(tmp_path, url, leaves, shutdown_stack)
            gate = graph.consent_gate
            assert type(gate) is CloudConsentGate and gate.withdrawn is False
            reasoners, _hermes, fast = _clients(graph)
            assert len(reasoners) == 2 and len(fast) == 1  # Main and compaction HttpReasoner, FastRouter.
            for client in [*reasoners, *fast]:
                assert client.policy.consent_gate is gate, client
            assert graph.dashboard.consent_gate is gate
            cloud = [p for p in _policies(graph) if p.destination is ProcessingDestination.CLOUD and p.cloud_consent]
            assert cloud and all(policy.consent_gate is gate for policy in cloud)


async def build(tmp_path, url, leaves, stack, **kwargs):  # noqa: F811
    from src.main import build_application
    return await build_application(_cloud_config(tmp_path, url, **kwargs), shutdown=stack, factories=leaves)


async def test_marker_boot_keeps_cloud_off_without_mutating_config(graph_http, leaves, tmp_path, monkeypatch):  # noqa: F811
    import src.main as main_module
    decision = {"origin": "inherited", "basis": "actor", "scope": "reasoner_cloud",
                "pinned_config_sha256": "d" * 64,
                "source": {"run_id": "3" * 32, "generation": 2, "record_sha256": "e" * 64, "reason": None}}
    monkeypatch.setattr(main_module, "boot_consent_decision", lambda registration: decision)
    async with offline_home() as (url, observations):
        async with AsyncExitStack() as shutdown_stack:
            config = _cloud_config(tmp_path, url)
            before = copy.deepcopy(config)
            graph = await main_module.build_application(config, shutdown=shutdown_stack, factories=leaves)
            assert graph.consent_gate.withdrawn
            assert graph.consent_gate.source is ConsentWithdrawalSource.DURABLE_MARKER
            reasoners, _hermes, fast = _clients(graph)
            # No cloud reasoner: the only HttpReasoner is the compaction fallback on the declared local 7B.
            assert len(reasoners) == 1
            assert reasoners[0].policy.destination is ProcessingDestination.LOCAL
            assert reasoners[0].base_url == "http://127.0.0.1:8101/v1"
            assert len(fast) == 1 and fast[0].policy.consent_gate is graph.consent_gate
            assert graph.config["reasoner"] == before["reasoner"]
            assert runtime_admission._config_digest(graph.config) == runtime_admission._config_digest(before)


async def test_marker_boot_keeps_a_local_declared_reasoner(graph_http, leaves, tmp_path, monkeypatch):  # noqa: F811
    import src.main as main_module
    decision = {"origin": "inherited", "basis": "unverifiable", "scope": "reasoner_cloud",
                "pinned_config_sha256": None,
                "source": {"run_id": None, "generation": None, "record_sha256": None, "reason": "duplicate_generation"}}
    monkeypatch.setattr(main_module, "boot_consent_decision", lambda registration: decision)
    async with offline_home() as (url, observations):
        async with AsyncExitStack() as shutdown_stack:
            graph = await build(tmp_path, url, leaves, shutdown_stack, destination="local")
            assert graph.consent_gate.withdrawn
            reasoners, _hermes, _fast = _clients(graph)
            main = [r for r in reasoners if r.base_url == "http://127.0.0.1:8200/v1"]
            assert len(main) == 1 and main[0].policy.destination is ProcessingDestination.LOCAL
            assert main[0].policy.consent_gate is graph.consent_gate


async def test_every_cloud_policy_shares_the_graph_gate_hermes(graph_http, leaves, tmp_path, monkeypatch):  # noqa: F811
    from src.llm import hermes_reasoner
    monkeypatch.setattr(hermes_reasoner.HermesCliReasoner, "load", lambda self: None)
    async with offline_home() as (url, observations):
        async with AsyncExitStack() as shutdown_stack:
            graph = await build(tmp_path, url, leaves, shutdown_stack, mode="hermes_cli")
            _reasoners, hermes, fast = _clients(graph)
            assert len(hermes) == 2  # Main reasoner and compaction.
            for client in [*hermes, *fast]:
                assert client.policy.consent_gate is graph.consent_gate
