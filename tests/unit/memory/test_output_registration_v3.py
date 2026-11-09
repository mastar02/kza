"""Strict V3 account publisher grammar beside immutable V1/V2 histories.

Decoder contracts with the synthetic controller seam of the delegated
inventory tests; real compiled ancestry is a separate oracle.
"""
from copy import deepcopy

import pytest

from scripts import stage_runtime as stage
from src.memory import output_registration as base
from src.memory import output_registration_v2 as registration
from src.memory import output_registration_v3 as registration_v3
# The shared synthetic controller fixture is used by name in each test.
from tests.unit.memory.test_delegated_output_inventory import _assert_unknown, _write, decoder  # noqa: F401

SELECTED = ("privacy_authority", "authority_sessions", "routines", "events", "suggestions")


def _v3_run(decoder, run, events, *, selected_accounts=True, header_update=None):  # noqa: F811
    root = decoder["root"]
    directory = root / "admission/output-inventory/v3/runs" / run
    directory.mkdir(parents=True)
    header = dict(decoder["header"], schema_version=3, run_id=run, generation=3,
                  namespace_identity=base._directory(directory),
                  selected=dict(dict.fromkeys(SELECTED, True), accounts=selected_accounts))
    header.update(header_update or {})
    raw = base._encoded(header)
    (directory / "epoch.json").write_bytes(raw)
    previous = base._sha(raw)
    revision = 0
    for revision, value in enumerate(events, 1):
        raw = base._encoded(dict(value, revision=revision, previous_sha256=previous))
        previous = base._sha(raw)
        (directory / f"{revision:012d}-{previous}.json").write_bytes(raw)
    (directory / "HEAD.json").write_bytes(base._encoded({"revision": revision, "sha256": previous}))
    _write(root / "admission/receipts" / f"{run}.json", {"run_id": run, "binding_id": header["binding_id"],
        "generation": 3, "status": "closed", "resources": dict.fromkeys(stage._CLOSED_RESOURCES, True)})
    return directory


def _plan(decoder):  # noqa: F811
    data = decoder["state"] / "data"
    data.mkdir(exist_ok=True)
    return registration_v3.account_plan(data / "users_accounts_v1.json", data / "users.json",
                                        data / "users.json", decoder["state"])


def _events(plan, *, committed=1, failed=0):
    event = {"kind": "account_publisher", "plan": plan}
    aggregate = {"attempts": committed + failed, "committed_replacements": committed,
                 "clean_failed_attempts": failed,
                 "last_committed_identity": {"device": 1, "inode": 2} if committed else plan["initial_identity"],
                 "native_drain_returned": True, "open_temp_count": 0, "cleanup_uncertain": False}
    composition = {"kind": "composition", "instrumented": sorted({*SELECTED, "lists", "reminders", "accounts"}),
                   "global_inventory_complete": False, "aggregate_copy_complete": False}
    return [dict(event, phase="intent"), dict(event, phase="attached"), composition,
            dict(event, phase="opening"), dict(event, phase="drained", aggregate=aggregate)]


def test_current_reader_joins_v2_and_v3_and_reports_accounts_as_partial(decoder):  # noqa: F811
    plan = _plan(decoder)
    assert set(plan) == {"kind", "producer", "role", "representation", "path", "configured_path",
        "legacy_import_path", "ancestor", "parent_identity", "missing_parents", "initial_identity",
        "creation_observation", "before_first_publish", "temporary_scope", "classification"}
    _v3_run(decoder, "4" * 32, _events(plan))
    records = registration.read_inventory(decoder["controller"])
    by_version = {record["version"]: record for record in records}
    assert set(by_version) == {"v2", "v3"}
    v3 = by_version["v3"]
    assert v3["phase"] == "observed" and v3["header"]["schema_version"] == 3
    assert v3["stores"]["accounts"]["phase"] == "drained"
    assert v3["partial_producers"] == sorted({*SELECTED, "accounts"})
    # No closed list/reminder family exists in this synthetic run: all 32 stay uncovered.
    assert v3["uncovered_producers"] == list(base.PRODUCERS)
    assert v3["instrumented_producers"] == ["accounts"]
    assert by_version["v2"]["partial_producers"] == sorted(SELECTED)
    assert all(record["capability"] is False and record["global_inventory_complete"] is False
               for record in records)
    header, _, states, _ = base._chain(decoder["root"] / "admission/output-inventory/v3/runs" / ("4" * 32))
    assert header["selected"]["accounts"] is True and states["accounts"]["aggregate"]["attempts"] == 1


def _mutate(events, case, plan):
    if case == "inconsistent_aggregate":
        events[-1]["aggregate"]["clean_failed_attempts"] = 1
    elif case == "opening_twice":
        events.insert(4, deepcopy(events[3]))
    elif case == "uncertain_cleanup":
        events[-1]["aggregate"]["cleanup_uncertain"] = True
    elif case == "v2_composition":
        events[2]["instrumented"] = sorted({*SELECTED, "lists", "reminders"})
    elif case == "plan_extra_key":
        for event in events:
            if "plan" in event:
                event["plan"] = dict(plan, sqlite_sidecars=[])
    elif case == "plan_changed":
        events[-1]["plan"] = dict(plan, before_first_publish=False)
    elif case == "drained_without_drain":
        events[-1]["aggregate"]["native_drain_returned"] = False
    elif case == "disabled_while_selected":
        events[:] = [{"kind": "account_publisher", "phase": "disabled",
                      "configured_path": plan["configured_path"], "path": plan["path"]}]
    elif case == "private_field":
        events[-1]["aggregate"]["names"] = ["Synthetic"]
    return events


@pytest.mark.parametrize("case", ["inconsistent_aggregate", "opening_twice", "uncertain_cleanup",
    "v2_composition", "plan_extra_key", "plan_changed", "drained_without_drain",
    "disabled_while_selected", "private_field"])
def test_strict_v3_transitions_and_aggregate_are_unknown_when_malformed(decoder, case):  # noqa: F811
    plan = _plan(decoder)
    _v3_run(decoder, "5" * 32, _mutate(_events(plan), case, plan))
    records = registration.read_inventory(decoder["controller"])
    v3, = (record for record in records if record["version"] == "v3")
    _assert_unknown(v3)


def test_disabled_publisher_is_a_separate_terminal_only_when_unselected(decoder):  # noqa: F811
    plan = _plan(decoder)
    events = [{"kind": "account_publisher", "phase": "disabled",
               "configured_path": plan["configured_path"], "path": plan["path"]}]
    _v3_run(decoder, "6" * 32, events, selected_accounts=False)
    v3, = (r for r in registration.read_inventory(decoder["controller"]) if r["version"] == "v3")
    assert v3["phase"] == "observed" and v3["stores"]["accounts"]["phase"] == "disabled"
    assert "accounts" in v3["uncovered_producers"]


def test_same_run_in_v2_and_v3_namespaces_is_conflicting_unknown(decoder):  # noqa: F811
    plan = _plan(decoder)
    _v3_run(decoder, decoder["header"]["run_id"], _events(plan))
    records = registration.read_inventory(decoder["controller"])
    assert {record["version"] for record in records} == {"v2", "v3"}
    assert all(record["reason"] == "conflicting_run_versions" and record["phase"] == "unknown"
               for record in records)


def test_v2_decoder_never_accepts_v3_and_v3_never_retries_as_v2(decoder):  # noqa: F811
    plan = _plan(decoder)
    directory = _v3_run(decoder, "7" * 32, _events(plan))
    with pytest.raises(RuntimeError, match="header"):
        registration.chain(directory)
    with pytest.raises(RuntimeError, match="header"):
        registration_v3.chain(decoder["directory"])
    assert base._chain(directory)[0]["schema_version"] == 3


def test_explicit_v2_construction_keeps_v2_format_and_v3_is_module_owned():
    assert registration.V2_FORMAT.schema_version == 2 and registration.V2_FORMAT.namespace_name == "v2"
    assert registration_v3.V3_FORMAT.schema_version == 3 and registration_v3.V3_FORMAT.namespace_name == "v3"
    assert "accounts" not in registration.V2_FORMAT.selected_producers
    assert registration_v3.V3_FORMAT.selected_producers == registration.V2_FORMAT.selected_producers | {"accounts"}
    with pytest.raises(TypeError):
        registration_v3.V3_FORMAT.__reduce_ex__(2)


async def test_explicit_v2_client_writes_genuine_v2_run_read_only_by_v2_decoder(tmp_path, monkeypatch):
    from contextlib import AsyncExitStack
    from uuid import uuid4
    from src.core import runtime_admission
    root, state = tmp_path / "admission-root", tmp_path / "state"
    (root / "admission").mkdir(parents=True)
    state.mkdir()
    config = {"memory": {"privacy_db_path": str(state / "privacy.db")}}
    ticket = {"admission_root": str(root), "run_id": uuid4().hex, "binding_id": "a" * 64, "generation": 1}
    admission = runtime_admission.RuntimeAdmission(ticket, {"state_root": str(state)},
        config["memory"]["privacy_db_path"], runtime_admission._config_digest(config))
    monkeypatch.setattr(admission, "revalidate", lambda: None)
    admission._validation_snapshot = object()
    monkeypatch.setattr(admission, "_revalidate_live", lambda snapshot=None: None)
    monkeypatch.setattr(runtime_admission, "_verified_admissions", {admission: False})
    monkeypatch.setattr(base, "_ACTIVE", set())
    stack = AsyncExitStack()
    admission.install_on(stack)
    coordinator = registration.OutputRegistrationV2(admission, config, key=base._KEY)
    admission._output_registration = coordinator
    try:
        assert coordinator._format is registration.V2_FORMAT and "accounts" not in coordinator._selected
        await coordinator._run(coordinator._start)
        assert coordinator._directory.parents[1].name == "v2"
        header, head, states, _ = registration.chain(coordinator._directory)
        assert header["schema_version"] == 2 and "accounts" not in header["selected"] and states == {}
        assert base._chain(coordinator._directory)[0] == header
        with pytest.raises(RuntimeError, match="header"):
            registration_v3.chain(coordinator._directory)
    finally:
        await coordinator._disk.aclose()
        await stack.__aexit__(None, None, None)


# ----------------------------------------------------------------------
# V3 must apply every V2 semantic check to the V2 event kinds it carries.
# Runs are written by real coordinators of each fixed format; a mutation
# rewrites only event bodies and recomputes digests, names and HEAD.
# ----------------------------------------------------------------------

def _format_writer(tmp_path, monkeypatch, name, journal_format):
    from contextlib import AsyncExitStack
    from uuid import uuid4
    from src.core import runtime_admission
    root, state = tmp_path / name / "admission-root", tmp_path / name / "state"
    (root / "admission").mkdir(parents=True)
    state.mkdir()
    config = {"memory": {"privacy_db_path": str(state / "privacy.db")}}
    ticket = {"admission_root": str(root), "run_id": uuid4().hex, "binding_id": "a" * 64, "generation": 1}
    admission = runtime_admission.RuntimeAdmission(ticket, {"state_root": str(state)},
        config["memory"]["privacy_db_path"], runtime_admission._config_digest(config))
    monkeypatch.setattr(admission, "revalidate", lambda: None)
    admission._validation_snapshot = object()
    monkeypatch.setattr(admission, "_revalidate_live", lambda snapshot=None: None)
    stack = AsyncExitStack()
    admission.install_on(stack)
    coordinator = registration.OutputRegistrationV2(admission, config, key=base._KEY, journal_format=journal_format)
    admission._output_registration = coordinator
    return coordinator, stack, state


def _observation_events(source):
    """One owner lifetime with a policy, a delegated target and a registry event."""
    owner = {"kind": "owner", "owner": "events.retention", "native_identity": None,
             "plan": {"kind": "worker", "producer": "events", "role": "retention"}}
    database = source / "events.db"
    database.touch()
    policy = {"schema_version": 1, "source_bindings": ["b" * 64], "owned_targets": [],
              "delegated_targets": ["c" * 64], "historical_targets": []}
    scope = {"scope_schema_version": 1, "source_binding": "b" * 64, "delegation_sha256": "d" * 64,
             "source_close_sha256": "e" * 64, "source_state_root": str(source),
             "source_state_identity": base._identity(source)}
    return [
        dict(owner, phase="intent"), dict(owner, phase="attached"),
        {"kind": "delegation_policy", "scope_schema_version": 1,
         "ownership_sha256": base._sha(base._encoded(policy)), "policy": policy},
        {"kind": "delegated_target", "owner": "events.retention", "id": "c" * 64, "status": "attached",
         "plan": registration._plan("events", "historical", database, source),
         "borrowed_reader": False, "scope": scope},
        {"kind": "registry", "owner": "events.retention", "digest": "f" * 64, "status": "complete"},
        dict(owner, phase="active"), dict(owner, phase="closed"),
    ]


def _rewrite(directory, events):
    """Replace the event chain after the original header; recompute every digest."""
    for path in directory.glob("[0-9]*.json"):
        path.unlink()
    previous = base._sha((directory / "epoch.json").read_bytes())
    revision = 0
    for revision, event in enumerate(events, 1):
        raw = base._encoded(dict(event, revision=revision, previous_sha256=previous))
        previous = base._sha(raw)
        (directory / f"{revision:012d}-{previous}.json").write_bytes(raw)
    (directory / "HEAD.json").write_bytes(base._encoded({"revision": revision, "sha256": previous}))


def _mutate_observation(events, case):
    events = deepcopy(events)
    if case == "delegated_scope":
        del events[3]["scope"]["source_binding"]
    elif case == "policy_digest":
        events[2]["policy"]["owned_targets"] = ["c" * 64]
    elif case == "registry_status":
        events[4]["status"] = "complete-but-unknown"
    elif case == "owner_name":
        for event in events:
            if "owner" in event:
                event["owner"] = "events.eraser"
    elif case == "owner_plan":
        for event in events:
            if event["kind"] == "owner":
                event["plan"] = dict(event["plan"], sqlite_path="/state/x.db")
    return events


async def _write_valid(coordinator, source):
    await coordinator._run(coordinator._start)
    for event in _observation_events(source):
        await coordinator._run(coordinator._record, event)
    return [base._read(path)[0] for path in sorted(coordinator._directory.glob("[0-9]*.json"))]


@pytest.mark.parametrize("case,code", [
    ("delegated_scope", "delegated_scope"), ("policy_digest", "delegated_policy"),
    ("registry_status", "event"), ("owner_name", "owner"), ("owner_plan", "plan"),
])
async def test_v3_chain_applies_original_v2_event_checks_with_the_same_code(tmp_path, monkeypatch, case, code):
    from src.core import runtime_admission
    writers = {name: _format_writer(tmp_path, monkeypatch, name, fmt)
               for name, fmt in (("v2", registration.V2_FORMAT), ("v3", registration_v3.V3_FORMAT))}
    monkeypatch.setattr(runtime_admission, "_verified_admissions",
                        {coordinator.admission: False for coordinator, _, _ in writers.values()})
    errors = {}
    try:
        for name, (coordinator, _, state) in writers.items():
            source = tmp_path / name / "source"
            source.mkdir()
            written = await _write_valid(coordinator, source)
            directory = coordinator._directory
            assert directory.parents[1].name == name
            # The genuine production-format run decodes before mutation.
            header, _, states, _ = coordinator._format.decode_chain(directory)
            assert header["schema_version"] == coordinator._format.schema_version
            assert states["events.retention"]["phase"] == "closed" and base._chain(directory)[2] == states
            events = [{k: v for k, v in event.items() if k not in {"revision", "previous_sha256"}}
                      for event in written]
            _rewrite(directory, events)
            assert coordinator._format.decode_chain(directory)[2] == states  # Rewrite alone is neutral.
            _rewrite(directory, _mutate_observation(events, case))
            with pytest.raises(RuntimeError) as decoded:
                coordinator._format.decode_chain(directory)
            with pytest.raises(RuntimeError) as dispatched:
                base._chain(directory)
            assert str(decoded.value) == str(dispatched.value)
            errors[name] = str(decoded.value)
    finally:
        for coordinator, stack, _ in writers.values():
            await coordinator._disk.aclose()
            await stack.__aexit__(None, None, None)
    assert errors == {"v2": "output_registration_" + code, "v3": "output_registration_" + code}
