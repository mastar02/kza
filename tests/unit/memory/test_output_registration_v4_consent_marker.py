"""OQ-8: configuration_publisher/2 consent marker grammar and its boot reader.

Decoder contracts reuse the synthetic controller seam of the V4 tests. The
/1 grammar and its functions stay byte-identical (pinned in
test_output_registration_v4.py); /2 is an append-only table entry.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

from src.memory import output_registration as base
from src.memory import output_registration_v2 as registration
from tests.unit.memory.test_delegated_output_inventory import decoder  # noqa: F401
from tests.unit.memory.test_output_registration_v4 import (
    _aggregate, _composition, _configuration_events, _configuration_plan, _record, _v4_run,
)

V1 = {"accounts": "account_publisher/1", "configuration": "configuration_publisher/1"}
V2 = {"accounts": "account_publisher/1", "configuration": "configuration_publisher/2"}
SCOPE = "reasoner_cloud"


def v4():
    from src.memory import output_registration_v4
    return output_registration_v4


def withdrawal(plan):
    return {"origin": "actor", "basis": "actor", "scope": SCOPE,
            "pinned_config_sha256": plan["initial_sha256"], "source": None}


def inherited(basis="actor", *, pinned="d" * 64, run="3" * 32, generation=2, record="e" * 64):
    return {"origin": "inherited", "basis": basis, "scope": SCOPE, "pinned_config_sha256": pinned,
            "source": {"run_id": run, "generation": generation, "record_sha256": record, "reason": None}}


def detected(reason="chain_invalid", *, run="3" * 32, generation=2):
    return {"origin": "inherited", "basis": "unverifiable", "scope": SCOPE, "pinned_config_sha256": None,
            "source": {"run_id": run, "generation": generation, "record_sha256": None, "reason": reason}}


def events_v2(plan, *, decision=None, withdrawn=False, outcome="no_pending", end=None):
    """intent, attached, composition[, withdrawn][, opening][, drained]; end truncates after a phase."""
    event = {"kind": "configuration_publisher", "plan": plan, "decision": decision}
    marker = withdrawal(plan) if withdrawn else None
    out = [dict(event, phase="intent"), dict(event, phase="attached"), _composition(V2)]
    if withdrawn:
        out.append(dict(event, phase="withdrawn", withdrawal=marker))
    if outcome in {"committed", "clean_failed"}:
        out.append(dict(event, phase="opening", withdrawal=marker))
    if outcome is not None:
        out.append(dict(event, phase="drained", withdrawal=marker, aggregate=_aggregate(plan, outcome)))
    if end is not None:
        index = max(i for i, item in enumerate(out) if item.get("phase") == end)
        out = out[:index + 1]
    return out


def run_v2(decoder, run, events, *, generation=3, grammars=V2, selected=None):  # noqa: F811
    return _v4_run(decoder, run, events, grammars=grammars, selected=selected,
                   header_update={"generation": generation})


def runs(decoder):  # noqa: F811
    return decoder["root"] / "admission/output-inventory/v4/runs"


def shape(event, prior=None, *, selected=True, decoder):  # noqa: F811
    state = decoder["state"]
    return v4().configuration_event_shape_v2(event, prior, selected, state, base._directory(state))


# ----------------------------------------------------------------------
# Table: /2 is appended, /1 stays the same object and function.
# ----------------------------------------------------------------------

def test_v2_grammar_is_appended_and_is_the_writer_grammar():
    module = v4()
    entry = module._GRAMMARS["configuration_publisher/2"]
    assert (entry.grammar_id, entry.producer, entry.event_kind, entry.decode_event) == (
        "configuration_publisher/2", "configuration", "configuration_publisher", module._configuration_event_v2)
    assert module._GRAMMARS["configuration_publisher/1"].decode_event is module._configuration_event
    assert dict(module.WRITER_GRAMMARS) == V2
    assert dict(module.CONFIGURATION_OUTCOMES_V2) == {
        "no_pending": frozenset({"attached", "withdrawn"}),
        "not_attempted_graph_failed": frozenset({"withdrawn"}),
        "refused_before_temporary": frozenset({"withdrawn", "opening"}),
        "clean_failed": frozenset({"opening"}), "committed": frozenset({"opening"})}
    # /1 outcomes are untouched.
    assert module.CONFIGURATION_OUTCOMES["no_pending"] == frozenset({"attached"})


# ----------------------------------------------------------------------
# Grammar /2 (decoder seam).
# ----------------------------------------------------------------------

@pytest.mark.parametrize("decision", [None, "actor", "unverifiable_detected", "unverifiable_inherited"])
@pytest.mark.parametrize("withdrawn,outcome", [
    (False, "no_pending"), (True, "no_pending"), (True, "not_attempted_graph_failed"),
    (True, "refused_before_temporary"), (True, "clean_failed"), (True, "committed"), (True, None),
    (False, None)])
def test_valid_v2_sequences_decode_with_sticky_marker(decoder, decision, withdrawn, outcome):  # noqa: F811
    plan = _configuration_plan(decoder)
    value = {None: None, "actor": inherited(), "unverifiable_detected": detected(),
             "unverifiable_inherited": inherited("unverifiable", pinned=None)}[decision]
    directory = run_v2(decoder, "4" * 32, events_v2(plan, decision=value, withdrawn=withdrawn, outcome=outcome))
    _, _, states, _ = v4().chain(directory)
    state = states["configuration"]
    assert state["decision"] == value and state["plan"] == plan and state["producer"] == "configuration"
    assert state["withdrawal"] == (withdrawal(plan) if withdrawn else None)
    # M6: the effective marker is sticky, whatever the drain outcome.
    assert state["marker_out"] == (withdrawal(plan) if withdrawn else value)
    assert state["phase"] == ("drained" if outcome else ("withdrawn" if withdrawn else "attached"))


def test_marker_out_stays_after_a_committed_shutdown_write(decoder):  # noqa: F811
    plan = _configuration_plan(decoder)
    directory = run_v2(decoder, "4" * 32, events_v2(plan, withdrawn=True, outcome="committed"))
    state = v4().chain(directory)[2]["configuration"]
    assert state["aggregate"]["outcome"] == "committed" and state["marker_out"] == withdrawal(plan)


@pytest.mark.parametrize("case", [
    "intent_without_decision", "attached_decision_differs", "opening_from_attached", "two_withdrawals",
    "pinned_differs", "withdrawal_identity_key", "decision_identity_key", "withdrawal_origin_inherited",
    "decision_origin_actor", "actor_decision_without_pinned", "unknown_reason", "detected_with_record",
    "source_run_not_hex", "source_generation_zero", "scope_other", "drained_drops_withdrawal",
    "drained_invents_withdrawal", "graph_failed_from_attached", "refused_from_attached",
    "attached_with_withdrawal_key", "withdrawn_decision_changed", "disabled_without_decision",
    "timestamp_in_decision", "opening_from_attached_consistent"])
def test_invalid_v2_events_are_rejected(decoder, case):  # noqa: F811
    plan = _configuration_plan(decoder)
    events = events_v2(plan, decision=None, withdrawn=True, outcome="committed")
    config = [event for event in events if event["kind"] == "configuration_publisher"]
    intent, attached, withdrawn_event, opening, drained = config
    selected = None
    if case == "intent_without_decision":
        del intent["decision"]
    elif case == "attached_decision_differs":
        attached["decision"] = inherited()
    elif case == "opening_from_attached":
        events = [e for e in events if e.get("phase") != "withdrawn"]
    elif case == "opening_from_attached_consistent":
        # M5: a shutdown write without a recorded withdrawal, otherwise self-consistent.
        events = [e for e in events if e.get("phase") != "withdrawn"]
        for event in (opening, drained):
            event["withdrawal"] = None
    elif case == "two_withdrawals":
        events.insert(events.index(withdrawn_event) + 1, deepcopy(withdrawn_event))
    elif case == "pinned_differs":
        for event in (withdrawn_event, opening, drained):
            event["withdrawal"] = dict(event["withdrawal"], pinned_config_sha256="f" * 64)
    elif case == "withdrawal_identity_key":
        for event in (withdrawn_event, opening, drained):
            event["withdrawal"] = dict(event["withdrawal"], actor_subject_sha256="a" * 64)
    elif case == "decision_identity_key":
        for event in config:
            event["decision"] = dict(inherited(), subject="synthetic-admin")
    elif case == "withdrawal_origin_inherited":
        for event in (withdrawn_event, opening, drained):
            event["withdrawal"] = dict(event["withdrawal"], origin="inherited")
    elif case == "decision_origin_actor":
        for event in config:
            event["decision"] = dict(inherited(), origin="actor")
    elif case == "actor_decision_without_pinned":
        for event in config:
            event["decision"] = inherited(pinned=None)
    elif case == "unknown_reason":
        for event in config:
            event["decision"] = detected("operator_said_so")
    elif case == "detected_with_record":
        for event in config:
            event["decision"] = dict(detected(), source=dict(detected()["source"], record_sha256="e" * 64))
    elif case == "source_run_not_hex":
        for event in config:
            event["decision"] = inherited(run="Z" * 32)
    elif case == "source_generation_zero":
        for event in config:
            event["decision"] = inherited(generation=0)
    elif case == "scope_other":
        for event in config:
            event["decision"] = dict(inherited(), scope="spotify")
    elif case == "drained_drops_withdrawal":
        drained["withdrawal"] = None
    elif case == "drained_invents_withdrawal":
        events = events_v2(plan, withdrawn=False, outcome="no_pending")
        events[-1]["withdrawal"] = withdrawal(plan)
    elif case == "graph_failed_from_attached":
        events = events_v2(plan, withdrawn=False, outcome="not_attempted_graph_failed")
    elif case == "refused_from_attached":
        events = events_v2(plan, withdrawn=False, outcome="refused_before_temporary")
    elif case == "attached_with_withdrawal_key":
        attached["withdrawal"] = None
    elif case == "withdrawn_decision_changed":
        for event in (withdrawn_event, opening, drained):
            event["decision"] = inherited()
    elif case == "disabled_without_decision":
        events = [{"kind": "configuration_publisher", "phase": "disabled",
                   "path": str(v4().configuration_path(decoder["state"]))}]
        selected = {"configuration": False}
    elif case == "timestamp_in_decision":
        for event in config:
            event["decision"] = dict(inherited(), recorded_at=1)
    directory = run_v2(decoder, "4" * 32, events, selected=selected)
    # Rejected by the /2 shape itself, never merely as an unknown grammar.
    with pytest.raises(RuntimeError, match=r"^output_registration_(?!unsupported_producer_grammar)"):
        v4().chain(directory)


def test_disabled_publisher_carries_the_decision(decoder):  # noqa: F811
    event = {"kind": "configuration_publisher", "phase": "disabled",
             "path": str(v4().configuration_path(decoder["state"])), "decision": inherited()}
    directory = run_v2(decoder, "4" * 32, [event], selected={"configuration": False})
    state = v4().chain(directory)[2]["configuration"]
    assert state["phase"] == "disabled" and state["plan"] is None
    assert state["decision"] == state["marker_out"] == inherited()


def test_v1_runs_still_decode_under_the_unchanged_grammar(decoder):  # noqa: F811
    plan = _configuration_plan(decoder)
    directory = _v4_run(decoder, "4" * 32, _configuration_events(plan), grammars=V1)
    state = v4().chain(directory)[2]["configuration"]
    assert state["phase"] == "drained" and "marker_out" not in state and "decision" not in state


def test_shape_function_returns_the_documented_state(decoder):  # noqa: F811
    plan = _configuration_plan(decoder)
    event = {"kind": "configuration_publisher", "phase": "intent", "plan": plan, "decision": None}
    assert shape(event, decoder=decoder) == {
        "phase": "intent", "plan": plan, "producer": "configuration", "aggregate": None,
        "decision": None, "withdrawal": None, "marker_out": None}


# ----------------------------------------------------------------------
# Boot reader (pure, read-only).
# ----------------------------------------------------------------------

def read(decoder, exclude="0" * 32):  # noqa: F811
    return v4().prior_consent_decision(runs(decoder), exclude)


def last_sha(directory):
    return v4().chain(directory)[2]["configuration"]["sha256"]


def test_reader_without_runs_or_v2_runs_returns_none(decoder):  # noqa: F811
    result = read(decoder)
    assert (result.basis, result.decision) == ("none", None)
    plan = _configuration_plan(decoder)
    _v4_run(decoder, "4" * 32, _configuration_events(plan), grammars=V1)
    result = read(decoder)
    assert (result.basis, result.decision) == ("none", None)
    with pytest.raises(Exception):
        result.basis = "actor"  # Frozen.


def test_reader_returns_an_inherited_copy_of_the_newest_marker(decoder):  # noqa: F811
    plan = _configuration_plan(decoder)
    directory = run_v2(decoder, "5" * 32, events_v2(plan, withdrawn=True, outcome=None), generation=7)
    result = read(decoder)
    assert result.basis == "actor"
    assert result.decision == {"origin": "inherited", "basis": "actor", "scope": SCOPE,
        "pinned_config_sha256": plan["initial_sha256"],
        "source": {"run_id": "5" * 32, "generation": 7, "record_sha256": last_sha(directory), "reason": None}}


def test_reader_carries_a_recorded_unverifiable_decision(decoder):  # noqa: F811
    plan = _configuration_plan(decoder)
    directory = run_v2(decoder, "5" * 32, events_v2(plan, decision=detected()), generation=7)
    result = read(decoder)
    assert result.basis == "unverifiable"
    assert result.decision == {"origin": "inherited", "basis": "unverifiable", "scope": SCOPE,
        "pinned_config_sha256": None,
        "source": {"run_id": "5" * 32, "generation": 7, "record_sha256": last_sha(directory), "reason": None}}


def test_newest_null_decision_wins_over_an_older_marker(decoder):  # noqa: F811
    plan = _configuration_plan(decoder)
    run_v2(decoder, "5" * 32, events_v2(plan, withdrawn=True), generation=2)
    run_v2(decoder, "6" * 32, events_v2(plan, decision=None), generation=3)
    assert read(decoder).basis == "none"


def test_ordering_is_by_generation_not_run_id(decoder):  # noqa: F811
    plan = _configuration_plan(decoder)
    run_v2(decoder, "f" * 32, events_v2(plan, withdrawn=True), generation=2)
    run_v2(decoder, "1" * 32, events_v2(plan), generation=5)
    assert read(decoder).basis == "none"


def test_ordering_is_by_generation_not_run_id_reversed(decoder):  # noqa: F811
    plan = _configuration_plan(decoder)
    run_v2(decoder, "1" * 32, events_v2(plan, withdrawn=True), generation=5)
    run_v2(decoder, "f" * 32, events_v2(plan), generation=2)
    assert read(decoder).basis == "actor"


def test_newest_run_without_configuration_state_steps_back(decoder):  # noqa: F811
    plan = _configuration_plan(decoder)
    run_v2(decoder, "5" * 32, events_v2(plan, withdrawn=True), generation=2)
    run_v2(decoder, "6" * 32, [], generation=3)  # Crashed before intent.
    result = read(decoder)
    assert result.basis == "actor" and result.decision["source"]["run_id"] == "5" * 32


def test_torn_newest_run_is_unverifiable(decoder):  # noqa: F811
    plan = _configuration_plan(decoder)
    run_v2(decoder, "5" * 32, events_v2(plan), generation=2)
    directory = run_v2(decoder, "6" * 32, events_v2(plan, withdrawn=True, outcome=None), generation=3)
    # A record file whose HEAD never advanced (crash inside the append).
    raw = base._encoded({"kind": "configuration_publisher", "torn": True})
    (directory / f"{99:012d}-{hashlib.sha256(raw).hexdigest()}.json").write_bytes(raw)
    result = read(decoder)
    assert result.basis == "unverifiable"
    assert result.decision == detected("chain_invalid", run="6" * 32, generation=3)


@pytest.mark.parametrize("files", [("epoch.json",), ("epoch.json", "HEAD.next"), ("HEAD.next",), ()])
def test_empty_run_is_ignored(decoder, files):  # noqa: F811
    plan = _configuration_plan(decoder)
    run_v2(decoder, "5" * 32, events_v2(plan, withdrawn=True), generation=2)
    directory = run_v2(decoder, "6" * 32, [], generation=3)
    for path in directory.iterdir():
        if path.name not in files:
            path.unlink()
    for name in files:
        if not (directory / name).exists():
            (directory / name).write_bytes(b"{")
    assert read(decoder).basis == "actor"


@pytest.mark.parametrize("unreadable", [b"", b"{not json", b"[]"])
def test_unreadable_epoch_beside_head_or_records_is_unverifiable(decoder, unreadable):  # noqa: F811
    plan = _configuration_plan(decoder)
    run_v2(decoder, "5" * 32, events_v2(plan), generation=9)
    directory = run_v2(decoder, "6" * 32, events_v2(plan), generation=2)
    (directory / "epoch.json").write_bytes(unreadable)
    result = read(decoder)
    assert result.basis == "unverifiable"
    assert result.decision == detected("epoch_unreadable_with_records", run="6" * 32, generation=None)


def test_duplicate_generation_is_unverifiable(decoder):  # noqa: F811
    plan = _configuration_plan(decoder)
    run_v2(decoder, "5" * 32, events_v2(plan), generation=4)
    run_v2(decoder, "6" * 32, events_v2(plan), generation=4)
    result = read(decoder)
    assert result.basis == "unverifiable"
    assert result.decision == detected("duplicate_generation", run=None, generation=4)


def test_current_run_is_excluded(decoder):  # noqa: F811
    plan = _configuration_plan(decoder)
    run_v2(decoder, "5" * 32, events_v2(plan, withdrawn=True), generation=4)
    assert read(decoder, exclude="5" * 32).basis == "none"


def test_reader_unexpected_exception_is_unverifiable(decoder, monkeypatch):  # noqa: F811
    plan = _configuration_plan(decoder)
    run_v2(decoder, "5" * 32, events_v2(plan, withdrawn=True), generation=4)

    def explode(directory):
        raise KeyError("synthetic decoder defect")
    monkeypatch.setattr(v4(), "chain", explode)
    result = read(decoder)
    assert result.basis == "unverifiable"
    assert result.decision == detected("chain_invalid", run="5" * 32, generation=4)


def test_reader_is_read_only(decoder):  # noqa: F811
    plan = _configuration_plan(decoder)
    run_v2(decoder, "5" * 32, events_v2(plan, withdrawn=True), generation=4)
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in runs(decoder).rglob("*") if path.is_file()}
    read(decoder)
    assert before == {path: (path.read_bytes(), path.stat().st_mtime_ns)
                      for path in runs(decoder).rglob("*") if path.is_file()}


# ----------------------------------------------------------------------
# m10(g): read_inventory over /2 runs.
# ----------------------------------------------------------------------

@pytest.mark.parametrize("end", ["withdrawn", "opening", "drained", "disabled"])
def test_read_inventory_observes_v2_runs(decoder, end):  # noqa: F811
    plan = _configuration_plan(decoder)
    if end == "disabled":
        events = [{"kind": "configuration_publisher", "phase": "disabled",
                   "path": str(v4().configuration_path(decoder["state"])), "decision": inherited()},
                  _composition(V2)]
        run_v2(decoder, "4" * 32, events, selected={"configuration": False})
    else:
        run_v2(decoder, "4" * 32, events_v2(plan, decision=inherited(), withdrawn=True, outcome="committed",
                                            end=end))
    value = _record(decoder)
    assert value["phase"] == "observed", value
    state = value["stores"]["configuration"]
    assert state["phase"] == end and state["plan"] == (None if end == "disabled" else plan)
    assert state["marker_out"] == (inherited() if end == "disabled" else withdrawal(plan))
    assert Path(value["header"]["state_root"]) == decoder["state"]
    assert registration.read_inventory  # The inherited reader is the one exercised.
