"""Extensible V4 grammar: header-declared producer grammars from a closed table.

Decoder contracts reuse the synthetic controller seam of the delegated
inventory tests; writer tests use the managed coordinator fixture. Real
compiled ancestry (native-03/06) is a separate oracle.
"""
import ast
from copy import deepcopy
import hashlib
import inspect
from pathlib import Path

import pytest

from scripts import stage_runtime as stage
from src.memory import output_registration as base
from src.memory import output_registration_v2 as registration
from src.memory import output_registration_v3 as registration_v3
# The shared synthetic controller fixture is used by name in each test.
from tests.unit.memory.test_delegated_output_inventory import _assert_unknown, _write, decoder  # noqa: F401
from tests.unit.memory.test_output_registration_families import managed
from tests.unit.memory.test_output_registration_v3 import (
    _format_writer, _mutate_observation, _rewrite, _v3_run, _write_valid,
)
from tests.unit.memory.test_output_registration_v3 import _events as _account_events
from tests.unit.memory.test_output_registration_v3 import _plan as _account_plan

SELECTED = ("privacy_authority", "authority_sessions", "routines", "events", "suggestions")
GRAMMARS = {"accounts": "account_publisher/1", "configuration": "configuration_publisher/1"}
BINDING_SHA = "b" * 64  # The synthetic binding's settings.yaml projection.


def v4():
    """Import lazily so each test fails on its own before the module exists."""
    from src.memory import output_registration_v4
    return output_registration_v4


# ----------------------------------------------------------------------
# T00: V1-V3 decoders and constants are byte-for-byte unchanged.
# ----------------------------------------------------------------------

PINNED = {
    "base._chain_v1": ("7ed8496d2357f558548c8455fcb9b57ea533c8c70dfe54fb7de491d5cdb5f9ac",
                       "8fed1110d9dbaf5fe39a7f0c74c18605b7577e7c3e8900186662f9ca0a193eeb"),
    "base._plan_shape": ("31788acf2a49d264aac8fefd566fa10dcd1566d7f86b9179fbd606019cff9e87",
                         "5190ae172b1c1b922b0c794c780b5c4b415f76615e3299863a6bdcf4288ba076"),
    "v2.chain": ("fd524d754438f1fc6965e7105d114c8049e89896e689198b0c5910da07acc03c",
                 "3e0c592cc181b6061f3f2c1e2906d32c9cec52b8d2a03cad63e14031f692a7f6"),
    "v2._namespace": ("7aec861838b591ba5d2c7d8ae0ee072722def4e6cc99285795dc6587bdb5d2c1",
                      "798c3337b44ed76bd85e8f5bf98eaa5bfd2e37473378e99d78f083a0f53f1eb2"),
    "v2._plan_shape": ("6e0a20da515e1a268f19778ded94214de7d2c33623692b7cbfa796e16eeaa4ca",
                       "6c3040523a27dbf984c84bc4a94dcc338a1c5d4a74dab3273673aab416eba43a"),
    "v2._owner_event_shape": ("6c5b6ed108b3f0c12ac5e422e58680e77ff39ec9a9408fd94ba0f0870f0a9a62",
                              "1c664eb9623f94ba764ae96f351cf13fc5e3678141b08cd877b815428f01af4f"),
    "v2._observation_shape": ("e484ebe520c797d5331629535f7e4bb28f57a3ed8c193f0897abfc52f6700c9b",
                              "a1bedbb03ff4ce3de04242a9cabb1b8b2410d2896dc3bff76dfdd5611d311cdc"),
    "v3.chain": ("eb378e3b67edabb5429971d7c2799c157d96a25ef61b9ef1f7b50a32e7309daf",
                 "1f36f117f77f107270c0b8e2c4cb125b940cce35acb2700a05e0f2552acd9abb"),
    "v3.account_event_shape": ("3973083d5c682b5448d5017ff03078037f3695ad39d286155e84fd6ce2d2a7e0",
                               "7ad7b7099debe252434fdd1f86d9df34f6e5f707b52b2bcc92c18a39bb8a702b"),
    "v3.account_plan_shape": ("870f7f74a6929b5b2149b04038a565a19f29f3eba5be2ad934622ad9f5477bb1",
                              "3afd87a996c3efb2ed3c5d9d01613aa9591eb1747d76dac3d3614345f83b1df6"),
    "v3._aggregate_shape": ("c8ec01e01dbe87f7d298535cbb784d38b3bcec8ee1f9ff1dafb6ea6907c53cd5",
                            "d46c9e538f980cf75c138791089934b9271b3991433d94e748697e8222b31b6b"),
    "v3.composition_shape": ("393f87829342bd439a8df249c71d44a186c3437d7a02319d8859d7ca7cbe301e",
                             "0354fad8c0e0f77a50d9405269051649050c4e8cdbd2dc98becd74eba341a3c0"),
}
V3_FILE_SHA256 = "00a2d4f22fb1fd3fbf458a3c3241938106513ead09979a5a8a8852a2e52b1d3c"


def test_v1_v2_v3_decoders_and_constants_are_byte_and_ast_pinned():
    modules = {"base": base, "v2": registration, "v3": registration_v3}
    for name, (source_sha, ast_sha) in PINNED.items():
        prefix, attribute = name.split(".", 1)
        source = inspect.getsource(getattr(modules[prefix], attribute))
        assert hashlib.sha256(source.encode()).hexdigest() == source_sha, name
        assert hashlib.sha256(ast.dump(ast.parse(source)).encode()).hexdigest() == ast_sha, name
    assert hashlib.sha256(Path(registration_v3.__file__).read_bytes()).hexdigest() == V3_FILE_SHA256
    assert registration._SELECTED == frozenset({"lists", "reminders", *SELECTED})
    assert registration_v3._SELECTED_KEYS == frozenset({*SELECTED, "accounts"})
    assert registration_v3._HEADER_KEYS == frozenset({"schema_version", "run_id", "generation", "binding_id",
        "config_sha256", "state_root", "state_identity", "namespace_identity", "producers", "selected"})
    for fmt, version, name, known, chain in (
            (registration.V2_FORMAT, 2, "v2", {"v1", "v2"}, registration.chain),
            (registration_v3.V3_FORMAT, 3, "v3", {"v1", "v2", "v3"}, registration_v3.chain)):
        assert (fmt.schema_version, fmt.namespace_name, fmt.known_versions, fmt.decode_chain) == (
            version, name, frozenset(known), chain)
    assert registration_v3.V3_FORMAT.selected_producers == registration._SELECTED | {"accounts"}


def test_v4_grammar_table_is_closed_and_reuses_the_unchanged_v3_account_function(decoder):  # noqa: F811
    module = v4()
    table = module._GRAMMARS
    assert {key: (entry.grammar_id, entry.producer, entry.event_kind) for key, entry in table.items()} == {
        "account_publisher/1": ("account_publisher/1", "accounts", "account_publisher"),
        "configuration_publisher/1": ("configuration_publisher/1", "configuration", "configuration_publisher")}
    with pytest.raises(TypeError):
        table["future_publisher/1"] = table["account_publisher/1"]
    with pytest.raises(TypeError):
        table["account_publisher/1"].__reduce_ex__(2)
    assert dict(module.WRITER_GRAMMARS) == GRAMMARS
    fmt = module.V4_FORMAT
    assert (fmt.schema_version, fmt.namespace_name, fmt.known_versions, fmt.decode_chain) == (
        4, "v4", frozenset({"v1", "v2", "v3", "v4"}), module.chain)
    assert fmt.selected_producers == registration._SELECTED | {"accounts", "configuration"}
    with pytest.raises(TypeError):
        fmt.__reduce_ex__(2)
    # The account grammar is exactly the frozen V3 function, called through the table.
    plan = _account_plan(decoder)
    event = {"kind": "account_publisher", "phase": "intent", "plan": plan}
    header = {"state_root": str(decoder["state"]), "state_identity": base._directory(decoder["state"])}
    assert table["account_publisher/1"].decode_event(event, None, True, header) == \
        registration_v3.account_event_shape(event, None, True, decoder["state"])


# T00b pins: the V4 decoders and every existing table entry are append-only.
# A later producer adds a table entry and its own pin; editing these fails here.
V4_PINNED = {
    "chain": ("2cc7742cb88d4ecdc14006c280b5d65ec0a560d6f8f16b9ed48a92ca85d568db",
              "02932525f7fff17e97c522ba297a8465d7f6754fec2a4aafa64ce27aed475ccd"),
    "declared_grammars": ("7189593138386f1bee3a7c75cd96117197735417ab744952d00d93b2048e7d15",
                          "8f5ac26674d7044db82dd712095282d158d5d2217e9e09936c197384f845cf9f"),
    "composition_shape": ("9cc6087b4ea13196c2b4c78a93e54d02dcbf397c419198427cb481c8981d221c",
                          "226715df9c7da6d21250ca932d1ffb939a1d4e057b5f210ee92bc35772047a06"),
    "configuration_event_shape": ("92e036f8e2c7a1f23ca1691d5b3477728174efc6a29067269c468d47a3bb6781",
                                  "b885c9dc1054bc2f549faade20395c06b17375fa1105710f6ff1964653613605"),
    "configuration_plan_shape": ("87a123026011c2925c47273f71bd52403035ce8cbce8e1c03babaac4240af2df",
                                 "d250ff24be792104e33e13f052914d621d105cfd7de91642daac3892442b4982"),
    "_aggregate_shape": ("070e725367c1a343d84bbe1312db2e37c0a0c0a3e2dc0416a9b3eb3d1aa0af79",
                         "fedbbb035e427b03ca9f58c953d08315f9fc1d58c37d3d2346249e9b7a38d4a4"),
    "_configuration_event": ("134687fd0d52e964265d4b549b3584b03907a0b197a021f335f4fd4ba893f80d",
                             "e1363a23bc03a78cd6ebd5933af5967367b2062e5128080c108e15a4ebbdb396"),
    "_account_event": ("e02329d8bc35724692cc346d940256daa4285b1ab038fc08fd23f1a253cb368f",
                       "2325ca04a540cba220e3d9797496c8b8f965db936379c9249861a9fa32a6c621"),
}


def test_v4_decoders_and_grammar_entries_are_byte_and_ast_pinned():
    module = v4()
    for name, (source_sha, ast_sha) in V4_PINNED.items():
        source = inspect.getsource(getattr(module, name))
        assert hashlib.sha256(source.encode()).hexdigest() == source_sha, name
        assert hashlib.sha256(ast.dump(ast.parse(source)).encode()).hexdigest() == ast_sha, name
    entries = {key: (entry.producer, entry.event_kind, entry.decode_event) for key, entry in module._GRAMMARS.items()}
    assert entries == {
        "account_publisher/1": ("accounts", "account_publisher", module._account_event),
        "configuration_publisher/1": ("configuration", "configuration_publisher", module._configuration_event)}
    assert module.V4_FORMAT.decode_chain is module.chain
    assert dict(module.CONFIGURATION_OUTCOMES) == {
        "no_pending": frozenset({"attached"}), "not_attempted_graph_failed": frozenset({"attached"}),
        "refused_before_temporary": frozenset({"attached", "opening"}),
        "clean_failed": frozenset({"opening"}), "committed": frozenset({"opening"})}


# ----------------------------------------------------------------------
# Synthetic V4 runs (decoder seam).
# ----------------------------------------------------------------------

def _settings(state: Path, raw: bytes = b"reasoner:\n  cloud:\n    consent: true\n") -> Path:
    path = state / "config" / "settings.yaml"
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        path.write_bytes(raw)
    return path


def _configuration_plan(decoder, *, binding=True):  # noqa: F811
    _settings(decoder["state"])
    plan = v4().configuration_plan(decoder["state"])
    if binding:
        plan["initial_sha256"] = BINDING_SHA  # Joined to the synthetic binding projection.
    return plan


def _v4_run(decoder, run, events, *, selected=None, grammars=None, header_update=None):  # noqa: F811
    root = decoder["root"]
    directory = root / "admission/output-inventory/v4/runs" / run
    directory.mkdir(parents=True)
    grammars = dict(GRAMMARS if grammars is None else grammars)
    chosen = dict(dict.fromkeys(SELECTED, True), **dict.fromkeys(grammars, True))
    chosen.update(selected or {})
    header = dict(decoder["header"], schema_version=4, run_id=run, generation=3,
                  namespace_identity=base._directory(directory), selected=chosen, grammars=grammars)
    header.update(header_update or {})
    raw = base._encoded(header)
    (directory / "epoch.json").write_bytes(raw)
    _rewrite(directory, events)
    _write(root / "admission/receipts" / f"{run}.json", {"run_id": run, "binding_id": header["binding_id"],
        "generation": 3, "status": "closed", "resources": dict.fromkeys(stage._CLOSED_RESOURCES, True)})
    return directory


def _composition(producers):
    return {"kind": "composition", "instrumented": sorted({*SELECTED, "lists", "reminders", *producers}),
            "global_inventory_complete": False, "aggregate_copy_complete": False}


def _aggregate(plan, outcome="committed", *, sha=None):
    committed = outcome == "committed"
    return {"outcome": outcome, "attempts": 1 if outcome in {"committed", "clean_failed"} else 0,
            "last_committed_identity": {"device": 1, "inode": 2} if committed else plan["initial_identity"],
            "last_committed_sha256": (sha or "c" * 64) if committed else plan["initial_sha256"],
            "writer_returned": True, "open_temp_count": 0, "cleanup_uncertain": False}


def _configuration_events(plan, outcome="committed", *, sha=None):
    event = {"kind": "configuration_publisher", "plan": plan}
    events = [dict(event, phase="intent"), dict(event, phase="attached"), _composition(GRAMMARS)]
    if outcome in {"committed", "clean_failed"}:
        events.append(dict(event, phase="opening"))
    events.append(dict(event, phase="drained", aggregate=_aggregate(plan, outcome, sha=sha)))
    return events


def _record(decoder, version="v4"):  # noqa: F811
    records = registration.read_inventory(decoder["controller"])
    value, = (record for record in records if record["version"] == version)
    return value


# ----------------------------------------------------------------------
# T01: production writer and version isolation.
# ----------------------------------------------------------------------

async def test_production_writer_begins_a_v4_run_declaring_its_grammars(tmp_path, monkeypatch):
    async with managed(tmp_path, monkeypatch) as (owner, stack, config):
        assert owner._format is v4().V4_FORMAT
        assert owner._directory.parents[1].name == "v4"
        header, head, states, _ = v4().chain(owner._directory)
        assert header["schema_version"] == 4 and header["grammars"] == GRAMMARS
        assert set(header["selected"]) == {*SELECTED, "accounts", "configuration"}
        assert header["selected"]["configuration"] is header["selected"]["authority_sessions"] is True
        assert owner._expected["configuration"] == str(Path(header["state_root"]) / "config/settings.yaml")
        assert base._chain(owner._directory) == (header, head, states, v4().chain(owner._directory)[3])
        with pytest.raises(RuntimeError, match="header"):
            registration_v3.chain(owner._directory)
        with pytest.raises(RuntimeError, match="header"):
            registration.chain(owner._directory)


def test_v3_decoder_rejects_v4_and_v4_never_retries_older_versions(decoder):  # noqa: F811
    v3_directory = _v3_run(decoder, "7" * 32, _account_events(_account_plan(decoder)))
    with pytest.raises(RuntimeError, match="header"):
        v4().chain(v3_directory)
    plan = _configuration_plan(decoder)
    directory = _v4_run(decoder, "8" * 32, _configuration_events(plan))
    with pytest.raises(RuntimeError, match="header"):
        registration_v3.chain(directory)
    assert base._chain(directory)[0]["schema_version"] == 4
    # A malformed V4 header is refused as V4; it is never decoded as V3 or V2.
    header, _ = base._read(directory / "epoch.json")
    del header["grammars"]
    (directory / "epoch.json").write_bytes(base._encoded(header))
    _rewrite(directory, _configuration_events(plan))
    with pytest.raises(RuntimeError, match="^output_registration_header$"):
        base._chain(directory)


# ----------------------------------------------------------------------
# T02: configuration_publisher/1 grammar.
# ----------------------------------------------------------------------

def _mutate(events, case, plan, state):
    if case == "opening_without_attached":
        del events[1]
    elif case == "plan_changed":
        events[-1]["plan"] = dict(plan, initial_mode=0o600 if plan["initial_mode"] != 0o600 else 0o644)
    elif case == "plan_extra_key":
        for event in events:
            if "plan" in event:
                event["plan"] = dict(plan, content="reasoner")
    elif case == "aggregate_private_field":
        events[-1]["aggregate"]["yaml"] = "consent: false"
    elif case == "disabled_while_selected":
        events[:] = [{"kind": "configuration_publisher", "phase": "disabled", "path": plan["path"]}]
    elif case == "second_intent":
        events.append(deepcopy(events[0]))
    elif case == "committed_same_identity":
        events[-1]["aggregate"]["last_committed_identity"] = plan["initial_identity"]
    elif case == "no_pending_after_opening":
        events[-1]["aggregate"] = _aggregate(plan, "no_pending")
    elif case == "clean_failed_from_attached":
        del events[3]
        events[-1]["aggregate"] = _aggregate(plan, "clean_failed")
    elif case == "writer_not_returned":
        events[-1]["aggregate"]["writer_returned"] = False
    elif case == "cleanup_uncertain":
        events[-1]["aggregate"]["cleanup_uncertain"] = True
    elif case == "committed_sha_not_hex":
        events[-1]["aggregate"]["last_committed_sha256"] = "C" * 64
    elif case == "plan_path_foreign":
        for event in events:
            if "plan" in event:
                event["plan"] = dict(plan, path=str(state / "data" / "settings.yaml"))
    elif case == "state_identity_changed":
        for event in events:
            if "plan" in event:
                event["plan"] = dict(plan, state_identity={"device": 0, "inode": 0})
    elif case == "attempts_bool":
        events[-1]["aggregate"]["attempts"] = True
    elif case == "unknown_outcome":
        events[-1]["aggregate"]["outcome"] = "written"
    return events


CASES = ["opening_without_attached", "plan_changed", "plan_extra_key", "aggregate_private_field",
         "disabled_while_selected", "second_intent", "committed_same_identity", "no_pending_after_opening",
         "clean_failed_from_attached", "writer_not_returned", "cleanup_uncertain", "committed_sha_not_hex",
         "plan_path_foreign", "state_identity_changed", "attempts_bool", "unknown_outcome"]


@pytest.mark.parametrize("case", CASES)
def test_configuration_grammar_rejects_malformed_history(decoder, case):  # noqa: F811
    plan = _configuration_plan(decoder)
    directory = _v4_run(decoder, "5" * 32, _configuration_events(plan))
    assert _record(decoder)["phase"] == "observed"  # The unmutated run is valid.
    _rewrite(directory, _mutate(_configuration_events(plan), case, plan, decoder["state"]))
    _assert_unknown(_record(decoder))


def test_unselected_configuration_accepts_only_its_disabled_terminal(decoder):  # noqa: F811
    plan = _configuration_plan(decoder)
    disabled = {"kind": "configuration_publisher", "phase": "disabled", "path": plan["path"]}
    directory = _v4_run(decoder, "6" * 32, [disabled], selected={"configuration": False})
    value = _record(decoder)
    assert value["phase"] == "observed"
    assert value["stores"]["configuration"] == {"phase": "disabled", "plan": None, "producer": "configuration",
        "path": plan["path"], "revision": 1, "sha256": value["stores"]["configuration"]["sha256"]}
    assert "configuration" in value["uncovered_producers"]
    for events in ([dict(disabled, path=str(decoder["state"] / "settings.yaml"))],
                   [{"kind": "configuration_publisher", "phase": "intent", "plan": plan}]):
        _rewrite(directory, events)
        _assert_unknown(_record(decoder))


def test_committed_outcome_accepts_identical_bytes_on_a_new_inode(decoder):  # noqa: F811
    """F1: re-encoding an already-withdrawn file can publish identical bytes."""
    plan = _configuration_plan(decoder)
    _v4_run(decoder, "9" * 32, _configuration_events(plan, sha=plan["initial_sha256"]))
    value = _record(decoder)
    assert value["phase"] == "observed"
    aggregate = value["stores"]["configuration"]["aggregate"]
    assert aggregate["outcome"] == "committed" and aggregate["last_committed_sha256"] == plan["initial_sha256"]
    assert aggregate["last_committed_identity"] != plan["initial_identity"]


# ----------------------------------------------------------------------
# T03: original V2 event checks keep the same code objects in V4.
# ----------------------------------------------------------------------

@pytest.mark.parametrize("case,code", [
    ("delegated_scope", "delegated_scope"), ("policy_digest", "delegated_policy"),
    ("registry_status", "event"), ("owner_name", "owner"), ("owner_plan", "plan"),
])
async def test_v4_chain_applies_original_v2_event_checks_with_the_same_code(tmp_path, monkeypatch, case, code):
    from src.core import runtime_admission
    formats = (("v2", registration.V2_FORMAT), ("v3", registration_v3.V3_FORMAT), ("v4", v4().V4_FORMAT))
    writers = {name: _format_writer(tmp_path, monkeypatch, name, fmt) for name, fmt in formats}
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
            header, _, states, _ = coordinator._format.decode_chain(directory)
            assert header["schema_version"] == coordinator._format.schema_version
            assert states["events.retention"]["phase"] == "closed" and base._chain(directory)[2] == states
            events = [{k: v for k, v in event.items() if k not in {"revision", "previous_sha256"}}
                      for event in written]
            _rewrite(directory, events)
            assert coordinator._format.decode_chain(directory)[2] == states
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
    assert errors == dict.fromkeys(("v2", "v3", "v4"), "output_registration_" + code)


def test_v4_composition_requires_the_declared_producer_list(decoder):  # noqa: F811
    plan = _configuration_plan(decoder)
    events = _configuration_events(plan)
    events[2] = _composition({"accounts"})  # The V3 eight-producer list.
    _v4_run(decoder, "a" * 32, events)
    _assert_unknown(_record(decoder))
    # A run that declares only the account grammar composes exactly eight producers.
    account_plan = _account_plan(decoder)
    _v4_run(decoder, "b" * 32, _account_events(account_plan), grammars={"accounts": "account_publisher/1"})
    records = {r["run_id"]: r for r in registration.read_inventory(decoder["controller"])}
    assert records["b" * 32]["phase"] == "observed"
    assert records["b" * 32]["partial_producers"] == sorted({*SELECTED, "accounts"})


# ----------------------------------------------------------------------
# T04: reader joins, typed unknown grammar and writer refusal.
# ----------------------------------------------------------------------

def test_reader_joins_v1_to_v4_and_reports_configuration_as_partial(decoder):  # noqa: F811
    plan = _configuration_plan(decoder)
    _v4_run(decoder, "4" * 32, _configuration_events(plan))
    records = registration.read_inventory(decoder["controller"])
    by_version = {record["version"]: record for record in records}
    assert set(by_version) == {"v2", "v4"}
    value = by_version["v4"]
    assert value["phase"] == "observed" and value["header"]["schema_version"] == 4
    state = value["stores"]["configuration"]
    assert state["phase"] == "drained" and state["producer"] == "configuration"
    assert state["aggregate"]["outcome"] == "committed"
    assert value["partial_producers"] == sorted({*SELECTED, "accounts", "configuration"})
    assert value["instrumented_producers"] == ["configuration"]
    assert value["uncovered_producers"] == list(base.PRODUCERS)
    # The two-key configuration projection is unchanged for every version.
    assert value["configuration"] == {"raw_sha256": BINDING_SHA, "effective_digest": "observed_only"}
    assert by_version["v2"]["configuration"] == value["configuration"]
    assert all(record["capability"] is False and record["global_inventory_complete"] is False
               for record in records)


@pytest.mark.parametrize("grammars,selected", [
    ({"accounts": "account_publisher/1", "configuration": "configuration_publisher/2"}, {}),
    (dict(GRAMMARS, audio_health="audio_health_publisher/1"), {"audio_health": True}),
])
def test_unknown_grammar_is_its_own_typed_reason_and_never_decoded(decoder, grammars, selected):  # noqa: F811
    plan = _configuration_plan(decoder)
    events = _configuration_events(plan)
    events[2] = _composition(grammars)
    directory = _v4_run(decoder, "c" * 32, events, grammars=grammars, selected=selected)
    with pytest.raises(v4().UnsupportedProducerGrammar, match="^output_registration_unsupported_producer_grammar$"):
        v4().chain(directory)
    with pytest.raises(v4().UnsupportedProducerGrammar):
        base._chain(directory)
    value = _record(decoder)
    assert value["phase"] == "unknown" and value["reason"] == "unsupported_producer_grammar"
    assert not {"header", "stores", "journal", "configuration", "material_observations"} & value.keys()
    assert value["capability"] is False and value["uncovered_producers"] == list(base.PRODUCERS)


@pytest.mark.parametrize("case", ["grammar_for_other_producer", "undeclared_kind", "malformed_grammar_id",
                                  "base_family_grammar"])
def test_known_grammar_bound_to_another_producer_or_undeclared_kind_is_malformed(decoder, case):  # noqa: F811
    plan = _configuration_plan(decoder)
    events, grammars, selected = _configuration_events(plan), dict(GRAMMARS), {}
    if case == "grammar_for_other_producer":
        grammars["configuration"] = "account_publisher/1"
    elif case == "undeclared_kind":
        grammars = {"accounts": "account_publisher/1"}
        events[2] = _composition(grammars)
    elif case == "malformed_grammar_id":
        grammars["configuration"] = "Configuration Publisher"
    else:
        grammars["events"] = "account_publisher/1"
    _v4_run(decoder, "d" * 32, events, grammars=grammars, selected=selected)
    _assert_unknown(_record(decoder))


async def test_writer_refuses_to_start_over_a_run_with_an_unknown_grammar(tmp_path, monkeypatch):
    from src.core import runtime_admission
    coordinator, stack, _ = _format_writer(tmp_path, monkeypatch, "w", v4().V4_FORMAT)
    monkeypatch.setattr(runtime_admission, "_verified_admissions", {coordinator.admission: False})
    runs = coordinator._namespace / "v4/runs"
    newer = runs / ("e" * 32)
    newer.mkdir(parents=True)
    header = {"schema_version": 4, "grammars": dict(GRAMMARS, configuration="configuration_publisher/2")}
    (newer / "epoch.json").write_bytes(base._encoded(header))
    try:
        with pytest.raises(RuntimeError, match="^output_registration_unsupported_producer_grammar$"):
            await coordinator._run(coordinator._start)
        assert sorted(path.name for path in runs.iterdir()) == ["e" * 32]
        # A run declaring only known grammars is history, not a refusal.
        (newer / "epoch.json").write_bytes(base._encoded(dict(header, grammars=GRAMMARS)))
        coordinator._failed = False
        await coordinator._run(coordinator._start)
        assert coordinator._directory.is_dir() and coordinator._directory.parent == runs
    finally:
        await coordinator._disk.aclose()
        await stack.__aexit__(None, None, None)


def test_reader_rejects_a_configuration_plan_that_differs_from_the_binding(decoder):  # noqa: F811
    """F7: the attach-time sha must equal the admission binding projection."""
    plan = _configuration_plan(decoder, binding=False)
    assert plan["initial_sha256"] != BINDING_SHA
    _v4_run(decoder, "f" * 32, _configuration_events(plan))
    _assert_unknown(_record(decoder))


def test_same_run_in_v3_and_v4_namespaces_is_conflicting(decoder):  # noqa: F811
    run = "1" * 32
    _v3_run(decoder, run, _account_events(_account_plan(decoder)))
    _v4_run(decoder, run, _configuration_events(_configuration_plan(decoder)))
    records = [r for r in registration.read_inventory(decoder["controller"]) if r["run_id"] == run]
    assert {record["version"] for record in records} == {"v3", "v4"}
    assert all(record["reason"] == "conflicting_run_versions" for record in records)
