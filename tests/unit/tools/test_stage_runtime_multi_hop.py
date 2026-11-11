"""U3b-0b: multi-hop erasure-target delegation with real admitted children.

Every slot here runs the committed HEAD artifact (pre-0b code) for its probe
and its admitted child, while the controller is the working tree. That is the
compatibility the flattened records promise: older managed code accepts them.
"""

from contextlib import closing
import json
from pathlib import Path
import sqlite3

import pytest

from scripts import stage_runtime as stage
from tests.unit.tools import test_stage_runtime_admission as admission
from tests.unit.tools.test_stage_runtime_admission import (
    _CLOSING_CHILD,
    _TARGETS_DDL,
    _closed_head,
    _command,
    _copy_candidate,
    _delegate_cli,
    _refused,
    _registered_targets,
    _registration,
    _witness,
)

admission_archive = admission.admission_archive  # The module-scoped built HEAD artifact fixture.

_INSERT = "INSERT INTO erasure_targets VALUES(?,?,?,?)"


def _local(state, name):
    directory = state / "data" / ("clips-" + name)
    directory.mkdir()
    return _registration("collectors", 1, {"kind": "wake_clips", "root": str(directory)})


def _run(root, name, rows=()):
    statements = [["data/privacy.db", _INSERT, row] for row in rows]
    stage.run_slot(root, name, _command(_CLOSING_CHILD, json.dumps(statements)), timeout=30)
    head = stage.read_admission_head(root)
    assert head["status"] == "closed"
    return head["binding_id"]


def _hop(archive, root, source_state, name, *, register=True):
    """Copy the closed head's state, delegate with the CLI, then run the copy closed."""
    state = _copy_candidate(archive, root, source_state, name)
    result = _delegate_cli(root, name)
    assert result.returncode == 0, result.stderr
    rows = [_local(state, name)] if register else []
    return {"state": state, "output": json.loads(result.stdout), "binding": _run(root, name, rows),
            "local": rows[0][0] if rows else None}


def _provenance(root, state):
    """target id -> owner binding, hop binding ids and record capability."""
    found = {}
    for record in stage._delegation_records(root, state):
        chain = stage._read_chain(root, record)
        for identity, hops in chain["chains"].items():
            found[identity] = {"owner": record["source_binding"], "capability": stage._digest(record),
                               "hops": [hop["binding_id"] for hop in hops], "via": [hop["via"] for hop in hops]}
    return found


def _policy(state):
    return json.loads((state / ".kza-target-ownership.json").read_text())


@pytest.fixture
def origin(admission_archive, tmp_path):
    """A: a real closed head owning one path target and one path-less target."""
    registered = []

    def statements(state):
        rows = _registered_targets(state)
        registered.extend(row[2][0] for row in rows[1:])
        return rows

    root, state = _closed_head(admission_archive, tmp_path, statements)
    return root, {"state": state, "binding": stage.read_admission_head(root)["binding_id"],
                  "local": registered[0], "pathless": registered[1]}


@pytest.fixture
def three_states(origin, admission_archive):
    """A -> B -> C, each a real closed head; C is the current head."""
    root, a = origin
    b = _hop(admission_archive, root, a["state"], "b")
    c = _hop(admission_archive, root, b["state"], "c")
    return root, a, b, c


def test_two_hop_flattens_records_to_owners_and_records_the_chain(three_states):
    root, a, b, c = three_states
    assert sorted(c["output"]["targets"]) == sorted([a["local"], b["local"]])
    assert len(c["output"]["capabilities"]) == 2
    provenance = _provenance(root, c["state"])
    assert provenance[a["local"]]["owner"] == a["binding"]
    assert provenance[a["local"]]["hops"] == [a["binding"], b["binding"]]
    [first] = b["output"]["capabilities"]
    assert provenance[a["local"]]["via"] == [None, first]  # B holds t_A through A's grant.
    assert provenance[b["local"]] == {"owner": b["binding"], "hops": [b["binding"]], "via": [None],
                                      "capability": provenance[b["local"]]["capability"]}
    policy = _policy(c["state"])
    assert set(policy["delegated_targets"]) == {a["local"], b["local"]}
    # The path-less target stays owned at every hop; C's own registration was sealed.
    assert set(policy["owned_targets"]) == {a["pathless"], c["local"]}
    assert {a["binding"], b["binding"]} <= set(policy["source_bindings"])


def test_three_hop_chains_every_inherited_target_to_its_owner(three_states, admission_archive):
    root, a, b, c = three_states
    d = _hop(admission_archive, root, c["state"], "d")
    provenance = _provenance(root, d["state"])
    assert provenance[a["local"]]["hops"] == [a["binding"], b["binding"], c["binding"]]
    assert provenance[b["local"]]["hops"] == [b["binding"], c["binding"]]
    assert provenance[c["local"]]["hops"] == [c["binding"]]
    assert {item["owner"] for item in provenance.values()} == {a["binding"], b["binding"], c["binding"]}
    assert len(d["output"]["capabilities"]) == 3
    assert stage.read_admission_head(root)["binding_id"] == d["binding"]


def test_cli_redelegates_an_inherited_pathless_target(origin, admission_archive):
    root, a = origin
    b_state = _copy_candidate(admission_archive, root, a["state"], "b")
    with stage.AdmissionController(root) as controller:
        controller.authorize_target_delegation(b_state, a["binding"], [a["local"], a["pathless"]])
    b_binding = _run(root, "b")
    c = _hop(admission_archive, root, b_state, "c")
    assert sorted(c["output"]["targets"]) == sorted([a["local"], a["pathless"]])
    provenance = _provenance(root, c["state"])
    assert provenance[a["pathless"]]["hops"] == [a["binding"], b_binding]
    assert a["pathless"] in _policy(c["state"])["delegated_targets"]


@pytest.mark.parametrize("hops", [1, 2])
def test_delegating_back_into_an_earlier_state_is_a_cycle(origin, admission_archive, hops):
    root, a = origin
    b = _hop(admission_archive, root, a["state"], "b")
    states = [a["state"], b["state"]]
    if hops == 2:  # A -> B -> C -> B
        c = _hop(admission_archive, root, b["state"], "c")
        states.append(c["state"])
        destination, held = b["state"], [a["local"], b["local"]]
    else:  # A -> B -> A
        destination, held = a["state"], [a["local"]]
    head = stage.read_admission_head(root)
    with stage.AdmissionController(root) as controller:
        _refused(root, states, lambda: controller.authorize_target_delegation(
            destination, head["binding_id"], held), "^runtime_admission_delegation_cycle$")


def _rewrite(path, value):
    path.chmod(0o600)
    path.write_bytes(stage._canonical_json(value) + b"\n")


def _forge(root, a, b, c, mutation):
    """Tamper with C's record for A's target or its provenance sidecar."""
    record = next(record for record in stage._delegation_records(root, c["state"])
                  if a["local"] in record["targets"])
    capability = stage._digest(record)
    path = stage._chain_path(root, capability)
    chain = json.loads(path.read_text())
    hops = chain["chains"][a["local"]]
    if mutation == "sidecar_deleted":
        path.unlink()
    elif mutation == "intermediate_dropped":
        chain["chains"][a["local"]] = hops[:1]
    elif mutation == "reordered":
        chain["chains"][a["local"]] = hops[::-1]
    elif mutation == "destination_inserted":
        intruder = dict(hops[1], binding_id=c["binding"], state_root=str(c["state"]),
                        state_identity=stage._identity(c["state"]))
        chain["chains"][a["local"]] = [hops[0], intruder, hops[1]]
    elif mutation == "via_elsewhere":
        other = next(stage._digest(item) for item in stage._delegation_records(root, c["state"])
                     if stage._digest(item) != capability and b["local"] in item["targets"])
        hops[1]["via"] = other
    elif mutation == "grantor":
        chain["grantor"] = {"binding_id": a["binding"], "close_sha256": hops[0]["close_sha256"]}
    elif mutation == "closure_forged":
        hops[1]["closure"] = dict(hops[1]["closure"], child_pid=hops[1]["closure"]["child_pid"] + 1)
        hops[1]["close_sha256"] = stage._digest(hops[1]["closure"])
        chain["grantor"]["close_sha256"] = hops[1]["close_sha256"]
    if mutation != "sidecar_deleted":
        _rewrite(path, chain)


@pytest.mark.parametrize(("mutation", "match"), [
    ("sidecar_deleted", "^runtime_admission_delegation_chain_missing$"),
    ("intermediate_dropped", "^runtime_admission_delegation_chain$"),
    ("reordered", "^runtime_admission_delegation_chain$"),
    ("destination_inserted", "^runtime_admission_delegation_cycle$"),
    ("via_elsewhere", "^runtime_admission_delegation_chain$"),
    ("grantor", "^runtime_admission_delegation_chain$"),
    ("closure_forged", "^runtime_admission_source_close_proof$"),
])
def test_forged_or_truncated_chain_is_refused(three_states, admission_archive, mutation, match):
    root, a, b, c = three_states
    d_state = _copy_candidate(admission_archive, root, c["state"], "d")
    _forge(root, a, b, c, mutation)
    with stage.AdmissionController(root) as controller:
        _refused(root, (a["state"], b["state"], c["state"], d_state),
                 lambda: controller.delegate("d"), match)
    result = _delegate_cli(root, "d")
    assert result.returncode != 0 and "ValueError: runtime_admission_" in result.stderr


@pytest.mark.parametrize("hop", [1, 2, 3])
@pytest.mark.parametrize("where", ["outside", "third_state"])
def test_out_of_state_path_at_any_hop_is_refused(origin, admission_archive, tmp_path, hop, where):
    """A head without a policy (hop 1) refuses the foreign path itself; a head with
    a policy refuses any registration its policy does not list, before path checks."""
    root, a = origin
    states, head = [a["state"]], a
    for name in ("b", "c")[:hop - 1]:
        head = _hop(admission_archive, root, head["state"], name)
        states.append(head["state"])
    if where == "outside":
        directory = tmp_path / "outside-clips"
    else:  # Inside another slot's state that never registered it.
        directory = _copy_candidate(admission_archive, root, a["state"], "other") / "data" / "unregistered"
    directory.mkdir()
    foreign = _registration("collectors", 1, {"kind": "conversations", "root": str(directory)})
    with closing(sqlite3.connect(head["state"] / "data" / "privacy.db")) as db, db:
        db.execute(_INSERT, foreign)
    nxt = _copy_candidate(admission_archive, root, head["state"], "next")
    match = "foreign_target" if hop == 1 else "active_inventory_mismatch"
    with stage.AdmissionController(root) as controller:
        _refused(root, (*states, nxt), lambda: controller.delegate("next"), "^runtime_admission_" + match + "$")
    result = _delegate_cli(root, "next")
    assert result.returncode != 0 and "ValueError: runtime_admission_" + match in result.stderr


@pytest.mark.parametrize("changed", ["owner", "owner_marker", "intermediate"])
def test_owner_closure_change_revokes_but_an_intermediate_one_does_not(origin, admission_archive, changed):
    root, a = origin
    b = _hop(admission_archive, root, a["state"], "b", register=False)  # B is a pure intermediate.
    c = _hop(admission_archive, root, b["state"], "c")
    assert _provenance(root, c["state"])[a["local"]]["hops"] == [a["binding"], b["binding"]]
    d_state = _copy_candidate(admission_archive, root, c["state"], "d")
    if changed == "owner_marker":  # The owner was selected again elsewhere: live revocation.
        marker = a["state"] / stage._MARKER
        _rewrite(marker, dict(json.loads(marker.read_text()), binding_id="0" * 64))
        with stage.AdmissionController(root) as controller:
            _refused(root, (a["state"], b["state"], c["state"], d_state),
                     lambda: controller.delegate("d"), "^runtime_admission_delegation_missing$")
        return
    binding = a["binding"] if changed == "owner" else b["binding"]
    path = root / "admission" / "closures" / (binding + ".json")
    proof = json.loads(path.read_text())
    # The mutable closure file is rewritten (a rerun rewrites it). Intermediates are
    # anchored on the embedded proof and its immutable receipt, so only the owner's
    # live closure is read; this rewrite no longer matches its receipt.
    _rewrite(path, dict(proof, child_pid=proof["child_pid"] + 1))
    with stage.AdmissionController(root) as controller:
        if changed == "owner":
            _refused(root, (a["state"], b["state"], c["state"], d_state),
                     lambda: controller.delegate("d"), "^runtime_admission_source_close_proof$")
            return
        controller.delegate("d")
        assert controller.activate("d") == "slots/c/runtime"


def test_owner_slot_artifact_change_is_refused_before_the_head_moves(three_states, admission_archive):
    root, a, b, c = three_states
    d_state = _copy_candidate(admission_archive, root, c["state"], "d")
    assert _delegate_cli(root, "d").returncode == 0
    runtime = root / "slots" / "old" / "runtime"
    manifest = json.loads((runtime / "release.json").read_text())
    payload = runtime / next(name for name in sorted(manifest["files"]) if name.endswith(".pyc"))
    payload.chmod(0o600)
    payload.write_bytes(payload.read_bytes() + b"\0")
    states = (a["state"], b["state"], c["state"], d_state)
    with stage.AdmissionController(root) as controller:
        _refused(root, states, lambda: controller.activate("d"), "integrity mismatch")
    assert stage.read_admission_head(root)["binding_id"] == c["binding"]
    e_state = _copy_candidate(admission_archive, root, c["state"], "e")
    with stage.AdmissionController(root) as controller:
        _refused(root, (*states, e_state), lambda: controller.delegate("e"), "integrity mismatch")


def test_owner_limit_is_a_coded_refusal(three_states, admission_archive, monkeypatch):
    root, a, b, c = three_states
    d_state = _copy_candidate(admission_archive, root, c["state"], "d")
    monkeypatch.setattr(stage, "_DELEGATION_OWNER_LIMIT", 2)  # C holds targets of A, B and itself.
    with stage.AdmissionController(root) as controller:
        _refused(root, (a["state"], b["state"], c["state"], d_state),
                 lambda: controller.delegate("d"), "^runtime_admission_delegation_owner_limit$")


def test_same_record_with_a_different_chain_is_a_conflict(origin, admission_archive):
    root, a = origin
    b = _hop(admission_archive, root, a["state"], "b")
    d_state = _copy_candidate(admission_archive, root, b["state"], "d")
    c_state = _copy_candidate(admission_archive, root, b["state"], "c")
    assert _delegate_cli(root, "d").returncode == 0  # A's target reaches D as [A, B].
    assert _delegate_cli(root, "c").returncode == 0
    c_binding = _run(root, "c")
    with stage.AdmissionController(root) as controller:
        # The same owner record for D, now proven as [A, B, C]: refused before any write.
        _refused(root, (a["state"], b["state"], c_state, d_state),
                 lambda: controller.authorize_target_delegation(d_state, c_binding, [a["local"], b["local"]]),
                 "^runtime_admission_delegation_conflict$")


def test_failure_on_the_second_record_write_leaves_inert_residue_and_retry_succeeds(
        three_states, admission_archive, monkeypatch):
    root, a, b, c = three_states
    d_state = _copy_candidate(admission_archive, root, c["state"], "d")
    head = stage.read_admission_head(root)
    before = _witness(root, a["state"], b["state"], c["state"], d_state)
    original, written = stage._atomic_record, []

    def failing(path, value, *, immutable=False):
        if path.parent.name == "delegations" and len([p for p in written if p.parent.name == "delegations"]) == 1:
            raise OSError(28, "No space left on device")
        original(path, value, immutable=immutable)
        written.append(path)

    monkeypatch.setattr(stage, "_atomic_record", failing)
    with stage.AdmissionController(root) as controller:
        with pytest.raises(OSError):
            controller.delegate("d")
    monkeypatch.setattr(stage, "_atomic_record", original)
    after = _witness(root, a["state"], b["state"], c["state"], d_state)
    added = set(after["root"]) - set(before["root"])
    assert {p.parent.name for p in written} == {"delegation-chains", "delegations"}
    assert added == {path.relative_to(root).as_posix() for path in written}
    assert after["states"] == before["states"]  # No policy, no marker change.
    assert stage.read_admission_head(root) == head
    result = _delegate_cli(root, "d")
    assert result.returncode == 0, result.stderr
    assert stage.activate_slot(root, "d") == "slots/c/runtime"


def test_probe_refuses_a_candidate_whose_own_ownership_validation_fails(three_states, admission_archive,
                                                                       monkeypatch):
    root, a, b, c = three_states
    d_state = _copy_candidate(admission_archive, root, c["state"], "d")
    assert _delegate_cli(root, "d").returncode == 0
    marker = "    validate(Path(root), binding)"
    assert stage._PROBE_SOURCE.count(marker) == 1
    monkeypatch.setattr(stage, "_PROBE_SOURCE", stage._PROBE_SOURCE.replace(
        marker, "    raise ValueError('runtime_admission_delegation_missing')"))
    with stage.AdmissionController(root) as controller:
        _refused(root, (a["state"], b["state"], c["state"], d_state), lambda: controller.activate("d"),
                 "^runtime_admission_candidate_rejected: runtime_admission_delegation_missing$")
    assert stage.read_admission_head(root)["binding_id"] == c["binding"]


def test_record_without_a_sidecar_is_refused(origin, admission_archive):
    """A 0a-era record (no provenance) is not honored by the 0b controller."""
    root, a = origin
    b = _hop(admission_archive, root, a["state"], "b")
    [capability] = b["output"]["capabilities"]
    stage._chain_path(root, capability).unlink()
    c_state = _copy_candidate(admission_archive, root, b["state"], "c")
    with stage.AdmissionController(root) as controller:
        _refused(root, (a["state"], b["state"], c_state), lambda: controller.delegate("c"),
                 "^runtime_admission_delegation_chain_missing$")


def test_three_hop_chain_with_its_middle_hop_removed_is_refused(three_states, admission_archive):
    """[A, B, C] forged to [A, C]: C's own grant proves the prefix [A, B], not [A]."""
    root, a, b, c = three_states
    d = _hop(admission_archive, root, c["state"], "d")
    record = next(record for record in stage._delegation_records(root, d["state"])
                  if a["local"] in record["targets"])
    path = stage._chain_path(root, stage._digest(record))
    chain = json.loads(path.read_text())
    hops = chain["chains"][a["local"]]
    assert [hop["binding_id"] for hop in hops] == [a["binding"], b["binding"], c["binding"]]
    chain["chains"][a["local"]] = [hops[0], hops[2]]
    _rewrite(path, chain)
    e_state = _copy_candidate(admission_archive, root, d["state"], "e")
    with stage.AdmissionController(root) as controller:
        _refused(root, (a["state"], b["state"], c["state"], d["state"], e_state),
                 lambda: controller.delegate("e"), "^runtime_admission_delegation_chain$")


def test_a_record_naming_a_holder_as_owner_is_refused(origin, admission_archive):
    """The owner must own the target: B holds A's path-less target only as delegated."""
    root, a = origin
    b_state = _copy_candidate(admission_archive, root, a["state"], "b")
    with stage.AdmissionController(root) as controller:
        controller.authorize_target_delegation(b_state, a["binding"], [a["local"], a["pathless"]])
    b_binding = _run(root, "b")
    c = _hop(admission_archive, root, b_state, "c")
    target = stage._target_inventory(c["state"])[a["pathless"]]
    proof = stage._closure_proof(root, b_binding)
    b_binding_record = stage.read_binding(root, b_binding)
    forged = {"schema_version": 1, "destination_state": str(c["state"]),
              "destination_identity": stage._identity(c["state"]), "source_binding": b_binding,
              "source_close_sha256": stage._digest(proof), "targets": {a["pathless"]: target}}
    hop = {"binding_id": b_binding, "close_sha256": stage._digest(proof), "closure": proof,
           "state_root": b_binding_record["state_root"], "state_identity": b_binding_record["state_identity"],
           "via": None}
    chain = {"schema_version": 1, "record_sha256": stage._digest(forged), "destination_state": str(c["state"]),
             "destination_identity": stage._identity(c["state"]),
             "grantor": {"binding_id": b_binding, "close_sha256": stage._digest(proof)},
             "chains": {a["pathless"]: [hop]}}
    sources = _policy(c["state"])["source_bindings"]
    with stage.AdmissionController(root):
        with pytest.raises(ValueError, match="^runtime_admission_delegation_chain$"):
            stage._delegation_candidate(root, forged, chain, target, sources)


def test_a_former_grantor_is_never_a_destination_even_outside_the_granted_chains(origin, admission_archive):
    """A -> B, B -> C, then A is selected again directly. Granting A's target back into
    B would rewrite B's policy and silently revoke C, although B is no hop of [A]."""
    root, a = origin
    b = _hop(admission_archive, root, a["state"], "b", register=False)
    c_state = _copy_candidate(admission_archive, root, b["state"], "c")
    assert _delegate_cli(root, "c").returncode == 0  # B is now a grantor (a hop of C's chain).
    assert stage.activate_slot(root, "old") == "slots/b/runtime"
    head = stage.read_admission_head(root)
    assert head["binding_id"] == a["binding"]
    with stage.AdmissionController(root) as controller:
        _refused(root, (a["state"], b["state"], c_state),
                 lambda: controller.authorize_target_delegation(b["state"], a["binding"], [a["local"]]),
                 "^runtime_admission_delegation_cycle$")


def test_every_hop_must_be_pinned_in_the_destination_policy(three_states):
    """The hop set is pinned through the destination policy (and so its binding)."""
    root, a, b, c = three_states
    record = next(record for record in stage._delegation_records(root, c["state"])
                  if a["local"] in record["targets"])
    chain = stage._read_chain(root, record)
    target = record["targets"][a["local"]]
    sources = _policy(c["state"])["source_bindings"]
    with stage.AdmissionController(root):
        assert stage._delegation_candidate(root, record, chain, target, sources)["capability"] == \
            stage._digest(record)
        unpinned = [binding for binding in sources if binding != b["binding"]]
        with pytest.raises(ValueError, match="^runtime_admission_delegation_chain$"):
            stage._delegation_candidate(root, record, chain, target, unpinned)


def test_a_real_owner_rerun_revokes_its_downstream_targets(origin, admission_archive):
    """0a live semantics kept: rerunning the owner rewrites its closure, so C loses A's target."""
    root, a = origin
    b = _hop(admission_archive, root, a["state"], "b", register=False)
    c = _hop(admission_archive, root, b["state"], "c", register=False)
    assert stage.activate_slot(root, "old") == "slots/c/runtime"
    assert _run(root, "old") == a["binding"]  # Same state, same binding, new closure proof.
    with stage.AdmissionController(root) as controller:
        _refused(root, (a["state"], b["state"], c["state"]), lambda: controller.activate("c"),
                 "^runtime_admission_delegation_missing$")


# --- U3b-0b independent review: F1, F2, F4, F7, F8 -------------------------------------

def _record_for(root, state, target_id):
    return next(record for record in stage._delegation_records(root, state) if target_id in record["targets"])


def test_spliced_intermediate_closure_is_refused(three_states, admission_archive):
    """Review probe P1 replayed: B's embedded closure replaced, in both sidecars that
    carry it, by C's real run closure relabelled with B's binding id (F1)."""
    root, a, b, c = three_states
    d = _hop(admission_archive, root, c["state"], "d", register=False)
    fake = dict(stage._closure_proof(root, c["binding"]), binding_id=b["binding"])
    fake_close = stage._digest(fake)
    path_c = stage._chain_path(root, stage._digest(_record_for(root, c["state"], a["local"])))
    chain_c = json.loads(path_c.read_text())
    for hops in chain_c["chains"].values():
        for hop in hops:
            if hop["binding_id"] == b["binding"]:
                hop["closure"], hop["close_sha256"] = fake, fake_close
    chain_c["grantor"]["close_sha256"] = fake_close
    _rewrite(path_c, chain_c)
    path_d = stage._chain_path(root, stage._digest(_record_for(root, d["state"], a["local"])))
    chain_d = json.loads(path_d.read_text())
    hops = chain_d["chains"][a["local"]]
    hops[1]["closure"], hops[1]["close_sha256"] = fake, fake_close
    _rewrite(path_d, chain_d)
    e_state = _copy_candidate(admission_archive, root, d["state"], "e")
    with stage.AdmissionController(root) as controller:
        _refused(root, (a["state"], b["state"], c["state"], d["state"], e_state),
                 lambda: controller.delegate("e"), "^runtime_admission_delegation_chain$")


@pytest.fixture
def two_owner_five_targets(origin, admission_archive):
    """C holds 5 delegated path targets: 1 owned by A, 4 owned by B (review probe P2)."""
    root, a = origin
    b_state = _copy_candidate(admission_archive, root, a["state"], "b")
    assert _delegate_cli(root, "b").returncode == 0
    _run(root, "b", [_local(b_state, "b%d" % index) for index in range(4)])
    c_state = _copy_candidate(admission_archive, root, b_state, "c")
    assert _delegate_cli(root, "c").returncode == 0
    assert len(_policy(c_state)["delegated_targets"]) == 5
    return root, c_state


def _counting(monkeypatch):
    calls, original = [], stage._binding_snapshot

    def counting(root, name, **kwargs):
        calls.append(name)
        return original(root, name, **kwargs)

    monkeypatch.setattr(stage, "_binding_snapshot", counting)
    return calls


def test_owner_snapshots_are_taken_once_per_owner_per_validation_pass(two_owner_five_targets, monkeypatch):
    """F2: one full owner snapshot per owner per pass, not one per delegated target."""
    root, c_state = two_owner_five_targets
    binding = stage._binding_snapshot(root, "c")
    calls = _counting(monkeypatch)
    stage.validate_target_ownership(root, binding)
    assert sorted(calls) == ["b", "old"]
    calls.clear()
    stage._target_ownership_validation(root, binding, allow_new_local=True, capture=True, artifacts={})
    assert sorted(calls) == ["b", "old"]


def test_probe_with_0b_code_snapshots_each_owner_once(two_owner_five_targets, monkeypatch, tmp_path):
    """F2 in the probe: a 0b candidate's own validation (this tree's code) also caches."""
    root, c_state = two_owner_five_targets
    counts = tmp_path / "probe-snapshots.json"
    source = stage._PROBE_SOURCE
    head = "module_path, root, slot = sys.argv[1:4]"
    assert source.count(head) == 1 and source.count("spec.loader.exec_module(module)") == 1
    source = source.replace(head, head + "\nmodule_path = " + repr(str(Path(stage.__file__))))
    source = source.replace("spec.loader.exec_module(module)", "\n".join([
        "spec.loader.exec_module(module)",
        "_seen, _original = [], module._binding_snapshot",
        "def _counted(root, name, **kwargs):",
        "    _seen.append(name)",
        "    open(" + repr(str(counts)) + ", 'w').write(json.dumps(_seen))",
        "    return _original(root, name, **kwargs)",
        "module._binding_snapshot = _counted"]))
    monkeypatch.setattr(stage, "_PROBE_SOURCE", source)
    binding = stage._binding_snapshot(root, "c")
    stage._probe_candidate(root, "c", binding, timeout=60)
    assert sorted(json.loads(counts.read_text())) == ["b", "c", "old"]


def test_via_record_must_carry_the_identical_target(three_states, admission_archive):
    """R2: a forged via record (same id, different descriptor) cannot prove a hop."""
    root, a, b, c = three_states
    d = _hop(admission_archive, root, c["state"], "d", register=False)
    record = _record_for(root, d["state"], a["local"])
    chain = stage._read_chain(root, record)
    hops = chain["chains"][a["local"]]
    via = stage._read_delegation_record(root, hops[2]["via"])
    via_chain = stage._read_chain(root, via)
    forged = json.loads(json.dumps(via))
    forged["targets"][a["local"]]["descriptor"]["root"] += "-elsewhere"
    capability = stage._digest(forged)
    forged_chain = dict(via_chain, record_sha256=capability)
    stage._atomic_record(root / "admission" / "delegations" / (capability + ".json"), forged, immutable=True)
    stage._atomic_record(stage._chain_path(root, capability), forged_chain, immutable=True)
    hops[2]["via"] = capability
    sources = _policy(d["state"])["source_bindings"]
    with stage.AdmissionController(root):
        with pytest.raises(ValueError, match="^runtime_admission_delegation_chain$"):
            stage._delegation_candidate(root, record, chain, record["targets"][a["local"]], sources)


def test_via_sidecar_without_the_target_is_a_coded_refusal(three_states, monkeypatch):
    """F7: a via sidecar lacking the target is runtime_admission_delegation_chain, not KeyError."""
    root, a, b, c = three_states
    record = _record_for(root, c["state"], a["local"])
    chain = stage._read_chain(root, record)
    original = stage._read_chain

    def without_target(actual_root, value):
        result = original(actual_root, value)
        if stage._digest(value) != stage._digest(record):
            result = dict(result, chains={})
        return result

    monkeypatch.setattr(stage, "_read_chain", without_target)
    with stage.AdmissionController(root):
        with pytest.raises(ValueError, match="^runtime_admission_delegation_chain$"):
            stage._verify_chain_hops(root, record["targets"][a["local"]], chain["chains"][a["local"]])


def _live_ownership(root, state):
    binding = stage._binding_snapshot(root, state.parent.name)
    artifacts = {}
    ownership = stage._target_ownership_validation(root, binding, allow_new_local=True, capture=True,
                                                   artifacts=artifacts)
    return binding, ownership, artifacts


@pytest.mark.parametrize(("mutation", "match"), [
    ("via_sidecar", "^runtime_admission_delegation_changed$"),
    ("via_record", "^runtime_admission_delegation_changed$"),  # Global record digest scan.
    ("via_record_deleted", "^runtime_admission_delegation_chain$"),  # Only the via re-read sees it.
])
def test_live_recheck_covers_via_records_and_sidecars(three_states, admission_archive, mutation, match):
    """R3: the live (child-side) re-check of a multi-hop selection re-reads every via."""
    root, a, b, c = three_states
    binding, ownership, artifacts = _live_ownership(root, c["state"])
    stage._validate_ownership_metadata(root, binding, ownership, artifacts)  # Unchanged: accepted.
    [capability] = ownership["delegations"][a["local"]]["vias"]
    if mutation == "via_sidecar":
        path = stage._chain_path(root, capability)
        chain = json.loads(path.read_text())
        chain["chains"][a["local"]][0]["closure"]["child_pid"] += 1
        _rewrite(path, chain)
    elif mutation == "via_record":
        path = root / "admission" / "delegations" / (capability + ".json")
        _rewrite(path, dict(json.loads(path.read_text()), schema_version=1, source_close_sha256="0" * 64))
    else:
        (root / "admission" / "delegations" / (capability + ".json")).unlink()
    with pytest.raises(ValueError, match=match):
        stage._validate_ownership_metadata(root, binding, ownership, artifacts)


def test_owner_limit_boundary_is_inclusive(three_states, admission_archive, monkeypatch):
    """R4: exactly _DELEGATION_OWNER_LIMIT owners passes (D needs A, B and C)."""
    root, a, b, c = three_states
    _copy_candidate(admission_archive, root, c["state"], "d")
    monkeypatch.setattr(stage, "_DELEGATION_OWNER_LIMIT", 3)
    with stage.AdmissionController(root) as controller:
        assert len(controller.delegate("d")["capabilities"]) == 3


@pytest.mark.parametrize("hops", ["one_hop", "two_hops"])
def test_owner_hop_must_name_the_owner_state(three_states, admission_archive, hops):
    """R6: the owner hop's state root and identity must be its binding's. On a one-hop
    chain ([B] for B's own target) nothing else compares them; on [A, B] the prefix
    held by B's own grant also differs."""
    root, a, b, c = three_states
    other = _copy_candidate(admission_archive, root, a["state"], "other")
    identity = b["local"] if hops == "one_hop" else a["local"]
    record = _record_for(root, c["state"], identity)
    path = stage._chain_path(root, stage._digest(record))
    chain = json.loads(path.read_text())
    assert len(chain["chains"][identity]) == (1 if hops == "one_hop" else 2)
    chain["chains"][identity][0].update(state_root=str(other), state_identity=stage._identity(other))
    if hops == "one_hop":  # The owner hop is also the grantor; keep the sidecar self-consistent.
        chain["grantor"] = {key: chain["chains"][identity][0][key] for key in ("binding_id", "close_sha256")}
    _rewrite(path, chain)
    d_state = _copy_candidate(admission_archive, root, c["state"], "d")
    with stage.AdmissionController(root) as controller:
        _refused(root, (a["state"], b["state"], c["state"], d_state), lambda: controller.delegate("d"),
                 "^runtime_admission_delegation_chain$")


def test_sidecar_destination_must_match_its_record(three_states, admission_archive):
    """R8: a sidecar naming another destination path does not prove the record."""
    root, a, b, c = three_states
    record = _record_for(root, c["state"], a["local"])
    path = stage._chain_path(root, stage._digest(record))
    _rewrite(path, dict(json.loads(path.read_text()), destination_state=str(b["state"])))
    d_state = _copy_candidate(admission_archive, root, c["state"], "d")
    with stage.AdmissionController(root) as controller:
        _refused(root, (a["state"], b["state"], c["state"], d_state), lambda: controller.delegate("d"),
                 "^runtime_admission_delegation_chain$")


def test_computed_records_are_validated_before_the_first_write(three_states, admission_archive, monkeypatch):
    """R7: a held selection that no longer proves its chain is refused before writing.

    The grantor's selection is altered after its own verification (its chain for A's
    target loses hop B). Only the pre-write validation of the computed records can
    see it; without it the records would be written and the post-write check would
    refuse too late, leaving residue.
    """
    root, a, b, c = three_states
    d_state = _copy_candidate(admission_archive, root, c["state"], "d")
    original = stage._target_ownership_validation

    def truncated(actual_root, binding, **kwargs):
        result = original(actual_root, binding, **kwargs)
        if binding["state_root"] == str(c["state"]) and a["local"] in result["delegations"]:
            selected = dict(result["delegations"][a["local"]])
            chain = json.loads(json.dumps(selected["chain"]))
            hops = chain["chains"][a["local"]]
            chain["chains"][a["local"]] = [hops[0], hops[-1]] if len(hops) > 2 else hops[:1]
            chain["grantor"] = {key: chain["chains"][a["local"]][-1][key]
                                for key in ("binding_id", "close_sha256")}
            selected["chain"] = chain
            result["delegations"] = dict(result["delegations"], **{a["local"]: selected})
        return result

    monkeypatch.setattr(stage, "_target_ownership_validation", truncated)
    with stage.AdmissionController(root) as controller:
        _refused(root, (a["state"], b["state"], c["state"], d_state), lambda: controller.delegate("d"),
                 "^runtime_admission_delegation_chain$")


_EVENTS = "CREATE TABLE events_policy_v1 (" + ", ".join(
    item.split(":")[0] + " " + item.split(":")[1] for item in stage._TABLE_COLUMNS["events_policy_v1"].split()) + ")"


def _event_row(identity, text_expires_at):
    sources = json.dumps([{"capture_id": "capture", "conversation_id": "U", "captured_at": 1.0e9,
                           "expires_at": 2.0e9}])
    return [identity, 1.0e9, "state_change", "light.room", "on", "u1", None, None, None,
            0, 0, 0, 0, "night", sources, "{}", 3.0e9, text_expires_at]


@pytest.fixture
def event_owner(admission_archive, tmp_path):
    """A owns an events target with one fenced row (text deadline shortened below its
    sources' ceiling, part of A's protected projection) and one unfenced row."""
    def statements(state):
        events = str(state / "data" / "events.db")
        target = _registration("events", 1, {"path": events, "table": "events_policy_v1"})
        statements.target = target[0]
        insert = "INSERT INTO events_policy_v1 VALUES(" + ",".join("?" * 18) + ")"
        return [["data/privacy.db", _TARGETS_DDL, []],
                ["data/privacy.db", _INSERT, target],
                [events, _EVENTS, []],
                [events, insert, _event_row(1, 1.5e9)],
                [events, insert, _event_row(2, None)]]

    root, state = _closed_head(admission_archive, tmp_path, statements)
    a = {"state": state, "binding": stage.read_admission_head(root)["binding_id"], "events": statements.target}
    b = _hop(admission_archive, root, state, "b", register=False)
    c_state = _copy_candidate(admission_archive, root, b["state"], "c")
    assert _delegate_cli(root, "c").returncode == 0
    return root, a, b, c_state


def _erase_in_run(root, name, database, row):
    statement = [str(database), "DELETE FROM events_policy_v1 WHERE id=?", [row]]
    return stage.run_slot(root, name, _command(_CLOSING_CHILD, json.dumps([statement])), timeout=30)


def test_ordinary_erasure_into_an_owner_keeps_the_chain(event_owner, admission_archive):
    """F8: deleting an unfenced (ordinary payload) event row in the owner during the
    head's run closes normally, and the next release still delegates."""
    root, a, b, c_state = event_owner
    _erase_in_run(root, "c", a["state"] / "data" / "events.db", 2)
    assert stage.read_admission_head(root)["status"] == "closed"
    d = _copy_candidate(admission_archive, root, c_state, "d")
    assert _delegate_cli(root, "d").returncode == 0
    assert a["events"] in _policy(d)["delegated_targets"]


@pytest.mark.xfail(strict=True, reason="U3b-0b F8 (inherited from 0a): an erasure that changes an owner's "
                   "protected projection stales the owner during the run; the close then fails and the "
                   "head is left UNKNOWN. Needs an owner re-binding design; not a local fix.")
def test_protected_erasure_into_an_owner_does_not_wedge_the_head(event_owner):
    root, a, b, c_state = event_owner
    try:
        _erase_in_run(root, "c", a["state"] / "data" / "events.db", 1)
    except Exception:  # noqa: BLE001 - the outcome under test is the head status, not the error.
        pass
    assert stage.read_admission_head(root)["status"] == "closed"


def test_failure_after_the_policy_write_is_completed_by_rerunning_delegate(three_states, admission_archive,
                                                                          monkeypatch):
    """F6 / review P3: the policy is written but the grant's manual head and head CAS are
    not. Documented recovery: run 'delegate' again; it rewrites identical records and
    completes the head CAS."""
    root, a, b, c = three_states
    d_state = _copy_candidate(admission_archive, root, c["state"], "d")
    head = stage.read_admission_head(root)
    original = stage._prepare_manual_head

    def failing(*args, **kwargs):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(stage, "_prepare_manual_head", failing)
    with stage.AdmissionController(root) as controller:
        with pytest.raises(OSError):
            controller.delegate("d")
    monkeypatch.setattr(stage, "_prepare_manual_head", original)
    assert (d_state / ".kza-target-ownership.json").exists()
    assert stage.read_admission_head(root) == head
    result = _delegate_cli(root, "d")
    assert result.returncode == 0, result.stderr
    assert stage.read_admission_head(root)["generation"] == head["generation"] + 1
    assert stage.activate_slot(root, "d") == "slots/c/runtime"


def test_residue_then_a_grantor_rerun_needs_a_fresh_slot(three_states, admission_archive, monkeypatch):
    """F6 / review P4: residue from a failed grant plus a rerun of the grantor makes the
    same destination a permanent conflict. Documented recovery: prepare a fresh slot."""
    root, a, b, c = three_states
    d_state = _copy_candidate(admission_archive, root, c["state"], "d")
    original, written = stage._atomic_record, []

    def failing(path, value, *, immutable=False):
        if path.parent.name == "delegations" and len([p for p in written if p.parent.name == "delegations"]) == 1:
            raise OSError(28, "No space left on device")
        original(path, value, immutable=immutable)
        written.append(path)

    monkeypatch.setattr(stage, "_atomic_record", failing)
    with stage.AdmissionController(root) as controller:
        with pytest.raises(OSError):
            controller.delegate("d")
    monkeypatch.setattr(stage, "_atomic_record", original)
    _run(root, "c")  # Same state, new closure proof: every recomputed sidecar differs.
    with stage.AdmissionController(root) as controller:
        _refused(root, (a["state"], b["state"], c["state"], d_state), lambda: controller.delegate("d"),
                 "^runtime_admission_delegation_conflict$")
    _copy_candidate(admission_archive, root, c["state"], "e")
    result = _delegate_cli(root, "e")
    assert result.returncode == 0, result.stderr
    assert stage.activate_slot(root, "e") == "slots/c/runtime"
