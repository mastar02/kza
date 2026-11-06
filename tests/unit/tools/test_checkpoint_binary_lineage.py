"""Inert checkpoint parsing and provenance joins, without application DTOs.

Fixtures encode the published JSON grammar directly. They grant no cleanup,
SDK retirement, serving, or authority and never construct DataCodec.
"""
import builtins
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import pytest

from tests.mocks import checkpoint_binary_lineage as lineage


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def dictionary(**values):
    return {"k": "dict", "v": [[key, value] for key, value in values.items()]}


def sequence(*values):
    return {"k": "list", "v": list(values)}


def source(conversation, *, dependencies=False):
    value = {"capture_id": str(uuid4()), "conversation_id": conversation,
        "captured_at": 1_800_000_000.0, "expires_at": 1_800_172_800.0,
        "origin": "synthetic-parser-fixture"}
    if dependencies:
        value["fact_dependencies"] = []
    return value


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path.resolve()
    blobs = root / "reconciliation-blobs"
    blobs.mkdir()
    subject, foreign = str(uuid4()), str(uuid4())
    return {"root": root, "blobs": blobs, "S": source(subject), "U": source(foreign),
            "scope": {"withdrawn_conversation_id": subject, "preserved_conversation_ids": (foreign,)}}


def binary(corpus, raw=b"synthetic binary value"):
    fingerprint = digest(raw)
    path = corpus["blobs"] / (fingerprint + ".bin")
    if path.exists():
        assert path.read_bytes() == raw
    else:
        path.write_bytes(raw)
    return {"k": "bytes", "sha256": fingerprint, "size": len(raw)}


def collector(ref, sources, *, identity=None, target=None):
    identity = identity or str(uuid4())
    row = dictionary(id=identity, target_id=target or "f" * 64,
        sources_json=json.dumps(sources, sort_keys=True), bucket="stt_correction", captured_at=1_800_000_000.0,
        expires_at=1_800_172_800.0)
    return dictionary(collectors=dictionary(rows=sequence(row), files=dictionary(**{identity: ref})))


def manifest_of(wire):
    found = {}

    def visit(value):
        if type(value) is dict:
            if value.get("k") == "bytes":
                assert value["sha256"] not in found or found[value["sha256"]] == value["size"]
                found[value["sha256"]] = value["size"]
            for child in value.values():
                visit(child)
        elif type(value) is list:
            for child in value:
                visit(child)

    visit(wire)
    return found


def checkpoint(corpus, payload, *, branch="O", links=None, commit=None, observed=True):
    """Wrap ordinary fixtures explicitly; namespace-negative tests opt out."""
    modes = {"O": "seed-O", "N": "advance-N", "R": "rollback-R", "F": "forward-F"}
    if observed:
        payload = dictionary(**{branch + "_observed": payload})
    if links:
        payload = deepcopy(payload)
        payload["v"].extend([[key, value] for key, value in links.items()])
    pins = {"unit": "T024-F1/material-code-data-reconciliation", "mode": modes[branch],
        "source_commit": commit or ("1" * 40 if branch in {"O", "R"} else "2" * 40),
        "runtime_root": str(corpus["root"] / "runtime" / branch),
        "state_root": str(corpus["root"] / "state" / branch),
        "release_sha256": "3" * 64, "helper_manifest_sha256": "4" * 64}
    document = dict(pins, schema_version=1, payload=payload, blob_manifest=manifest_of(payload))
    path = corpus["root"] / (branch + "-state.json")
    assert not path.exists()
    path.write_text(json.dumps(document, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    return {"path": path, "expected_sha256": digest(path.read_bytes()), "expected_pins": pins}


def analyze(corpus, spec, *, anchors=None):
    return lineage.analyze_checkpoint(**spec, blob_root=corpus["blobs"],
        source_anchors=anchors if anchors is not None else (corpus["S"], corpus["U"]), **corpus["scope"])


def rewrite(spec, change):
    document = json.loads(spec["path"].read_text())
    change(document)
    spec["path"].write_text(json.dumps(document, sort_keys=True, allow_nan=False) + "\n")
    # Adversarial fixtures may provide a new external pin only to reach a
    # deeper validator; this helper never upgrades an observed runtime grant.
    return dict(spec, expected_sha256=digest(spec["path"].read_bytes()))


def test_wire_parser_builds_only_inert_values_and_preserves_encoded_positions(corpus, monkeypatch):
    ref = binary(corpus)
    fields = dictionary(**corpus["S"])
    wire = dictionary(source={"k": "dto", "n": "MemorySource", "v": fields},
        binary=ref, label={"k": "path", "v": "/unopened/historical/path"},
        pair={"k": "tuple", "v": [True, None]})
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert not (name == "src" or name.startswith("src.") or name.endswith("compiled_restore_child"))
        return real_import(name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", guarded)
        root, tags, refs = lineage.normalize_wire(wire)
    assert type(root) is dict and type(root["source"]) is dict
    assert root["source"] == corpus["S"] and "fact_dependencies" not in root["source"]
    assert tags[id(root["source"])] == ("MemorySource", True)
    assert tags["positions"][id(root)] == ""
    assert type(root["binary"]) is lineage.BlobRef and refs == [root["binary"]]
    assert refs[0].tag_sha256 == digest(canonical(ref))
    assert root["label"] == "/unopened/historical/path" and root["pair"] == (True, None)


@pytest.mark.parametrize("wire", [
    {"k": "unsupported", "v": []}, {"k": "dict", "v": [["a", 1], ["a", 2]]},
    {"k": "bytes", "sha256": "a" * 64, "size": True},
    {"k": "bytes", "sha256": "A" * 64, "size": 1},
    {"k": "bytes", "sha256": "a" * 64, "size": 1, "extra": None},
    {"k": "list", "v": {}},
], ids=["unsupported-tag", "duplicate-key", "bool-size", "uppercase-hash", "extra-field", "wrong-container"])
def test_malformed_wire_is_rejected_instead_of_dropping_references(wire):
    with pytest.raises(ValueError, match="^binary_lineage_"):
        lineage.normalize_wire(wire)


@pytest.mark.parametrize("which,expected", [("S", "S_only"), ("U", "retained"), ("mixed", "shared")])
def test_exact_collector_sources_classify_without_cleanup_permission(corpus, which, expected):
    ref = binary(corpus)
    sources = [corpus["S"], corpus["U"]] if which == "mixed" else [corpus[which]]
    spec = checkpoint(corpus, collector(ref, sources))
    before = {p.name: p.read_bytes() for p in corpus["blobs"].iterdir()}
    report = analyze(corpus, spec)
    assert report["blobs"][ref["sha256"]]["classification"] == expected
    assert report["references"][0]["sources"] == sources
    assert report["complete_reference_scope"] is False
    assert report["deletion_authorized"] is False and report["cleanup_ack"] is False
    assert {p.name: p.read_bytes() for p in corpus["blobs"].iterdir()} == before


@pytest.mark.parametrize("family", ["captures", "collectors", "voices"])
def test_typed_locations_pin_exact_encoded_tag_row_and_source(corpus, family):
    ref = binary(corpus)
    identity = str(uuid4())
    if family == "captures":
        row = dictionary(id=identity, capture_id=corpus["S"]["capture_id"],
            conversation_id=corpus["S"]["conversation_id"])
        payload = dictionary(captures=dictionary(audio=ref, rows=sequence(row)))
    elif family == "voices":
        row = dictionary(id=identity, embedding=ref, sources_json=json.dumps([corpus["S"]]))
        payload = dictionary(voices=dictionary(rows=sequence(row)))
    else:
        payload = collector(ref, [corpus["S"]], identity=identity)
        row = payload["v"][0][1]["v"][0][1]["v"][0]
    report = analyze(corpus, checkpoint(corpus, payload))
    record, = report["references"]
    assert record["family"] == family and record["categories"] == ["S"]
    assert record["row_id"] == identity and record["reason"] == "typed_join"
    assert record["tag_sha256"] == digest(canonical(ref))
    assert record["row_sha256"] == digest(canonical(row))
    document = json.loads(Path(record["checkpoint"]).read_text())

    def at(pointer):
        current = document
        for part in pointer.split("/")[1:]:
            part = part.replace("~1", "/").replace("~0", "~")
            current = current[int(part)] if type(current) is list else current[part]
        return current

    assert at(record["encoded_pointer"]) == ref and at(record["row_pointer"]) == row
    assert record["physical"]["path"] == str(corpus["blobs"] / (ref["sha256"] + ".bin"))


def test_optional_dependencies_are_preserved_without_adding_defaults(corpus):
    ref = binary(corpus)
    six = dict(corpus["S"], fact_dependencies=[])
    spec = checkpoint(corpus, collector(ref, [six]))
    assert analyze(corpus, spec)["blobs"][ref["sha256"]]["classification"] == "blocked"
    report = analyze(corpus, spec, anchors=(six, corpus["U"]))
    assert report["blobs"][ref["sha256"]]["classification"] == "S_only"
    assert report["references"][0]["sources"] == [six]
    wire = {"k": "dto", "n": "MemorySource", "v": dictionary(**{**corpus["S"], "fact_dependencies": sequence()})}
    normalized, tags, _ = lineage.normalize_wire(wire)
    assert tags[id(normalized)] == ("MemorySource", True)
    assert normalized == six and "fact_dependencies" not in corpus["S"]


@pytest.mark.parametrize("problem", ["missing-row", "duplicate-row", "unanchored-capture", "changed-source", "UUID-target"])
def test_missing_or_ambiguous_collector_join_is_blocked(corpus, problem):
    ref = binary(corpus)
    payload = collector(ref, [corpus["S"]])
    rows = payload["v"][0][1]["v"][0][1]["v"]
    row = rows[0]
    if problem == "missing-row":
        rows.clear()
    elif problem == "duplicate-row":
        rows.append(deepcopy(row))
    else:
        keyed = dict(row["v"])
        if problem == "UUID-target":
            keyed["target_id"] = str(uuid4())
        else:
            changed = dict(corpus["S"])
            if problem == "unanchored-capture":
                changed["capture_id"] = str(uuid4())
            else:
                changed["expires_at"] += 1
            keyed["sources_json"] = json.dumps([changed])
        row["v"] = [[key, value] for key, value in keyed.items()]
    report = analyze(corpus, checkpoint(corpus, payload))
    assert report["blobs"][ref["sha256"]]["classification"] == "blocked"
    assert report["references"][0]["categories"] == ["unknown"]


@pytest.mark.parametrize("changed_field", ["conversation_id", "expires_at"])
def test_conflicting_anchors_for_one_capture_are_rejected(corpus, changed_field):
    ref = binary(corpus)
    spec = checkpoint(corpus, collector(ref, [corpus["S"]]))
    changed = dict(corpus["S"])
    changed[changed_field] = str(uuid4()) if changed_field == "conversation_id" else changed[changed_field] + 1
    with pytest.raises(ValueError, match="^binary_lineage_source_anchor_conflict$"):
        analyze(corpus, spec, anchors=(corpus["S"], changed, corpus["U"]))


def opaque(wire):
    return {"k": "dto", "n": "UnknownLineageRecord", "v": wire}


def manual(rows):
    # DataCodec's exact ManualSentinels field grammar, not a live actor/DTO.
    fields = {key: None for key in ("profile_id", "profile_revision", "link_revision", "user_id", "user",
        "fact_id", "group_request", "actor_subject", "voice_ids", "suggestion_ids", "config_consent",
        "deleted_profile_id", "home_token")}
    fields["rows"] = rows
    return {"k": "dto", "n": "ManualSentinels", "v": dictionary(**fields)}


@pytest.mark.parametrize("where", ["collector-container", "collector-row", "capture-row", "voice-row", "manual-nested"])
def test_unknown_DTO_ancestry_blocks_remote_row_joins_and_manual_overrides(corpus, where):
    ref = binary(corpus)
    if where.startswith("collector"):
        payload = collector(ref, [corpus["S"]])
        fields = payload["v"][0]
        if where == "collector-container":
            fields[1] = opaque(fields[1])
        else:
            rows = fields[1]["v"][0][1]["v"]
            rows[0] = opaque(rows[0])
    elif where == "capture-row":
        row = dictionary(id=str(uuid4()), capture_id=corpus["S"]["capture_id"],
            conversation_id=corpus["S"]["conversation_id"])
        payload = dictionary(captures=dictionary(audio=ref, rows=sequence(opaque(row))))
    elif where == "voice-row":
        row = dictionary(id=str(uuid4()), sources_json=json.dumps([corpus["S"]]), embedding=ref)
        payload = dictionary(voices=dictionary(rows=sequence(opaque(row))))
    else:
        payload = dictionary(manual=manual(dictionary(unknown=opaque(dictionary(binary=ref)))))
    report = analyze(corpus, checkpoint(corpus, payload))
    assert report["blobs"][ref["sha256"]]["classification"] == "blocked"
    assert all(item["categories"] == ["unknown"] for item in report["references"])


@pytest.mark.parametrize("unknown", [False, True])
def test_hash_union_follows_each_independent_source_join(corpus, unknown):
    ref = binary(corpus)
    second = collector(ref, [corpus["U"]])
    if unknown:
        rows = second["v"][0][1]["v"][0][1]["v"]
        rows[0] = opaque(rows[0])
    payload = dictionary(N_observed=collector(ref, [corpus["S"]]), N_only_observed=second)
    links = {"O_source_sha256": "a" * 64}
    report = analyze(corpus, checkpoint(corpus, payload, branch="N", observed=False, links=links))
    assert report["parent_links"] == links
    result = report["blobs"][ref["sha256"]]
    assert result["classification"] == ("blocked" if unknown else "shared")
    assert result["reference_count"] == len(report["references"]) == 2
    assert len({item["encoded_pointer"] for item in report["references"]}) == 2
    assert any(item["categories"] == ["S"] for item in report["references"])
    assert any(item["categories"] == (["unknown"] if unknown else ["U"]) for item in report["references"])


@pytest.mark.parametrize("also_subject", [False, True])
def test_typed_manual_rows_are_retained_or_shared_without_erasure_grant(corpus, also_subject):
    ref = binary(corpus)
    payload = dictionary(manual=manual(dictionary(retained=ref)))
    if also_subject:
        payload["v"].extend(collector(ref, [corpus["S"]])["v"])
    report = analyze(corpus, checkpoint(corpus, payload))
    assert report["blobs"][ref["sha256"]]["classification"] == ("shared" if also_subject else "retained")
    assert any(item["categories"] == ["manual"] for item in report["references"])
    assert report["cleanup_ack"] is False and report["deletion_authorized"] is False


@pytest.mark.parametrize("problem", ["bare", "arbitrary", "nested-observed", "wrong-mode",
    "unsupported-binary-role", "nested-binary"])
def test_familiar_fields_outside_original_namespaces_are_blocked(corpus, problem):
    ref = binary(corpus)
    familiar = collector(ref, [corpus["S"]])
    if problem == "bare":
        payload = familiar
    elif problem == "arbitrary":
        payload = dictionary(arbitrary=familiar)
    elif problem == "nested-observed":
        payload = dictionary(arbitrary=dictionary(O_observed=familiar))
    elif problem == "wrong-mode":
        payload = dictionary(N_observed=familiar)
    elif problem == "unsupported-binary-role":
        payload = dictionary(binary_observed=dictionary(arbitrary=familiar))
    else:
        payload = dictionary(arbitrary=dictionary(binary_observed=dictionary(S_private=familiar)))
    # No implicit observed wrapper: this is the exact rejected namespace.
    report = analyze(corpus, checkpoint(corpus, payload, observed=False))
    assert report["blobs"][ref["sha256"]]["classification"] == "blocked"
    assert report["references"][0]["reason"] == "unsupported_binary_location"


@pytest.mark.parametrize("branch,alias", [("N", "N_only_observed"), ("F", "F_N_only_observed")])
def test_original_additional_observed_aliases_keep_their_mode(corpus, branch, alias):
    ref = binary(corpus)
    payload = dictionary(**{alias: collector(ref, [corpus["S"]])})
    # Single-checkpoint scope pins syntactic predecessors; four_checkpoints
    # separately binds these links to the actual original checkpoint hashes.
    parents = ("O",) if branch == "N" else ("O", "N", "R")
    links = {key + "_source_sha256": digest(("external fixture " + key).encode()) for key in parents}
    report = analyze(corpus, checkpoint(corpus, payload, branch=branch, observed=False, links=links))
    assert report["parent_links"] == links
    assert report["blobs"][ref["sha256"]]["classification"] == "S_only"
    assert report["references"][0]["logical_pointer"].startswith("/" + alias + "/collectors/")


@pytest.mark.parametrize("family,container", [("collectors", "list"), ("collectors", "tuple"),
    ("collectors", "dto"), ("voices", "list"), ("voices", "tuple"), ("voices", "dto")])
def test_sources_json_requires_original_SQL_text_not_source_containers(corpus, family, container):
    ref = binary(corpus)
    fields = dictionary(**corpus["S"])
    value = ({"k": "dto", "n": "MemorySource", "v": fields} if container == "dto"
        else {"k": container, "v": [fields]})
    if family == "collectors":
        payload = collector(ref, [corpus["S"]])
        row = payload["v"][0][1]["v"][0][1]["v"][0]
        row["v"] = [[key, value if key == "sources_json" else original] for key, original in row["v"]]
    else:
        row = dictionary(id=str(uuid4()), embedding=ref, sources_json=value)
        payload = dictionary(voices=dictionary(rows=sequence(row)))
    report = analyze(corpus, checkpoint(corpus, payload))
    assert report["blobs"][ref["sha256"]]["classification"] == "blocked"
    assert report["references"][0]["categories"] == ["unknown"]


def binary_observation(corpus, ref, sources):
    identity, target = str(uuid4()), "f" * 64
    node = collector(ref, sources, identity=identity, target=target)
    node["v"].extend([["id", identity], ["current_file_locator", dictionary(
        root=str(corpus["root"] / "collector-store"), target_id=target,
        relative="policy_v1/" + identity + ".wav")]])
    return node


@pytest.mark.parametrize("role", ["S_private", "S_shared", "U_shared_manual"])
def test_binary_roles_use_current_locator_and_independent_row_sources(corpus, role):
    ref = binary(corpus)
    source_value = corpus["U"] if role == "U_shared_manual" else corpus["S"]
    payload = dictionary(binary_observed=dictionary(**{role: binary_observation(corpus, ref, [source_value])}))
    report = analyze(corpus, checkpoint(corpus, payload, observed=False))
    assert report["blobs"][ref["sha256"]]["classification"] == ("retained" if role == "U_shared_manual" else "S_only")
    record, = report["references"]
    assert record["sources"] == [source_value] and record["reason"] == "typed_join"
    assert record["logical_pointer"].startswith("/binary_observed/" + role + "/collectors/")


@pytest.mark.parametrize("problem", ["missing", "target", "id", "relative", "root", "unknown"])
def test_binary_locator_must_match_the_actual_row_and_file(corpus, problem):
    ref = binary(corpus)
    node = binary_observation(corpus, ref, [corpus["S"]])
    fields = dict(node["v"])
    locator = dict(fields["current_file_locator"]["v"])
    if problem == "missing":
        fields.pop("current_file_locator")
    elif problem == "id":
        fields["id"] = str(uuid4())
    elif problem == "unknown":
        fields["current_file_locator"] = opaque(fields["current_file_locator"])
    else:
        locator[{"target": "target_id", "relative": "relative", "root": "root"}[problem]] = {
            "target": "e" * 64, "relative": "policy_v1/../borrowed.wav", "root": "relative-store"}[problem]
        fields["current_file_locator"]["v"] = list(map(list, locator.items()))
    node["v"] = list(map(list, fields.items()))
    payload = dictionary(binary_observed=dictionary(S_private=node))
    report = analyze(corpus, checkpoint(corpus, payload, observed=False))
    assert report["blobs"][ref["sha256"]]["classification"] == "blocked"
    assert report["references"][0]["categories"] == ["unknown"]


@pytest.mark.parametrize("problem", ["raw-checkpoint", "control-identity"])
def test_external_checkpoint_and_original_branch_pins_are_not_self_repaired(corpus, problem):
    ref = binary(corpus)
    spec = checkpoint(corpus, collector(ref, [corpus["S"]]))
    if problem == "raw-checkpoint":
        document = json.loads(spec["path"].read_text())
        spec["path"].write_text(json.dumps(document, indent=2) + "\n")
        error = "checkpoint_pin"
    else:
        spec = rewrite(spec, lambda document: document.update(state_root="/different/original-state"))
        error = "control_pin"
    with pytest.raises(ValueError, match="^binary_lineage_" + error + "$"):
        analyze(corpus, spec)


@pytest.mark.parametrize("problem", ["missing-entry", "size-conflict", "unused-entry"])
def test_manifest_and_typed_reference_must_close_exactly(corpus, problem):
    ref = binary(corpus)
    spec = checkpoint(corpus, collector(ref, [corpus["S"]]))

    def changed(document):
        if problem == "missing-entry":
            document["blob_manifest"].clear()
        elif problem == "size-conflict":
            document["blob_manifest"][ref["sha256"]] += 1
        else:
            document["blob_manifest"]["e" * 64] = 0

    spec = rewrite(spec, changed)
    before = {p.name: p.read_bytes() for p in corpus["blobs"].iterdir()}
    with pytest.raises(ValueError, match="^binary_lineage_"):
        analyze(corpus, spec)
    assert {p.name: p.read_bytes() for p in corpus["blobs"].iterdir()} == before


@pytest.mark.parametrize("problem", ["changed-bytes", "missing", "symlink", "hardlink"])
def test_physical_binary_pins_cannot_follow_borrowed_or_changed_files(corpus, problem):
    ref = binary(corpus)
    spec = checkpoint(corpus, collector(ref, [corpus["S"]]))
    path = corpus["blobs"] / (ref["sha256"] + ".bin")
    if problem == "changed-bytes":
        path.write_bytes(b"x" * ref["size"])
    elif problem == "missing":
        path.unlink()
    else:
        saved = corpus["root"] / "borrowed.bin"
        path.rename(saved)
        if problem == "symlink":
            path.symlink_to(saved)
        else:
            path.hardlink_to(saved)
    with pytest.raises((ValueError, OSError)):
        analyze(corpus, spec)


def four_checkpoints(corpus, *, blocked=False, wrong_parent=False, wrong_code=False):
    ref = binary(corpus, b"one physical blob shared by four original branches")
    specs = {}
    for branch in ("O", "N", "R", "F"):
        source_value = corpus["S"] if branch in {"O", "R"} else corpus["U"]
        payload = collector(ref, [source_value])
        if branch == "R" and blocked:
            rows = payload["v"][0][1]["v"][0][1]["v"]
            rows[0] = opaque(rows[0])
        links = {key + "_source_sha256": spec["expected_sha256"] for key, spec in specs.items()}
        if branch == "N" and wrong_parent:
            links["O_source_sha256"] = "a" * 64
        specs[branch] = checkpoint(corpus, payload, branch=branch, links=links,
            commit="5" * 40 if branch == "R" and wrong_code else None)
    return ref, specs


def aggregate(corpus, checkpoints):
    return lineage.analyze_checkpoints(checkpoints, corpus["blobs"],
        source_anchors=(corpus["S"], corpus["U"]), **corpus["scope"])


@pytest.mark.parametrize("blocked", [False, True])
def test_four_branch_union_closes_scope_but_unknown_reference_still_vetoes(corpus, blocked):
    ref, specs = four_checkpoints(corpus, blocked=blocked)
    single = analyze(corpus, specs["O"])
    assert single["blobs"][ref["sha256"]]["classification"] == "S_only"
    assert single["complete_reference_scope"] is False
    report = aggregate(corpus, specs)
    assert report["complete_reference_scope"] is True and set(report["checkpoints"]) == {"O", "N", "R", "F"}
    assert report["blobs"][ref["sha256"]]["classification"] == ("blocked" if blocked else "shared")
    assert report["blobs"][ref["sha256"]]["reference_count"] == 4
    assert report["deletion_authorized"] is False and report["cleanup_ack"] is False and report["capability"] is False


@pytest.mark.parametrize("problem", ["missing-branch", "wrong-parent", "wrong-code"])
def test_four_branch_original_relationships_cannot_be_relabelled(corpus, problem):
    _, specs = four_checkpoints(corpus, wrong_parent=problem == "wrong-parent", wrong_code=problem == "wrong-code")
    if problem == "missing-branch":
        specs.pop("F")
    expected = {"missing-branch": "checkpoint_set", "wrong-parent": "parent_link_pin", "wrong-code": "code_pair"}[problem]
    with pytest.raises(ValueError, match="^binary_lineage_" + expected + "$"):
        aggregate(corpus, specs)


def test_only_global_analysis_rejects_bytes_unreferenced_by_any_branch(corpus):
    ref, specs = four_checkpoints(corpus)
    orphan = binary(corpus, b"globally unreferenced retained bytes")
    single = analyze(corpus, specs["O"])
    assert single["complete_reference_scope"] is False
    assert set(single["blobs"]) == {ref["sha256"]}
    with pytest.raises(ValueError, match="^binary_lineage_physical_inventory_closure$"):
        aggregate(corpus, specs)
    assert (corpus["blobs"] / (orphan["sha256"] + ".bin")).read_bytes() == b"globally unreferenced retained bytes"
