"""Inert DataCodec wire analysis; classifications never authorize cleanup.

Only stdlib objects are created. DTO names remain annotations; no application
class, DataCodec instance, SDK or model is imported or constructed.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from uuid import UUID

SOURCE_FIELDS = {"capture_id", "conversation_id", "captured_at", "expires_at", "origin", "fact_dependencies"}
REQUIRED_SOURCE_FIELDS = SOURCE_FIELDS - {"fact_dependencies"}
PIN_FIELDS = {"unit", "mode", "source_commit", "runtime_root", "state_root", "release_sha256", "helper_manifest_sha256"}
DTO_FIELDS = {
    "MemorySource": SOURCE_FIELDS,
    "Cohort": {"source", "fact", "ambient_id", "archive_path", "clip_ids", "event_ids", "suggestion_ids", "voice_ids", "user", "browser_client"},
    "ManualSentinels": {"profile_id", "profile_revision", "link_revision", "user_id", "user", "fact_id", "group_request", "actor_subject", "voice_ids", "suggestion_ids", "rows", "config_consent", "deleted_profile_id", "home_token"},
    "DataSnapshot": {"data", "root_identities", "diagnostic_files", "index_path", "index_collection", "sdk_version"},
    "ReplayWitness": {"root", "schemas", "root_identities", "index_path", "index_collection", "sdk_version", "index_metadata", "index_lineage"},
    "ErasureStep": {"operation_id", "conversation_id", "participant", "target_id", "version", "descriptor", "attempts", "claim_token", "cursor"},
}


@dataclass(frozen=True)
class BlobRef:
    """An analysis node, never an application DTO or live binary handle."""

    sha256: str
    size: int
    pointer: str
    tag_sha256: str


def _require(value, code):
    if not value:
        raise ValueError("binary_lineage_" + code)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _hash(value):
    return type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None


def _pairs(pairs):
    value = {}
    for key, item in pairs:
        _require(key not in value, "duplicate_JSON_key")
        value[key] = item
    return value


def _load(raw):
    def reject(_value):
        raise ValueError("binary_lineage_nonfinite")
    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=reject)


def _escape(value):
    return str(value).replace("~", "~0").replace("/", "~1")


def normalize_wire(value):
    """Return builtins, DTO annotations and every encoded binary occurrence."""
    tags, references, budget = {"positions": {}}, [], [0]

    def decode(node, pointer, depth):
        budget[0] += 1
        _require(depth <= 100 and budget[0] <= 1_000_000, "wire_budget")
        if node is None or type(node) in (str, bool, int):
            return node
        if type(node) is float:
            _require(math.isfinite(node), "nonfinite")
            return node
        _require(type(node) is dict and type(node.get("k")) is str, "wire_tag")
        kind = node["k"]
        if kind == "bytes":
            _require(set(node) == {"k", "sha256", "size"} and _hash(node["sha256"])
                and type(node["size"]) is int and 0 <= node["size"] <= 16*1024*1024, "bytes_tag")
            ref = BlobRef(node["sha256"], node["size"], pointer, _sha(_json(node)))
            references.append(ref)
            return ref
        if kind == "dto":
            _require(set(node) == {"k", "n", "v"} and type(node["n"]) is str, "dto_tag")
            fields = decode(node["v"], pointer + "/v", depth + 1)
            _require(type(fields) is dict, "dto_fields")
            valid = set(fields) == DTO_FIELDS.get(node["n"])
            if node["n"] == "MemorySource":
                valid = set(fields) in (SOURCE_FIELDS, REQUIRED_SOURCE_FIELDS)
            tags[id(fields)] = (node["n"], valid)
            return fields
        _require(kind in {"path", "dict", "list", "tuple"} and set(node) == {"k", "v"}, "container_tag")
        if kind == "path":
            _require(type(node["v"]) is str and Path(node["v"]).is_absolute(), "path_tag")
            return node["v"]
        _require(type(node["v"]) is list and len(node["v"]) <= 100000, "container_size")
        if kind in {"list", "tuple"}:
            result = [decode(item, pointer + "/v/" + str(i), depth + 1) for i, item in enumerate(node["v"])]
            result = tuple(result) if kind == "tuple" else result
            tags["positions"][id(result)] = pointer
            return result
        result = {}
        for i, pair in enumerate(node["v"]):
            _require(type(pair) is list and len(pair) == 2 and type(pair[0]) is str
                and pair[0] not in result, "dict_pair")
            result[pair[0]] = decode(pair[1], pointer + "/v/" + str(i) + "/1", depth + 1)
        tags["positions"][id(result)] = pointer
        return result

    return decode(value, "", 0), tags, references


def _read(path, limit):
    _require(isinstance(path, Path) and path.is_absolute() and path.resolve(strict=True) == path, "canonical_path")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        _require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_size <= limit, "physical_file")
        raw = stream.read(limit + 1)
    after = path.stat()
    _require(not path.is_symlink() and path.resolve(strict=True) == path and len(raw) == before.st_size
        and (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) and after.st_nlink == 1, "file_changed")
    return raw, {"device": after.st_dev, "inode": after.st_ino}


def _source(source):
    if type(source) is not dict or set(source) not in (SOURCE_FIELDS, REQUIRED_SOURCE_FIELDS):
        return False
    try:
        if any(type(source[k]) is not str or str(UUID(source[k])) != source[k]
               for k in ("capture_id", "conversation_id")):
            return False
        if any(type(source[k]) not in (int, float) or not math.isfinite(source[k])
               for k in ("captured_at", "expires_at")):
            return False
        return (source["expires_at"] > source["captured_at"] and type(source["origin"]) is str
            and bool(source["origin"]) and ("fact_dependencies" not in source
                or type(source["fact_dependencies"]) is list
                and all(type(v) is str for v in source["fact_dependencies"])))
    except (ValueError, TypeError):
        return False


def _references(root, tags, blobs, anchors, subject, preserved, wire, mode):
    observed_paths = {"seed-O": {"/O_observed"},
        "advance-N": {"/N_observed", "/N_only_observed"},
        "rollback-R": {"/R_observed"},
        "forward-F": {"/F_observed", "/F_N_only_observed"}}[mode]
    binary_paths = {"/binary_observed/" + role for role in ("S_private", "S_shared", "U_shared_manual")}
    known = {}
    for source in anchors:
        _require(_source(source), "source_anchor")
        key = source["capture_id"]
        _require(key not in known or _json(known[key]) == _json(source), "source_anchor_conflict")
        known[key] = source
    uncertain = set()

    def uncertainty(node, inherited=False):
        dto = tags.get(id(node))
        inherited = inherited or dto is not None and not dto[1]
        if type(node) in (dict, tuple, list, BlobRef):
            if inherited:
                uncertain.add(id(node))
            values = node.values() if type(node) is dict else node if type(node) in (tuple, list) else ()
            for value in values:
                uncertainty(value, inherited)

    uncertainty(root)
    records = {ref.pointer: {"encoded_pointer": "/payload" + ref.pointer,
        "tag_sha256": ref.tag_sha256, "sha256": ref.sha256, "size": ref.size,
        "categories": ["unknown"], "reason": "unsupported_binary_location"} for ref in blobs}

    def leaves(node):
        if type(node) is BlobRef:
            yield node
        elif type(node) is dict:
            for value in node.values():
                yield from leaves(value)
        elif type(node) in (tuple, list):
            for value in node:
                yield from leaves(value)

    def sources(value):
        try:
            if type(value) is not str:
                return None
            items = _load(value)
            if type(items) not in (list, tuple) or not items:
                return None
            result = []
            for source in items:
                if not _source(source) or source["capture_id"] not in known:
                    return None
                if _json(source) != _json(known[source["capture_id"]]):
                    return None
                result.append(source)
            if len({v["capture_id"] for v in result}) != len(result):
                return None
            return result
        except (ValueError, TypeError):
            return None

    def assign(ref, row, joined, family, path, source_json=None):
        if type(ref) is not BlobRef:
            return
        result = records[ref.pointer]
        if id(ref) in uncertain or id(row) in uncertain:
            result.update(categories=["unknown"], reason="unknown_DTO_in_binary_or_lineage_ancestry")
            return
        categories = set()
        for source in joined or ():
            conversation = source["conversation_id"]
            categories.add("S" if conversation == subject else "U" if conversation in preserved else "unknown")
        if not categories:
            categories.add("unknown")
        if result["reason"] != "unsupported_binary_location":
            categories = {"unknown"}
        pointer = tags["positions"].get(id(row))
        original = wire
        if pointer is not None:
            for part in pointer.split("/")[1:]:
                part = part.replace("~1", "/").replace("~0", "~")
                original = original[int(part)] if type(original) is list else original[part]
        result.update(categories=sorted(categories), reason="typed_join" if "unknown" not in categories else "missing_or_conflicting_source_join",
            family=family, logical_pointer=path, row_id=row.get("id"),
            row_pointer="/payload" + pointer if pointer is not None else None,
            row_sha256=_sha(_json(original)) if pointer is not None else None,
            sources=joined or [], source_json_sha256=_sha(source_json.encode()) if type(source_json) is str else None)

    def visit(node, path="", blocked=False):
        if type(node) is dict:
            dto = tags.get(id(node))
            if dto is not None and not dto[1]:
                blocked = True
            if dto is not None and dto[0] == "ManualSentinels" and dto[1] and not blocked:
                for ref in leaves(node["rows"]):
                    if id(ref) not in uncertain:
                        records[ref.pointer].update(categories=["manual"], reason="typed_ManualSentinels_rows",
                            logical_pointer=path + "/rows", family="manual")
                return
            if not blocked and path in observed_paths | binary_paths:
                capture = node.get("captures")
                if path in observed_paths and type(capture) is dict and type(capture.get("audio")) is BlobRef:
                    rows = capture.get("rows")
                    row = rows[0] if type(rows) in (list, tuple) and len(rows) == 1 and type(rows[0]) is dict else {}
                    source = known.get(row.get("capture_id"))
                    joined = [source] if source and source["conversation_id"] == row.get("conversation_id") else None
                    assign(capture["audio"], row, joined, "captures", path + "/captures/audio")
                collector = node.get("collectors")
                if type(collector) is dict and type(collector.get("files")) is dict:
                    rows = collector.get("rows")
                    for key, ref in collector["files"].items():
                        matches = [r for r in rows if type(r) is dict and r.get("id") == key] if type(rows) in (list, tuple) else []
                        row = matches[0] if len(matches) == 1 else {}
                        locator_ok = True
                        if path in binary_paths:
                            locator = node.get("current_file_locator")
                            locator_ok = (type(locator) is dict and id(locator) not in uncertain
                                and locator.get("target_id") == row.get("target_id")
                                and node.get("id") == key
                                and locator.get("relative") == "policy_v1/" + key + ".wav"
                                and type(locator.get("root")) is str and Path(locator["root"]).is_absolute())
                        joined = sources(row.get("sources_json")) if _hash(row.get("target_id")) and locator_ok else None
                        assign(ref, row, joined, "collectors", path + "/collectors/files/" + _escape(key), row.get("sources_json"))
                voices = node.get("voices")
                if path in observed_paths and type(voices) is dict and type(voices.get("rows")) in (tuple, list):
                    rows = voices["rows"]
                    for index, row in enumerate(rows):
                        if type(row) is dict and type(row.get("embedding")) is BlobRef:
                            unique = type(row.get("id")) is str and sum(type(r) is dict and r.get("id") == row["id"] for r in rows) == 1
                            assign(row["embedding"], row, sources(row.get("sources_json")) if unique else None,
                                "voices", path + "/voices/rows/" + str(index) + "/embedding", row.get("sources_json"))
            for key, value in node.items():
                visit(value, path + "/" + _escape(key), blocked)
        elif type(node) in (tuple, list):
            for index, value in enumerate(node):
                visit(value, path + "/" + str(index), blocked)

    visit(root)
    return list(records.values())


def classify_blobs(references):
    """Aggregate only after every reference has an independent typed join."""
    result = {}
    for ref in references:
        result.setdefault(ref["sha256"], []).append(ref)
    output = {}
    for digest, refs in result.items():
        categories = set().union(*(set(ref["categories"]) for ref in refs))
        conversations = {source["conversation_id"] for ref in refs for source in ref.get("sources", [])}
        classification = ("blocked" if "unknown" in categories else "S_only" if categories == {"S"}
            else "shared" if len(categories) > 1 or len(conversations) > 1 else "retained")
        output[digest] = {"classification": classification, "categories": sorted(categories),
            "reference_count": len(refs), "pointers": [ref["encoded_pointer"] for ref in refs]}
    return output


def analyze_checkpoint(path: Path, blob_root: Path, *, expected_sha256: str, expected_pins: dict,
                       source_anchors: tuple[dict, ...], withdrawn_conversation_id: str,
                       preserved_conversation_ids: tuple[str, ...]) -> dict:
    """Verify one checkpoint; only the four-branch API closes shared inventory."""
    _require(_hash(expected_sha256) and type(expected_pins) is dict and set(expected_pins) == PIN_FIELDS, "external_pins")
    _require(type(source_anchors) is tuple and type(preserved_conversation_ids) is tuple, "source_scope")
    try:
        ids = (withdrawn_conversation_id,) + preserved_conversation_ids
        _require(all(type(v) is str and str(UUID(v)) == v for v in ids)
            and len(set(ids)) == len(ids), "source_scope")
    except (ValueError, TypeError, AttributeError) as error:
        raise ValueError("binary_lineage_source_scope") from error
    raw, identity = _read(path, 64 * 1024 * 1024)
    _require(_sha(raw) == expected_sha256, "checkpoint_pin")
    document = _load(raw)
    _require(type(document) is dict and set(document) == PIN_FIELDS | {"schema_version", "payload", "blob_manifest"}
        and type(document["schema_version"]) is int and document["schema_version"] == 1, "envelope")
    _require({key: document[key] for key in PIN_FIELDS} == expected_pins, "control_pin")
    _require(document["unit"] == "T024-F1/material-code-data-reconciliation"
        and document["mode"] in {"seed-O", "advance-N", "rollback-R", "forward-F"}
        and type(document["source_commit"]) is str and re.fullmatch("[0-9a-f]{40}", document["source_commit"])
        and all(_hash(document[k]) for k in ("release_sha256", "helper_manifest_sha256"))
        and all(type(document[k]) is str and Path(document[k]).is_absolute() for k in ("runtime_root", "state_root")), "control_fields")
    manifest = document["blob_manifest"]
    _require(type(manifest) is dict and all(_hash(k) and type(v) is int
        and 0 <= v <= 16*1024*1024 for k, v in manifest.items()), "manifest")
    root, tags, blobs = normalize_wire(document["payload"])
    _require(type(root) is dict and {ref.sha256 for ref in blobs} == set(manifest), "manifest_reference_closure")
    _require(blob_root.is_absolute() and blob_root.is_dir() and blob_root.resolve(strict=True) == blob_root, "blob_root")
    root_info, physical = blob_root.stat(), {}
    for digest, size in manifest.items():
        data, file_identity = _read(blob_root / (digest + ".bin"), 16*1024*1024)
        _require(len(data) == size and _sha(data) == digest, "blob_pin")
        physical[digest] = {"path": str(blob_root / (digest + ".bin")), "size": size, "identity": file_identity}
    _require(all(ref.size == manifest[ref.sha256] for ref in blobs), "reference_size")
    after = blob_root.stat()
    _require(blob_root.resolve(strict=True) == blob_root
        and (root_info.st_dev, root_info.st_ino) == (after.st_dev, after.st_ino), "blob_root_changed")
    references = _references(root, tags, blobs, source_anchors, withdrawn_conversation_id,
        set(preserved_conversation_ids), document["payload"], document["mode"])
    for ref in references:
        ref.update(checkpoint_sha256=expected_sha256, checkpoint=str(path), mode=document["mode"],
            manifest_sha256=_sha(_json(manifest)), physical=physical[ref["sha256"]])
    link_keys = {"O_source_sha256", "N_source_sha256", "R_source_sha256"}
    links = {key: root[key] for key in link_keys if key in root}
    expected_keys = {"seed-O": set(), "advance-N": {"O_source_sha256"},
        "rollback-R": {"O_source_sha256", "N_source_sha256"}, "forward-F": link_keys}[document["mode"]]
    _require(set(links) == expected_keys and all(_hash(v) for v in links.values()), "parent_link_fields")
    return {"checkpoint": str(path), "checkpoint_sha256": expected_sha256, "checkpoint_identity": identity,
        "pins": expected_pins, "manifest": manifest, "manifest_sha256": _sha(_json(manifest)),
        "references": references, "blobs": classify_blobs(references), "parent_links": links,
        "complete_reference_scope": False, "deletion_authorized": False, "cleanup_ack": False}


def analyze_checkpoints(checkpoints: dict, blob_root: Path, *, source_anchors: tuple[dict, ...],
                        withdrawn_conversation_id: str, preserved_conversation_ids: tuple[str, ...]) -> dict:
    """Close four-branch physical/reference inventory without mutation/grants."""
    _require(type(checkpoints) is dict and set(checkpoints) == {"O", "N", "R", "F"}, "checkpoint_set")
    reports, manifest, references = {}, {}, []
    modes = {"O": "seed-O", "N": "advance-N", "R": "rollback-R", "F": "forward-F"}
    for branch in ("O", "N", "R", "F"):
        spec = checkpoints[branch]
        _require(type(spec) is dict and set(spec) == {"path", "expected_sha256", "expected_pins"}, "checkpoint_spec")
        report = analyze_checkpoint(blob_root=blob_root, source_anchors=source_anchors,
            withdrawn_conversation_id=withdrawn_conversation_id,
            preserved_conversation_ids=preserved_conversation_ids, **spec)
        _require(report["pins"]["mode"] == modes[branch], "checkpoint_mode")
        expected_links = {key + "_source_sha256": prior["checkpoint_sha256"] for key, prior in reports.items()}
        _require(report["parent_links"] == expected_links, "parent_link_pin")
        reports[branch] = report
        for digest, size in report["manifest"].items():
            _require(digest not in manifest or manifest[digest] == size, "manifest_size_conflict")
            manifest[digest] = size
        references.extend(report["references"])
    _require(reports["O"]["pins"]["source_commit"] == reports["R"]["pins"]["source_commit"]
        and reports["N"]["pins"]["source_commit"] == reports["F"]["pins"]["source_commit"]
        and reports["O"]["pins"]["source_commit"] != reports["N"]["pins"]["source_commit"], "code_pair")
    _require({p.name for p in blob_root.iterdir()} == {digest + ".bin" for digest in manifest}, "physical_inventory_closure")
    return {"schema_version": 1, "checkpoints": reports, "manifest": manifest,
        "references": references, "blobs": classify_blobs(references), "complete_reference_scope": True,
        "deletion_authorized": False, "cleanup_ack": False, "capability": False}
