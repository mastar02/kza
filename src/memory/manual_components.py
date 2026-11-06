"""Typed, payload-free witnesses for manual collector components.

This module uses only the standard library. Callers must establish actual
closed-state admission and validate each CollectorOwner's binding/capability;
the DTO itself confers no authority. Captures retain hashes, never row payload.
"""

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
from uuid import UUID


TABLES = {
    "wake_clips_v1": frozenset("sequence id target_id bucket room_id score sources_json captured_at expires_at published".split()),
    "stt_corrections_v1": frozenset("id original_text corrected_text user_id sample_rate used".split()),
    "conversation_turns_v1": frozenset("id group_id input_sources_json quality digest".split()),
    "conversation_export_parents_v1": frozenset(("export_id", "parent_id")),
}
_KINDS = {"wake_clips": (".wav", None), "stt_corrections": (".wav", "stt_correction"),
          "conversations": (".json", "conversation_turn"),
          "conversation_exports": (".jsonl", "conversation_export")}


class ManualComponentError(ValueError):
    """A closed collector component could not be interpreted exactly."""


def _require(value, code):
    if not value:
        raise ManualComponentError("manual_components_" + code)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value):
    return type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None


def _uuid(value):
    _require(type(value) is str, "identity")
    try:
        _require(str(UUID(value)) == value, "identity")
    except ValueError:
        raise ManualComponentError("manual_components_identity") from None
    return value


def _identity(path):
    info = path.stat()
    return {"device": info.st_dev, "inode": info.st_ino}


def _identity_dto(value):
    return (type(value) is dict and set(value) == {"device", "inode"}
            and all(type(item) is int for item in value.values())
            and value["device"] >= 0 and value["inode"] > 0)


def _relative(value):
    _require(type(value) is str and value and PurePosixPath(value).as_posix() == value
             and not PurePosixPath(value).is_absolute()
             and not {".", ".."}.intersection(PurePosixPath(value).parts), "relative_path")


def _typed(value):
    if type(value) is bytes:
        return {"blob_sha256": hashlib.sha256(value).hexdigest(), "size": len(value)}
    _require(value is None or type(value) in (str, int)
             or type(value) is float and math.isfinite(value), "sql_value")
    return value


def _row_hash(row, *, omit_target=False):
    return _digest({key: _typed(value) for key, value in row.items()
                    if not (omit_target and key == "target_id")})


def _target_id(kind, root):
    descriptor = {"kind": kind, "root": str(root)}
    encoded = json.dumps(descriptor, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(json.dumps(["collectors", 1, encoded]).encode()).hexdigest()


def collector_manual_closure(stt_rows, turn_rows, edges) -> set[str]:
    """Return the indivisible manual/export component without I/O.

    An export and every parent form one component. A manually reviewed parent
    therefore protects co-parents and any further exports sharing them.
    """
    def identity(value):
        _require(type(value) is str and bool(value), "identity")
        return value

    protected = {identity(row["id"]) for row in stt_rows}
    for row in turn_rows:
        _require(row["quality"] in {"unmarked", "good", "bad", "corrected"}, "feedback")
        if row["quality"] != "unmarked":
            protected.add(identity(row["id"]))
    edges = tuple(edges)
    for edge in edges:
        identity(edge["export_id"])
        identity(edge["parent_id"])
    while True:
        expanded = (protected | {edge["export_id"] for edge in edges if edge["parent_id"] in protected}
                    | {edge["parent_id"] for edge in edges if edge["export_id"] in protected})
        if expanded == protected:
            return protected
        protected = expanded


def protected_collector_records(tables: dict) -> set[str]:
    """Validate complete collector rows before computing their manual closure."""
    _require(type(tables) is dict and set(tables) == set(TABLES), "table_inventory")
    for name, fields in TABLES.items():
        _require(type(tables[name]) is list, "table_rows")
        for row in tables[name]:
            _require(type(row) is dict and set(row) == fields, "columns")
    clips = tables["wake_clips_v1"]
    keyed = {row["id"]: row for row in clips}
    _require(len(keyed) == len(clips), "duplicate_clip")
    children = {}
    for name, bucket in (("stt_corrections_v1", "stt_correction"),
                         ("conversation_turns_v1", "conversation_turn")):
        rows = tables[name]
        identifiers = {_uuid(row["id"]) for row in rows}
        _require(len(identifiers) == len(rows) and not identifiers.intersection(children), "duplicate_child")
        _require(all(identity in keyed and keyed[identity]["bucket"] == bucket
                     for identity in identifiers), "child_clip_missing")
        children.update((identity, bucket) for identity in identifiers)
    for edge in tables["conversation_export_parents_v1"]:
        _require(edge["export_id"] in keyed and keyed[edge["export_id"]]["bucket"] == "conversation_export"
                 and children.get(edge["parent_id"]) == "conversation_turn", "export_parent_missing")
    return collector_manual_closure(tables["stt_corrections_v1"], tables["conversation_turns_v1"],
                                    tables["conversation_export_parents_v1"])


@dataclass(frozen=True)
class CollectorOwner:
    """An audit locator whose admission proof is supplied by the controller."""

    target_id: str
    kind: str
    root: Path
    state_root: Path
    state_identity: dict
    binding_id: str


def _locator(owner: CollectorOwner) -> dict:
    _require(type(owner) is CollectorOwner and owner.kind in _KINDS
             and _sha(owner.target_id) and _sha(owner.binding_id), "owner")
    state, root = owner.state_root, owner.root
    _require(_identity_dto(owner.state_identity)
             and isinstance(state, Path) and state.is_absolute() and state.resolve(strict=True) == state
             and state.is_dir() and _identity(state) == owner.state_identity, "owner_state")
    _require(isinstance(root, Path) and root.is_absolute() and root.is_relative_to(state)
             and root.resolve() == root and not root.is_symlink(), "owner_root")
    _require(_target_id(owner.kind, root) == owner.target_id, "owner_target")
    present = root.exists()
    _require(not present or root.is_dir(), "owner_root")
    return {"target_id": owner.target_id, "kind": owner.kind, "root": str(root),
            "relative_root": root.relative_to(state).as_posix(),
            "root_identity": _identity(root) if present else None,
            "owner_state_root": str(state), "owner_state_identity": owner.state_identity,
            "owner_binding_id": owner.binding_id}


def _file(owner: CollectorOwner, identity: str) -> dict:
    relative = "policy_v1/" + _uuid(identity) + _KINDS[owner.kind][0]
    path = owner.root / relative
    _require(path.resolve() == path and not path.is_symlink(), "file_redirected")
    try:
        before = path.lstat()
    except FileNotFoundError:
        return {"relative_file": relative, "status": "absent", "sha256": None,
                "size": None, "identity": None}
    _require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, "file_type")
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        opened = os.fstat(stream.fileno())
        _require((opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino)
                 and stat.S_ISREG(opened.st_mode) and opened.st_nlink == 1, "file_changed")
        while block := stream.read(1024 * 1024):
            digest.update(block)
    after = path.lstat()
    _require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
             == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), "file_changed")
    return {"relative_file": relative, "status": "present", "sha256": digest.hexdigest(),
            "size": before.st_size, "identity": {"device": before.st_dev, "inode": before.st_ino}}


def capture_manual_components(state_root: Path, stores: dict, owners: dict[str, CollectorOwner]) -> dict:
    """Hash complete typed rows/edges and exact file presence under known owners."""
    state_root = Path(state_root)
    _require(state_root.is_absolute() and state_root.resolve(strict=True) == state_root
             and state_root.is_dir() and type(stores) is dict and type(owners) is dict, "state")
    _require(all(type(owner) is CollectorOwner and key == owner.target_id
                 for key, owner in owners.items()), "owner_inventory")
    locators = {key: _locator(owner) for key, owner in owners.items()}
    _require(len({value["root"] for value in locators.values()}) == len(locators), "owner_ambiguous")
    result = {}
    for relative, tables in sorted(stores.items()):
        name = PurePosixPath(relative)
        _require(type(relative) is str and not name.is_absolute() and ".." not in name.parts
                 and name.parts and str(name) == relative, "database")
        protected = protected_collector_records(tables)
        clips = {row["id"]: row for row in tables["wake_clips_v1"]}
        _require(len(clips) == len(tables["wake_clips_v1"]) and protected <= clips.keys(), "clip_closure")
        captured = {}
        for identity in sorted(protected):
            row = clips[identity]
            _uuid(identity)
            _require(row["target_id"] in owners and type(row["published"]) is int
                     and row["published"] in (0, 1), "clip_owner")
            owner = owners[row["target_id"]]
            locator = locators[row["target_id"]]
            bucket = _KINDS[owner.kind][1]
            _require(bucket is None or row["bucket"] == bucket, "clip_kind")
            captured[identity] = {"row_sha256": _row_hash(row),
                                  "relocatable_row_sha256": _row_hash(row, omit_target=True),
                                  "locator": locator, "file": _file(owner, identity)}
        hashes = {}
        for table in ("stt_corrections_v1", "conversation_turns_v1"):
            selected = [row for row in tables[table] if row["id"] in protected]
            values = {row["id"]: _row_hash(row) for row in selected}
            _require(len(values) == len(selected), "duplicate_child")
            hashes[table] = values
        edges = [dict(edge) for edge in tables["conversation_export_parents_v1"] if edge["export_id"] in protected]
        _require(len({_canonical(edge) for edge in edges}) == len(edges), "duplicate_edge")
        edges = [dict(edge, row_sha256=_row_hash(edge)) for edge in edges]
        result[relative] = {"clips": captured, "child_rows": hashes,
                            "edges": sorted(edges, key=_canonical)}
    result = {"schema_version": 1, "state_root": str(state_root),
              "state_identity": _identity(state_root), "stores": result}
    _validate_capture(result)
    return result


def _validate_capture(value):
    """Validate the content-free wire shape without reopening historical paths."""
    _require(type(value) is dict and set(value) == {"schema_version", "state_root", "state_identity", "stores"}
             and type(value["schema_version"]) is int and value["schema_version"] == 1
             and type(value["state_root"]) is str and Path(value["state_root"]).is_absolute()
             and _identity_dto(value["state_identity"]) and type(value["stores"]) is dict, "projection")
    for relative, store in value["stores"].items():
        _relative(relative)
        _require(type(store) is dict and set(store) == {"clips", "child_rows", "edges"}
                 and type(store["clips"]) is dict and type(store["child_rows"]) is dict
                 and set(store["child_rows"]) == {"stt_corrections_v1", "conversation_turns_v1"}
                 and type(store["edges"]) is list, "projection_store")
        for identity, clip in store["clips"].items():
            _uuid(identity)
            _require(type(clip) is dict and set(clip) == {"row_sha256", "relocatable_row_sha256", "locator", "file"}
                     and _sha(clip["row_sha256"]) and _sha(clip["relocatable_row_sha256"]), "projection_clip")
            locator, file = clip["locator"], clip["file"]
            _require(type(locator) is dict and set(locator) == {"target_id", "kind", "root", "relative_root",
                "root_identity", "owner_state_root", "owner_state_identity", "owner_binding_id"}
                and _sha(locator["target_id"]) and locator["kind"] in _KINDS
                and _sha(locator["owner_binding_id"]) and _identity_dto(locator["owner_state_identity"])
                and (locator["root_identity"] is None or _identity_dto(locator["root_identity"])), "projection_locator")
            _relative(locator["relative_root"])
            _require(type(locator["root"]) is str and type(locator["owner_state_root"]) is str
                     and Path(locator["owner_state_root"]).is_absolute()
                     and Path(locator["owner_state_root"]) / locator["relative_root"] == Path(locator["root"]),
                     "projection_locator")
            _require(_target_id(locator["kind"], locator["root"]) == locator["target_id"], "projection_locator")
            _require(type(file) is dict and set(file) == {"relative_file", "status", "sha256", "size", "identity"}
                     and file["relative_file"] == "policy_v1/" + identity + _KINDS[locator["kind"]][0], "projection_file")
            _require(file["status"] == "absent" and file["sha256"] is None and file["size"] is None and file["identity"] is None
                or file["status"] == "present" and _sha(file["sha256"]) and type(file["size"]) is int
                and file["size"] >= 0 and _identity_dto(file["identity"]), "projection_file")
        for name, rows in store["child_rows"].items():
            _require(type(rows) is dict, "projection_child")
            for identity, digest in rows.items():
                _uuid(identity)
                _require(identity in store["clips"] and _sha(digest), "projection_child")
        seen = set()
        for edge in store["edges"]:
            _require(type(edge) is dict and set(edge) == {"export_id", "parent_id", "row_sha256"}, "projection_edge")
            pair = {key: edge[key] for key in ("export_id", "parent_id")}
            _require(all(identity in store["clips"] for identity in pair.values())
                     and edge["row_sha256"] == _row_hash(pair) and _canonical(pair) not in seen, "projection_edge")
            seen.add(_canonical(pair))


def verify_manual_components(actual: dict, sealed: dict) -> None:
    """A closed source must still have the exact physical witnessed state."""
    _validate_capture(actual)
    _validate_capture(sealed)
    _require(_canonical(actual) == _canonical(sealed), "closed_state_changed")


def equivalent_manual_components(latest: dict, candidate: dict, *, target_relocations: dict[str, str]) -> bool:
    """Compare only explicit collector relocation, never other row/file fields.

    Both captures must have been made with controller-validated CollectorOwners.
    This comparison does not validate or manufacture an origin capability.
    """
    _require(type(target_relocations) is dict
             and all(_sha(old) and _sha(new) and old != new for old, new in target_relocations.items()), "relocations")
    _validate_capture(latest)
    _validate_capture(candidate)
    if set(latest["stores"]) != set(candidate["stores"]):
        return False
    used = {}
    for relative, source in latest["stores"].items():
        target = candidate["stores"][relative]
        if (set(source["clips"]) != set(target["clips"]) or source["edges"] != target["edges"]
                or source["child_rows"] != target["child_rows"]):
            return False
        for identity, before in source["clips"].items():
            after = target["clips"][identity]
            old, new = before["locator"]["target_id"], after["locator"]["target_id"]
            if old != new:
                if target_relocations.get(old) != new:
                    return False
                used[old] = new
            else:
                if before["row_sha256"] != after["row_sha256"]:
                    return False
                if any(before["locator"][field] != after["locator"][field] for field in
                       ("root", "root_identity", "owner_state_root", "owner_state_identity")):
                    return False
                if before["file"]["identity"] != after["file"]["identity"]:
                    return False
            if before["relocatable_row_sha256"] != after["relocatable_row_sha256"]:
                return False
            for field in ("kind", "relative_root"):
                if before["locator"][field] != after["locator"][field]:
                    return False
            for field in ("relative_file", "status", "sha256", "size"):
                if before["file"][field] != after["file"][field]:
                    return False
    return used == target_relocations
