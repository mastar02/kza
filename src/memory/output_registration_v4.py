"""Extensible V4 output-inventory grammar: header-declared producer grammars.

V4 is the last journal version bump. The run header declares, for each
producer outside the V2 families, the grammar id its events use; the decoder
validates every declaration against ``_GRAMMARS``, a closed, append-only
module table. A later producer adds a table entry (and its pin), never a new
journal version. An unknown grammar id fails closed with its own typed reason
and is never decoded under another grammar.

This module is pure journal encoding and decoding policy. It owns no
coordinator, lifecycle, close task or authority. V2 owner, target, delegation
and physical events keep their unchanged V2 checks, and the account publisher
keeps the unchanged V3 function.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import stat
from types import MappingProxyType

from src.memory import output_registration as base
from src.memory import output_registration_v2 as v2
from src.memory import output_registration_v3 as v3

CONFIGURATION_REPRESENTATION = "kza-settings-yaml-raw-v1"
CONFIGURATION_PREFIX = ".consent-"
MAX_CONFIGURATION_BYTES = 2 * 1024 * 1024  # The owner's own settings.yaml bound.
_GRAMMAR_ID = re.compile(r"[a-z][a-z0-9_]*/[1-9][0-9]{0,8}")
_BASE_SELECTED = frozenset({"privacy_authority", "authority_sessions", "routines", "events", "suggestions"})
_HEADER_KEYS = v3._HEADER_KEYS | {"grammars"}
_CONFIGURATION_PHASES = {
    "intent": {None}, "attached": {"intent"}, "opening": {"attached"},
    "drained": {"attached", "opening"}, "disabled": {None},
}
_CONFIGURATION_PLAN_KEYS = frozenset({
    "kind", "producer", "role", "representation", "path", "parent", "state_identity", "parent_identity",
    "initial_identity", "initial_mode", "initial_sha256", "before_first_publish", "write_window",
    "temporary_scope", "classification",
})
_CONFIGURATION_AGGREGATE_KEYS = frozenset({
    "outcome", "attempts", "last_committed_identity", "last_committed_sha256", "writer_returned",
    "open_temp_count", "cleanup_uncertain",
})
# Outcome -> prior phases from which a drained terminal may report it.
CONFIGURATION_OUTCOMES = MappingProxyType({
    "no_pending": frozenset({"attached"}),
    "not_attempted_graph_failed": frozenset({"attached"}),
    "refused_before_temporary": frozenset({"attached", "opening"}),
    "clean_failed": frozenset({"opening"}),
    "committed": frozenset({"opening"}),
})


class UnsupportedProducerGrammar(RuntimeError):
    """A run declares a grammar this release does not know; never decoded."""


def _hex64(value: object) -> bool:
    return type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None


def configuration_path(state: Path) -> Path:
    """Return the only destination this grammar accepts: <state>/config/settings.yaml."""
    return Path(state) / "config" / "settings.yaml"


def configuration_plan_shape(plan: dict, state: Path, state_identity: dict | None = None) -> None:
    """Validate exact raw-YAML plan keys, constants and path relationships."""
    path = configuration_path(state)
    base._require(type(plan) is dict and set(plan) == _CONFIGURATION_PLAN_KEYS
        and plan["kind"] == "atomic_yaml" and plan["producer"] == "configuration"
        and plan["role"] == "publisher" and plan["representation"] == CONFIGURATION_REPRESENTATION
        and plan["path"] == str(path) and plan["parent"] == str(path.parent)
        and base._identity_shape(plan["state_identity"])
        and (state_identity is None or plan["state_identity"] == state_identity)
        and base._identity_shape(plan["parent_identity"]) and base._identity_shape(plan["initial_identity"])
        and type(plan["initial_mode"]) is int and 0 <= plan["initial_mode"] <= 0o7777
        and _hex64(plan["initial_sha256"])
        and plan["before_first_publish"] is True and plan["write_window"] == "after_graph_close"
        and plan["temporary_scope"] == {"directory": str(path.parent), "prefix": CONFIGURATION_PREFIX,
                                        "allocator": "mkstemp"}
        and plan["classification"] == "protected_policy_input", "configuration_plan")


def _read_settings(path: Path) -> tuple[os.stat_result, bytes]:
    before = path.lstat()
    base._require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
                  and before.st_size <= MAX_CONFIGURATION_BYTES, "destination")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        base._require((info.st_dev, info.st_ino) == (before.st_dev, before.st_ino), "destination")
        raw = stream.read(MAX_CONFIGURATION_BYTES + 1)
    base._require(len(raw) <= MAX_CONFIGURATION_BYTES, "destination")
    return before, raw


def configuration_plan(state: str | Path) -> dict:
    """Observe the canonical settings file; content is hashed, never serialized."""
    state = Path(state)
    path = configuration_path(state)
    try:
        base._require(not path.is_symlink() and not path.parent.is_symlink(), "destination")
        state_identity = base._directory(state)
        parent_identity = base._directory(path.parent)
        info, raw = _read_settings(path)
    except OSError:
        base._require(False, "destination")
    plan = {"kind": "atomic_yaml", "producer": "configuration", "role": "publisher",
            "representation": CONFIGURATION_REPRESENTATION, "path": str(path), "parent": str(path.parent),
            "state_identity": state_identity, "parent_identity": parent_identity,
            "initial_identity": {"device": info.st_dev, "inode": info.st_ino},
            "initial_mode": stat.S_IMODE(info.st_mode), "initial_sha256": hashlib.sha256(raw).hexdigest(),
            "before_first_publish": True, "write_window": "after_graph_close",
            "temporary_scope": {"directory": str(path.parent), "prefix": CONFIGURATION_PREFIX,
                                "allocator": "mkstemp"},
            "classification": "protected_policy_input"}
    configuration_plan_shape(plan, state, state_identity)
    return plan


def configuration_observation(plan: dict) -> tuple[dict, str]:
    """Return the destination's current identity and content digest."""
    path = Path(plan["path"])
    try:
        base._require(not path.is_symlink(), "destination_changed")
        info, raw = _read_settings(path)
    except OSError:
        base._require(False, "destination_changed")
    return {"device": info.st_dev, "inode": info.st_ino}, hashlib.sha256(raw).hexdigest()


def _aggregate_shape(aggregate: dict, plan: dict, prior_phase: str) -> None:
    base._require(type(aggregate) is dict and set(aggregate) == _CONFIGURATION_AGGREGATE_KEYS
        and aggregate["outcome"] in CONFIGURATION_OUTCOMES
        and prior_phase in CONFIGURATION_OUTCOMES[aggregate["outcome"]], "configuration_aggregate")
    committed = aggregate["outcome"] == "committed"
    attempted = aggregate["outcome"] in {"committed", "clean_failed"}
    base._require(type(aggregate["attempts"]) is int and aggregate["attempts"] == (1 if attempted else 0)
        and aggregate["writer_returned"] is True
        and type(aggregate["open_temp_count"]) is int and aggregate["open_temp_count"] == 0
        and aggregate["cleanup_uncertain"] is False
        and (base._identity_shape(aggregate["last_committed_identity"])
             and aggregate["last_committed_identity"] != plan["initial_identity"]
             and _hex64(aggregate["last_committed_sha256"])
             if committed else
             aggregate["last_committed_identity"] == plan["initial_identity"]
             and aggregate["last_committed_sha256"] == plan["initial_sha256"]), "configuration_aggregate")


def configuration_event_shape(event: dict, prior: dict | None, selected: bool, state: Path,
                              state_identity: dict | None = None) -> dict:
    """Check one configuration publisher event (without chain fields); return its state."""
    phase = event.get("phase")
    prior_phase = None if prior is None else prior["phase"]
    base._require(phase in _CONFIGURATION_PHASES and prior_phase in _CONFIGURATION_PHASES[phase], "transition")
    if phase == "disabled":
        base._require(selected is False and set(event) == {"kind", "phase", "path"}
                      and event["path"] == str(configuration_path(state)), "configuration_event")
        return {"phase": "disabled", "plan": None, "producer": "configuration", "path": event["path"]}
    fields = {"kind", "phase", "plan"} | ({"aggregate"} if phase == "drained" else set())
    base._require(selected is True and set(event) == fields, "configuration_event")
    plan = event["plan"]
    configuration_plan_shape(plan, state, state_identity)
    base._require(prior is None or prior["plan"] == plan, "plan_changed")
    aggregate = None
    if phase == "drained":
        _aggregate_shape(event["aggregate"], plan, prior_phase)
        aggregate = event["aggregate"]
    return {"phase": phase, "plan": plan, "producer": "configuration", "aggregate": aggregate}


# ----------------------------------------------------------------------
# The closed grammar table. Append-only: an entry is never edited; a new
# producer adds an entry and its own pin.
# ----------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class _Grammar:
    """One fixed producer grammar; never a coordinator, plugin or authority."""
    grammar_id: str
    producer: str
    event_kind: str
    decode_event: Callable

    def __reduce_ex__(self, protocol: int):
        raise TypeError("output_journal_grammar_not_serializable")


def _account_event(event: dict, prior: dict | None, selected: bool, header: dict) -> dict:
    return v3.account_event_shape(event, prior, selected, Path(header["state_root"]))


def _configuration_event(event: dict, prior: dict | None, selected: bool, header: dict) -> dict:
    return configuration_event_shape(event, prior, selected, Path(header["state_root"]), header["state_identity"])


_GRAMMARS = MappingProxyType({
    "account_publisher/1": _Grammar("account_publisher/1", "accounts", "account_publisher", _account_event),
    "configuration_publisher/1": _Grammar("configuration_publisher/1", "configuration",
                                          "configuration_publisher", _configuration_event),
})
WRITER_GRAMMARS = MappingProxyType({"accounts": "account_publisher/1", "configuration": "configuration_publisher/1"})


def declared_grammars(header: dict) -> dict[str, _Grammar]:
    """Resolve the header's declarations; unknown ids raise their own type."""
    grammars = header.get("grammars")
    base._require(type(grammars) is dict
        and all(type(producer) is str and producer in base.PRODUCERS and producer not in v2._SELECTED
                for producer in grammars)
        and all(type(value) is str and _GRAMMAR_ID.fullmatch(value) for value in grammars.values()), "header")
    if any(value not in _GRAMMARS for value in grammars.values()):
        raise UnsupportedProducerGrammar("output_registration_unsupported_producer_grammar")
    resolved = {producer: _GRAMMARS[value] for producer, value in grammars.items()}
    base._require(all(grammar.producer == producer for producer, grammar in resolved.items()), "header")
    return resolved


def refuse_unknown_grammars(runs: Path) -> None:
    """Writer start gate: an existing run declaring an unknown grammar blocks a new run.

    The caller already bounded the namespace entry count. Unreadable or
    grammar-less headers are history for the reader, not declarations.
    """
    if not runs.exists():
        return
    for directory in sorted(runs.iterdir()):
        try:
            header, _ = base._read(directory / "epoch.json")
        except (OSError, RuntimeError, ValueError):
            continue
        grammars = header.get("grammars") if type(header) is dict else None
        if type(grammars) is dict and any(type(value) is not str or value not in _GRAMMARS
                                          for value in grammars.values()):
            raise UnsupportedProducerGrammar("output_registration_unsupported_producer_grammar")


def composition_shape(event: dict, producers: set) -> None:
    base._require(set(event) == {"kind", "revision", "previous_sha256", "instrumented",
                                 "global_inventory_complete", "aggregate_copy_complete"}
        and event["instrumented"] == sorted(v2._SELECTED | producers)
        and event["global_inventory_complete"] is False and event["aggregate_copy_complete"] is False, "event")


def chain(directory: Path) -> tuple[dict, dict, dict, int]:
    """Read V4 only; malformed V4 is never retried as V3, V2 or V1."""
    namespace = directory.parents[2]
    v2._namespace(namespace)
    header, raw_header = base._read(directory / "epoch.json")
    base._require(header.get("schema_version") == 4 and header.get("run_id") == directory.name
        and re.fullmatch("[0-9a-f]{32}", directory.name)
        and header.get("producers") == list(base.PRODUCERS)
        and header.get("namespace_identity") == base._directory(directory)
        and base._identity_shape(header.get("state_identity"))
        and all(_hex64(header.get(k)) for k in ("binding_id", "config_sha256"))
        and type(header.get("generation")) is int and header["generation"] > 0
        and type(header.get("grammars")) is dict
        and type(header.get("selected")) is dict
        and set(header["selected"]) == _BASE_SELECTED | set(header["grammars"])
        and all(type(value) is bool for value in header["selected"].values())
        and set(header) == _HEADER_KEYS, "header")
    state = Path(header["state_root"])
    base._require(state.is_absolute() and str(state) == header["state_root"] and ".." not in state.parts, "header")
    grammars = declared_grammars(header)
    by_kind = {grammar.event_kind: grammar for grammar in grammars.values()}
    base._require(len(by_kind) == len(grammars), "header")
    head, raw_head = base._read(directory / "HEAD.json")
    base._require(set(head) == {"revision", "sha256"} and type(head["revision"]) is int
                  and 0 <= head["revision"] <= base.MAX_EVENTS, "head")
    paths = list(directory.iterdir())
    by_revision = {}
    for path in paths:
        match = re.fullmatch(r"(\d{12})-([0-9a-f]{64})\.json", path.name)
        if match:
            revision = int(match[1])
            base._require(revision not in by_revision, "orphan")
            by_revision[revision] = path
    base._require(len(paths) == head["revision"] + 2 and len(by_revision) == head["revision"], "orphan")
    previous, total, states = base._sha(raw_header), len(raw_header) + len(raw_head), {}
    targets, policy_seen, publishers = set(), False, {}
    for revision in range(1, head["revision"] + 1):
        path = by_revision[revision]
        event, raw = base._read(path)
        total += len(raw)
        digest = base._sha(raw)
        base._require(path.name == f"{revision:012d}-{digest}.json" and event["revision"] == revision
                      and event["previous_sha256"] == previous, "chain")
        if event["kind"] == "owner":
            base._require(set(event) == {"kind", "owner", "phase", "plan", "native_identity", "revision", "previous_sha256"}, "event")
            key, phase, plan = event["owner"], event["phase"], event["plan"]
            v2._plan_shape(plan, state)
            prior = states.get(key)
            base._require(phase in v2._PHASES and (None if prior is None else prior["phase"]) in v2._PHASES[phase], "transition")
            base._require(prior is None or prior["plan"] == plan, "plan_changed")
            v2._owner_event_shape(event, prior, header["selected"])
            states[key] = {"phase": phase, "plan": plan, "native_identity": event["native_identity"],
                           "revision": revision, "sha256": digest}
        elif event["kind"] in by_kind:
            grammar = by_kind[event["kind"]]
            local = {key: value for key, value in event.items() if key not in {"revision", "previous_sha256"}}
            decoded = grammar.decode_event(local, publishers.get(grammar.producer),
                                           header["selected"][grammar.producer], header)
            publishers[grammar.producer] = dict(decoded, revision=revision, sha256=digest)
        elif event["kind"] == "composition":
            composition_shape(event, set(grammars))
        else:
            if event["kind"] == "delegation_policy":
                base._require(not policy_seen, "delegated_policy")
                policy_seen = True
            if event["kind"] == "delegated_target":
                base._require(policy_seen, "delegated_policy")
            v2._observation_shape(event, state, states, targets)
        previous = digest
    base._require(previous == head["sha256"] and total <= base.MAX_RUN_BYTES, "head")
    base._require(base._read(directory / "HEAD.json")[1] == raw_head
                  and base._read(directory / "epoch.json")[1] == raw_header, "journal_changed")
    # Owner plans accept only V2 families, so publisher names never collide.
    base._require(not set(publishers) & set(states), "event")
    return header, head, dict(states, **publishers), total


V4_FORMAT = v2._JournalFormat(
    schema_version=4, namespace_name="v4", known_versions=frozenset({"v1", "v2", "v3", "v4"}),
    selected_producers=v2._SELECTED | set(WRITER_GRAMMARS), decode_chain=chain,
)
