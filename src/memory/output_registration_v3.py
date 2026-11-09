"""Strict V3 output-inventory grammar: V2 families plus the account publisher.

This module is pure journal encoding and decoding policy. It owns no
coordinator, lifecycle, close task or authority. Existing V2 owner, target,
delegation and physical events keep their unchanged V2 semantic checks; only
the V3 header, composition and ``account_publisher`` events are new grammar.
"""
from __future__ import annotations

from pathlib import Path
import re
import stat

from src.memory import output_registration as base
from src.memory import output_registration_v2 as v2

ACCOUNT_REPRESENTATION = "kza-users-accounts-v1"
ACCOUNT_PREFIX = ".accounts-"
MAX_ACCOUNT_COUNT = 2**31 - 1
_ACCOUNT_PHASES = {
    "intent": {None}, "attached": {"intent"}, "opening": {"attached"},
    "drained": {"attached", "opening"}, "disabled": {None},
}
_PLAN_KEYS = frozenset({
    "kind", "producer", "role", "representation", "path", "configured_path", "legacy_import_path",
    "ancestor", "parent_identity", "missing_parents", "initial_identity", "creation_observation",
    "before_first_publish", "temporary_scope", "classification",
})
_AGGREGATE_KEYS = frozenset({
    "attempts", "committed_replacements", "clean_failed_attempts", "last_committed_identity",
    "native_drain_returned", "open_temp_count", "cleanup_uncertain",
})
_SELECTED_KEYS = frozenset({"privacy_authority", "authority_sessions", "routines", "events", "suggestions", "accounts"})
_HEADER_KEYS = frozenset({"schema_version", "run_id", "generation", "binding_id", "config_sha256",
                          "state_root", "state_identity", "namespace_identity", "producers", "selected"})


def _expected_name(legacy: Path) -> str:
    return legacy.name if legacy.stem.endswith("_accounts_v1") else legacy.stem + "_accounts_v1.json"


def account_plan_shape(plan: dict, state: Path) -> None:
    """Validate exact atomic-JSON plan keys, types and path relationships."""
    base._require(type(plan) is dict and set(plan) == _PLAN_KEYS
        and plan["kind"] == "atomic_json" and plan["producer"] == "accounts"
        and plan["role"] == "publisher" and plan["representation"] == ACCOUNT_REPRESENTATION
        and plan["before_first_publish"] is True and plan["classification"] == "protected_authority", "account_plan")
    path, ancestor = v2._canonical_path(plan["path"]), v2._canonical_path(plan["ancestor"])
    configured = v2._canonical_path(plan["configured_path"])
    legacy = v2._canonical_path(plan["legacy_import_path"])
    base._require(path.is_relative_to(state) and path != state and ancestor.is_relative_to(state)
        and path.parent.is_relative_to(ancestor)
        and type(plan["missing_parents"]) is list
        and plan["missing_parents"] == list(path.parent.relative_to(ancestor).parts)
        and (plan["initial_identity"] is None or not plan["missing_parents"])
        and plan["creation_observation"] == ("observed_existing" if plan["initial_identity"] is not None else "planned_absent")
        and base._identity_shape(plan["parent_identity"])
        and (plan["initial_identity"] is None or base._identity_shape(plan["initial_identity"]))
        and legacy.parent == path.parent and path.name == _expected_name(legacy)
        and configured.name == legacy.name
        and plan["temporary_scope"] == {"directory": str(path.parent), "prefix": ACCOUNT_PREFIX,
                                        "allocator": "mkstemp"}, "account_plan")


def account_plan(path: str | Path, configured: str | Path, legacy: str | Path, state: Path) -> dict:
    """Observe the canonical account destination before its first publication."""
    supplied = Path(path).absolute()
    base._require(not supplied.is_symlink(), "destination")
    physical = supplied.resolve(strict=False)
    base._require(physical.is_relative_to(state) and physical != state, "destination")
    ancestor, missing = physical.parent, []
    while not ancestor.exists():
        missing.append(ancestor.name)
        ancestor = ancestor.parent
    identity = base._directory(ancestor)
    existing = None
    if physical.exists() or physical.is_symlink():
        info = physical.lstat()
        base._require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "destination")
        existing = base._identity(physical)
    plan = {"kind": "atomic_json", "producer": "accounts", "role": "publisher",
            "representation": ACCOUNT_REPRESENTATION, "path": str(physical),
            "configured_path": str(Path(configured).absolute()),
            "legacy_import_path": str(Path(legacy).absolute().resolve(strict=False)),
            "ancestor": str(ancestor), "parent_identity": identity,
            "missing_parents": list(reversed(missing)), "initial_identity": existing,
            "creation_observation": "observed_existing" if existing is not None else "planned_absent",
            "before_first_publish": True,
            "temporary_scope": {"directory": str(physical.parent), "prefix": ACCOUNT_PREFIX, "allocator": "mkstemp"},
            "classification": "protected_authority"}
    account_plan_shape(plan, state)
    return plan


def _count(value: object) -> bool:
    return type(value) is int and 0 <= value <= MAX_ACCOUNT_COUNT


def _aggregate_shape(aggregate: dict, plan: dict, prior_phase: str) -> None:
    base._require(type(aggregate) is dict and set(aggregate) == _AGGREGATE_KEYS
        and all(_count(aggregate[key]) for key in ("attempts", "committed_replacements", "clean_failed_attempts"))
        and aggregate["attempts"] == aggregate["committed_replacements"] + aggregate["clean_failed_attempts"]
        and aggregate["native_drain_returned"] is True
        and type(aggregate["open_temp_count"]) is int and aggregate["open_temp_count"] == 0
        and aggregate["cleanup_uncertain"] is False
        and (aggregate["attempts"] == 0 if prior_phase == "attached" else aggregate["attempts"] >= 1)
        and (aggregate["last_committed_identity"] == plan["initial_identity"]
             if aggregate["committed_replacements"] == 0
             else base._identity_shape(aggregate["last_committed_identity"])), "account_aggregate")


def account_event_shape(event: dict, prior: dict | None, selected: bool, state: Path) -> dict:
    """Check one account publisher event (without chain fields); return its state."""
    phase = event.get("phase")
    prior_phase = None if prior is None else prior["phase"]
    base._require(phase in _ACCOUNT_PHASES and prior_phase in _ACCOUNT_PHASES[phase], "transition")
    if phase == "disabled":
        base._require(selected is False and set(event) == {"kind", "phase", "configured_path", "path"}, "account_event")
        path, configured = v2._canonical_path(event["path"]), v2._canonical_path(event["configured_path"])
        base._require(path.is_relative_to(state) and path != state and configured.is_absolute(), "account_event")
        return {"phase": "disabled", "plan": None, "producer": "accounts",
                "path": event["path"], "configured_path": event["configured_path"]}
    fields = {"kind", "phase", "plan"} | ({"aggregate"} if phase == "drained" else set())
    base._require(selected is True and set(event) == fields, "account_event")
    plan = event["plan"]
    account_plan_shape(plan, state)
    base._require(prior is None or prior["plan"] == plan, "plan_changed")
    aggregate = None
    if phase == "drained":
        _aggregate_shape(event["aggregate"], plan, prior_phase)
        aggregate = event["aggregate"]
    return {"phase": phase, "plan": plan, "producer": "accounts", "aggregate": aggregate}


def composition_shape(event: dict) -> None:
    base._require(set(event) == {"kind", "revision", "previous_sha256", "instrumented",
                                 "global_inventory_complete", "aggregate_copy_complete"}
        and event["instrumented"] == sorted(V3_FORMAT.selected_producers)
        and event["global_inventory_complete"] is False and event["aggregate_copy_complete"] is False, "event")


def chain(directory: Path) -> tuple[dict, dict, dict, int]:
    """Read V3 only; malformed V3 is never retried as V2 or any other version."""
    namespace = directory.parents[2]
    v2._namespace(namespace)
    header, raw_header = base._read(directory / "epoch.json")
    base._require(header.get("schema_version") == 3 and header.get("run_id") == directory.name
        and re.fullmatch("[0-9a-f]{32}", directory.name)
        and header.get("producers") == list(base.PRODUCERS)
        and header.get("namespace_identity") == base._directory(directory)
        and base._identity_shape(header.get("state_identity"))
        and all(type(header.get(k)) is str and re.fullmatch("[0-9a-f]{64}", header[k])
                for k in ("binding_id", "config_sha256"))
        and type(header.get("generation")) is int and header["generation"] > 0
        and type(header.get("selected")) is dict and set(header["selected"]) == _SELECTED_KEYS
        and all(type(value) is bool for value in header["selected"].values())
        and set(header) == _HEADER_KEYS, "header")
    state = Path(header["state_root"])
    base._require(state.is_absolute() and str(state) == header["state_root"] and ".." not in state.parts, "header")
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
    targets, policy_seen, account = set(), False, None
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
        elif event["kind"] == "account_publisher":
            local = {key: value for key, value in event.items() if key not in {"revision", "previous_sha256"}}
            base._require(set(event) == set(local) | {"revision", "previous_sha256"}, "account_event")
            account = dict(account_event_shape(local, account, header["selected"]["accounts"], state),
                           revision=revision, sha256=digest)
        elif event["kind"] == "composition":
            composition_shape(event)
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
    if account is not None:
        states = dict(states, accounts=account)
    return header, head, states, total


V3_FORMAT = v2._JournalFormat(
    schema_version=3, namespace_name="v3", known_versions=frozenset({"v1", "v2", "v3"}),
    selected_producers=v2._SELECTED | {"accounts"}, decode_chain=chain,
)
