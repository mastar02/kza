"""Closed T024 grammars for the feature, integration, dashboard and latency owners.

The composition root supplies selected roles before opening and independently
observed drained roles at shutdown. These builders only produce journal metadata.
"""

from __future__ import annotations

import asyncio
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from functools import partial
from hashlib import sha256
from pathlib import PurePath
import re
import sys
from typing import TYPE_CHECKING
from urllib.parse import urlsplit
from uuid import uuid4

from src.memory import output_registration as base
from src.memory.output_publisher import GenericOutputPublisher, GrammarSpec, publisher_grammar, publisher_plan

if TYPE_CHECKING:
    from src.memory.output_registration_v2 import OutputRegistrationV2

FEATURES_PRODUCERS = ("household_features", "home_integrations", "dashboard_transport", "latency")

# Role IDs join the nominal graph. Boundaries describe where each original
# owner operates, including delegates to an existing store or protocol.
_OWNERS = {
    "household_features": {
        "feature_manager": {"kind": "resource_only", "owner": "feature_manager"},
        "timers_runtime": {
            "kind": "resource_only", "owner": "named_timer_manager",
            "durable_boundary": "reminders", "store": "timer_store",
        },
        "smart_notifications": {"kind": "resource_only", "owner": "smart_notification_manager"},
        "alerts_runtime": {
            "kind": "resource_only", "owner": "alert_manager",
            "durable_boundary": "reminders", "table": "alert_preferences",
        },
        "intercom_delivery": {"kind": "resource_only", "owner": "intercom_system"},
    },
    "home_integrations": {
        "ha_client": {"kind": "transport_session", "owner": "home_assistant_client"},
        "home_actions": {"kind": "external_effect", "owner": "home_action_service"},
        "home_capabilities": {"kind": "readonly_input", "owner": "home_capability_service"},
        "energy_monitor": {"kind": "readonly_input", "owner": "energy_monitor"},
        "camera_health": {"kind": "readonly_input", "owner": "camera_health_monitor"},
        "calendar_reader": {"kind": "readonly_input", "owner": "calendar_reader", "input": "ha_response_or_ics"},
        "media_control": {"kind": "external_effect", "owner": "media_controller"},
        "ha_todo_sync": {"kind": "external_effect", "owner": "ha_sync_manager", "local_boundary": "lists"},
    },
    "dashboard_transport": {
        "dashboard_http": {"kind": "transport_session", "owner": "dashboard_api"},
        "live_event_bus": {"kind": "resource_only", "owner": "live_event_bus"},
        "live_receipt_store": {
            "kind": "sqlite", "owner": "live_receipt_store", "durable_boundary": "privacy_authority",
        },
        "browser_delivery": {"kind": "transport_session", "owner": "live_retention"},
    },
    "latency": {
        "latency_sqlite": {
            "kind": "sqlite", "owner": "latency_monitor",
            "recent_records": 1000, "recent_per_component": 100,
        },
        "health_aggregator": {"kind": "resource_only", "owner": "health_aggregator", "recent_failures": 50},
        "llm_metrics": {
            "kind": "resource_only", "owner": "llm_metrics_tracker",
            "window_seconds": 300, "recent_per_endpoint": 200,
        },
    },
}
_NOMINAL_OWNERS = {
    "feature_manager": ("src.pipeline.feature_manager", "FeatureManager"),
    "timers_runtime": ("src.timers.named_timers", "NamedTimerManager"),
    "smart_notifications": ("src.notifications.smart_notifications", "SmartNotificationManager"),
    "alerts_runtime": ("src.alerts.alert_manager", "AlertManager"),
    "intercom_delivery": ("src.intercom.intercom_system", "IntercomSystem"),
    "ha_client": ("src.home_assistant.ha_client", "HomeAssistantClient"),
    "home_actions": ("src.home_assistant.home_actions", "HomeActionService"),
    "home_capabilities": ("src.home_assistant.capability_service", "HomeCapabilityService"),
    "energy_monitor": ("src.integrations.energy_monitor", "EnergyMonitor"),
    "camera_health": ("src.integrations.camera_health", "CameraHealthMonitor"),
    "calendar_reader": ("src.integrations.calendar_reader", "CalendarReader"),
    "media_control": ("src.integrations.media_control", "MediaController"),
    "ha_todo_sync": ("src.lists.ha_sync", "HASyncManager"),
    "dashboard_http": ("src.dashboard.api", "DashboardAPI"),
    "live_event_bus": ("src.dashboard.live_event_bus", "LiveEventBus"),
    "live_receipt_store": ("src.dashboard.live_receipts", "LiveReceiptStore"),
    "browser_delivery": ("src.dashboard.live_retention", "LiveRetention"),
    "latency_sqlite": ("src.monitoring.latency_monitor", "LatencyMonitor"),
    "health_aggregator": ("src.monitoring.health_aggregator", "HealthAggregator"),
    "llm_metrics": ("src.llm.metrics", "LLMMetricsTracker"),
}
_RETAINED_DRAINS = {
    "ha_client": "_close_task",
    "home_capabilities": "_close_task",
    "dashboard_http": "_stop_task",
    "live_receipt_store": "_closing",
    "browser_delivery": "_closing",
}
_KINDS = {
    "household_features": "resource_only",
    "home_integrations": "transport_session",
    "dashboard_transport": "transport_session",
    "latency": "sqlite",
}
_FIXED = {
    "household_features": {},
    "home_integrations": {"selection": "home_assistant_control_selection_path"},
    "dashboard_transport": {
        # Versions of the erasure targets in dashboard/live_erasure.py.
        "receipt_targets": {
            "live_browser": {"version": 1, "kind": "live_browser_receipts", "protocol": 1},
            "conversation": {"version": 3, "kind": "conversation_rest_and_live_receipts", "protocol": 2},
            "voice_review": {"version": 4, "kind": "voice_review_rest_and_live_receipts", "protocol": 3},
        },
        "sent_remote_bytes_locally_erasable": False,
    },
    "latency": {"database": "monitoring_db_path", "scope": "technical_metrics"},
}
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _role_flags(producer: str, active_roles: Collection[str]) -> dict[str, bool]:
    base._require(type(producer) is str and producer in _OWNERS, "features_producer")
    selected = tuple(active_roles)
    owners = _OWNERS[producer]
    base._require(all(type(role) is str and role in owners for role in selected)
                  and len(set(selected)) == len(selected), "features_roles")
    return {role: role in selected for role in owners}


def _plan(producer: str, enabled: dict[str, bool], **destination: object) -> dict:
    return publisher_plan(
        _KINDS[producer], roles=tuple(_OWNERS[producer]), owners=_OWNERS[producer],
        enabled=enabled, **_FIXED[producer], **destination,
    )


def household_features_plan(*, selected_roles: Collection[str]) -> dict:
    """Describe the planned feature owners and their existing storage boundary."""
    return _plan("household_features", _role_flags("household_features", selected_roles))


def home_integrations_plan(
    *, selected_roles: Collection[str], ha_url: str | None = None, selection_configured: bool = False,
) -> dict:
    """Hash the HA origin and record the logical selection resolver."""
    enabled = _role_flags("home_integrations", selected_roles)
    configured = any(enabled.values())
    base._require(type(selection_configured) is bool and (configured or not selection_configured),
                  "features_plan")
    base._require(type(ha_url) is str if configured else ha_url is None, "features_ha_origin")
    origin = None
    if ha_url is not None:
        try:
            url = urlsplit(ha_url)
            host, port = url.hostname, url.port
        except ValueError:
            raise RuntimeError("features_ha_origin") from None
        base._require(url.scheme in {"http", "https"} and bool(host) and port != 0, "features_ha_origin")
        port = port if port is not None else (443 if url.scheme == "https" else 80)
        origin = sha256(f"{url.scheme}|{host}|{port}".encode()).hexdigest()
    return _plan("home_integrations", enabled, origin_sha256=origin, selection_configured=selection_configured)


def dashboard_transport_plan(*, selected_roles: Collection[str]) -> dict:
    """Describe the dashboard transport and existing receipt target versions."""
    return _plan("dashboard_transport", _role_flags("dashboard_transport", selected_roles))


def latency_plan(*, selected_roles: Collection[str], db_path: str | PurePath | None = None) -> dict:
    """Hash the effective monitoring path supplied by the original resolver."""
    enabled = _role_flags("latency", selected_roles)
    if isinstance(db_path, PurePath):
        db_path = db_path.as_posix()
    base._require(type(db_path) is str and bool(db_path) if enabled["latency_sqlite"]
                  else db_path is None, "features_latency_database")
    digest = sha256(db_path.encode()).hexdigest() if db_path is not None else None
    return _plan("latency", enabled, database_sha256=digest)


def features_aggregate(producer: str, *, instance_sha256: Mapping[str, str]) -> dict:
    """Build detached terminal observations from captured owner identities."""
    drained = _role_flags(producer, instance_sha256.keys())
    base._require(all(_digest_shape(value, True) for value in instance_sha256.values())
                  and len(set(instance_sha256.values())) == len(instance_sha256), "features_aggregate")
    return {"owners": {
        role: {"observation": "drained" if enabled else "disabled",
               "instance_sha256": instance_sha256.get(role)}
        for role, enabled in drained.items()
    }}


def _digest_shape(value: object, present: bool) -> bool:
    return bool(type(value) is str and _SHA256.fullmatch(value)) if present else value is None


def _plan_shape(producer: str, plan: dict, header: dict) -> None:
    enabled = plan.get("enabled")
    base._require(type(enabled) is dict and set(enabled) == _OWNERS[producer].keys()
                  and all(type(flag) is bool for flag in enabled.values()), "features_plan")
    selected = any(enabled.values())
    base._require(header["selected"][producer] is selected, "features_selection")

    destination = {}
    if producer == "home_integrations":
        origin, configured = plan.get("origin_sha256"), plan.get("selection_configured")
        base._require(_digest_shape(origin, selected) and type(configured) is bool
                      and (selected or not configured), "features_plan")
        destination = {"origin_sha256": origin, "selection_configured": configured}
    elif producer == "latency":
        digest = plan.get("database_sha256")
        base._require(_digest_shape(digest, enabled["latency_sqlite"]), "features_plan")
        destination = {"database_sha256": digest}
    # Canonical encoding keeps booleans distinct from their numeric equivalents.
    base._require(base._encoded(plan) == base._encoded(_plan(producer, enabled, **destination)), "features_plan")


def _aggregate_shape(aggregate: dict, plan: dict, _header: dict) -> None:
    owners = aggregate.get("owners")
    enabled = plan["enabled"]
    base._require(set(aggregate) == {"owners"} and type(owners) is dict
                  and owners.keys() == enabled.keys(), "features_aggregate")
    identities = set()
    for role, evidence in owners.items():
        selected = enabled[role]
        base._require(type(evidence) is dict and set(evidence) == {"observation", "instance_sha256"}
                      and evidence["observation"] == ("drained" if selected else "disabled")
                      and _digest_shape(evidence["instance_sha256"], selected), "features_aggregate")
        if selected:
            digest = evidence["instance_sha256"]
            base._require(digest not in identities, "features_aggregate")
            identities.add(digest)


GRAMMAR_SPECS: tuple[GrammarSpec, ...] = tuple(
    publisher_grammar(producer, roles=tuple(_OWNERS[producer]),
                      plan_shape=partial(_plan_shape, producer), aggregate_shape=_aggregate_shape)
    for producer in FEATURES_PRODUCERS
)
FEATURES_GRAMMAR_SPECS = GRAMMAR_SPECS


def features_family_selection(plans: Mapping[str, dict]) -> dict[str, bool]:
    """Return the producer selection for coordinator construction."""
    return {producer: any(plans[producer]["enabled"].values()) for producer in FEATURES_PRODUCERS}


def _nominal_owner(role: str, owner: object) -> bool:
    module_name, class_name = _NOMINAL_OWNERS[role]
    # The graph constructs the runtime owners. Keep this module metadata-only
    # until an instance has been constructed by its original factory.
    module = sys.modules.get(module_name)
    return module is not None and type(owner) is vars(module).get(class_name)


def _observed_drain(role: str, owner: object) -> bool:
    attr = _RETAINED_DRAINS.get(role)
    if attr is None:
        # These roles rely on the original shutdown sequence supplied by I1.
        return True
    task = getattr(owner, attr)
    return (isinstance(task, asyncio.Task) and task.done() and not task.cancelled()
            and task.exception() is None
            and (role != "ha_client" or owner.get_health_status().retirements_pending == 0))


@dataclass(frozen=True, slots=True, repr=False, eq=False)
class _AttachedOwner:
    instance: object
    instance_sha256: str


@dataclass(slots=True, repr=False, eq=False)
class FeaturesFamilyPublisher:
    """Pin exact nominal owner instances across one feature row's lifecycle."""

    _publisher: GenericOutputPublisher
    _enabled: dict[str, bool]
    _owners: dict[str, _AttachedOwner] = field(default_factory=dict, init=False)
    _opening: set[str] = field(default_factory=set, init=False)
    _opened: set[str] = field(default_factory=set, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    _terminal: bool = field(default=False, init=False)

    def _matches_owner(self, role: str, owner: object) -> bool:
        bound = self._owners.get(role)
        return bound is not None and bound.instance is owner and _nominal_owner(role, owner)

    def matches(self, owners: Mapping[str, object]) -> bool:
        """Check the composition or shutdown graph against the attached instances."""
        return (owners.keys() == {role for role, selected in self._enabled.items() if selected}
                and all(self._matches_owner(role, owner) for role, owner in owners.items()))

    async def attach(self, role: str, owner: object) -> None:
        """Pin the exact owner after intent and before native work starts."""
        async with self._lock:
            base._require(not self._terminal and self._enabled.get(role) is True
                          and role not in self._owners and _nominal_owner(role, owner), "features_owner")
            if not self._owners:
                await self._publisher.attached()
            self._owners[role] = _AttachedOwner(owner, sha256(uuid4().bytes).hexdigest())

    async def opening(self, role: str, owner: object) -> None:
        """Write the opening boundary before invoking this owner's work."""
        async with self._lock:
            base._require(not self._terminal and role not in self._opening
                          and self._matches_owner(role, owner), "features_owner")
            if not self._opening:
                await self._publisher.opening()
            self._opening.add(role)

    async def opened(self, role: str, owner: object) -> None:
        """Confirm completion of the original opening or graph handoff."""
        async with self._lock:
            base._require(not self._terminal and role in self._opening
                          and self._matches_owner(role, owner), "features_owner")
            self._opened.add(role)

    async def drained(self, owners: Mapping[str, object]) -> None:
        """Publish terminal evidence after the original shutdown owners drain."""
        async with self._lock:
            base._require(not self._terminal and self.matches(owners) and self._opened == owners.keys()
                          and all(_observed_drain(role, owner) for role, owner in owners.items()),
                          "features_owner")
            instance_sha256 = {role: bound.instance_sha256 for role, bound in self._owners.items()}
            await self._publisher.drained(features_aggregate(self._publisher.producer, instance_sha256=instance_sha256))
            self._terminal = True


async def bind_features_family_publishers(
    coordinator: OutputRegistrationV2, plans: Mapping[str, dict],
) -> dict[str, FeaturesFamilyPublisher]:
    """Reserve all four rows on an initialized coordinator before construction.

    Supply the same plans to `features_family_selection` when building the
    coordinator. Attach the real graph owners, pair opening/opened around
    their original work and call drained after observing their original drains.
    """
    base._require(plans.keys() == set(FEATURES_PRODUCERS), "features_plan")
    selection = features_family_selection(plans)
    header = {"selected": coordinator._selected}
    validated = {}
    for spec in FEATURES_GRAMMAR_SPECS:
        producer = spec.producer
        base._require(coordinator._selected.get(producer) is selection[producer]
                      and coordinator._writer_grammars.get(producer) == spec.grammar_id, "features_selection")
        phase = "intent" if selection[producer] else "disabled"
        event = {"kind": spec.event_kind, "phase": phase, "plan": plans[producer]}
        validated[producer] = spec.decode_event(event, None, selection[producer], header)["plan"]

    await coordinator._run(coordinator.admission.revalidate)
    bindings = {}
    for producer, plan in validated.items():
        if selection[producer]:
            publisher = await coordinator.publisher_intent(producer, plan)
            bindings[producer] = FeaturesFamilyPublisher(publisher, plan["enabled"])
        else:
            await coordinator.publisher_disabled(producer, plan)
    return bindings
