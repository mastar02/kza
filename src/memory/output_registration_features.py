"""Closed T024 grammars for the feature, integration, dashboard and latency owners.

The composition root supplies selected roles before opening and independently
observed drained roles at shutdown. These builders only produce journal metadata.
"""

from collections.abc import Collection
from functools import partial
from hashlib import sha256
from pathlib import PurePath
import re
from urllib.parse import urlsplit

from src.memory import output_registration as base
from src.memory.output_publisher import GrammarSpec, publisher_grammar, publisher_plan

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
        "browser_delivery": {"kind": "transport_session", "owner": "dashboard_clients"},
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


def features_aggregate(producer: str, *, drained_roles: Collection[str]) -> dict:
    """Build terminal observations from original owners whose drain completed."""
    return {"drained": _role_flags(producer, drained_roles)}


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
    base._require(base._encoded(aggregate) == base._encoded({"drained": plan["enabled"]}), "features_aggregate")


GRAMMAR_SPECS: tuple[GrammarSpec, ...] = tuple(
    publisher_grammar(producer, roles=tuple(_OWNERS[producer]),
                      plan_shape=partial(_plan_shape, producer), aggregate_shape=_aggregate_shape)
    for producer in FEATURES_PRODUCERS
)
