"""Translate the durable Python scheduling and household stores."""

import hashlib
import json
import sqlite3
from collections import defaultdict

from privacy import Privacy
from storage import Catalog, decode, encode, identifier, millis, require, timestamp

TABLES = {
    "timers", "reminders", "reminder_home_actions", "reminder_notices", "routines",
    "routine_runs", "routine_run_actions", "lists", "list_items", "alerts", "alert_preferences",
}
PRIORITIES = {"critical": 0, "high": 1, "medium": 2, "low": 3}
REMINDER_STATES = {"active": "pending", "fired": "completed", "cancelled": "cancelled",
                   "pending": "pending", "completed": "completed"}
ACTION_STATES = {"confirmed": "succeeded", "sending": "uncertain", "failed": "failed",
                 "uncertain": "uncertain"}


def name_key(name: str) -> str:
    return " ".join(name.split()).lower()


def ordered(rows: list[dict], key: str = "id") -> list[dict]:
    return sorted(rows, key=lambda row: str(row[key]))


def insert(db: sqlite3.Connection, table: str, values: tuple) -> int:
    args = ",".join("?" for _ in values)
    return db.execute(f"INSERT INTO {table} VALUES ({args})", values).lastrowid


def legacy_id(row: dict) -> int | None:
    value = row.get("id")
    return value if type(value) is int else None


def migrate_household(source: Catalog, db: sqlite3.Connection, privacy: Privacy,
                      alerts: list[dict] | None, now: float) -> dict[str, int]:
    for row in ordered(source.rows("timers")):
        name = row.get("name", row.get("label"))
        room = row.get("room", row.get("delivery_room")) or ""
        deadline = row.get("deadline") if "deadline" in row else optional_time(row.get("ends_at"))
        remaining = (row.get("remaining_ms") if "remaining_ms" in row else
                     optional_time(row.get("remaining_s")))
        state = row["state"]
        if state in {"completed", "cancelled"}:
            deadline, remaining = None, None
        insert(db, "timers", (legacy_id(row), identifier(name), name_key(name), room, state,
                              deadline, remaining, row.get("ring_until")))
        source.count("timers", "migrated")
    recurring = migrate_reminders(source, db, privacy)
    constrained = migrate_routines(source, db, now)
    migrate_lists(source, db, privacy)
    migrate_alerts(source, db, alerts)
    return {"recurring_reminders_imported_as_single_occurrence": recurring,
            "routines_with_legacy_schedule": constrained[0],
            "routines_disabled_due_to_legacy_conditions": constrained[1]}


def optional_time(value: object) -> int | None:
    return None if value is None else timestamp(value)


def task_reason(privacy: Privacy, row: dict, owner: str | None) -> str | None:
    if owner is None:
        return None
    return privacy.reason((row.get("conversation_id"),), (owner,))


def home_action(action: dict) -> dict:
    entity = identifier(action["entity_id"])
    service = identifier(action.get("operation", action.get("service")))
    data = action.get("data") or {}
    if "operation" in action:
        data = {key: value for key, value in action.items() if key not in {"operation", "entity_id"}}
    require(type(data) is dict, "invalid legacy home action parameters")
    if service == "brightness" and "brightness" in data:
        data["brightness_pct"] = round(data.pop("brightness") * 100 / 255)
    if action.get("domain"):
        require(entity.partition(".")[0] == action["domain"], "home action domain mismatch")
        service = action["domain"] + "." + service
    return {"entity_id": entity, "service": service, "data": data}


def migrate_reminders(source: Catalog, db: sqlite3.Connection, privacy: Privacy) -> int:
    actions = source.rows("reminder_home_actions")
    receipts = {row["request_id"]: row for row in actions if "request_id" in row}
    require(len(receipts) == sum("request_id" in row for row in actions), "duplicate reminder receipt")
    notices = defaultdict(list)
    used = set()
    for key, row in receipts.items():
        body = decode(row["body"])
        require(isinstance(body, dict), "invalid reminder receipt")
        if body.get("kind") in {"voice_notice", "grouped_notice"}:
            notices[(body["reminder_id"], body["trigger_at"])].append((key, row, body))
    ids, excluded = {}, {}
    recurring = 0
    for row in ordered(source.rows("reminders")):
        key = row["id"]
        owner = row.get("owner", row.get("user_id"))
        scope = row.get("scope", "private" if owner else "shared")
        require(scope in {"private", "shared"} and (scope == "shared" or bool(owner)),
                "invalid reminder privacy scope")
        owner = None if scope == "shared" else identifier(owner)
        reason = task_reason(privacy, row, owner)
        trigger = row.get("trigger_at")
        specifications = decode(row["ha_actions"]) if row.get("ha_actions") else []
        require(type(specifications) is list, "invalid legacy reminder actions")
        attempts = notices.get((key, trigger), [])
        action_receipts = [
            "reminder:" + hashlib.sha256(json.dumps(
                [key, trigger, index], separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            for index in range(len(specifications))
        ]
        if reason:
            source.count("reminders", reason)
            for receipt in set(action_receipts) | {item[0] for item in attempts}:
                if receipt in receipts:
                    source.count("reminder_home_actions", reason)
                    used.add(receipt)
            excluded[key] = reason
            continue
        state = REMINDER_STATES[row["state"]]
        due = row["due_at"] if "due_at" in row else millis(trigger)
        terminal_notice = next((receipt for _, receipt, _ in attempts
                                if receipt["status"] in {"confirmed", "sending", "uncertain"}), None)
        if state == "pending" and terminal_notice is not None:
            state = "completed"
        ids[key] = insert(db, "reminders", (
            legacy_id(row), row.get("message", row.get("text")), owner,
            row.get("room", row.get("delivery_room")) or "", due, state,
        ))
        source.count("reminders", "migrated")
        recurring += bool(row.get("recurrence"))
        for index, action in enumerate(specifications):
            receipt_key = action_receipts[index]
            receipt = receipts.get(receipt_key)
            status = ("pending" if state == "pending" and owner is None else
                      "cancelled" if state == "cancelled" else "skipped")
            if receipt:
                status = ("late_not_executed" if receipt.get("reason") == "late_not_executed"
                          else ACTION_STATES[receipt["status"]])
                source.count("reminder_home_actions", "migrated")
                used.add(receipt_key)
            insert(db, "reminder_home_actions", (ids[key], index, encode(home_action(action)), status))
        if "due_at" not in row:
            if terminal_notice:
                status = {"confirmed": "delivered", "uncertain": "uncertain",
                          "sending": "uncertain"}[terminal_notice["status"]]
                if terminal_notice.get("reason") == "dismissed":
                    status = "dismissed"
            else:
                status = {"pending": "pending", "completed": "delivered", "cancelled": "cancelled"}[state]
            count = sum(body["kind"] == "voice_notice" for _, _, body in attempts)
            insert(db, "reminder_notices",
                   (ids[key], status, due if status == "pending" else None, None, count, count, 0))
            source.count("reminder_home_actions", "migrated", int(bool(attempts)))
            source.count("reminder_home_actions", "discarded", max(len(attempts) - 1, 0))
            used.update(item[0] for item in attempts)
    source.count("reminder_home_actions", "discarded", len(receipts.keys() - used))
    for table in ("reminder_home_actions", "reminder_notices"):
        rows = actions if table == "reminder_home_actions" else source.rows(table)
        for row in rows:
            if "reminder_id" not in row:
                continue
            key = row["reminder_id"]
            if key in excluded:
                source.count(table, excluded[key])
                continue
            require(key in ids, "orphan reminder record")
            if table == "reminder_home_actions":
                values = (ids[key], row["position"], row["command"], row["status"])
            else:
                values = (ids[key], *(row[column] for column in (
                    "status", "next_attempt_at", "claimed_at", "attempts",
                    "round_attempts", "fallback_rounds",
                )))
            insert(db, table, values)
            source.count(table, "migrated")
    return recurring


def routine_steps(definition: dict) -> str:
    return encode([{"kind": "home", "action": home_action(action)} for action in definition["actions"]])


def migrate_routines(source: Catalog, db: sqlite3.Connection, now: float) -> tuple[int, int]:
    ids = {}
    scheduled = disabled = 0
    for row in sorted(source.rows("routines"), key=lambda x: str(x.get("routine_id", x.get("id")))):
        key = row.get("routine_id", row.get("id"))
        if "definition_json" in row:
            definition = decode(row["definition_json"])
            name = definition["name"]
            conditions = bool(definition.get("conditions"))
            enabled = int(definition["enabled"] and not conditions)
            disabled += bool(definition["enabled"] and conditions)
            scheduled += bool(definition.get("schedule"))
            steps = routine_steps(definition)
        else:
            name, enabled, steps = row["name"], row["enabled"], row["steps"]
        ids[key] = insert(db, "routines", (legacy_id(row), identifier(name), name_key(name),
                                          enabled, row["revision"], steps, row.get("deleted_at")))
        source.count("routines", "migrated")
    actions = defaultdict(list)
    for row in source.rows("routine_run_actions"):
        actions[row["run_id"]].append(row)
    for row in sorted(source.rows("routine_runs"), key=lambda x: str(x.get("run_id", x.get("id")))):
        key, routine = row.get("run_id", row.get("id")), row["routine_id"]
        run_source = row["source"]
        if "snapshot_json" in row:
            snapshot = decode(row["snapshot_json"])
            if run_source == "household_voice" and snapshot["routine_id"] != routine:
                # A brake receipt has a separate namespace and performs no steps.
                source.count("routine_runs", "discarded")
                source.count("routine_run_actions", "discarded", len(actions.pop(key, [])))
                continue
            # A running or ambiguous Python run cannot resume effects after cutover.
            if run_source == "household_voice":
                run_source = "voice"
            status = {"running": "interrupted", "uncertain": "interrupted"}.get(row["status"], row["status"])
            started = timestamp(row.get("scheduled_for") or snapshot["created_at"])
            indexed = actions.pop(key, [])
            next_step = sum(item["status"] == "confirmed" for item in indexed)
            source.count("routine_run_actions", "discarded", len(indexed))
            steps = routine_steps(snapshot)
            finished, claimed, due = max(started, millis(now)), 0, None
            if routine not in ids:
                # Python physically deletes idle routines but retains their runs.
                name = identifier(snapshot["name"])
                ids[routine] = insert(db, "routines", (
                    None, name, name_key(name), 0, snapshot["revision"], steps, millis(now),
                ))
        else:
            status, started, finished, steps, next_step, claimed, due = (
                row[column] for column in ("status", "started_at", "finished_at", "steps", "next_step",
                                           "step_claimed", "step_due_at")
            )
        require(routine in ids, "orphan routine run")
        insert(db, "routine_runs", (
            legacy_id(row), ids[routine], row["occurrence_key"], run_source, started,
            finished, status, steps, next_step, claimed, due,
        ))
        source.count("routine_runs", "migrated")
    if actions:
        source.count("routine_run_actions", "discarded", sum(map(len, actions.values())))
    return scheduled, disabled


def migrate_lists(source: Catalog, db: sqlite3.Connection, privacy: Privacy) -> None:
    ids, excluded = {}, {}
    for row in ordered(source.rows("lists")):
        owner = row.get("owner", row.get("owner_id"))
        owner_type = row.get("owner_type", "user" if owner else "shared")
        require(owner_type in {"shared", "user"} and (owner_type == "shared" or bool(owner)),
                "invalid list privacy scope")
        owner = identifier(owner) if owner_type == "user" else None
        reason = task_reason(privacy, row, owner)
        if reason:
            excluded[row["id"]] = reason
            source.count("lists", reason)
            continue
        name = identifier(row["name"])
        ids[row["id"]] = insert(db, "lists", (legacy_id(row), name, name_key(name), owner,
                                               row.get("ha_entity"), row.get("revision", 1)))
        source.count("lists", "migrated")
    items = source.rows("list_items")
    for row in sorted(items, key=lambda item: (item.get("created_at", 0), str(item["id"]))):
        key = row["list_id"]
        if key in excluded:
            source.count("list_items", excluded[key])
            continue
        require(key in ids, "orphan household list item")
        insert(db, "list_items", (legacy_id(row), ids[key], row["text"],
                                   row.get("ha_uid"), row["completed"]))
        source.count("list_items", "migrated")


def migrate_alerts(source: Catalog, db: sqlite3.Connection, alerts: list[dict] | None) -> None:
    rows = source.rows("alerts")
    require(not rows or alerts is None, "ambiguous alert authorities")
    if alerts is not None:
        rows = alerts
        source.count("alerts", "source", len(rows))
    for row in rows:
        key = row.get("condition_key")
        if key is None:
            parts = (row.get("alert_type"), row.get("source"), row.get("condition") or row.get("alert_id"))
            key = encode(parts)
        priority = row["priority"]
        if isinstance(priority, str):
            priority = PRIORITIES[priority.lower()]
        opened = row["opened_at"] if "opened_at" in row else timestamp(row["timestamp"])
        exported = "opened_at" not in row
        last_seen = (optional_time(row.get("last_seen_at")) if exported else row.get("last_seen_at"))
        resolved = optional_time(row.get("resolved_at")) if exported else row.get("resolved_at")
        acknowledged = (optional_time(row.get("processed_at")) if exported
                        else row.get("acknowledged_at"))
        expires = optional_time(row.get("expires_at")) if exported else row.get("expires_at")
        insert(db, "alerts", (legacy_id(row), key, priority, row["message"], opened,
                                opened if last_seen is None else last_seen, expires, acknowledged,
                                resolved, row.get("revision", 1)))
        source.count("alerts", "migrated")
    for row in source.rows("alert_preferences"):
        if row.get("key", "alert_zones") != "alert_zones":
            source.count("alert_preferences", "discarded")
            continue
        zones = decode(row.get("zones", row.get("value_json")))
        require(isinstance(zones, list) and all(isinstance(zone, str) for zone in zones),
                "invalid alert zone preferences")
        insert(db, "alert_preferences", (1, encode(zones), row["revision"]))
        source.count("alert_preferences", "migrated")
