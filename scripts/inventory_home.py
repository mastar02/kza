#!/usr/bin/env python3
"""Read HA states/services into a private, unapproved stabilization inventory."""

import argparse
import asyncio
from datetime import UTC, datetime
import json
import os
from pathlib import Path

import aiohttp
from dotenv import dotenv_values


CONTROL_DOMAINS = {"light", "switch", "climate", "cover", "scene", "automation", "script"}
SENSITIVE_DOMAINS = {"lock", "alarm_control_panel"}
ALERT_DOMAINS = {"sensor", "binary_sensor"}
CAPABILITY_ATTRIBUTES = {
    "friendly_name", "supported_features", "supported_color_modes", "color_mode",
    "min_mireds", "max_mireds", "min_color_temp_kelvin", "max_color_temp_kelvin",
    "hvac_modes", "hvac_mode", "preset_modes", "fan_modes", "swing_modes",
    "min_temp", "max_temp", "target_temp_step", "temperature", "target_temp_low",
    "target_temp_high", "current_temperature", "current_position", "current_tilt_position",
    "device_class", "unit_of_measurement", "entity_id", "editable", "id",
}


def summarize_snapshot(states: list[dict], services: list[dict]) -> dict:
    """Describe available devices without granting or inferring authorization.

    Args:
        states: HA /api/states response or an equivalent private snapshot.
        services: HA /api/services response, including parameter schemas.

    Returns:
        Inventory for private review. Scene effects remain unverified even if
        the state lists members; membership does not prove nested effects.
    """
    if not isinstance(states, list) or not isinstance(services, list):
        raise ValueError("Snapshot must contain states and services lists")
    entities = []
    domains = CONTROL_DOMAINS | SENSITIVE_DOMAINS | ALERT_DOMAINS
    for state in states:
        entity_id = state["entity_id"]
        domain = entity_id.partition(".")[0]
        if domain not in domains:
            continue
        attributes = state.get("attributes", {})
        entity = {
            "entity_id": entity_id,
            "state": state.get("state"),
            "attributes": {k: v for k, v in attributes.items() if k in CAPABILITY_ATTRIBUTES},
            "selection": {"approved": False, "operations": [], "limits": {}},
        }
        if domain in SENSITIVE_DOMAINS:
            entity["restriction"] = "blocked_by_roadmap"
        if domain in {"scene", "automation", "script"}:
            entity["effects"] = {"verified": False, "actions": [], "reason": "definition_review_required"}
        entities.append(entity)
    return {
        "format": "kza-home-inventory-v1",
        "captured_at": datetime.now(UTC).isoformat(),
        "authorization": "none; requires explicit entity/operation/parameter review",
        "entities": sorted(entities, key=lambda item: item["entity_id"]),
        "services": [item for item in services if item.get("domain") in domains],
        "missing_evidence": [
            "area/entity/device registry assignments",
            "scene/automation/script definitions and all nested effects",
            "user selection of exact entities, operations and parameter bounds",
            "alert source usefulness and delivery configuration",
        ],
    }


async def capture_snapshot(url: str, token: str) -> tuple[list[dict], list[dict]]:
    """Fetch read-only HA endpoints without sending service calls."""
    headers = {"Authorization": f"Bearer {token}"}
    async with aiohttp.ClientSession(
        headers=headers, timeout=aiohttp.ClientTimeout(total=20),
    ) as session:
        async def fetch(endpoint: str) -> list[dict]:
            async with session.get(f"{url.rstrip('/')}/api/{endpoint}", allow_redirects=False) as response:
                if response.status != 200:
                    raise RuntimeError(f"HA inventory read failed: HTTP {response.status}")
                return await response.json()

        states, services = await asyncio.gather(fetch("states"), fetch("services"))
        return states, services


def main() -> None:
    """Write a new private inventory; never replace an existing file."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--snapshot", type=Path, help="Private JSON with states and services lists")
    source.add_argument("--env-file", type=Path, help="Private dotenv; defaults to exported HA variables")
    parser.add_argument("output", type=Path, help="Private destination outside tracked files (e.g. data/)")
    args = parser.parse_args()
    if args.snapshot:
        snapshot = json.loads(args.snapshot.read_text())
        inventory = summarize_snapshot(snapshot["states"], snapshot["services"])
    else:
        values = {**(dotenv_values(args.env_file) if args.env_file else {}), **os.environ}
        url = values.get("HOME_ASSISTANT_URL")
        token = values.get("HOME_ASSISTANT_TOKEN")
        if not url or not token:
            parser.error("HOME_ASSISTANT_URL and HOME_ASSISTANT_TOKEN are required")
        states, services = asyncio.run(capture_snapshot(url, token))
        inventory = summarize_snapshot(states, services)
    # Build the complete output before creating it, to avoid a partial private file.
    data = (json.dumps(inventory, ensure_ascii=False, indent=2) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as output:
        output.write(data)
    print(json.dumps({"entities": len(inventory["entities"]), "approved": 0}))


if __name__ == "__main__":
    main()
