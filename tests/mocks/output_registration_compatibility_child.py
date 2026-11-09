"""Isolated real-graph and closed-controller phases for V1 and current (V4) compatibility."""

import argparse
import asyncio
import hashlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def _require(value: object, reason: str) -> None:
    if not value:
        raise AssertionError("output_compatibility_" + reason)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bootstrap_module():
    path = Path(__file__).with_name("compiled_restore_child.py")
    spec = importlib.util.spec_from_file_location("_output_compatibility_bootstrap", path)
    _require(spec is not None and spec.loader is not None, "bootstrap_spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def _account_admin(graph: object) -> dict:
    """Create, update and delete one synthetic non-admin account through the real app.

    Uses the existing synthetic fixture root token and the original login,
    routes and admin owner; nothing seeds or mutates the UserManager directly.
    """
    from httpx import ASGITransport, AsyncClient
    from tests.mocks.restore_application_graph import TOKEN

    dashboard = graph.dashboard
    users = dashboard.user_manager
    _require(dashboard is not None and users is not None and users.voice_review_managed is True,
             "account_admin_owner")
    path = Path(users.data_path)
    before = set(users._users)

    def on_disk() -> dict:
        return {row["user_id"]: row["permission_level"] for row in json.loads(path.read_text())["users"]}

    async with AsyncClient(transport=ASGITransport(app=dashboard.app), base_url="http://testserver") as client:
        login = await client.post("/api/admin/auth/login", json={"token": TOKEN})
        _require(login.status_code == 200, "account_admin_login")
        created = await client.post("/api/admin/users",
                                    json={"name": "Synthetic Compat Guest", "permission_level": "guest"})
        _require(created.status_code == 200, "account_admin_create")
        user_id = created.json()["user_id"]
        _require(set(users._users) == before | {user_id} and user_id not in before, "account_create_ram")
        level = users._users[user_id].permission_level
        _require(level.name == "GUEST" and on_disk()[user_id] == level.value, "account_create_file")
        updated = await client.put(f"/api/admin/users/{user_id}/permissions", json={"permission_level": "teen"})
        _require(updated.status_code == 200, "account_admin_update")
        level = users._users[user_id].permission_level
        _require(level.name == "TEEN" and on_disk()[user_id] == level.value, "account_update_file_ram")
        deleted = await client.delete(f"/api/admin/users/{user_id}")
        _require(deleted.status_code == 200, "account_admin_delete")
        _require(set(users._users) == before and user_id not in on_disk(), "account_delete_file_ram")
    info = os.lstat(path)
    return {"operations": ["create", "update", "delete"], "statuses": [created.status_code,
            updated.status_code, deleted.status_code], "accounts_before": len(before),
            "accounts_after": len(users._users),
            "final_identity": {"device": info.st_dev, "inode": info.st_ino}}


async def _graph(args: argparse.Namespace, control: dict, guard: object) -> dict:
    import pytest
    import src.main as main
    from src.core import runtime_admission
    from src.memory import output_registration
    from tests.mocks.memory_graph import Clock, MemoryClient
    from tests.mocks.managed_slot_graph import managed_application_graph
    from tests.mocks.slot_rollback_child import _resources

    # Both selected releases have their own managed main factory. The separately
    # pinned bridge must never replace the old producer-registration writer.
    _require(hasattr(main, "verify_runtime_admission"), "original_managed_factory")
    ticket = runtime_admission._ticket()
    _require(type(ticket) is dict, "actual_ticket_missing")
    _require(Path(ticket["admission_root"]).resolve() == Path(args.admission_root), "ticket_root")
    version = {"graph-v1": 1, "graph-current": 4}[args.mode]
    patch, marker = pytest.MonkeyPatch(), len(MemoryClient._created)
    try:
        async with managed_application_graph(Path(args.state_root), patch, Clock(time.time()),
                                             control=control) as bundle:
            _require(bundle.graph.privacy is bundle.privacy, "original_privacy_owner")
            # Plan step 5: only the current graph exercises the account publisher.
            admin = await _account_admin(bundle.graph) if version == 4 else None
            guard.audit()
        MemoryClient.close_since(marker)
        resources = _resources(bundle)
        directory = (Path(args.admission_root) / "admission/output-inventory"
                     / f"v{version}/runs" / ticket["run_id"])
        header, journal, stores, _ = output_registration._chain(directory)
        _require(header["schema_version"] == version, "real_writer_version")
        _require(all(header[key] == ticket[key] for key in ("run_id", "binding_id", "generation")),
                 "epoch_ticket_join")
        _require(header["state_root"] == str(Path(args.state_root).parents[1]), "epoch_state")
        _require(all(stores[name]["phase"] == "closed" for name in ("lists", "reminders")),
                 "original_store_close")
        for other in {1, 2, 3, 4} - {version}:
            _require(not (directory.parents[2] / f"v{other}/runs" / ticket["run_id"]).exists(),
                     "writer_did_not_change_version")
        if version == 1:
            _require("src.memory.output_registration_v2" not in sys.modules, "old_writer_replaced")
        if version < 3:
            # The genuine historical releases never load the V3 codec.
            _require("src.memory.output_registration_v3" not in sys.modules, "historical_writer_promoted")
        if version < 4:
            _require("src.memory.output_registration_v4" not in sys.modules, "historical_writer_promoted")
        if version == 4:
            # The current writer reuses the frozen V3 account grammar inside V4.
            _require("src.memory.output_registration_v3" in sys.modules
                     and "src.memory.output_registration_v4" in sys.modules, "current_writer_codecs")
            _require(header["grammars"] == {"accounts": "account_publisher/1",
                                            "configuration": "configuration_publisher/1"}, "current_grammars")
            accounts = stores.get("accounts", {})
            events = [event for event in (output_registration._read(path)[0]
                                          for path in sorted(directory.glob("[0-9]*.json")))
                      if event["kind"] == "account_publisher"]
            _require([event["phase"] for event in events] == ["intent", "attached", "opening", "drained"],
                     "account_publisher_phases")
            _require(accounts.get("phase") == "drained" and events[-1]["aggregate"] == accounts["aggregate"] == {
                "attempts": 3, "committed_replacements": 3, "clean_failed_attempts": 0,
                "last_committed_identity": admin["final_identity"], "native_drain_returned": True,
                "open_temp_count": 0, "cleanup_uncertain": False}, "account_publisher_drained")
            raw = b"".join(path.read_bytes() for path in sorted(directory.iterdir()) if path.is_file())
            _require(b"Synthetic Compat Guest" not in raw and b"teen" not in raw, "account_journal_private")
            # No withdrawal is queued in this phase: the registered owner drains no_pending.
            configuration = stores.get("configuration", {})
            phases = [event["phase"] for event in (output_registration._read(path)[0]
                                                   for path in sorted(directory.glob("[0-9]*.json")))
                      if event["kind"] == "configuration_publisher"]
            _require(phases == ["intent", "attached", "drained"] and configuration.get("phase") == "drained"
                     and configuration["aggregate"]["outcome"] == "no_pending", "configuration_publisher_drained")
        receipt_path = Path(args.admission_root) / "admission/receipts" / (ticket["run_id"] + ".json")
        receipt = json.loads(receipt_path.read_text())
        _require(receipt["child_pid"] == os.getpid() and receipt["status"] == "closed",
                 "real_child_receipt")
        _require(all(receipt[key] == header[key] for key in ("run_id", "binding_id", "generation")),
                 "receipt_epoch_join")
        return {"version": f"v{version}", "header": header, "journal": journal, "account_admin": admin,
                "resources": resources, "receipt": receipt, "receipt_sha256": _sha(receipt_path),
                "factory": "src.main.build_application", "bridge_supplied_writer": False,
                "fixture_scope": "actual managed graph plus original _bundle initialization and original owned close"}
    finally:
        MemoryClient.close_since(marker)
        patch.undo()


def _read(args: argparse.Namespace) -> dict:
    # Imports satisfy the existing sourceless guard's public-module audit; this
    # phase does not construct an ApplicationGraph or install a RuntimeAdmission.
    import src.main
    import src.application
    from src.core import runtime_admission
    from src.memory.output_registration import read_output_inventory

    _require("KZA_RUNTIME_ADMISSION" not in os.environ, "reader_inherited_ticket")
    expectation = json.loads(Path(args.expectation).read_text())
    stage = runtime_admission._stage()
    stage_path = Path(stage.__file__).resolve()
    _require(stage_path == Path(args.runtime) / "scripts/stage_runtime.pyc", "actual_reader_controller")
    _require(stage.__spec__ is not None
             and isinstance(stage.__spec__.loader, importlib.machinery.SourcelessFileLoader),
             "reader_controller_sourceless_loader")
    with stage.AdmissionController(Path(args.admission_root)) as controller:
        before = stage.read_admission_head(controller.root)
        _require(before == expectation["head"] and before["status"] == "closed", "reader_closed_head")
        values = read_output_inventory(controller)
        observed = [value for value in values if value["phase"] == "observed"]
        _require(len(observed) == 2, "both_real_histories")
        _require(sorted(value["version"] for value in observed) == ["v1", "v4"], "mixed_versions")
        for expected in expectation["runs"]:
            matched = [value for value in observed if value["run_id"] == expected["header"]["run_id"]]
            _require(len(matched) == 1, "unique_run")
            value = matched[0]
            _require(value["version"] == expected["version"] and value["header"] == expected["header"],
                     "actual_history_header")
            header = value["header"]
            binding = stage.read_binding(controller.root, header["binding_id"])
            receipt_path = controller.root / "admission/receipts" / (header["run_id"] + ".json")
            receipt = json.loads(receipt_path.read_text())
            _require(_sha(receipt_path) == expected["receipt_sha256"] and receipt == expected["receipt"],
                     "immutable_actual_receipt")
            _require(binding["state_root"] == header["state_root"]
                     and binding["state_identity"] == header["state_identity"], "binding_state_join")
            _require(receipt["status"] == "closed"
                     and set(receipt["resources"]) == stage._CLOSED_RESOURCES
                     and all(item is True for item in receipt["resources"].values()), "closed_resource_receipt")
            _require(all(receipt[key] == header[key] for key in ("run_id", "binding_id", "generation")),
                     "run_binding_generation_join")
            _require(all(value["stores"][name]["phase"] == "closed" for name in ("lists", "reminders")),
                     "closed_registered_stores")
            _require(len(value["required_producers"]) == 32 and len(value["uncovered_producers"]) == 30,
                     "partial_scope_preserved")
            if value["version"] == "v4":
                # V4 adds exactly the account and configuration publishers to the seven V2 families.
                _require(len(value["instrumented_producers"]) == 9 and len(value["partial_producers"]) == 7
                         and {"accounts", "configuration"} <= set(value["instrumented_producers"])
                         and {"accounts", "configuration"} <= set(value["partial_producers"]),
                         "instrumented_is_not_covered")
                _require(value["stores"]["accounts"]["phase"] == "drained", "v3_account_publisher_drained")
                _require(value["stores"]["configuration"]["phase"] == "drained", "configuration_publisher_drained")
                policies = [event for event in value["material_observations"] if event["kind"] == "delegation_policy"]
                _require(len(policies) == 1 and policies[0]["scope_schema_version"] == 1,
                         "single_original_running_policy")
                policy = policies[0]
                _require(hashlib.sha256(stage._canonical_json(policy["policy"]) + b"\n").hexdigest()
                         == policy["ownership_sha256"] == binding["ownership_sha256"],
                         "original_running_policy_hash")
                delegated = [event for event in value["material_observations"]
                             if event["kind"] == "delegated_target" and event["status"] == "attached"]
                _require({event["plan"]["producer"] for event in delegated} == {"events", "suggestions"},
                         "both_historical_families_registered")
                actual = expectation["delegations"][value["version"]]
                record = stage._read_json(controller.root / "admission/delegations" / (actual["id"] + ".json"))
                source = stage.read_binding(controller.root, actual["source_closed_binding"])
                closure = stage._closure_proof(controller.root, actual["source_closed_binding"])
                for event in delegated:
                    scope = event["scope"]
                    target = record["targets"][event["id"]]
                    _require(event["id"] in actual["target_ids"]
                             and scope["scope_schema_version"] == 1
                             and scope["delegation_sha256"] == actual["id"] == stage._digest(record)
                             and scope["source_binding"] == actual["source_closed_binding"]
                             and scope["source_close_sha256"] == stage._digest(closure)
                             and scope["source_state_root"] == source["state_root"]
                             and scope["source_state_identity"] == source["state_identity"]
                             and record["destination_state"] == header["state_root"] == actual["destination_state"]
                             and event["plan"]["path"] == target["descriptor"]["path"]
                             and event["plan"]["producer"] == target["participant"],
                             "actual_delegated_target_source_and_close_join")
        unknown = [value for value in values if value["phase"] == "unknown"]
        if args.mode == "read-valid":
            _require(len(values) == 2 and not unknown, "valid_exact_histories")
        else:
            _require(len(values) == 6 and len(unknown) == 4, "negative_exact_histories")
            _require(any(value.get("version") == "v999" and value["reason"] == "unsupported_version"
                         for value in unknown), "unknown_version_refused")
            for version, run_id in expectation["malformed_runs"].items():
                _require([value.get("version") for value in unknown if value.get("run_id") == run_id
                          and value["reason"] == "invalid_or_unfinished_history"] == [version],
                         "malformed_known_version_refused")
            _require(all(field not in value for value in unknown
                         for field in ("header", "stores", "journal", "configuration", "material_observations")),
                     "unknown_has_no_positive_projection")
        _require(all(value[flag] is False for value in values
                     for flag in ("capability", "global_inventory_complete", "aggregate_copy_complete")),
                 "observation_is_not_authority")
        after = stage.read_admission_head(controller.root)
        _require(after == before, "reader_changed_head")
    versions = {value["run_id"]: value["version"] for value in observed}
    return {"histories": values, "head_before": before, "head_after": after,
            "version_by_run": versions,
            "graph_constructed": False, "inherited_ticket": False,
            "controller_module": str(stage_path), "controller_sha256": _sha(stage_path),
            "controller_loader": type(stage.__spec__.loader).__name__}


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True,
                        choices=("graph-v1", "graph-current", "read-valid", "read-invalid"))
    for flag in ("control", "runtime", "state-root", "evidence-root", "admission-root"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--expectation")
    args = parser.parse_args()
    bootstrap = _bootstrap_module()
    control, release, guard, hits = bootstrap._bootstrap(args)
    own = Path(__file__).resolve().relative_to(Path(control["island_root"])).as_posix()
    _require(control["island_files"].get(own) == _sha(Path(__file__)), "entry_hashlisted")
    observations = (asyncio.run(_graph(args, control, guard)) if args.mode.startswith("graph-")
                    else _read(args))
    origins = guard.audit()
    _require(not hits, "forbidden_sdk_or_hardware_call")
    for relative, digest in release["files"].items():
        _require(_sha(Path(args.runtime) / relative) == digest, "artifact_changed")
    _require(_sha(Path(args.runtime) / "release.json") == control["release_sha256"], "release_changed")
    for relative, digest in control["island_files"].items():
        _require(_sha(Path(control["island_root"]) / relative) == digest, "helper_changed")
    result = {"schema_version": 1, "mode": args.mode, "pid": os.getpid(),
              "source_commit": release["git_commit"], "release_sha256": control["release_sha256"],
              "runtime_root": args.runtime, "state_root": args.state_root,
              "module_origins": origins, "observations": observations,
              "sys_flags": {name: int(getattr(sys.flags, name)) for name in
                            ("isolated", "no_site", "no_user_site", "ignore_environment")},
              "dont_write_bytecode": bool(sys.dont_write_bytecode)}
    bootstrap._atomic_json(Path(args.evidence_root) / (args.mode + "-result.json"), result)
    print(json.dumps({"mode": args.mode, "source_commit": release["git_commit"], "pid": os.getpid()}))


if __name__ == "__main__":
    _main()
