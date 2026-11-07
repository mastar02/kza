"""Isolated real-graph and closed-controller phases for v1/v2 compatibility."""

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
    version = 1 if args.mode == "graph-v1" else 2
    patch, marker = pytest.MonkeyPatch(), len(MemoryClient._created)
    try:
        async with managed_application_graph(Path(args.state_root), patch, Clock(time.time()),
                                             control=control) as bundle:
            _require(bundle.graph.privacy is bundle.privacy, "original_privacy_owner")
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
        other = 2 if version == 1 else 1
        _require(not (directory.parents[2] / f"v{other}/runs" / ticket["run_id"]).exists(),
                 "writer_did_not_change_version")
        if version == 1:
            _require("src.memory.output_registration_v2" not in sys.modules, "old_writer_replaced")
        receipt_path = Path(args.admission_root) / "admission/receipts" / (ticket["run_id"] + ".json")
        receipt = json.loads(receipt_path.read_text())
        _require(receipt["child_pid"] == os.getpid() and receipt["status"] == "closed",
                 "real_child_receipt")
        _require(all(receipt[key] == header[key] for key in ("run_id", "binding_id", "generation")),
                 "receipt_epoch_join")
        return {"version": f"v{version}", "header": header, "journal": journal,
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
        _require({value["version"] for value in observed} == {"v1", "v2"}, "mixed_versions")
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
            if value["version"] == "v2":
                _require(len(value["instrumented_producers"]) == 7
                         and len(value["partial_producers"]) == 5, "instrumented_is_not_covered")
        unknown = [value for value in values if value["phase"] == "unknown"]
        if args.mode == "read-valid":
            _require(len(values) == 2 and not unknown, "valid_exact_histories")
        else:
            _require(len(values) == 4 and len(unknown) == 2, "negative_exact_histories")
            _require(any(value.get("version") == "v999" and value["reason"] == "unsupported_version"
                         for value in unknown), "unknown_version_refused")
            _require(any(value.get("run_id") == expectation["malformed_run_id"]
                         and value["reason"] == "invalid_or_unfinished_history" for value in unknown),
                     "malformed_known_version_refused")
            _require(all(field not in value for value in unknown
                         for field in ("header", "stores", "journal", "configuration", "material_observations")),
                     "unknown_has_no_positive_projection")
        _require(all(value[flag] is False for value in values
                     for flag in ("capability", "global_inventory_complete", "aggregate_copy_complete")),
                 "observation_is_not_authority")
        after = stage.read_admission_head(controller.root)
        _require(after == before, "reader_changed_head")
    return {"histories": values, "head_before": before, "head_after": after,
            "graph_constructed": False, "inherited_ticket": False,
            "controller_module": str(stage_path), "controller_sha256": _sha(stage_path),
            "controller_loader": type(stage.__spec__.loader).__name__}


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True,
                        choices=("graph-v1", "graph-v2", "read-valid", "read-invalid"))
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
