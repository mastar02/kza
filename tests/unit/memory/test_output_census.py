"""Keep registration, nominal coverage and global predicates distinct."""
from copy import deepcopy

import pytest

from src.memory.output_census import InventoryReason, _census
from src.memory.output_publisher import ALL_MATRIX_PRODUCERS
from src.memory.output_registration_audio import AUDIO_GRAMMAR_SPECS
from src.memory.output_registration_features import FEATURES_GRAMMAR_SPECS
from src.memory.output_registration_memory import MEMORY_GRAMMAR_SPECS
from src.memory.output_registration_v2 import _SELECTED
from src.memory.output_registration_v4 import ALL_RUNTIME_EXTRA_GRAMMARS, WRITER_GRAMMARS


@pytest.fixture
def observed_run():
    """Use the states of a recovered run as admitted by the family codecs."""
    grammars = dict(WRITER_GRAMMARS) | {spec.producer: spec.grammar_id for spec in ALL_RUNTIME_EXTRA_GRAMMARS}
    selected = dict.fromkeys(grammars.keys() | (_SELECTED - {"lists", "reminders"}), True)
    stores = {producer: {"phase": "closed"} for producer in _SELECTED}
    stores.update({f"{producer}.{role}": {"phase": "closed"}
                   for producer in ("events", "suggestions") for role in ("eraser", "retention")})
    for spec in AUDIO_GRAMMAR_SPECS:
        stores[spec.producer] = {"phase": "drained", "aggregate": {
            "owner_counts": {"role": 1}, "owner_state": "drained", "owner_replaced": False,
        }}
    for spec in MEMORY_GRAMMAR_SPECS + FEATURES_GRAMMAR_SPECS:
        stores[spec.producer] = {"phase": "drained", "aggregate": {"owners": {
            "role": {"observation": "drained", "instance_sha256": "a" * 64},
        }}}
    for producer, evidence in {
        "accounts": {"native_drain_returned": True, "cleanup_uncertain": False},
        "configuration": {"writer_returned": True, "cleanup_uncertain": False},
        "training": {"observation": "instance_reported"},
        "offline_copies": {"observation": "catalog_observed"},
    }.items():
        stores[producer] = {"phase": "drained", "aggregate": evidence}
    for producer in ("offline_capture_tools", "offline_exports_training", "build_forensic_outputs", "audio_health"):
        selected[producer] = False
        stores[producer] = {"phase": "disabled"}
    return {
        "phase": "observed", "version": "v4", "run_id": "d" * 32,
        "header": {"grammars": grammars, "selected": selected},
        "stores": stores, "uncovered_producers": list(ALL_MATRIX_PRODUCERS),
        "material_observations": ({"kind": "composition", "instrumented": ALL_MATRIX_PRODUCERS},),
    }


def test_local_coverage_does_not_certify_external_inventory_or_erasure(observed_run):
    snapshot = deepcopy(observed_run)
    census = _census(observed_run)
    assert census.registered_producers == 32
    assert census.producer_coverage_complete and not census.uncovered_producers
    assert not census.destination_enumeration_complete
    assert not census.representation_classification_complete
    assert not census.subject_erasure_complete
    assert not census.global_inventory_complete
    assert set(census.inventory_reasons) == {InventoryReason.EXTERNAL_PRIVATE, InventoryReason.EXTERNAL_UNCONFIRMED}
    assert observed_run == snapshot


@pytest.mark.parametrize(("producer", "metadata"), [
    ("offline_copies", {"observation": "catalog_unresolved"}),
    ("training", {"observation": "blocked_instance"}),
    ("audio_input", {"owner_state": "drained", "owner_counts": {"role": 1}, "owner_replaced": True}),
    ("capture", {"owner_state": "drained", "owner_counts": {"role": 1}, "owner_replaced": False,
                 "_capture_unreviewed": True}),
    ("model_resources", {"owners": {"role": {"instance_sha256": None}}}),
    ("configuration", {}),
])
def test_degraded_terminal_record_keeps_row_registered_but_uncovered(observed_run, producer, metadata):
    observed_run["uncovered_producers"] = []
    observed_run["stores"][producer]["aggregate"] = metadata
    census = _census(observed_run)
    assert census.registered_producers == 32 and census.uncovered_producers == (producer,)
    assert not census.producer_coverage_complete
    assert not census.global_inventory_complete
    assert (InventoryReason.PENDING_CLEANUP in census.inventory_reasons) == (
        producer in ("training", "offline_copies")
    )


def test_missing_receipt_or_unknown_codec_never_confers_coverage(observed_run):
    missing = _census({"run_id": observed_run["run_id"], "phase": "unknown", "version": "v4"})
    assert missing.registered_producers == 0 and missing.uncovered_producers == ALL_MATRIX_PRODUCERS
    observed_run["header"]["grammars"]["audio_input"] = "audio_input_publisher/99"
    unknown = _census(observed_run)
    assert unknown.registered_producers == 31 and unknown.uncovered_producers == ("audio_input",)
