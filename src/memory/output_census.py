"""Closed census of a recovered V4 run and its local producer lifetimes."""

from dataclasses import dataclass
from enum import StrEnum

from src.memory.output_publisher import ALL_MATRIX_PRODUCERS
from src.memory.output_registration import read_output_inventory
from src.memory.output_registration_audio import AUDIO_GRAMMAR_SPECS
from src.memory.output_registration_features import FEATURES_GRAMMAR_SPECS
from src.memory.output_registration_memory import MEMORY_GRAMMAR_SPECS
from src.memory.output_registration_v2 import _SELECTED
from src.memory.output_registration_v4 import ALL_RUNTIME_EXTRA_GRAMMARS, WRITER_GRAMMARS


class InventoryReason(StrEnum):
    """Closed vocabulary for unresolved destinations or obligations."""

    EXTERNAL_PRIVATE = "external_private"
    EXTERNAL_UNCONFIRMED = "external_unconfirmed"
    PENDING_CLEANUP = "pending_cleanup"


@dataclass(frozen=True, slots=True)
class OutputCensus:
    """Predicates and gaps for one receipt-verified runtime epoch."""

    run_id: str
    registered_producers: int
    producer_coverage_complete: bool
    destination_enumeration_complete: bool
    representation_classification_complete: bool
    subject_erasure_complete: bool
    global_inventory_complete: bool
    uncovered_producers: tuple[str, ...]
    inventory_reasons: tuple[InventoryReason, ...]


_GRAMMARS = dict(WRITER_GRAMMARS) | {spec.producer: spec.grammar_id for spec in ALL_RUNTIME_EXTRA_GRAMMARS}
_AUDIO = {spec.producer for spec in AUDIO_GRAMMAR_SPECS}
_IDENTITIES = {spec.producer for spec in MEMORY_GRAMMAR_SPECS + FEATURES_GRAMMAR_SPECS}
_OFFLINE_EVIDENCE = {
    "training": {"instance_reported"},
    "offline_copies": {"catalog_observed"},
    "offline_capture_tools": {"native_retired", "resource_session_reported"},
    "offline_exports_training": set(),  # The invocation declaration carries no owner receipt.
    "build_forensic_outputs": {"declared_manifests_observed"},
}
# These boundaries survive disabled selections because prior runs and
# third-party retention have independent owners. Codes are intentionally
# detached from endpoint addresses, paths and subject identifiers.
_EXTERNAL_BOUNDARIES = {
    "dashboard_transport": InventoryReason.EXTERNAL_PRIVATE,
    "home_integrations": InventoryReason.EXTERNAL_PRIVATE,
    "reasoner_transports": InventoryReason.EXTERNAL_PRIVATE,
    "training": InventoryReason.EXTERNAL_PRIVATE,
    "diagnostics": InventoryReason.EXTERNAL_UNCONFIRMED,
    "logging": InventoryReason.EXTERNAL_UNCONFIRMED,
    "house_vector_index": InventoryReason.EXTERNAL_UNCONFIRMED,
    "model_resources": InventoryReason.EXTERNAL_UNCONFIRMED,
}


def _witnessed(producer: str, state: dict, selected: bool) -> bool:
    """Recognize positive evidence from the closed family codecs."""
    phase = state.get("phase")
    if producer in _SELECTED:
        return phase == ("closed" if selected else "disabled_never_opened")
    if not selected:
        return phase == "disabled"
    if phase != "drained":
        return False
    aggregate = state.get("aggregate")
    if not aggregate or aggregate.get("owner_replaced") is True or aggregate.get("_capture_unreviewed"):
        return False
    if producer in _AUDIO:
        return (aggregate.get("owner_state") == "drained" and aggregate.get("owner_replaced") is False
                and any(aggregate.get("owner_counts", {}).values()))
    if producer in _IDENTITIES:
        owners = aggregate.get("owners", {})
        observed = [item for item in owners.values() if item.get("observation") != "disabled"]
        return bool(observed) and all(item.get("instance_sha256") and not item.get("_capture_unreviewed")
                                      and item.get("observation") in {"drained", "lazy_unopened"}
                                      for item in observed)
    if producer in _OFFLINE_EVIDENCE:
        return aggregate.get("observation") in _OFFLINE_EVIDENCE[producer]
    if producer == "configuration":
        return aggregate.get("writer_returned") is True and aggregate.get("cleanup_uncertain") is False
    # The legacy account grammar pins the actual admin publisher to its
    # destination and accepts drained only after observing the dashboard stop.
    return (producer == "accounts" and aggregate.get("native_drain_returned") is True
            and aggregate.get("cleanup_uncertain") is False)


def _census(run: dict) -> OutputCensus:
    """Evaluate a run from read_output_inventory, which joins the close receipt."""
    registered = set()
    covered = set()
    if run["phase"] == "observed" and run["version"] == "v4":
        header, states = run["header"], run["stores"]
        declared = _SELECTED | {
            name for name, grammar in header["grammars"].items() if _GRAMMARS.get(name) == grammar
        }
        composition = next((event for event in run["material_observations"] if event["kind"] == "composition"), {})
        registered = declared.intersection(composition.get("instrumented", ()), states)
        covered = {name for name in registered if _witnessed(name, states[name], header["selected"].get(name, True))}
        # Erasers and retention workers are part of the effective legacy rows.
        for name in ("events", "suggestions"):
            if any(states.get(f"{name}.{role}", {}).get("phase") != "closed" for role in ("eraser", "retention")):
                covered.discard(name)

    uncovered = tuple(name for name in ALL_MATRIX_PRODUCERS if name not in covered)
    # The supported catalog includes private remote outputs and opaque SDK
    # extents. A local journal and its close receipt cannot enumerate or type
    # those universes, or supply an exact applicable subject erasure receipt.
    reasons = {_EXTERNAL_BOUNDARIES[name] for name in ALL_MATRIX_PRODUCERS if name in _EXTERNAL_BOUNDARIES}
    states = run.get("stores", {})
    if any((item.get("aggregate") or {}).get("observation") in {"catalog_unresolved", "blocked_instance"}
           for item in states.values()):
        reasons.add(InventoryReason.PENDING_CLEANUP)
    producer_complete = not uncovered
    destinations_complete = producer_complete and not reasons.intersection({
        InventoryReason.EXTERNAL_PRIVATE, InventoryReason.EXTERNAL_UNCONFIRMED,
    })
    classification_complete = destinations_complete
    # The physical census is still open until copies, SDK extents and their
    # historical boundaries can be reconciled in one observed generation.
    coherent_physical_census = False
    return OutputCensus(
        run["run_id"], len(registered), producer_complete, destinations_complete,
        classification_complete, False,
        producer_complete and destinations_complete and classification_complete and coherent_physical_census,
        uncovered, tuple(sorted(reasons)),
    )


def read_output_census(controller: object) -> tuple[OutputCensus, ...]:
    """Read census predicates without acquiring native or erasure authority."""
    return tuple(_census(run) for run in read_output_inventory(controller))
