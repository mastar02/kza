"""C02 deterministic matrix entries (design ``c02_matrix_format``, P3-F0/c02-matrix/v1).

Each family unit contributes tagged entries in its own test module; an entry
counts toward C02 only when its test asserts every effect counter, the task
status and the persistence (``counting_rules``). Layers: ``classifier`` (the
pure P3-F2 classifier), ``dispatcher`` (RequestDispatcher.dispatch),
``router`` (RequestRouter.process_command with a real orchestrator),
``graph`` (build_application) and ``web`` (a dashboard HTTP task route with a
real session authority, actors ``web_home``/``web_person``; T035-U1).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

EFFECT_COUNTERS = (
    "ha_service_calls", "vector_searches", "music_calls", "list_mutations", "reminder_rows_delta",
    "reminder_transitions", "timer_delta", "timer_transitions", "queue_cancels", "routine_runs",
    "routine_changes", "llm_8101_calls", "cloud_calls", "ledger_claims", "spoken_utterances",
)
CLASSIFIER = "not_applicable:classifier"


def effects(**counts: int) -> dict[str, int]:
    """Every effect counter, zero unless given."""
    unknown = set(counts) - set(EFFECT_COUNTERS)
    if unknown:
        raise ValueError(f"unknown effect counters: {sorted(unknown)}")
    return {name: counts.get(name, 0) for name in EFFECT_COUNTERS}


@dataclass(frozen=True)
class C02Entry:
    """One tagged C02 matrix entry.

    Attributes:
        id: Stable ``C02-<FAMILY>-<NNNN>``; design seed ids are kept when the
            entry implements a seed (``seed`` notes any interim difference).
        text: Utterance as the STT would deliver it; for ``web``, the request
            ("METHOD path [body]").
        layer: classifier | router | dispatcher | graph | web.
        family, effect_class, expected_outcome, actor: C02 tags.
        path: PathType value, or None (classifier layer, or a gated turn).
        task_status: TaskStatus value, None for 'no_turn', or CLASSIFIER.
        persistence: PersistenceState value, None for 'no_turn', or CLASSIFIER.
        spoken: none | clarification | refusal | readback | answer | notice.
        effects: Every counter in EFFECT_COUNTERS.
        conditions: Layer conditions (wake, llm, state, service_filter, ...).
        classification: Classifier layer only: {family, operation, rule[, parts]}.
        unit, finding, seed, note: Provenance.
    """

    id: str
    text: str
    layer: str
    family: str
    effect_class: str
    expected_outcome: str
    path: str | None
    task_status: str | None
    persistence: str | None
    spoken: str
    effects: dict = field(default_factory=effects)
    conditions: dict = field(default_factory=dict)
    classification: dict | None = None
    actor: str = "household_unknown_voice"
    unit: str = "P3-F2"
    finding: str | None = None
    seed: str | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if set(self.effects) != set(EFFECT_COUNTERS):
            raise ValueError(f"{self.id}: every effect counter must be stated")
        if self.layer not in ("classifier", "router", "dispatcher", "graph", "web"):
            raise ValueError(f"{self.id}: unknown layer {self.layer}")


def tag_counts(entries) -> dict:
    """Counts per family and per effect_class (evidence export)."""
    return {"total": len(entries),
            "by_layer": dict(Counter(e.layer for e in entries)),
            "by_family": dict(Counter(e.family for e in entries)),
            "by_effect_class": dict(Counter(e.effect_class for e in entries))}
