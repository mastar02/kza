"""T037-U6: the compound runner's order, request ids, independence, reply and aggregate outcome."""

import asyncio
from types import SimpleNamespace

import pytest

from src.core.task_outcome import (
    CURRENT_REQUEST_ID,
    DispatchState,
    EffectState,
    PersistenceState,
    ResponseRequirement,
    TaskOutcome,
    TaskStatus,
)
from src.orchestrator.compound import CompoundRunner, PartVerdict, aggregate_outcome, part_outcome

DONE_HA = TaskOutcome(TaskStatus.COMPLETED, persistence=PersistenceState.VOLATILE, dispatch=DispatchState.ACCEPTED,
                      effect=EffectState.OBSERVED, observation_scope="ha_state")
UNCERTAIN_HA = TaskOutcome(TaskStatus.UNCERTAIN, persistence=PersistenceState.VOLATILE,
                           dispatch=DispatchState.ACCEPTED, effect=EffectState.UNCERTAIN)
REFUSED = TaskOutcome(TaskStatus.UNSUPPORTED, persistence=PersistenceState.NOT_WRITTEN,
                      dispatch=DispatchState.NOT_SENT, effect=EffectState.NOT_OBSERVED)
TIMER_DONE = TaskOutcome(TaskStatus.COMPLETED, persistence=PersistenceState.VOLATILE, effect=EffectState.OBSERVED,
                         observation_scope="software_state")


def _part(segment, intent=None):
    return SimpleNamespace(segment=segment, intent=intent)


def _result(outcome, *, response="", entity=None, service=None, silent=False, path="fast_domotics"):
    return SimpleNamespace(outcome=outcome, response=response, success=outcome.task_completed,
                           action={"entity_id": entity, "service": service, "data": {}} if entity else None,
                           receipt=None, intent="domotics", path=SimpleNamespace(value=path),
                           response_requirement=ResponseRequirement.SILENT if silent else ResponseRequirement.REQUIRED)


async def _run(parts, results, **kwargs):
    runner = kwargs.pop("runner", None) or CompoundRunner()
    seen = []

    async def run_part(context):
        seen.append((context.index, context.request_id, CURRENT_REQUEST_ID.get()))
        produced = results[context.index - 1]
        return produced(context) if callable(produced) else produced

    token = CURRENT_REQUEST_ID.set(kwargs.pop("turn", "voice:t1"))
    try:
        return await runner.run(parts, run_part, **kwargs), seen
    finally:
        CURRENT_REQUEST_ID.reset(token)


async def test_parts_run_in_order_with_their_own_request_ids():
    run, seen = await _run([_part("apagá la luz"), _part("pausá la música")],
                           [_result(DONE_HA, entity="light.a", service="turn_off", silent=True),
                            _result(TaskOutcome(TaskStatus.COMPLETED, dispatch=DispatchState.ACCEPTED),
                                    response="Pausé la música", path="fast_music")])
    assert seen == [(1, "voice:t1:part1", "voice:t1:part1"), (2, "voice:t1:part2", "voice:t1:part2")]
    assert CURRENT_REQUEST_ID.get() is None
    assert [r.verdict for r in run.reports] == [PartVerdict.DONE, PartVerdict.DONE]
    assert run.outcome.status is TaskStatus.COMPLETED and run.outcome.dispatch is DispatchState.ACCEPTED
    assert run.outcome.parts == (DONE_HA, TaskOutcome(TaskStatus.COMPLETED, dispatch=DispatchState.ACCEPTED))
    assert run.response == "Apagá la luz: hecho. Pausá la música: Pausé la música."


async def test_refused_part_never_stops_the_next_and_the_reply_says_why():
    run, seen = await _run([_part("abrí la cerradura"), _part("apagá la luz")],
                           [_result(REFUSED, response="No manejo cerraduras por voz"),
                            _result(DONE_HA, entity="light.a", service="turn_off", silent=True)])
    assert len(seen) == 2
    assert run.outcome.status is TaskStatus.FAILED and run.outcome.dispatch is DispatchState.PARTIAL
    assert run.response == "Abrí la cerradura: no lo hice. No manejo cerraduras por voz. Apagá la luz: hecho."
    assert "Listo" not in run.response


async def test_uncertain_part_blocks_only_later_parts_on_the_same_entity():
    def later(context):
        assert context.check("light.a", "turn_off") is PartVerdict.SAME_TARGET
        assert context.check("light.b", "turn_on") is None
        return _result(DONE_HA, entity="light.b", service="turn_on", silent=True)

    run, _ = await _run([_part("apagá la luz"), _part("prendé la otra")],
                        [_result(UNCERTAIN_HA, entity="light.a", service="turn_off"), later])
    # Both parts were sent; the first one's effect is what is uncertain.
    assert run.outcome.status is TaskStatus.UNCERTAIN and run.outcome.dispatch is DispatchState.ACCEPTED
    assert run.outcome.effect is EffectState.PARTIAL
    assert run.response.startswith("Apagá la luz: no sé si se hizo; fijate antes de repetirlo.")


async def test_duplicate_and_conflict_on_one_entity_are_not_sent():
    def check(context):
        verdict = context.check("light.a", "turn_off" if context.index == 2 else "turn_on")
        return _result(context.skip(verdict), entity="light.a", service="x")

    run, _ = await _run([_part("apagá la luz"), _part("apagá la luz del living"), _part("prendé la luz del living")],
                        [_result(DONE_HA, entity="light.a", service="turn_off", silent=True), check, check])
    assert [r.verdict for r in run.reports] == [PartVerdict.DONE, PartVerdict.DUPLICATE, PartVerdict.CONFLICT]
    assert [r.outcome.dispatch for r in run.reports[1:]] == [DispatchState.NOT_SENT] * 2
    assert run.outcome.status is TaskStatus.FAILED


async def test_interruption_stops_the_remaining_parts():
    flag = {"stop": False}

    def first(context):
        flag["stop"] = True
        return _result(DONE_HA, entity="light.a", service="turn_off", silent=True)

    run, seen = await _run([_part("apagá la luz"), _part("pausá la música"), _part("prendé la tele")],
                           [first, None, None], interrupted=lambda: flag["stop"])
    assert len(seen) == 1
    assert [r.verdict for r in run.reports] == [PartVerdict.DONE, PartVerdict.INTERRUPTED, PartVerdict.INTERRUPTED]
    assert run.outcome.status is TaskStatus.CANCELLED and run.outcome.dispatch is DispatchState.PARTIAL
    assert run.response.endswith("Prendé la tele: no lo intenté porque me interrumpiste.")


async def test_household_target_known_before_running_blocks_after_a_failure():
    failed = TaskOutcome(TaskStatus.FAILED, persistence=PersistenceState.NOT_WRITTEN, dispatch=DispatchState.NOT_SENT,
                         effect=EffectState.NOT_OBSERVED)
    run, seen = await _run([_part("poné un timer", "timer"), _part("pausá el timer", "timer")],
                           [_result(failed, path="fast_timer"), None], part_target=lambda part: part.intent)
    assert len(seen) == 1 and run.reports[1].verdict is PartVerdict.SAME_TARGET


async def test_handler_exception_is_uncertain_and_later_parts_still_run():
    def boom(context):
        raise RuntimeError("transport")

    run, seen = await _run([_part("apagá la luz"), _part("pausá la música")],
                           [boom, _result(TaskOutcome(TaskStatus.COMPLETED, dispatch=DispatchState.ACCEPTED),
                                          path="fast_music")])
    assert len(seen) == 2 and run.reports[0].verdict is PartVerdict.UNCERTAIN
    assert run.reports[0].reason == "part_exception" and run.outcome.status is TaskStatus.UNCERTAIN


async def test_cancellation_propagates():
    async def cancelled(context):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await CompoundRunner().run([_part("apagá la luz")], cancelled)
    assert CURRENT_REQUEST_ID.get() is None


async def test_retried_turn_reports_without_running_again():
    runner = CompoundRunner()
    calls = []

    def counted(context):
        calls.append(context.index)
        return _result(DONE_HA, entity=f"light.{context.index}", service="turn_off", silent=True)

    parts = [_part("apagá la luz"), _part("apagá la otra")]
    await _run(parts, [counted, counted], runner=runner, turn="voice:r")
    again, _ = await _run(parts, [counted, counted], runner=runner, turn="voice:r")
    assert calls == [1, 2]
    assert [r.retried for r in again.reports] == [True, True] and again.outcome.status is TaskStatus.COMPLETED
    other, _ = await _run([_part("prendé la luz"), parts[1]], [counted, counted], runner=runner, turn="voice:r")
    assert calls == [1, 2] and other.reports[0].reason == "request_id_conflict"
    assert other.reports[0].outcome.dispatch is DispatchState.NOT_SENT


async def test_without_a_turn_request_id_parts_still_get_distinct_ids():
    seen = []

    async def run_part(context):
        seen.append(context.request_id)
        return _result(DONE_HA, entity=f"light.{context.index}", service="turn_off", silent=True)

    await CompoundRunner().run([_part("a"), _part("b")], run_part)
    assert seen[0].startswith("compound:") and seen[0].endswith(":part1") and seen[1].endswith(":part2")
    assert seen[0].rsplit(":", 1)[0] == seen[1].rsplit(":", 1)[0]


def test_aggregate_mixed_observations_and_storage():
    reports = [SimpleNamespace(outcome=DONE_HA), SimpleNamespace(outcome=TIMER_DONE)]
    outcome = aggregate_outcome(reports, False)
    assert outcome.status is TaskStatus.COMPLETED and outcome.effect is EffectState.OBSERVED
    assert outcome.observation_scope == "software_state" and outcome.persistence is PersistenceState.VOLATILE
    assert outcome.to_dict()["parts"] == [DONE_HA.to_dict(), TIMER_DONE.to_dict()]
    assert "parts" not in DONE_HA.to_dict()


def test_part_outcome_without_an_outcome():
    music = SimpleNamespace(outcome=None, success=True, path=SimpleNamespace(value="fast_music"))
    assert part_outcome(music) == TaskOutcome(TaskStatus.COMPLETED, dispatch=DispatchState.ACCEPTED)
    music_failed = SimpleNamespace(outcome=None, success=False, path=SimpleNamespace(value="slow_music"))
    assert part_outcome(music_failed).dispatch is DispatchState.UNKNOWN
    conflict = SimpleNamespace(outcome=None, success=False, path=SimpleNamespace(value="fast_domotics"))
    assert part_outcome(conflict) == TaskOutcome(TaskStatus.FAILED, dispatch=DispatchState.NOT_SENT,
                                                 effect=EffectState.NOT_OBSERVED)


def test_task_outcome_parts_are_flat_outcomes():
    with pytest.raises(ValueError, match="invalid_task_parts"):
        TaskOutcome(TaskStatus.COMPLETED, parts=(TaskOutcome(TaskStatus.COMPLETED, parts=(DONE_HA,)),))
    with pytest.raises(ValueError, match="invalid_task_parts"):
        TaskOutcome(TaskStatus.COMPLETED, parts=[DONE_HA])
