"""Completion must follow operation and delivery evidence independently."""

from dataclasses import replace

import pytest

from src.core.task_outcome import (
    DeliveryOutcome, DispatchState, EffectState, PersistenceState, ResponseRequirement,
    TaskOutcome, TaskStatus, TurnOutcome, project_turn_outcome,
)
from src.core.voice_delivery import DeliveryState, GenerationState
from src.home_assistant.home_actions import HomeActionResult, HomeEffectResult
from src.home_assistant.effect_plans import EffectIntent
from src.orchestrator.dispatcher import DispatchResult, PathType
from src.orchestrator.priority_queue import Priority


@pytest.mark.parametrize("fields", [
    {"persistence": PersistenceState.UNCERTAIN}, {"persistence": PersistenceState.NOT_WRITTEN},
    {"dispatch": DispatchState.UNCERTAIN}, {"dispatch": DispatchState.PARTIAL},
    {"effect": EffectState.UNCONFIRMED}, {"effect": EffectState.PARTIAL},
    {"effect": EffectState.OBSERVED}, {"handled": False},
])
def test_completion_cannot_contradict_required_stage(fields):
    with pytest.raises(ValueError, match="completion_without_required_evidence"):
        TaskOutcome(TaskStatus.COMPLETED, **fields)


def test_committed_operation_survives_failed_confirmation_without_overall_success():
    task = TaskOutcome(TaskStatus.COMPLETED, persistence=PersistenceState.COMMITTED,
                       effect=EffectState.OBSERVED, observation_scope="software_state")
    turn = TurnOutcome(task, DeliveryOutcome(GenerationState.COMPLETE, DeliveryState.FAILED, error=True))
    result = turn.to_dict()
    assert result["task_completed"] and not result["response_completed"] and not result["success"]
    assert result["outcome"]["persistence"] == "committed"


def test_optional_replication_is_not_evidence_that_ha_list_matches():
    local = TaskOutcome(TaskStatus.COMPLETED, persistence=PersistenceState.COMMITTED,
                        replication=DispatchState.UNCERTAIN)
    assert local.task_completed
    assert local.to_dict()["replication"] == "uncertain"
    # The selected HA-authoritative integration must require its own dispatch/effect.
    with pytest.raises(ValueError):
        replace(local, dispatch=DispatchState.UNCERTAIN)


def test_silent_effect_has_no_fabricated_spoken_completion():
    result = TurnOutcome(TaskOutcome(TaskStatus.COMPLETED),
                         response_requirement=ResponseRequirement.SILENT).to_dict()
    assert result["success"] and result["response_completed"] is None
    assert not result["response_required"] and result["outcome"]["response"] is None


@pytest.mark.parametrize("status", [TaskStatus.RECEIVED, TaskStatus.QUEUED, TaskStatus.NEEDS_INPUT,
                                   TaskStatus.UNSUPPORTED, TaskStatus.UNCERTAIN])
def test_successful_feedback_is_not_completed_operation(status):
    result = TurnOutcome(TaskOutcome(status), DeliveryOutcome(GenerationState.COMPLETE, DeliveryState.COMPLETE)).to_dict()
    assert result["handled"] and result["response_completed"]
    assert not result["task_completed"] and not result["success"]


def test_later_success_cannot_hide_failed_required_delivery():
    turn = TurnOutcome(TaskOutcome(TaskStatus.COMPLETED))
    turn.record_delivery(DeliveryOutcome(GenerationState.COMPLETE, DeliveryState.FAILED, error=True))
    turn.record_delivery(DeliveryOutcome(GenerationState.COMPLETE, DeliveryState.COMPLETE))
    assert not turn.to_dict()["success"]


def test_projection_rejects_arbitrary_content_in_closed_fields():
    result = project_turn_outcome({"handled": "yes", "task_status": "private text", "receipt": {"id": "private"},
        "task_completed": False, "dispatch_state": "uncertain", "user": "private"})
    assert result == {"task_completed": False, "dispatch_state": "uncertain"}


def test_legacy_or_unscoped_home_receipt_cannot_claim_completion():
    receipt = HomeActionResult("test", "light.fixture", "confirmed")
    assert not receipt.task_outcome.task_completed
    receipt = replace(receipt, dispatch=DispatchState.ACCEPTED, effect=EffectState.OBSERVED)
    assert not receipt.task_outcome.task_completed
    assert replace(receipt, observation_scope="ha_state").task_outcome.task_completed


def test_composite_cannot_invent_observation_scope():
    intent = EffectIntent("light.fixture", "turn_on")
    leaf = HomeEffectResult(intent, "confirmed", dispatch=DispatchState.ACCEPTED, effect=EffectState.OBSERVED)
    receipt = HomeActionResult.composite("test", "light.group", "confirmed", None, (leaf,))
    assert receipt.observation_scope is None and not receipt.task_outcome.task_completed


def test_dispatch_serialization_keeps_task_and_failed_delivery_independently():
    result = DispatchResult(path=PathType.FAST_LIST, priority=Priority.MEDIUM, success=True,
        response="synthetic", intent="list", outcome=TaskOutcome(TaskStatus.COMPLETED,
            persistence=PersistenceState.COMMITTED, effect=EffectState.OBSERVED, observation_scope="software_state"),
        delivery_outcome=DeliveryOutcome(GenerationState.COMPLETE, DeliveryState.FAILED))
    document = result.to_dict()
    assert document["task_completed"] and not document["success"]
    assert document["outcome"]["response"]["delivery"] == "failed"
