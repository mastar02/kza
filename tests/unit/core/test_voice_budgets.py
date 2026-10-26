"""Reject inputs that would disable bounded voice admission or work."""

import math

import pytest

from src.core.voice_delivery import ResponseBudgets
from src.orchestrator.dispatcher import MultiUserOrchestrator
from src.pipeline.response_handler import ResponseHandler


@pytest.mark.parametrize('name', ResponseBudgets.__dataclass_fields__)
@pytest.mark.parametrize('value', [True, 0, -1, math.nan, math.inf, -math.inf, '1', None])
def test_invalid_stage_budget_is_rejected(name, value):
    with pytest.raises(ValueError):
        ResponseBudgets(**{name: value})


def test_terminal_grace_cannot_overflow():
    with pytest.raises(ValueError):
        ResponseBudgets(cleanup_s=1e308)


@pytest.mark.parametrize('value', [True, 0, -1, math.nan, math.inf, -math.inf, '1'])
def test_orchestrator_timeout_is_validated_before_work(value):
    with pytest.raises(ValueError):
        MultiUserOrchestrator(None, None, None, response_timeout_s=value)


@pytest.mark.asyncio
@pytest.mark.parametrize('value', [True, 0, -1, math.nan, math.inf, -math.inf, '1'])
async def test_utterance_timeout_is_validated_before_admission(value):
    owner = ResponseHandler(None)
    with pytest.raises(ValueError):
        await owner.aspeak('synthetic', timeout_s=value)
    assert owner.get_delivery_status()['controllers'] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize('value', [True, math.nan, math.inf, -math.inf, '1'])
async def test_absolute_work_deadline_must_be_finite(value):
    owner = ResponseHandler(None)
    with pytest.raises(ValueError):
        await owner.arespond('synthetic', deadline=value)
    assert owner.get_delivery_status()['controllers'] == 0
