"""OQ-8/D8: a consent-withdrawn cloud reasoner fails over to the local 7B."""
import httpx
import openai
import pytest

from src.core.egress import EgressDenied
from src.llm.cooldown import CooldownManager
from src.llm.error_classifier import classify_error, is_consent_withdrawn
from src.llm.router import FallbackSummaryError, LLMRouter
from src.llm.types import EndpointKind, ErrorKind, LLMEndpoint


def _sdk_wrapped():
    try:
        try:
            raise EgressDenied("cloud_consent_withdrawn")
        except EgressDenied as error:
            raise openai.APIConnectionError(request=httpx.Request("GET", "https://gateway.test/v1")) from error
    except openai.APIConnectionError as wrapped:
        return wrapped


def _hermes_killed():
    try:
        try:
            raise RuntimeError("hermes -z failed (exit=-9): ")
        except RuntimeError as error:
            raise EgressDenied("cloud_consent_withdrawn") from error
    except EgressDenied as denied:
        return denied


def test_kind_is_failover_worthy():
    assert ErrorKind.CONSENT_WITHDRAWN.value == "consent_withdrawn"
    assert ErrorKind.CONSENT_WITHDRAWN.is_failover_worthy()


@pytest.mark.parametrize("make", [_sdk_wrapped, _hermes_killed,
                                  lambda: EgressDenied("cloud_consent_withdrawn")])
def test_classifier_finds_the_withdrawal_in_the_chain(make):
    assert classify_error(make()) is ErrorKind.CONSENT_WITHDRAWN


def test_classifier_context_chain_and_other_denials():
    try:
        try:
            raise EgressDenied("cloud_consent_withdrawn")
        except EgressDenied:
            raise RuntimeError("cleanup failed while handling")
    except RuntimeError as error:
        assert classify_error(error) is ErrorKind.CONSENT_WITHDRAWN
    assert classify_error(EgressDenied("cloud_consent_required")) is ErrorKind.PERMANENT
    assert classify_error(PermissionError("cloud_consent_withdrawn")) is ErrorKind.PERMANENT
    assert not is_consent_withdrawn(RuntimeError("cloud_consent_withdrawn"))


def test_walker_is_bounded_on_cycles():
    first, second = RuntimeError("a"), RuntimeError("b")
    first.__context__, second.__context__ = second, first
    assert is_consent_withdrawn(first) is False


class _Client:
    def __init__(self, result):
        self.result, self.calls = result, 0

    async def complete(self, prompt, **kwargs):
        self.calls += 1
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


def _router(tmp_path, cloud):
    local = _Client("respuesta local")
    endpoints = [LLMEndpoint("reasoner_cloud", EndpointKind.HTTP_REASONER, _Client(cloud), 0, timeout_s=0),
                 LLMEndpoint("fast_router_7b", EndpointKind.FAST_ROUTER, local, 1, timeout_s=0)]
    cooldowns = CooldownManager(tmp_path / "cooldown.json")
    return LLMRouter(endpoints, cooldowns), cooldowns, local


@pytest.mark.parametrize("make", [_sdk_wrapped, _hermes_killed])
async def test_router_fails_over_without_recording_a_cooldown(tmp_path, make):
    router, cooldowns, local = _router(tmp_path, make())
    result = await router.complete("consulta")
    assert result.text == "respuesta local" and result.endpoint_id == "fast_router_7b" and result.attempts == 2
    assert local.calls == 1
    state = cooldowns.get_state("reasoner_cloud")
    assert state.error_count == 0 and state.cooldown_until == 0.0 and state.last_error_kind is None
    assert cooldowns.is_available("reasoner_cloud")


async def test_router_fails_over_after_hermes_kill(tmp_path):
    await test_router_fails_over_without_recording_a_cooldown(tmp_path, _hermes_killed)


async def test_summary_records_the_withdrawn_attempt(tmp_path):
    router, _cooldowns, local = _router(tmp_path, _sdk_wrapped())
    local.result = EgressDenied("cloud_consent_withdrawn")
    with pytest.raises(FallbackSummaryError) as caught:
        await router.complete("consulta")
    assert [a.error_kind for a in caught.value.attempts] == [ErrorKind.CONSENT_WITHDRAWN] * 2
    assert is_consent_withdrawn(caught.value)
