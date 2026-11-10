"""OQ-8: the slow path explains a withdrawn cloud consent in Spanish."""
from src.core.egress import CloudConsentGate, ConsentWithdrawalSource, EgressDenied
from src.llm.error_classifier import is_consent_withdrawn
from src.llm.reasoner import HttpReasoner
from src.llm.router import FailedAttempt, FallbackSummaryError
from src.llm.types import ErrorKind
from src.orchestrator import dispatcher as dispatcher_module
from tests.integration.test_async_conversation_generation import dispatch, graph
from tests.mocks.llm_http import event

MESSAGE = ("El razonamiento en la nube está desactivado porque se retiró el consentimiento. "
           "Puedo seguir con los comandos de la casa.")


def test_constant_is_the_designed_spanish_message():
    assert dispatcher_module._CLOUD_CONSENT_WITHDRAWN_RESPONSE == MESSAGE


def test_fallback_summary_and_groups_are_detected():
    summary = FallbackSummaryError([FailedAttempt("reasoner_cloud", ErrorKind.CONSENT_WITHDRAWN, "x"),
                                    FailedAttempt("fast_router_7b", ErrorKind.TIMEOUT, "y")])
    assert is_consent_withdrawn(summary)
    assert not is_consent_withdrawn(FallbackSummaryError([FailedAttempt("a", ErrorKind.TIMEOUT, "y")]))
    group = BaseExceptionGroup("close", [RuntimeError("other"), EgressDenied("cloud_consent_withdrawn")])
    assert is_consent_withdrawn(group)
    assert is_consent_withdrawn(ExceptionGroup("outer", [ExceptionGroup("inner", [group.exceptions[1]])]))
    assert not is_consent_withdrawn(BaseExceptionGroup("close", [RuntimeError("other")]))


async def test_withdrawn_reasoner_speaks_the_consent_message(monkeypatch):
    async with graph(monkeypatch, [event("respuesta"), event(finish="stop")]) as (owner, endpoint):
        gate = CloudConsentGate()
        reasoner = owner.llm
        assert isinstance(reasoner, HttpReasoner)
        reasoner.policy = type(reasoner.policy)(reasoner.policy.destination, True, reasoner.policy.endpoint,
                                                consent_gate=gate)
        gate.withdraw(ConsentWithdrawalSource.ACTOR)
        result = await dispatch(owner)
        assert not result.success and result.intent == "error"
        assert result.response == MESSAGE
        assert endpoint.requests == []


async def test_other_failures_keep_the_generic_message(monkeypatch):
    async with graph(monkeypatch, [event("partial"), event(finish="length")]) as (owner, endpoint):
        result = await dispatch(owner)
        assert not result.success
        assert result.response == "No pude procesar eso ahora mismo, probá de nuevo."
