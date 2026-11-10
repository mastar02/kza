"""Processing declarations cannot acquire authority from transport topology."""

import pytest

from src.core.egress import EgressDenied, EgressPolicy


@pytest.mark.parametrize("url", [
    "http://127.0.0.1:8200/v1", "http://localhost:8200/v1",
    "http://[::1]:8200/v1", "https://gateway.test/v1",
])
def test_unknown_or_declared_cloud_endpoint_requires_consent(url):
    for policy in (EgressPolicy(endpoint=url), EgressPolicy("cloud", False, url)):
        with pytest.raises(EgressDenied):
            policy.require_request(url + "/chat/completions")


def test_explicit_local_destination_is_bound_to_its_own_endpoint():
    policy = EgressPolicy("local", False, "http://127.0.0.1:8101/v1")
    policy.require_request("http://127.0.0.1:8101/v1/chat/completions")
    with pytest.raises(EgressDenied):
        policy.require_allowed("http://127.0.0.1:8200/v1")
    with pytest.raises(EgressDenied):
        policy.require_request("https://cloud.test/v1/chat/completions")


@pytest.mark.parametrize("path", ["/v10/chat/completions", "/v1/../other", "/v1/%2e%2e/other", "/v1/%5cother"])
def test_request_cannot_escape_bound_path(path):
    policy = EgressPolicy("local", False, "http://127.0.0.1:8101/v1")
    with pytest.raises((EgressDenied, ValueError)):
        policy.require_request("http://127.0.0.1:8101" + path)


@pytest.mark.parametrize("consent", ["false", "true", 1, 0, None, [], {}])
def test_consent_never_uses_truthiness(consent):
    with pytest.raises(ValueError):
        EgressPolicy("cloud", consent, "http://127.0.0.1:8200/v1")


@pytest.mark.parametrize("url", ["${GATEWAY}", "file:///tmp/data", "http://[::1", "https://user:secret@host/v1", "https://host/v1?key=secret"])
def test_invalid_transport_cannot_be_declared_local(url):
    with pytest.raises(ValueError):
        EgressPolicy("local", False, url)


# ----------------------------------------------------------------------
# OQ-8: the shared runtime consent gate on a frozen policy.
# ----------------------------------------------------------------------

CLOUD_URL = "https://gateway.test/v1"
LOCAL_URL = "http://127.0.0.1:8101/v1"


def _withdrawn_gate():
    from src.core.egress import CloudConsentGate, ConsentWithdrawalSource
    gate = CloudConsentGate()
    gate.withdraw(ConsentWithdrawalSource.ACTOR)
    return gate


def test_cloud_policy_is_denied_once_its_gate_is_withdrawn():
    from src.core.egress import CloudConsentGate, ConsentWithdrawalSource
    gate = CloudConsentGate()
    policy = EgressPolicy.for_http(CLOUD_URL, "cloud", True, consent_gate=gate)
    assert policy.allowed is True
    policy.require_allowed(CLOUD_URL)
    policy.require_request(CLOUD_URL + "/chat/completions")
    gate.withdraw(ConsentWithdrawalSource.ACTOR)
    assert policy.allowed is False
    for check in (policy.require_allowed, lambda: policy.require_allowed(CLOUD_URL),
                  lambda: policy.require_request(CLOUD_URL + "/chat/completions")):
        with pytest.raises(EgressDenied, match="^cloud_consent_withdrawn$"):
            check()
    with pytest.raises(EgressDenied, match="^cloud_consent_withdrawn$"):
        EgressPolicy(cloud_consent=True, consent_gate=gate).require_allowed()


def test_local_policy_is_never_denied_by_a_withdrawn_gate():
    policy = EgressPolicy("local", False, LOCAL_URL, consent_gate=_withdrawn_gate())
    assert policy.allowed is True
    policy.require_allowed(LOCAL_URL)
    policy.require_request(LOCAL_URL + "/completions")


def test_missing_consent_is_reported_before_withdrawal():
    policy = EgressPolicy("cloud", False, CLOUD_URL, consent_gate=_withdrawn_gate())
    with pytest.raises(EgressDenied, match="^cloud_consent_required$"):
        policy.require_allowed()
    with pytest.raises(EgressDenied, match="^cloud_consent_required$"):
        policy.require_request(CLOUD_URL + "/models")


def test_gate_does_not_change_equality_hash_or_frozenness():
    from dataclasses import FrozenInstanceError
    from src.core.egress import CloudConsentGate
    plain = EgressPolicy("cloud", True, CLOUD_URL)
    gated = EgressPolicy("cloud", True, CLOUD_URL, consent_gate=CloudConsentGate())
    assert plain == gated and hash(plain) == hash(gated)
    with pytest.raises(FrozenInstanceError):
        gated.consent_gate = None


@pytest.mark.parametrize("value", [object(), True, "gate", 1])
def test_gate_must_be_the_exact_gate_type(value):
    with pytest.raises(TypeError, match="^cloud consent gate must be a CloudConsentGate$"):
        EgressPolicy("cloud", True, CLOUD_URL, consent_gate=value)
    with pytest.raises(TypeError, match="^cloud consent gate must be a CloudConsentGate$"):
        EgressPolicy.for_http(CLOUD_URL, "cloud", True, consent_gate=value)


def test_client_constructors_require_a_gate_for_consented_cloud():
    from src.core.egress import CloudConsentGate, require_consent_gate
    with pytest.raises(ValueError, match="^consent_gate_required$"):
        require_consent_gate(EgressPolicy("cloud", True, CLOUD_URL))
    require_consent_gate(EgressPolicy("cloud", True, CLOUD_URL, consent_gate=CloudConsentGate()))
    require_consent_gate(EgressPolicy("cloud", False, CLOUD_URL))  # Denied anyway.
    require_consent_gate(EgressPolicy("local", False, LOCAL_URL))
