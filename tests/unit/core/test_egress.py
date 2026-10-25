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
