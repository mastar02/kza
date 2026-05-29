"""Tests: classify_error handles network/TLS/DNS/5xx errors as failover-worthy."""

import socket
import ssl

from src.llm.error_classifier import classify_error
from src.llm.types import ErrorKind


def test_ssl_error_is_failover_worthy():
    kind = classify_error(ssl.SSLError("handshake failed"))
    assert kind == ErrorKind.TIMEOUT
    assert kind.is_failover_worthy()


def test_dns_error_is_failover_worthy():
    kind = classify_error(socket.gaierror("Name or service not known"))
    assert kind == ErrorKind.TIMEOUT
    assert kind.is_failover_worthy()


def test_auth_wins_over_transient_substring():
    # Regresión: un error de auth que ADEMÁS contiene un patrón transitorio
    # ("ssl error") NO debe clasificarse TIMEOUT — AUTH no es failover-worthy.
    for m in ("401 Unauthorized: ssl error",
              "403 Forbidden: ssl error in handshake",
              "401 unauthorized - connection error"):
        kind = classify_error(Exception(m))
        assert kind == ErrorKind.AUTH, f"{m!r} → {kind} (esperado AUTH)"
        assert not kind.is_failover_worthy()


def test_pure_transient_still_timeout():
    # Sin substring de auth, los transitorios siguen siendo TIMEOUT.
    for m in ("503 Service Unavailable", "SSLError: handshake failure", "request timed out"):
        assert classify_error(Exception(m)) == ErrorKind.TIMEOUT, m


def test_openai_timeout_message_is_failover_worthy():
    assert classify_error(Exception("Request timed out.")) == ErrorKind.TIMEOUT


def test_5xx_is_failover_worthy():
    assert classify_error(Exception("503 Service Unavailable")).is_failover_worthy()


def test_auth_still_wins_over_transient():
    # 401 must still be AUTH (not failover-worthy), not misclassified as transient.
    assert classify_error(Exception("401 Unauthorized")) == ErrorKind.AUTH


def test_auth_message_mentioning_ssl_is_not_misclassified():
    # "ssl" substring must not shadow AUTH — 401 wins.
    from src.llm.error_classifier import classify_error
    from src.llm.types import ErrorKind
    assert classify_error(Exception("401 Unauthorized: ssl client cert required")) == ErrorKind.AUTH


def test_stringified_ssl_error_still_transient():
    from src.llm.error_classifier import classify_error
    from src.llm.types import ErrorKind
    assert classify_error(Exception("SSLError: handshake failure")) == ErrorKind.TIMEOUT
