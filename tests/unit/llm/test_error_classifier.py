"""Tests for LLM error classifier."""

import asyncio
import pytest
from src.llm.error_classifier import classify_error
from src.llm.types import ErrorKind


class TestClassifyError:
    def test_asyncio_timeout(self):
        assert classify_error(asyncio.TimeoutError()) == ErrorKind.TIMEOUT

    def test_timeout_error_builtin(self):
        assert classify_error(TimeoutError("read")) == ErrorKind.TIMEOUT

    def test_connection_error(self):
        assert classify_error(ConnectionError("refused")) == ErrorKind.TIMEOUT

    def test_rate_limit_message(self):
        e = RuntimeError("429 Too Many Requests")
        assert classify_error(e) == ErrorKind.RATE_LIMIT

    def test_rate_limit_throttling(self):
        e = RuntimeError("ThrottlingException: too many requests")
        assert classify_error(e) == ErrorKind.RATE_LIMIT

    def test_rate_limit_concurrency(self):
        e = RuntimeError("concurrency limit reached")
        assert classify_error(e) == ErrorKind.RATE_LIMIT

    def test_rate_limit_quota(self):
        e = RuntimeError("quota limit exceeded for this minute")
        assert classify_error(e) == ErrorKind.RATE_LIMIT

    def test_billing_credits(self):
        e = RuntimeError("insufficient credits in your account")
        assert classify_error(e) == ErrorKind.BILLING

    def test_billing_balance(self):
        e = RuntimeError("credit balance too low")
        assert classify_error(e) == ErrorKind.BILLING

    def test_auth_401(self):
        e = RuntimeError("401 Unauthorized: invalid api key")
        assert classify_error(e) == ErrorKind.AUTH

    def test_auth_403(self):
        e = RuntimeError("403 Forbidden")
        assert classify_error(e) == ErrorKind.AUTH

    def test_format_invalid_json(self):
        e = ValueError("Invalid JSON in response")
        assert classify_error(e) == ErrorKind.FORMAT

    def test_idle_timeout_explicit(self):
        # Custom exception class for idle timeout
        from src.llm.error_classifier import IdleTimeoutError
        assert classify_error(IdleTimeoutError(15.0)) == ErrorKind.IDLE_TIMEOUT

    def test_unknown_error_is_permanent(self):
        e = RuntimeError("something completely unexpected")
        assert classify_error(e) == ErrorKind.PERMANENT
