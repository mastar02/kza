"""
Tests for structured logging system.
"""

import pytest
import logging
import json
from io import StringIO
from unittest.mock import patch

from src.core.logging import (
    get_logger,
    configure_logging,
    LogContext,
    LogFormat,
    LogConfig,
    set_context,
    clear_context,
    get_context,
    generate_request_id,
    StructuredFormatter,
    ContextLogger,
)


class TestLogConfig:
    """Tests for LogConfig"""

    def test_default_config(self):
        config = LogConfig()
        assert config.level == "INFO"
        assert config.format == LogFormat.COLORED
        assert config.include_timestamp is True

    def test_custom_config(self):
        config = LogConfig(
            level="DEBUG",
            format=LogFormat.JSON,
            include_location=False
        )
        assert config.level == "DEBUG"
        assert config.format == LogFormat.JSON
        assert config.include_location is False


class TestStructuredFormatter:
    """Tests for StructuredFormatter"""

    def test_json_format(self):
        config = LogConfig(format=LogFormat.JSON, include_timestamp=False)
        formatter = StructuredFormatter(config)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert data["message"] == "Test message"
        assert data["level"] == "INFO"
        assert data["logger"] == "test"

    def test_plain_format(self):
        config = LogConfig(format=LogFormat.PLAIN, include_timestamp=False)
        formatter = StructuredFormatter(config)

        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="test.py",
            lineno=20,
            msg="Warning message",
            args=(),
            exc_info=None
        )

        output = formatter.format(record)
        assert "WARNING" in output
        assert "Warning message" in output

    def test_colored_format(self):
        config = LogConfig(format=LogFormat.COLORED)
        formatter = StructuredFormatter(config)

        record = logging.LogRecord(
            name="test.module",
            level=logging.ERROR,
            pathname="test.py",
            lineno=30,
            msg="Error message",
            args=(),
            exc_info=None
        )

        output = formatter.format(record)
        assert "Error message" in output


class TestLogContext:
    """Tests for LogContext"""

    def test_context_manager(self):
        clear_context()

        with LogContext(user_id="user_123", request_id="req_abc"):
            ctx = get_context()
            assert ctx["user_id"] == "user_123"
            assert ctx["request_id"] == "req_abc"

        # Context should be cleared after exiting
        ctx = get_context()
        assert "user_id" not in ctx

    def test_nested_context(self):
        clear_context()

        with LogContext(user_id="user_1"):
            assert get_context()["user_id"] == "user_1"

            with LogContext(zone_id="living"):
                ctx = get_context()
                assert ctx["user_id"] == "user_1"
                assert ctx["zone_id"] == "living"

            # Inner context cleared, outer remains
            ctx = get_context()
            assert ctx["user_id"] == "user_1"
            assert "zone_id" not in ctx

    def test_set_context(self):
        clear_context()

        set_context(user_id="user_456")
        ctx = get_context()
        assert ctx["user_id"] == "user_456"

        clear_context()
        assert get_context() == {}


class TestContextLogger:
    """Tests for ContextLogger"""

    def test_logger_with_context(self):
        logger = get_logger("test")
        assert isinstance(logger, ContextLogger)

    def test_timed_context_manager(self):
        logger = get_logger("test")

        # Just verify it doesn't raise
        with logger.timed("test_operation"):
            pass


class TestGetLogger:
    """Tests for get_logger function"""

    def test_get_logger(self):
        logger = get_logger("my.module")
        assert logger is not None
        assert isinstance(logger, ContextLogger)

    def test_same_logger_returned(self):
        logger1 = get_logger("same.name")
        logger2 = get_logger("same.name")
        # Should be same underlying logger
        assert logger1.logger is logger2.logger


class TestGenerateRequestId:
    """Tests for generate_request_id"""

    def test_generates_unique_ids(self):
        ids = [generate_request_id() for _ in range(100)]
        assert len(set(ids)) == 100  # All unique

    def test_id_format(self):
        req_id = generate_request_id()
        assert len(req_id) == 8
        # Should be valid hex
        assert all(c in "0123456789abcdef-" for c in req_id)


class TestConfigureLogging:
    """Tests for configure_logging"""

    def test_configure_json(self):
        configure_logging(level="DEBUG", format=LogFormat.JSON)
        logger = get_logger("test.json")
        # Just verify it doesn't raise
        logger.info("Test JSON log")

    def test_configure_plain(self):
        configure_logging(level="INFO", format=LogFormat.PLAIN)
        logger = get_logger("test.plain")
        logger.info("Test plain log")

    def test_configure_colored(self):
        configure_logging(level="WARNING", format=LogFormat.COLORED)
        logger = get_logger("test.colored")
        logger.warning("Test colored log")
