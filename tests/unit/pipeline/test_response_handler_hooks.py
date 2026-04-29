"""Tests for ResponseHandler integration with plugin hooks (plan #3 OpenClaw)."""

from unittest.mock import MagicMock

import pytest

from src.hooks import BlockResult, HookRegistry, RewriteResult, TtsCall
from src.pipeline.response_handler import ResponseHandler


@pytest.fixture
def handler_with_hooks():
    """Minimal ResponseHandler with mocked TTS + fresh HookRegistry."""
    hooks = HookRegistry()
    tts = MagicMock()
    tts.sample_rate = 24000
    tts.synthesize = MagicMock(return_value=(b"\x00" * 100, 24000))
    tts.speak = MagicMock(return_value=None)
    # Streaming path attrs absent so _speak_direct falls through to tts.speak()
    if hasattr(tts, "speak_stream"):
        del tts.speak_stream
    handler = ResponseHandler(tts=tts, streaming_enabled=False, hooks=hooks)
    return handler, hooks, tts


def test_block_prevents_tts(handler_with_hooks):
    handler, hooks, tts = handler_with_hooks

    def block(call):
        return BlockResult(reason="silence", rule_name="test")

    hooks.register_before("before_tts_speak", block, priority=10)
    handler.speak("hello")

    # Neither speak() nor synthesize() should have been invoked.
    assert tts.speak.call_count == 0
    assert tts.synthesize.call_count == 0


def test_rewrite_modifies_text_before_tts(handler_with_hooks):
    handler, hooks, tts = handler_with_hooks
    received_text = []

    def capture(text, *a, **kw):
        received_text.append(text)
        return None

    tts.speak = MagicMock(side_effect=capture)

    def upper(call):
        return RewriteResult(
            modified=TtsCall(
                text=call.text.upper(),
                voice=call.voice,
                lang=call.lang,
                user_id=call.user_id,
                zone_id=call.zone_id,
            ),
            rule_name="upper",
        )

    hooks.register_before("before_tts_speak", upper, priority=10)
    handler.speak("hola")

    assert received_text and "HOLA" in str(received_text[0])


def test_no_hooks_unchanged(handler_with_hooks):
    """With no handlers registered, speak() does its normal work."""
    handler, hooks, tts = handler_with_hooks
    # No handlers registered

    handler.speak("hola")
    assert tts.speak.call_count == 1


def test_no_hooks_attribute_unchanged():
    """ResponseHandler with hooks=None behaves identically to baseline."""
    tts = MagicMock()
    tts.sample_rate = 24000
    tts.speak = MagicMock(return_value=None)
    if hasattr(tts, "speak_stream"):
        del tts.speak_stream

    handler = ResponseHandler(tts=tts, streaming_enabled=False, hooks=None)
    handler.speak("hola")
    assert tts.speak.call_count == 1
