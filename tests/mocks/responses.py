"""Explicit async response contract for router/producer tests without audio."""

from unittest.mock import AsyncMock, MagicMock

from src.pipeline.response_handler import DeliveryState, GenerationState, ResponseResult


def response_handler_mock():
    handler = MagicMock()
    handler.is_responding = handler.is_speaking = False

    async def speak(text, **kwargs):
        handler.speak(text, **kwargs)
        return ResponseResult(text, GenerationState.COMPLETE, DeliveryState.COMPLETE)

    async def earcon(**kwargs):
        handler.play_earcon(**kwargs)
        return ResponseResult('', GenerationState.COMPLETE, DeliveryState.COMPLETE)

    async def respond(prompt, **kwargs):
        text = handler.speak_with_llm_stream(prompt, **kwargs)
        text = text if isinstance(text, str) else 'Respuesta'
        return ResponseResult(text, GenerationState.COMPLETE, DeliveryState.COMPLETE)

    handler.aspeak = AsyncMock(side_effect=speak)
    handler.aplay_earcon = AsyncMock(side_effect=earcon)
    handler.arespond = AsyncMock(side_effect=respond)
    handler.cancel = AsyncMock(return_value=False)
    return handler
