"""Voice delivery outcomes and awaited callbacks shared by local producers."""

import asyncio
import inspect
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable


class GenerationState(StrEnum):
    INCOMPLETE = "incomplete"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DeliveryState(StrEnum):
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"
    EMPTY = "empty"


@dataclass(frozen=True)
class ResponseResult:
    """Keep generation validity separate from actual voice delivery."""

    text: str
    generation: GenerationState
    delivery: DeliveryState
    error: str | None = None
    speech_handled: bool = True
    spoken_segments: int = 0

    @property
    def success(self) -> bool:
        return self.generation is GenerationState.COMPLETE and self.delivery is DeliveryState.COMPLETE


class VoiceDeliveryError(RuntimeError):
    """The configured voice output did not confirm completed delivery."""


async def invoke_voice_callback(callback: Callable[..., Any] | None, *args, **kwargs) -> Any:
    """Await async callbacks and keep legacy synchronous callbacks off-loop.

    The application callbacks return ResponseResult, whose failed/blocked/
    cancelled delivery must propagate to the producer. Legacy callbacks that
    return no receipt retain their existing acknowledgement convention.
    """
    if callback is None:
        raise VoiceDeliveryError("voice_output_unavailable")
    if inspect.iscoroutinefunction(callback):
        result = await callback(*args, **kwargs)
    else:
        result = await asyncio.to_thread(callback, *args, **kwargs)
        if inspect.isawaitable(result):
            result = await result
    if isinstance(result, ResponseResult) and not result.success:
        raise VoiceDeliveryError(result.error or "voice_delivery_" + result.delivery.value)
    if result is False:
        raise VoiceDeliveryError("voice_delivery_failed")
    return result


@dataclass(frozen=True)
class LocalVoiceDelivery:
    """Bind background speech to a configured route instead of an active room."""

    handler: Any
    default_zone: str | None = None
    require_zone: bool = False

    async def aspeak(self, text: str, zone_id: str | None = None) -> ResponseResult:
        """Deliver to the captured producer zone or its fixed configured default."""
        if zone_id == "all":
            return await self.handler.aspeak_to_all_zones(text)
        target = self.default_zone if zone_id in (None, "default") else zone_id
        if target is None and self.require_zone:
            raise VoiceDeliveryError("voice_output_zone_required")
        return await self.handler.aspeak(text, zone_id=target)
