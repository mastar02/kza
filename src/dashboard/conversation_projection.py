"""Bounded conversation snapshots with the original capture deadline."""

from dataclasses import dataclass
import math

from src.analytics.event_logger import Event
from src.memory.fact_ledger import sources_current


def _text(value, limit: int = 256) -> str:
    return value[:limit] if type(value) is str else ""


@dataclass(frozen=True)
class ConversationProjection:
    """Build a detached scalar projection without forwarding context objects."""

    event: Event
    observed_at: float

    def to_dict(self) -> dict | None:
        """Return a bounded snapshot only while every original source is current."""
        event, now = self.event, self.observed_at
        if (type(event.record_id) is not int or event.record_id <= 0
                or not sources_current(event.sources, now) or not event.trigger_phrase):
            return None
        deadlines = (event.expires_at, event.text_expires_at, *(source.expires_at for source in event.sources))
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in deadlines):
            return None
        expiry = min(deadlines)
        if now >= expiry:
            return None
        context = event.context if type(event.context) is dict else {}
        entity, action = _text(event.entity_id), _text(event.action)
        domain = entity.split(".", 1)[0] if "." in entity else "—"
        latency = context.get("latency_ms", 0)
        if type(latency) not in (int, float) or not math.isfinite(latency) or latency < 0:
            latency = 0
        args = {}
        for key in ("brightness_pct", "color_temp_kelvin", "temperature", "hvac_mode", "position"):
            value = context.get(key)
            if (type(value) in (int, float) and math.isfinite(value)
                    or type(value) is str and len(value) <= 256):
                args[key] = value
        return {
            "id": f"turn_{event.record_id}_{event.sources[0].capture_id}",
            "ts": event.datetime.strftime("%H:%M:%S"),
            "user": _text(event.user_name or event.user_id) or "—",
            "zone": _text(context.get("zone") or context.get("room_id")) or "—",
            "path": context.get("path") if context.get("path") in ("fast", "slow", "music") else "unknown",
            "stt": _text(event.trigger_phrase, 8192), "tts": _text(context.get("tts"), 8192),
            "intent": f"{domain}.{action}", "target": entity, "args": args,
            "latency_ms": latency, "success": context.get("success") if type(context.get("success")) is bool else None,
            "observed_at": now, "expires_at": expiry,
        }
