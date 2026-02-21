# Analytics module - Smart Automations
from .event_logger import EventLogger, Event, EventType
from .pattern_analyzer import PatternAnalyzer, Pattern, PatternType
from .suggestion_engine import SuggestionEngine, Suggestion

__all__ = [
    "EventLogger",
    "Event",
    "EventType",
    "PatternAnalyzer",
    "Pattern",
    "PatternType",
    "SuggestionEngine",
    "Suggestion"
]
