"""NLU helpers: intent classifier + slot extractors + streaming parser."""
from src.nlu.slot_extractor import (
    classify_intent,
    extract_slots,
    merge_service_data,
    SLOT_BRIGHTNESS,
    SLOT_COLOR,
    SLOT_COLOR_TEMP,
)
from src.nlu.command_grammar import (
    PartialCommand,
    parse_partial_command,
    extract_entity,
    extract_room,
    has_wake_word,
    ENTITY_TERMS,
    ROOM_ALIASES,
    WAKE_TERMS,
)

__all__ = [
    "classify_intent",
    "extract_slots",
    "merge_service_data",
    "SLOT_BRIGHTNESS",
    "SLOT_COLOR",
    "SLOT_COLOR_TEMP",
    "PartialCommand",
    "parse_partial_command",
    "extract_entity",
    "extract_room",
    "has_wake_word",
    "ENTITY_TERMS",
    "ROOM_ALIASES",
    "WAKE_TERMS",
]
