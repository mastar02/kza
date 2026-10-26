"""Reconstruct pattern text and statistics from verified activity metadata."""

import hashlib
import json
import re

from src.analytics.event_evidence import PatternInputs
from src.analytics.pattern_analyzer import Pattern, PatternAnalyzer, PatternType


def pattern_spec(pattern: Pattern) -> dict:
    """Persist only structured selection; arbitrary descriptions are not sources."""
    return {
        "type": pattern.pattern_type.value,
        "entity_id": pattern.entity_id,
        "action": pattern.action,
        "hour": pattern.hour,
        "minute": pattern.minute,
        "weekdays": pattern.weekdays,
        "trigger_entity": pattern.trigger_entity,
        "trigger_action": pattern.trigger_action,
    }


def pattern_key(pattern: Pattern) -> str:
    """Deduplicate the complete automation selection, including its trigger."""
    return hashlib.sha256(
        json.dumps(pattern_spec(pattern), sort_keys=True).encode()
    ).hexdigest()


def project_pattern(spec: dict, inputs: PatternInputs) -> Pattern:
    """Verify a selection and regenerate its description from actual event rows."""
    if type(spec) is not dict or set(spec) != {
        "type",
        "entity_id",
        "action",
        "hour",
        "minute",
        "weekdays",
        "trigger_entity",
        "trigger_action",
    }:
        raise ValueError("unknown_suggestion_pattern")
    kind = PatternType(spec["type"])
    for key, expression in (
        ("entity_id", r"[a-z_]+\.[a-z0-9_]+"),
        ("action", r"[a-z_]+"),
    ):
        if type(spec[key]) is not str or re.fullmatch(expression, spec[key]) is None:
            raise ValueError("unknown_suggestion_pattern")
    for key, limit in (("hour", 24), ("minute", 60)):
        if spec[key] is not None and (
            type(spec[key]) is not int or not 0 <= spec[key] < limit
        ):
            raise ValueError("unknown_suggestion_pattern")
    if type(spec["weekdays"]) is not list or any(
        type(day) is not int or not 0 <= day <= 6 for day in spec["weekdays"]
    ):
        raise ValueError("unknown_suggestion_pattern")
    analyzer = PatternAnalyzer(None)
    events = inputs.events
    if kind is PatternType.SEQUENCE:
        if not all(
            (event.entity_id, event.action)
            in {
                (spec["entity_id"], spec["action"]),
                (spec["trigger_entity"], spec["trigger_action"]),
            }
            for event in events
        ):
            raise ValueError("unknown_suggestion_pattern")
        candidates = analyzer._analyze_sequences(events)
    else:
        detectors = {
            PatternType.DAILY_TIME: analyzer._detect_daily_pattern,
            PatternType.WEEKDAY_TIME: analyzer._detect_weekday_pattern,
            PatternType.WEEKEND_TIME: analyzer._detect_weekend_pattern,
        }
        if (
            kind not in detectors
            or len(events) < analyzer.MIN_OCCURRENCES
            or not all(
                (event.entity_id, event.action) == (spec["entity_id"], spec["action"])
                for event in events
            )
        ):
            raise ValueError("unknown_suggestion_pattern")
        candidate = detectors[kind](spec["entity_id"], spec["action"], list(events))
        candidates = [candidate] if candidate is not None else []
    for candidate in candidates:
        if pattern_spec(candidate) == spec:
            candidate.evidence = tuple(event.dependency for event in events)
            return candidate
    raise ValueError("unknown_suggestion_pattern")
