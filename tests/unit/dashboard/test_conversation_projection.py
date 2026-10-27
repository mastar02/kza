"""Conversation transport keeps source deadlines and a bounded scalar shape."""

from src.analytics.event_logger import Event, EventType
from src.dashboard.conversation_projection import ConversationProjection
from tests.mocks.memory_sources import memory_source


def event(**fields):
    """Create a sourced event without a database or a fabricated authority."""
    source = memory_source(now=1000, ttl=60)
    return Event(timestamp=1000, event_type=EventType.COMMAND, entity_id='light.synthetic', action='turn_on',
        trigger_phrase='Synthetic private turn', sources=(source,), expires_at=2000, text_expires_at=1060, record_id=1, **fields)


def test_projection_carries_the_earliest_original_deadline_across_multiple_sources():
    item = event()
    item.sources += (memory_source(now=1000, ttl=10),)
    row = ConversationProjection(item, 1001).to_dict()
    assert row['expires_at'] == 1010 and row['observed_at'] == 1001
    assert ConversationProjection(item, 1010).to_dict() is None


def test_unsourced_or_missing_nonfinite_deadlines_are_not_projected():
    item = event()
    for deadline in (None, float('inf'), float('nan'), True):
        item.text_expires_at = deadline
        assert ConversationProjection(item, 1001).to_dict() is None
    item.text_expires_at = 1060
    item.sources = ()
    assert ConversationProjection(item, 1001).to_dict() is None


def test_unknown_context_and_nested_values_never_enter_the_transport_projection():
    item = event(context={'nested': {'raw': 'Private nested context'}, 'tts': {'raw': 'Invalid response'},
        'temperature': {'raw': 'Invalid parameter'}, 'latency_ms': float('nan'), 'success': 'not a receipt'})
    row = ConversationProjection(item, 1001).to_dict()
    assert row['args'] == {} and row['tts'] == '' and row['latency_ms'] == 0 and row['success'] is None
    assert 'Private nested context' not in str(row)


def test_text_fields_are_bounded_and_user_names_cannot_override_source_deadlines():
    item = event(user_name='x'*1000, context={'tts':'y'*10000})
    item.trigger_phrase = 'z'*10000
    row = ConversationProjection(item, 1001).to_dict()
    assert len(row['user']) == 256 and len(row['tts']) == len(row['stt']) == 8192
    assert row['expires_at'] == 1060
