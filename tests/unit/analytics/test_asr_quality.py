from src.analytics.asr_quality import log_asr_outcome


class _FakeEventLogger:
    def __init__(self):
        self.calls = []

    def log(self, entity_id, action, event_type=None, trigger_phrase=None, extra_context=None):
        self.calls.append(
            dict(entity_id=entity_id, action=action, trigger_phrase=trigger_phrase,
                 extra_context=extra_context)
        )


def test_logs_one_event_with_room_and_reason():
    el = _FakeEventLogger()
    log_asr_outcome(el, room_id="living", outcome="gate_rejected",
                    reason="empty", text="", signals={"compression_ratio": None},
                    wake_score=0.81, rms=0.03)
    assert len(el.calls) == 1
    call = el.calls[0]
    assert call["entity_id"] == "asr_quality:living"
    assert call["action"] == "gate_rejected:empty"
    assert call["extra_context"]["wake_score"] == 0.81
    assert call["extra_context"]["rms"] == 0.03


def test_truncates_text_to_60_chars():
    el = _FakeEventLogger()
    log_asr_outcome(el, room_id="cocina", outcome="accepted", reason="ok",
                    text="x" * 200, signals={}, wake_score=1.0, rms=0.1)
    assert len(el.calls[0]["trigger_phrase"]) == 60


def test_none_event_logger_is_noop():
    # No debe explotar si no hay logger (fail-open): no exception.
    log_asr_outcome(None, room_id="hall", outcome="accepted", reason="ok",
                    text="dale", signals={}, wake_score=1.0, rms=0.1)
