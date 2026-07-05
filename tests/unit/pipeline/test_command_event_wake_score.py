import numpy as np
from src.pipeline.command_event import CommandEvent


def test_command_event_default_wake_score():
    e = CommandEvent(audio=np.zeros(16, dtype=np.float32), room_id="living")
    assert e.wake_score == 1.0


def test_command_event_carries_wake_score():
    e = CommandEvent(audio=np.zeros(16, dtype=np.float32), room_id="living", wake_score=0.81)
    assert e.wake_score == 0.81
