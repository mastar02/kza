"""FollowUpMode end listeners: how a held reminder notice learns that a conversation ended (T034-U2 Q33)."""

import threading

import pytest

from src.conversation.follow_up_mode import ConversationState, FollowUpMode


@pytest.fixture
def follow_up():
    mode = FollowUpMode(follow_up_window=600)
    yield mode
    mode.end_conversation("test_teardown")


def test_listeners_run_once_after_the_conversation_is_idle(follow_up):
    seen = []
    follow_up.add_end_listener(lambda: seen.append(("first", follow_up.is_active, follow_up.state)))
    follow_up.add_end_listener(lambda: seen.append(("second", follow_up.is_active, follow_up.state)))
    follow_up.start_conversation()
    follow_up.start_conversation()  # the person keeps talking: the same conversation goes on
    assert seen == []
    follow_up.end_conversation("timeout")
    assert seen == [("first", False, ConversationState.IDLE), ("second", False, ConversationState.IDLE)]
    follow_up.end_conversation("timeout")  # nothing active: no second end
    assert len(seen) == 2


def test_ending_without_a_conversation_calls_no_listener(follow_up):
    seen = []
    follow_up.add_end_listener(lambda: seen.append(True))
    follow_up.end_conversation("capture_paused")
    assert seen == []


def test_a_failing_listener_breaks_neither_the_end_nor_the_other_listeners(follow_up, caplog):
    seen, legacy = [], []

    def broken():
        raise RuntimeError("listener bug")

    follow_up.on_conversation_end(lambda context, reason: legacy.append(reason))
    follow_up.add_end_listener(broken)
    follow_up.add_end_listener(lambda: seen.append(True))
    follow_up.start_conversation()
    follow_up.end_conversation("timeout")
    assert seen == [True] and legacy == ["timeout"]
    assert follow_up.state is ConversationState.IDLE
    assert "Conversation end listener failed: RuntimeError" in caplog.text


def test_a_timeout_on_the_timer_thread_calls_the_listener_there():
    """Started off the event loop (audio thread), the window ends on a threading.Timer."""
    mode, ended = FollowUpMode(follow_up_window=.05), threading.Event()
    threads = []
    mode.add_end_listener(lambda: (threads.append(threading.current_thread()), ended.set()))
    starter = threading.Thread(target=mode.start_conversation)
    starter.start()
    starter.join(2)
    assert ended.wait(2)
    assert not mode.is_active
    assert threads[0] is not threading.main_thread()
