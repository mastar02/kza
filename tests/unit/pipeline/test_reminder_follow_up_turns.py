"""A held reminder notice never talks over a follow-up command (T034-U2 review round 3, Q33-1).

The follow-up window's timer ends the FollowUpMode conversation on its own; a
command the person started inside the window may still be being said, then
transcribed and answered. The audio loop's CommandTurns hold the output-free
round until that command is done.
"""

import asyncio
import sys
import time
from unittest.mock import AsyncMock, MagicMock

sys.modules.setdefault('sounddevice', MagicMock())
sys.modules.setdefault('soundfile', MagicMock())
sys.modules.setdefault('pyaudio', MagicMock())
sys.modules.setdefault('torch', MagicMock())
sys.modules.setdefault('torch.cuda', MagicMock())

import numpy as np
import pytest
import pytest_asyncio

from src.conversation.command_turns import CommandTurns
from src.conversation.follow_up_mode import FollowUpMode
from src.core.voice_delivery import DeliveryState, GenerationState, ResponseResult, ResponseStage
from src.pipeline.audio_loop import AudioLoop
from src.pipeline.command_event import CommandEvent
from src.pipeline.multi_room_audio_loop import CHUNK_SIZE, MultiRoomAudioLoop, RoomStream
from src.reminders.reminder_scheduler import ReminderScheduler
from src.reminders.reminder_store import ReminderScope, ReminderState, ReminderStore
from tests.mocks.home_control import selected_actions
from tests.mocks.responses import delivered, voice_output

GAP = 60


def output_busy() -> ResponseResult:
    """Not started: the output stayed owned by another response past the queue budget."""
    return ResponseResult("x", GenerationState.COMPLETE, DeliveryState.TIMED_OUT, spoken_segments=0,
                          timeout_stage=ResponseStage.QUEUE, output_started=False)


def echo_suppressor() -> MagicMock:
    echo = MagicMock()
    echo.is_safe_to_listen = True
    echo.should_process_audio = MagicMock(return_value=(True, "ok"))
    echo.is_human_voice = MagicMock(return_value=True)
    echo.config = MagicMock()
    echo.config.post_speech_buffer_ms = 400
    return echo


class MultiRoom:
    """The production multi-room loop with one room and no wake word heard (the person keeps talking)."""

    def __init__(self, follow_up: FollowUpMode):
        detector = MagicMock()
        detector.detect = MagicMock(return_value=None)
        detector.get_active_models = MagicMock(return_value=[])
        self.rs = RoomStream(room_id="living", device_index=0, wake_detector=detector,
                             echo_suppressor=echo_suppressor())
        self.loop = MultiRoomAudioLoop(room_streams={"living": self.rs}, follow_up=follow_up)
        self._process = self.loop._make_audio_processor(self.rs)

    def speak(self) -> None:
        self._process(np.full((CHUNK_SIZE, 1), 0.05, dtype=np.float32), CHUNK_SIZE, None, None)

    @property
    def listening(self) -> bool:
        return self.rs.listening

    async def step(self, event: CommandEvent | None) -> None:
        """One capture step that ends the capture: hands ``event`` over, or discards it (None)."""
        async def ended(*, handoff=None):
            self.loop._reset_listening(self.rs)
            return event
        self.loop._capture_rooms["living"].step = ended
        await self.loop._consume_once(self.rs)

    @property
    def dispatches(self) -> set:
        return self.loop._dispatch_tasks


class SingleRoom:
    """The single-room fallback AudioLoop, no wake word heard (the person keeps talking)."""

    def __init__(self, follow_up: FollowUpMode):
        manager = MagicMock()
        manager.detect_wake_word = MagicMock(return_value=False)
        self.loop = AudioLoop(manager, echo_suppressor(), follow_up)

    def speak(self) -> None:
        self.loop._process_audio(np.full((1280, 1), 0.05, dtype=np.float32), 1280, time.time(), None)

    @property
    def listening(self) -> bool:
        return self.loop._state.listening

    async def step(self, event: CommandEvent | None) -> None:
        async def ended(*, handoff=None):
            self.loop._clear_capture()
            return event
        self.loop._room.step = ended
        await self.loop._consume_once()

    @property
    def dispatches(self) -> set:
        return self.loop._dispatches


class Clock:
    def __init__(self):
        self.now = time.time()

    def __call__(self) -> float:
        return self.now


@pytest_asyncio.fixture
async def store(tmp_path):
    s = ReminderStore(str(tmp_path / "turns.db"))
    await s.initialize()
    yield s
    await s.close()


@pytest.fixture
def follow_up():
    mode = FollowUpMode(follow_up_window=600)  # only the test ends it, as the window timer would
    yield mode
    mode.end_conversation("test_teardown")


async def held_notice(store, follow_up, audio, clock):
    """A notice that lost the output to KZA's answer to the wake word: reopened, held by the window."""
    output = voice_output()
    output.aspeak.return_value = output_busy()
    scheduler = ReminderScheduler(store, output, None, AsyncMock(), {"output_zone": "living",
                                  "notice_retry_min_gap_s": GAP}, clock=clock,
                                  home_actions=selected_actions(AsyncMock(), ("light.patio",)))
    scheduler.watch_conversation(follow_up, audio.loop.command_turns)
    r = await store.create("u1", "regar", clock() - 1, scope=ReminderScope.SHARED)
    await scheduler._deliver_due(clock())
    follow_up.start_conversation()  # the wake word
    scheduler.notify_output_available()  # KZA's answer ended
    assert scheduler._holds[r.id].output_round
    return scheduler, output, r


@pytest.mark.parametrize("loop_kind", [MultiRoom, SingleRoom])
@pytest.mark.asyncio
async def test_a_follow_up_command_still_being_said_when_the_window_ends_holds_the_notice(
    store, follow_up, loop_kind,
):
    """The person starts a follow-up command inside the window and is still talking when its timer
    ends the conversation: the notice waits while the command is said and answered, then speaks."""
    clock, audio = Clock(), loop_kind(follow_up)
    scheduler, output, r = await held_notice(store, follow_up, audio, clock)
    audio.speak()  # no wake word: the follow-up window takes it as a command
    assert audio.listening
    follow_up.end_conversation("timeout")  # the window's timer, while the person is still talking
    assert scheduler._wake_event.is_set()
    await scheduler._deliver_due(clock())
    assert output.aspeak.await_count == 1
    assert await scheduler._next_timeout() == GAP  # no spin: only the backoff the round replaced
    answered = asyncio.Event()

    async def answer(event):
        await answered.wait()
        return {"success": True, "text": "apaga la luz", "intent": "domotics"}

    audio.loop.on_command(answer)
    scheduler._wake_event.clear()
    await audio.step(CommandEvent(audio=np.zeros(16000, dtype=np.float32), room_id="living"))
    assert not audio.listening and len(audio.dispatches) == 1
    assert not scheduler._wake_event.is_set()  # the command is being answered: still held
    await scheduler._deliver_due(clock())
    assert output.aspeak.await_count == 1
    scheduler._wake_event.clear()  # a direct _deliver_due sets it; only the end of the answer may set it now
    answered.set()
    await asyncio.gather(*audio.dispatches)
    await asyncio.sleep(0)  # the dispatch's done callbacks
    assert not audio.dispatches
    assert scheduler._wake_event.is_set()
    assert await scheduler._next_timeout() == 0
    output.aspeak.return_value = delivered()
    await scheduler._deliver_due(clock())
    assert output.aspeak.await_count == 2
    assert (await store.get_by_id(r.id)).state is ReminderState.FIRED


@pytest.mark.parametrize("loop_kind", [MultiRoom, SingleRoom])
@pytest.mark.asyncio
async def test_a_discarded_follow_up_capture_releases_the_notice(store, follow_up, loop_kind):
    """A follow-up capture that ends without a command (noise, too short) releases the round at the
    end of that capture step."""
    clock, audio = Clock(), loop_kind(follow_up)
    scheduler, output, r = await held_notice(store, follow_up, audio, clock)
    audio.speak()
    follow_up.end_conversation("timeout")
    await scheduler._deliver_due(clock())
    assert output.aspeak.await_count == 1
    scheduler._wake_event.clear()
    await audio.step(None)
    assert not audio.listening and not audio.dispatches
    assert scheduler._wake_event.is_set()
    output.aspeak.return_value = delivered()
    await scheduler._deliver_due(clock())
    assert output.aspeak.await_count == 2
    assert (await store.get_by_id(r.id)).state is ReminderState.FIRED


def test_an_idle_loop_wakes_nobody():
    """Settling is called after every capture step: it notifies only activity an observer saw."""
    busy, seen = [False], []
    turns = CommandTurns(lambda: busy[0])
    turns.add_end_listener(lambda: seen.append("end"))
    for _ in range(3):
        turns.settle()
    busy[0] = True
    turns.settle()  # busy but nobody looked
    busy[0] = False
    turns.settle()
    assert seen == []
    busy[0] = True
    assert turns.active
    turns.settle()  # still busy
    assert seen == []
    busy[0] = False
    assert not turns.active
    turns.settle()
    turns.settle()
    assert seen == ["end"]


def test_a_failing_turn_listener_breaks_neither_the_other_listeners_nor_the_loop(caplog):
    busy, seen = [True], []
    turns = CommandTurns(lambda: busy[0])

    def broken():
        raise RuntimeError("listener bug")

    turns.add_end_listener(broken)
    turns.add_end_listener(lambda: seen.append("second"))
    assert turns.active
    busy[0] = False
    turns.settle()
    assert seen == ["second"]
    assert "Command turn end listener failed: RuntimeError" in caplog.text
