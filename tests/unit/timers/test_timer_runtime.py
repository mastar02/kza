"""T030-U2: timer runtime on a virtual clock and a fake voice output.

Deadlines (monotonic clock + wall-clock ends_at) instead of per-tick
decrements; warnings and rings in owned tasks off the tick loop; a ring stops
on a voice stop and on a barge-in; a failed or partial ResponseResult leaves a
ring_failed/undelivered receipt, never a delivered one; stop() leaves no task.
"""

import asyncio
import heapq

import pytest

from src.core.voice_delivery import (
    DeliveryState,
    GenerationState,
    ResponseResult,
    ResponseStage,
    VoiceDeliveryError,
)
from src.timers.named_timers import (
    RING_GAP_S,
    RING_REPEATS,
    NamedTimerManager,
    RingDelivery,
    TimerState,
)

START = 1000.0
WALL_OFFSET = 1_800_000_000.0 - START
RING_S = 2.0  # the simulated length of one spoken ring


class VirtualClock:
    """Monotonic time that moves only when the test advances it; sleepers wake in time order."""

    def __init__(self) -> None:
        self.now = START
        self._sleepers: list[tuple[float, int, asyncio.Future]] = []
        self._seq = 0

    def __call__(self) -> float:
        return self.now

    def wall(self) -> float:
        return self.now + WALL_OFFSET

    async def sleep(self, seconds: float) -> None:
        future = asyncio.get_running_loop().create_future()
        self._seq += 1
        heapq.heappush(self._sleepers, (self.now + max(0.0, seconds), self._seq, future))
        await future

    @staticmethod
    async def settle() -> None:
        for _ in range(30):
            await asyncio.sleep(0)

    async def advance(self, seconds: float) -> None:
        target = self.now + seconds
        await self.settle()
        for _ in range(100_000):
            if not self._sleepers or self._sleepers[0][0] > target:
                break
            wake, _, future = heapq.heappop(self._sleepers)
            self.now = max(self.now, wake)
            if not future.done():
                future.set_result(None)
                await self.settle()
        else:
            raise AssertionError("virtual clock spun")
        self.now = target
        await self.settle()


def _result(delivery=DeliveryState.COMPLETE, *, started=True, stage=None) -> ResponseResult:
    generation = GenerationState.CANCELLED if delivery is DeliveryState.CANCELLED else GenerationState.COMPLETE
    error = None if delivery is DeliveryState.COMPLETE else "x"
    return ResponseResult("x", generation, delivery, error, spoken_segments=0, timeout_stage=stage,
                          output_started=started)


class FakeOutput:
    """A voice output that speaks each sentence for RING_S virtual seconds; it can be barged in."""

    def __init__(self, clock: VirtualClock, results=None) -> None:
        self.clock = clock
        self.events: list[tuple[float, str, str, str]] = []
        self.results = list(results or [])
        self._interrupt: asyncio.Future | None = None

    async def aspeak(self, text: str, zone_id: str | None = None) -> ResponseResult:
        if self.results and isinstance(self.results[0], BaseException):
            raise self.results.pop(0)
        if self.results and not self.results[0].success and not self.results[0].output_started:
            self.events.append((self.clock.now, "refused", text, zone_id))
            return self.results.pop(0)
        self.events.append((self.clock.now, "start", text, zone_id))
        self._interrupt = asyncio.get_running_loop().create_future()
        sleeper = asyncio.ensure_future(self.clock.sleep(RING_S))
        try:
            await asyncio.wait({sleeper, self._interrupt}, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            self.events.append((self.clock.now, "cut", text, zone_id))
            raise
        finally:
            sleeper.cancel()
        if self._interrupt.done():
            self.events.append((self.clock.now, "barged", text, zone_id))
            return _result(DeliveryState.CANCELLED, started=True)
        self.events.append((self.clock.now, "end", text, zone_id))
        return self.results.pop(0) if self.results else _result()

    def barge_in(self) -> None:
        assert self._interrupt is not None and not self._interrupt.done()
        self._interrupt.set_result(None)

    def starts(self, fragment: str = "") -> list[float]:
        return [round(t - START, 6) for t, kind, text, _ in self.events if kind == "start" and fragment in text]


async def _manager(results=None, *, start=True):
    clock = VirtualClock()
    output = FakeOutput(clock, results)
    timers = NamedTimerManager(tts_callback=output.aspeak, clock=clock, wall_clock=clock.wall, sleep=clock.sleep)
    if start:
        await timers.start()
    return clock, output, timers


def _manager_tasks() -> list[asyncio.Task]:
    return [t for t in asyncio.all_tasks() if not t.done()
            and t.get_coro().__qualname__.startswith("NamedTimerManager.")]


async def test_concurrent_timers_keep_their_deadlines_while_one_rings():
    """A 2 s ring no longer delays the others: each starts ringing at its own deadline."""
    clock, output, timers = await _manager()
    try:
        first = timers.create_timer("pasta", 10)
        second = timers.create_timer("salsa", 11)
        third = timers.create_timer("arroz", 13)
        assert [t.ends_at for t in (first, second, third)] == [clock.wall() + d for d in (10, 11, 13)]
        await clock.advance(40)
        starts = {name: output.starts(name) for name in ("pasta", "salsa", "arroz")}
        gap = RING_S + RING_GAP_S
        for name, due in (("pasta", 10), ("salsa", 11), ("arroz", 13)):
            assert starts[name] == pytest.approx([due + k * gap for k in range(RING_REPEATS)], abs=1e-6)
        assert {t.state for t in (first, second, third)} == {TimerState.COMPLETED}
        assert {t.ring for t in (first, second, third)} == {RingDelivery.DELIVERED}
    finally:
        await timers.stop()


async def test_remaining_time_follows_the_deadline_not_the_ticks():
    """Without any tick (loop not running) the time left still falls; pause and resume keep it exact."""
    clock, _, timers = await _manager(start=False)
    timer = timers.create_timer("pasta", 60)
    await clock.advance(20)
    assert timer.remaining_seconds == pytest.approx(40)
    assert timers.pause_timer(timer.timer_id)
    assert timer.ends_at is None and timer.deadline is None
    await clock.advance(100)
    assert timer.remaining_seconds == pytest.approx(40)
    assert timers.resume_timer(timer.timer_id)
    assert timer.ends_at == pytest.approx(clock.wall() + 40)
    await clock.advance(10)
    assert timer.remaining_seconds == pytest.approx(30)
    assert timers.cancel_timer(timer.timer_id)
    await clock.advance(10)
    assert timer.remaining_seconds == pytest.approx(30) and timer.ends_at is None


async def test_a_voice_stop_during_the_ring_cuts_it_and_prevents_every_repeat():
    clock, output, timers = await _manager()
    try:
        timer = timers.create_timer("pasta", 5)
        await clock.advance(6)  # one second into the first repetition
        assert [t.timer_id for t in timers.ringing_timers()] == [timer.timer_id]
        assert await timers.stop_ring(timer.timer_id) is True
        assert output.events[-1][:2] == (START + 6, "cut")
        await clock.advance(60)
        assert output.starts() == [5.0]
        assert timer.ring is RingDelivery.STOPPED and timer.state is TimerState.COMPLETED
        assert timers.ringing_timers() == [] and await timers.stop_ring(timer.timer_id) is False
        assert timers.ringing_timers(within_s=30) == []  # 60 s after: no longer a follow-up target
    finally:
        await timers.stop()


async def test_a_voice_stop_between_repetitions_prevents_the_next_one():
    clock, output, timers = await _manager()
    try:
        timer = timers.create_timer("", 5)
        await clock.advance(5 + RING_S + RING_GAP_S / 2)  # first repetition done, inside the gap
        assert await timers.stop_ring(timer.timer_id) is True
        await clock.advance(60)
        assert output.starts() == [5.0] and [e[1] for e in output.events] == ["start", "end"]
        assert timer.ring is RingDelivery.STOPPED
    finally:
        await timers.stop()


async def test_a_barge_in_on_the_ring_stops_it():
    clock, output, timers = await _manager()
    try:
        timer = timers.create_timer("pasta", 5)
        await clock.advance(5.5)
        output.barge_in()  # ResponseHandler.cancel(zone): CANCELLED after the first PCM chunk
        await clock.advance(1)
        assert timer.ring is RingDelivery.STOPPED and timers.ringing_timers() == []
        # The command that follows the barge-in still names this ring (voice_commands RING_FOLLOW_UP_S).
        assert [t.timer_id for t in timers.ringing_timers(within_s=30)] == [timer.timer_id]
        await clock.advance(60)
        assert output.starts() == [5.0] and [e[1] for e in output.events] == ["start", "barged"]
    finally:
        await timers.stop()


async def test_a_repetition_swept_before_any_audio_does_not_end_the_ring():
    """Cancelled while queued (no output started): nobody heard it, so the ring goes on."""
    clock, output, timers = await _manager([_result(DeliveryState.CANCELLED, started=False)])
    try:
        timer = timers.create_timer("", 5)
        await clock.advance(60)
        assert [e[1] for e in output.events].count("refused") == 1
        assert len(output.starts()) == RING_REPEATS - 1 and timer.ring is RingDelivery.DELIVERED
    finally:
        await timers.stop()


@pytest.mark.parametrize(("first", "receipt"), [
    (_result(DeliveryState.FAILED, started=False), RingDelivery.UNDELIVERED),
    (_result(DeliveryState.BLOCKED, started=False), RingDelivery.UNDELIVERED),
    (_result(DeliveryState.TIMED_OUT, started=False, stage=ResponseStage.QUEUE), RingDelivery.UNDELIVERED),
    (VoiceDeliveryError("voice_output_zone_required"), RingDelivery.UNDELIVERED),
    (_result(DeliveryState.TIMED_OUT, started=False, stage=ResponseStage.DELIVERY), RingDelivery.RING_FAILED),
    (RuntimeError("boom"), RingDelivery.RING_FAILED),
])
async def test_a_failed_ring_is_never_a_delivered_completion(first, receipt):
    clock, output, timers = await _manager([first])
    try:
        timer = timers.create_timer("pasta", 5)
        await clock.advance(60)
        assert timer.state is TimerState.COMPLETED and timer.ring is receipt
        assert output.starts() == []  # a failure ends the ring: it is not repeated
        status = timers.get_status()
        assert (status["ring_not_heard"], status["ringing"]) == (1, 0)
    finally:
        await timers.stop()


async def test_a_ring_that_breaks_after_its_audio_started_is_ring_failed():
    clock, output, timers = await _manager([_result(DeliveryState.FAILED, started=True)])
    try:
        timer = timers.create_timer("pasta", 5)
        await clock.advance(60)
        assert output.starts() == [5.0] and timer.ring is RingDelivery.RING_FAILED
    finally:
        await timers.stop()


async def test_no_audio_after_a_timer_is_cancelled_during_its_warning():
    clock, output, timers = await _manager()
    try:
        timer = timers.create_timer("pasta", 180)
        await clock.advance(121)  # the 1-minute warning started at 120 s and is still speaking
        assert output.starts("le queda 1 minuto") == [120.0]
        assert timers.cancel_timer(timer.timer_id)
        await clock.advance(1)
        assert output.events[-1][:2] == (START + 121, "cut")
        await clock.advance(300)
        assert output.starts() == [120.0] and timer.ring is RingDelivery.PENDING
        assert timers.get_status()["delivery_tasks"] == 0
    finally:
        await timers.stop()


async def test_stop_cancels_and_awaits_every_warning_and_ring():
    clock, output, timers = await _manager()
    ringing = timers.create_timer("pasta", 125)
    warning = timers.create_timer("salsa", 185)  # its 1-minute warning comes due when pasta rings
    await clock.advance(125.5)  # pasta rings, salsa's warning is speaking
    assert len(_manager_tasks()) == 3 and timers.get_status()["delivery_tasks"] == 2
    await timers.stop()
    assert _manager_tasks() == [] and timers.running is False
    assert (timers._deliveries, timers._rings, timers._update_task.done()) == ({}, {}, True)
    assert [e[1] for e in output.events].count("cut") == 2
    spoken = len(output.events)
    await clock.advance(300)
    assert len(output.events) == spoken
    assert ringing.ring is RingDelivery.UNDELIVERED and warning.state is TimerState.RUNNING


async def test_a_warning_found_too_late_is_skipped_and_the_ring_still_comes():
    """A stalled loop must not say 'le queda 1 minuto' with seconds left."""
    clock, output, timers = await _manager(start=False)
    timers.create_timer("pasta", 180)
    await clock.advance(170)
    timers._tick()
    await clock.advance(0)
    assert output.starts() == []
    await clock.advance(10)
    timers._tick()
    await clock.advance(0)
    assert output.starts() == [180.0]
    await timers.stop()


async def test_a_repeating_timer_rearms_from_its_old_deadline():
    clock, output, timers = await _manager()
    try:
        timer = timers.create_timer("", 10)
        timer.repeat = True
        await clock.advance(35)
        assert output.starts("Terminó")[::RING_REPEATS] == [10.0, 20.0, 30.0]
        assert timer.state is TimerState.RUNNING and timer.remaining_seconds == pytest.approx(5)
        assert timers.cancel_timer(timer.timer_id)
        await clock.advance(60)
        assert max(output.starts()) < 35
    finally:
        await timers.stop()
