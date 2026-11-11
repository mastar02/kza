"""U3b-4a-2 review T: the verification room check reads the microphone snapshot and the capture pause."""

from types import SimpleNamespace

import pytest

from src.audio.microphone_status import CaptureState
from src.dashboard.api import DashboardAPI


def mic(room_id, state):
    return SimpleNamespace(room_id=room_id, capture=SimpleNamespace(state=state))


def snapshot(*mics, stale=False):
    return SimpleNamespace(stale=stale, microphones=tuple(mics))


def check(provider, capture=None, room="living"):
    api = SimpleNamespace(microphone_snapshot_provider=provider, capture_control=capture)
    return DashboardAPI._voice_room_available(api, room)


ALLOWED = SimpleNamespace(capture_allowed=True)


@pytest.mark.parametrize("state", [CaptureState.WAITING, CaptureState.RECEIVING])
def test_a_fresh_capturing_room_is_available(state):
    assert check(lambda: snapshot(mic("living", state))) is True
    assert check(lambda: snapshot(mic("living", state)), ALLOWED) is True


@pytest.mark.parametrize("state", [CaptureState.UNAVAILABLE, CaptureState.CLOSED, CaptureState.STALE])
def test_a_room_that_is_not_capturing_is_unavailable(state):
    assert check(lambda: snapshot(mic("living", state))) is False


def test_stale_snapshot_paused_capture_other_room_and_no_provider_are_unavailable():
    fresh = snapshot(mic("living", CaptureState.RECEIVING))
    assert check(lambda: snapshot(mic("living", CaptureState.RECEIVING), stale=True)) is False
    for capture in (SimpleNamespace(capture_allowed=False), SimpleNamespace(capture_allowed=None),
                    SimpleNamespace(capture_allowed=1)):
        assert check(lambda: fresh, capture) is False
    assert check(lambda: snapshot(mic("kitchen", CaptureState.RECEIVING), mic("living", CaptureState.CLOSED))) is False
    assert check(lambda: fresh, room="kitchen") is False
    assert check(None) is False
    assert check(lambda: None) is False


def test_a_provider_error_is_refused_by_the_ledger_not_read_as_available():
    """The route passes the check to FactLedger.open_voice_window, which treats any error as unavailable."""
    def broken():
        raise RuntimeError("telemetry down")
    with pytest.raises(RuntimeError):
        check(broken)
