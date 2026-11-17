"""Two-stage wake verification in shadow mode (V2, voice-diag 2026-11-17)."""

import json
import logging
import os
import stat
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

sys.modules.setdefault("sounddevice", MagicMock())
sys.modules.setdefault("soundfile", MagicMock())
sys.modules.setdefault("pyaudio", MagicMock())
sys.modules.setdefault("torch", MagicMock())
sys.modules.setdefault("torch.cuda", MagicMock())

import numpy as np
import pytest
import yaml

from tests.mocks.responses import response_handler_mock
from src.nlu.command_gate import CommandAcceptanceGate, contains_wake_word
from src.pipeline.command_event import CommandEvent
from src.pipeline.multi_room_audio_loop import CHUNK_SIZE, MultiRoomAudioLoop, RoomStream
from src.pipeline.request_router import RequestRouter
from src.pipeline.wake_verification import (
    STATUS_SCHEMA, WakeCheckCode, WakeVerificationConfig, WakeVerificationMode, WakeVerifier,
)

ROOT = Path(__file__).resolve().parents[3]
WORDS = ["nexa", "alexa", "nexia", "nexta", "kaza", "neska", "nexca"]
SECRET = "prendé la lámpara azul"


def _verifier(mode="shadow", tmp_path=None, clock=None):
    cfg = WakeVerificationConfig.from_dict(
        {"mode": mode, "status_path": str(tmp_path / "wv.json") if tmp_path else None, "flush_interval_s": 0},
        WORDS)
    return WakeVerifier(cfg, clock=clock) if clock else WakeVerifier(cfg)


def test_shadow_marks_missing_wake_without_rejecting(caplog):
    verifier = _verifier()
    with caplog.at_level(logging.INFO):
        check = verifier.evaluate(SECRET, room_id="escritorio", wake_score=0.47, wake_vad=0.03, gate_accept=True)
    assert check.code is WakeCheckCode.WOULD_REJECT and check.reject is False
    line = next(r.getMessage() for r in caplog.records if "[wake-verify]" in r.getMessage())
    assert "code=would_reject" in line and "room=escritorio" in line and "score=0.47" in line and "vad=0.03" in line
    # Privacy: never the transcript.
    assert "lámpara" not in caplog.text and "prend" not in caplog.text


@pytest.mark.parametrize("text", ["Nexa, prendé la luz", "Alexa apagá todo", "nexia subí", "NEXTA bajá la persiana"])
def test_any_configured_variant_counts_as_present(text):
    check = _verifier().evaluate(text, room_id="living", wake_score=0.6, wake_vad=None, gate_accept=True)
    assert check.code is WakeCheckCode.PRESENT and not check.reject


def test_enforce_rejects_with_the_gate_reason_and_off_does_nothing():
    check = _verifier("enforce").evaluate(SECRET, room_id="hall", wake_score=0.5, wake_vad=0.1, gate_accept=True)
    assert check.reject and check.code is WakeCheckCode.REJECTED and check.reason == "missing_wake:'nexa'"
    assert _verifier("off").evaluate(SECRET, room_id="hall", wake_score=0.5, wake_vad=0.1, gate_accept=True) is None
    with pytest.raises(ValueError):
        WakeVerificationConfig.from_dict({"mode": "maybe"}, WORDS)
    with pytest.raises(ValueError):
        WakeVerificationConfig.from_dict({"mode": "shadow"}, [])


def test_rule_is_the_gate_missing_wake_rule():
    gate = CommandAcceptanceGate(wake_words=WORDS)
    for text in (SECRET, "Nexa prendé la luz", "¿Alexa? apagá"):
        gate_rejects = gate.evaluate(text).reason.startswith("missing_wake")
        assert gate_rejects == (not contains_wake_word(text, WORDS))


def test_counters_status_file_is_private_atomic_and_resumable(tmp_path):
    now = [1_800_000_000.0]
    verifier = _verifier(tmp_path=tmp_path, clock=lambda: now[0])
    verifier.evaluate(SECRET, room_id="cocina", wake_score=0.55, wake_vad=0.2, gate_accept=True)
    verifier.evaluate("Nexa prendé la luz", room_id="cocina", wake_score=0.8, wake_vad=0.7, gate_accept=True)
    verifier.evaluate(SECRET, room_id="living", wake_score=0.45, wake_vad=None, gate_accept=False)
    snapshot = verifier.snapshot()
    assert snapshot["totals"] == {"would_reject": 2, "wake_present": 1, "gate_rejected": 1}
    verifier.write_status()
    path = tmp_path / "wv.json"
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    raw = path.read_text()
    assert "lámpara" not in raw and "prend" not in raw.lower()
    data = json.loads(raw)
    assert data["schema"] == STATUS_SCHEMA and data["mode"] == "shadow"
    assert [e["code"] for e in data["events"]] == ["would_reject", "wake_present", "would_reject"]
    assert data["events"][0] == {"ts": 1_800_000_000.0, "room": "cocina", "code": "would_reject",
                                 "score": 0.55, "vad": 0.2, "gate_accept": True}
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".wake_verification.")]
    resumed = _verifier(tmp_path=tmp_path, clock=lambda: now[0])
    resumed.load()
    resumed.evaluate(SECRET, room_id="cocina", wake_score=0.5, wake_vad=0.1, gate_accept=True)
    assert resumed.snapshot()["totals"]["would_reject"] == 3
    assert len(resumed.snapshot()["events"]) == 4


def test_events_and_days_older_than_keep_days_are_dropped(tmp_path):
    day = 86400.0
    now = [1_800_000_000.0]
    verifier = _verifier(tmp_path=tmp_path, clock=lambda: now[0])
    verifier.evaluate(SECRET, room_id="cocina", wake_score=0.55, wake_vad=0.2, gate_accept=True)
    verifier.write_status()
    # A quiet period: 15 days later a single wake; the old event and day go.
    now[0] += 15 * day
    verifier.evaluate(SECRET, room_id="living", wake_score=0.5, wake_vad=None, gate_accept=True)
    snapshot = verifier.snapshot()
    assert [e["room"] for e in snapshot["events"]] == ["living"]
    assert len(snapshot["days"]) == 1 and snapshot["totals"] == {"would_reject": 1}
    # Loading a stale file keeps nothing older than keep_days either.
    resumed = _verifier(tmp_path=tmp_path, clock=lambda: now[0])
    resumed.load()
    assert resumed.snapshot()["events"] == [] and resumed.snapshot()["days"] == {}
    # The snapshot written on shutdown is pruned too, even without new wakes.
    idle = _verifier(tmp_path=tmp_path, clock=lambda: now[0] - 15 * day)
    idle.load()
    assert len(idle.snapshot()["events"]) == 1
    idle._clock = lambda: now[0]
    idle.write_status()
    assert json.loads((tmp_path / "wv.json").read_text())["events"] == []


def test_settings_default_is_shadow():
    cfg = yaml.safe_load((ROOT / "config" / "settings.yaml").read_text())
    wake = cfg["rooms"]["wake_word"]
    parsed = WakeVerificationConfig.from_dict(wake["verification"], wake["words"])
    assert parsed.mode is WakeVerificationMode.SHADOW
    assert set(parsed.words) == set(WORDS)


# ---------------------------------------------------------------- router

def _router(verifier, text):
    cmd = MagicMock(text=text, user=None, emotion=None, timings={}, stt_confidence=None)
    processor = MagicMock()
    processor.process_command = AsyncMock(return_value=cmd)
    orch = MagicMock()
    orch.process = AsyncMock(return_value=MagicMock(
        intent="domotics", response="ok", success=True, action=None, path=None, timings={},
        was_queued=False, queue_position=None))
    router = RequestRouter(
        command_processor=processor, orchestrator=orch, orchestrator_enabled=True,
        response_handler=response_handler_mock(), audio_manager=MagicMock(), wake_words=("nexa",),
        command_gate=CommandAcceptanceGate(), wake_acoustically_confirmed=True, wake_verifier=verifier)
    return router, orch


def _event(trigger="wake", **kwargs):
    return CommandEvent(audio=np.zeros(16000, dtype=np.float32), room_id="escritorio", wake_score=0.9,
                        wake_vad=0.12, capture_trigger=trigger, capture_complete=True, **kwargs)


@pytest.mark.asyncio
async def test_router_shadow_counts_but_does_not_change_the_result():
    verifier = _verifier()
    router, orch = _router(verifier, "prendé la luz")
    shadow_result = await router.process_command(_event())
    assert verifier.snapshot()["totals"] == {"would_reject": 1}
    assert verifier.snapshot()["events"][0]["vad"] == 0.12
    baseline_router, baseline_orch = _router(None, "prendé la luz")
    baseline = await baseline_router.process_command(_event())
    assert shadow_result["intent"] == baseline["intent"]
    assert orch.process.await_count == baseline_orch.process.await_count


@pytest.mark.asyncio
async def test_router_enforce_rejects_as_missing_wake():
    verifier = _verifier("enforce")
    router, orch = _router(verifier, "prendé la luz")
    result = await router.process_command(_event())
    assert result["intent"] == "gate_rejected:missing_wake:'nexa'" and result["success"] is False
    orch.process.assert_not_awaited()
    present_router, present_orch = _router(_verifier("enforce"), "Nexa prendé la luz")
    present = await present_router.process_command(_event())
    assert not str(present["intent"]).startswith("gate_rejected")


@pytest.mark.asyncio
@pytest.mark.parametrize("event_kwargs", [{"trigger": "follow_up"}, {"trigger": "barge_in"}, {"trigger": None},
                                          {"trigger": "wake", "textual_wake": True}])
async def test_router_only_checks_acoustic_wakes(event_kwargs):
    verifier = _verifier("enforce")
    router, _orch = _router(verifier, "prendé la luz")
    result = await router.process_command(_event(**event_kwargs))
    assert verifier.snapshot()["totals"] == {}
    assert not str(result["intent"]).startswith("gate_rejected:missing_wake")


@pytest.mark.asyncio
async def test_router_raw_audio_is_not_checked():
    verifier = _verifier("enforce")
    router, _orch = _router(verifier, "prendé la luz")
    await router.process_command(np.zeros(16000, dtype=np.float32))
    assert verifier.snapshot()["totals"] == {}


# ---------------------------------------------------------------- wake log

class _Detector:
    peek_pending_text = None
    pop_pending_command_audio = None
    pop_pending_text = None

    def __init__(self):
        self.calls = 0

    def load(self):
        pass

    def reset(self):
        pass

    def detect(self, chunk):
        self.calls += 1
        return ("nexa", 0.51) if self.calls == 2 else None


def test_every_wake_logs_its_vad(caplog):
    echo = MagicMock(is_safe_to_listen=True)
    echo.should_process_audio.return_value = (True, "ok")
    rs = RoomStream("hall", 0, _Detector(), echo)
    loop = MultiRoomAudioLoop({"hall": rs}, MagicMock(is_active=False), wake_preroll_s=0.24)
    loop._wake_vad_predictors["hall"] = lambda _audio: 0.27
    process = loop._make_audio_processor(rs)
    with caplog.at_level(logging.INFO):
        for _ in range(2):
            process(np.full((CHUNK_SIZE, 1), 0.01, dtype=np.float32), CHUNK_SIZE, None, None)
    assert rs.listening and rs.wake_vad == pytest.approx(0.27)
    assert any("Wake word in hall" in r.getMessage() and "vad=0.27" in r.getMessage() for r in caplog.records)
    no_vad = RoomStream("hall", 0, _Detector(), echo)
    loop2 = MultiRoomAudioLoop({"hall": no_vad}, MagicMock(is_active=False))
    process2 = loop2._make_audio_processor(no_vad)
    caplog.clear()
    with caplog.at_level(logging.INFO):
        for _ in range(2):
            process2(np.full((CHUNK_SIZE, 1), 0.01, dtype=np.float32), CHUNK_SIZE, None, None)
    assert any("Wake word in hall" in r.getMessage() and "vad=na" in r.getMessage() for r in caplog.records)


def test_guard_rejection_logs_the_vad(caplog):
    guard = MagicMock()
    guard.on_wake.return_value = MagicMock(accept=False, reason="strict_score", state=MagicMock(value="strict"))
    echo = MagicMock(is_safe_to_listen=True)
    rs = RoomStream("living", 0, _Detector(), echo)
    loop = MultiRoomAudioLoop({"living": rs}, MagicMock(is_active=False), ambient_guard=guard)
    with caplog.at_level(logging.INFO):
        assert loop._should_accept_wakeword("living", 0.01, 1.0, wake_score=0.44, wake_vad=0.05) is False
    assert any("vad=0.05" in r.getMessage() for r in caplog.records)
