"""U3b-4b-2: the router, the registry join and the attributor/verifier on one shared privacy ledger.

Synthetic only: a fake encoder returns chosen 192-dimension unit vectors, the
STT is a stub that returns a chosen transcript, audio is a constant tone. The
real RequestRouter, VoiceRegistry (observe, model admission, ledger put and
review clip), FactLedger and ResponseHandler (synthetic speech) run.
"""

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
import json
import logging
import shutil
import subprocess
import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import numpy as np
import pytest

from src.memory.erasure_coordinator import ErasureCoordinator
from src.memory.fact_ledger import MemoryOrigin, MemorySource, MemoryVoiceConsentActor, VoiceSwitchActor
from src.memory.privacy_services import PrivacyServices
from src.memory.producer_context import PRODUCER_CONTEXT
from src.memory.services import build_memory_services
from src.memory.voice_attribution import VoiceConsentState, VoiceProposalSettings
from src.memory.voice_phrase import VERIFICATION_REPLY
from src.orchestrator.dispatcher import MultiUserOrchestrator
from src.pipeline import request_router as router_module
from src.pipeline.command_event import CommandEvent
from src.pipeline.request_router import RequestRouter
from src.pipeline.voice_pipeline import VoicePipeline
from src.users import voice_attributor as attributor_module
from src.users.voice_attributor import build_voice_attribution
from src.users.voice_ledger import VoiceReviewer
from src.users.voice_registry import build_voice_registry, encoder_model_key
from tests.mocks.memory_graph import Clock, MemoryClient
from tests.mocks.voice_graph import voice_graph

pytestmark = pytest.mark.asyncio

ROOT = Path(__file__).resolve().parents[2]
BASE = "cc78261"  # U3b-4a-3 docs commit: the router before U3b-4b
ACTOR = VoiceReviewer("synthetic-admin")
PERSON, OTHER = "user_" + "a" * 32, "user_" + "b" * 32
AUDIO = np.full(32000, 0.01, dtype=np.float32)  # 2 s


def unit(**weights) -> np.ndarray:
    vector = np.zeros(192, dtype=np.float64)
    for name, weight in weights.items():
        vector[int(name[1:])] = weight
    vector[191] = max(0.0, 1.0 - float(np.dot(vector, vector))) ** 0.5
    return vector.astype(np.float32)


class Encoder:
    """Synthetic encoder: returns ``vector``; optionally holds until ``release`` is set."""

    model_name = "synthetic-ecapa"
    sample_rate = 16000
    _model_type = "synthetic"

    def __init__(self):
        self.vector = unit(e0=1.0)
        self.release: threading.Event | None = None
        self.delay = 0.0
        self.calls = 0

    def get_embedding(self, audio):
        self.calls += 1
        if self.release is not None:
            self.release.wait(5)
        if self.delay:
            time.sleep(self.delay)
        return self.vector.copy()


class Processor:
    """Stands in for CommandProcessor: observes the capture like the real one, returns a chosen transcript."""

    def __init__(self, voices, encoder):
        self.voices, self.encoder, self.text, self.observed = voices, encoder, "", 0

    async def process_command(self, audio, use_parallel=True, pretranscribed_text=None, **options):
        context = PRODUCER_CONTEXT.get()
        if self.voices.observe(self.encoder, audio, context.sources, transcript=self.text, room_id=context.room_id or None):
            self.observed += 1
        return SimpleNamespace(text=self.text, user=None, emotion=None, timings={}, stt_confidence=None)


class Gate:
    """Records what reaches the command gate; accepts only memory requests (so they reach their path)."""

    def __init__(self):
        self.texts = []

    def evaluate(self, text, stt_confidence=None):
        self.texts.append(text)
        accept = "acordate que" in text
        return SimpleNamespace(accept=accept, reason=None if accept else "synthetic_reject", signals={})


@asynccontextmanager
async def stage(tmp_path, monkeypatch, *, enabled=True, wait=2.0, orchestrated=False, router_class=RequestRouter,
                asr_module=router_module, draw=None):
    clock = Clock()
    asr = []

    async def record_asr(logger, *args, **kwargs):
        asr.append((*args, len(kwargs.get("sources", ()))))

    monkeypatch.setattr(asr_module, "alog_asr_outcome", record_asr)
    settings = VoiceProposalSettings(enabled=enabled, embedding_wait_s=wait)
    async with AsyncExitStack() as stack:
        privacy = PrivacyServices(str(tmp_path / "privacy.db"), clock=clock, voice_settings=settings)
        stack.push_async_callback(privacy.aclose)
        await privacy.acall(privacy.initialize)
        owner = ErasureCoordinator(privacy)
        stack.push_async_callback(owner.aclose)
        voices = await build_voice_registry(privacy=privacy, owner=owner, shutdown=stack)
        services = await build_memory_services(
            {"memory": {"privacy_db_path": str(tmp_path / "privacy.db"), "preferences_path": str(tmp_path / "p.json")}},
            shutdown=stack, chroma_client=MemoryClient(tmp_path / "chroma"), privacy=privacy)
        authority = SimpleNamespace(memory_owner_retired=lambda ids: False, memory_owner_active=lambda person: True)
        if draw is not None:
            privacy.ledger.voice_windows._draw = draw
        attributor = build_voice_attribution(privacy, voices, authority, shutdown=stack)
        handler, *_ = voice_graph(monkeypatch)
        stack.push_async_callback(handler.aclose)
        encoder = Encoder()
        voices.register_encoder(encoder)
        processor = Processor(voices, encoder)
        gate = Gate()
        orchestrator = (MultiUserOrchestrator(None, None, None, llm=handler.llm, response_handler=handler,
                                              memory_manager=services.manager) if orchestrated else None)
        options = {"voice_attributor": attributor} if router_class is RequestRouter else {}
        router = router_class(processor, handler, MagicMock(), orchestrator=orchestrator, orchestrator_enabled=orchestrated,
                              memory_manager=services.manager, privacy=privacy, command_gate=gate,
                              chroma_sync=SimpleNamespace(search_command=lambda *a, **kw: None),
                              routine_manager=SimpleNamespace(handle=AsyncMock(return_value={"handled": False})),
                              **options)
        stack.push_async_callback(router.aclose)
        spoken = []
        speak = router._speak_response

        async def spy(text, **kwargs):
            spoken.append(text)
            return await speak(text, **kwargs)

        router._speak_response = spy
        order = []
        if attributor is not None:
            verify = attributor.verify

            def verify_spy(*args):
                order.append(("verify", len(spoken)))
                return verify(*args)

            attributor.verify = verify_spy
        if enabled:
            await privacy.acall(privacy.ledger.set_voice_switch, True, 0, VoiceSwitchActor("synthetic-operator"))
        yield SimpleNamespace(clock=clock, privacy=privacy, ledger=privacy.ledger, voices=voices, services=services,
                              attributor=attributor, router=router, processor=processor, encoder=encoder, gate=gate,
                              spoken=spoken, asr=asr, order=order, key=encoder_model_key(encoder), handler=handler)
        await settle(attributor)


async def settle(attributor):
    async with asyncio.timeout(10):
        while attributor is not None and attributor._tasks:
            await asyncio.gather(*list(attributor._tasks), return_exceptions=True)


async def sql(s, statement, *args):
    def run():
        with s.ledger._transaction() as db:
            return [tuple(row) for row in db.execute(statement, args).fetchall()]
    return await s.privacy.acall(run)


async def profile(s, vector, name, person=None):
    identities = []
    for _ in range(3):
        source = MemorySource(str(uuid4()), str(uuid4()), s.clock.now - 1, s.clock.now + 47 * 3600, MemoryOrigin.VOICE)
        identities.append(await s.voices.acall(s.voices.ledger.put, s.key, vector.copy(), 2.0, (source,)))
    made = await s.voices.acall(s.voices.ledger.confirm, tuple(identities), name, str(uuid4()), ACTOR)
    if person is not None:
        await sql(s, "INSERT INTO voice_profile_people VALUES (?,?,1,?,'synthetic-reviewer')", made["id"], person, s.clock.now)
    return made["id"]


async def household(s):
    """A (PERSON) on e0, B (OTHER) on e1, X unlinked on e2."""
    s.a = await profile(s, unit(e0=1.0), "Voz A", PERSON)
    s.b = await profile(s, unit(e1=1.0), "Voz B", OTHER)
    s.x = await profile(s, unit(e2=1.0), "Voz X")


def me(person=PERSON):
    return MemoryVoiceConsentActor("synthetic-home", person)


async def open_window(s, person=PERSON, room="living"):
    consent = await s.privacy.acall(s.ledger.voice_consent, me(person))
    if consent.state is None:
        await s.privacy.acall(s.ledger.request_voice_consent, me(person), consent.revision)
        consent = await s.privacy.acall(s.ledger.voice_consent, me(person))
    return await s.privacy.acall(s.ledger.open_voice_window, me(person), consent.revision, room, lambda room_id: True)


async def say(s, text, room="living", **event_options):
    s.processor.text = text
    event = CommandEvent(AUDIO.copy(), room, timestamp=s.clock.now, captured_at=s.clock.now, **event_options)
    return event, await s.router.process_command(event)


async def turn_payload(s, event, result):
    published = []
    bus = SimpleNamespace(publish=AsyncMock(side_effect=lambda live: published.append(live.payload)))
    await VoicePipeline._publish_turn_event(SimpleNamespace(_event_bus=bus, request_router=s.router), event, result)
    return published[0]


@pytest.mark.parametrize("orchestrated", [False, True])
async def test_a_phrase_is_routed_before_the_gate_redacted_and_verified(tmp_path, monkeypatch, caplog, orchestrated):
    caplog.set_level(logging.DEBUG)
    async with stage(tmp_path, monkeypatch, orchestrated=orchestrated) as s:
        await household(s)
        window = await open_window(s)
        s.encoder.vector = unit(e0=0.9, e1=0.3)
        spoken_form = "Nexa, " + window.phrase.title().replace(" ", ", ", 1) + "."
        event, result = await say(s, spoken_form)
        assert (result["intent"], result["text"], result["response"]) == ("voice_verification", "", VERIFICATION_REPLY)
        assert s.gate.texts == [] and s.asr == [], "never reaches the gate, its log or the ASR event"
        assert s.spoken == [VERIFICATION_REPLY]
        assert s.order == [("verify", 1)], "the evaluation is scheduled only after the reply was spoken"
        payload = await turn_payload(s, event, result)
        assert payload["stt"] == "" and window.phrase not in json.dumps(payload, ensure_ascii=False).casefold()
        await settle(s.attributor)
        consent = await s.privacy.acall(s.ledger.voice_consent, me())
        assert consent.state is VoiceConsentState.ACTIVE and consent.verified_profile_id == s.a
        assert await sql(s, "SELECT outcome FROM voice_verification_attempts WHERE person_id=? AND outcome!='window_opened'",
                         PERSON) == [("passed",)]
        assert await sql(s, "SELECT COUNT(*) FROM voice_fingerprints") >= [(1,)]
        assert await sql(s, "SELECT COUNT(*) FROM voice_capture_reviews") == [(0,)], "the clip with the phrase is gone"
        assert s.ledger.voice_windows.current(PERSON) is None, "consumed"
        assert s.attributor.counts().verification == {"owner": 1}
        logged = "\n".join(record.getMessage() for record in caplog.records).casefold()
        assert all(word not in logged for word in window.phrase.split()), "no phrase word in any log record"


async def test_every_outcome_speaks_the_same_reply_and_a_failure_counts(tmp_path, monkeypatch):
    async with stage(tmp_path, monkeypatch) as s:
        await household(s)
        results = []
        for vector in (unit(e2=0.9, e0=0.3), unit(e1=0.9, e0=0.3), unit(e0=0.9, e1=0.3)):
            window = await open_window(s)
            s.encoder.vector = vector
            _event, result = await say(s, window.phrase)
            results.append((result["intent"], result["response"], result["text"]))
            await settle(s.attributor)
        assert results == [("voice_verification", VERIFICATION_REPLY, "")] * 3
        assert s.spoken == [VERIFICATION_REPLY] * 3 and s.gate.texts == []
        assert sorted(row[0] for row in await sql(
            s, "SELECT outcome FROM voice_verification_attempts WHERE outcome!='window_opened'")) == [
            "failed", "failed_foreign", "passed"]


async def test_only_multi_room_microphone_captures_can_match(tmp_path, monkeypatch):
    """Any other channel speaking the phrase gets the same reply (review L2) but consumes and evaluates nothing."""
    async with stage(tmp_path, monkeypatch) as s:
        await household(s)
        window = await open_window(s)
        results = [(await say(s, window.phrase, textual_wake=True))[1],   # textual wake: AMBIENT source
                   (await say(s, window.phrase, room=""))[1]]             # single-mic loop
        results.append((await say(s, window.phrase, room="kitchen"))[1])  # another room
        assert [(r["intent"], r["text"], r["response"]) for r in results] == [
            ("voice_verification", "", VERIFICATION_REPLY)] * 3
        s.processor.text = window.phrase
        raw = await s.router.process_command(AUDIO.copy())       # a raw array has no room (nor a live source)
        assert raw["intent"] == "cancelled" and raw["text"] == "" and s.processor.observed == 3
        assert s.gate.texts == [] and s.asr == [] and s.order == []
        assert s.ledger.voice_windows.current(PERSON) is window, "still open"
        assert await sql(s, "SELECT COUNT(*) FROM voice_verification_attempts WHERE outcome!='window_opened'") == [(0,)]
        assert s.spoken == [VERIFICATION_REPLY] * 3
        assert s.attributor.counts().verification == {"near_miss": 3}
        _event, result = await say(s, window.phrase)
        assert result["intent"] == "voice_verification" and s.order == [("verify", 4)]
        await settle(s.attributor)
        _event, again = await say(s, window.phrase)
        assert again["intent"].startswith("gate_rejected"), "a replay after consumption matches nothing"


NEAR = "foca pino ajedrez tiburon"  # test-only phrase: foca/pino were dropped from the list (review W)
LIGHT = "Nexa, prendo el foco ajedrez tiburón"  # foca->foco, pino->prendo


@pytest.mark.parametrize("orchestrated", [False, True])
async def test_a_near_miss_of_an_open_phrase_is_a_failed_phrase_turn(tmp_path, monkeypatch, caplog, orchestrated):
    """Review L2: before the gate, the same reply, nothing logged, no light command, the window untouched."""
    caplog.set_level(logging.DEBUG)
    async with stage(tmp_path, monkeypatch, orchestrated=orchestrated, draw=lambda: NEAR) as s:
        await household(s)
        window = await open_window(s)
        assert window.phrase == NEAR
        from src.nlu.command_grammar import parse_partial_command
        assert parse_partial_command(LIGHT).intent == "turn_on", "on the normal route this reads as a light command"
        parsed = []
        monkeypatch.setattr(router_module, "parse_home_command", lambda *a, **kw: parsed.append(a) or None)
        lanes = [await say(s, LIGHT), await say(s, LIGHT, room="kitchen"), await say(s, LIGHT, textual_wake=True),
                 await say(s, "Nexa, foco pinos ajedrez"), await say(s, "foca pino ajedres tiburon por favor")]
        for event, result in lanes:
            assert (result["intent"], result["text"], result["response"]) == ("voice_verification", "", VERIFICATION_REPLY)
            payload = await turn_payload(s, event, result)
            assert payload["stt"] == ""
        assert s.gate.texts == [] and s.asr == [] and s.order == []
        assert s.spoken == [VERIFICATION_REPLY] * len(lanes)
        assert parsed == [], "no home (light) command was parsed, let alone produced"
        assert s.ledger.voice_windows.current(PERSON) is window, "a near miss consumes nothing"
        assert await sql(s, "SELECT COUNT(*) FROM voice_verification_attempts WHERE outcome!='window_opened'") == [(0,)]
        logged = "\n".join(record.getMessage() for record in caplog.records).casefold()
        assert not any(word in logged for word in ("foco", "prendo", "ajedrez", "tiburon", "tiburón", "pinos", "ajedres"))
        ordinary = ["prendé la luz del living", "Nexa, el foco del ajedrez"]  # two of four words is not near
        for text in ordinary:
            _event, result = await say(s, text)
            assert result["intent"] == "gate_rejected:synthetic_reject", text
        assert s.gate.texts == ordinary and s.ledger.voice_windows.current(PERSON) is window
        s.clock.now += 60.0
        _event, result = await say(s, LIGHT)
        assert result["intent"].startswith("gate_rejected") and s.gate.texts == [*ordinary, LIGHT], "expired: normal route"


async def test_with_no_open_window_ordinary_commands_are_untouched(tmp_path, monkeypatch):
    """Review L2: the guard never fires with no window open, even for a phrase-like or command utterance."""
    async with stage(tmp_path, monkeypatch, draw=lambda: NEAR) as s:
        await household(s)
        screened = []
        screen = s.attributor.screen

        def spy(*args):
            screened.append(screen(*args))
            return screened[-1]

        s.attributor.screen = spy
        texts = [LIGHT, NEAR, "prendé la luz del living", "Nexa, acordate que me gusta el té de menta"]
        results = [(await say(s, text))[1] for text in texts]
        assert screened == [None] * 4
        assert [r["intent"] for r in results[:3]] == ["gate_rejected:synthetic_reject"] * 3
        assert results[3]["intent"] == "memory_proposal"
        assert s.gate.texts == texts and len(s.asr) == 4 and VERIFICATION_REPLY not in s.spoken
        assert s.attributor.counts().verification == {}


async def test_screen_treats_a_multi_source_capture_as_a_near_miss_and_consumes_nothing(tmp_path, monkeypatch):
    async with stage(tmp_path, monkeypatch, draw=lambda: NEAR) as s:
        await household(s)
        window = await open_window(s)
        voice = MemorySource(str(uuid4()), str(uuid4()), 1.0, 2.0, MemoryOrigin.VOICE)
        screen = s.attributor.screen
        assert screen(NEAR, "living", False, (voice, voice), ("nexa",)) == (None, None)
        assert screen(LIGHT, "living", False, (voice, voice), ("nexa",)) == (None, None)
        assert screen("prendé la luz", "living", False, (voice,), ("nexa",)) is None
        assert screen("", "living", False, (voice,), ("nexa",)) is None
        assert s.ledger.voice_windows.current(PERSON) is window
        assert screen(NEAR, "living", False, (voice,), ("nexa",)) == (window, voice)
        assert s.voices._redacted == {voice.capture_id: None}, "the matched capture keeps no review clip"
        assert screen(NEAR, "living", False, (voice,), ("nexa",)) is None, "consumed: no window is open"


async def test_several_windows_in_one_room_are_matched_by_phrase(tmp_path, monkeypatch):
    async with stage(tmp_path, monkeypatch) as s:
        await household(s)
        mine = await open_window(s, PERSON)
        theirs = await open_window(s, OTHER)
        assert mine.room_id == theirs.room_id == "living" and mine.phrase != theirs.phrase
        s.encoder.vector = unit(e1=0.9, e0=0.3)
        _event, result = await say(s, theirs.phrase)
        assert result["intent"] == "voice_verification"
        await settle(s.attributor)
        assert s.ledger.voice_windows.current(PERSON) is mine and s.ledger.voice_windows.current(OTHER) is None
        assert (await s.privacy.acall(s.ledger.voice_consent, me(OTHER))).state is VoiceConsentState.ACTIVE
        assert (await s.privacy.acall(s.ledger.voice_consent, me())).state is VoiceConsentState.PENDING_VERIFICATION


async def activate(s, person=PERSON, vector=None):
    window = await open_window(s, person)
    s.encoder.vector = vector if vector is not None else unit(e0=0.95, e1=0.2)
    await say(s, window.phrase)
    await settle(s.attributor)
    assert (await s.privacy.acall(s.ledger.voice_consent, me(person))).state is VoiceConsentState.ACTIVE
    s.spoken.clear()
    s.gate.texts.clear()


async def test_the_proposal_reply_is_identical_for_every_gate_row_and_attribution_follows_it(tmp_path, monkeypatch):
    async with stage(tmp_path, monkeypatch) as s:
        await household(s)
        await activate(s)
        order = []
        schedule = s.attributor.schedule

        def spy(*args):
            order.append(("schedule", len(s.spoken)))
            return schedule(*args)

        s.attributor.schedule = spy
        ledger_steps = []
        attribute = s.ledger.attribute_voice_proposal

        def step(*args, **kwargs):
            ledger_steps.append(args[0])
            return attribute(*args, **kwargs)

        monkeypatch.setattr(s.ledger, "attribute_voice_proposal", step)
        rows = [(unit(e0=0.9, e1=0.3), "owner"), (unit(e0=0.7), "below_score"), (unit(e2=0.95, e0=0.2), "top_not_grantable"),
                (None, "no_fingerprint")]
        replies, facts = [], []
        for vector, _reason in rows:
            if vector is None:
                s.encoder.vector = unit(e0=0.9)
                s.voices._encoders.clear()  # observe drops the capture: no fingerprint
            else:
                s.encoder.vector = vector
            _event, result = await say(s, "Nexa, acordate que me gusta el té de menta")
            replies.append((result["intent"], result["response"], result["success"], len(s.spoken)))
            facts.append(result["memory_proposal"]["id"])
            await settle(s.attributor)
            s.spoken.clear()
        assert len({reply for reply in replies}) == 1 and replies[0][3] == 1
        assert order == [("schedule", 1)] * 4, "scheduled only after the reply was spoken"
        owners = [await sql(s, "SELECT person_id FROM fact_owners WHERE fact_id=?", fact) for fact in facts]
        assert owners == [[(PERSON,)], [], [], []]
        assert ledger_steps == facts[:3], "no ledger step without a fingerprint"
        assert s.attributor.counts().attribution == {reason: 1 for _vector, reason in rows}


async def test_with_the_flag_off_there_is_no_stage_no_join_and_no_spy_call(tmp_path, monkeypatch):
    calls = []
    for name in ("schedule", "verify", "match_phrase", "screen"):
        monkeypatch.setattr(attributor_module.VoiceProposalAttributor, name,
                            lambda self, *args, _name=name: calls.append(_name))
    async with stage(tmp_path, monkeypatch, enabled=False) as s:
        assert s.attributor is None and s.voices._completions is None and s.router._voice_attributor is None
        await household(s)
        _event, result = await say(s, "gato perro barco nube")
        assert result["intent"].startswith("gate_rejected") and s.gate.texts == ["gato perro barco nube"]
        _event, result = await say(s, "Nexa, acordate que me gusta el té de menta")
        assert result["intent"] == "memory_proposal"
        await asyncio.sleep(0.05)
        assert calls == [] and s.voices._completions is None
        assert await sql(s, "SELECT COUNT(*) FROM fact_owners") == [(0,)]


def _base_router_class():
    """RequestRouter as of the base commit, loaded from git (the pre-4b behaviour)."""
    git = shutil.which("git")
    if git is None:
        pytest.skip("git unavailable")
    shown = subprocess.run([git, "show", f"{BASE}:src/pipeline/request_router.py"], cwd=ROOT, capture_output=True, text=True)
    if shown.returncode != 0:
        pytest.skip("base commit unavailable")
    module = types.ModuleType("request_router_base")
    module.__file__ = str(ROOT / "src/pipeline/request_router.py")
    sys.modules["request_router_base"] = module
    try:
        exec(compile(shown.stdout, module.__file__, "exec"), module.__dict__)
    finally:
        sys.modules.pop("request_router_base", None)
    return module.RequestRouter, module


PROJECTED = ("intent", "text", "response", "success", "handled", "task_completed", "outcome", "voice_delivery", "path")


@pytest.mark.parametrize("orchestrated", [False, True])
async def test_flag_off_behaviour_is_identical_to_the_base_router(tmp_path, monkeypatch, orchestrated):
    base, base_module = _base_router_class()
    runs = {}
    for name, router_class in (("base", base), ("head", RequestRouter)):
        module = base_module if router_class is base else router_module
        async with stage(tmp_path / name, monkeypatch, enabled=False, orchestrated=orchestrated, router_class=router_class,
                         asr_module=module) as s:
            await household(s)
            seen = []
            for text in ("gato perro barco nube", "Nexa, acordate que me gusta el té de menta", "", "nexa", LIGHT, NEAR):
                _event, result = await say(s, text)
                seen.append({key: result.get(key) for key in PROJECTED})
            runs[name] = (seen, list(s.spoken), list(s.gate.texts), s.processor.observed, list(s.asr))
    assert runs["head"][4], "ASR events were recorded and compared"
    assert runs["head"] == runs["base"]


async def test_the_registry_touches_windows_on_the_loop_and_the_ledger_off_it(tmp_path, monkeypatch):
    async with stage(tmp_path, monkeypatch) as s:
        await household(s)
        loop_thread = threading.get_ident()
        threads = {}
        windows = s.ledger.voice_windows
        match = windows.match

        def match_spy(*args):
            threads.setdefault("match", set()).add(threading.get_ident())
            return match(*args)

        monkeypatch.setattr(windows, "match", match_spy)
        for name in ("verify_voice_consent", "attribute_voice_proposal"):
            original = getattr(s.ledger, name)

            def spy(*args, _original=original, _name=name, **kwargs):
                threads.setdefault(_name, set()).add(threading.get_ident())
                return _original(*args, **kwargs)

            monkeypatch.setattr(s.ledger, name, spy)
        gaps, running = [], True

        async def heartbeat():
            last = time.perf_counter()
            while running:
                await asyncio.sleep(0.01)
                now = time.perf_counter()
                gaps.append(now - last)
                last = now

        beat = asyncio.create_task(heartbeat())
        s.encoder.delay = 0.3  # the model takes 300 ms in its own thread
        await activate(s)
        _event, _result = await say(s, "Nexa, acordate que me gusta el té de menta")
        await settle(s.attributor)
        running = False
        await beat
        assert threads["match"] == {loop_thread}
        assert loop_thread not in threads["verify_voice_consent"] | threads["attribute_voice_proposal"]
        assert max(gaps) < 0.2, "nothing blocked the event loop"


async def test_bounded_tasks_and_shutdown_write_nothing(tmp_path, monkeypatch):
    async with stage(tmp_path, monkeypatch) as s:
        await household(s)
        await activate(s)
        s.encoder.release = threading.Event()
        s.encoder.vector = unit(e0=0.9, e1=0.3)
        try:
            facts = []
            for _ in range(6):
                _event, result = await say(s, "Nexa, acordate que me gusta el té de menta")
                facts.append(result["memory_proposal"]["id"])
            assert len(s.attributor._tasks) == 4 and s.attributor.counts().overflow == 2
            started = time.perf_counter()
            await s.attributor.aclose()
            assert time.perf_counter() - started < 0.5, "shutdown cancels; it does not wait for the embeddings"
            assert not s.attributor._tasks and s.attributor.counts().attribution == {}
            assert s.attributor.schedule(facts[0], MemorySource(str(uuid4()), str(uuid4()), s.clock.now, s.clock.now + 60,
                                                                MemoryOrigin.VOICE)) is False
        finally:
            s.encoder.release.set()
        async with asyncio.timeout(5):
            while s.voices.resource_status()["pending"]:
                await asyncio.sleep(0.01)
        assert await sql(s, "SELECT COUNT(*) FROM fact_owners") == [(0,)], "a cancelled attribution writes nothing"


async def test_a_late_embedding_is_unavailable_and_its_clip_is_deleted_when_it_lands(tmp_path, monkeypatch):
    async with stage(tmp_path, monkeypatch, wait=0.05) as s:
        await household(s)
        window = await open_window(s)
        s.encoder.release = threading.Event()
        s.encoder.vector = unit(e0=0.9, e1=0.3)
        discarded = []
        original = s.ledger.discard_voice_review

        def spy(*args):
            discarded.append(original(*args))
            return discarded[-1]

        monkeypatch.setattr(s.ledger, "discard_voice_review", spy)
        _event, result = await say(s, window.phrase)
        assert result["response"] == VERIFICATION_REPLY
        await asyncio.sleep(0.2)
        assert await sql(s, "SELECT outcome FROM voice_verification_attempts WHERE outcome!='window_opened'") == [
            ("unavailable",)]
        assert s.attributor.counts().verification == {"late_embedding": 1}
        s.encoder.release.set()
        async with asyncio.timeout(5):
            while not discarded:
                await asyncio.sleep(0.01)
        await settle(s.attributor)
        assert discarded == [False], "the capture was redacted at the match: no clip was ever built (review W20)"
        assert await sql(s, "SELECT COUNT(*) FROM voice_capture_reviews") == [(0,)]
        assert (await s.privacy.acall(s.ledger.voice_consent, me())).state is VoiceConsentState.PENDING_VERIFICATION


async def test_an_ordinary_capture_keeps_its_review_clip(tmp_path, monkeypatch):
    """Control for the deletion above: only a matched phrase loses its clip."""
    async with stage(tmp_path, monkeypatch) as s:
        await household(s)
        await say(s, "gato perro barco nube")
        async with asyncio.timeout(5):
            while s.voices.resource_status()["pending"]:
                await asyncio.sleep(0.01)
        assert await sql(s, "SELECT COUNT(*) FROM voice_capture_reviews") == [(1,)]


def test_only_a_room_microphone_voice_source_is_a_verification_channel():
    """channel_source on its own: each condition refuses independently (room, textual wake, origin, one source)."""
    from src.users.voice_attributor import VoiceProposalAttributor
    voice = MemorySource(str(uuid4()), str(uuid4()), 1.0, 2.0, MemoryOrigin.VOICE)
    ambient = MemorySource(str(uuid4()), str(uuid4()), 1.0, 2.0, MemoryOrigin.AMBIENT)
    channel = VoiceProposalAttributor.channel_source
    assert channel("living", False, (voice,)) is voice
    for room, textual, sources in (("", False, (voice,)), (None, False, (voice,)), ("living", True, (voice,)),
                                   ("living", False, (ambient,)), ("living", False, (voice, voice)), ("living", False, ()),
                                   ("living", None, (voice,)), ("living", False, [voice])):
        assert channel(room, textual, sources) is None, (room, textual, sources)


async def test_a_late_clip_is_discarded_even_after_shutdown(tmp_path, monkeypatch):
    """Review W20: shutdown does not stop the deletion of a phrase clip whose fingerprint lands late."""
    async with stage(tmp_path, monkeypatch) as s:
        await household(s)
        await say(s, "gato perro barco nube")  # an ordinary capture: its fingerprint and clip are stored
        async with asyncio.timeout(5):
            while s.voices.resource_status()["pending"]:
                await asyncio.sleep(0.01)
        [(identity,)] = await sql(s, "SELECT fingerprint_id FROM voice_capture_reviews")
        await s.attributor.aclose()
        landed = asyncio.get_running_loop().create_future()
        landed.set_result(identity)
        s.attributor._discard_late_clip(landed)
        await settle(s.attributor)
        assert await sql(s, "SELECT COUNT(*) FROM voice_capture_reviews") == [(0,)]


async def test_a_phrase_capture_whose_embedding_lands_after_shutdown_stores_no_clip(tmp_path, monkeypatch):
    """Review W20: the capture is redacted at the match, so a late embedding never builds the clip."""
    async with stage(tmp_path, monkeypatch, wait=0.05) as s:
        await household(s)
        window = await open_window(s)
        s.encoder.release = threading.Event()
        _event, result = await say(s, window.phrase)
        assert result["response"] == VERIFICATION_REPLY
        await asyncio.sleep(0.2)
        discarded = []
        original = s.ledger.discard_voice_review

        def spy(*args):
            discarded.append(original(*args))
            return discarded[-1]

        monkeypatch.setattr(s.ledger, "discard_voice_review", spy)
        before = await sql(s, "SELECT COUNT(*) FROM voice_fingerprints")
        await s.attributor.aclose()
        s.encoder.release.set()
        async with asyncio.timeout(5):
            while not discarded:
                await asyncio.sleep(0.01)
        await settle(s.attributor)
        assert await sql(s, "SELECT COUNT(*) FROM voice_fingerprints") == [(before[0][0] + 1,)], "the fingerprint landed"
        assert discarded == [False], "the clip was never built, so there was nothing to delete"
        assert await sql(s, "SELECT COUNT(*) FROM voice_capture_reviews") == [(0,)]


async def test_the_whisper_wake_engine_keeps_the_stage_off(tmp_path, monkeypatch, caplog):
    """Review E: whisper_wake logs transcripts, so with it there is no voice stage at all."""
    caplog.set_level(logging.DEBUG)
    async with stage(tmp_path, monkeypatch) as s:
        stack = AsyncExitStack()
        authority = SimpleNamespace(memory_owner_retired=lambda ids: False, memory_owner_active=lambda person: True)
        for engine in ("whisper", None, "", "Openwakeword"):
            assert build_voice_attribution(s.privacy, s.voices, authority, shutdown=stack, wake_engine=engine) is None
        assert build_voice_attribution(s.privacy, s.voices, authority, shutdown=stack) is not None
        await stack.aclose()
        assert "the wake engine is not phrase-safe" in caplog.text


def test_the_configured_wake_engine_is_openwakeword_and_main_passes_it():
    """Review E: pins the engine choice and its wiring into the stage."""
    import ast as python_ast
    import yaml
    config = yaml.safe_load((ROOT / "config/settings.yaml").read_text(encoding="utf-8"))
    assert config["rooms"]["wake_word"]["engine"] == "openwakeword"
    tree = python_ast.parse((ROOT / "src/main.py").read_text(encoding="utf-8"))
    [call] = [node for node in python_ast.walk(tree) if isinstance(node, python_ast.Call)
              and getattr(node.func, "id", None) == "build_voice_attribution"]
    keywords = {keyword.arg: python_ast.unparse(keyword.value) for keyword in call.keywords}
    assert keywords["wake_engine"] == "config.get('rooms', {}).get('wake_word', {}).get('engine', 'openwakeword')"
    assert keywords["home_names"] == "home_capabilities.cached_names"


async def test_the_stage_keeps_phrase_words_away_from_home_names(tmp_path, monkeypatch):
    """Review W: build_voice_attribution installs the home names on the windows."""
    async with stage(tmp_path, monkeypatch) as s:
        stack = AsyncExitStack()
        draws = iter(["tiburon ajedrez trineo vitral", "mosaico ajedrez trineo vitral"])
        s.ledger.voice_windows._draw = lambda: next(draws)
        authority = SimpleNamespace(memory_owner_retired=lambda ids: False, memory_owner_active=lambda person: True)
        build_voice_attribution(s.privacy, s.voices, authority, shutdown=stack, home_names=lambda: ("Tiburón",))
        await household(s)
        window = await open_window(s)
        assert window.phrase == "mosaico ajedrez trineo vitral"
        await stack.aclose()


def test_cached_names_are_the_last_snapshots_device_and_area_names():
    from src.home_assistant.capability_service import HomeCapabilityService
    service = HomeCapabilityService(MagicMock(), MagicMock())
    assert service.cached_names() == ()
    service._last_snapshot = SimpleNamespace(
        states=SimpleNamespace(states=({"entity_id": "light.tiburon_grande", "attributes": {"friendly_name": "Lámpara Delfín"}},
                                       {"entity_id": "switch.bomba", "attributes": {}}, {"attributes": None})),
        areas=SimpleNamespace(area_names={"a1": "Quincho", "a2": None}))
    assert service.cached_names() == ("Lámpara Delfín", "tiburon grande", "bomba", "Quincho")


async def test_without_redaction_the_evaluation_still_deletes_the_phrase_clip(tmp_path, monkeypatch):
    """Two independent layers: with the capture redaction disabled, the verification transaction deletes the clip."""
    async with stage(tmp_path, monkeypatch) as s:
        await household(s)
        monkeypatch.setattr(s.voices, "redact_capture", lambda capture_id: None)
        window = await open_window(s)
        s.encoder.vector = unit(e0=0.9, e1=0.3)
        _event, result = await say(s, window.phrase)
        assert result["intent"] == "voice_verification"
        await settle(s.attributor)
        assert await sql(s, "SELECT COUNT(*) FROM voice_fingerprints") >= [(1,)]
        assert await sql(s, "SELECT COUNT(*) FROM voice_capture_reviews") == [(0,)]
        assert (await s.privacy.acall(s.ledger.voice_consent, me())).state is VoiceConsentState.ACTIVE
