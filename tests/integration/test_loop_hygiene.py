"""U3b-0d2: blocking work stays off the event loop, and the lag probe sees it when it does not."""

import asyncio
import json
import logging
from pathlib import Path
import threading
import time
from unittest.mock import MagicMock

import chromadb
import numpy as np
import pytest
import yaml

from src.monitoring.health_aggregator import HealthAggregator
from src.monitoring.loop_lag import LoopLagConfig, LoopLagProbe
from src.pipeline.voice_pipeline import VoicePipeline
from src.vectordb.chroma_sync import ChromaSync, CommandSyncError

REPO = Path(__file__).resolve().parents[2]
BUDGET_MS = 200


def _probe() -> LoopLagProbe:
    return LoopLagProbe(LoopLagConfig(interval_s=0.02, report_interval_s=3600, warn_ms=BUDGET_MS))


async def test_probe_measures_a_block_on_the_loop(caplog):
    probe = _probe()
    caplog.set_level(logging.INFO, logger="src.monitoring.loop_lag")
    probe.start()
    await asyncio.sleep(0.1)
    time.sleep(0.35)  # deliberately on the loop
    await asyncio.sleep(0.1)
    await probe.stop()
    snapshot = probe.snapshot()
    assert snapshot["max_ms"] >= 300 and snapshot["over_warn"] >= 1 and not snapshot["running"]
    assert "[LoopLag] event loop blocked for about" in caplog.text
    assert "[LoopLag] run max block" in caplog.text


async def test_probe_reports_each_window_and_stays_quiet_when_idle(caplog):
    probe = LoopLagProbe(LoopLagConfig(interval_s=0.02, report_interval_s=0.1, warn_ms=BUDGET_MS))
    caplog.set_level(logging.INFO, logger="src.monitoring.loop_lag")
    probe.start()
    probe.start()  # idempotent
    await asyncio.sleep(0.5)
    await probe.stop()
    snapshot = probe.snapshot()
    assert snapshot["samples"] >= 10 and snapshot["max_ms"] < BUDGET_MS
    assert snapshot["last_window_max_ms"] is not None
    assert "[LoopLag] max block" in caplog.text


def test_sustained_lag_warns_once_per_window_and_counts_the_rest(caplog):
    """S7: ten blocks a second must not flood the console log."""
    clock = [0.0]
    probe = LoopLagProbe(LoopLagConfig(interval_s=0.1, report_interval_s=60, warn_ms=BUDGET_MS),
                         clock=lambda: clock[0])
    caplog.set_level(logging.INFO, logger="src.monitoring.loop_lag")
    probe._window_started = 0.0
    for _ in range(50):
        probe.record(0.3)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and probe.snapshot()["window_over_warn"] == 50
    clock[0] = 60.0
    probe._close_window(60.0)
    assert "(50 >= 200 ms;" in caplog.text
    probe.record(0.3)
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 2
    assert probe.snapshot()["over_warn"] == 51 and probe.snapshot()["window_over_warn"] == 1


def test_probe_config_defaults_match_settings_and_bad_values_are_refused():
    settings = yaml.safe_load((REPO / "config/settings.yaml").read_text())
    assert LoopLagConfig.from_config(settings["monitoring"]["loop_lag"]) == LoopLagConfig()
    assert LoopLagConfig.from_config(None) == LoopLagConfig()
    for bad in ({"interval_s": 0}, {"warn_ms": 0}, {"interval_s": 1, "report_interval_s": 0.5}):
        with pytest.raises(ValueError, match="loop_lag_config_invalid"):
            LoopLagConfig.from_config(bad)


async def test_disabled_probe_never_starts():
    probe = LoopLagProbe(LoopLagConfig(enabled=False))
    probe.start()
    assert not probe.running
    await probe.stop()


def test_metrics_publish_the_loop_lag_snapshot():
    probe = _probe()
    probe.record(0.25)
    metrics = HealthAggregator(loop_lag_provider=probe.snapshot).get_metrics()
    assert metrics["loop_lag"]["max_ms"] == 250.0 and metrics["loop_lag"]["over_warn"] == 1
    assert "loop_lag" not in HealthAggregator().get_metrics()


class _SlowEncoder:
    def __init__(self, threads, delay=0.05):
        self.threads = threads
        self.delay = delay

    def encode(self, text):
        self.threads.add(threading.current_thread() is threading.main_thread())
        time.sleep(self.delay)  # BGE-M3 CPU encode
        return np.array([0.1, 0.2, 0.3])


class _Home:
    """Async HA reads, as HomeAssistantClient exposes them."""

    def __init__(self, count=4):
        self.count = count

    async def get_domotics_entities(self):
        return [{"entity_id": f"light.room_{n}", "attributes": {"friendly_name": f"Room {n}"}}
                for n in range(self.count)]

    async def get_services_by_domain(self):
        return {"light": ["turn_on", "turn_off"]}


class _SlowLLM:
    """The reasoners' async ``complete``: each description takes ``delay`` seconds.

    ``fail_at`` raises on that call (cloud failure); ``started`` is set at the
    first call; ``cancelled`` counts calls cancelled by a close.
    """

    def __init__(self, verb="prende", delay=0.3, fail_at=None, phrases=2):
        self.verb, self.delay, self.fail_at, self.phrases = verb, delay, fail_at, phrases
        self.calls = 0
        self.cancelled = 0
        self.started = asyncio.Event()

    async def complete(self, prompt, max_tokens=512, temperature=0.7):
        self.calls += 1
        self.started.set()
        try:
            await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        if self.fail_at == self.calls:
            raise ConnectionError("gateway unreachable")
        entity = prompt.split("Entidad: ", 1)[1].split("\n", 1)[0]
        return json.dumps([{"entity_id": entity, "domain": "light", "service": "turn_on",
                            "description": "Encender",
                            "phrases": [f"{self.verb} {entity} {n}" for n in range(self.phrases)]}])


def _documents(owner):
    return sorted(owner.commands.get()["documents"])


def _collections(owner):
    return sorted(c.name for c in owner._client.list_collections())


@pytest.fixture
def chroma_stub(monkeypatch):
    import src.vectordb.chroma_sync as module

    threads = set()
    encoders = []

    def build(*args, **kwargs):
        encoders.append(_SlowEncoder(threads))
        return encoders[-1]
    original = chromadb.PersistentClient
    monkeypatch.setattr(chromadb, "PersistentClient", lambda **kwargs: original(**kwargs))
    monkeypatch.setattr(module, "SentenceTransformer", build)
    return threads, encoders


@pytest.fixture
async def index(tmp_path, chroma_stub):
    threads, _ = chroma_stub
    owner = ChromaSync(str(tmp_path / "index"), "synthetic", "cpu")
    await asyncio.to_thread(owner.initialize)
    try:
        yield owner, threads
    finally:
        await owner.aclose()


@pytest.fixture
async def seeded(index):
    """An index whose active commands are the 8 'viejo' phrases."""
    owner, threads = index
    assert await owner.sync_commands(_Home(), _SlowLLM(verb="viejo", delay=0)) == 8
    return owner, threads


async def test_sync_commands_keeps_the_loop_responsive(index):
    owner, encoder_threads = index
    llm = _SlowLLM()
    probe = _probe()
    probe.start()
    started = time.monotonic()
    count = await owner.sync_commands(_Home(), llm)
    elapsed = time.monotonic() - started
    await probe.stop()
    assert count == 8 and await asyncio.to_thread(owner.commands.count) == 8
    assert elapsed >= 1.2  # 4 x 300 ms of LLM plus 8 encodes really ran
    assert llm.calls == 4 and encoder_threads == {False}
    assert probe.snapshot()["max_ms"] < BUDGET_MS, probe.snapshot()


async def test_sync_commands_without_a_reasoner_fails_before_touching_the_index(index):
    owner, _ = index
    for reasoner in (None, object()):
        with pytest.raises(CommandSyncError, match="chroma_sync_reasoner_unavailable") as caught:
            await owner.sync_commands(_Home(), reasoner)
        assert caught.value.index_kept


async def test_searches_see_the_previous_index_throughout_a_sync(seeded):
    """S1: neither the LLM phase nor the encode phase empties the active index."""
    owner, _ = seeded
    old = await asyncio.to_thread(_documents, owner)
    owner._embedder.delay = 0.1
    llm = _SlowLLM(verb="nuevo", delay=0.2)
    running = asyncio.create_task(owner.sync_commands(_Home(), llm))
    seen = []
    while not running.done():
        seen.append(await asyncio.to_thread(_documents, owner))
        match = await owner.asearch_command("viejo light.room_0 0", threshold=0.1)
        if seen[-1] == old:
            assert match is not None and match["entity_id"].startswith("light.room_")
        await asyncio.sleep(0.05)
    assert await running == 8
    assert len(seen) > 10 and all(docs in (old, sorted(d.replace("viejo", "nuevo") for d in old))
                                  for docs in seen)
    assert seen[0] == old and old in seen[: len(seen) - 1]
    assert await asyncio.to_thread(_documents, owner) == sorted(d.replace("viejo", "nuevo") for d in old)


async def test_a_failure_mid_sync_keeps_the_previous_index_persisted(seeded, tmp_path, chroma_stub):
    """S1: a cloud failure at entity 3 leaves the old 8 phrases, also after a restart."""
    owner, _ = seeded
    old = await asyncio.to_thread(_documents, owner)
    before = await asyncio.to_thread(_collections, owner)
    with pytest.raises(CommandSyncError, match="chroma_sync_failed") as caught:
        await owner.sync_commands(_Home(), _SlowLLM(verb="nuevo", delay=0.01, fail_at=3))
    assert caught.value.index_kept and isinstance(caught.value.__cause__, ConnectionError)
    assert await asyncio.to_thread(_documents, owner) == old
    assert await asyncio.to_thread(_collections, owner) == before
    await owner.aclose()
    reopened = ChromaSync(str(tmp_path / "index"), "synthetic", "cpu")
    await asyncio.to_thread(reopened.initialize)
    try:
        assert await asyncio.to_thread(_documents, reopened) == old
    finally:
        await reopened.aclose()


async def test_a_failure_while_writing_the_staging_collection_drops_it(seeded, monkeypatch):
    owner, _ = seeded
    old = await asyncio.to_thread(_documents, owner)
    before = await asyncio.to_thread(_collections, owner)
    original = owner._client.create_collection

    def failing_staging(**kwargs):
        collection = original(**kwargs)
        monkeypatch.setattr(collection, "add", MagicMock(side_effect=OSError("disk full")))
        return collection
    monkeypatch.setattr(owner._client, "create_collection", failing_staging)
    with pytest.raises(CommandSyncError, match="chroma_sync_failed") as caught:
        await owner.sync_commands(_Home(), _SlowLLM(verb="nuevo", delay=0))
    assert caught.value.index_kept
    assert await asyncio.to_thread(_documents, owner) == old
    assert await asyncio.to_thread(_collections, owner) == before


async def test_an_uncertain_switch_keeps_the_staging_collection(seeded, monkeypatch):
    """If the pointer write itself fails, neither possible pointer may name a dropped collection."""
    owner, _ = seeded
    old = await asyncio.to_thread(_documents, owner)
    before = await asyncio.to_thread(_collections, owner)
    monkeypatch.setattr(owner._pointer_collection, "modify", MagicMock(side_effect=OSError("io")))
    with pytest.raises(CommandSyncError, match="chroma_sync_switch_uncertain") as caught:
        await owner.sync_commands(_Home(), _SlowLLM(verb="nuevo", delay=0))
    assert caught.value.index_kept is False
    assert await asyncio.to_thread(_documents, owner) == old
    after = await asyncio.to_thread(_collections, owner)
    assert set(before) < set(after) and len(after) == len(before) + 1
    # R1: no further sync in this process could drop what the pointer may name.
    assert owner.index_health()["sync_latched"] == "switch_uncertain"
    monkeypatch.undo()
    with pytest.raises(CommandSyncError, match="chroma_sync_restart_required") as again:
        await owner.sync_commands(_Home(), _SlowLLM(verb="otro", delay=0))
    assert again.value.index_kept
    assert await asyncio.to_thread(_collections, owner) == after
    health = HealthAggregator(command_index_provider=owner.index_health).get_system_health()
    assert [s.status.value for s in health.subsystems if s.name == "command_index"] == ["degraded"]


async def test_a_pointer_write_that_committed_but_raised_is_honoured(seeded, monkeypatch, tmp_path):
    """R1: the persisted pointer may name the kept staging collection; cleanup keeps it
    and the next start opens it."""
    owner, _ = seeded
    real = owner._pointer_collection.modify

    def committed_then_failed(**kwargs):
        real(**kwargs)
        raise OSError("ack lost")
    monkeypatch.setattr(owner._pointer_collection, "modify", committed_then_failed)
    with pytest.raises(CommandSyncError, match="chroma_sync_switch_uncertain"):
        await owner.sync_commands(_Home(), _SlowLLM(verb="nuevo", delay=0))
    persisted = owner._client.get_collection("home_assistant_commands__pointer").metadata["active"]
    assert persisted != owner._commands_collection.name
    # Cleanup with only the in-memory active name still keeps the persisted target.
    await asyncio.to_thread(owner._drop_stale_collections, {owner._commands_collection.name})
    assert persisted in await asyncio.to_thread(_collections, owner)
    await owner.aclose()
    reopened = ChromaSync(str(tmp_path / "index"), "synthetic", "cpu")
    await asyncio.to_thread(reopened.initialize)
    try:
        assert reopened._commands_collection.name == persisted
        assert reopened.index_health()["degraded"] is None
        assert (await asyncio.to_thread(_documents, reopened))[0].startswith("nuevo")
    finally:
        await reopened.aclose()


async def test_a_cancelled_caller_keeps_the_sync_single_flight(seeded):
    """R2: cancelling the voice request does not release the sync while its worker runs."""
    owner, _ = seeded
    owner._embedder.delay = 0.15  # 8 encodes: about 1.2 s in the owned worker
    first = asyncio.create_task(owner.sync_commands(_Home(), _SlowLLM(verb="nuevo", delay=0)))
    async with asyncio.timeout(5):
        while not owner._index_lock.locked():
            await asyncio.sleep(0.01)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert owner._index_lock.locked()  # the shielded worker is still indexing
    second = _SlowLLM(verb="otro", delay=0)
    with pytest.raises(CommandSyncError, match="chroma_sync_already_running"):
        await owner.sync_commands(_Home(), second)
    assert second.calls == 0  # refused at entry: no HA reads or cloud calls are spent
    while owner._index_lock.locked():
        await asyncio.sleep(0.02)
    # The first worker finished its switch; a later sync runs normally.
    assert (await asyncio.to_thread(_documents, owner))[0].startswith("nuevo")
    owner._embedder.delay = 0
    assert await owner.sync_commands(_Home(), _SlowLLM(verb="otro", delay=0)) == 8


async def test_a_pointer_to_a_missing_collection_is_degraded_not_silently_empty(seeded, tmp_path):
    """R3: the pointer target is opened, never created; the base collection serves."""
    owner, _ = seeded
    owner._pointer_collection.modify(metadata={"active": "home_assistant_commands__gone"})
    legacy = sorted(owner._client.get_collection("home_assistant_commands").get()["documents"])
    await owner.aclose()
    reopened = ChromaSync(str(tmp_path / "index"), "synthetic", "cpu")
    await asyncio.to_thread(reopened.initialize)
    try:
        assert reopened._commands_collection.name == "home_assistant_commands"
        assert reopened.index_health()["degraded"] == "pointer_target_missing"
        assert "home_assistant_commands__gone" not in await asyncio.to_thread(_collections, reopened)
        assert await asyncio.to_thread(_documents, reopened) == legacy
        health = HealthAggregator(command_index_provider=reopened.index_health).get_system_health()
        index_health = [s for s in health.subsystems if s.name == "command_index"]
        assert [s.status.value for s in index_health] == ["degraded"]
        assert "pointer_target_missing" in index_health[0].detail
        # A successful sync repairs it.
        assert await reopened.sync_commands(_Home(), _SlowLLM(verb="nuevo", delay=0)) == 8
        assert reopened.index_health()["degraded"] is None
    finally:
        await reopened.aclose()


async def test_the_legacy_collection_is_kept_for_rollback(index):
    """R4: pre-0d2 code opens 'home_assistant_commands'; syncs never drop it."""
    owner, _ = index
    legacy = owner._client.get_collection("home_assistant_commands")
    legacy.add(ids=["pre0d2"], embeddings=[[0.1, 0.2, 0.3]], documents=["prende el living (pre-0d2)"])
    for verb in ("uno", "dos"):
        assert await owner.sync_commands(_Home(), _SlowLLM(verb=verb, delay=0)) == 8
    assert owner._client.get_collection("home_assistant_commands").get()["documents"] == [
        "prende el living (pre-0d2)"]
    assert owner._commands_collection.name != "home_assistant_commands"


async def test_a_sync_switches_the_persisted_pointer_and_retires_old_collections(seeded, tmp_path):
    owner, _ = seeded
    owner._client.create_collection(name="home_assistant_commands__leftover1")  # an interrupted sync
    assert await owner.sync_commands(_Home(count=2), _SlowLLM(verb="nuevo", delay=0)) == 4
    names = await asyncio.to_thread(_collections, owner)
    active = owner._commands_collection.name
    assert names == sorted(["home_assistant_commands", "home_assistant_commands__pointer",
                            "home_assistant_routines", active])
    assert owner._pointer_collection.metadata == {"active": active}
    await owner.aclose()
    reopened = ChromaSync(str(tmp_path / "index"), "synthetic", "cpu")
    await asyncio.to_thread(reopened.initialize)
    try:
        assert reopened._commands_collection.name == active
        assert await asyncio.to_thread(_documents, reopened) == sorted(
            f"nuevo light.room_{e} {n}" for e in range(2) for n in range(2))
    finally:
        await reopened.aclose()


async def test_no_phrases_never_replaces_the_index(seeded):
    owner, _ = seeded
    old = await asyncio.to_thread(_documents, owner)
    with pytest.raises(CommandSyncError, match="chroma_sync_no_phrases"):
        await owner.sync_commands(_Home(), _SlowLLM(delay=0, phrases=0))
    assert await asyncio.to_thread(_documents, owner) == old


async def test_a_second_concurrent_sync_is_refused(seeded):
    owner, _ = seeded
    llm = _SlowLLM(verb="nuevo", delay=0.1)
    running = asyncio.create_task(owner.sync_commands(_Home(), llm))
    await llm.started.wait()
    with pytest.raises(CommandSyncError, match="chroma_sync_already_running") as caught:
        await owner.sync_commands(_Home(), _SlowLLM(verb="otro", delay=0))
    assert caught.value.index_kept
    assert await running == 8
    assert (await asyncio.to_thread(_documents, owner))[0].startswith("nuevo")


@pytest.mark.parametrize("phase", ["llm", "encode"])
async def test_aclose_during_a_sync_aborts_it_within_its_bound(seeded, phase):
    """S2: the production close path (aclose, 1 s bound) during a sync.

    The LLM call in flight is cancelled; in the encode phase the worker stops at
    its next phrase. No WorkCloseTimeout, the staging work is dropped, and the
    persisted index is the previous one.
    """
    owner, _ = seeded
    old = await asyncio.to_thread(_documents, owner)
    before = await asyncio.to_thread(_collections, owner)
    path = owner.chroma_path
    if phase == "llm":
        llm = _SlowLLM(verb="nuevo", delay=30)  # a hung cloud call
    else:
        llm = _SlowLLM(verb="nuevo", delay=0, phrases=50)
        owner._embedder.delay = 0.1  # 200 encodes: 20 s if not aborted
    running = asyncio.create_task(owner.sync_commands(_Home(), llm))
    await llm.started.wait()
    if phase == "encode":
        while owner._async_work.status()["operations"] == 0:
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.3)
    started = time.monotonic()
    await owner.aclose()  # raises WorkCloseTimeout if the sync is still running after 1 s
    assert time.monotonic() - started < 1.0
    with pytest.raises(CommandSyncError, match="chroma_sync_aborted") as caught:
        await running
    assert caught.value.index_kept
    if phase == "llm":
        assert llm.cancelled == 1
    assert owner.resource_status()["client_closed"]
    reopened = ChromaSync(path, "synthetic", "cpu")
    await asyncio.to_thread(reopened.initialize)
    try:
        assert await asyncio.to_thread(_documents, reopened) == old
        assert await asyncio.to_thread(_collections, reopened) == before
    finally:
        await reopened.aclose()


async def test_a_sync_after_close_is_refused(index):
    owner, _ = index
    await owner.aclose()
    with pytest.raises(CommandSyncError, match="chroma_sync_closed"):
        await owner.sync_commands(_Home(), _SlowLLM(delay=0))


class _SlowIndex:
    def __init__(self):
        self.initialized_on_main = self.stats_on_main = None

    def initialize(self):
        self.initialized_on_main = threading.current_thread() is threading.main_thread()
        time.sleep(0.5)  # PersistentClient + SentenceTransformer construction

    def get_stats(self):
        self.stats_on_main = threading.current_thread() is threading.main_thread()
        time.sleep(0.3)  # collection count on a large index
        return {"commands_phrases": 0}


async def test_pipeline_initializes_the_index_off_the_loop():
    index = _SlowIndex()
    processor = MagicMock()
    processor.load_models = MagicMock()
    pipeline = VoicePipeline(audio_loop=None, command_processor=processor, request_router=MagicMock(),
                             response_handler=MagicMock(), feature_manager=None, chroma_sync=index)
    probe = _probe()
    probe.start()
    await pipeline.run()
    await probe.stop()
    assert index.initialized_on_main is False and index.stats_on_main is False
    assert probe.snapshot()["max_ms"] < BUDGET_MS, probe.snapshot()


class _SlowSave:
    """CooldownManager persistence that takes 300 ms (fsync under disk load)."""

    def __init__(self, manager):
        self.on_main = []
        self._original = manager._save_to_disk
        manager._save_to_disk = self

    def __call__(self, *args, **kwargs):
        self.on_main.append(threading.current_thread() is threading.main_thread())
        time.sleep(0.3)
        return self._original(*args, **kwargs)


class _Client:
    def __init__(self, error=None):
        self.error = error

    async def complete(self, prompt, max_tokens=256, **_):
        if self.error is not None:
            raise self.error
        return "ok"


async def test_router_cooldown_persistence_stays_off_the_loop(tmp_path):
    from src.llm.cooldown import CooldownManager
    from src.llm.router import LLMRouter
    from src.llm.types import EndpointKind, LLMEndpoint

    manager = CooldownManager(persistence_path=tmp_path / "cooldown.json")
    save = _SlowSave(manager)
    router = LLMRouter([
        LLMEndpoint(id="primary", kind=EndpointKind.FAST_ROUTER, client=_Client(TimeoutError()), priority=0),
        LLMEndpoint(id="secondary", kind=EndpointKind.HTTP_REASONER, client=_Client(), priority=1),
    ], manager)
    probe = _probe()
    probe.start()
    result = await router.complete("hola", max_tokens=8)
    await probe.stop()
    assert result.endpoint_id == "secondary"
    assert save.on_main == [False, False]  # failure cooldown + success reset
    assert not manager.is_available("primary") and (tmp_path / "cooldown.json").exists()
    assert probe.snapshot()["max_ms"] < BUDGET_MS, probe.snapshot()


async def test_graph_runs_the_probe_and_metrics_publish_it(tmp_path, monkeypatch):
    from contextlib import AsyncExitStack

    from src.application import ApplicationFactories
    from src.main import build_application
    import src.vectordb.chroma_sync as index_module
    from tests.mocks.application_graph import OfflineEncoder, OfflineSTT, OfflineTTS, offline_config, offline_home

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(index_module, "SentenceTransformer", lambda *a, **kw: OfflineEncoder())
    stt, tts = OfflineSTT(), OfflineTTS()
    leaves = ApplicationFactories(stt=lambda _: stt, tts=lambda _: tts, preflight=lambda: None)
    async with offline_home() as (url, _):
        async with AsyncExitStack() as stack:
            config = offline_config(tmp_path, url)
            config["monitoring"] = {"enabled": False, "loop_lag": {"interval_s": 0.02}}
            graph = await build_application(config, shutdown=stack, factories=leaves)
            probe = graph.loop_lag
            assert isinstance(probe, LoopLagProbe) and probe.config.interval_s == 0.02

            async def idle():
                await asyncio.Event().wait()

            async def voice():
                await asyncio.sleep(0.3)

            monkeypatch.setattr(graph.dashboard, "start", idle)
            monkeypatch.setattr(graph.pipeline, "run", voice)
            await graph.run()
            assert not probe.running
            metrics = graph.dashboard.health_aggregator.get_metrics()["loop_lag"]
            assert metrics["samples"] >= 5 and metrics["interval_ms"] == 20.0
