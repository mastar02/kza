"""Offline promotion must select compatible code and data as one slot."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sqlite3
import os
import subprocess
import sys
import tarfile

import pytest

from scripts.build_runtime import build_runtime
from scripts.stage_runtime import activate_slot, prepare_slot


@pytest.fixture
def archive(tmp_path):
    """Use the real builder and a committed application, without starting it."""
    output = tmp_path / "runtime.tar.gz"
    build_runtime(Path(__file__).resolve().parents[3], output)
    return output


@pytest.fixture
def state(tmp_path):
    source = tmp_path / "offline-backup"
    (source / "config").mkdir(parents=True)
    (source / "data").mkdir()
    (source / "config" / "settings.yaml").write_text("schema: old\n")
    (source / ".env").write_text("TEST_ONLY=private\n")
    with sqlite3.connect(source / "data" / "state.db") as db:
        db.execute("CREATE TABLE memories (value TEXT)")
        db.execute("INSERT INTO memories VALUES ('before migration')")
    return source


def test_promote_and_rollback_restore_runtime_and_sqlite(archive, state, tmp_path):
    root = tmp_path / "staging"
    old_archive = tmp_path / "old.tar.gz"
    build_runtime(Path(__file__).resolve().parents[3], old_archive,
                  "0fd3d166f1b17726f8ef998b320f37a42ff11d81")
    original = prepare_slot(old_archive, root, "old", state)
    assert activate_slot(root, "old") is None
    old_runtime = (root / "current").resolve()

    # A real copy of stopped state becomes a candidate's independent state.
    candidate = prepare_slot(archive, root, "candidate", old_runtime.parent / "state")
    assert original["git_commit"] != candidate["git_commit"]
    assert original["files"]["src/main.pyc"] != candidate["files"]["src/main.pyc"]
    new_runtime = root / "slots" / "candidate" / "runtime"
    with sqlite3.connect(new_runtime / "data" / "state.db") as db:
        db.execute("ALTER TABLE memories ADD COLUMN provenance TEXT")
        db.execute("UPDATE memories SET value = 'after migration'")
    (new_runtime / "config" / "settings.yaml").write_text("schema: new\n")
    assert activate_slot(root, "candidate") == "slots/old/runtime"
    assert (root / "current" / "config" / "settings.yaml").read_text() == "schema: new\n"

    assert activate_slot(root, "old") == "slots/candidate/runtime"
    assert (root / "current").resolve() == old_runtime
    with sqlite3.connect(root / "current" / "data" / "state.db") as db:
        assert db.execute("SELECT * FROM memories").fetchall() == [("before migration",)]
        assert len(db.execute("PRAGMA table_info(memories)").fetchall()) == 1
    assert (root / "current" / "config" / "settings.yaml").read_text() == "schema: old\n"
    assert json.loads((old_runtime / "release.json").read_text()) == original
    assert not list(old_runtime.rglob("*.py"))
    assert (old_runtime.parent.stat().st_mode & 0o777) == 0o700
    assert (old_runtime.parent / "state" / ".env").stat().st_mode & 0o777 == 0o600


def test_compiled_main_and_frontend_outside_checkout(archive, state, tmp_path):
    root = tmp_path / "staging"
    prepare_slot(archive, root, "candidate", state)
    activate_slot(root, "candidate")
    environment = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME"}}
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(
        [sys.executable, "-c", "\n".join([
            "from pathlib import Path",
            "import os",
            "import src.main",
            "from fastapi.testclient import TestClient",
            "from src.dashboard.api import DashboardAPI",
            "assert Path(src.main.__file__).suffix == '.pyc'",
            "assert Path(src.main.__file__).resolve().is_relative_to(Path.cwd().resolve())",
            "from src.core.egress import EgressDenied",
            "from src.llm.reasoner import HttpReasoner",
            "reasoner = HttpReasoner(base_url='http://127.0.0.1:8200/v1')",
            "try:",
            "    reasoner.generate('offline smoke')",
            "except EgressDenied:",
            "    pass",
            "else:",
            "    raise AssertionError('Cloud gateway must require consent')",
            "import asyncio",
            "from contextlib import aclosing",
            "from src.llm.generation_stream import GenerationStream",
            "class OfflineReasoner:",
            "    def generate_stream(self, prompt):",
            "        yield {'token': 'compiled async generation'}",
            "async def check_generation():",
            "    async with aclosing(GenerationStream(OfflineReasoner()).astream('offline')) as stream:",
            "        assert ''.join([chunk['token'] async for chunk in stream]) == 'compiled async generation'",
            "    try:",
            "        async with aclosing(reasoner.astream('offline smoke')) as stream:",
            "            await anext(stream)",
            "    except EgressDenied:",
            "        pass",
            "    else:",
            "        raise AssertionError('Async gateway must require consent')",
            "asyncio.run(check_generation())",
            "import httpx, json",
            "from src.llm.http_generation import HTTPGenerationClient",
            "async def compiled_wire(request):",
            "    if request.method == 'GET':",
            "        return httpx.Response(200, json={'data': [{'id': 'compiled-model'}]})",
            "    choice = {'index': 0, 'finish_reason': 'stop'}",
            "    if json.loads(request.content)['stream']:",
            "        choice['delta'] = {'content': 'compiled HTTP'}",
            "        class Body(httpx.AsyncByteStream):",
            "            async def __aiter__(self):",
            "                yield ('data: ' + json.dumps({'choices': [choice]}) + '\\n\\n').encode()",
            "        return httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=Body())",
            "    choice['message'] = {'content': 'compiled HTTP'}",
            "    return httpx.Response(200, json={'choices': [choice], 'usage': {'completion_tokens': 2}})",
            "async def check_compiled_http():",
            "    native = HttpReasoner(model='compiled-model', api_style='chat', cloud_consent=True)",
            "    native._new_async_client = lambda url: HTTPGenerationClient(url, api_key='synthetic', http_client=httpx.AsyncClient(transport=httpx.MockTransport(compiled_wire)))",
            "    async with aclosing(native.astream('synthetic')) as stream:",
            "        assert ''.join([chunk['token'] async for chunk in stream]) == 'compiled HTTP'",
            "    assert await native.complete('synthetic') == 'compiled HTTP'",
            "    assert native._last_metrics['tokens'] == 2",
            "asyncio.run(check_compiled_http())",
            "from src.pipeline.response_handler import ResponseHandler, DeliveryState",
            "empty_voice = asyncio.run(ResponseHandler(None).aspeak(''))",
            "assert empty_voice.delivery is DeliveryState.EMPTY and not empty_voice.speech_handled",
            "from contextlib import AsyncExitStack",
            "from src.pipeline.voice_services import build_voice_services",
            "async def check_voice_services():",
            "    async with AsyncExitStack() as stack:",
            "        services = build_voice_services({'alerts': {'enabled': False}, 'tts': {'streaming': {'response_timeouts': {'generation_s': 12.0}}}}, tts=None, shutdown=stack)",
            "        assert services.responses.budgets.generation_s == 12.0",
            "        await services.features.start()",
            "    status = services.responses.get_delivery_status()",
            "    assert status['closed'] and status['workers'] == status['controllers'] == status['generations'] == 0",
            "    assert services.features.intercom._process_task.done()",
            "asyncio.run(check_voice_services())",
            "from src.ambient.store import AmbientStore",
            "from src.ambient.types import AmbientUtterance",
            "from tempfile import TemporaryDirectory",
            "async def check_capture_retention():",
            "    with TemporaryDirectory() as private:",
            "        clock = [2000000000.0]",
            "        store = AmbientStore(str(Path(private)/'capture.db'), clock=lambda: clock[0], audio_dir=private)",
            "        await store.init()",
            "        try:",
            "            capture = AmbientUtterance('offline', clock[0], clock[0]+1, text='synthetic')",
            "            row_id = await store.add(capture)",
            "            clip = Path(private)/'clip.flac'",
            "            clip.write_bytes(b'synthetic clip')",
            "            assert await store.set_audio_path(row_id, str(clip))",
            "            assert await store.read_audio(row_id) == b'synthetic clip'",
            "            clock[0] += 48*3600",
            "            assert await store.undistilled_live() == []",
            "            assert await store.read_audio(row_id) is None",
            "            assert await store.purge_expired() == 1 and not clip.exists()",
            "            assert (await store.audio_cleanup_status())['pending'] == 0",
            "        finally:",
            "            await store.close()",
            "asyncio.run(check_capture_retention())",
            "from src.memory.services import build_memory_services",
            "from src.memory.fact_ledger import MemoryReviewer",
            "from unittest.mock import MagicMock",
            "from uuid import uuid4",
            "async def check_memory_services():",
            "    with TemporaryDirectory() as private:",
            "        clock = [2000000000.0]",
            "        client = MagicMock()",
            "        client.get_or_create_collection.return_value.count.return_value = 0",
            "        config = {'memory': {'privacy_db_path': str(Path(private)/'privacy.db'), 'preferences_path': str(Path(private)/'preferences.json')}}",
            "        async with AsyncExitStack() as stack:",
            "            services = await build_memory_services(config, shutdown=stack, chroma_client=client, clock=lambda: clock[0])",
            "            memory = services.manager.long_term",
            "            source = services.manager.source_for_capture(str(uuid4()), clock[0], 'offline')",
            "            proposal = await memory.acall(memory.propose, 'synthetic memory', (source,))",
            "            assert await memory.asearch_relevant('synthetic') == []",
            "            confirmed = await memory.acall(memory.confirm, proposal['id'], MemoryReviewer('offline-admin'))",
            "            assert confirmed['state'] == 'confirmed' and confirmed['expires_at'] is None",
            "            assert await memory.acall(memory.delete_fact, proposal['id'], MemoryReviewer('offline-admin'))",
            "            from src.orchestrator.context_manager import ContextManager",
            "            from src.orchestrator.context_persister import ContextPersister",
            "            persister = ContextPersister(Path(private)/'contexts', source_policy=memory.ledger, clock=lambda: clock[0])",
            "            contexts = ContextManager(persister=persister, source_policy=memory.ledger, clock=lambda: clock[0], io_worker=memory.acall)",
            "            stack.push_async_callback(contexts.aclose)",
            "            context = await contexts.aget_or_create('offline-user')",
            "            context.compacted_summary = 'synthetic sourced summary'",
            "            context.summary_sources = (source,)",
            "            await memory.acall(persister.save, context)",
            "            assert 'synthetic sourced summary' in contexts.build_prompt('offline-user', 'query')",
            "            clock[0] += 48*3600",
            "            assert 'synthetic sourced summary' not in contexts.build_prompt('offline-user', 'query')",
            "            assert await memory.acall(persister.load, 'offline-user') is None",
            "            assert await memory.acall(persister.purge_expired) == 1",
            "            from src.analytics.services import build_event_logger",
            "            analytics = await build_event_logger({'analytics': {'events_db': str(Path(private)/'events.db')}}, shutdown=stack, source_policy=memory.ledger, clock=lambda: clock[0], io_worker=memory.acall)",
            "            capture = services.manager.source_for_capture(str(uuid4()), clock[0], 'offline')",
            "            await analytics.alog('light.synthetic', 'turn_on', trigger_phrase='synthetic private text', extra_context={'rms': .2}, sources=(capture,))",
            "            assert (await analytics.acall(analytics.get_events))[0].trigger_phrase == 'synthetic private text'",
            "            clock[0] += 48*3600",
            "            assert (await analytics.acall(analytics.get_events))[0].trigger_phrase is None",
            "            await analytics.acall(analytics.cleanup_old_events)",
            "            from src.memory.transient import TransientCache",
            "            from src.orchestrator.priority_queue import PriorityRequestQueue",
            "            from src.nlu.llm_router import LLMCommandRouter",
            "            current_source = services.manager.source_for_capture(str(uuid4()), clock[0], 'offline')",
            "            cache = TransientCache(2, source_policy=memory.ledger)",
            "            assert cache.put('synthetic private key', {'value': 'synthetic'}, (current_source,))",
            "            queue = PriorityRequestQueue(source_policy=memory.ledger)",
            "            receipt = queue.enqueue('offline-user', 'synthetic private input', memory_sources=(current_source,))",
            "            assert receipt.complete('synthetic response')",
            "            clock[0] += 48*3600",
            "            assert cache.get('synthetic private key') is None",
            "            queue.prune_private()",
            "            assert receipt.text == '' and receipt.result is None and queue._user_requests == {}",
            "            classifier = LLMCommandRouter(None, source_policy=memory.ledger)",
            "            assert (await classifier.classify('expired input', sources=(current_source,))).is_command is None",
            "            await classifier.aclose()",
            "        assert services.maintenance.done()",
            "        assert memory.resource_status()['operations'] == 0 and not memory.resource_status()['writer_lease']",
            "asyncio.run(check_memory_services())",
            "from src.orchestrator.dispatcher import MultiUserOrchestrator",
            "orchestrator = MultiUserOrchestrator(None, None, None)",
            "assert orchestrator.dispatcher.queue is orchestrator.queue",
            "from unittest.mock import MagicMock",
            "from src.home_assistant.control_policy import ActionOrigin, HomeActor",
            "from src.home_assistant.home_actions import HomeActionRequest, HomeActionService",
            "ha = MagicMock()",
            "request = HomeActionRequest('light.test_lamp', 'turn_on', 'compiled-control', origin=ActionOrigin.VOICE)",
            "result = asyncio.run(HomeActionService(ha).execute(request, actor=HomeActor.household()))",
            "assert result.status == 'failed' and result.reason == 'action_not_selected'",
            "assert ha.mock_calls == []",
            "from src.home_assistant.operations import compile_effect",
            "typed = HomeActionRequest('climate.test_ac', 'set_temperature', 'compiled-temperature', temperature=23.5)",
            "effect = compile_effect(typed)",
            "assert effect.domain == 'climate' and effect.service == 'set_temperature' and effect.data == {'temperature': 23.5}",
            "from src.nlu.home_command import parse_home_command",
            "spoken = parse_home_command('Nexa poné el aire a veintidós coma cinco grados', wake_words=('nexa',))",
            "assert spoken.operation == 'set_temperature' and spoken.parameters == {'temperature': 22.5}",
            "from src.home_assistant.voice_resolver import HomeVoiceResolver",
            "assert asyncio.run(HomeVoiceResolver(HomeActionService(ha)).resolve(spoken, room_id=None)).reason == 'no_selected_target'",
            "from src.home_assistant.voice_actions import VoiceHomeActionExecutor",
            "denied = asyncio.run(VoiceHomeActionExecutor(ha, HomeActionService(ha)).execute({'domain': 'climate', 'service': 'set_temperature', 'entity_id': 'climate.test_ac', 'data': {'temperature': 22.5}}))",
            "assert denied.status == 'failed' and denied.reason == 'action_not_selected'",
            "ha.call_service_once.assert_not_called()",
            "from src.home_assistant.control_policy import EntityGrant",
            "from src.home_assistant.effect_plans import expand_effect_plan, scene_effects_from_states",
            "effects = scene_effects_from_states({'light.test_group': {'state': 'on', 'color_temp_kelvin': 2700, 'brightness': 179}})",
            "grants = (EntityGrant('scene.test_evening', frozenset({'activate_scene'}), scene_effects=effects), EntityGrant('light.test_group', frozenset({'turn_on'}), members=('light.test_lamp',)), EntityGrant('light.test_lamp', frozenset({'turn_on'})))",
            "plan = expand_effect_plan(HomeActionRequest('scene.test_evening', 'activate_scene', 'compiled-scene'), grants)",
            "assert len(plan.effects) == 1 and plan.effects[0].entity_id == 'light.test_lamp' and plan.effects[0].brightness == 179",
            "assert compile_effect(plan.effects[0]).data == {'color_temp_kelvin': 2700, 'brightness': 179}",
            "from src.tts.piper_tts import StreamingAudioPlayer",
            "player = StreamingAudioPlayer(sample_rate=24000)",
            "assert player.play_stream(iter(())) is True",
            "assert player.get_stats()['state'] == 'completed' and player.get_stats()['played_samples'] == 0",
            "from src.routines.models import parse_routine_spec, spec_to_dict",
            "spec = {'name': 'compiled', 'description': '', 'enabled': True, 'timezone': 'America/Montevideo', 'schedule': None, 'conditions': [], 'actions': [{'entity_id': 'climate.test_ac', 'operation': 'set_temperature', 'temperature': 23.5}]}",
            "assert spec_to_dict(parse_routine_spec(spec)) == spec",
            "client = TestClient(DashboardAPI(home_enabled=True).app)",
            "for path in ('/', '/home/home.css', '/home/home.js', '/home/request-id.js', '/home/routines.js', '/obs/src/session.js'):",
            "    assert client.get(path).status_code == 200, path",
            "os.environ.pop('KZA_DASHBOARD_TOKEN', None)",
            "assert client.get('/api/health').status_code == 503",
            "os.environ['KZA_DASHBOARD_TOKEN'] = 'compiled-smoke-session'",
            "assert client.get('/api/health').status_code == 401",
            "assert client.get('/api/health', headers={'Authorization': 'Bearer compiled-smoke-session'}).status_code == 200",
        ])],
        cwd=root / "current", env=environment, check=True,
        capture_output=True, text=True, timeout=60,
    )


def test_bad_candidate_does_not_replace_active_slot(archive, state, tmp_path):
    root = tmp_path / "staging"
    prepare_slot(archive, root, "old", state)
    activate_slot(root, "old")
    prepare_slot(archive, root, "broken", state)
    (root / "slots" / "broken" / "runtime" / "src" / "main.pyc").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        activate_slot(root, "broken")
    assert (root / "current").readlink() == Path("slots/old/runtime")


def test_extra_sources_or_redirected_state_cannot_be_activated(archive, state, tmp_path):
    root = tmp_path / "staging"
    prepare_slot(archive, root, "candidate", state)
    runtime = root / "slots" / "candidate" / "runtime"
    (runtime / "src" / "main.py").write_text("raise RuntimeError('unexpected source')")
    with pytest.raises(ValueError, match="Unexpected"):
        activate_slot(root, "candidate")
    (runtime / "src" / "main.py").unlink()
    (runtime / "data").unlink()
    (runtime / "data").symlink_to(state / "data")
    with pytest.raises(ValueError, match="same slot"):
        activate_slot(root, "candidate")
    assert not (root / "current").exists()


def test_refuses_overwrite_and_linked_state(archive, state, tmp_path):
    root = tmp_path / "staging"
    prepare_slot(archive, root, "old", state)
    with pytest.raises(FileExistsError):
        prepare_slot(archive, root, "old", state)
    (state / "data" / "escape").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="links"):
        prepare_slot(archive, root, "bad", state)
    assert not (root / "slots" / "bad").exists()


@pytest.mark.parametrize("name,kind", [
    ("../escape", "file"), ("/escape", "file"), ("src/evil.py", "file"),
    ("src/link", "link"), ("src/main.pyc", "duplicate"), ("data/private", "file"),
    ("src//alias", "file"),
])
def test_rejects_unsafe_archives(name, kind, state, tmp_path):
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        member = tarfile.TarInfo(name)
        member.size = 1
        if kind == "link":
            member.type = tarfile.SYMTYPE
            member.linkname = "../../escape"
        bundle.addfile(member, io.BytesIO(b"x"))
        if kind == "duplicate":
            bundle.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="unsafe"):
        prepare_slot(archive, tmp_path / "root", "bad", state)
    assert not (tmp_path / "escape").exists()


def test_refuses_manifest_checksum_or_python_mismatch(state, tmp_path):
    archive = tmp_path / "bad-manifest.tar.gz"
    files = {"src/main.pyc": b"main", "src/__init__.pyc": b"init"}
    manifest = {
        "format": "kza-cpython-bytecode-v1", "git_commit": "a" * 40,
        "python": "3.13", "python_magic": importlib.util.MAGIC_NUMBER.hex(),
        "external_paths": ["config", "data", "models", "logs"],
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
    }
    files["src/main.pyc"] = b"altered"
    files["release.json"] = json.dumps(manifest).encode()
    with tarfile.open(archive, "w:gz") as bundle:
        for name, data in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            bundle.addfile(member, io.BytesIO(data))
    with pytest.raises(ValueError, match="manifest"):
        prepare_slot(archive, tmp_path / "root", "bad", state)
