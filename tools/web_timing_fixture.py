#!/usr/bin/env python3
"""Deterministic fixture backend for the T046-U1 web timing probe.

Serves the real ``DashboardAPI`` (NEXO at ``/``) on loopback HTTP through
uvicorn with a real ``DashboardAuthority`` and three individual credentials
(operator, home bound to a synthetic person, evaluator), a real voice
registry seeded from a fixed seed, a synthetic Home Assistant catalog with
receipts, and the line protocol on stdin/stdout that the probe uses as the
fixture's control pipe (t046/preregistration.json ``fixture_backend``).

Determinism (R4-5): every source capture id and conversation id is a UUID
derived from ``sha256(seed, kind, index)``; ``created_at`` and each source's
``captured_at`` are the fixture start minus a seed-determined offset in
[60 s, 6 h), distinct per capture, through the fixture's injected clock; each
source's TTL is its offset plus 12 h, so every capture expires 12 h after the
fixture start. Capture ids are fingerprints over sources that include absolute
times, so they differ between fixture starts; the digest leaves them out.

Control pipe (one JSON object per line on stdout; commands on stdin):

* first stdout line: ``{"event": "ready", ...}`` with origin, digest and the
  synthetic credential tokens;
* ``position_check <ordinal>``: projects the first captures page through
  ``VoiceLedger.project('captures', None, <operator VoiceReviewer>)`` on the
  served ledger (R5-2), compares its first 50 rows with the position map,
  writes the result into the run directory and answers on stdout;
* ``health``: answers whether the HTTP server is started;
* ``catalog <ok|unavailable>``: switches the home catalog to 503 (freshness);
* ``quit``: stops the server.

Usage::

    python -m tools.web_timing_fixture --run-dir DIR [--seed 46001] [--harness-pages]
"""

import argparse
import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
import hashlib
import json
import logging
import os
from pathlib import Path
import socket
import sys
import tempfile
import time
from uuid import UUID

import numpy as np


logger = logging.getLogger(__name__)

SEED = 46001
CAPTURE_COUNT = 120
GROUP_COUNT = 4
PROFILE_COUNT = 3
PROFILE_CAPTURES = 3
POSITIONS = 50
OFFSET_MIN_S = 60
OFFSET_MAX_S = 6 * 3600
TTL_AFTER_START_S = 12 * 3600
SESSION_TTL_S = 12 * 3600
FIXTURE_LIFETIME_S = 10 * 3600
VOICE_ROOMS = ('Sala sintetica', 'Cocina sintetica')
WORDS = ('luz', 'cocina', 'música', 'persiana', 'mañana', 'temperatura', 'sala', 'recordatorio', 'noche', 'café')
EMBEDDING_DIMENSIONS = 192
HOME_ROOMS = (
    ('sala', 'Sala de estar', 'la sala'),
    ('cocina', 'Cocina', 'la cocina'),
    ('dormitorio', 'Dormitorio', 'el dormitorio'),
    ('estudio', 'Estudio', 'el estudio'),
)
UNASSIGNED_LIGHTS = ('light.pasillo', 'light.entrada')
CLIMATE_MODES = ['off', 'cool', 'dry', 'fan_only', 'auto', 'heat']
# The one home action the core script activates on #acciones (R5-4 lookup in the probe).
ACTION_ENTITY = 'light.sala'
ACTION_OPERATION = 'turn_on'


def derived_uuid(seed: int, kind: str, index: int) -> str:
    """Return a UUID derived from sha256(seed, kind, index) (R4-5)."""
    digest = hashlib.sha256(f'{seed}:{kind}:{index}'.encode()).digest()[:16]
    return str(UUID(bytes=digest, version=4))


@dataclass
class CapturePlan:
    """Seed-determined content of one synthetic capture."""

    index: int
    offset_s: int
    duration_s: float
    room_id: str
    transcript: str
    group: int
    audio: np.ndarray = field(repr=False)

    def audio_sha256(self) -> str:
        return hashlib.sha256(self.audio.tobytes()).hexdigest()


@dataclass
class FixturePlan:
    """Everything the fixture serves that the seed decides."""

    seed: int
    captures: list
    profiles: list
    permutation: list

    def positions(self) -> list:
        """Capture indices in server order: newest first, distinct offsets (R4-5)."""
        return [capture.index for capture in sorted(self.captures, key=lambda item: item.offset_s)]

    def digest(self) -> str:
        """sha256 over seed-determined content only; no absolute time or capture id (R4-5)."""
        offsets = sorted(range(len(self.captures)), key=lambda index: self.captures[index].offset_s)
        content = {
            'seed': self.seed,
            'counts': {'captures': len(self.captures), 'groups': GROUP_COUNT, 'profiles': len(self.profiles),
                       'positions': POSITIONS},
            'captures': [{'index': capture.index, 'transcript': capture.transcript, 'room': capture.room_id,
                          'duration_s': capture.duration_s, 'audio_sha256': capture.audio_sha256(),
                          'group': capture.group} for capture in self.captures],
            'profiles': self.profiles,
            'offset_order': offsets,
            'preview_permutation': self.permutation,
        }
        return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def build_plan(seed: int = SEED) -> FixturePlan:
    """Derive the dataset from the seed alone (fixture_backend.seed)."""
    rng = np.random.default_rng(seed)
    offsets = rng.choice(np.arange(OFFSET_MIN_S, OFFSET_MAX_S), size=CAPTURE_COUNT, replace=False)
    groups = rng.permutation(np.repeat(np.arange(GROUP_COUNT), CAPTURE_COUNT // GROUP_COUNT))
    captures = []
    for index in range(CAPTURE_COUNT):
        duration = round(6.0 + 0.1 * int(rng.integers(0, 31)), 1)
        words = ' '.join(WORDS[int(value)] for value in rng.integers(0, len(WORDS), size=6))
        samples = int(round(duration * 16000))
        audio = (rng.standard_normal(samples) * 0.05).clip(-0.5, 0.5).astype(np.float32)
        captures.append(CapturePlan(index, int(offsets[index]), duration, VOICE_ROOMS[int(rng.integers(0, 2))],
                                    f'Frase sintética {index}: {words}', int(groups[index]), audio))
    # Three confirmed profiles from the three newest captures of groups 0..2.
    newest = sorted(captures, key=lambda item: item.offset_s)
    profiles = []
    for group in range(PROFILE_COUNT):
        members = [capture.index for capture in newest if capture.group == group][:PROFILE_CAPTURES]
        profiles.append({'group': group, 'name': f'Perfil sintético {"ABC"[group]}', 'captures': members})
    permutation = [int(value) for value in rng.permutation(POSITIONS)]
    return FixturePlan(seed, captures, profiles, permutation)


class FixtureClock:
    """Injected privacy clock: a fixed override while seeding, wall time afterwards."""

    def __init__(self):
        self.override = None

    def __call__(self) -> float:
        return time.time() if self.override is None else self.override

    @property
    def now(self) -> float:
        return self()


class SeededEncoder:
    """Synthetic encoder whose embedding is the capture's group vector (no model, no GPU)."""

    model_name = 'synthetic-ecapa-t046'
    sample_rate = 16000
    _model_type = 'synthetic'

    def __init__(self, plan: FixturePlan):
        self._groups = {capture.audio_sha256(): capture.group for capture in plan.captures}

    def get_embedding(self, audio):
        vector = np.zeros(EMBEDDING_DIMENSIONS, dtype=np.float32)
        vector[self._groups[hashlib.sha256(np.asarray(audio, dtype=np.float32).tobytes()).hexdigest()]] = 1.0
        return vector


def write_json_atomic(path: Path, value) -> None:
    """Write JSON through a temp file with fsync and an atomic replace, mode 0600."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(handle, 'w') as output:
            json.dump(value, output, indent=1, sort_keys=True, ensure_ascii=False)
            output.write('\n')
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def home_graph():
    """Synthetic installation: 4 rooms, 14 entities, receipts through the real action service."""
    from src.home_assistant.control_policy import EntityGrant, HomeControlPolicy
    from src.home_assistant.home_actions import HomeActionService
    from src.rooms.room_context import RoomConfig
    from tests.integration.test_typed_home_operations import TypedHA

    ha = TypedHA()
    ha.states.clear()
    grants, rooms = [], {}
    for index, (room_id, name, display) in enumerate(HOME_ROOMS):
        light, climate, cover = f'light.{room_id}', f'climate.{room_id}', f'cover.{room_id}'
        ha.states[light] = {'entity_id': light, 'state': 'on' if index % 2 == 0 else 'off',
                            'attributes': {'friendly_name': f'Luz {name.lower()}', 'supported_color_modes': ['brightness'],
                                           'brightness': 160}}
        ha.states[climate] = {'entity_id': climate, 'state': 'cool', 'attributes': {
            'friendly_name': f'Clima {name.lower()}', 'supported_features': 1, 'hvac_modes': CLIMATE_MODES.copy(),
            'min_temp': 7, 'max_temp': 35, 'target_temp_step': 0.5, 'temperature': 22, 'current_temperature': 24}}
        ha.states[cover] = {'entity_id': cover, 'state': 'open', 'attributes': {
            'friendly_name': f'Persiana {name.lower()}', 'supported_features': 15, 'device_class': 'blind',
            'current_position': 40}}
        grants += [EntityGrant(light, frozenset({'turn_on', 'turn_off', 'brightness'})),
                   EntityGrant(climate, frozenset({'set_temperature', 'set_hvac_mode'}), temperature_min=7,
                               temperature_max=35, hvac_modes=frozenset(CLIMATE_MODES)),
                   EntityGrant(cover, frozenset({'open_cover', 'close_cover', 'stop_cover', 'set_cover_position'}),
                               cover_device_class='blind')]
        rooms[room_id] = RoomConfig(room_id, name, display, default_light=light, default_climate=climate,
                                    default_cover=cover)
    for light in UNASSIGNED_LIGHTS:
        ha.states[light] = {'entity_id': light, 'state': 'off', 'attributes': {
            'friendly_name': 'Luz ' + light.split('.')[1], 'supported_color_modes': ['brightness'], 'brightness': 90}}
        grants.append(EntityGrant(light, frozenset({'turn_on', 'turn_off', 'brightness'})))
    return ha, HomeActionService(ha, policy=HomeControlPolicy(tuple(grants))), rooms


HARNESS_DIRECTORY = Path(__file__).resolve().parent / 'web_timing_pages'
REPOSITORY = Path(__file__).resolve().parents[1]


def repository_modules() -> list:
    """Repository files among this process's modules (recorded closure, R5-12)."""
    paths = set()
    for module in list(sys.modules.values()):
        location = getattr(module, '__file__', None)
        try:
            relative = Path(location).resolve().relative_to(REPOSITORY) if location else None
        except ValueError:
            relative = None
        if relative is not None and relative.parts and relative.parts[0] != '.venv':
            paths.add(str(relative))
    return sorted(paths)


class Fixture:
    """One fixture start: graph, seeded voices, credentials and the served app."""

    def __init__(self, run_dir: Path, seed: int = SEED, *, harness_pages: bool = False):
        self.run_dir = run_dir
        self.seed = seed
        self.harness_pages = harness_pages
        self.plan = build_plan(seed)
        self.clock = FixtureClock()
        self.started_at = None
        self.position_ids = []
        self.catalog_available = True
        self.server = None
        self.origin = None
        self.voices = None
        self.reviewer = None
        self.tokens = {}

    async def seed_voices(self, voices, encoder) -> None:
        """Observe every planned capture at its own injected time, then confirm the profiles."""
        from src.memory.fact_ledger import MemoryOrigin, MemorySource

        ids = {}
        for capture in self.plan.captures:
            captured = self.started_at - capture.offset_s
            source = MemorySource(derived_uuid(self.seed, 'capture', capture.index),
                                  derived_uuid(self.seed, 'conversation', capture.index),
                                  captured, captured + capture.offset_s + TTL_AFTER_START_S, MemoryOrigin.VOICE)
            before = await self.all_capture_ids(voices)
            self.clock.override = captured
            try:
                if not voices.observe(encoder, capture.audio, (source,), transcript=capture.transcript,
                                      room_id=capture.room_id):
                    raise RuntimeError(f'fixture_capture_refused:{capture.index}')
                async with asyncio.timeout(10):
                    while voices.resource_status()['pending'] or voices.resource_status()['operations']:
                        await asyncio.sleep(0.001)
            finally:
                self.clock.override = None
            after = await self.all_capture_ids(voices)
            new = after - before
            if len(new) != 1:
                raise RuntimeError(f'fixture_capture_missing:{capture.index}')
            ids[capture.index] = new.pop()
        self.capture_ids = ids
        for profile in self.plan.profiles:
            members = tuple(ids[index] for index in profile['captures'])
            await voices.acall(voices.ledger.confirm, members, profile['name'],
                               derived_uuid(self.seed, 'profile-request', profile['group']), self.reviewer)
        self.position_ids = [ids[index] for index in self.plan.positions()[:POSITIONS]]

    @staticmethod
    async def all_capture_ids(voices) -> set:
        identities, cursor = set(), None
        while True:
            snapshot = await voices.acall(voices.ledger.snapshot, cursor)
            identities |= {row['id'] for row in snapshot['fingerprints']}
            cursor = snapshot.get('next_cursor')
            if not cursor:
                return identities

    def digest(self) -> str:
        """The digest this fixture start reports (fixture_backend.digest)."""
        return self.plan.digest()

    def position_map(self) -> dict:
        positions = self.plan.positions()
        return {'seed': self.seed, 'digest': self.digest(),
                'positions': [{'position': position, 'index': positions[position],
                               'capture_id': self.position_ids[position]} for position in range(POSITIONS)],
                'preview_permutation': self.plan.permutation}

    async def position_check(self, ordinal: int) -> dict:
        """R5-2: the served projection's first 50 rows against the map, in the fixture process."""
        result = {'ordinal': ordinal, 'ok': False, 'ids': [], 'time': time.time()}
        try:
            projection = await self.voices.acall(self.voices.ledger.project, 'captures', None, self.reviewer)
            read = [row['id'] for row in projection.rows[:POSITIONS]]
            result.update(ids=read, ok=read == self.position_ids)
        except Exception as error:  # noqa: BLE001 - every error is a fixture fault, recorded.
            result['error'] = type(error).__name__ + ':' + str(error)[:200]
        write_json_atomic(self.run_dir / f'position-check-{ordinal:05d}.json', result)
        return result

    async def build(self, stack: AsyncExitStack, directory: Path):
        """Create the real authority, voices, live services, users, home graph and app."""
        from starlette.responses import FileResponse, JSONResponse
        from starlette.routing import Route

        from src.dashboard.api import DashboardAPI
        from src.dashboard.authority import DashboardAuthority, DashboardRole, EvaluationModality, EvaluationSample
        from src.dashboard.live_erasure import build_live_services
        from src.memory.erasure_coordinator import ErasureCoordinator
        from src.memory.privacy_services import PrivacyServices
        from src.users.user_manager import PermissionLevel
        from src.users.voice_ledger import VoiceReviewer
        from src.users.voice_registry import build_voice_registry

        os.environ['KZA_DASHBOARD_TOKEN'] = 'synthetic-t046-root'
        self.started_at = time.time()
        expires = self.started_at + SESSION_TTL_S
        users = None
        authority = DashboardAuthority(directory / 'auth.db', session_ttl_s=SESSION_TTL_S,
                                       account_provider=lambda: users.active_accounts() if users is not None else None)
        stack.push_async_callback(authority.aclose)
        await authority.initialize()
        root = authority.resolve('synthetic-t046-root')
        privacy = PrivacyServices(str(directory / 'privacy.db'), clock=self.clock)
        stack.push_async_callback(privacy.aclose)
        await privacy.acall(privacy.initialize)
        owner = ErasureCoordinator(privacy)
        stack.push_async_callback(owner.aclose)
        bus = await build_live_services({}, privacy=privacy, owner=owner, shutdown=stack)
        voices = await build_voice_registry(privacy=privacy, owner=owner, shutdown=stack)
        users = await voices.load_accounts(str(directory / 'users.json'),
                                           retirement_proof=lambda identities: authority.users_retired(identities)
                                           and voices.ledger.people.user_links_cleared(identities))
        person, _ = await voices.acall(users.add_user, 'Persona sintética', PermissionLevel.ADULT)
        if person is None:
            raise RuntimeError('fixture_person_missing')
        operator = await authority.issue_credential(root, subject='synthetic-t046-operator', role=DashboardRole.OPERATOR,
                                                    expires_at=expires)
        home = await authority.issue_credential(root, subject='synthetic-t046-home', role=DashboardRole.HOME,
                                                expires_at=expires, user_id=person.user_id)
        # Same synthetic selection as the other NEXO harnesses (browser_nexo_home.py).
        selection = EvaluationSample('vf_' + 'a' * 64, frozenset({'b' * 64}),
                                     frozenset({EvaluationModality.AUDIO_PREVIEW}), expires)
        evaluator = await authority.issue_credential(root, subject='synthetic-t046-evaluator',
                                                     role=DashboardRole.EVALUATOR, expires_at=expires, samples=(selection,))
        self.tokens = {'operator': operator.token, 'home': home.token, 'evaluator': evaluator.token}
        self.reviewer = VoiceReviewer.from_dashboard(operator.principal)
        self.voices = voices
        encoder = SeededEncoder(self.plan)
        voices.register_encoder(encoder)
        await self.seed_voices(voices, encoder)
        ha, actions, rooms = home_graph()
        dashboard = DashboardAPI(authority=authority, event_bus=bus, voice_registry=voices, user_manager=users,
                                 ha_client=ha, home_actions=actions, home_rooms=rooms)
        fixture = self

        @dashboard.app.middleware('http')
        async def catalog_switch(request, call_next):
            if request.url.path == '/api/home/catalog' and not fixture.catalog_available:
                return JSONResponse({'detail': 'synthetic_catalog_unavailable'}, status_code=503)
            return await call_next(request)

        if self.harness_pages:
            async def harness_page(request):
                name = request.path_params['name']
                path = HARNESS_DIRECTORY / name
                if '/' in name or not name.endswith('.html') or not path.is_file():
                    return JSONResponse({'detail': 'not_found'}, status_code=404)
                return FileResponse(path, headers={'Cache-Control': 'no-store'})
            dashboard.app.router.routes.insert(0, Route('/__harness/{name}', harness_page))
        return dashboard

    async def serve(self) -> None:
        import uvicorn

        with tempfile.TemporaryDirectory(prefix='kza-t046-fixture-') as temporary:
            async with AsyncExitStack() as stack:
                dashboard = await self.build(stack, Path(temporary))
                listener = socket.socket()
                listener.bind(('127.0.0.1', 0))
                port = listener.getsockname()[1]
                self.origin = f'http://127.0.0.1:{port}'
                self.server = uvicorn.Server(uvicorn.Config(dashboard.app, log_level='warning'))
                serving = asyncio.create_task(self.server.serve(sockets=[listener]))
                async with asyncio.timeout(15):
                    while not self.server.started:
                        await asyncio.sleep(0.02)
                write_json_atomic(self.run_dir / 'fixture-positions.json', self.position_map())
                self.emit({'event': 'ready', 'origin': self.origin, 'digest': self.digest(), 'seed': self.seed,
                           'started_at': self.started_at, 'expires_at': self.started_at + TTL_AFTER_START_S,
                           'lifetime_s': FIXTURE_LIFETIME_S, 'tokens': self.tokens, 'pid': os.getpid(),
                           'position_map': str(self.run_dir / 'fixture-positions.json')})
                try:
                    await self.control()
                finally:
                    self.server.should_exit = True
                    await serving

    @staticmethod
    def emit(message: dict) -> None:
        sys.stdout.write(json.dumps(message, separators=(',', ':')) + '\n')
        sys.stdout.flush()

    async def control(self) -> None:
        """Read commands from stdin until 'quit' or EOF."""
        loop = asyncio.get_running_loop()
        reader = asyncio.StreamReader()
        await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)
        while True:
            line = await reader.readline()
            if not line:
                return
            words = line.decode().split()
            if not words:
                continue
            if words[0] == 'quit':
                write_json_atomic(self.run_dir / 'fixture-modules.json', repository_modules())
                self.emit({'reply': 'quit'})
                return
            if words[0] == 'health':
                self.emit({'reply': 'health', 'ok': bool(self.server and self.server.started)})
            elif words[0] == 'position_check' and len(words) == 2 and words[1].isdigit():
                result = await self.position_check(int(words[1]))
                self.emit({'reply': 'position_check', 'ordinal': result['ordinal'], 'ok': result['ok']})
            elif words[0] == 'catalog' and len(words) == 2 and words[1] in ('ok', 'unavailable'):
                self.catalog_available = words[1] == 'ok'
                self.emit({'reply': 'catalog', 'available': self.catalog_available, 'time': time.time()})
            else:
                self.emit({'reply': 'error', 'detail': 'unknown_command'})


def main(argv=None) -> int:
    import src  # noqa: F401  entry guard (N8): refuse before any store is opened

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--harness-pages', action='store_true')
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    arguments.run_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    asyncio.run(Fixture(arguments.run_dir, arguments.seed, harness_pages=arguments.harness_pages).serve())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
