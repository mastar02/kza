"""Manual diagnostics keep capture and typed action authority independent."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
from pathlib import Path
import runpy
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from src.audio.manual_command_action import ManualVoiceActionClient
from src.audio.native_capture_authority import ManualVoiceCommandAuthorityClient
from src.dashboard.authority import DashboardRole, EvaluationModality, EvaluationSample
from src.dashboard.home_api import register_home_routes
from src.home_assistant.control_policy import HomeControlPolicy
from src.home_assistant.home_actions import HomeActionService
from tests.integration.test_capture_browser_owners import browser_app
from tests.integration.test_capture_producer_privacy import until
from tests.integration.test_global_capture_control import actor, capture_phase, controlled
from tests.integration.test_home_control_policy import ObservedHA, write_selection
from tests.mocks.calibration_authority import serve_capture_app
from tests.mocks.manual_command import ManualCommandLeaves, NativeSegmentsActor, PipeInput
from tests.mocks.wakeword_capture import CaptureCall


SCRIPT = Path('scripts/test_voice_command.py')
CAPTURE_ENV = 'KZA_MANUAL_TEST_CAPTURE'
ACTION_ENV = 'KZA_MANUAL_TEST_ACTION'


@asynccontextmanager
async def manual_graph(tmp_path, monkeypatch, dashboard_auth):
    async with controlled(tmp_path, monkeypatch) as graph:
        async with browser_app(graph.control, tmp_path) as app:
            selection = tmp_path / 'manual-selection.json'
            write_selection(selection)
            ha = ObservedHA()
            ha.states['light.test_lamp']['state'] = 'on'
            actions = HomeActionService(ha, policy=HomeControlPolicy(selection_path=str(selection)))
            register_home_routes(app, catalog=None, actions=actions)
            action_state = SimpleNamespace(hold=False, entered=asyncio.Event(), release=asyncio.Event())
            original_send = ha.call_service_once

            async def held_send(*args, **kwargs):
                if action_state.hold:
                    action_state.entered.set()
                    await action_state.release.wait()
                return await original_send(*args, **kwargs)

            ha.call_service_once = held_send
            authority = app.state.dashboard_authority
            home = await authority.issue_credential(authority.resolve(dashboard_auth['Authorization']),
                subject='synthetic-manual-home', role=DashboardRole.HOME, expires_at=time.time() + 120)
            async with serve_capture_app(app) as url:
                with monkeypatch.context() as patch:
                    sdk = ManualCommandLeaves(patch)
                    pipe = PipeInput(patch, sdk)
                    clients, action_clients, calls = [], [], []
                    capture_tokens, action_tokens, action_attempts = [], [], []
                    credential = dashboard_auth['Authorization'].split(' ', 1)[1]
                    patch.setenv(CAPTURE_ENV, credential)
                    patch.setenv(ACTION_ENV, home.token)
                    patch.setenv('HOME_ASSISTANT_URL', 'http://127.0.0.1:9')
                    patch.setenv('HOME_ASSISTANT_TOKEN', 'synthetic-unused-raw-ha-token')
                    capture_init = ManualVoiceCommandAuthorityClient.__init__
                    action_init = ManualVoiceActionClient.__init__
                    action_execute = ManualVoiceActionClient.execute

                    def observe_capture(client, *args, **kwargs):
                        capture_init(client, *args, **kwargs)
                        clients.append(client)
                        capture_tokens.append(args[1] if len(args) > 1 else kwargs['token'])

                    def observe_action(client, *args, **kwargs):
                        action_init(client, *args, **kwargs)
                        action_clients.append(client)
                        action_tokens.append(args[1] if len(args) > 1 else kwargs['token'])

                    def observe_execute(client, *args, **kwargs):
                        action_attempts.append((args, kwargs.copy()))
                        return action_execute(client, *args, **kwargs)

                    patch.setattr(ManualVoiceCommandAuthorityClient, '__init__', observe_capture)
                    patch.setattr(ManualVoiceActionClient, '__init__', observe_action)
                    patch.setattr(ManualVoiceActionClient, 'execute', observe_execute)
                    namespace = runpy.run_path(str(SCRIPT))

                    def factory():
                        return ManualVoiceCommandAuthorityClient(url, credential, poll_interval_s=.02)

                    def call(operation):
                        pending = CaptureCall(operation)
                        calls.append(pending)
                        return pending

                    def cli(*, execute=False, enter=True, extra=()):
                        args = ['--capture-url', url, '--capture-token-env', CAPTURE_ENV,
                            '--duration', '.01', '--once']
                        args += ['--action-token-env', ACTION_ENV] if execute else ['--no-execute']
                        args += list(extra)
                        if enter:
                            pipe.enter()
                        return call(lambda: namespace['main'](args))

                    try:
                        yield SimpleNamespace(graph=graph, app=app, sdk=sdk, pipe=pipe,
                            clients=clients, action_clients=action_clients, call=call, cli=cli,
                            factory=factory, namespace=namespace, ha=ha, home=home,
                            action_state=action_state, capture_tokens=capture_tokens,
                            action_tokens=action_tokens, action_attempts=action_attempts)
                    finally:
                        action_state.release.set()
                        pipe.release()
                        sdk.release()
                        for pending in calls:
                            await asyncio.to_thread(pending.join)
                        sdk.release()
                        for client in action_clients + clients:
                            await asyncio.to_thread(client.close)
                        pipe.close()


async def reached(event):
    await until(event.is_set)


async def finished(call):
    await reached(call.done)
    await asyncio.to_thread(call.join)
    assert call.worker._handle.is_done()


def strict_local_retirement(sdk, *, models=True):
    for stream in sdk.streams:
        assert stream.stop_kwargs == [{'ignore_errors': False}]
        assert stream.close_kwargs == [{'ignore_errors': False}]
        assert stream.exited.is_set() and stream.worker._handle.is_done()
    for state in sdk.vector_states:
        assert state['close_calls'] == 1 and state['close_returned']
        assert state['worker']._handle.is_done()
    for row in sdk.segment_rows:
        assert row['close_calls'] == 1 and row['close_returned']
    if models:
        assert all(row['released'] for row in sdk.models)
        assert all(ref() is None for ref in sdk.model_refs + sdk.collection_refs + sdk.embedding_refs)
    assert sdk.raw_ha_attempts == sdk.private_env_attempts == []


@pytest.mark.parametrize('foreign', [False, True])
@pytest.mark.asyncio
async def test_direct_record_denies_missing_or_foreign_authority_before_sdk(tmp_path, monkeypatch, dashboard_auth, foreign):
    async with manual_graph(tmp_path, monkeypatch, dashboard_auth) as b:
        factory = (lambda: object()) if foreign else None
        call = b.call(lambda: b.namespace['record'](.01, 17, authority_factory=factory))
        await finished(call)
        assert isinstance(call.failure, RuntimeError)
        assert str(call.failure) == 'manual_command_authority_required'
        assert not b.sdk.streams and not b.sdk.models and not b.sdk.vector_states
        assert not b.action_clients and not b.ha.sends


@pytest.mark.parametrize('role', [DashboardRole.HOME, DashboardRole.EVALUATOR])
@pytest.mark.asyncio
async def test_main_requires_capture_admin_even_when_home_or_voice_claims_exist(tmp_path, monkeypatch, dashboard_auth, role):
    async with manual_graph(tmp_path, monkeypatch, dashboard_auth) as b:
        now = time.time()
        samples = (EvaluationSample('vf_' + 'a' * 64, frozenset({'b' * 64}),
            frozenset({EvaluationModality.AUDIO_PREVIEW}), now + 60),) if role is DashboardRole.EVALUATOR else ()
        authority = b.app.state.dashboard_authority
        grant = await authority.issue_credential(authority.resolve(dashboard_auth['Authorization']),
            subject='synthetic-manual-member', role=role, expires_at=now + 120, samples=samples)
        monkeypatch.setenv(CAPTURE_ENV, grant.token)
        call = b.cli()
        await finished(call)
        assert call.failure is not None or call.result != 0
        assert len(b.clients) == 1
        assert not b.sdk.events and not b.sdk.streams and not b.sdk.models
        assert not b.action_clients and not b.ha.sends
        assert b.graph.control.snapshot()['pending_native_clients'] == 0


@pytest.mark.parametrize('api', ['record', 'main'])
@pytest.mark.asyncio
async def test_muted_real_backend_denies_before_any_manual_sdk_constructor(tmp_path, monkeypatch, dashboard_auth, api):
    async with manual_graph(tmp_path, monkeypatch, dashboard_auth) as b:
        b.graph.control.request(True, b.graph.control.snapshot()['revision'], actor())
        await capture_phase(b.graph.control, 'muted')
        call = b.cli() if api == 'main' else b.call(lambda: b.namespace['record'](
            .01, 17, authority_factory=b.factory))
        await finished(call)
        assert call.failure is not None or (api == 'main' and call.result != 0)
        assert not b.sdk.events and not b.sdk.streams and not b.sdk.models
        assert len(b.clients) == 1 and not b.ha.sends
        assert b.graph.control.snapshot()['pending_native_clients'] == 0


@pytest.mark.asyncio
async def test_public_record_preserves_six_channels_rms_and_borrowed_mono_after_finish(tmp_path, monkeypatch, dashboard_auth):
    async with manual_graph(tmp_path, monkeypatch, dashboard_auth) as b:
        call = b.call(lambda: b.namespace['record'](.01, 17, authority_factory=b.factory))
        await finished(call)
        assert call.failure is None
        assert call.result.shape == (160,) and call.result.dtype == np.float32
        assert np.all(call.result == np.float32(.25))
        assert float(np.sqrt(np.mean(call.result ** 2))) == .25
        assert b.sdk.streams[0].kwargs['channels'] == 6
        assert b.sdk.streams[0].kwargs['samplerate'] == 16000
        assert sum(frames for _, frames in b.sdk.reads) == 160
        strict_local_retirement(b.sdk)
        assert b.graph.control.snapshot()['pending_native_clients'] == 0
        assert np.all(call.result == np.float32(.25))


@pytest.mark.parametrize('outcome', ['normal', 'empty', 'no-results', 'below-threshold'])
@pytest.mark.asyncio
async def test_once_no_execute_preserves_stt_search_math_without_action_acquisition(tmp_path, monkeypatch, dashboard_auth, caplog, outcome):
    async with manual_graph(tmp_path, monkeypatch, dashboard_auth) as b:
        monkeypatch.delenv(ACTION_ENV, raising=False)
        if outcome == 'empty':
            b.sdk.text = ''
        elif outcome == 'no-results':
            b.sdk.no_results = True
        elif outcome == 'below-threshold':
            b.sdk.distance = 1.0
        call = b.cli()
        await finished(call)
        assert call.failure is None
        assert call.result == (0 if outcome == 'normal' else 1)
        assert not b.action_clients and not b.action_attempts and not b.ha.sends
        whisper, embedder = b.sdk.models
        assert whisper['kwargs'] == {'device': 'cuda', 'device_index': 0, 'compute_type': 'float16'}
        assert embedder['args'] == ('BAAI/bge-m3',) and embedder['kwargs'] == {'device': 'cuda:0'}
        if outcome != 'empty':
            assert len(b.sdk.query_calls) == 1
            query = b.sdk.query_calls[0]
            assert query['where'] == {'service': 'turn_off'} and query['n_results'] == 3
            assert query['include'] == ['metadatas', 'distances', 'documents']
            assert np.allclose(query['query_embeddings'], [[.1, .2, .3]])
        if outcome == 'normal':
            assert 'sim=0.900' in caplog.text and 'MATCH: light.turn_off(light.test_lamp)' in caplog.text
        assert all(reader is call.worker for reader in b.pipe.readers)
        assert not b.pipe.stream.closed
        strict_local_retirement(b.sdk)
        assert all(ref() is None for ref in b.sdk.audio_refs)
        assert b.graph.control.snapshot()['pending_native_clients'] == 0


@pytest.mark.asyncio
async def test_main_uses_home_action_grant_distinct_from_original_capture_actor(tmp_path, monkeypatch, dashboard_auth):
    async with manual_graph(tmp_path, monkeypatch, dashboard_auth) as b:
        call = b.cli(execute=True)
        await finished(call)
        assert call.failure is None and call.result == 0
        assert b.action_tokens == [b.home.token]
        assert b.capture_tokens == [dashboard_auth['Authorization'].split(' ', 1)[1]]
        assert b.action_tokens != b.capture_tokens
        assert len(b.action_attempts) == 1
        _args, kwargs = b.action_attempts[0]
        assert isinstance(kwargs['request_id'], str) and kwargs['request_id']
        assert len(b.ha.sends) == 1 and b.ha.states['light.test_lamp']['state'] == 'off'
        assert b.ha.sends[0][:3] == ('light', 'turn_off', 'light.test_lamp')
        strict_local_retirement(b.sdk)
        assert b.graph.control.snapshot()['pending_native_clients'] == 0


@pytest.mark.asyncio
async def test_missing_action_credential_never_falls_back_to_capture_or_ha_env(tmp_path, monkeypatch, dashboard_auth):
    async with manual_graph(tmp_path, monkeypatch, dashboard_auth) as b:
        monkeypatch.delenv(ACTION_ENV, raising=False)
        call = b.cli(execute=True)
        await finished(call)
        assert call.failure is not None or call.result != 0
        assert b.action_tokens == [] and not b.action_attempts and not b.ha.sends
        assert b.sdk.raw_ha_attempts == b.sdk.private_env_attempts == []
        assert b.graph.control.snapshot()['pending_native_clients'] == 0


@pytest.mark.asyncio
async def test_unobserved_action_is_nonzero_without_repeating_post_or_poisoning_local_retirement(tmp_path, monkeypatch, dashboard_auth):
    async with manual_graph(tmp_path, monkeypatch, dashboard_auth) as b:
        b.ha.observe_effect = False
        call = b.cli(execute=True)
        await finished(call)
        assert call.failure is None and call.result != 0
        assert len(b.action_attempts) == 1 and len(b.ha.sends) == 1
        assert b.ha.states['light.test_lamp']['state'] == 'on'
        strict_local_retirement(b.sdk)
        assert b.graph.control.snapshot()['pending_native_clients'] == 0


@pytest.mark.parametrize('boundary', ['start', 'read', 'transcribe', 'segment', 'encode',
    'query', 'segment-close', 'chroma-close', 'model-release', 'report'])
@pytest.mark.asyncio
async def test_global_ack_waits_for_actual_sdk_iterator_model_and_report_return(tmp_path, monkeypatch, dashboard_auth, boundary):
    async with manual_graph(tmp_path, monkeypatch, dashboard_auth) as b:
        b.sdk.hold.add(boundary)
        original_info = logging.Logger.info

        def info_and_hold(logger, message, *args, **kwargs):
            result = original_info(logger, message, *args, **kwargs)
            if logger.name == 'voice_test' and str(message).startswith('STT ['):
                b.sdk.checkpoint('report')
            return result

        monkeypatch.setattr(logging.Logger, 'info', info_and_hold)
        call = b.cli()
        try:
            await reached(b.sdk.event(boundary))
            assert b.graph.control.snapshot()['pending_native_clients'] == 1
            b.graph.control.request(True, b.graph.control.snapshot()['revision'], actor())
            await until(lambda: bool(b.clients) and not b.clients[0].current())
            assert call.worker.is_alive() and not call.done.is_set()
            state = b.graph.control.snapshot()
            assert state['pending_native_clients'] == 1 and not state['total_mute_confirmed']
            if boundary == 'chroma-close':
                assert b.sdk.vector_states[0]['worker']._handle.is_done()
                assert not b.sdk.vector_states[0]['close_returned']
        finally:
            b.sdk.release_event(boundary).set()
        await finished(call)
        strict_local_retirement(b.sdk)
        assert all(ref() is None for ref in b.sdk.audio_refs)
        state = await capture_phase(b.graph.control, 'muted')
        assert state['pending_native_clients'] == 0 and state['total_mute_confirmed']
        assert not b.action_clients and not b.ha.sends


@pytest.mark.asyncio
async def test_action_http_return_is_part_of_capture_retirement_barrier(tmp_path, monkeypatch, dashboard_auth):
    async with manual_graph(tmp_path, monkeypatch, dashboard_auth) as b:
        b.action_state.hold = True
        call = b.cli(execute=True)
        try:
            await reached(b.action_state.entered)
            b.graph.control.request(True, b.graph.control.snapshot()['revision'], actor())
            await until(lambda: bool(b.clients) and not b.clients[0].current())
            assert call.worker.is_alive() and not call.done.is_set()
            assert not b.graph.control.snapshot()['total_mute_confirmed']
            assert b.graph.control.snapshot()['pending_native_clients'] == 1
        finally:
            b.action_state.release.set()
        await finished(call)
        assert len(b.action_attempts) == 1 and len(b.ha.sends) == 1
        strict_local_retirement(b.sdk)
        state = await capture_phase(b.graph.control, 'muted')
        assert state['pending_native_clients'] == 0 and state['total_mute_confirmed']


@pytest.mark.asyncio
async def test_muted_prompt_uses_borrowed_pipe_select_without_detached_reader(tmp_path, monkeypatch, dashboard_auth):
    async with manual_graph(tmp_path, monkeypatch, dashboard_auth) as b:
        call = b.cli(enter=False)
        await reached(b.sdk.event('prompt-select'))
        assert b.pipe.selectors and all(thread is call.worker for thread in b.pipe.selectors)
        assert not b.pipe.readers and call.worker.is_alive()
        b.graph.control.request(True, b.graph.control.snapshot()['revision'], actor())
        await finished(call)
        assert not b.pipe.stream.closed
        assert not b.sdk.streams
        strict_local_retirement(b.sdk)
        state = await capture_phase(b.graph.control, 'muted')
        assert state['pending_native_clients'] == 0 and state['total_mute_confirmed']


@pytest.mark.asyncio
async def test_unreturned_vector_constructor_remains_pending_without_ack(tmp_path, monkeypatch, dashboard_auth):
    async with manual_graph(tmp_path, monkeypatch, dashboard_auth) as b:
        b.sdk.faults.add('chroma-construct')
        call = b.cli()
        await finished(call)
        assert call.failure is not None or call.result != 0
        assert b.sdk.vector_states[0]['worker'].is_alive()
        assert not b.sdk.vector_states[0]['worker']._handle.is_done()
        assert b.sdk.vector_states[0]['close_calls'] == 0
        assert b.graph.control.snapshot()['pending_native_clients'] == 1
        await asyncio.to_thread(b.clients[0].close)
        b.graph.control.request(True, b.graph.control.snapshot()['revision'], actor())
        await capture_phase(b.graph.control, 'muting')
        assert not b.graph.control.snapshot()['total_mute_confirmed']


@pytest.mark.asyncio
async def test_sdk_failure_drops_pcm_and_models_while_original_observer_retains_failure(tmp_path, monkeypatch, dashboard_auth):
    async with manual_graph(tmp_path, monkeypatch, dashboard_auth) as b:
        b.sdk.faults.add('transcribe')
        call = b.cli()
        await finished(call)
        assert call.failure is not None or call.result != 0
        strict_local_retirement(b.sdk)
        assert b.sdk.audio_refs and all(ref() is None for ref in b.sdk.audio_refs)
        assert b.graph.control.snapshot()['pending_native_clients'] == 0


@pytest.mark.asyncio
async def test_original_operator_revocation_retires_an_admitted_read_without_new_actor(tmp_path, monkeypatch, dashboard_auth):
    async with manual_graph(tmp_path, monkeypatch, dashboard_auth) as b:
        b.sdk.hold.add('read')
        call = b.call(lambda: b.namespace['record'](.01, 17, authority_factory=b.factory))
        try:
            await reached(b.sdk.event('read'))
            monkeypatch.setenv('KZA_DASHBOARD_TOKEN', 'different-manual-operator')
            await until(lambda: not b.clients[0].current())
            assert call.worker.is_alive() and not call.done.is_set()
            assert b.graph.control.snapshot()['pending_native_clients'] == 1
        finally:
            b.sdk.release_event('read').set()
        await finished(call)
        assert call.failure is not None
        strict_local_retirement(b.sdk)
        assert len(b.clients) == 1 and not b.clients[0].current()
        assert b.graph.control.snapshot()['pending_native_clients'] == 0


@pytest.mark.asyncio
async def test_lexical_helper_and_unbound_action_do_not_infer_authority(tmp_path, monkeypatch, dashboard_auth):
    async with manual_graph(tmp_path, monkeypatch, dashboard_auth) as b:
        assert b.namespace['classify_intent']('prende y apaga la luz') == 'turn_off'
        assert b.namespace['classify_intent']('prende la luz') == 'turn_on'
        with pytest.raises(RuntimeError):
            b.namespace['call_ha_service']('light', 'turn_off', 'light.test_lamp')
        assert not b.sdk.events and not b.sdk.models and not b.sdk.streams
        assert not b.action_clients and not b.ha.sends
        assert b.sdk.raw_ha_attempts == b.sdk.private_env_attempts == []


@pytest.mark.parametrize('boundary', ['source', 'iterator'])
@pytest.mark.asyncio
async def test_iterator_slot_reservation_failure_cannot_ack_a_returned_native_actor(tmp_path, monkeypatch, dashboard_auth, boundary):
    from src.audio import manual_command_capture as capture_module

    async with manual_graph(tmp_path, monkeypatch, dashboard_auth) as b:
        actors, returned_sources = [], []
        calls = {'slots': 0, 'faults': 0, 'transcribe': 0, 'iter': 0}
        target = 1 if boundary == 'source' else 2
        original_slot = capture_module._Iterator

        class FailingSlot(original_slot):
            def __init__(self, *args, **kwargs):
                calls['slots'] += 1
                if calls['slots'] == target:
                    calls['faults'] += 1
                    raise RuntimeError('synthetic_iterator_slot_reservation_failed')
                super().__init__(*args, **kwargs)

        class Source:
            def __init__(self, audio, model):
                self.audio, self.model = audio, model
                self.close_calls = 0

            def __iter__(self):
                calls['iter'] += 1
                sdk_actor = NativeSegmentsActor(self.audio, self.model, 'iterator')
                actors.append(sdk_actor)
                return sdk_actor

            def close(self):
                self.close_calls += 1
                self.audio = self.model = None

        def transcribe(model, audio, **kwargs):
            assert kwargs == {'language': 'es', 'vad_filter': True, 'beam_size': 1}
            assert audio.shape == (160,) and np.all(audio == np.float32(.25))
            calls['transcribe'] += 1
            if boundary == 'source':
                source = NativeSegmentsActor(audio, model, 'source')
                actors.append(source)
            else:
                source = Source(audio, model)
            returned_sources.append(source)
            return source, SimpleNamespace(language='es')

        monkeypatch.setattr(capture_module, '_Iterator', FailingSlot)
        monkeypatch.setattr(sys.modules['faster_whisper'].WhisperModel, 'transcribe', transcribe)
        try:
            call = b.cli()
            await finished(call)
            assert call.failure is not None or call.result != 0
            assert calls['faults'] == 1 and calls['slots'] >= target
            assert len(b.clients) == 1 and len(b.sdk.models) == 2
            assert sum(frames for _, frames in b.sdk.reads) == 160
            assert b.sdk.streams[0].worker._handle.is_done()
            live = [sdk_actor for sdk_actor in actors if not sdk_actor.worker._handle.is_done()]
            before = b.graph.control.snapshot()
            b.graph.control.request(True, before['revision'], actor())
            if before['pending_native_clients'] == 0:
                state = await capture_phase(b.graph.control, 'muted')
            else:
                await until(lambda: b.graph.control.snapshot()['phase'] != 'running')
                state = b.graph.control.snapshot()
            print({'scenario': 'returned_iterator_registration_fault', 'boundary': boundary,
                'slot_calls': calls['slots'], 'transcribe_calls': calls['transcribe'],
                'iter_calls': calls['iter'], 'returned_actors': len(actors),
                'live_native_handles': len(live), 'pending_native_clients': state['pending_native_clients'],
                'total_mute_confirmed': state['total_mute_confirmed']})
            assert not live or not state['total_mute_confirmed'], 'Live returned SDK actor permitted total mute ACK'
            assert not live or state['pending_native_clients'] == 1
            assert all(source.close_calls > 0 for source in returned_sources)
        finally:
            for returned_actor in actors:
                returned_actor.close()
                assert returned_actor.worker._handle.is_done()
            print({'scenario': 'returned_actor_fixture_cleanup', 'boundary': boundary,
                'native_handles_done': [sdk_actor.worker._handle.is_done() for sdk_actor in actors],
                'close_calls': [sdk_actor.close_calls for sdk_actor in actors],
                'cleanup_only_in_finally': True})


def failure_leaves(error):
    if isinstance(error, BaseExceptionGroup):
        return [leaf for member in error.exceptions for leaf in failure_leaves(member)]
    return [error]


@pytest.mark.parametrize('control_type', [KeyboardInterrupt, SystemExit])
@pytest.mark.asyncio
async def test_record_retains_read_control_and_close_failure_with_unconfirmed_retirement(tmp_path, monkeypatch, dashboard_auth, control_type):
    async with manual_graph(tmp_path, monkeypatch, dashboard_auth) as b:
        stream_type = sys.modules['sounddevice'].InputStream
        original_read = stream_type.read

        def read_then_control(stream, frames):
            _audio, _overflowed = original_read(stream, frames)
            if control_type is SystemExit:
                raise SystemExit(7)
            raise KeyboardInterrupt()

        monkeypatch.setattr(stream_type, 'read', read_then_control)
        b.sdk.faults.add('close')
        call = b.call(lambda: b.namespace['record'](.01, 17, authority_factory=b.factory))
        await finished(call)
        assert call.failure is not None
        stream = b.sdk.streams[0]
        assert stream.worker._handle.is_done() and stream.exited.is_set()
        assert stream.stop_kwargs == [{'ignore_errors': False}]
        assert stream.close_kwargs == [{'ignore_errors': False}]
        b.graph.control.request(True, b.graph.control.snapshot()['revision'], actor())
        await capture_phase(b.graph.control, 'muting')
        state = b.graph.control.snapshot()
        assert state['pending_native_clients'] == 1 and not state['total_mute_confirmed']
        leaves = failure_leaves(call.failure)
        print({'scenario': 'read_control_and_close_failure', 'control': control_type.__name__,
            'leaf_types': [type(error).__name__ for error in leaves],
            'native_handle_done': stream.worker._handle.is_done(),
            'pending_native_clients': state['pending_native_clients'],
            'total_mute_confirmed': state['total_mute_confirmed']})
        assert any(isinstance(error, RuntimeError) for error in leaves)
        assert any(isinstance(error, control_type) for error in leaves), 'Read control was masked by SDK close failure'
        if control_type is SystemExit:
            assert any(isinstance(error, SystemExit) and error.code == 7 for error in leaves)
        assert all(ref() is None for ref in b.sdk.audio_refs)
