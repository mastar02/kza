"""
Benchmark de Latencia del Pipeline KZA

Mide la latencia real de cada componente del pipeline:
- STT (Speech-to-Text)
- Speaker ID
- Emotion Detection
- Router (clasificación)
- LLM (razonamiento)
- TTS (Text-to-Speech)
- Home Assistant

Uso:
    python tools/benchmark_latency.py [--iterations N] [--audio FILE]
"""
from __future__ import annotations
import argparse
import asyncio
import logging
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import ExitStack, contextmanager
from contextvars import Context
from functools import wraps
import threading

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.core.resource_retirement import ResourceOwner, ResourceFailure, raise_errors  # noqa: E402

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Resultado de un benchmark individual"""
    component: str
    latency_ms: float
    success: bool
    details: str = ""


@dataclass
class BenchmarkSummary:
    """Resumen de benchmarks"""
    component: str
    iterations: int
    min_ms: float
    max_ms: float
    avg_ms: float
    median_ms: float
    p95_ms: float
    std_ms: float
    success_rate: float


_LATENCY_OWNERS: set[object] = set()
_LATENCY_OWNERS_LOCK = threading.RLock()

@dataclass(eq=False)
class _LatencyOperation:
    caller: asyncio.Task | None
    stack: ExitStack = field(default_factory=ExitStack)
    owner_ids: set[int] = field(default_factory=set)
    active: bool = True
    finished: bool = False

@dataclass(eq=False)
class _LatencyHA:
    resource: object | None = None
    confirmed: bool = False

@dataclass(eq=False)
class _LatencyCloseAdmission:
    caller: asyncio.Task | None
    finished: bool = False

def _capture_failures(stage: str, error: BaseException) -> list[ResourceFailure]:
    """Copy safe metadata per leaf; never retain the source group or frames."""
    failures = []
    pending = [error]
    current = None
    try:
        while pending:
            current = pending.pop()
            if isinstance(current, BaseExceptionGroup):
                pending.extend(reversed(current.exceptions))
            else:
                failures.append(ResourceFailure.capture(stage, current))
        return failures
    finally:
        current = error = None
        pending.clear()

def _raise_failures(failures) -> None:
    """Preserve every known control leaf, without source group labels/shape."""
    errors = [failure.exception() for failure in failures]
    if len(errors) > 1 and any((not isinstance(error, Exception) for error in errors)):
        raise BaseExceptionGroup('latency_failures', errors)
    raise_errors(errors, message='latency_failures')

def _failure_kind(error: BaseException) -> str:
    return ResourceFailure.capture('latency_sample_failed', error).kind

def _component(slot: str | None=None):

    def decorate(function):

        @wraps(function)
        async def invoke(self, *args, **kwargs):
            result, failures = (None, [])
            try:
                with self._operation():
                    try:
                        if slot is not None:
                            self._use_sync(slot)
                        result = await function(self, *args, **kwargs)
                    except BaseException as error:
                        failures.extend(_capture_failures('latency_component_failed', error))
                    finally:
                        args = kwargs = None
            except BaseException as error:
                failures.extend(_capture_failures('latency_operation_failed', error))
            finally:
                args = kwargs = None
            if failures:
                result = None
                _raise_failures(failures)
            return result
        return invoke
    return decorate

class LatencyBenchmark:
    """
    Benchmark de latencia del pipeline KZA.

    Mide cada componente individualmente y el pipeline completo.
    """

    def __init__(self, config_path: str='config/settings.yaml'):
        self.config_path = config_path
        self.results: dict[str, list[BenchmarkResult]] = {}
        self._stt = None
        self._tts = None
        self._router = None
        self._llm = None
        self._speaker_id = None
        self._emotion = None
        self._ha = None
        self._ownership = threading.RLock()
        self._operations: list[_LatencyOperation] = []
        self._sync_owners: list[ResourceOwner] = []
        self._owned_fields: dict[str, object] = {}
        self._owned_owner: dict[str, ResourceOwner] = {}
        self._owned_ha = None
        self._ha_records: list[_LatencyHA] = []
        self._closed = False
        self._closing = None
        self._close_coroutine = None
        self._close_admission = None
        self._close_failures: list[ResourceFailure] = []
        self._close_waiters: set[asyncio.Task] = set()
        self._retirement_confirmed = False
        with _LATENCY_OWNERS_LOCK:
            _LATENCY_OWNERS.add(self)

    def _generate_test_audio(self, duration_s: float=1.0, sample_rate: int=16000) -> np.ndarray:
        """Generar audio de prueba (ruido simulando habla)"""
        samples = int(duration_s * sample_rate)
        noise = np.random.randn(samples).astype(np.float32) * 0.3
        envelope = np.concatenate([np.linspace(0, 1, samples // 4), np.ones(samples // 2), np.linspace(1, 0, samples // 4)])
        return noise * envelope[:len(noise)]

    @_component('_stt')
    async def benchmark_stt(self, audio: np.ndarray=None, iterations: int=10) -> BenchmarkSummary:
        """Benchmark de Speech-to-Text"""
        try:
            print('\n📝 Benchmarking STT...')
            if self._stt is None:
                try:
                    from src.stt.whisper_fast import FastWhisperSTT
                    self._ensure_sync('_stt', lambda: FastWhisperSTT(model='distil-whisper/distil-small.en', device='cuda:0', language='es'), lambda resource: resource.load())
                except Exception as e:
                    print(f'  ⚠️ STT no disponible: {_failure_kind(e)}')
                    return None
            if audio is None:
                audio = self._generate_test_audio(1.0)
            results = []
            for i in range(iterations):
                start = time.perf_counter()
                try:
                    text, _ = self._stt.transcribe(audio, 16000)
                    latency = (time.perf_counter() - start) * 1000
                    results.append(BenchmarkResult('stt', latency, True, text[:30]))
                except Exception as e:
                    results.append(BenchmarkResult('stt', 0, False, _failure_kind(e)))
            return self._summarize('STT', results)
        finally:
            audio = text = _ = results = FastWhisperSTT = None

    @_component('_stt')
    async def benchmark_stt_with_vad(self, audio: np.ndarray=None, iterations: int=10) -> BenchmarkSummary:
        """Benchmark de STT con VAD temprano"""
        try:
            print('\n📝 Benchmarking STT + VAD...')
            if self._stt is None:
                return None
            if audio is None:
                speech = self._generate_test_audio(0.8)
                silence = np.zeros(int(0.4 * 16000), dtype=np.float32)
                audio = np.concatenate([speech, silence])
            results = []
            for i in range(iterations):
                start = time.perf_counter()
                try:
                    _text, _, early = self._stt.transcribe_with_early_vad(audio, 16000)
                    latency = (time.perf_counter() - start) * 1000
                    results.append(BenchmarkResult('stt_vad', latency, True, f'early={early}'))
                except Exception as e:
                    results.append(BenchmarkResult('stt_vad', 0, False, _failure_kind(e)))
            return self._summarize('STT+VAD', results)
        finally:
            audio = speech = silence = _text = _ = results = None

    @_component('_tts')
    async def benchmark_tts(self, text: str='Hola, esta es una prueba de síntesis', iterations: int=10) -> BenchmarkSummary:
        """Benchmark de Text-to-Speech"""
        try:
            print('\n🔊 Benchmarking TTS...')
            if self._tts is None:
                try:
                    from src.tts.piper_tts import PiperTTS
                    self._ensure_sync('_tts', lambda: PiperTTS(), lambda resource: resource.load(warmup=True))
                except Exception as e:
                    print(f'  ⚠️ TTS no disponible: {_failure_kind(e)}')
                    return None
            results = []
            for i in range(iterations):
                start = time.perf_counter()
                try:
                    audio, _ = self._tts.synthesize(text)
                    latency = (time.perf_counter() - start) * 1000
                    results.append(BenchmarkResult('tts', latency, True, f'{len(audio)} samples'))
                except Exception as e:
                    results.append(BenchmarkResult('tts', 0, False, _failure_kind(e)))
            return self._summarize('TTS', results)
        finally:
            text = audio = _ = results = PiperTTS = None

    @_component('_router')
    async def benchmark_router(self, text: str='prende la luz del living', iterations: int=10) -> BenchmarkSummary:
        """Benchmark del Router/Clasificador"""
        try:
            print('\n🧠 Benchmarking Router...')
            if self._router is None:
                try:
                    from src.llm.reasoner import FastRouter
                    self._ensure_sync('_router', lambda: FastRouter(model='Qwen/Qwen2.5-7B-Instruct', device='cuda:2', enable_prefix_caching=True), lambda resource: resource.load())
                except Exception as e:
                    print(f'  ⚠️ Router no disponible: {_failure_kind(e)}')
                    return None
            results = []
            for i in range(iterations):
                start = time.perf_counter()
                try:
                    needs_deep, _response = self._router.classify_and_respond(text)
                    latency = (time.perf_counter() - start) * 1000
                    results.append(BenchmarkResult('router', latency, True, f'deep={needs_deep}'))
                except Exception as e:
                    results.append(BenchmarkResult('router', 0, False, _failure_kind(e)))
            return self._summarize('Router', results)
        finally:
            text = _response = results = FastRouter = None

    @_component('_emotion')
    async def benchmark_emotion(self, audio: np.ndarray=None, iterations: int=10) -> BenchmarkSummary:
        """Benchmark de detección de emociones"""
        try:
            print('\n😊 Benchmarking Emotion Detection...')
            if self._emotion is None:
                try:
                    from src.users.emotion_detector import EmotionDetector
                    self._ensure_sync('_emotion', lambda: EmotionDetector(model_name='audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim', device='cuda:0'), lambda resource: resource.load())
                except Exception as e:
                    print(f'  ⚠️ Emotion no disponible: {_failure_kind(e)}')
                    return None
            if audio is None:
                audio = self._generate_test_audio(1.0)
            results = []
            for i in range(iterations):
                start = time.perf_counter()
                try:
                    result = self._emotion.detect(audio)
                    latency = (time.perf_counter() - start) * 1000
                    results.append(BenchmarkResult('emotion', latency, True, result.emotion))
                except Exception as e:
                    results.append(BenchmarkResult('emotion', 0, False, _failure_kind(e)))
            return self._summarize('Emotion', results)
        finally:
            audio = result = results = EmotionDetector = None

    @_component('_emotion')
    async def benchmark_emotion_batch(self, iterations: int=5) -> BenchmarkSummary:
        """Benchmark de detección de emociones en batch"""
        try:
            print('\n😊 Benchmarking Emotion Batch...')
            if self._emotion is None:
                return None
            audios = [self._generate_test_audio(1.0) for _ in range(3)]
            results = []
            for i in range(iterations):
                start = time.perf_counter()
                try:
                    batch_results = self._emotion.batch_detect(audios)
                    latency = (time.perf_counter() - start) * 1000
                    results.append(BenchmarkResult('emotion_batch', latency, True, f'{len(batch_results)} results'))
                except Exception as e:
                    results.append(BenchmarkResult('emotion_batch', 0, False, _failure_kind(e)))
            return self._summarize('Emotion Batch (3)', results)
        finally:
            audios = batch_results = results = None

    @_component()
    async def benchmark_ha_call(self, iterations: int=10) -> BenchmarkSummary:
        """Measure one actual retained HA REST snapshot per iteration."""
        try:
            print('\n🏠 Benchmarking Home Assistant...')
            if self._ha is None:
                try:
                    from src.home_assistant.ha_client import HomeAssistantClient
                    import yaml
                    with open(self.config_path) as f:
                        config = yaml.safe_load(f)
                    ha_config = config.get('home_assistant', {})
                    self._ensure_ha(lambda: HomeAssistantClient(url=ha_config.get('url'), token=ha_config.get('token')))
                except Exception as e:
                    print(f'  ⚠️ HA no disponible: {_failure_kind(e)}')
                    return None
            results = []
            for i in range(iterations):
                start = time.perf_counter()
                try:
                    snapshot = await self._ha.get_state_snapshot()
                    latency = (time.perf_counter() - start) * 1000
                    if snapshot.available is True:
                        states = snapshot.states
                        results.append(BenchmarkResult('ha', latency, True, f'{len(states)} states'))
                    else:
                        results.append(BenchmarkResult('ha', 0, False, 'source_unavailable'))
                except Exception as e:
                    results.append(BenchmarkResult('ha', 0, False, _failure_kind(e)))
            return self._summarize('Home Assistant', results)
        finally:
            states = snapshot = config = ha_config = f = HomeAssistantClient = yaml = results = None

    def _summarize(self, name: str, results: list[BenchmarkResult]) -> BenchmarkSummary:
        """Generar resumen de resultados"""
        successful = [r for r in results if r.success]
        latencies = [r.latency_ms for r in successful]
        if not latencies:
            return BenchmarkSummary(component=name, iterations=len(results), min_ms=0, max_ms=0, avg_ms=0, median_ms=0, p95_ms=0, std_ms=0, success_rate=0)
        return BenchmarkSummary(component=name, iterations=len(results), min_ms=min(latencies), max_ms=max(latencies), avg_ms=statistics.mean(latencies), median_ms=statistics.median(latencies), p95_ms=sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else latencies[0], std_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0, success_rate=len(successful) / len(results) * 100)

    async def _run_full_body(self, iterations: int=10) -> dict:
        """Ejecutar benchmark completo"""
        try:
            print('=' * 60)
            print('🚀 KZA Latency Benchmark')
            print('=' * 60)
            print(f'Iterations per component: {iterations}')
            summaries = {}
            summary = await self.benchmark_stt(iterations=iterations)
            if summary:
                summaries['stt'] = summary
            summary = await self.benchmark_stt_with_vad(iterations=iterations)
            if summary:
                summaries['stt_vad'] = summary
            summary = await self.benchmark_tts(iterations=iterations)
            if summary:
                summaries['tts'] = summary
            summary = await self.benchmark_router(iterations=iterations)
            if summary:
                summaries['router'] = summary
            summary = await self.benchmark_emotion(iterations=iterations)
            if summary:
                summaries['emotion'] = summary
            summary = await self.benchmark_emotion_batch(iterations=iterations // 2)
            if summary:
                summaries['emotion_batch'] = summary
            summary = await self.benchmark_ha_call(iterations=iterations)
            if summary:
                summaries['ha'] = summary
            return summaries
        finally:
            summary = summaries = None

    def _print_summary(self, summaries: dict):
        """Imprimir resumen formateado"""
        print('\n' + '=' * 60)
        print('📊 RESUMEN DE LATENCIAS')
        print('=' * 60)
        print(f'{'Componente':<20} {'Min':>8} {'Avg':>8} {'Med':>8} {'P95':>8} {'Max':>8} {'Success':>8}')
        print('-' * 60)
        total_avg = 0
        for name, s in summaries.items():
            print(f'{s.component:<20} {s.min_ms:>7.1f}ms {s.avg_ms:>7.1f}ms {s.median_ms:>7.1f}ms {s.p95_ms:>7.1f}ms {s.max_ms:>7.1f}ms {s.success_rate:>7.0f}%')
            total_avg += s.avg_ms
        print('-' * 60)
        pipeline_components = ['stt', 'router', 'tts']
        pipeline_total = sum((summaries[c].avg_ms for c in pipeline_components if c in summaries))
        print(f'\n{'Pipeline estimado:':<20} {pipeline_total:>7.1f}ms (STT + Router + TTS)')
        target = 300
        if pipeline_total <= target:
            print(f'✅ Dentro del objetivo de {target}ms')
        else:
            print(f'⚠️ Excede objetivo de {target}ms por {pipeline_total - target:.1f}ms')
        if 'stt' in summaries and 'stt_vad' in summaries:
            vad_saving = summaries['stt'].avg_ms - summaries['stt_vad'].avg_ms
            if vad_saving > 0:
                print(f'\n💡 VAD early detection ahorra ~{vad_saving:.1f}ms en promedio')

    @contextmanager
    def _operation(self):
        token = _LatencyOperation(asyncio.current_task())
        try:
            with self._ownership:
                nested = any((item.active and item.caller is token.caller for item in self._operations))
                if self._closed and (not nested):
                    raise RuntimeError('latency_closed')
                self._operations.append(token)
            yield token
        finally:
            token.active = False
            try:
                token.stack.close()
            finally:
                token.owner_ids.clear()
                token.caller = None
                token.finished = True
                with self._ownership:
                    if token in self._operations:
                        self._operations.remove(token)

    def _current_operation(self):
        caller = asyncio.current_task()
        with self._ownership:
            for token in reversed(self._operations):
                if token.active and token.caller is caller:
                    return token
        raise RuntimeError('latency_operation_required')

    def _enter_owner(self, owner: ResourceOwner) -> None:
        token = self._current_operation()
        if id(owner) not in token.owner_ids:
            token.stack.enter_context(owner.operation())
            token.owner_ids.add(id(owner))

    def _use_sync(self, slot: str) -> None:
        resource = getattr(self, slot)
        owner = self._owned_owner.get(slot)
        if owner is not None and self._owned_fields.get(slot) is resource:
            self._enter_owner(owner)

    def _ensure_sync(self, slot: str, factory, load):
        resource = owner = guard = None
        try:
            resource = getattr(self, slot)
            if resource is not None:
                self._use_sync(slot)
                return resource
            owner = ResourceOwner(owns_resource=True, error_prefix='latency' + slot, thread_name='latency' + slot + '-close')
            self._sync_owners.append(owner)
            self._enter_owner(owner)
            resource = factory()
            guard = getattr(resource, '_adapter_require_close_caller', None)
            if guard is None:
                guard = getattr(resource, '_router_require_close_caller', None)
            owner.adopt(resource, caller_guard=guard)
            self._owned_fields[slot] = resource
            self._owned_owner[slot] = owner
            setattr(self, slot, resource)
            load(resource)
            return resource
        finally:
            resource = owner = guard = factory = load = None

    def _ensure_ha(self, factory):
        record = resource = None
        try:
            if self._ha is None:
                record = _LatencyHA()
                self._ha_records.append(record)
                resource = factory()
                record.resource = resource
                self._owned_ha = resource
                self._owned_fields['_ha'] = resource
                self._ha = resource
            return self._ha
        finally:
            record = resource = factory = None

    def _require_close_caller(self) -> None:
        try:
            caller = asyncio.current_task()
        except RuntimeError:
            caller = None
        with self._ownership:
            if caller is not None and any((item.active and item.caller is caller for item in self._operations)):
                raise RuntimeError('latency_close_from_operation')
            if caller is not None and (caller is self._closing or caller in self._close_waiters):
                raise RuntimeError('latency_close_from_owner')
            admission = self._close_admission
            if caller is not None and admission is not None and (not admission.finished) and (admission.caller is caller):
                raise RuntimeError('latency_close_from_admission')
            owners = tuple(self._sync_owners)
            records = tuple(self._ha_records)
        if caller is not None and any((getattr(record.resource, '_close_task', None) is caller for record in records)):
            raise RuntimeError('latency_close_from_owner')
        for owner in owners:
            owner.require_close_caller()

    def begin_close(self) -> asyncio.Task:
        """Seal once and retain the actual retirement coordinator."""
        self._require_close_caller()
        loop = asyncio.get_running_loop()
        candidate = _LatencyCloseAdmission(asyncio.current_task())
        try:
            with self._ownership:
                if self._close_admission is None:
                    self._close_admission = candidate
                    self._closed = True
            if self._close_admission is candidate:
                self._close_coroutine = self._close_owned()
                task = loop.create_task(self._close_coroutine, context=Context())
                self._closing = task
                task.add_done_callback(lambda done: None if done.cancelled() else done.exception(), context=Context())
        except BaseException as error:
            if self._close_admission is not candidate:
                raise
            self._close_failures.extend(_capture_failures('latency_close_admission_failed', error))
        finally:
            if self._close_admission is candidate:
                candidate.finished = True
                candidate.caller = None
        if self._closing is None:
            raise RuntimeError('latency_close_unconfirmed')
        return self._closing

    async def _wait_sync_owner(self, owner: ResourceOwner) -> None:
        try:
            await asyncio.to_thread(owner.wait_closed)
        except BaseException as error:
            self._close_failures.extend(_capture_failures('latency_native_close_failed', error))
        finally:
            owner = None

    async def _wait_ha(self, record: _LatencyHA) -> None:
        resource = record.resource
        try:
            if resource is not None:
                await resource.close()
            record.confirmed = True
        except BaseException as error:
            self._close_failures.extend(_capture_failures('latency_ha_close_failed', error))
        finally:
            resource = record = None

    async def _close_owned(self) -> None:
        owners = records = waiters = owner = record = resource = None
        try:
            await asyncio.sleep(0)
            if self._closing is None:
                self._closing = asyncio.current_task()
            while True:
                with self._ownership:
                    pending = any((not token.finished for token in self._operations))
                if not pending:
                    break
                await asyncio.sleep(0.005)
            owners = tuple(self._sync_owners)
            records = tuple(self._ha_records)
            for owner in owners:
                try:
                    owner.request_close()
                except BaseException as error:
                    self._close_failures.extend(_capture_failures('latency_native_seal_failed', error))
            for owner in owners:
                try:
                    owner.begin_close()
                except BaseException as error:
                    self._close_failures.extend(_capture_failures('latency_native_begin_failed', error))
            waiters = []
            for record in records:
                try:
                    task = asyncio.create_task(self._wait_ha(record), context=Context())
                    self._close_waiters.add(task)
                    waiters.append(task)
                except BaseException as error:
                    self._close_failures.extend(_capture_failures('latency_ha_wait_start_failed', error))
            for owner in owners:
                try:
                    task = asyncio.create_task(self._wait_sync_owner(owner), context=Context())
                    self._close_waiters.add(task)
                    waiters.append(task)
                except BaseException as error:
                    self._close_failures.extend(_capture_failures('latency_native_wait_start_failed', error))
            if waiters:
                await asyncio.gather(*(asyncio.shield(task) for task in waiters))
            for slot, resource in tuple(self._owned_fields.items()):
                if slot == '_ha':
                    confirmed = any((record.resource is resource and record.confirmed for record in records))
                else:
                    owner = self._owned_owner.get(slot)
                    confirmed = owner is not None and owner._confirmed
                if confirmed:
                    if getattr(self, slot) is resource:
                        setattr(self, slot, None)
                    self._owned_fields.pop(slot, None)
                    self._owned_owner.pop(slot, None)
            self._sync_owners[:] = [owner for owner in owners if not owner._confirmed]
            for record in records:
                if record.confirmed:
                    if self._owned_ha is record.resource:
                        self._owned_ha = None
                    record.resource = None
            self._ha_records[:] = [record for record in records if not record.confirmed]
            if not self._sync_owners and (not self._ha_records) and (not self._owned_fields) and (not self._close_failures):
                self._retirement_confirmed = True
                with _LATENCY_OWNERS_LOCK:
                    _LATENCY_OWNERS.discard(self)
        except BaseException as error:
            self._close_failures.extend(_capture_failures('latency_retirement_failed', error))
        finally:
            while True:
                pending_tasks = tuple((task for task in self._close_waiters if not task.done()))
                if not pending_tasks:
                    break
                try:
                    await asyncio.gather(*(asyncio.shield(task) for task in pending_tasks))
                except BaseException as error:
                    self._close_failures.extend(_capture_failures('latency_waiter_drain_failed', error))
            self._close_waiters.clear()
            owners = records = waiters = owner = record = resource = None
        if self._close_failures or not self._retirement_confirmed:
            raise RuntimeError('latency_retirement_failed')
        self._close_coroutine = None

    async def aclose(self) -> None:
        """Observe the same coordinator through repeated caller cancellation."""
        self._require_close_caller()
        cancelled = []
        try:
            closing = self.begin_close()
        except BaseException as error:
            items = _capture_failures('latency_close_observer_failed', error)
            closing = self._closing
            if closing is None:
                _raise_failures((*items, *self._close_failures))
            cancelled.extend(items)
        while not closing.done():
            try:
                await asyncio.shield(closing)
            except asyncio.CancelledError as error:
                cancelled.extend(_capture_failures('latency_close_observer_cancelled', error))
            except Exception:
                break
        if closing.cancelled():
            self._close_failures.append(ResourceFailure('latency_close_owner_cancelled', 'cancel'))
        elif closing.exception() is not None and (not self._close_failures):
            self._close_failures.append(ResourceFailure('latency_retirement_failed', 'RuntimeError'))
        _raise_failures((*self._close_failures, *cancelled))

    async def __aenter__(self):
        """Use the benchmark until its terminal asynchronous context exit."""
        if self._closed:
            raise RuntimeError('latency_closed')
        return self

    async def __aexit__(self, error_type, error, traceback):
        """Retire all owned resources before propagating safe failure metadata."""
        failures = []
        if error is not None:
            failures.extend(_capture_failures('latency_context_failed', error))
        error = None
        del error_type, traceback
        try:
            await self.aclose()
        except BaseException as close_error:
            failures.extend(_capture_failures('latency_context_close_failed', close_error))
        _raise_failures(failures)
        return False

    async def run_full_benchmark(self, iterations: int=10) -> dict:
        """Run the unchanged sample order, retire, then print the summary."""
        summaries, failures = (None, [])
        try:
            with self._operation():
                try:
                    summaries = await self._run_full_body(iterations=iterations)
                except BaseException as error:
                    failures.extend(_capture_failures('latency_benchmark_failed', error))
        except BaseException as error:
            failures.extend(_capture_failures('latency_operation_failed', error))
        try:
            await self.aclose()
        except BaseException as error:
            failures.extend(_capture_failures('latency_benchmark_close_failed', error))
        if not failures:
            try:
                self._print_summary(summaries)
            except BaseException as error:
                failures.extend(_capture_failures('latency_report_failed', error))
        if failures:
            summaries = None
            _raise_failures(failures)
        return summaries

async def main():
    parser = argparse.ArgumentParser(description="Benchmark de latencia KZA")
    parser.add_argument("--iterations", "-n", type=int, default=10, help="Iteraciones por componente")
    parser.add_argument("--config", "-c", default="config/settings.yaml", help="Path a configuración")
    args = parser.parse_args()

    benchmark = LatencyBenchmark(config_path=args.config)
    await benchmark.run_full_benchmark(iterations=args.iterations)


if __name__ == "__main__":
    asyncio.run(main())
