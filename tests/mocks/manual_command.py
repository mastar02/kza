"""Manual-command SDK leaves, borrowed stdin and real native receipts."""
from __future__ import annotations

import os
from pathlib import Path
import select
import sys
import threading
from types import ModuleType, SimpleNamespace
import urllib.request
import weakref

import numpy as np

from tests.mocks.zone_capture import ZoneCaptureLeaves


class PipeInput:
    def __init__(self, monkeypatch, leaf):
        read_fd, self.write_fd = os.pipe()
        self.stream = os.fdopen(read_fd, 'r', encoding='utf-8')
        self.read_fd = read_fd
        self.readers = []
        self.selectors = []
        original_select, original_read = select.select, os.read

        def observe_select(readers, writers, errors, timeout=None):
            descriptors = [value if isinstance(value, int) else value.fileno() for value in readers]
            if read_fd in descriptors:
                self.selectors.append(threading.current_thread())
                leaf.event('prompt-select').set()
            return original_select(readers, writers, errors, timeout)

        def observe_read(fd, size):
            result = original_read(fd, size)
            if fd == read_fd:
                self.readers.append(threading.current_thread())
                leaf.event('prompt-read-return').set()
            return result

        monkeypatch.setattr(sys, 'stdin', self.stream)
        monkeypatch.setattr(select, 'select', observe_select)
        monkeypatch.setattr(os, 'read', observe_read)

    def enter(self):
        os.write(self.write_fd, b'\n')

    def release(self):
        try:
            self.enter()
        except (BrokenPipeError, OSError):
            pass

    def close(self):
        os.close(self.write_fd)
        self.stream.close()


class ManualCommandLeaves(ZoneCaptureLeaves):
    def __init__(self, monkeypatch):
        super().__init__(monkeypatch)
        self.models = []
        self.model_refs = []
        self.audio_refs = []
        self.embedding_refs = []
        self.collection_refs = []
        self.segment_rows = []
        self.query_calls = []
        self.vector_states = []
        self.raw_ha_attempts = []
        self.private_env_attempts = []
        self.text = 'apaga la luz'
        self.distance = .2
        self.no_results = False
        leaf = self
        original_stream = sys.modules['sounddevice'].InputStream

        class SixChannelStream(original_stream):
            def read(self, frames):
                audio, overflowed = super().read(frames)
                assert self.kwargs['channels'] == 6
                audio[:, 1:] = np.float32(.875)
                leaf.audio_refs.append(weakref.ref(audio))
                return audio, overflowed

        sys.modules['sounddevice'].InputStream = SixChannelStream

        class Segments:
            def __init__(self, model, audio):
                self.model = model
                self.audio = audio
                self.words = iter(leaf.text.split())
                self.row = {'close_calls': 0, 'close_returned': False}
                leaf.segment_rows.append(self.row)

            def __iter__(self):
                return self

            def __next__(self):
                word = next(self.words)
                leaf.checkpoint('segment')
                return SimpleNamespace(text=word)

            def close(self):
                self.row['close_calls'] += 1
                leaf.checkpoint('segment-close')
                self.model = self.audio = None
                self.row['close_returned'] = True

        class Whisper:
            def __init__(self, *args, **kwargs):
                self.row = {'kind': 'whisper', 'args': args, 'kwargs': kwargs, 'released': False}
                leaf.models.append(self.row)
                leaf.model_refs.append(weakref.ref(self))
                leaf.checkpoint('whisper-construct')

            def transcribe(self, audio, **kwargs):
                assert audio.ndim == 1 and audio.dtype == np.float32
                assert np.all(audio == np.float32(.25))
                assert kwargs == {'language': 'es', 'vad_filter': True, 'beam_size': 1}
                leaf.audio_refs.append(weakref.ref(audio))
                leaf.checkpoint('transcribe')
                return Segments(self, audio), SimpleNamespace(language='es')

            def __del__(self):
                leaf.checkpoint('model-release')
                self.row['released'] = True

        class Embedder:
            def __init__(self, *args, **kwargs):
                self.row = {'kind': 'embedder', 'args': args, 'kwargs': kwargs, 'released': False}
                leaf.models.append(self.row)
                leaf.model_refs.append(weakref.ref(self))
                leaf.checkpoint('embedder-construct')

            def encode(self, text):
                assert text == leaf.text
                leaf.checkpoint('encode')
                result = np.array([.1, .2, .3], dtype=np.float32)
                leaf.embedding_refs.append(weakref.ref(result))
                return result

            def __del__(self):
                leaf.checkpoint('model-release')
                self.row['released'] = True

        class Collection:
            def __init__(self, client):
                self.client = client
                leaf.collection_refs.append(weakref.ref(self))

            def count(self):
                return 1

            def query(self, **kwargs):
                leaf.query_calls.append(kwargs)
                leaf.checkpoint('query')
                if leaf.no_results:
                    return {'ids': [[]], 'metadatas': [[]], 'distances': [[]], 'documents': [[]]}
                return {'ids': [['synthetic-command']],
                    'metadatas': [[{'entity_id': 'light.test_lamp', 'domain': 'light', 'service': 'turn_off'}]],
                    'distances': [[leaf.distance]], 'documents': [['apaga la luz']]}

        class VectorClient:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.state = {'entered': threading.Event(), 'release': threading.Event(),
                    'close_calls': 0, 'close_returned': False}
                state = self.state
                leaf.vector_states.append(state)

                def native():
                    state['entered'].set()
                    assert state['release'].wait(20), 'synthetic vector native work not released'

                worker = threading.Thread(target=native, daemon=True)
                state['worker'] = worker
                leaf.threads.append(worker)
                worker.start()
                assert state['entered'].wait(3)
                leaf.model_refs.append(weakref.ref(self))
                leaf.checkpoint('chroma-construct')

            def get_collection(self, name):
                if name == 'home_assistant_commands__pointer':
                    # No pointer: the base collection is active (src/vectordb/command_index.py).
                    from chromadb.errors import NotFoundError
                    raise NotFoundError(f'Collection [{name}] does not exist')
                assert name == 'home_assistant_commands'
                return Collection(self)

            def close(self):
                self.state['close_calls'] += 1
                self.state['release'].set()
                leaf.join(self.state['worker'])
                leaf.checkpoint('chroma-close')
                self.state['close_returned'] = True

        whisper = ModuleType('faster_whisper')
        whisper.WhisperModel = Whisper
        embedder = ModuleType('sentence_transformers')
        embedder.SentenceTransformer = Embedder
        monkeypatch.setitem(sys.modules, 'faster_whisper', whisper)
        monkeypatch.setitem(sys.modules, 'sentence_transformers', embedder)
        import chromadb
        monkeypatch.setattr(chromadb, 'PersistentClient', VectorClient)

        original_read_text = Path.read_text
        private_env = Path(__file__).resolve().parents[2] / '.env'

        def refuse_private_env(path, *args, **kwargs):
            if path == private_env:
                leaf.private_env_attempts.append('blocked')
                raise AssertionError('Manual command must not read project dotenv')
            return original_read_text(path, *args, **kwargs)

        def refuse_raw_ha(*args, **kwargs):
            leaf.raw_ha_attempts.append('blocked')
            raise AssertionError('Raw HA transport is not dashboard action authority')

        monkeypatch.setattr(Path, 'read_text', refuse_private_env)
        monkeypatch.setattr(urllib.request, 'urlopen', refuse_raw_ha)

    def release(self):
        for state in self.vector_states:
            state['release'].set()
        super().release()


class NativeSegmentsActor:
    """Returned SDK iterator whose native work ends only after actual close."""

    def __init__(self, audio, model, kind):
        self.audio = audio
        self.model = model
        self.close_calls = 0
        self.entered = threading.Event()
        self.release = threading.Event()
        self.exited = threading.Event()
        self.emitted = False

        def native():
            self.entered.set()
            try:
                assert self.release.wait(20), 'returned SDK actor was not closed'
            finally:
                self.exited.set()

        self.worker = threading.Thread(target=native, name=f'h-returned-{kind}', daemon=True)
        self.worker.start()
        assert self.entered.wait(3)

    def __iter__(self):
        return self

    def __next__(self):
        if self.emitted:
            raise StopIteration
        self.emitted = True
        return SimpleNamespace(text='apaga la luz')

    def close(self):
        self.close_calls += 1
        self.release.set()
        self.worker.join(5)
        assert self.exited.is_set()
        assert not self.worker.is_alive() and self.worker._handle.is_done()
        self.audio = self.model = None
