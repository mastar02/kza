"""Synthetic diagnostic SDK leaves; Model has no invented close/reset API."""
from __future__ import annotations

import sys
import threading
from types import ModuleType
import weakref

import numpy as np


class DiagnosticLeaves:
    def __init__(self, monkeypatch):
        self.values = np.array([.002, .001, .25, .03, -.04, .001], dtype=np.float32)
        self.events = []
        self.hold = set()
        self.faults = set()
        self.entered = {}
        self.releases = {}
        self.streams = []
        self.read_frames = []
        self.models = []
        self.model_refs = []
        self.pcm_refs = []
        leaf = self

        class InputStream:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.stop_kwargs = []
                self.close_kwargs = []
                self.close_returned = False
                leaf.streams.append(self)
                leaf.checkpoint('stream-construct')

            def start(self, **kwargs):
                leaf.checkpoint('start')
                return self

            def read(self, frames):
                leaf.read_frames.append(frames)
                leaf.checkpoint('read')
                audio = np.tile(leaf.values, (frames, 1))
                leaf.pcm_refs.append(weakref.ref(audio))
                return audio, False

            def stop(self, **kwargs):
                self.stop_kwargs.append(kwargs)
                assert kwargs == {'ignore_errors': False}
                leaf.checkpoint('stop')

            def close(self, **kwargs):
                self.close_kwargs.append(kwargs)
                assert kwargs == {'ignore_errors': False}
                leaf.checkpoint('close')
                self.close_returned = True

            def __enter__(self):
                return self.start()

            def __exit__(self, *args):
                # Mirror sounddevice's default-error-ignoring convenience path;
                # it cannot fabricate the explicit strict receipt above.
                self.stop()
                self.close()
                return False

        class Model:
            def __init__(self, **kwargs):
                assert all(stream.close_returned for stream in leaf.streams), 'model constructed before owned input close returned'
                self.ordinal = len(leaf.models)
                self.row = {'kwargs': kwargs, 'calls': 0, 'first_values': [], 'released': False}
                leaf.models.append(self.row)
                leaf.model_refs.append(weakref.ref(self))
                assert kwargs == {'wakeword_models': ['hey_jarvis'], 'inference_framework': 'onnx'}
                leaf.checkpoint('model-construct')

            def predict(self, chunk):
                assert chunk.shape == (1280,) and chunk.dtype == np.float32
                leaf.checkpoint('predict')
                if not self.row['first_values']:
                    self.row['first_values'].append(float(chunk[0]))
                index = self.row['calls']
                self.row['calls'] += 1
                leaf.pcm_refs.append(weakref.ref(chunk))
                score = [.1, .5, .5, .2, .4][index] if index < 5 else .1
                return {'hey_jarvis': score}

            def __del__(self):
                if hasattr(self, 'row'):
                    if self.ordinal == 0:
                        leaf.checkpoint('model-release')
                    self.row['released'] = True

        def forbidden(*args, **kwargs):
            raise AssertionError('global rec/wait/stop cannot retire an owned stream')

        sounddevice = ModuleType('sounddevice')
        sounddevice.InputStream = InputStream
        sounddevice.rec = sounddevice.wait = sounddevice.stop = forbidden
        openwakeword = ModuleType('openwakeword')
        openwakeword.Model = Model
        monkeypatch.setitem(sys.modules, 'sounddevice', sounddevice)
        monkeypatch.setitem(sys.modules, 'openwakeword', openwakeword)

    def event(self, name):
        return self.entered.setdefault(name, threading.Event())

    def release_event(self, name):
        return self.releases.setdefault(name, threading.Event())

    def checkpoint(self, name):
        self.events.append(name)
        self.event(name).set()
        if name in self.hold:
            assert self.release_event(name).wait(8), f'{name} gate not released'
        if name in self.faults:
            raise RuntimeError(f'synthetic_{name}_failure')

    def release(self):
        for name in self.hold | set(self.releases):
            self.release_event(name).set()
