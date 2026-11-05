"""Synthetic audio leaves with real native Thread retirement receipts."""
from __future__ import annotations

import sys
import threading
from types import ModuleType

import numpy as np


class ZoneCaptureLeaves:
    def __init__(self, monkeypatch):
        self.streams = []
        self.threads = []
        self.events = []
        self.hold = set()
        self.faults = set()
        self.entered = {}
        self.releases = {}
        self.reads = []
        self.callback_errors = []
        self.auto_audio = {}
        self.start_failure_device = None
        self.value = .25
        leaf = self

        class InputStream:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.stop_kwargs = []
                self.close_kwargs = []
                self.callbacks = []
                self.started = threading.Event()
                self.release = threading.Event()
                self.exited = threading.Event()
                self.worker = None
                leaf.streams.append(self)
                leaf.checkpoint('construct')

            def start(self):
                def native():
                    self.started.set()
                    try:
                        audio = leaf.auto_audio.get(self.kwargs['device'])
                        if audio is not None and self.kwargs.get('callback'):
                            leaf.invoke(self, audio)
                        assert self.release.wait(20), 'synthetic native stream was not released'
                    finally:
                        self.exited.set()

                self.worker = threading.Thread(target=native, daemon=True)
                leaf.threads.append(self.worker)
                self.worker.start()
                assert self.started.wait(3)
                leaf.checkpoint('start')
                if self.kwargs['device'] == leaf.start_failure_device:
                    raise OSError('synthetic_start_failed_after_native_dispatch')
                return self

            def read(self, frames):
                leaf.reads.append((self.kwargs['device'], frames))
                leaf.checkpoint('read')
                return np.full((frames, self.kwargs['channels']), leaf.value,
                    dtype=np.float32), False

            def stop(self, **kwargs):
                self.stop_kwargs.append(kwargs)
                assert kwargs == {'ignore_errors': False}
                leaf.checkpoint('stop')
                self.release.set()
                if self.worker is not None:
                    leaf.join(self.worker)

            def close(self, **kwargs):
                self.close_kwargs.append(kwargs)
                assert kwargs == {'ignore_errors': False}
                self.release.set()
                if self.worker is not None:
                    leaf.join(self.worker)
                for worker in self.callbacks:
                    leaf.join(worker)
                leaf.checkpoint('close')

        def forbidden(*args, **kwargs):
            raise AssertionError('Global sounddevice rec/wait/stop cannot prove stream ownership')

        module = ModuleType('sounddevice')
        module.InputStream = InputStream
        module.rec = module.wait = module.stop = forbidden
        monkeypatch.setitem(sys.modules, 'sounddevice', module)

    def event(self, name):
        return self.entered.setdefault(name, threading.Event())

    def release_event(self, name):
        return self.releases.setdefault(name, threading.Event())

    def checkpoint(self, name):
        self.events.append(name)
        self.event(name).set()
        if name in self.hold:
            assert self.release_event(name).wait(8), f'{name} gate was not released'
        if name in self.faults:
            raise RuntimeError(f'synthetic_{name}_failure')

    def invoke(self, stream, audio):
        try:
            stream.kwargs['callback'](audio, len(audio), None, None)
        except BaseException as error:
            self.callback_errors.append(error)
        finally:
            self.event('callback-returned').set()

    def emit(self, stream, audio):
        worker = threading.Thread(target=lambda: self.invoke(stream, audio), daemon=True)
        stream.callbacks.append(worker)
        self.threads.append(worker)
        worker.start()
        return worker

    @staticmethod
    def join(worker):
        worker.join(5)
        assert not worker.is_alive() and worker._handle.is_done(), 'native Thread has not exited'

    def release(self):
        for name in self.hold | set(self.releases):
            self.release_event(name).set()
        for stream in self.streams:
            stream.release.set()
        for worker in self.threads:
            self.join(worker)
