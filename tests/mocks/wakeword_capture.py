"""Synthetic device/model leaves with actual caller and callback Thread receipts."""
from __future__ import annotations

import sys
import threading
from types import ModuleType

import numpy as np


class WakeCaptureLeaves:
    def __init__(self, monkeypatch):
        self.events = []
        self.streams = []
        self.detectors = []
        self.threads = []
        self.hold = set()
        self.faults = set()
        self.entered = {}
        self.releases = {}
        self.read_frames = []
        self.emit_callback = False
        self.callback_error = None
        self.interrupt = threading.Event()
        self.detector_closed = threading.Event()
        self.stream_closed = threading.Event()
        self.value = .25
        leaf = self

        class InputStream:
            def __init__(self, **kwargs):
                leaf.streams.append(self)
                self.kwargs = kwargs
                self.stop_kwargs = []
                self.close_kwargs = []
                self.callbacks = []
                leaf.checkpoint('construct')

            def start(self, **kwargs):
                leaf.checkpoint('start')
                callback = self.kwargs.get('callback')
                if callback is not None and leaf.emit_callback:
                    def native_callback():
                        try:
                            frames = self.kwargs.get('blocksize', 1280)
                            callback(np.full((frames, 1), leaf.value, dtype=np.float32), frames, None, None)
                        except BaseException as error:
                            leaf.callback_error = error
                        finally:
                            leaf.event('callback-returned').set()

                    worker = threading.Thread(target=native_callback, daemon=True)
                    self.callbacks.append(worker)
                    leaf.threads.append(worker)
                    worker.start()
                return self

            def read(self, frames):
                leaf.read_frames.append(frames)
                leaf.checkpoint('read')
                return np.full((frames, 1), leaf.value, dtype=np.float32), False

            def stop(self, **kwargs):
                self.stop_kwargs.append(kwargs)
                assert kwargs == {'ignore_errors': False}, 'stop must expose its real SDK error'
                leaf.checkpoint('stop')

            def close(self, **kwargs):
                self.close_kwargs.append(kwargs)
                assert kwargs == {'ignore_errors': False}, 'close must expose its real SDK error'
                leaf.checkpoint('close')
                for worker in self.callbacks:
                    leaf.join(worker)
                leaf.stream_closed.set()

            def __enter__(self):
                return self.start()

            def __exit__(self, *args):
                # sounddevice's convenience context uses the default error-
                # ignoring API. It must not manufacture a strict owner receipt.
                self.stop()
                self.close()
                return False

        class Detector:
            PRETRAINED_MODELS = ['synthetic-pretrained']

            def __init__(self, **kwargs):
                self.kwargs = kwargs
                leaf.detectors.append(self)
                leaf.checkpoint('detector-construct')

            def load(self):
                leaf.checkpoint('detector-load')

            def get_active_models(self):
                return ['synthetic.onnx']

            def detect(self, audio):
                assert audio.ndim == 1 and np.all(audio == leaf.value)
                leaf.checkpoint('detect')
                return 'synthetic.onnx', .8

            def request_close(self):
                leaf.events.append('detector-seal')

            def close(self):
                leaf.checkpoint('detector-close')
                leaf.detector_closed.set()

        def sleep(milliseconds):
            if leaf.interrupt.wait(milliseconds / 1000):
                raise KeyboardInterrupt('bounded synthetic fixture stop')

        def forbidden(*args, **kwargs):
            raise AssertionError('global sounddevice rec/wait/stop is not an owned stream receipt')

        sounddevice = ModuleType('sounddevice')
        sounddevice.InputStream = InputStream
        sounddevice.sleep = sleep
        sounddevice.rec = sounddevice.wait = sounddevice.stop = forbidden
        detector = ModuleType('src.wakeword.detector')
        detector.WakeWordDetector = Detector
        monkeypatch.setitem(sys.modules, 'sounddevice', sounddevice)
        monkeypatch.setitem(sys.modules, 'src.wakeword.detector', detector)

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

    @staticmethod
    def join(worker):
        worker.join(5)
        assert not worker.is_alive() and worker._handle.is_done(), 'native Thread has not actually exited'

    def release(self):
        self.interrupt.set()
        for name in self.hold | set(self.releases):
            self.release_event(name).set()
        for worker in self.threads:
            self.join(worker)


class CaptureCall:
    def __init__(self, operation):
        self.result = None
        self.failure = None
        self.done = threading.Event()

        def invoke():
            try:
                self.result = operation()
            except BaseException as error:
                self.failure = error
            finally:
                self.done.set()

        self.worker = threading.Thread(target=invoke, daemon=True)
        self.worker.start()

    def join(self):
        WakeCaptureLeaves.join(self.worker)


class TrainingAuthorityFactory:
    def __init__(self, url, credential):
        self.url = url
        self.credential = credential
        self.clients = []

    def __call__(self):
        from src.audio.native_capture_authority import WakewordTrainingAuthorityClient

        client = WakewordTrainingAuthorityClient(self.url, self.credential)
        self.clients.append(client)
        return client

    def close(self):
        for client in self.clients:
            client.close()
