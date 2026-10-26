"""Callback-driven output double with observable drain and cancellation."""

import itertools
import threading
import time
from types import SimpleNamespace

import numpy as np


class CallbackStop(Exception):
    """Device should drain its queued tail before marking inactive."""


class CallbackAbort(Exception):
    """Device should discard its queued tail immediately."""


class AudioOutput:
    """Own a driver thread; keep copies only in this test observation sink."""

    def __init__(self, backend, **options):
        self.backend = backend
        self.callback = options["callback"]
        self.finished_callback = options["finished_callback"]
        self.channels = options["channels"]
        self.device = options.get("device")
        self.owner = self.callback.__self__
        self.aborted = threading.Event()
        self.draining = threading.Event()
        self.delivered = threading.Event()
        self.finished = threading.Event()
        self.closed = False
        self.error = None
        self.frames = []
        self.full_frames = []
        self.thread = None

    def start(self):
        if self.backend.fail_start:
            raise RuntimeError("synthetic output start failure")
        self.thread = threading.Thread(target=self._run, name="test-audio-output", daemon=True)
        self.thread.start()

    def _run(self):
        try:
            if self.backend.stall:
                self.aborted.wait()
                return
            for index, count in enumerate(itertools.cycle(self.backend.frame_sizes)):
                if self.aborted.is_set():
                    return
                if self.backend.fail_after is not None and index >= self.backend.fail_after:
                    return
                output = np.empty((count, self.channels), dtype=np.float32)
                before = self.owner.get_stats()["played_samples"]
                terminal = False
                try:
                    self.callback(output, count, None, SimpleNamespace(output_underflow=False))
                except CallbackStop:
                    terminal = True
                except CallbackAbort:
                    return
                after = self.owner.get_stats()["played_samples"]
                self.frames.append(output[:after - before].copy())
                self.full_frames.append(output.copy())
                if after > before:
                    self.delivered.set()
                if terminal:
                    self.draining.set()
                    while not self.backend.release_tail.wait(0.01):
                        if self.aborted.is_set():
                            return
                    return
                time.sleep(self.backend.callback_pause_s)
        except Exception as error:
            self.error = error
        finally:
            self.finished.set()
            self.finished_callback()

    def stop(self):
        if not self.finished.is_set():
            self.error = AssertionError("Normal close preceded device drain")
        self.aborted.set()

    def abort(self):
        self.aborted.set()

    def close(self):
        self.closed = True
        self.aborted.set()
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=1)
            assert not self.thread.is_alive(), "Synthetic driver thread leaked"


class AudioBackend:
    """Minimal sounddevice surface; never opens a hardware output."""

    CallbackStop = CallbackStop
    CallbackAbort = CallbackAbort

    def __init__(self, *, frame_sizes=(128, 512, 1024), stall=False, hold_tail=False, fail_start=False, fail_after=None):
        self.frame_sizes = frame_sizes
        self.stall, self.fail_start, self.fail_after = stall, fail_start, fail_after
        self.callback_pause_s = 0.0002
        self.release_tail = threading.Event()
        if not hold_tail:
            self.release_tail.set()
        self.created = threading.Event()
        self.outputs = []

    def OutputStream(self, **options):
        output = AudioOutput(self, **options)
        self.outputs.append(output)
        self.created.set()
        return output

    def heard(self):
        arrays = [chunk for output in self.outputs for chunk in output.frames]
        return np.concatenate(arrays) if arrays else np.empty((0, 1), dtype=np.float32)
