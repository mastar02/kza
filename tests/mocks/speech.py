"""Deterministic PCM engine with optional blocked synthesis and call tracking."""

import threading

import numpy as np


class PCMEngine:
    sample_rate = 24000

    def __init__(self, frames=24000):
        self.frames = frames
        self.calls = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.failure = None
        self.active = self.peak_active = 0
        self.lock = threading.Lock()

    def audio(self, text):
        return np.full(self.frames, (sum(map(ord, text)) % 100 + 1) / 200, dtype=np.float32)

    def synthesize(self, text):
        with self.lock:
            self.calls.append(text)
            self.active += 1
            self.peak_active = max(self.active, self.peak_active)
        try:
            self.entered.set()
            assert self.release.wait(2), 'Synthetic engine was not released'
            if self.failure:
                raise self.failure
            return self.audio(text), self.frames / self.sample_rate
        finally:
            with self.lock:
                self.active -= 1

    def synthesize_stream(self, text):
        yield self.synthesize(text)[0]
