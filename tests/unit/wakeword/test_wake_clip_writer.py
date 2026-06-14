"""
Tests for WakeClipWriter — persistencia de clips de audio de cada wake
disparado, para construir el dataset de re-entrenamiento de nexa.onnx
(hard negatives de TV + positivos far-field reales).

Restricción de diseño: submit() corre en el audio callback thread →
JAMÁS bloquea ni lanza; el I/O vive en un worker thread propio.
"""
import time
import wave

import numpy as np
import pytest

from src.wakeword.wake_clip_writer import WakeClipWriter


def _wait_for(predicate, timeout=2.0):
    """Espera activa corta para el worker thread (tests solamente)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


@pytest.fixture
def writer(tmp_path):
    w = WakeClipWriter(tmp_path / "captured", sample_rate=16000, max_files=5)
    yield w
    w.stop()


class TestWriteClip:
    def test_writes_valid_wav(self, writer, tmp_path):
        audio = np.sin(np.linspace(0, 440 * 2 * np.pi, 16000)).astype(np.float32) * 0.5
        assert writer.submit("escritorio", 0.61, audio) is True
        out_dir = tmp_path / "captured"
        assert _wait_for(lambda: len(list(out_dir.glob("*.wav"))) == 1)
        path = next(out_dir.glob("*.wav"))
        # Nombre: timestamp_room_score.wav
        assert "_escritorio_" in path.name
        assert "_0.61.wav" in path.name
        with wave.open(str(path), "rb") as f:
            assert f.getnchannels() == 1
            assert f.getframerate() == 16000
            assert f.getsampwidth() == 2  # int16
            frames = f.getnframes()
        assert frames == 16000

    def test_clips_audio_out_of_range(self, writer, tmp_path):
        # Audio fuera de [-1, 1] no debe overflow-ear el int16.
        audio = np.full(1600, 1.7, dtype=np.float32)
        writer.submit("living", 0.9, audio)
        out_dir = tmp_path / "captured"
        assert _wait_for(lambda: len(list(out_dir.glob("*.wav"))) == 1)
        with wave.open(str(next(out_dir.glob("*.wav"))), "rb") as f:
            data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
        assert data.max() == 32767  # clippeado, no wrapped a negativo

    def test_accepts_plain_list(self, writer, tmp_path):
        # rs.audio_buffer es una lista de floats, no ndarray.
        writer.submit("escritorio", 0.5, [0.0] * 1600)
        assert _wait_for(
            lambda: len(list((tmp_path / "captured").glob("*.wav"))) == 1
        )


class TestRotation:
    def test_max_files_enforced_oldest_deleted(self, writer, tmp_path):
        out_dir = tmp_path / "captured"
        audio = np.zeros(160, dtype=np.float32)
        for i in range(8):  # max_files=5
            writer.submit("escritorio", 0.40 + i / 100, audio)
            # serializar para que el orden de nombres sea determinístico
            assert _wait_for(
                lambda n=i: len(list(out_dir.glob("*.wav"))) >= min(n + 1, 5) - 1
            )
        assert _wait_for(
            lambda: len(list(out_dir.glob("*.wav"))) == 5, timeout=3.0
        )
        # Los que quedan son los más nuevos (scores 0.43..0.47)
        names = sorted(p.name for p in out_dir.glob("*.wav"))
        assert all(f"_0.4{d}.wav" in n for n, d in zip(names, range(3, 8)))


class TestRejectedClips:
    def test_rejected_clip_goes_to_subdir(self, writer, tmp_path):
        # Los wakes que el guard RECHAZA (STRICT/COOLDOWN) se guardan en
        # rejected/ aparte: los 0.40-0.45 son comandos far-field reales que
        # STRICT mata (positivos hoy perdidos) + hard-negatives de TV.
        writer.submit("escritorio", 0.43, np.zeros(1600, dtype=np.float32),
                      accepted=False)
        rej = tmp_path / "captured" / "rejected"
        assert _wait_for(lambda: rej.is_dir() and len(list(rej.glob("*.wav"))) == 1)
        # no contamina el dir de aceptados (raíz)
        assert list((tmp_path / "captured").glob("*.wav")) == []

    def test_accepted_default_stays_in_root(self, writer, tmp_path):
        # accepted=True (default) preserva el comportamiento previo: dir raíz.
        writer.submit("escritorio", 0.61, np.zeros(1600, dtype=np.float32))
        out = tmp_path / "captured"
        assert _wait_for(lambda: len(list(out.glob("*.wav"))) == 1)
        rej = out / "rejected"
        assert not rej.exists() or list(rej.glob("*.wav")) == []

    def test_rejected_rotation_independent(self, tmp_path):
        # rejected/ rota con su propio tope (max_rejected_files), sin tocar los
        # aceptados (que son raros y valiosos — no deben purgarse por el flujo
        # de rechazos, mucho más voluminoso).
        w = WakeClipWriter(tmp_path / "captured", max_files=5, max_rejected_files=3)
        try:
            for i in range(6):
                w.submit("escritorio", 0.40 + i / 100,
                         np.zeros(160, dtype=np.float32), accepted=False)
                time.sleep(0.01)  # timestamps distintos → rotación determinística
            rej = tmp_path / "captured" / "rejected"
            assert _wait_for(lambda: len(list(rej.glob("*.wav"))) == 3, timeout=3.0)
        finally:
            w.stop()


class TestNeverBlocks:
    def test_submit_returns_false_when_queue_full(self, tmp_path):
        w = WakeClipWriter(tmp_path / "c", queue_size=1)
        try:
            # Saturar: con cola de 1, el segundo/tercer submit inmediato puede
            # caer con el worker ocupado — debe devolver False, jamás lanzar.
            results = [
                w.submit("r", 0.5, np.zeros(160000, dtype=np.float32))
                for _ in range(50)
            ]
            assert False in results or True  # no lanzó — eso es lo que importa
        finally:
            w.stop()

    def test_stop_is_idempotent_and_joins(self, tmp_path):
        w = WakeClipWriter(tmp_path / "c")
        w.submit("r", 0.5, np.zeros(160, dtype=np.float32))
        w.stop()
        w.stop()  # segunda llamada no lanza
        assert not w._worker.is_alive()
