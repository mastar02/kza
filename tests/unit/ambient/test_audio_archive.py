"""Tests: AudioArchiver — persistencia FLAC best-effort del ambient."""
import asyncio

import numpy as np
import soundfile as sf

from src.ambient.audio_archive import AudioArchiver


def _run(coro):
    return asyncio.run(coro)


def _audio(n: int = 1600, ch: int = 2) -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.standard_normal((n, ch)).astype(np.float32) * 0.1


def test_escribe_flac_y_devuelve_ruta(tmp_path):
    arch = AudioArchiver(base_dir=str(tmp_path), enabled=True)
    path = _run(arch.write("escritorio", 42, _audio()))
    assert path is not None
    assert path.endswith("escritorio/42.flac")
    data, sr = sf.read(path)
    assert sr == 16000
    assert data.ndim == 1          # se guarda mono
    assert len(data) == 1600


def test_deshabilitado_no_escribe_nada(tmp_path):
    arch = AudioArchiver(base_dir=str(tmp_path), enabled=False)
    assert _run(arch.write("escritorio", 1, _audio())) is None
    assert list(tmp_path.iterdir()) == []


def test_audio_mono_1d_tambien_funciona(tmp_path):
    arch = AudioArchiver(base_dir=str(tmp_path), enabled=True)
    mono = _audio(ch=1).reshape(-1)
    path = _run(arch.write("cocina", 3, mono))
    data, _ = sf.read(path)
    assert len(data) == 1600


def test_fallo_de_escritura_devuelve_none_sin_lanzar(tmp_path):
    arch = AudioArchiver(base_dir=str(tmp_path / "no" / "existe"), enabled=True)
    # base_dir se crea solo; forzamos el fallo con un audio inválido
    assert _run(arch.write("escritorio", 1, np.array([], dtype=np.float32))) is None


def test_disco_lleno_desactiva_la_escritura(tmp_path):
    arch = AudioArchiver(base_dir=str(tmp_path), enabled=True,
                         min_free_bytes=10**18)  # piso imposible de cumplir
    assert _run(arch.write("escritorio", 1, _audio())) is None
    assert not (tmp_path / "escritorio").exists()
