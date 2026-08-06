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


def test_deshabilitado_no_cuenta_en_stats(tmp_path):
    """M14 (review PR #15): el docstring de write() promete que
    deshabilitado 'retorna None sin contar ni loguear' — sin este test esa
    mitad del contrato no tenía cobertura, solo el efecto en disco."""
    arch = AudioArchiver(base_dir=str(tmp_path), enabled=False)
    _run(arch.write("escritorio", 1, _audio()))
    assert arch.stats == {"written": 0, "skipped_disk": 0, "failed": 0}


def test_audio_mono_1d_tambien_funciona(tmp_path):
    arch = AudioArchiver(base_dir=str(tmp_path), enabled=True)
    mono = _audio(ch=1).reshape(-1)
    path = _run(arch.write("cocina", 3, mono))
    data, _ = sf.read(path)
    assert len(data) == 1600


def test_audio_vacio_devuelve_none(tmp_path):
    """Audio vacío: fallo en validación de entrada."""
    arch = AudioArchiver(base_dir=str(tmp_path), enabled=True)
    # Forzamos fallo de validación con audio inválido
    assert _run(arch.write("escritorio", 1, np.array([], dtype=np.float32))) is None


def test_sf_write_falla_devuelve_none_sin_propagar(tmp_path, monkeypatch):
    """Fallo en sf.write(): excepción capturada, devuelve None sin propagar."""
    import soundfile as sf

    def mock_write_fail(*args, **kwargs):
        raise OSError("Permission denied / Disco lleno / I/O error")

    monkeypatch.setattr(sf, "write", mock_write_fail)

    arch = AudioArchiver(base_dir=str(tmp_path), enabled=True)
    # El audio es válido; el fallo ocurre en _write_sync() → sf.write()
    result = _run(arch.write("escritorio", 1, _audio()))
    assert result is None  # write() devuelve None, sin propagar la excepción


def test_disco_lleno_desactiva_la_escritura(tmp_path):
    arch = AudioArchiver(base_dir=str(tmp_path), enabled=True,
                         min_free_bytes=10**18)  # piso imposible de cumplir
    assert _run(arch.write("escritorio", 1, _audio())) is None
    assert not (tmp_path / "escritorio").exists()


def test_piso_de_disco_corta_en_el_thread_y_cuenta_skipped(tmp_path):
    # min_free_bytes imposible (1 exabyte): _has_room da False sin monkeypatch.
    arch = AudioArchiver(base_dir=str(tmp_path), enabled=True,
                         min_free_bytes=10**18)
    audio = np.full(1600, 0.1, dtype=np.float32)
    path = _run(arch.write("cocina", 1, audio))
    assert path is None
    assert list(tmp_path.rglob("*.flac")) == []
    assert arch.stats == {"written": 0, "skipped_disk": 1, "failed": 0}


def test_escritura_ok_cuenta_written(tmp_path):
    arch = AudioArchiver(base_dir=str(tmp_path), enabled=True)
    audio = np.full(1600, 0.1, dtype=np.float32)
    path = _run(arch.write("cocina", 2, audio))
    assert path is not None and path.endswith("cocina/2.flac")
    assert arch.stats == {"written": 1, "skipped_disk": 0, "failed": 0}


def test_fallo_de_escritura_cuenta_failed(tmp_path):
    arch = AudioArchiver(base_dir=str(tmp_path), enabled=True)
    audio = np.zeros((0,), dtype=np.float32)   # audio vacío → ValueError interna
    path = _run(arch.write("cocina", 3, audio))
    assert path is None
    assert arch.stats["failed"] == 1
