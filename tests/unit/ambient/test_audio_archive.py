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


def test_channel_mode_stereo_2ch_preserva_ch0_y_ch1(tmp_path):
    import pytest

    from src.ambient.audio_archive import ArchiveChannelMode

    t = np.linspace(0, 0.1, 1600, endpoint=False, dtype=np.float32)
    ch0 = 0.2 * np.sin(2 * np.pi * 440 * t)
    ch1 = 0.5 * np.sin(2 * np.pi * 880 * t)
    stereo = np.stack([ch0, ch1], axis=1)

    arch = AudioArchiver(
        base_dir=str(tmp_path), enabled=True, channel_mode=ArchiveChannelMode.STEREO_2CH
    )
    path = _run(arch.write("living", 7, stereo))
    assert path is not None
    data, sr = sf.read(path, dtype="float32")
    assert sr == 16000
    assert data.shape == (1600, 2)
    assert np.allclose(data[:, 0], ch0, atol=1e-3)
    assert np.allclose(data[:, 1], ch1, atol=1e-3)

    with pytest.raises(ValueError, match="invalid_archive_channel_mode"):
        AudioArchiver(base_dir=str(tmp_path), enabled=True, channel_mode="quad_4ch")


def test_channel_mode_mono_ch1_y_fallback_1ch(tmp_path):
    t = np.linspace(0, 0.1, 1600, endpoint=False, dtype=np.float32)
    ch0 = 0.1 * np.ones_like(t)
    ch1 = 0.4 * np.ones_like(t)
    stereo = np.stack([ch0, ch1], axis=1)

    arch_ch1 = AudioArchiver(base_dir=str(tmp_path), enabled=True, channel_mode="mono_ch1")
    path = _run(arch_ch1.write("cocina", 8, stereo))
    data, _ = sf.read(path, dtype="float32")
    assert data.ndim == 1
    assert np.allclose(data, ch1, atol=1e-3)

    # 1ch input with stereo_2ch degrades cleanly to mono instead of failing
    arch_2ch = AudioArchiver(base_dir=str(tmp_path), enabled=True, channel_mode="stereo_2ch")
    path_1ch = _run(arch_2ch.write("cocina", 9, ch0))
    data_1ch, _ = sf.read(path_1ch, dtype="float32")
    assert data_1ch.ndim == 1
    assert len(data_1ch) == 1600
