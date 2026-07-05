"""Tests: GCC-PHAT y DoAEstimator — azimut relativo desde mics crudos."""
import numpy as np

from src.ambient.doa import DoAEstimator, gcc_phat

SR = 16000


def _delayed_noise(n: int, delay_samples: int, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 0.3, size=n + abs(delay_samples) + 8).astype(np.float32)
    ref = base[: n]
    sig = base[delay_samples : delay_samples + n] if delay_samples >= 0 else base[: n]
    if delay_samples < 0:
        sig, ref = ref, base[-delay_samples : -delay_samples + n]
    return sig, ref


def test_gcc_phat_recovers_known_delay():
    delay = 4  # samples = 250µs @ 16k
    sig, ref = _delayed_noise(SR, delay)
    tau = gcc_phat(sig, ref, fs=SR, max_tau=0.001)
    assert abs(tau - (-delay / SR)) < 0.5 / SR  # sig adelantada → tau negativo


def test_estimator_stable_for_coherent_source():
    # Fuente coherente: mismo ruido con delays fijos entre canales → estabilidad alta
    rng = np.random.default_rng(11)
    n = SR * 2
    base = rng.normal(0, 0.3, size=n + 16).astype(np.float32)
    seg = np.zeros((n, 6), dtype=np.float32)
    # cols 2-5 = mics crudos; delays fijos simulan dirección estable
    seg[:, 2] = base[:n]
    seg[:, 3] = base[2 : n + 2]
    seg[:, 4] = base[4 : n + 4]
    seg[:, 5] = base[1 : n + 1]
    est = DoAEstimator(sample_rate=SR, raw_first_col=2, n_raw=4, win_s=0.5)
    res = est.estimate(seg)
    assert res is not None
    assert res.stability > 0.9
    assert -np.pi <= res.azimuth <= np.pi


def test_estimator_unstable_for_incoherent_noise():
    rng = np.random.default_rng(12)
    seg = rng.normal(0, 0.3, size=(SR * 2, 6)).astype(np.float32)
    est = DoAEstimator(sample_rate=SR, raw_first_col=2, n_raw=4, win_s=0.5)
    res = est.estimate(seg)
    assert res is not None
    assert res.stability < 0.9  # azimuts dispersos entre sub-ventanas


def test_estimator_returns_none_without_raw_channels():
    seg = np.zeros((SR, 2), dtype=np.float32)  # firmware 2ch: sin mics crudos
    est = DoAEstimator(sample_rate=SR, raw_first_col=2, n_raw=4)
    assert est.estimate(seg) is None
