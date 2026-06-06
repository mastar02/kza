"""DoA por software — GCC-PHAT sobre los mics crudos del XVF3800 (fw 6ch).

Azimut RELATIVO: atan2 de los TDOAs de los dos pares diagonales del array.
Sin geometría absoluta es consistente consigo mismo — suficiente para
clasificar contra una firma calibrada (tv_azimuth se mide en Fase 2 con la
TV sonando). stability = módulo del promedio circular entre sub-ventanas:
1.0 = dirección clavada (fuente puntual), ~0 = difuso/ruido.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def gcc_phat(
    sig: np.ndarray,
    ref: np.ndarray,
    fs: int = 16000,
    max_tau: float | None = None,
    interp: int = 4,
) -> float:
    """TDOA (s) entre sig y ref por correlación cruzada generalizada PHAT."""
    n = sig.shape[0] + ref.shape[0]
    SIG = np.fft.rfft(sig, n=n)
    REF = np.fft.rfft(ref, n=n)
    r = SIG * np.conj(REF)
    denom = np.abs(r)
    denom[denom < 1e-15] = 1e-15
    cc = np.fft.irfft(r / denom, n=interp * n)
    max_shift = int(interp * n / 2)
    if max_tau is not None:
        max_shift = min(int(interp * fs * max_tau), max_shift)
    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
    shift = int(np.argmax(np.abs(cc))) - max_shift
    return shift / float(interp * fs)


@dataclass
class DoAResult:
    azimuth: float    # rad relativo [-pi, pi]
    stability: float  # 0-1


class DoAEstimator:
    """Azimut relativo + estabilidad de un segmento multicanal."""

    def __init__(
        self,
        sample_rate: int = 16000,
        raw_first_col: int = 2,
        n_raw: int = 4,
        win_s: float = 0.5,
        mic_max_tau_s: float = 0.0005,  # ~17cm de apertura máx — clamp anti-outlier
    ):
        self.sample_rate = sample_rate
        self.raw_first_col = raw_first_col
        self.n_raw = n_raw
        self.win_s = win_s
        self.mic_max_tau_s = mic_max_tau_s

    def estimate(self, audio: np.ndarray) -> DoAResult | None:
        """None si el audio no trae los mics crudos (firmware 2ch)."""
        if audio.ndim != 2 or audio.shape[1] < self.raw_first_col + self.n_raw:
            return None
        c = self.raw_first_col
        m0, m1, m2, m3 = (audio[:, c + i] for i in range(4))

        win = int(self.win_s * self.sample_rate)
        if audio.shape[0] < win:
            win = audio.shape[0]
        azimuths = []
        for start in range(0, audio.shape[0] - win + 1, win):
            sl = slice(start, start + win)
            # pares diagonales del array: TDOA en ejes ~ortogonales
            tau02 = gcc_phat(m0[sl], m2[sl], fs=self.sample_rate, max_tau=self.mic_max_tau_s)
            tau13 = gcc_phat(m1[sl], m3[sl], fs=self.sample_rate, max_tau=self.mic_max_tau_s)
            azimuths.append(np.arctan2(tau13, tau02))
        if not azimuths:
            return None
        vec = np.exp(1j * np.array(azimuths))
        mean_vec = vec.mean()
        return DoAResult(
            azimuth=float(np.angle(mean_vec)),
            stability=float(np.abs(mean_vec)),
        )


def angular_distance(a: float, b: float) -> float:
    """Distancia angular |a-b| envuelta a [0, pi]."""
    d = abs(a - b) % (2 * np.pi)
    return float(min(d, 2 * np.pi - d))
