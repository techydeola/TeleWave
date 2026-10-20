"""
Wavelet Spectral Analysis & Continuous Wavelet Transform (CWT) power spectrum generator.
Detects dominant non-stationary periodicities (e.g., Annual, Quasi-Biennial, ENSO 2-7yr, Decadal cycles).
Fully vectorized using NumPy for 2D convolution and power spectrum integration.
"""

import math
from typing import List, Dict, Tuple, Any, Union
import numpy as np


class WaveletSpectralAnalysis:
    """Computes Continuous Wavelet Transform (Morlet) power spectrum across scale/frequency using NumPy."""

    @staticmethod
    def morlet_wavelet(t: float, scale: float, omega0: float = 6.0) -> Tuple[float, float]:
        """Morlet complex wavelet: psi(t) = pi^(-1/4) * exp(i*w0*t) * exp(-t^2 / 2)."""
        norm = (math.pi ** -0.25) / math.sqrt(scale)
        t_scaled = t / scale
        envelope = math.exp(-0.5 * (t_scaled ** 2))
        real_part = norm * envelope * math.cos(omega0 * t_scaled)
        imag_part = norm * envelope * math.sin(omega0 * t_scaled)
        return real_part, imag_part

    @classmethod
    def compute_power_spectrum(
        cls,
        signal: Union[List[float], np.ndarray],
        dt: float = 1.0,
        num_scales: int = 32,
        s0: float = 2.0,
        dj: float = 0.25,
        omega0: float = 6.0,
    ) -> Dict[str, Any]:
        """
        Computes the 2D Wavelet Power Spectrum (WPS = |W(s, t)|^2) and global power spectrum.
        Fully vectorized with NumPy 2D convolution tensors.
        """
        sig = np.asarray(signal, dtype=np.float64)
        N = len(sig)
        mean_val = float(np.mean(sig)) if N > 0 else 0.0
        norm_sig = sig - mean_val

        scales = s0 * (2.0 ** (np.arange(num_scales, dtype=np.float64) * dj))
        # Exact Fourier equivalent period for Morlet wavelet with omega0=6.0:
        # lambda = (4 * pi * scale) / (omega0 + sqrt(2 + omega0^2)) ≈ 1.033044 * scale
        fourier_factor = (4.0 * math.pi) / (omega0 + math.sqrt(2.0 + omega0 ** 2))
        periods = scales * fourier_factor

        power_matrix = np.empty((num_scales, N), dtype=np.float64)
        norm_const = math.pi ** -0.25

        # 2D distance matrix: shape (N, N), diff[t, k] = (t - k) * dt
        t_diff = (np.arange(N)[:, None] - np.arange(N)[None, :]) * dt

        for s_idx, scale in enumerate(scales):
            half_win = int(min(N, int(4.0 * scale / dt)))
            mask = np.abs(t_diff) <= half_win * dt
            t_scaled = t_diff / scale
            envelope = np.where(mask, np.exp(-0.5 * (t_scaled ** 2)), 0.0)
            norm = norm_const / math.sqrt(scale)

            wr = norm * envelope * np.cos(omega0 * t_scaled) * dt
            wi = norm * envelope * np.sin(omega0 * t_scaled) * dt

            real_conv = np.dot(wr, norm_sig)
            imag_conv = np.dot(wi, norm_sig)

            power_matrix[s_idx] = real_conv ** 2 + imag_conv ** 2

        global_power = np.mean(power_matrix, axis=1)
        max_idx = int(np.argmax(global_power)) if len(global_power) > 0 else 0
        dominant_period = round(float(periods[max_idx]), 1)

        return {
            "periods": [round(float(p), 2) for p in periods],
            "scales": [round(float(s), 2) for s in scales],
            "global_power": [round(float(g), 4) for g in global_power],
            "dominant_period": dominant_period,
            "power_matrix": power_matrix.tolist(),
        }
