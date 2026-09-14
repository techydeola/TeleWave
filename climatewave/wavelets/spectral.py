"""
Wavelet Spectral Analysis & Continuous Wavelet Transform (CWT) power spectrum generator.
Detects dominant non-stationary periodicities (e.g., Annual, Quasi-Biennial, ENSO 2-7yr, Decadal cycles).
"""

import math
from typing import List, Dict, Tuple, Any


class WaveletSpectralAnalysis:
    """Computes Continuous Wavelet Transform (Morlet) power spectrum across scale/frequency."""

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
        signal: List[float],
        dt: float = 1.0,
        num_scales: int = 32,
        s0: float = 2.0,
        dj: float = 0.25,
    ) -> Dict[str, Any]:
        """
        Computes the 2D Wavelet Power Spectrum (WPS = |W(s, t)|^2) and global power spectrum.
        """
        N = len(signal)
        mean_val = sum(signal) / N if N > 0 else 0.0
        norm_sig = [x - mean_val for x in signal]

        scales = [s0 * (2.0 ** (j * dj)) for j in range(num_scales)]
        # Exact Fourier equivalent period for Morlet wavelet with omega0=6.0:
        # lambda = (4 * pi * scale) / (omega0 + sqrt(2 + omega0^2)) ≈ 1.033044 * scale
        fourier_factor = (4.0 * math.pi) / (6.0 + math.sqrt(2.0 + 6.0 ** 2))
        periods = [scale * fourier_factor for scale in scales]

        power_matrix = []
        global_power = [0.0] * num_scales

        for s_idx, scale in enumerate(scales):
            row_power = []
            half_win = int(min(N, int(4.0 * scale / dt)))

            for t_idx in range(N):
                real_conv = 0.0
                imag_conv = 0.0
                start_k = max(0, t_idx - half_win)
                end_k = min(N, t_idx + half_win + 1)

                for k in range(start_k, end_k):
                    t_diff = (t_idx - k) * dt
                    wr, wi = cls.morlet_wavelet(t_diff, scale)
                    real_conv += norm_sig[k] * wr * dt
                    imag_conv += norm_sig[k] * wi * dt

                p = real_conv ** 2 + imag_conv ** 2
                row_power.append(p)
                global_power[s_idx] += p

            global_power[s_idx] /= N
            power_matrix.append(row_power)

        # Identify dominant period
        max_idx = global_power.index(max(global_power)) if global_power else 0
        dominant_period = round(periods[max_idx], 1)

        return {
            "periods": [round(p, 2) for p in periods],
            "scales": [round(s, 2) for s in scales],
            "global_power": [round(g, 4) for g in global_power],
            "dominant_period": dominant_period,
            "power_matrix": power_matrix,
        }
