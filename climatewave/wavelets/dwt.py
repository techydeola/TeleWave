"""
Discrete Wavelet Transform (DWT), Inverse DWT (IDWT), and Multiresolution Analysis (MRA).
Provides pure-Python numerical implementations with zero required external dependencies,
ensuring perfect reconstruction and mathematical rigor (Parseval energy conservation).
"""

import math
from typing import List, Tuple, Dict, Any, Union
from climatewave.wavelets.filters import get_wavelet_filters


def convolve_downsample(signal: List[float], filter_kernel: List[float], step: int = 2) -> List[float]:
    """1D convolution followed by dyadic downsampling (decimation by step)."""
    N = len(signal)
    K = len(filter_kernel)
    out_len = (N + K - 1) // step
    result = []

    # Periodic boundary extension for energy conservation
    extended = signal + signal[:K]
    for i in range(0, N, step):
        val = 0.0
        for k in range(K):
            idx = (i + k) % N
            val += extended[idx] * filter_kernel[k]
        result.append(val)
    return result


def upsample_convolve(coeffs: List[float], filter_kernel: List[float], target_len: int) -> List[float]:
    """Dyadic upsampling (inserting zeros) followed by convolution."""
    K = len(filter_kernel)
    # Dyadically upsample
    upsampled = []
    for c in coeffs:
        upsampled.append(c)
        upsampled.append(0.0)

    # Convolve with reconstruction filter
    N = len(upsampled)
    result = [0.0] * target_len
    for i in range(target_len):
        val = 0.0
        for k in range(K):
            idx = (i - k) % N
            val += upsampled[idx] * filter_kernel[k]
        result[i] = val
    return result


class WaveletTransform:
    """Core multiresolution wavelet transform engine."""

    def __init__(self, wavelet: str = "db4"):
        self.wavelet_name = wavelet
        self.h_dec, self.g_dec, self.h_rec, self.g_rec = get_wavelet_filters(wavelet)

    def dwt_single_level(self, signal: List[float]) -> Tuple[List[float], List[float]]:
        """Decomposes a 1D signal into approximation (cA) and detail (cD) coefficients."""
        N = len(signal)
        # Ensure even length
        sig = list(signal)
        if len(sig) % 2 != 0:
            sig.append(sig[-1])

        K = len(self.h_dec)
        half_len = len(sig) // 2
        cA = [0.0] * half_len
        cD = [0.0] * half_len

        for i in range(half_len):
            sum_a = 0.0
            sum_d = 0.0
            for k in range(K):
                idx = (2 * i + k) % len(sig)
                sum_a += sig[idx] * self.h_dec[k]
                sum_d += sig[idx] * self.g_dec[k]
            cA[i] = sum_a
            cD[i] = sum_d

        return cA, cD

    def idwt_single_level(self, cA: List[float], cD: List[float], target_len: int) -> List[float]:
        """Reconstructs 1D signal from approximation (cA) and detail (cD) coefficients."""
        half_len = max(len(cA), len(cD))
        K = len(self.h_rec)
        rec = [0.0] * (half_len * 2)

        for i in range(half_len):
            val_a = cA[i] if i < len(cA) else 0.0
            val_d = cD[i] if i < len(cD) else 0.0
            for k in range(K):
                out_idx = (2 * i + k) % len(rec)
                rec[out_idx] += val_a * self.h_rec[k] + val_d * self.g_rec[k]

        return rec[:target_len]

    def wavedec(self, signal: List[float], level: int = 3) -> List[List[float]]:
        """
        Multilevel 1D Discrete Wavelet Decomposition.
        Returns coefficients list: [cA_J, cD_J, cD_{J-1}, ..., cD_1].
        """
        coeffs = []
        current_a = list(signal)
        for _ in range(level):
            if len(current_a) < len(self.h_dec):
                break
            cA, cD = self.dwt_single_level(current_a)
            coeffs.append(cD)
            current_a = cA
        coeffs.append(current_a)
        # Return in standard PyWavelets order: [cA_n, cD_n, cD_n-1, ..., cD_1]
        return list(reversed(coeffs))

    def waverec(self, coeffs: List[List[float]], original_len: int) -> List[float]:
        """
        Multilevel Inverse Discrete Wavelet Reconstruction.
        """
        current_a = coeffs[0]
        for cD in coeffs[1:]:
            target_len = len(cD) * 2
            current_a = self.idwt_single_level(current_a, cD, target_len)
        return current_a[:original_len]

    def mra(self, signal: List[float], level: int = 3) -> Dict[str, List[float]]:
        """
        Multiresolution Analysis (MRA) Decomposition:
        Decomposes the raw signal into zero-phase additive components of equal length:
        x(t) = A_J(t) + D_J(t) + D_{J-1}(t) + ... + D_1(t)
        """
        orig_len = len(signal)
        coeffs = self.wavedec(signal, level=level)
        num_levels = len(coeffs) - 1

        components = {}

        # 1. Reconstruct Approximation Component A_J (set all details to 0)
        zeros_coeffs = [coeffs[0]] + [[0.0] * len(cD) for cD in coeffs[1:]]
        components[f"A{num_levels}"] = self.waverec(zeros_coeffs, orig_len)

        # 2. Reconstruct Detail Components D_j (keep only j-th detail, set others to 0)
        for j in range(1, len(coeffs)):
            detail_level = num_levels - j + 1
            isolated_coeffs = [[0.0] * len(coeffs[0])] + [
                coeffs[k] if k == j else [0.0] * len(coeffs[k])
                for k in range(1, len(coeffs))
            ]
            components[f"D{detail_level}"] = self.waverec(isolated_coeffs, orig_len)

        return components

    def energy_distribution(self, signal: List[float], level: int = 3) -> Dict[str, float]:
        """
        Computes the relative wavelet energy percentage across decomposed frequency bands.
        """
        mra_dict = self.mra(signal, level=level)
        total_energy = sum(x ** 2 for x in signal)
        if total_energy == 0:
            total_energy = 1.0

        energies = {}
        for k, comp in mra_dict.items():
            comp_energy = sum(x ** 2 for x in comp)
            energies[k] = round((comp_energy / total_energy) * 100.0, 2)
        return energies
