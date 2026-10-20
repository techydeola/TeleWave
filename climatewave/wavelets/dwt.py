"""
Discrete Wavelet Transform (DWT), Inverse DWT (IDWT), and Multiresolution Analysis (MRA).
High-performance vectorized implementation using NumPy for dyadic filtering and multiresolution decomposition,
guaranteeing mathematical equivalence and energy conservation.
"""

import math
from typing import List, Tuple, Dict, Any, Union
import numpy as np
from climatewave.wavelets.filters import get_wavelet_filters


def convolve_downsample(
    signal: Union[List[float], np.ndarray],
    filter_kernel: Union[List[float], np.ndarray],
    step: int = 2,
) -> List[float]:
    """1D circular convolution followed by dyadic downsampling (decimation by step) using NumPy."""
    sig = np.asarray(signal, dtype=np.float64)
    kernel = np.asarray(filter_kernel, dtype=np.float64)
    N = len(sig)
    K = len(kernel)
    out_indices = np.arange(0, N, step)
    gather_idx = (out_indices[:, None] + np.arange(K)[None, :]) % N
    result = np.dot(sig[gather_idx], kernel)
    return result.tolist()


def upsample_convolve(
    coeffs: Union[List[float], np.ndarray],
    filter_kernel: Union[List[float], np.ndarray],
    target_len: int,
) -> List[float]:
    """Dyadic upsampling (inserting zeros) followed by convolution using NumPy."""
    c = np.asarray(coeffs, dtype=np.float64)
    kernel = np.asarray(filter_kernel, dtype=np.float64)
    upsampled = np.zeros(len(c) * 2, dtype=np.float64)
    upsampled[::2] = c
    N = len(upsampled)
    i_idx = np.arange(target_len)[:, None]
    k_idx = np.arange(len(kernel))[None, :]
    idx = (i_idx - k_idx) % N
    result = np.dot(upsampled[idx], kernel)
    return result.tolist()


class WaveletTransform:
    """Core multiresolution wavelet transform engine using NumPy vectorization."""

    def __init__(self, wavelet: str = "db4"):
        self.wavelet_name = wavelet
        h_dec, g_dec, h_rec, g_rec = get_wavelet_filters(wavelet)
        self.h_dec = np.asarray(h_dec, dtype=np.float64)
        self.g_dec = np.asarray(g_dec, dtype=np.float64)
        self.h_rec = np.asarray(h_rec, dtype=np.float64)
        self.g_rec = np.asarray(g_rec, dtype=np.float64)

    def dwt_single_level(self, signal: Union[List[float], np.ndarray]) -> Tuple[List[float], List[float]]:
        """Decomposes a 1D signal into approximation (cA) and detail (cD) coefficients."""
        sig = np.asarray(signal, dtype=np.float64)
        # Ensure even length by repeating last element if odd
        if len(sig) % 2 != 0:
            sig = np.append(sig, sig[-1])

        N = len(sig)
        K = len(self.h_dec)
        half_len = N // 2
        i_idx = np.arange(half_len)[:, None]
        k_idx = np.arange(K)[None, :]
        idx = (2 * i_idx + k_idx) % N
        samples = sig[idx]

        cA = np.dot(samples, self.h_dec)
        cD = np.dot(samples, self.g_dec)
        return cA.tolist(), cD.tolist()

    def idwt_single_level(
        self,
        cA: Union[List[float], np.ndarray],
        cD: Union[List[float], np.ndarray],
        target_len: int,
    ) -> List[float]:
        """Reconstructs 1D signal from approximation (cA) and detail (cD) coefficients."""
        cA_arr = np.asarray(cA, dtype=np.float64)
        cD_arr = np.asarray(cD, dtype=np.float64)
        half_len = max(len(cA_arr), len(cD_arr))
        K = len(self.h_rec)
        M = half_len * 2

        val_a = np.zeros(half_len, dtype=np.float64)
        val_d = np.zeros(half_len, dtype=np.float64)
        val_a[: len(cA_arr)] = cA_arr
        val_d[: len(cD_arr)] = cD_arr

        terms = val_a[:, None] * self.h_rec[None, :] + val_d[:, None] * self.g_rec[None, :]
        i_idx = np.arange(half_len)[:, None]
        k_idx = np.arange(K)[None, :]
        idx = (2 * i_idx + k_idx) % M

        rec = np.zeros(M, dtype=np.float64)
        np.add.at(rec, idx.ravel(), terms.ravel())
        return rec[:target_len].tolist()

    def wavedec(self, signal: Union[List[float], np.ndarray], level: int = 3) -> List[List[float]]:
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
        # Return in standard order: [cA_n, cD_n, cD_n-1, ..., cD_1]
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

    def mra(self, signal: Union[List[float], np.ndarray], level: int = 3) -> Dict[str, List[float]]:
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

    def energy_distribution(self, signal: Union[List[float], np.ndarray], level: int = 3) -> Dict[str, float]:
        """
        Computes the relative wavelet energy percentage across decomposed frequency bands.
        """
        sig_arr = np.asarray(signal, dtype=np.float64)
        mra_dict = self.mra(signal, level=level)
        total_energy = float(np.sum(sig_arr ** 2))
        if total_energy == 0.0:
            total_energy = 1.0

        energies = {}
        for k, comp in mra_dict.items():
            comp_arr = np.asarray(comp, dtype=np.float64)
            comp_energy = float(np.sum(comp_arr ** 2))
            energies[k] = round((comp_energy / total_energy) * 100.0, 2)
        return energies
