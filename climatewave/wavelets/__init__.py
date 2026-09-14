"""
Wavelet analysis and multiresolution decomposition module for TeleWave.
"""

from climatewave.wavelets.filters import get_wavelet_filters, WAVELET_FILTERS
from climatewave.wavelets.dwt import WaveletTransform
from climatewave.wavelets.spectral import WaveletSpectralAnalysis

__all__ = [
    "get_wavelet_filters",
    "WAVELET_FILTERS",
    "WaveletTransform",
    "WaveletSpectralAnalysis",
]
