"""
TeleWave: Hybrid Wavelet + Deep Learning Framework for Climate & Rainfall Forecasting.
"""

from climatewave.wavelets.dwt import WaveletTransform
from climatewave.wavelets.spectral import WaveletSpectralAnalysis
from climatewave.data.loader import MinMaxScaler, ClimateDataLoader
from climatewave.models.wavelet_lstm import WaveletLSTMForecaster
from climatewave.models.baseline_lstm import BaselineLSTMForecaster
from climatewave.models.statistical import PersistenceForecaster, SeasonalClimatologyForecaster
from climatewave.evaluation.metrics import compute_all_metrics
from climatewave.evaluation.comparator import ModelBenchmarkHarness

__version__ = "1.0.0"
__author__ = "Adeola Johnson"

__all__ = [
    "WaveletTransform",
    "WaveletSpectralAnalysis",
    "MinMaxScaler",
    "ClimateDataLoader",
    "WaveletLSTMForecaster",
    "BaselineLSTMForecaster",
    "PersistenceForecaster",
    "SeasonalClimatologyForecaster",
    "compute_all_metrics",
    "ModelBenchmarkHarness",
]
