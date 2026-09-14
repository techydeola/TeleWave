"""
Predictive and forecasting models for TeleWave.
"""

from climatewave.models.base import BaseClimateForecaster
from climatewave.models.baseline_lstm import BaselineLSTMForecaster
from climatewave.models.wavelet_lstm import WaveletLSTMForecaster
from climatewave.models.statistical import PersistenceForecaster, SeasonalClimatologyForecaster

__all__ = [
    "BaseClimateForecaster",
    "BaselineLSTMForecaster",
    "WaveletLSTMForecaster",
    "PersistenceForecaster",
    "SeasonalClimatologyForecaster",
]
