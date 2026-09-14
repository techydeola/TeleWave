"""
Classical statistical and climatological baseline forecasters.
"""

from typing import List, Dict, Any
from climatewave.models.base import BaseClimateForecaster


class PersistenceForecaster(BaseClimateForecaster):
    """
    Persistence / Naive baseline: Forecasts the most recent observed value forward:
    y_hat_{t+1} = y_t.
    """

    def __init__(self, name: str = "Persistence-Baseline"):
        super().__init__(name=name)

    def fit(self, X_train: List[List[List[float]]], y_train: List[List[float]], **kwargs) -> Dict[str, Any]:
        self.is_fitted = True
        return {"status": "persistence_fitted"}

    def predict(self, X: List[List[List[float]]]) -> List[List[float]]:
        # Returns the last time-step of the primary target feature
        return [[seq[-1][0]] for seq in X]


class SeasonalClimatologyForecaster(BaseClimateForecaster):
    """
    Seasonal Climatology baseline: Forecasts the historical mean for the corresponding season/month.
    """

    def __init__(self, name: str = "Seasonal-Climatology", period: int = 12):
        super().__init__(name=name)
        self.period = period
        self.seasonal_means = [0.0] * period

    def fit(self, X_train: List[List[List[float]]], y_train: List[List[float]], **kwargs) -> Dict[str, Any]:
        sums = [0.0] * self.period
        counts = [0] * self.period

        for idx, y_val in enumerate(y_train):
            m = idx % self.period
            sums[m] += y_val[0]
            counts[m] += 1

        self.seasonal_means = [sums[m] / counts[m] if counts[m] > 0 else 0.0 for m in range(self.period)]
        self.is_fitted = True
        return {"seasonal_means": self.seasonal_means}

    def predict(self, X: List[List[List[float]]]) -> List[List[float]]:
        preds = []
        for idx in range(len(X)):
            m = idx % self.period
            preds.append([self.seasonal_means[m]])
        return preds
