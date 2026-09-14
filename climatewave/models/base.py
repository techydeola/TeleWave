"""
Base model interface for climate time-series forecasters.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple


class BaseClimateForecaster(ABC):
    """Abstract interface for climate and precipitation predictive models."""

    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False

    @abstractmethod
    def fit(self, X_train: List[List[List[float]]], y_train: List[List[float]], **kwargs) -> Dict[str, Any]:
        """Fit the model to historical time-series sequences."""
        pass

    @abstractmethod
    def predict(self, X: List[List[List[float]]]) -> List[List[float]]:
        """Generate forecasts for input time-series sequences."""
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name}>"
