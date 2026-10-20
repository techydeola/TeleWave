"""
Scientific evaluation metrics for hydrological, climate, and time-series forecasting.
Vectorized implementation using NumPy for high-throughput evaluation.
"""

import math
from typing import List, Dict, Any, Union
import numpy as np


def rmse(y_true: Union[List[float], np.ndarray], y_pred: Union[List[float], np.ndarray]) -> float:
    """Root Mean Squared Error (RMSE)."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.size == 0:
        return 0.0
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mae(y_true: Union[List[float], np.ndarray], y_pred: Union[List[float], np.ndarray]) -> float:
    """Mean Absolute Error (MAE)."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.size == 0:
        return 0.0
    return float(np.mean(np.abs(yt - yp)))


def mape(y_true: Union[List[float], np.ndarray], y_pred: Union[List[float], np.ndarray], eps: float = 1e-5) -> float:
    """Mean Absolute Percentage Error (MAPE)."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.size == 0:
        return 0.0
    return float(np.mean(np.abs((yt - yp) / (np.abs(yt) + eps))) * 100.0)


def r2_score(y_true: Union[List[float], np.ndarray], y_pred: Union[List[float], np.ndarray]) -> float:
    """Coefficient of Determination (R^2)."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.size == 0:
        return 0.0
    mean_true = np.mean(yt)
    ss_tot = float(np.sum((yt - mean_true) ** 2))
    ss_res = float(np.sum((yt - yp) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return float(1.0 - (ss_res / ss_tot))


def nash_sutcliffe(y_true: Union[List[float], np.ndarray], y_pred: Union[List[float], np.ndarray]) -> float:
    """Nash-Sutcliffe Efficiency (NSE) for hydrological rainfall-runoff evaluation."""
    return r2_score(y_true, y_pred)


def peak_capture_index(
    y_true: Union[List[float], np.ndarray],
    y_pred: Union[List[float], np.ndarray],
    top_percentile: float = 90.0,
) -> float:
    """
    Evaluates forecasting accuracy specifically on extreme heavy rainfall peaks (top 10% events).
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.size == 0:
        return 1.0
    sorted_vals = np.sort(yt)
    idx = int((top_percentile / 100.0) * len(sorted_vals))
    threshold = sorted_vals[min(idx, len(sorted_vals) - 1)]

    mask = yt >= threshold
    if not np.any(mask):
        return 1.0

    peak_true = yt[mask]
    peak_pred = yp[mask]
    mean_err = float(np.mean(np.abs(peak_true - peak_pred) / (np.abs(peak_true) + 1e-5)))
    return float(max(0.0, 1.0 - mean_err))


def compute_all_metrics(
    y_true: Union[List[float], np.ndarray],
    y_pred: Union[List[float], np.ndarray],
) -> Dict[str, float]:
    """Computes comprehensive score dictionary for a forecast."""
    return {
        "RMSE": round(rmse(y_true, y_pred), 4),
        "MAE": round(mae(y_true, y_pred), 4),
        "MAPE": round(mape(y_true, y_pred), 2),
        "R2": round(r2_score(y_true, y_pred), 4),
        "NSE": round(nash_sutcliffe(y_true, y_pred), 4),
        "PCI": round(peak_capture_index(y_true, y_pred), 4),
    }
