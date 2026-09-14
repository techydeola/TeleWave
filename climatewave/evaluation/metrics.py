"""
Scientific evaluation metrics for hydrological, climate, and time-series forecasting.
"""

import math
from typing import List, Dict, Any, Union


def rmse(y_true: List[float], y_pred: List[float]) -> float:
    """Root Mean Squared Error (RMSE)."""
    N = len(y_true)
    if N == 0:
        return 0.0
    mse = sum((y_t - y_p) ** 2 for y_t, y_p in zip(y_true, y_pred)) / N
    return math.sqrt(mse)


def mae(y_true: List[float], y_pred: List[float]) -> float:
    """Mean Absolute Error (MAE)."""
    N = len(y_true)
    if N == 0:
        return 0.0
    return sum(abs(y_t - y_p) for y_t, y_p in zip(y_true, y_pred)) / N


def mape(y_true: List[float], y_pred: List[float], eps: float = 1e-5) -> float:
    """Mean Absolute Percentage Error (MAPE)."""
    N = len(y_true)
    if N == 0:
        return 0.0
    return (sum(abs((y_t - y_p) / (abs(y_t) + eps)) for y_t, y_p in zip(y_true, y_pred)) / N) * 100.0


def r2_score(y_true: List[float], y_pred: List[float]) -> float:
    """Coefficient of Determination (R^2)."""
    N = len(y_true)
    if N == 0:
        return 0.0
    mean_true = sum(y_true) / N
    ss_tot = sum((y - mean_true) ** 2 for y in y_true)
    ss_res = sum((y_t - y_p) ** 2 for y_t, y_p in zip(y_true, y_pred))
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - (ss_res / ss_tot)


def nash_sutcliffe(y_true: List[float], y_pred: List[float]) -> float:
    """Nash-Sutcliffe Efficiency (NSE) for hydrological rainfall-runoff evaluation."""
    return r2_score(y_true, y_pred)


def peak_capture_index(y_true: List[float], y_pred: List[float], top_percentile: float = 90.0) -> float:
    """
    Evaluates forecasting accuracy specifically on extreme heavy rainfall peaks (top 10% events).
    """
    if not y_true:
        return 1.0
    sorted_vals = sorted(y_true)
    idx = int((top_percentile / 100.0) * len(sorted_vals))
    threshold = sorted_vals[min(idx, len(sorted_vals) - 1)]

    peak_true, peak_pred = [], []
    for yt, yp in zip(y_true, y_pred):
        if yt >= threshold:
            peak_true.append(yt)
            peak_pred.append(yp)

    if not peak_true:
        return 1.0

    mean_err = sum(abs(yt - yp) / (abs(yt) + 1e-5) for yt, yp in zip(peak_true, peak_pred)) / len(peak_true)
    return max(0.0, 1.0 - mean_err)


def compute_all_metrics(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
    """Computes comprehensive score dictionary for a forecast."""
    return {
        "RMSE": round(rmse(y_true, y_pred), 4),
        "MAE": round(mae(y_true, y_pred), 4),
        "MAPE": round(mape(y_true, y_pred), 2),
        "R2": round(r2_score(y_true, y_pred), 4),
        "NSE": round(nash_sutcliffe(y_true, y_pred), 4),
        "PCI": round(peak_capture_index(y_true, y_pred), 4),
    }
