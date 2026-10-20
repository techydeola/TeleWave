"""
Data loading, parsing, normalization, and sliding-window sequence generators for TeleWave.
"""

import os
from typing import List, Dict, Any, Tuple, Optional, Union
import numpy as np
import pandas as pd


class SlidingWindowArray(np.ndarray):
    """
    NumPy ndarray subclass providing Python container protocol compatibility
    (truthiness and list equality) so unrefactored downstream modules work seamlessly.
    """

    def __new__(cls, input_array):
        return np.asanyarray(input_array).view(cls)

    def __bool__(self) -> bool:
        if self.ndim == 0:
            return bool(self.item())
        return len(self) > 0

    def __eq__(self, other):
        if isinstance(other, (list, tuple)):
            other_arr = np.asarray(other)
            if self.shape != other_arr.shape:
                return False
            return bool(np.allclose(np.asarray(self), other_arr, atol=1e-9))
        return super().__eq__(other)


class MinMaxScaler:
    """Feature normalization to [0, 1] range using NumPy vectorization."""

    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0)):
        self.feature_range = feature_range
        self.min_val = 0.0
        self.max_val = 1.0

    def fit(self, data: Union[List[float], np.ndarray]) -> "MinMaxScaler":
        arr = np.asarray(data, dtype=np.float64)
        if arr.size == 0:
            return self
        self.min_val = float(np.min(arr))
        self.max_val = float(np.max(arr))
        if self.min_val == self.max_val:
            self.max_val += 1e-6
        return self

    def transform(self, data: Union[List[float], np.ndarray]) -> SlidingWindowArray:
        arr = np.asarray(data, dtype=np.float64)
        if arr.size == 0:
            return SlidingWindowArray(np.array([], dtype=np.float64))
        r_min, r_max = self.feature_range
        scale = (r_max - r_min) / (self.max_val - self.min_val)
        return SlidingWindowArray(r_min + (arr - self.min_val) * scale)

    def fit_transform(self, data: Union[List[float], np.ndarray]) -> SlidingWindowArray:
        return self.fit(data).transform(data)

    def inverse_transform(self, data: Union[List[float], np.ndarray]) -> SlidingWindowArray:
        arr = np.asarray(data, dtype=np.float64)
        if arr.size == 0:
            return SlidingWindowArray(np.array([], dtype=np.float64))
        r_min, r_max = self.feature_range
        scale = (self.max_val - self.min_val) / (r_max - r_min)
        return SlidingWindowArray(self.min_val + (arr - r_min) * scale)


class ClimateDataLoader:
    """Loads and formats meteorological datasets for training and forecasting using Pandas & NumPy."""

    @staticmethod
    def get_default_data_dir() -> str:
        base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return os.path.join(base, "data", "raw")

    @classmethod
    def load_teleconnections_monthly(cls, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Loads the Nigeria Monthly Precipitation + Teleconnections (1961 - 2022) dataset.
        Columns: Date, Prec, NINA, AMM, SOI, PDO, NAO, AMO
        """
        path = file_path or os.path.join(cls.get_default_data_dir(), "nigeria_teleconnections_monthly_1961_2022.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found at: {path}")

        df = pd.read_csv(path, encoding="utf-8-sig")
        dates = df["Date"].astype(str).str.strip().tolist()
        prec = df["Prec"].to_numpy(dtype=np.float64)
        nina = df["NINA"].to_numpy(dtype=np.float64)
        amm = df["AMM"].to_numpy(dtype=np.float64)
        soi = df["SOI"].to_numpy(dtype=np.float64)
        pdo = df["PDO"].to_numpy(dtype=np.float64)
        nao = df["NAO"].to_numpy(dtype=np.float64)
        amo = df["AMO"].to_numpy(dtype=np.float64)

        n_pts = len(dates)
        month_indices = np.arange(n_pts) % 12
        sin_month = np.sin(2.0 * np.pi * month_indices / 12.0)
        cos_month = np.cos(2.0 * np.pi * month_indices / 12.0)

        return {
            "dates": dates,
            "precipitation": SlidingWindowArray(prec),
            "features": {
                "NINA": SlidingWindowArray(nina),
                "AMM": SlidingWindowArray(amm),
                "SOI": SlidingWindowArray(soi),
                "PDO": SlidingWindowArray(pdo),
                "NAO": SlidingWindowArray(nao),
                "AMO": SlidingWindowArray(amo),
                "sin_month": SlidingWindowArray(sin_month),
                "cos_month": SlidingWindowArray(cos_month),
            },
            "total_records": n_pts,
        }

    @classmethod
    def load_daily_meteorology(cls, station: str = "ondo", file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Loads daily meteorological observations (Ondo or Calabar, 1991 - 2021).
        Columns: Date, u10, v10, mx2t, mxtpr, mn2t, mntpr, tp
        """
        fname = f"{station.lower()}_daily_meteorology_1991_2021.csv"
        path = file_path or os.path.join(cls.get_default_data_dir(), fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Station file not found: {path}")

        df = pd.read_csv(path, encoding="utf-8-sig")
        dates = df["Date"].astype(str).str.strip().tolist()
        u10 = df["u10"].to_numpy(dtype=np.float64) if "u10" in df else np.zeros(len(dates), dtype=np.float64)
        v10 = df["v10"].to_numpy(dtype=np.float64) if "v10" in df else np.zeros(len(dates), dtype=np.float64)
        mx2t = df["mx2t"].to_numpy(dtype=np.float64) if "mx2t" in df else np.full(len(dates), 300.0, dtype=np.float64)
        mn2t = df["mn2t"].to_numpy(dtype=np.float64) if "mn2t" in df else np.full(len(dates), 295.0, dtype=np.float64)
        tp = df["tp"].to_numpy(dtype=np.float64) if "tp" in df else np.zeros(len(dates), dtype=np.float64)

        return {
            "station": station,
            "dates": dates,
            "precipitation": SlidingWindowArray(tp),
            "features": {
                "u10": SlidingWindowArray(u10),
                "v10": SlidingWindowArray(v10),
                "mx2t": SlidingWindowArray(mx2t),
                "mn2t": SlidingWindowArray(mn2t),
            },
            "total_records": len(dates),
        }

    @staticmethod
    def create_sliding_windows(
        series: Union[List[float], np.ndarray],
        exog_features: Optional[Union[List[List[float]], np.ndarray, List[np.ndarray]]] = None,
        lookback: int = 12,
        horizon: int = 1,
    ) -> Tuple[SlidingWindowArray, SlidingWindowArray]:
        """
        Creates (X, Y) sliding window sequences for temporal neural network training using NumPy strided windows.
        X shape: (N_samples, lookback, num_features)
        Y shape: (N_samples, horizon)
        """
        s = np.asarray(series, dtype=np.float64)
        N = len(s)
        n_samples = N - lookback - horizon + 1

        if exog_features is not None and len(exog_features) > 0:
            exog_arr = np.asarray(exog_features, dtype=np.float64)
            if exog_arr.ndim == 2:
                if exog_arr.shape[1] == N and exog_arr.shape[0] != N:
                    exog_cols = exog_arr.T
                elif exog_arr.shape[0] == N:
                    exog_cols = exog_arr
                else:
                    raise ValueError(f"Exogenous features dimension {exog_arr.shape} does not match series length {N}")
            elif exog_arr.ndim == 1:
                if len(exog_arr) != N:
                    raise ValueError(f"Exogenous feature length {len(exog_arr)} does not match series length {N}")
                exog_cols = exog_arr[:, np.newaxis]
            else:
                raise ValueError(f"Exogenous features must be 1D or 2D, got shape {exog_arr.shape}")
            full_data = np.column_stack([s, exog_cols])
        else:
            full_data = s[:, np.newaxis]

        num_features = full_data.shape[1]

        if n_samples <= 0:
            return (
                SlidingWindowArray(np.empty((0, lookback, num_features), dtype=np.float64)),
                SlidingWindowArray(np.empty((0, horizon), dtype=np.float64)),
            )

        X = np.lib.stride_tricks.sliding_window_view(full_data, window_shape=lookback, axis=0).swapaxes(1, 2)[:n_samples].copy()
        Y = np.lib.stride_tricks.sliding_window_view(s[lookback:], window_shape=horizon, axis=0)[:n_samples].copy()

        return SlidingWindowArray(X), SlidingWindowArray(Y)

    @staticmethod
    def train_test_split(
        X: Union[List[Any], np.ndarray],
        Y: Union[List[Any], np.ndarray],
        train_ratio: float = 0.8,
    ) -> Tuple[Union[List[Any], np.ndarray], Union[List[Any], np.ndarray], Union[List[Any], np.ndarray], Union[List[Any], np.ndarray]]:
        """Temporal train-test split maintaining chronological order."""
        split_idx = int(len(X) * train_ratio)
        return X[:split_idx], Y[:split_idx], X[split_idx:], Y[split_idx:]
