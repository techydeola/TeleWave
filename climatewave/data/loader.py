"""
Data loading, parsing, normalization, and sliding-window sequence generators for TeleWave.
"""

import os
import csv
import math
from typing import List, Dict, Any, Tuple, Optional


class MinMaxScaler:
    """Feature normalization to [0, 1] range."""

    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0)):
        self.feature_range = feature_range
        self.min_val = 0.0
        self.max_val = 1.0

    def fit(self, data: List[float]) -> "MinMaxScaler":
        if not data:
            return self
        self.min_val = min(data)
        self.max_val = max(data)
        if self.min_val == self.max_val:
            self.max_val += 1e-6
        return self

    def transform(self, data: List[float]) -> List[float]:
        r_min, r_max = self.feature_range
        scale = (r_max - r_min) / (self.max_val - self.min_val)
        return [r_min + (x - self.min_val) * scale for x in data]

    def fit_transform(self, data: List[float]) -> List[float]:
        return self.fit(data).transform(data)

    def inverse_transform(self, data: List[float]) -> List[float]:
        r_min, r_max = self.feature_range
        scale = (self.max_val - self.min_val) / (r_max - r_min)
        return [self.min_val + (x - r_min) * scale for x in data]


class ClimateDataLoader:
    """Loads and formats meteorological datasets for training and forecasting."""

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

        dates, prec, nina, amm, soi, pdo, nao, amo = [], [], [], [], [], [], [], []

        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dates.append(row.get("Date", "").strip())
                prec.append(float(row.get("Prec", 0.0)))
                nina.append(float(row.get("NINA", 0.0)))
                amm.append(float(row.get("AMM", 0.0)))
                soi.append(float(row.get("SOI", 0.0)))
                pdo.append(float(row.get("PDO", 0.0)))
                nao.append(float(row.get("NAO", 0.0)))
                amo.append(float(row.get("AMO", 0.0)))

        N_pts = len(dates)
        sin_month = [math.sin(2.0 * math.pi * (i % 12) / 12.0) for i in range(N_pts)]
        cos_month = [math.cos(2.0 * math.pi * (i % 12) / 12.0) for i in range(N_pts)]

        return {
            "dates": dates,
            "precipitation": prec,
            "features": {
                "NINA": nina,
                "AMM": amm,
                "SOI": soi,
                "PDO": pdo,
                "NAO": nao,
                "AMO": amo,
                "sin_month": sin_month,
                "cos_month": cos_month,
            },
            "total_records": len(dates),
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

        dates, u10, v10, mx2t, mn2t, tp = [], [], [], [], [], []

        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dates.append(row.get("Date", "").strip())
                u10.append(float(row.get("u10", 0.0)))
                v10.append(float(row.get("v10", 0.0)))
                mx2t.append(float(row.get("mx2t", 300.0)))
                mn2t.append(float(row.get("mn2t", 295.0)))
                tp.append(float(row.get("tp", 0.0)))

        return {
            "station": station,
            "dates": dates,
            "precipitation": tp,
            "features": {
                "u10": u10,
                "v10": v10,
                "mx2t": mx2t,
                "mn2t": mn2t,
            },
            "total_records": len(dates),
        }

    @staticmethod
    def create_sliding_windows(
        series: List[float],
        exog_features: Optional[List[List[float]]] = None,
        lookback: int = 12,
        horizon: int = 1,
    ) -> Tuple[List[List[List[float]]], List[List[float]]]:
        """
        Creates (X, Y) sliding window sequences for temporal neural network training.
        X shape: [N_samples, lookback, num_features]
        Y shape: [N_samples, horizon]
        """
        N = len(series)
        X, Y = [], []

        num_exog = len(exog_features) if exog_features else 0

        for i in range(N - lookback - horizon + 1):
            # Target lag sequence
            window_steps = []
            for t in range(i, i + lookback):
                step_features = [series[t]]
                if exog_features:
                    for ex in exog_features:
                        step_features.append(ex[t])
                window_steps.append(step_features)

            target_steps = series[i + lookback : i + lookback + horizon]
            X.append(window_steps)
            Y.append(target_steps)

        return X, Y

    @staticmethod
    def train_test_split(
        X: List[Any], Y: List[Any], train_ratio: float = 0.8
    ) -> Tuple[List[Any], List[Any], List[Any], List[Any]]:
        """Temporal train-test split maintaining chronological order."""
        split_idx = int(len(X) * train_ratio)
        X_train, Y_train = X[:split_idx], Y[:split_idx]
        X_test, Y_test = X[split_idx:], Y[split_idx:]
        return X_train, Y_train, X_test, Y_test
