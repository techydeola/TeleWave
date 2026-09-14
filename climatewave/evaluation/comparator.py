"""
Comparative model benchmark harness.
Evaluates Wavelet-LSTM against standard deep learning and statistical baselines.
"""

from typing import List, Dict, Any
from climatewave.models.wavelet_lstm import WaveletLSTMForecaster
from climatewave.models.baseline_lstm import BaselineLSTMForecaster
from climatewave.models.statistical import PersistenceForecaster, SeasonalClimatologyForecaster
from climatewave.evaluation.metrics import compute_all_metrics


class ModelBenchmarkHarness:
    """Runs automated side-by-side performance benchmarks across model architectures."""

    @classmethod
    def run_comparison(
        cls,
        X_train: List[List[List[float]]],
        y_train: List[List[float]],
        X_test: List[List[List[float]]],
        y_test: List[List[float]],
        epochs: int = 20,
    ) -> Dict[str, Any]:
        y_test_flat = [y[0] for y in y_test]

        models = [
            WaveletLSTMForecaster(name="Wavelet-LSTM (Hybrid)", wavelet="db4", decomposition_level=2),
            BaselineLSTMForecaster(name="Standard LSTM", hidden_dim=16),
            SeasonalClimatologyForecaster(name="Seasonal Climatology"),
            PersistenceForecaster(name="Persistence Baseline"),
        ]

        leaderboard = []
        predictions = {}

        for m in models:
            m.fit(X_train, y_train, epochs=epochs)
            preds = m.predict(X_test)
            preds_flat = [p[0] for p in preds]
            predictions[m.name] = preds_flat

            score = compute_all_metrics(y_test_flat, preds_flat)
            score["Model"] = m.name
            leaderboard.append(score)

        # Sort leaderboard by lowest RMSE
        leaderboard.sort(key=lambda x: x["RMSE"])

        return {
            "leaderboard": leaderboard,
            "predictions": predictions,
            "y_true": y_test_flat,
            "best_model": leaderboard[0]["Model"],
        }
