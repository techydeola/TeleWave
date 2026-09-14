"""
Unit tests for evaluation metrics and comparator.
"""

import unittest
from climatewave.evaluation.metrics import (
    rmse,
    mae,
    mape,
    r2_score,
    nash_sutcliffe,
    peak_capture_index,
    compute_all_metrics,
)


class TestEvaluation(unittest.TestCase):

    def test_metrics_perfect_prediction(self):
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [1.0, 2.0, 3.0, 4.0, 5.0]

        self.assertAlmostEqual(rmse(y_true, y_pred), 0.0)
        self.assertAlmostEqual(mae(y_true, y_pred), 0.0)
        self.assertAlmostEqual(mape(y_true, y_pred), 0.0)
        self.assertAlmostEqual(r2_score(y_true, y_pred), 1.0)
        self.assertAlmostEqual(nash_sutcliffe(y_true, y_pred), 1.0)

    def test_metrics_imperfect_prediction(self):
        y_true = [10.0, 20.0, 30.0, 40.0]
        y_pred = [12.0, 18.0, 33.0, 38.0]

        scores = compute_all_metrics(y_true, y_pred)
        self.assertGreater(scores["RMSE"], 0.0)
        self.assertGreater(scores["MAE"], 0.0)
        self.assertGreater(scores["R2"], 0.8)


if __name__ == "__main__":
    unittest.main()
