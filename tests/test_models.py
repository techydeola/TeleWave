"""
Unit tests for forecasting models.
"""

import unittest
from climatewave.models.baseline_lstm import BaselineLSTMForecaster
from climatewave.models.wavelet_lstm import WaveletLSTMForecaster
from climatewave.models.statistical import PersistenceForecaster, SeasonalClimatologyForecaster


class TestModels(unittest.TestCase):

    def setUp(self):
        # Synthetic (X, y) sliding windows
        self.X_train = [[[float(i + t)]] for i in range(40) for t in range(12)]
        # Re-shape into [40, 12, 1]
        self.X_train = [[ [float(i + t)] for t in range(12) ] for i in range(40)]
        self.y_train = [[float(i + 12)] for i in range(40)]

        self.X_test = [[ [float(i + t)] for t in range(12) ] for i in range(40, 50)]
        self.y_test = [[float(i + 12)] for i in range(40, 50)]

    def test_baseline_lstm_fit_predict(self):
        model = BaselineLSTMForecaster(hidden_dim=8)
        res = model.fit(self.X_train, self.y_train, epochs=5)
        self.assertTrue(model.is_fitted)
        self.assertGreater(len(res["loss_history"]), 0)

        preds = model.predict(self.X_test)
        self.assertEqual(len(preds), len(self.X_test))

    def test_wavelet_lstm_fit_predict(self):
        model = WaveletLSTMForecaster(wavelet="db4", decomposition_level=2, hidden_dim=8)
        res = model.fit(self.X_train, self.y_train, epochs=5)
        self.assertTrue(model.is_fitted)
        self.assertIn("branches", res)

        preds = model.predict(self.X_test)
        self.assertEqual(len(preds), len(self.X_test))

    def test_persistence_baseline(self):
        model = PersistenceForecaster()
        model.fit(self.X_train, self.y_train)
        preds = model.predict(self.X_test)
        # Persistence returns last element of sequence
        self.assertEqual(preds[0][0], self.X_test[0][-1][0])


if __name__ == "__main__":
    unittest.main()
