"""
Unit tests for data loading and preprocessing.
"""

import unittest
from climatewave.data.loader import MinMaxScaler, ClimateDataLoader


class TestData(unittest.TestCase):

    def test_minmax_scaler(self):
        scaler = MinMaxScaler()
        raw = [10.0, 20.0, 30.0, 40.0, 50.0]
        scaled = scaler.fit_transform(raw)

        self.assertAlmostEqual(min(scaled), 0.0)
        self.assertAlmostEqual(max(scaled), 1.0)

        inversed = scaler.inverse_transform(scaled)
        for r, inv in zip(raw, inversed):
            self.assertAlmostEqual(r, inv, places=4)

    def test_load_teleconnections(self):
        data = ClimateDataLoader.load_teleconnections_monthly()
        self.assertGreater(data["total_records"], 700)
        self.assertIn("NINA", data["features"])
        self.assertIn("AMO", data["features"])
        self.assertEqual(len(data["precipitation"]), data["total_records"])

    def test_sliding_windows(self):
        series = list(range(100))
        X, Y = ClimateDataLoader.create_sliding_windows(series, lookback=12, horizon=1)
        self.assertEqual(len(X), 100 - 12 - 1 + 1)
        self.assertEqual(len(X[0]), 12)
        self.assertEqual(len(Y[0]), 1)


if __name__ == "__main__":
    unittest.main()
