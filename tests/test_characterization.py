"""
Characterization regression tests comparing TeleWave against saved golden outputs.
These tests verify that any refactored code (e.g. NumPy) maintains exact numerical
parity with the v1 pure standard library reference.
"""

import unittest
import json
import os
import math
from climatewave.data.loader import ClimateDataLoader, MinMaxScaler
from climatewave.wavelets.filters import get_wavelet_filters
from climatewave.wavelets.dwt import WaveletTransform
from climatewave.wavelets.spectral import WaveletSpectralAnalysis
from climatewave.evaluation.metrics import (
    rmse,
    mae,
    mape,
    r2_score,
    nash_sutcliffe,
    peak_capture_index,
    compute_all_metrics,
)
from climatewave.models.baseline_lstm import solve_ridge_regression, BaselineLSTMForecaster
from climatewave.models.wavelet_lstm import WaveletLSTMForecaster
from climatewave.models.statistical import PersistenceForecaster, SeasonalClimatologyForecaster

_HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN_DIR = os.path.join(_HERE, "golden")


class TestCharacterizationGolden(unittest.TestCase):
    """Tests against frozen golden JSON snapshots from pure-Python v1."""

    def test_metrics_golden(self):
        with open(os.path.join(GOLDEN_DIR, "golden_metrics.json"), "r") as f:
            gold = json.load(f)

        yt = gold["y_true"]
        yp = gold["y_pred"]

        self.assertAlmostEqual(rmse(yt, yp), gold["rmse"], places=6)
        self.assertAlmostEqual(mae(yt, yp), gold["mae"], places=6)
        self.assertAlmostEqual(mape(yt, yp), gold["mape"], places=6)
        self.assertAlmostEqual(r2_score(yt, yp), gold["r2"], places=6)
        self.assertAlmostEqual(nash_sutcliffe(yt, yp), gold["nse"], places=6)
        self.assertAlmostEqual(peak_capture_index(yt, yp), gold["pci"], places=6)
        self.assertEqual(compute_all_metrics(yt, yp), gold["compute_all_metrics"])

    def test_scaler_and_windowing_golden(self):
        with open(os.path.join(GOLDEN_DIR, "golden_scaler_windowing.json"), "r") as f:
            gold = json.load(f)

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(gold["scaler_raw"])
        inversed = scaler.inverse_transform(scaled)

        for s, g in zip(scaled, gold["scaler_scaled"]):
            self.assertAlmostEqual(s, g, delta=1e-9)
        for inv, g in zip(inversed, gold["scaler_inversed"]):
            self.assertAlmostEqual(inv, g, delta=1e-9)

        self.assertAlmostEqual(scaler.min_val, gold["scaler_min_val"], delta=1e-9)
        self.assertAlmostEqual(scaler.max_val, gold["scaler_max_val"], delta=1e-9)

        # Sliding window test
        seq_series = [float(i) for i in range(20)]
        seq_exog = [[float(i * 2) for i in range(20)], [float(i * -1) for i in range(20)]]
        X_win, Y_win = ClimateDataLoader.create_sliding_windows(seq_series, exog_features=seq_exog, lookback=4, horizon=2)
        self.assertEqual(X_win, gold["sliding_window_X"])
        self.assertEqual(Y_win, gold["sliding_window_Y"])

        X_tr, Y_tr, X_te, Y_te = ClimateDataLoader.train_test_split(X_win, Y_win, train_ratio=0.7)
        self.assertEqual(len(X_tr), gold["train_test_split"]["len_X_train"])
        self.assertEqual(len(X_te), gold["train_test_split"]["len_X_test"])
        self.assertEqual(X_tr[0], gold["train_test_split"]["X_train_0"])
        self.assertEqual(Y_tr[0], gold["train_test_split"]["Y_train_0"])

    def test_wavelets_golden(self):
        with open(os.path.join(GOLDEN_DIR, "golden_wavelets.json"), "r") as f:
            gold = json.load(f)

        sig = gold["signal"]
        for wname, expected in gold["results"].items():
            h_dec, g_dec, h_rec, g_rec = get_wavelet_filters(wname)
            self.assertEqual(h_dec, expected["filters"]["h_dec"])
            self.assertEqual(g_dec, expected["filters"]["g_dec"])

            wt = WaveletTransform(wavelet=wname)
            cA, cD = wt.dwt_single_level(sig)
            for a, exp_a in zip(cA, expected["cA"]):
                self.assertAlmostEqual(a, exp_a, places=6)
            for d, exp_d in zip(cD, expected["cD"]):
                self.assertAlmostEqual(d, exp_d, places=6)

            rec = wt.idwt_single_level(cA, cD, target_len=len(sig))
            for r, exp_r in zip(rec, expected["rec_single"]):
                self.assertAlmostEqual(r, exp_r, places=6)

            mra = wt.mra(sig, level=2)
            for comp_k, comp_v in mra.items():
                for v, exp_v in zip(comp_v, expected["mra"][comp_k]):
                    self.assertAlmostEqual(v, exp_v, places=6)

            energies = wt.energy_distribution(sig, level=2)
            self.assertEqual(energies, expected["energies"])

    def test_spectral_cwt_golden(self):
        with open(os.path.join(GOLDEN_DIR, "golden_spectral.json"), "r") as f:
            gold = json.load(f)

        sig = gold["signal"]
        spec = WaveletSpectralAnalysis.compute_power_spectrum(sig, dt=1.0, num_scales=8, s0=2.0, dj=0.5)

        self.assertEqual(spec["dominant_period"], gold["spectrum"]["dominant_period"])
        self.assertEqual(spec["periods"], gold["spectrum"]["periods"])
        self.assertEqual(spec["scales"], gold["spectrum"]["scales"])
        self.assertEqual(spec["global_power"], gold["spectrum"]["global_power"])

    def test_ridge_regression_golden(self):
        with open(os.path.join(GOLDEN_DIR, "golden_ridge.json"), "r") as f:
            gold = json.load(f)

        X = gold["X"]
        y = gold["y"]
        w_1e4 = solve_ridge_regression(X, y, lambda_reg=1e-4)
        w_01 = solve_ridge_regression(X, y, lambda_reg=0.01)

        for w, exp_w in zip(w_1e4, gold["weights_1e4"]):
            self.assertAlmostEqual(w, exp_w, places=6)
        for w, exp_w in zip(w_01, gold["weights_01"]):
            self.assertAlmostEqual(w, exp_w, places=6)

    def test_statistical_baselines_golden(self):
        with open(os.path.join(GOLDEN_DIR, "golden_statistical_models.json"), "r") as f:
            gold = json.load(f)

        stat_X = [
            [[1.0], [2.0], [3.0]],
            [[2.0], [3.0], [4.0]],
            [[3.0], [4.0], [5.0]],
            [[4.0], [5.0], [6.0]],
        ]
        stat_y = [[4.0], [5.0], [6.0], [7.0]]

        pers = PersistenceForecaster()
        pers.fit(stat_X, stat_y)
        self.assertEqual(pers.predict(stat_X), gold["persistence_preds"])

        seas = SeasonalClimatologyForecaster(period=2)
        seas.fit(stat_X, stat_y)
        self.assertEqual(seas.seasonal_means, gold["seasonal_means"])
        self.assertEqual(seas.predict(stat_X), gold["seasonal_preds"])

    def test_wavelet_lstm_deterministic_golden(self):
        with open(os.path.join(GOLDEN_DIR, "golden_wavelet_lstm.json"), "r") as f:
            gold = json.load(f)

        X_wl = [[[float(i + t), float((i+t)*0.5)] for t in range(8)] for i in range(12)]
        y_wl = [[float(i + 8)] for i in range(12)]

        w_lstm = WaveletLSTMForecaster(wavelet="db4", decomposition_level=2, hidden_dim=4, seed=42)
        features_0 = w_lstm._extract_wavelet_features(X_wl[0])
        for f_val, exp_val in zip(features_0, gold["sample_feature_vector_0"]):
            self.assertAlmostEqual(f_val, exp_val, places=6)

        w_lstm.fit(X_wl, y_wl)
        for w_val, exp_w in zip(w_lstm.fusion_weights, gold["fusion_weights"]):
            self.assertAlmostEqual(w_val, exp_w, places=6)

        preds = w_lstm.predict(X_wl)
        for p, exp_p in zip(preds, gold["predictions"]):
            self.assertAlmostEqual(p[0], exp_p[0], places=6)

    def test_baseline_lstm_injected_weights_golden(self):
        """
        Tests LSTM cell recurrence forward equations with explicit injected weights.
        This isolates the mathematical recurrence from PRNG implementation differences.
        """
        with open(os.path.join(GOLDEN_DIR, "golden_baseline_lstm.json"), "r") as f:
            gold = json.load(f)["injected_weights"]

        b_injected = BaselineLSTMForecaster(hidden_dim=4, seed=0)
        b_injected.input_dim = 2
        b_injected.weights = gold["weights"]

        hidden = b_injected._extract_hidden_features(gold["sample_input"])
        for h_val, exp_h in zip(hidden, gold["expected_hidden_features_0"]):
            self.assertAlmostEqual(h_val, exp_h, places=6)


if __name__ == "__main__":
    unittest.main()
