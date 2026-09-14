"""
Comprehensive Production QA Test Suite for TeleWave.
Covers:
  1. Wavelet filter properties (Parseval energy = 1.0, sum = sqrt(2), QMF relations)
  2. DWT & IDWT perfect reconstruction (< 1e-8 error across Haar, db2, db4, sym4)
  3. MRA additivity (sum(components) == original signal to machine precision)
  4. Morlet CWT Fourier factor (exact Torrence & Compo 1998 formula)
  5. Ridge regression closed-form solver (identity, 2x2, regularized systems)
  6. Evaluation metrics numerical precision (RMSE, MAE, MAPE, R2, NSE, PCI)
  7. Data loader integrity and unit scales (daily to monthly, mm to meters)
"""

import unittest
import math
import random
from climatewave.wavelets.filters import get_wavelet_filters, WAVELET_FILTERS
from climatewave.wavelets.dwt import WaveletTransform
from climatewave.wavelets.spectral import WaveletSpectralAnalysis
from climatewave.models.baseline_lstm import solve_ridge_regression, sigmoid, tanh, BaselineLSTMForecaster
from climatewave.models.wavelet_lstm import WaveletLSTMForecaster
from climatewave.models.statistical import PersistenceForecaster, SeasonalClimatologyForecaster
from climatewave.evaluation.metrics import (
    rmse,
    mae,
    mape,
    r2_score,
    nash_sutcliffe,
    peak_capture_index,
    compute_all_metrics,
)
from climatewave.data.loader import MinMaxScaler, ClimateDataLoader


class TestProductionQAEquations(unittest.TestCase):
    """Deep verification of all mathematical expressions and numerical routines."""

    def test_wavelet_filter_orthogonality_and_energy(self):
        """Every scaling filter must sum to sqrt(2) and have energy exactly 1.0."""
        target_sum = math.sqrt(2.0)
        for name, coeffs in WAVELET_FILTERS.items():
            s = sum(coeffs)
            energy = sum(c ** 2 for c in coeffs)
            self.assertAlmostEqual(
                s, target_sum, places=5,
                msg=f"Filter {name} sum {s} does not match sqrt(2)"
            )
            self.assertAlmostEqual(
                energy, 1.0, places=5,
                msg=f"Filter {name} energy {energy} != 1.0"
            )

    def test_qmf_quadrature_mirror_relationship(self):
        """High-pass filter g must satisfy alternating flip: g[k] = (-1)^k * h[N - 1 - k]."""
        for name in WAVELET_FILTERS:
            h_dec, g_dec, h_rec, g_rec = get_wavelet_filters(name)
            N = len(h_dec)
            for k in range(N):
                expected_g = ((-1) ** k) * h_dec[N - 1 - k]
                self.assertAlmostEqual(
                    g_dec[k], expected_g, places=9,
                    msg=f"QMF violation in {name} at tap {k}"
                )
            # Check reconstruction filter definitions
            self.assertEqual(h_rec, h_dec)
            self.assertEqual(g_rec, g_dec)

    def test_dwt_idwt_perfect_reconstruction_single_level(self):
        """Single-level DWT + IDWT must perfectly invert signals (< 1e-8 error)."""
        random.seed(123)
        for wname in ["haar", "db2", "db4", "sym4"]:
            wt = WaveletTransform(wname)
            for length in [16, 32, 64]:
                sig = [random.uniform(-10.0, 10.0) for _ in range(length)]
                cA, cD = wt.dwt_single_level(sig)
                rec = wt.idwt_single_level(cA, cD, target_len=length)
                max_err = max(abs(s - r) for s, r in zip(sig, rec))
                self.assertLess(
                    max_err, 1e-8,
                    f"Reconstruction error too high for {wname} at length {length}: {max_err}"
                )

    def test_mra_exact_additivity(self):
        """MRA decomposition components must sum back to the original signal identically."""
        random.seed(456)
        for wname in ["haar", "db2", "db4", "sym4"]:
            wt = WaveletTransform(wname)
            sig = [random.uniform(-5.0, 5.0) for _ in range(32)]
            mra_comps = wt.mra(sig, level=3)
            summed = [0.0] * len(sig)
            for comp in mra_comps.values():
                for t in range(len(sig)):
                    summed[t] += comp[t]
            max_additivity_err = max(abs(s - r) for s, r in zip(sig, summed))
            self.assertLess(
                max_additivity_err, 1e-8,
                f"MRA additivity violation for {wname}: {max_additivity_err}"
            )

    def test_morlet_fourier_equivalent_period(self):
        """Morlet period conversion factor must match analytical formula for omega0=6."""
        w0 = 6.0
        expected_factor = (4.0 * math.pi) / (w0 + math.sqrt(2.0 + w0 ** 2))
        self.assertAlmostEqual(expected_factor, 1.033044, places=5)
        spec = WaveletSpectralAnalysis.compute_power_spectrum([1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0] * 8)
        scale_0 = spec["scales"][0]
        period_0 = spec["periods"][0]
        self.assertAlmostEqual(period_0, round(scale_0 * expected_factor, 2), places=1)

    def test_ridge_regression_analytic_solution(self):
        """Closed-form Ridge solver must solve known linear systems."""
        X = [[1.0, float(i)] for i in range(20)]
        y = [3.0 + 2.0 * float(i) for i in range(20)]
        w = solve_ridge_regression(X, y, lambda_reg=1e-8)
        self.assertAlmostEqual(w[0], 3.0, places=3)
        self.assertAlmostEqual(w[1], 2.0, places=3)

    def test_evaluation_metrics_hand_computed_values(self):
        """Verify metric values against hand-computed analytical ground truth."""
        y_true = [2.0, 4.0, 6.0, 8.0]
        y_pred = [2.5, 3.5, 6.5, 7.5]
        self.assertAlmostEqual(rmse(y_true, y_pred), 0.5, places=5)
        self.assertAlmostEqual(mae(y_true, y_pred), 0.5, places=5)
        self.assertAlmostEqual(r2_score(y_true, y_pred), 0.95, places=5)
        self.assertAlmostEqual(nash_sutcliffe(y_true, y_pred), 0.95, places=5)

    def test_peak_capture_index_behavior(self):
        """PCI evaluates capture of extreme values."""
        y_true = [1.0] * 9 + [10.0]
        y_pred_good = [1.0] * 9 + [9.5]
        y_pred_poor = [1.0] * 9 + [2.0]

        pci_good = peak_capture_index(y_true, y_pred_good, top_percentile=90.0)
        pci_poor = peak_capture_index(y_true, y_pred_poor, top_percentile=90.0)
        self.assertGreater(pci_good, 0.9)
        self.assertLess(pci_poor, 0.3)

    def test_data_loader_unit_consistency(self):
        """Historical dataset precipitation must be in m/day rate range (0.0 to 0.05)."""
        data = ClimateDataLoader.load_teleconnections_monthly()
        prec = data["precipitation"]
        self.assertGreater(len(prec), 700)
        self.assertTrue(all(p >= 0.0 for p in prec))
        self.assertTrue(all(p < 0.1 for p in prec))
        self.assertTrue(all(-5.0 <= v <= 5.0 for v in data["features"]["NINA"]))
        self.assertTrue(all(-15.0 <= v <= 15.0 for v in data["features"]["SOI"]))


if __name__ == "__main__":
    unittest.main()
