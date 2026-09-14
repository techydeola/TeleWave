"""
Unit tests for TeleWave Wavelet Transform & Multiresolution Analysis.
"""

import unittest
import math
from climatewave.wavelets.filters import get_wavelet_filters, WAVELET_FILTERS
from climatewave.wavelets.dwt import WaveletTransform
from climatewave.wavelets.spectral import WaveletSpectralAnalysis


class TestWavelets(unittest.TestCase):

    def test_wavelet_filters(self):
        for w_name in ["haar", "db2", "db4", "sym4"]:
            h_dec, g_dec, h_rec, g_rec = get_wavelet_filters(w_name)
            self.assertEqual(len(h_dec), len(g_dec))
            self.assertEqual(len(h_rec), len(g_rec))
            # Check filter energy normalization
            energy_h = sum(x ** 2 for x in h_dec)
            self.assertAlmostEqual(energy_h, 1.0, places=3)

    def test_dwt_idwt_reconstruction(self):
        wt = WaveletTransform(wavelet="db4")
        # Synthetic signal
        signal = [math.sin(0.1 * i) + 0.5 * math.cos(0.4 * i) for i in range(64)]
        cA, cD = wt.dwt_single_level(signal)
        self.assertEqual(len(cA), 32)
        self.assertEqual(len(cD), 32)

        reconstructed = wt.idwt_single_level(cA, cD, len(signal))
        self.assertEqual(len(reconstructed), len(signal))

    def test_multiresolution_analysis_additive(self):
        wt = WaveletTransform(wavelet="db4")
        signal = [math.sin(0.05 * i) + 0.3 * math.sin(0.3 * i) for i in range(128)]
        mra_dict = wt.mra(signal, level=3)

        self.assertIn("A3", mra_dict)
        self.assertIn("D1", mra_dict)
        self.assertIn("D2", mra_dict)
        self.assertIn("D3", mra_dict)

        for comp in mra_dict.values():
            self.assertEqual(len(comp), len(signal))

        energies = wt.energy_distribution(signal, level=3)
        self.assertGreaterEqual(sum(energies.values()), 80.0)

    def test_wavelet_spectral_analysis(self):
        # 12-month periodic wave
        signal = [math.sin(2.0 * math.pi * i / 12.0) for i in range(120)]
        spec = WaveletSpectralAnalysis.compute_power_spectrum(signal)
        self.assertIn("dominant_period", spec)
        self.assertGreater(len(spec["scales"]), 0)


if __name__ == "__main__":
    unittest.main()
