"""
Unit tests for LiveClimateDataFetcher parsing logic and temporal alignment.
"""

import unittest
from climatewave.data.live_fetcher import LiveClimateDataFetcher


class TestLiveFetcher(unittest.TestCase):

    def test_parse_nina34(self):
        sample = """
 YR   MON  NINO1+2  ANOM   NINO3    ANOM   NINO4    ANOM   NINO3.4  ANOM
1950   1   23.01   -1.55   23.56   -2.10   26.94   -1.38   24.55   -1.99
2026   6   25.94    2.82   28.33    1.71   30.19    1.22   29.17    1.44
"""
        res = LiveClimateDataFetcher.parse_nina34(sample)
        self.assertEqual(len(res), 2)
        self.assertIn((1950, 1), res)
        self.assertAlmostEqual(res[(1950, 1)], -1.99)
        self.assertIn((2026, 6), res)
        self.assertAlmostEqual(res[(2026, 6)], 1.44)

    def test_parse_soi(self):
        sample = """
YEAR   JAN   FEB   MAR   APR   MAY   JUN   JUL   AUG   SEP   OCT   NOV   DEC
1951   2.5   1.5  -0.2  -0.5  -1.1   0.3  -1.7  -0.4  -1.8  -1.6  -1.3  -1.2
2026   1.1   1.4   1.2  -0.6  -0.9  -1.4  -2.4  -1.1-999.9-999.9-999.9-999.9
"""
        res = LiveClimateDataFetcher.parse_soi(sample)
        self.assertIn((1951, 1), res)
        self.assertAlmostEqual(res[(1951, 1)], 2.5)
        self.assertAlmostEqual(res[(1951, 12)], -1.2)
        self.assertIn((2026, 1), res)
        self.assertAlmostEqual(res[(2026, 1)], 1.1)
        self.assertAlmostEqual(res[(2026, 8)], -1.1)
        self.assertNotIn((2026, 9), res)  # -999.9 filtered out

    def test_parse_pdo(self):
        sample = """
Year  Jan   Feb   Mar   Apr   May   Jun   Jul   Aug   Sep   Oct   Nov   Dec
1854  0.11 -0.24 -0.40 -0.44 -0.54 -0.30 -0.10 -1.24 -1.00 -2.23 -1.68 -1.76
2026 -1.24 -1.00 -1.42 -1.62 -1.68 -1.72 -2.14 -1.84 99.99 99.99 99.99 99.99
"""
        res = LiveClimateDataFetcher.parse_pdo(sample)
        self.assertIn((1854, 1), res)
        self.assertAlmostEqual(res[(1854, 1)], 0.11)
        self.assertIn((2026, 1), res)
        self.assertAlmostEqual(res[(2026, 1)], -1.24)
        self.assertNotIn((2026, 9), res)  # 99.99 filtered out

    def test_parse_nao(self):
        sample = """
      Jan    Feb    Mar    Apr    May    Jun    Jul    Aug    Sep    Oct    Nov    Dec
1950   0.92   0.40  -0.36   0.73  -0.59  -0.06  -1.26  -0.05   0.25   0.85  -1.26  -1.02
2026  -0.36   0.68   2.69   1.39  -0.74   0.10  -0.31  -1.92
"""
        res = LiveClimateDataFetcher.parse_nao(sample)
        self.assertIn((1950, 1), res)
        self.assertAlmostEqual(res[(1950, 1)], 0.92)
        self.assertIn((2026, 1), res)
        self.assertAlmostEqual(res[(2026, 1)], -0.36)
        self.assertAlmostEqual(res[(2026, 8)], -1.92)
        self.assertNotIn((2026, 9), res)

    def test_parse_amm(self):
        sample = """
 1948        2026
 1948    0.660   0.000   1.000   1.860   1.300   2.510   1.630   0.550  -0.480  -0.240   0.720  -0.930
 2026    1.580  -0.160 -99.000 -99.000 -99.000 -99.000 -99.000 -99.000 -99.000 -99.000 -99.000 -99.000
"""
        res = LiveClimateDataFetcher.parse_amm(sample)
        self.assertIn((1948, 1), res)
        self.assertAlmostEqual(res[(1948, 1)], 0.660)
        self.assertIn((2026, 1), res)
        self.assertAlmostEqual(res[(2026, 1)], 1.580)
        self.assertAlmostEqual(res[(2026, 2)], -0.160)
        self.assertNotIn((2026, 3), res)  # -99.000 filtered out


if __name__ == "__main__":
    unittest.main()
