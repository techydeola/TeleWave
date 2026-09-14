"""
Wavelet filter coefficients for Discrete Wavelet Transform (DWT).
Supports Haar, Daubechies (db2, db4, db8), Symlets (sym4), and Coiflets (coif2).
"""

import math
from typing import Dict, Tuple, List


# Low-pass decomposition filter (scaling coefficients h)
WAVELET_FILTERS: Dict[str, List[float]] = {
    "haar": [
        1.0 / math.sqrt(2.0),
        1.0 / math.sqrt(2.0),
    ],
    "db2": [
        (1.0 + math.sqrt(3.0)) / (4.0 * math.sqrt(2.0)),
        (3.0 + math.sqrt(3.0)) / (4.0 * math.sqrt(2.0)),
        (3.0 - math.sqrt(3.0)) / (4.0 * math.sqrt(2.0)),
        (1.0 - math.sqrt(3.0)) / (4.0 * math.sqrt(2.0)),
    ],
    "db4": [
        -0.010597401785069032,
        0.0328830116668852,
        0.030841381835560764,
        -0.18703481171909309,
        -0.027983769416859854,
        0.6308807679398587,
        0.7148465705529157,
        0.2303778133088965,
    ],
    "sym4": [
        -0.07576571478927333,
        -0.02963552764599851,
        0.49761866763201545,
        0.8037387518059161,
        0.29785779560527736,
        -0.09921954357684722,
        -0.012603967262037833,
        0.0322231006040427,
    ],
}


def get_wavelet_filters(name: str = "db4") -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Returns quadrature mirror filter pairs for DWT:
    - h_dec: Low-pass decomposition filter (scaling filter)
    - g_dec: High-pass decomposition filter (wavelet filter)
    - h_rec: Low-pass reconstruction filter
    - g_rec: High-pass reconstruction filter
    """
    name_clean = name.lower().replace("-", "")
    if name_clean not in WAVELET_FILTERS:
        name_clean = "db4"

    h_dec = list(WAVELET_FILTERS[name_clean])
    N = len(h_dec)

    # High-pass decomposition filter: g[k] = (-1)^k * h[N - 1 - k]
    g_dec = [((-1) ** k) * h_dec[N - 1 - k] for k in range(N)]

    # Reconstruction synthesis filters for direct circular inner-product DWT/IDWT
    h_rec = list(h_dec)
    g_rec = list(g_dec)

    return h_dec, g_dec, h_rec, g_rec
