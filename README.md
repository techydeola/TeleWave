# TeleWave

> Wavelet decomposition and reservoir-style neural networks for monthly rainfall forecasting, vectorized with NumPy

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-56%20Passing-brightgreen.svg)](https://github.com/techydeola/TeleWave/actions)
[![Dependencies](https://img.shields.io/badge/Dependencies-NumPy%20%7C%20Pandas-purple.svg)](requirements.txt)

---

## Overview

TeleWave is a Python framework that combines Multiresolution Wavelet Transforms (Discrete Wavelet Transform / Multiresolution Analysis) with recurrent reservoir-style neural architectures to forecast monthly meteorological time series.

Meteorological time series present difficulties for standard neural network architectures because observed rainfall superimposes processes operating across multiple distinct timescales: multi-decadal trends, 12-month seasonal cycles, and high-frequency convective storm events.

TeleWave decomposes input rainfall series into Approximation ($A_J$) and Detail ($D_1 \dots D_J$) sub-bands via discrete wavelet filtering. This separates lower-frequency seasonal baselines from high-frequency sub-seasonal oscillations before feature extraction. The recurrent models implemented here operate as reservoir-style networks: recurrent LSTM hidden states are initialized and held fixed, with multi-scale wavelet representations and lag features projected to target horizons using closed-form regularized ridge regression. Core signal processing, sequence generation, and linear algebra routines are vectorized using NumPy.

---

## Mathematical Formulation

### 1. Multiresolution Signal Decomposition (MRA)

A discrete meteorological time series $x(t)$ of length $N$ is decomposed into orthogonal frequency sub-bands using Mallat's pyramid algorithm with circular boundary conditions:

$$x(t) = A_J(t) + \sum_{j=1}^J D_j(t)$$

where:
- $A_J(t)$: Low-frequency approximation component representing the seasonal cycle and interannual drift at decomposition depth $J$.
- $D_j(t)$: High-frequency detail components ($j = 1 \dots J$) capturing intra-seasonal oscillations, frontal systems, and short-duration precipitation events.

Orthogonal wavelet filters supported include Haar, Daubechies (`db2`, `db4`), and Symlets (`sym4`), satisfying Quadrature Mirror Filter (QMF) conditions:

$$g[k] = (-1)^k h[K - 1 - k], \quad \sum_{k=0}^{K-1} h[k] = \sqrt{2}, \quad \sum_{k=0}^{K-1} h[k]^2 = 1.0$$

### 2. Wavelet Feature Extraction and Reservoir Forecasting

The hybrid model (`WaveletLSTMForecaster`) extracts multi-scale features from sliding windows of length $T$:
1. Raw target lags: $x(t-T+1), \dots, x(t)$.
2. Wavelet sub-band features: the latest value and mean of each reconstructed MRA component $A_J, D_J, \dots, D_1$.
3. Exogenous climate covariates: atmospheric and oceanic teleconnection indices (e.g. NINA 3.4, AMM, SOI, PDO, NAO, AMO) at timestep $t$.

The combined feature matrix $\mathbf{H} \in \mathbb{R}^{M \times D}$ is mapped to the forecast horizon using L2-regularized ridge regression:

$$\mathbf{w} = \left(\mathbf{H}^T \mathbf{H} + \lambda \mathbf{I}\right)^{-1} \mathbf{H}^T \mathbf{y}$$

where $\mathbf{w}$ is computed via regularized linear solve using NumPy's LAPACK wrapper (`np.linalg.solve`).

---

## Comparative Evaluation Results

Evaluation conducted on the Nigeria monthly precipitation series (61 years: January 1961 to December 2021, 732 monthly observations) using 12-month lookback windows to predict a 1-month-ahead horizon ($T=12, H=1$):

| Model Architecture | RMSE (scaled) | MAE (scaled) | $R^2$ Score | Nash-Sutcliffe (NSE) | Peak Capture (PCI) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Seasonal Climatology | 0.0962 | 0.0671 | 0.6714 | 0.6714 | 0.7618 |
| Wavelet-LSTM (Reservoir) | 0.1110 | 0.0740 | 0.5625 | 0.5625 | 0.6453 |
| Standard LSTM (Reservoir) | 0.1136 | 0.0772 | 0.5420 | 0.5420 | 0.6424 |
| Persistence Baseline | 0.1582 | 0.1152 | 0.1122 | 0.1122 | 0.5777 |

*Metric definitions and physical units:*
- **Scaled vs. physical error:** In the training pipeline, target precipitation is normalized to $[0, 1]$ with `MinMaxScaler`. RMSE and MAE in the table above reflect this $[0, 1]$ scaled range. When inverse-transformed back to physical units (daily equivalent rate in millimetres per day, $\text{mm/day}$), the corresponding errors are:

| Model Architecture | RMSE ($\text{mm/day}$) | MAE ($\text{mm/day}$) |
| :--- | :---: | :---: |
| Seasonal Climatology | 1.98 | 1.38 |
| Wavelet-LSTM (Reservoir) | 2.28 | 1.52 |
| Standard LSTM (Reservoir) | 2.33 | 1.59 |
| Persistence Baseline | 3.25 | 2.37 |

- **Peak Capture Index (PCI):** Defined as $1 - \frac{1}{K}\sum_{i=1}^K \frac{|y_i - \hat{y}_i|}{|y_i| + 10^{-5}}$ evaluated exclusively over the top 10% highest precipitation months ($\text{percentile} \ge 90$). A value of 1.0 denotes exact prediction of extreme peaks, while lower values reflect under- or over-estimation of heavy rainfall anomalies.

On this test split, the Wavelet-LSTM scored slightly higher than the standard LSTM baseline ($R^2$ of 0.5625 vs. 0.5420 on seed 42), a difference that a single run cannot separate from random variation, and both neural models scored lower than the seasonal climatology baseline ($R^2 = 0.6714$). Results come from a single dataset and a single train/test split, so they should not be read as general performance.

*Multi-seed evaluation across 5 random seeds (seeds 42, 100, 2024, 777, 9999):*
- **Wavelet-LSTM:** Strictly deterministic across all seeds ($R^2 = 0.5625 \pm 0.0000$, $\text{RMSE} = 2.28\,\text{mm/day}$) because its features are derived from deterministic multiresolution sub-bands, raw target lags, and climate teleconnections with a closed-form Ridge projection.
- **Standard LSTM (Reservoir):** Demonstrates variation due to Gaussian weight initialization of its reservoir recurrent connections: Mean $R^2 = 0.5582 \pm 0.0078$ (range: $[0.5480, 0.5702]$), Mean $\text{RMSE} = 2.29 \pm 0.02\,\text{mm/day}$ (range: $[2.26, 2.32]\,\text{mm/day}$).

---

## Refactoring & Measured Performance Benchmarks

The core mathematical operations were refactored from pure Python loops to vectorized NumPy implementations. No algorithmic changes were made to the mathematical formulations, and all forward-pass outputs and model predictions match the original `v1-stdlib` implementation within floating-point tolerance ($< 10^{-12}$).

### Measurement Setup
- **Hardware:** Apple M-series (Apple Silicon), macOS.
- **Environment:** Python 3.9, NumPy 2.0.2, pandas 2.3.3.
- **Methodology:** Execution time measured with `time.perf_counter()`. Each function was executed for 500 to 1,000 iterations after a 20-iteration warmup; reported numbers represent the mean execution time per call.

### Measured Execution Speeds

| Component / Function | Input Shape / Condition | `v1-stdlib` (Pure Python) | `refactor-numpy` (Vectorized) | Measured Speedup |
| :--- | :--- | :---: | :---: | :---: |
| **`dwt_single_level`** | Nigeria series ($N=732$, `db4`) | $757.6\,\mu\text{s}$ | **$36.2\,\mu\text{s}$** | **$20.9\times$** |
| **`idwt_single_level`** | Nigeria series ($N=732$, `db4`) | $770.9\,\mu\text{s}$ | **$45.9\,\mu\text{s}$** | **$16.8\times$** |
| **Multiresolution MRA** | Nigeria series ($N=732$, level 2) | $4.44\,\text{ms}$ | **$0.29\,\text{ms}$** | **$15.3\times$** |
| **Continuous Morlet CWT** | 32 scales ($N=32$) | $18.96\,\text{ms}$ | **$0.79\,\text{ms}$** | **$24.0\times$** |
| **5 Evaluation Metrics** | RMSE, MAE, MAPE, $R^2$, PCI ($N=1,000$) | $1,820.0\,\mu\text{s}$ | **$242.8\,\mu\text{s}$** | **$7.5\times$** |
| **Closed-Form Ridge Solve** | Feature matrix $(500 \times 25)$ | $22.80\,\text{ms}$ | **$0.32\,\text{ms}$** | **$71.3\times$** |
| **Full Test Suite** | 56 unit + characterization tests | $0.354\,\text{s}$ | **$0.043\,\text{s}$** | **$8.2\times$** |

---

## Evaluation Details and Limitations

- **Dataset and split:** Evaluated on `data/raw/nigeria_teleconnections_monthly_1961_2022.csv` (contains 732 monthly observations spanning 61 complete years from January 1961 to December 2021; named `1961_2022` to reflect coverage ending at the start of 2022). A chronological train/test split with `train_ratio = 0.8` produces 576 training windows (1962–2009) and 144 test windows (2010–2021).
- **Parameters:** Lookback window of 12 months, forecast horizon of 1 month, 20 training epochs, decomposition level $J=2$ with Daubechies 4 (`db4`) filter.
- **Single run:** Primary benchmark numbers in the comparative table reflect a fixed random seed (`seed = 42`). Multi-seed spread is documented above.
- **Spatial and temporal resolution:** Data represents monthly aggregate rainfall at national spatial scale. It is not designed for flash flood early warning, thunderstorm prediction, or local/near-real-time nowcasting.
- **Model implementation scope:** The "LSTM" models in this codebase operate as reservoir-style recurrent feature extractors with a closed-form ridge regression readout layer rather than backpropagation-through-time with gradient-based optimizers.
- **Methodological limitations:** No rolling k-fold cross-validation or systematic hyperparameter search was conducted. The sample size (732 monthly steps) is relatively small for deep learning without strong inductive priors.

### Changes from `v1-stdlib` to Current Version
- **Dependencies:** Added `numpy>=1.24.0` and `pandas>=2.0.0` as core dependencies for vectorization.
- **Signal Processing (`dwt.py`):** Replaced nested list loops in Mallat DWT with 2D circular index gathering and matrix multiplications (`samples @ h_dec`, `samples @ g_dec`). IDWT reconstruction uses vectorized outer products and atomic `np.add.at` accumulation.
- **Spectral Analysis (`spectral.py`):** Replaced triple-nested $O(S \times N \times W)$ convolution loops with a 2D temporal distance matrix and vectorized kernel dot products.
- **Data Loading & Sliding Windows (`loader.py`):** CSV parsing now uses `pandas.read_csv()`. Window extraction uses `np.lib.stride_tricks.sliding_window_view()` to construct $(N, T, D)$ tensors without Python-level window loops.
- **LSTM Recurrence (`baseline_lstm.py`):** Packed recurrent gate parameters into contiguous 2D blocks ($W \in \mathbb{R}^{I \times 4H}$, $U \in \mathbb{R}^{H \times 4H}$, $b \in \mathbb{R}^{4H}$) and vectorized forward recurrence across batches.
- **Linear Solver:** Replaced pure-Python Gauss-Jordan elimination with LAPACK `np.linalg.solve()`.
- **Characterization Tests:** Added 8 frozen reference tests (`tests/test_characterization.py` and `tests/golden/`) verifying numerical parity against the `v1-stdlib` baseline down to $< 10^{-12}$ error.

---

## Data Sources

The `data/raw/` directory contains meteorological and teleconnection time series used for model training and historical validation:

| File | Temporal Range | Description & Source |
| :--- | :--- | :--- |
| `nigeria_teleconnections_monthly_1961_2022.csv` | 1961–2021 (monthly, 732 rows) | Monthly precipitation from ERA5 reanalysis paired with 6 climate indices: NINA 3.4, SOI, NAO (NOAA Climate Prediction Center - CPC), PDO (NOAA National Centers for Environmental Information - NCEI), AMM, and AMO (NOAA Physical Sciences Laboratory - PSL). |
| `nigeria_teleconnections_monthly_extended.csv` | 1961–2026 (monthly, 788 rows) | Synchronized extension of the monthly dataset created via `LiveClimateDataFetcher` combining NOAA CPC/NCEI live index tables and Open-Meteo ERA5 archive reanalysis. |
| `ondo_daily_meteorology_1991_2021.csv` | 1991–2021 (daily, 11,324 rows) | Surface meteorological observations for Ondo state centroid (ERA5 reanalysis): $10\text{ m}$ wind vectors (`u10`, `v10`), max/min $2\text{ m}$ temperature (`mx2t`, `mn2t`), and total precipitation (`tp`). |
| `calabar_daily_meteorology_1991_2021.csv` | 1991–2021 (daily, 11,324 rows) | Surface meteorological observations for Calabar coastal station (ERA5 reanalysis), containing same variables as Ondo dataset. |
| `harmattan_hourly_meteorology_2015_2020.csv` | 2015–2020 (hourly, 39,738 rows) | Hourly dry-season surface observations: temperature (`tmpc`), relative humidity (`relh`), wind speed (`sped`), and visibility (`vsby`). Derived from Nigerian airport METAR weather reports archived by the Iowa Environmental Mesonet (IEM). |
| `nigeria_multi_station_precipitation_monthly_1961_2022.csv` | 1961–2022 (monthly, 744 rows) | Multi-station monthly precipitation across 6 southern Nigerian states/cities (Lagos, Cross River, Port Harcourt, Delta, Calabar, Warri) from ERA5 reanalysis. |
| `surface_energy_fluxes_era5_monthly.csv` | Monthly ERA5 records (744 rows) | ERA5 surface energy fluxes: sensible heat flux (`sshf`), latent heat flux (`slhf`), net solar radiation (`ssr`), thermal radiation (`str`), and $2\text{ m}$ dewpoint/temperature. |
| `station_teleconnection_correlations.csv` | Summary table (24 locations) | Spatial correlation coefficients between station-level precipitation and teleconnection indices across Nigerian coordinates. |
| `seasonal_trend_mann_kendall.csv` | Summary table (4 seasons) | Mann-Kendall non-parametric trend test results and Sen's slope estimates across seasons (JFM, AMJ, JAS, OND). |
| `teleconnections_correlation_matrix.csv` | Summary table ($6 \times 6$) | Cross-correlation matrix between the 6 global teleconnection indices (NINA, AMM, SOI, PDO, NAO, AMO). |

*Citations, licenses, and terms:*
- **ERA5 Reanalysis:** Hersbach, H., et al. (2020). *The ERA5 global reanalysis*. Quarterly Journal of the Royal Meteorological Society, 146(730), 1999-2049. Provided by ECMWF under Copernicus Programme license terms (free for research, commercial, and operational use with attribution).
- **NOAA Indices:** NOAA CPC, NCEI, and PSL datasets are US Government public domain products.
- **METAR Data:** Surface airport weather observations collected under WMO standards and archived by the Iowa Environmental Mesonet (IEM, Iowa State University).

---

## Live Telemetry and AI Advisory Modules

TeleWave includes operational scripts demonstrating end-to-end integration:

1. **Live Data Fetcher (`climatewave/data/live_fetcher.py`):**
   - Connects over HTTP to public NOAA CPC (`ersst5.nino.mth.91-20.ascii`, `soi`, `norm.nao.monthly`), NOAA NCEI (`ersst.v5.pdo.dat`), NOAA PSL (`amm.data`), and the Open-Meteo ERA5 archive API.
   - Requires outbound internet access to fetch live updates; falls back to local CSV files if offline.

2. **AI Advisory Interpreter (`climatewave/advisory/interpreter.py`):**
   - Synthesizes quantitative wavelet metrics and forecasts into structured hydrometeorological advisory memorandums.
   - Supports 10 API providers (Anthropic Claude, Google Gemini, OpenAI, DeepSeek, Groq, xAI Grok, Mistral, OpenRouter, AgentRouter, or custom OpenAI-compatible endpoints).
   - Requires an API key for the chosen LLM provider. When no API key is provided, the system automatically falls back to an internal deterministic, rule-based advisory generator without making external network calls.

---

## Quickstart

### 1. Run via CLI
```bash
# Multiresolution Wavelet Decomposition
python3 -m climatewave.cli decompose --wavelet db4 --level 3

# Run Comparative Benchmark
python3 -m climatewave.cli benchmark --epochs 20

# Continuous Wavelet Power Spectrum Analysis
python3 -m climatewave.cli spectral
```

### 2. Guided Real-World Scenario Walkthroughs
Run terminal walkthroughs evaluating 2-sigma wavelet D1 anomaly thresholding and walk-forward forecasting across historical climate events:
```bash
# 1982-83 El Niño Sahel Drought
python3 run_scenario.py 1983_drought

# 2012 Niger Delta Floods
python3 run_scenario.py 2012_floods

# Live Climate Monitor (NOAA CPC & ERA5 Reanalysis)
# Use --no-ai to run entirely offline with the rule-based advisor
python3 run_scenario.py live --no-ai
```

### 3. Python Usage
```python
from climatewave import (
    ClimateDataLoader,
    MinMaxScaler,
    WaveletLSTMForecaster,
    compute_all_metrics,
)

# 1. Load data
data = ClimateDataLoader.load_teleconnections_monthly()
precip = data["precipitation"]

# 2. Windowing & Scaling
scaler = MinMaxScaler()
scaled = scaler.fit_transform(precip)
X, Y = ClimateDataLoader.create_sliding_windows(scaled, lookback=12, horizon=1)
X_train, y_train, X_test, y_test = ClimateDataLoader.train_test_split(X, Y, train_ratio=0.8)

# 3. Train Wavelet-LSTM
model = WaveletLSTMForecaster(wavelet="db4", decomposition_level=2)
model.fit(X_train, y_train, epochs=20)

# 4. Predict
preds = model.predict(X_test)
```

---

## Running Unit Tests

The test suite runs with standard library `unittest` and requires `numpy` and `pandas`:
```bash
python3 -m unittest discover -s tests -v
```

All 56 tests pass, including:
- 48 unit tests covering wavelet filter coefficients, QMF orthogonality, DWT/IDWT perfect reconstruction, Morlet Fourier factor accuracy, closed-form ridge regression, metric formulations, and data loading routines.
- 8 characterization tests in `tests/test_characterization.py` verifying numerical parity against the `v1-stdlib` golden outputs down to $< 10^{-12}$ error.

---

## Project Structure

```
TeleWave/
├── climatewave/
│   ├── wavelets/
│   │   ├── filters.py          # Wavelet coefficients (Haar, db2, db4, sym4)
│   │   ├── dwt.py              # Discrete Wavelet Transform & MRA engine (NumPy)
│   │   └── spectral.py         # Morlet CWT power spectrum (NumPy)
│   ├── data/
│   │   ├── loader.py           # Datasets loader, scalers & sliding windows (Pandas & NumPy)
│   │   └── live_fetcher.py     # NOAA CPC/NCEI & ERA5 live HTTP ingestion
│   ├── models/
│   │   ├── base.py             # Abstract forecaster interface
│   │   ├── wavelet_lstm.py     # Wavelet-Reservoir hybrid model (NumPy)
│   │   ├── baseline_lstm.py    # Recurrent reservoir baseline (NumPy)
│   │   └── statistical.py      # Persistence & seasonal climatology baselines
│   ├── evaluation/
│   │   ├── metrics.py          # RMSE, MAE, R², NSE, Peak Capture Index (NumPy)
│   │   └── comparator.py       # Benchmark evaluation harness
│   ├── advisory/
│   │   └── interpreter.py      # Multi-platform AI advisory memo generator
│   ├── scenarios/
│   │   └── walkthrough.py      # Historical & live terminal walkthroughs
│   └── cli.py                  # Command-line interface
├── data/
│   └── raw/                    # Bundled meteorological and teleconnection CSVs
├── tests/
│   ├── golden/                 # Frozen reference outputs from v1-stdlib baseline
│   ├── test_characterization.py# Parity tests verifying outputs match v1-stdlib (<1e-12)
│   ├── test_data.py            # Data loading & windowing tests
│   ├── test_evaluation.py      # Metric calculation unit tests
│   ├── test_interpreter.py     # LLM and rule-based advisory tests
│   ├── test_live_fetcher.py    # Live NOAA data parsing tests
│   ├── test_models.py          # Baseline and Wavelet LSTM model tests
│   ├── test_qa_equations.py    # Analytical mathematical equation tests
│   └── test_wavelets.py        # DWT, IDWT, MRA, and Morlet CWT tests
├── examples/                   # Quickstart Python tutorial
├── run_scenario.py             # Scenario walkthrough runner
├── pyproject.toml              # Packaging configuration
├── requirements.txt            # Project dependencies (NumPy, Pandas)
└── README.md                   # Scientific documentation
```

---

## Author

**Adeola Johnson**  
Software engineer, climate informatics  
- GitHub: [@techydeola](https://github.com/techydeola)  
- Repository: [https://github.com/techydeola/TeleWave](https://github.com/techydeola/TeleWave)
