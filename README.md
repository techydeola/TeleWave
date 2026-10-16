# TeleWave

> Hybrid Wavelet + Deep Learning Framework for Climate, Weather & Rainfall Forecasting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-48%20Passing-brightgreen.svg)]()
[![Zero External Dependencies](https://img.shields.io/badge/Dependencies-Zero%20External%20Req-purple.svg)]()

---

## Overview

TeleWave is a Python framework that combines Multiresolution Wavelet Transforms (Discrete Wavelet Transform / Multiresolution Analysis) with recurrent deep learning models to forecast meteorological and hydrological time series.

Meteorological time series present difficulties for standard neural network architectures because observed rainfall superimposes processes operating across multiple distinct timescales: multi-decadal trends, 12-month seasonal cycles, and high-frequency convective storm events.

TeleWave decomposes input rainfall series into Approximation ($A_J$) and Detail ($D_1 \dots D_J$) sub-bands via discrete wavelet filtering. This separates lower-frequency seasonal baselines from high-frequency sub-seasonal oscillations before feature extraction and forecasting.

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

### 2. Wavelet Feature Extraction and Hybrid Forecasting

The hybrid model (`WaveletLSTMForecaster`) extracts multi-scale features from sliding windows of length $T$:
1. Raw target lags: $x(t-T+1), \dots, x(t)$.
2. Wavelet sub-band features: the latest value and mean of each reconstructed MRA component $A_J, D_J, \dots, D_1$.
3. Exogenous climate covariates: atmospheric and oceanic teleconnection indices (e.g. NINA 3.4, AMM, SOI, PDO, NAO, AMO) at timestep $t$.

The combined feature matrix $\mathbf{H} \in \mathbb{R}^{M \times D}$ is mapped to the forecast horizon using L2-regularized ridge regression:

$$\mathbf{w} = \left(\mathbf{H}^T \mathbf{H} + \lambda \mathbf{I}\right)^{-1} \mathbf{H}^T \mathbf{y}$$

where $\mathbf{w}$ is computed via Gauss-Jordan elimination with partial pivoting in pure Python standard library.

---

## Comparative Evaluation Results

Evaluation conducted on the Nigeria monthly precipitation series (1961–2022) using 12-month lookback windows to predict a 1-month-ahead horizon ($T=12, H=1$):

| Model Architecture | RMSE (scaled) | MAE (scaled) | $R^2$ Score | Nash-Sutcliffe (NSE) | Peak Capture (PCI) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Seasonal Climatology | 0.0962 | 0.0671 | 0.6714 | 0.6714 | 0.7618 |
| Wavelet-LSTM (Hybrid) | 0.1110 | 0.0740 | 0.5625 | 0.5625 | 0.6453 |
| Standard LSTM | 0.1136 | 0.0772 | 0.5420 | 0.5420 | 0.6424 |
| Persistence Baseline | 0.1582 | 0.1152 | 0.1122 | 0.1122 | 0.5777 |

*Note on units:* In the standard training pipeline, target precipitation is normalized to $[0, 1]$ with `MinMaxScaler`. RMSE and MAE in the table above reflect this $[0, 1]$ scaled range. When inverse-transformed back to physical units (ERA5 equivalent mean daily precipitation rate in meters per day, $m/\text{day}$), the corresponding errors are:

| Model Architecture | RMSE ($m/\text{day}$) | MAE ($m/\text{day}$) |
| :--- | :---: | :---: |
| Seasonal Climatology | 0.0020 | 0.0014 |
| Wavelet-LSTM (Hybrid) | 0.0023 | 0.0015 |
| Standard LSTM | 0.0023 | 0.0016 |
| Persistence Baseline | 0.0033 | 0.0024 |

On this test split, the Wavelet-LSTM scored higher than the standard LSTM baseline and the persistence benchmark, but lower than the seasonal climatology baseline. Results come from a single dataset and a single train/test split, so they should not be read as general performance.

---

## Evaluation Details and Limitations

- **Dataset and split:** Evaluated on `data/raw/nigeria_teleconnections_monthly_1961_2022.csv` (732 monthly observations, January 1961 to December 2021). Chronological train/test split with `train_ratio = 0.8` produces 576 training windows and 144 test windows (test period: 2010 to 2021).
- **Parameters:** Lookback window of 12 months, forecast horizon of 1 month, 20 training epochs, decomposition level $J=2$ with Daubechies 4 (`db4`) filter.
- **Single run:** Results are from a single run with a fixed random seed (`seed = 42`), not averaged across multiple random initializations.
- **Spatial and temporal resolution:** Data represents monthly aggregate rainfall at national spatial scale. It is not designed for flash flood early warning, thunderstorm prediction, or local/near-real-time nowcasting.
- **Model implementation scope:** The LSTM implementations in this repository operate as reservoir-style networks with regularized closed-form readout layers rather than full backpropagation-through-time with Adam/SGD optimizers.
- **Methodological limitations:** No rolling k-fold cross-validation or systematic hyperparameter search was conducted. The sample size (732 monthly steps) is relatively small for deep learning without strong inductive priors.

---

## Data Sources

The `data/raw/` directory contains meteorological and teleconnection time series used for model training and historical validation:

| File | Temporal Range | Description & Source |
| :--- | :--- | :--- |
| `nigeria_teleconnections_monthly_1961_2022.csv` | 1961–2021 (monthly, 732 rows) | Monthly precipitation (ERA5 reanalysis, ECMWF / Copernicus Open Access) paired with 6 climate indices: NINA 3.4, SOI, NAO (NOAA Climate Prediction Center - CPC), PDO (NOAA National Centers for Environmental Information - NCEI), AMM, and AMO (NOAA Physical Sciences Laboratory - PSL). |
| `nigeria_teleconnections_monthly_extended.csv` | 1961–2026 (monthly, 788 rows) | Synchronized extension of the monthly dataset created via `LiveClimateDataFetcher` combining NOAA CPC/NCEI live index tables and Open-Meteo ERA5 archive reanalysis. |
| `ondo_daily_meteorology_1991_2021.csv` | 1991–2021 (daily, 11,324 rows) | Surface meteorological observations for Ondo state centroid (ERA5 reanalysis): $10\text{ m}$ wind vectors (`u10`, `v10`), max/min $2\text{ m}$ temperature (`mx2t`, `mn2t`), and total precipitation (`tp`). |
| `calabar_daily_meteorology_1991_2021.csv` | 1991–2021 (daily, 11,324 rows) | Surface meteorological observations for Calabar coastal station (ERA5 reanalysis), containing same variables as Ondo dataset. |
| `harmattan_hourly_meteorology_2015_2020.csv` | 2015–2020 (hourly, 39,738 rows) | Hourly dry-season observations: temperature (`tmpc`), relative humidity (`relh`), wind speed (`sped`), and visibility (`vsby`). Derived from surface METAR station observations. |
| `nigeria_multi_station_precipitation_monthly_1961_2022.csv` | 1961–2022 (monthly, 744 rows) | Multi-station monthly precipitation across 6 southern Nigerian states/cities (Lagos, Cross River, Port Harcourt, Delta, Calabar, Warri) from ERA5 reanalysis. |
| `surface_energy_fluxes_era5_monthly.csv` | Monthly ERA5 records (744 rows) | ERA5 surface energy fluxes: sensible heat flux (`sshf`), latent heat flux (`slhf`), net solar radiation (`ssr`), thermal radiation (`str`), and $2\text{ m}$ dewpoint/temperature. |
| `station_teleconnection_correlations.csv` | Summary table (24 locations) | Spatial correlation coefficients between station-level precipitation and teleconnection indices across Nigerian coordinates. |
| `seasonal_trend_mann_kendall.csv` | Summary table (4 seasons) | Mann-Kendall non-parametric trend test results and Sen's slope estimates across seasons (JFM, AMJ, JAS, OND). |
| `teleconnections_correlation_matrix.csv` | Summary table ($6 \times 6$) | Cross-correlation matrix between the 6 global teleconnection indices (NINA, AMM, SOI, PDO, NAO, AMO). |

*License and terms:*
- NOAA CPC, NCEI, and PSL datasets are US Government public domain products.
- ERA5 reanalysis data is provided by ECMWF under Copernicus Programme license terms (free for research, commercial, and operational use with attribution).

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
Run terminal walkthroughs evaluating anomaly detection and forecasting during historical events:
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

The test suite runs with standard library unittest and requires no external packages:
```bash
python3 -m unittest discover -s tests -v
```

All 48 tests pass, covering wavelet filter coefficients, QMF orthogonality, DWT/IDWT perfect reconstruction, Morlet Fourier factor accuracy, closed-form ridge regression, metric formulations, and data loading routines.

---

## Project Structure

```
TeleWave/
├── climatewave/
│   ├── wavelets/
│   │   ├── filters.py          # Wavelet coefficients (Haar, db2, db4, sym4)
│   │   ├── dwt.py              # Discrete Wavelet Transform & MRA engine
│   │   └── spectral.py         # Morlet CWT power spectrum
│   ├── data/
│   │   ├── loader.py           # Datasets loader, scalers & sliding windows
│   │   └── live_fetcher.py     # NOAA CPC/NCEI & ERA5 live HTTP ingestion
│   ├── models/
│   │   ├── base.py             # Abstract forecaster interface
│   │   ├── wavelet_lstm.py     # Multi-Branch Wavelet-LSTM network
│   │   ├── baseline_lstm.py    # Standard recurrent LSTM baseline
│   │   └── statistical.py      # Persistence & seasonal climatology baselines
│   ├── evaluation/
│   │   ├── metrics.py          # RMSE, MAE, R², NSE, Peak Capture Index
│   │   └── comparator.py       # Benchmark evaluation harness
│   ├── advisory/
│   │   └── interpreter.py      # Multi-platform AI advisory memo generator
│   ├── scenarios/
│   │   └── walkthrough.py      # Historical & live terminal walkthroughs
│   └── cli.py                  # Command-line interface
├── data/
│   └── raw/                    # Bundled meteorological and teleconnection CSVs
├── tests/                      # 48 passing unit tests
├── examples/                   # Quickstart Python tutorial
├── run_scenario.py             # Scenario walkthrough runner
├── pyproject.toml              # Packaging configuration
└── README.md                   # Scientific documentation
```

---

## Author

**Adeola Johnson**  
Software engineer, climate informatics  
- GitHub: [@techydeola](https://github.com/techydeola)  
- Repository: [https://github.com/techydeola/TeleWave](https://github.com/techydeola/TeleWave)
