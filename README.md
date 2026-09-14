# 🌊 TeleWave

> **Hybrid Wavelet + Deep Learning Framework for Climate, Extreme Weather & Rainfall Forecasting**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-100%25%20Passing-brightgreen.svg)]()
[![Zero External Dependencies](https://img.shields.io/badge/Dependencies-Zero%20External%20Req-purple.svg)]()
[![Physics Informed](https://img.shields.io/badge/Domain-Physics%20%2B%20Deep%20Learning-cyan.svg)]()

---

## 🌟 Overview

**TeleWave** is a scientific machine learning framework that combines **Multiresolution Wavelet Transforms (DWT / MRA)** with **Multi-Branch Deep Learning (Wavelet-LSTM & Attention Networks)** to forecast non-stationary hydrological and atmospheric time-series.

Standard neural networks frequently struggle with meteorological time-series because weather signals contain complex superposition of multi-scale phenomena: long-term multi-decadal trends, 12-month seasonal cycles, and sudden high-frequency convective storm bursts.

By decomposing raw precipitation signals into **Approximation ($A_J$)** and **Detail ($D_1 \dots D_J$)** sub-bands, **TeleWave** enables dedicated recurrent branches to capture both slow-moving climate modes (e.g., ENSO, AMO, NAO) and sudden extreme rainfall events simultaneously.

---

## 📐 Mathematical Formulation

### 1. Multiresolution Signal Decomposition (MRA)
Any raw meteorological time series $x(t)$ is decomposed into orthogonal frequency sub-bands via the Discrete Wavelet Transform (DWT):

$$x(t) = A_J(t) + \sum_{j=1}^J D_j(t)$$

where:
- $A_J(t)$: Low-frequency **Approximation** component capturing annual/interannual seasonal climatology and multi-decadal drift.
- $D_j(t)$: High-frequency **Detail** components ($j = 1 \dots J$) capturing intra-seasonal oscillations, frontal systems, and localized storm events.

### 2. Multi-Branch Recurrent Neural Network
Each decomposed sub-band is routed into a dedicated recurrent sub-network:

$$h_t^{(A_J)} = \text{LSTM}^{(A_J)}\left(A_J(t), h_{t-1}^{(A_J)}\right)$$
$$h_t^{(D_j)} = \text{LSTM}^{(D_j)}\left(D_j(t), h_{t-1}^{(D_j)}\right), \quad j \in \{1, \dots, J\}$$

### 3. Multiresolution Forecast Synthesis
The final multi-horizon prediction $\hat{y}_{t+h}$ is synthesized additively:

$$\hat{y}_{t+h} = \hat{y}_{t+h}^{(A_J)} + \sum_{j=1}^J \hat{y}_{t+h}^{(D_j)}$$

---

## 🏗️ Architecture & Pipeline

```
+---------------------------------------------------------------------------------+
|                       TeleWave Forecasting Pipeline                       |
+---------------------------------------------------------------------------------+
                                         |
                                         v
                      +--------------------------------------+
                      | Raw Meteorological Series x(t)       |
                      | & Climate Teleconnections (ENSO/AMO) |
                      +--------------------------------------+
                                         |
                     [ Multiresolution Wavelet Transform ]
                                         |
         +-------------------------------+-------------------------------+
         |                               |                               |
         v                               v                               v
+------------------+            +------------------+            +------------------+
| Approx Branch A_J|            | Detail Branch D_2|            | Detail Branch D_1|
| (Seasonal Mode)  |            | (Intra-Seasonal) |            | (Convective Peak)|
+------------------+            +------------------+            +------------------+
         |                               |                               |
         v                               v                               v
+------------------+            +------------------+            +------------------+
|  LSTM Branch A   |            |  LSTM Branch D2  |            |  LSTM Branch D1  |
+------------------+            +------------------+            +------------------+
         |                               |                               |
         +-------------------------------+-------------------------------+
                                         |
                            [ Multi-Scale Fusion Layer ]
                                         |
                                         v
                      +--------------------------------------+
                      | Integrated Multi-Horizon Forecast    |
                      | + Extreme Peak-Capture Index (PCI)   |
                      +--------------------------------------+
```

---

## 📊 Benchmark Datasets Included

The project bundles real-world multi-decadal meteorological datasets:

| Dataset | Period | Frequency | Variables |
| :--- | :--- | :--- | :--- |
| **`nigeria_teleconnections_monthly_1961_2022.csv`** | 1961 – 2022 | Monthly | Precipitation (`Prec`), ENSO (`NINA`), Atlantic Mode (`AMM`), `SOI`, `PDO`, `NAO`, `AMO` |
| **`ondo_daily_meteorology_1991_2021.csv`** | 1991 – 2021 | Daily (11,324 rows) | 10m Winds (`u10`, `v10`), Max/Min Temp (`mx2t`, `mn2t`), Total Precip (`tp`) |
| **`calabar_daily_meteorology_1991_2021.csv`** | 1991 – 2021 | Daily (11,324 rows) | Coastal meteorological station time series |
| **`harmattan_hourly_meteorology_2015_2020.csv`** | 2015 – 2020 | Hourly (39,738 rows)| Temperature (`tmpc`), Humidity (`relh`), Wind (`sped`), Visibility (`vsby`) |

---

## 🏆 Comparative Evaluation Results

Evaluation on Nigeria Monthly Precipitation Test Set (1961 – 2022):

| Model Architecture | RMSE (mm) | MAE (mm) | $R^2$ Score | Nash-Sutcliffe (NSE) | Peak Capture (PCI) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Wavelet-LSTM (Hybrid)** 🏆 | **0.0021** | **0.0016** | **0.794** | **0.794** | **0.862** |
| **Standard Raw LSTM** | 0.0034 | 0.0028 | 0.461 | 0.461 | 0.584 |
| **Seasonal Climatology** | 0.0041 | 0.0033 | 0.220 | 0.220 | 0.410 |
| **Persistence Baseline** | 0.0053 | 0.0042 | -0.312 | -0.312 | 0.320 |

> **Key Takeaway:** Wavelet-LSTM achieves a **~38% reduction in RMSE** and boosts extreme rainfall peak capture from 0.58 to 0.86 compared to standard deep learning baselines!

---

## 🚀 Quickstart

### 1. Run via CLI
```bash
# 1. Multiresolution Wavelet Decomposition
python3 -m climatewave.cli decompose --wavelet db4 --level 3

# 2. Run Comparative Benchmark
python3 -m climatewave.cli benchmark --epochs 20

# 3. Wavelet Power Spectrum Analysis
python3 -m climatewave.cli spectral
```

### 2. Guided Real-World Scenario Walkthroughs
Experience how TeleWave detects pre-disaster anomalies and generates operational decision memos:
```bash
# 1982-83 El Niño Sahel Drought
python3 run_scenario.py 1983_drought

# 2012 Niger Delta Floods
python3 run_scenario.py 2012_floods

# Real-Time 2026 Climate Monitor (Live NOAA CPC & ERA5 Reanalysis)
python3 run_scenario.py live
```

### 3. Python SDK Usage
```python
from climatewave import ClimateDataLoader, MinMaxScaler, WaveletLSTMForecaster, compute_all_metrics

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

## 🧪 Running Unit Tests

Execute the test suite with 100% pass rate:
```bash
python3 -m unittest discover -s tests
```

---

## 📁 Project Structure

```
TeleWave/
├── climatewave/
│   ├── wavelets/
│   │   ├── filters.py          # Wavelet coefficients (Haar, db2, db4, sym4)
│   │   ├── dwt.py              # Discrete Wavelet Transform & MRA engine
│   │   └── spectral.py         # Morlet CWT power spectrum
│   ├── data/
│   │   ├── loader.py           # Datasets loader, scalers & sliding windows
│   │   └── live_fetcher.py     # Real-time NOAA CPC/NCEI & ERA5 ingestion
│   ├── models/
│   │   ├── base.py             # Abstract forecaster interface
│   │   ├── wavelet_lstm.py     # Multi-Branch Wavelet-LSTM network
│   │   ├── baseline_lstm.py    # Standard raw LSTM baseline
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
│   └── raw/                    # Bundled real-world climate datasets
├── tests/                      # Full unittest test suite (46 tests)
├── examples/                   # Quickstart Python tutorial
├── run_scenario.py             # Scenario walkthrough runner
├── pyproject.toml              # Packaging configuration
└── README.md                   # Scientific documentation
```

---

## 👨‍💻 Author & Portfolio

**Adeola Johnson**  
*Machine Learning Researcher & Engineer | Climate Informatics & Deep Learning*
- **GitHub:** [@techydeola](https://github.com/techydeola)
- **Repository:** `TeleWave`
