"""
Quickstart demonstration of TeleWave.
Decomposes precipitation, trains Hybrid Wavelet-LSTM, and outputs forecast metrics in 20 lines.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from climatewave import (
    ClimateDataLoader,
    MinMaxScaler,
    WaveletTransform,
    WaveletLSTMForecaster,
    compute_all_metrics,
)


def main():
    print("🌊 TeleWave | Quickstart Precipitation Forecast")

    # 1. Load multi-decadal meteorological & teleconnection data
    data = ClimateDataLoader.load_teleconnections_monthly()
    precip = data["precipitation"]
    exog = [data["features"]["NINA"], data["features"]["AMM"], data["features"]["SOI"]]
    print(f"[*] Loaded {len(precip)} monthly observations (1961 - 2022).")

    # 2. Multiresolution Wavelet Decomposition (MRA)
    wt = WaveletTransform(wavelet="db4")
    energies = wt.energy_distribution(precip, level=3)
    print(f"[*] Wavelet Energy Distribution: {energies}")

    # 3. Sliding window dataset & temporal split
    scaler = MinMaxScaler()
    precip_scaled = scaler.fit_transform(precip)
    X, Y = ClimateDataLoader.create_sliding_windows(precip_scaled, exog_features=exog, lookback=12, horizon=1)
    X_train, y_train, X_test, y_test = ClimateDataLoader.train_test_split(X, Y, train_ratio=0.8)

    # 4. Train Hybrid Wavelet-LSTM Network
    print("[*] Training Multi-Branch Wavelet-LSTM Network...")
    model = WaveletLSTMForecaster(wavelet="db4", decomposition_level=2, hidden_dim=16)
    model.fit(X_train, y_train, epochs=20)

    # 5. Predict and Evaluate
    preds = model.predict(X_test)
    y_test_inv = scaler.inverse_transform([y[0] for y in y_test])
    preds_inv = scaler.inverse_transform([p[0] for p in preds])

    metrics = compute_all_metrics(y_test_inv, preds_inv)
    print("\n" + "=" * 50)
    print("      🌊  FORECAST EVALUATION RESULTS (TEST SET)      ")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:<10} : {v}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
