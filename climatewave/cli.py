"""
Command-Line Interface (CLI) for TeleWave.
Supports wavelet multiresolution decomposition, benchmark evaluation,
precipitation forecasting, and guided real-world scenario walkthroughs.
"""

import sys
import os
import argparse
from climatewave.data.loader import ClimateDataLoader, MinMaxScaler
from climatewave.wavelets.dwt import WaveletTransform
from climatewave.wavelets.spectral import WaveletSpectralAnalysis
from climatewave.models.wavelet_lstm import WaveletLSTMForecaster
from climatewave.models.baseline_lstm import BaselineLSTMForecaster
from climatewave.evaluation.comparator import ModelBenchmarkHarness
from climatewave.scenarios.walkthrough import ScenarioWalkthrough


def color(text: str, c: str) -> str:
    codes = {
        "cyan": "\033[96m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "bold": "\033[1m",
        "red": "\033[91m",
        "reset": "\033[0m",
    }
    return f"{codes.get(c, '')}{text}{codes['reset']}"


def cmd_decompose(args):
    print(color("\n=======================================================", "cyan"))
    print(color("   🌊 TeleWave | Multiresolution Wavelet MRA    ", "bold"))
    print(color("=======================================================\n", "cyan"))

    data = ClimateDataLoader.load_teleconnections_monthly()
    precip = data["precipitation"]
    dates = data["dates"]

    wt = WaveletTransform(wavelet=args.wavelet)
    mra_dict = wt.mra(precip, level=args.level)
    energies = wt.energy_distribution(precip, level=args.level)

    print(f"[*] Input Signal: {color('Nigeria Monthly Precipitation (1961-2022)', 'bold')} ({len(precip)} records)")
    print(f"[*] Wavelet Filter: {color(args.wavelet.upper(), 'yellow')} | Decomposition Level: {args.level}\n")

    print(color("----------------- MULTIRESOLUTION ENERGY BREAKDOWN -----------------", "bold"))
    print(f"{'Component':<12} | {'Description':<35} | {'Energy (%)':<12}")
    print("-" * 65)

    num_levels = args.level
    print(f"{f'A{num_levels}':<12} | {'Low-Frequency Trend / Seasonal Cycle':<35} | {energies.get(f'A{num_levels}', 0.0):.2f}%")
    for j in range(num_levels, 0, -1):
        desc = f"Detail Level {j} (Intra-Seasonal / High-Freq Fluctuations)"
        print(f"{f'D{j}':<12} | {desc:<35} | {energies.get(f'D{j}', 0.0):.2f}%")
    print(color("-" * 65 + "\n", "bold"))


def cmd_benchmark(args):
    print(color("\n=======================================================", "cyan"))
    print(color("   🚀 TeleWave | Model Benchmark & Evaluation   ", "bold"))
    print(color("=======================================================\n", "cyan"))

    data = ClimateDataLoader.load_teleconnections_monthly()
    precip = data["precipitation"]
    exog = [
        data["features"]["NINA"],
        data["features"]["AMM"],
        data["features"]["SOI"],
        data["features"]["PDO"],
        data["features"]["NAO"],
        data["features"]["AMO"],
    ]

    scaler = MinMaxScaler()
    precip_scaled = scaler.fit_transform(precip)

    X, Y = ClimateDataLoader.create_sliding_windows(precip_scaled, exog_features=exog, lookback=12, horizon=1)
    X_train, y_train, X_test, y_test = ClimateDataLoader.train_test_split(X, Y, train_ratio=0.8)

    print(f"[*] Dataset: Nigeria Precipitation + 6 Climate Teleconnections (NINA, AMM, SOI, PDO, NAO, AMO)")
    print(f"[*] Training Windows: {len(X_train)} | Test Windows: {len(X_test)} | Epochs: {args.epochs}\n")
    print("[*] Training models and computing comparative metrics...")

    results = ModelBenchmarkHarness.run_comparison(X_train, y_train, X_test, y_test, epochs=args.epochs)
    leaderboard = results["leaderboard"]

    print("\n" + color("---------------------- BENCHMARK LEADERBOARD ----------------------", "bold"))
    print(f"{'Model Architecture':<28} | {'RMSE':<8} | {'MAE':<8} | {'R² Score':<8} | {'Peak Capture':<12}")
    print("-" * 72)
    for row in leaderboard:
        m_name = row["Model"]
        rmse_val = f"{row['RMSE']:.4f}"
        mae_val = f"{row['MAE']:.4f}"
        r2_val = f"{row['R2']:.4f}"
        pci_val = f"{row['PCI']:.2f}"
        is_best = m_name == results["best_model"]
        line = f"{m_name:<28} | {rmse_val:<8} | {mae_val:<8} | {r2_val:<8} | {pci_val:<12}"
        if is_best:
            print(color(line, "green"))
        else:
            print(line)
    print(color("-" * 72 + "\n", "bold"))
    print(f"🏆 Top Performing Architecture: {color(results['best_model'], 'green')}\n")


def cmd_spectral(args):
    print(color("\n=======================================================", "cyan"))
    print(color("   📡 TeleWave | Continuous Wavelet Power Spectrum ", "bold"))
    print(color("=======================================================\n", "cyan"))

    data = ClimateDataLoader.load_teleconnections_monthly()
    precip = data["precipitation"]

    spec = WaveletSpectralAnalysis.compute_power_spectrum(precip)
    print(f"[*] Time-Series: Monthly Precipitation (1961 - 2022)")
    dom_period = spec["dominant_period"]
    print(f"[*] Dominant Periodic Oscillation: {color(f'{dom_period} months', 'yellow')} (Annual / Seasonal Harmonic)")
    print(f"[*] Peak Scales Analysed: {len(spec['scales'])} frequency bands\n")


def cmd_scenario(args):
    interactive = getattr(args, "interactive", False)
    scenario = getattr(args, "scenario", "all")
    api_key = getattr(args, "api_key", None)
    base_url = getattr(args, "base_url", None)
    model = getattr(args, "model", None)
    no_ai = getattr(args, "no_ai", False)
    sw = ScenarioWalkthrough(
        interactive=interactive,
        api_key=api_key,
        base_url=base_url,
        model=model,
        no_ai=no_ai,
    )
    valid = ["1983_drought", "2012_floods", "live", "all"]
    if scenario not in valid:
        print(f"  Unknown scenario: '{scenario}'. Choose from: {', '.join(valid)}")
        return
    if scenario == "all":
        sw.run("1983_drought")
        sw.run("2012_floods")
        sw.run("live")
    else:
        sw.run(scenario)


def cmd_sync(args):
    print(color("\n=======================================================", "cyan"))
    print(color("   🌐 TeleWave | NOAA & ERA5 Live Data Sync     ", "bold"))
    print(color("=======================================================\n", "cyan"))
    from climatewave.data.live_fetcher import LiveClimateDataFetcher
    print("[*] Contacting NOAA CPC, NCEI, and PSL servers for 6 climate indices...")
    print("[*] Ingesting ECMWF / Open-Meteo ERA5 precipitation rates...")
    out_path, new_cnt = LiveClimateDataFetcher.sync_to_file()
    print(f"[*] Success! Ingested {color(str(new_cnt), 'green')} new monthly records.")
    print(f"[*] Synchronized extended dataset written to: {color(out_path, 'yellow')}\n")


def main():
    parser = argparse.ArgumentParser(
        prog="climatewave",
        description="TeleWave: Hybrid Wavelet + Deep Learning Climate Forecasting Engine",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Decompose
    p_dec = subparsers.add_parser("decompose", help="Multiresolution wavelet decomposition of precipitation")
    p_dec.add_argument("--wavelet", type=str, default="db4", choices=["haar", "db2", "db4", "sym4"], help="Wavelet filter")
    p_dec.add_argument("--level", type=int, default=3, help="Decomposition depth level")

    # Benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run comparative model benchmark")
    p_bench.add_argument("--epochs", type=int, default=20, help="Training epochs")

    # Spectral
    p_spec = subparsers.add_parser("spectral", help="Compute continuous wavelet power spectrum")

    # Helper for adding LLM arguments
    def add_llm_args(p):
        p.add_argument("--api-key", type=str, default=None, help="AgentRouter or OpenAI-compatible API key for AI interpretation")
        p.add_argument("--base-url", type=str, default=None, help="Custom API Base URL (default: https://agentrouter.org/v1)")
        p.add_argument("--model", type=str, default=None, help="LLM model name (e.g. gpt-4o, llama-3.3-70b-versatile)")
        p.add_argument("--no-ai", action="store_true", help="Disable AI interpretation and interactive AI prompt")

    # Scenario
    p_scenario = subparsers.add_parser("scenario", help="Run guided real-world scenario walkthrough")
    p_scenario.add_argument(
        "scenario",
        nargs="?",
        default="all",
        choices=["1983_drought", "2012_floods", "live", "all"],
        help="Which scenario to run (default: all)",
    )
    p_scenario.add_argument(
        "--interactive",
        action="store_true",
        help="Pause between steps waiting for ENTER",
    )
    add_llm_args(p_scenario)

    # Live
    p_live = subparsers.add_parser("live", help="Run real-time 2026 climate monitor with live NOAA data")
    p_live.add_argument(
        "--interactive",
        action="store_true",
        help="Pause between steps waiting for ENTER",
    )
    add_llm_args(p_live)

    # Sync
    p_sync = subparsers.add_parser("sync", help="Synchronize historical dataset with live NOAA and ERA5 streams")

    args = parser.parse_args()

    if args.command == "decompose":
        cmd_decompose(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "spectral":
        cmd_spectral(args)
    elif args.command == "scenario":
        cmd_scenario(args)
    elif args.command == "live":
        args.scenario = "live"
        cmd_scenario(args)
    elif args.command == "sync":
        cmd_sync(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
