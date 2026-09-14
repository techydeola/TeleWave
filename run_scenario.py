#!/usr/bin/env python3
"""
TeleWave — Real-World Scenario Walkthrough Launcher
==========================================================

Usage:
  python3 run_scenario.py [scenario] [--interactive]

Scenarios:
  1983_drought   The 1982-83 El Niño Drought (Sahel / Nigeria)
  2012_floods    The 2012 Niger Delta Floods
  all            Run both scenarios in sequence

Flags:
  --interactive  Pause between each step waiting for ENTER
                 (default: runs straight through)

Examples:
  python3 run_scenario.py 1983_drought
  python3 run_scenario.py 2012_floods --interactive
  python3 run_scenario.py all
"""

import sys
import os

# Ensure project root is on the path
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from climatewave.scenarios.walkthrough import ScenarioWalkthrough

VALID_SCENARIOS = ["1983_drought", "2012_floods", "live", "all"]

HELP = """
TeleWave — Scenario Walkthrough

  python3 run_scenario.py [scenario] [--interactive]

Scenarios:
  1983_drought   The 1982-83 El Niño Drought
  2012_floods    The 2012 Niger Delta Floods
  live           Real-time 2026 climate monitor (live NOAA + ERA5)
  all            Run all scenarios

Flags:
  --interactive  Pause between steps for ENTER key
  --api-key KEY  AgentRouter or OpenAI-compatible API key
  --model MODEL  LLM model name (e.g. gpt-4o, llama-3.3-70b-versatile)
  --base-url URL Custom API base URL (default: https://agentrouter.org/v1)
  --no-ai        Disable AI interpretation and interactive setup

Examples:
  python3 run_scenario.py live
  python3 run_scenario.py live --api-key "sk-..." --model "gpt-4o"
  python3 run_scenario.py 1983_drought
  python3 run_scenario.py 2012_floods --interactive
  python3 run_scenario.py all
"""


def main():
    args = sys.argv[1:]

    if not args or "--help" in args or "-h" in args:
        print(HELP)
        sys.exit(0)

    interactive = "--interactive" in args
    no_ai = "--no-ai" in args
    
    # Parse optional API credentials
    api_key = None
    base_url = None
    model = None
    
    i = 0
    clean_args = []
    while i < len(args):
        a = args[i]
        if a == "--api-key" and i + 1 < len(args):
            api_key = args[i + 1]
            i += 2
        elif a == "--base-url" and i + 1 < len(args):
            base_url = args[i + 1]
            i += 2
        elif a == "--model" and i + 1 < len(args):
            model = args[i + 1]
            i += 2
        elif a.startswith("--"):
            i += 1
        else:
            clean_args.append(a)
            i += 1

    scenario = clean_args[0] if clean_args else "all"
    if scenario not in VALID_SCENARIOS:
        print(f"  Unknown scenario: '{scenario}'")
        print(f"  Valid options: {', '.join(VALID_SCENARIOS)}")
        sys.exit(1)

    sw = ScenarioWalkthrough(
        interactive=interactive,
        api_key=api_key,
        base_url=base_url,
        model=model,
        no_ai=no_ai,
    )

    if scenario == "all":
        print("\n" + "=" * 78)
        print("  TELEWAVE: REAL-WORLD SCENARIO WALKTHROUGHS".center(78))
        print("  Two historically verified Nigerian climate disasters".center(78))
        print("=" * 78)
        sw.run("1983_drought")
        print("\n" + "─" * 78)
        print("  Starting Scenario 2 in 2 seconds...".center(78))
        print("─" * 78 + "\n")
        import time; time.sleep(2)
        sw.run("2012_floods")
    else:
        sw.run(scenario)


if __name__ == "__main__":
    main()
