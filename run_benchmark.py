"""Benchmark the system over many random scenes.

Compares:
  - pick-order strategies: naive (vision order), greedy, optimal
  - controller: PID alone vs PID + model-based feedforward

Usage:  python run_benchmark.py [--trials 50]
Writes results/benchmark.json and prints a summary table.
"""
import argparse
import json
import os

import numpy as np

from pickplace import sim


def run(strategy, feedforward, trials):
    rows = []
    for seed in range(trials):
        rng = np.random.default_rng(seed)
        objects = sim.random_scene(rng)
        rows.append(sim.Episode(objects, rng, strategy=strategy, feedforward=feedforward).run())
    n_obj = sum(r["n_objects"] for r in rows)
    return {
        "strategy": strategy,
        "feedforward": feedforward,
        "trials": trials,
        "objects": n_obj,
        "success_rate_pct": 100 * sum(r["placed_ok"] for r in rows) / n_obj,
        "mean_cycle_time_s": float(np.mean([r["cycle_time_s"] for r in rows])),
        "mean_time_per_object_s": float(sum(r["cycle_time_s"] for r in rows) / n_obj),
        "rms_tracking_error_mm": float(np.mean([r["rms_tracking_error_mm"] for r in rows])),
        "max_tracking_error_mm": float(np.max([r["max_tracking_error_mm"] for r in rows])),
        "mean_localisation_error_mm": float(np.mean([r["mean_localisation_error_mm"] for r in rows])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    args = parser.parse_args()

    configs = [("naive", True), ("greedy", True), ("optimal", True), ("greedy", False)]
    results = []
    for strategy, ff in configs:
        print(f"running {strategy:8s} feedforward={ff} ...", flush=True)
        results.append(run(strategy, ff, args.trials))

    os.makedirs("results", exist_ok=True)
    with open("results/benchmark.json", "w") as f:
        json.dump(results, f, indent=2)

    print()
    print(f"{'order':8s} {'feedfwd':8s} {'success%':>9s} {'time/obj s':>11s} "
          f"{'RMS track mm':>13s} {'max track mm':>13s} {'vision err mm':>14s}")
    for r in results:
        print(f"{r['strategy']:8s} {str(r['feedforward']):8s} {r['success_rate_pct']:9.1f} "
              f"{r['mean_time_per_object_s']:11.2f} {r['rms_tracking_error_mm']:13.2f} "
              f"{r['max_tracking_error_mm']:13.2f} {r['mean_localisation_error_mm']:14.2f}")
    naive, greedy, optimal = results[0], results[1], results[2]
    saving = 100 * (1 - greedy["mean_cycle_time_s"] / naive["mean_cycle_time_s"])
    print(f"\nGreedy ordering cuts cycle time by {saving:.1f}% vs naive "
          f"(optimal: {100 * (1 - optimal['mean_cycle_time_s'] / naive['mean_cycle_time_s']):.1f}%).")


if __name__ == "__main__":
    main()
