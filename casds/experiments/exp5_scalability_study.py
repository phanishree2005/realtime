"""
Experiment 5: Scalability Analysis of Multiple Criticality Levels (Section IV-E & Table III).
Evaluates DSR across 4 ISO 26262 ASIL criticality levels (S_4, S_3, S_2, S_1)
under arrival rates lambda in {20, 40} DAGs/s.
"""

import json
import os
import time
from typing import Dict, List, Any
import numpy as np

from ..benchmarks.adsuite_benchmark import create_adsuite_platform
from ..benchmarks.trace_generator import TraceGenerator
from ..schedulers.casds_scheduler import CASDSScheduler
from ..schedulers.baselines import (
    ADSScheduler, ASDYSScheduler, HetSchedScheduler, TD_ADSScheduler, ULSScheduler
)
from ..simulation.simulator import ACPSSimulator


def _std(values):
    """Sample std (ddof=1) when N>=2; population std (ddof=0) for N==1 (variance not estimable)."""
    return float(np.std(values, ddof=1)) if len(values) >= 2 else float(np.std(values, ddof=0))


def _variance_note(n: int) -> str:
    return "" if n >= 2 else "SINGLE_RUN_VARIANCE_NOT_ESTIMABLE"


def run_experiment_5(
    num_dags: int = 150,
    num_runs: int = 3,
    output_dir: str = "results"
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    lambdas = [20.0, 40.0]

    schedulers_factory = {
        "ADS": lambda: ADSScheduler(),
        "ASDYS": lambda: ASDYSScheduler(),
        "HetSched": lambda: HetSchedScheduler(),
        "TD_ADS": lambda: TD_ADSScheduler(),
        "ULS": lambda: ULSScheduler(),
        "CASDS": lambda: CASDSScheduler(),
    }

    results: Dict[str, Any] = {
        "table_iii": {}
    }

    print(f"=== Starting Experiment 5 (Scalability Study - Table III) ===")
    start_t = time.time()

    for lam in lambdas:
        lam_key = f"lambda_{int(lam)}"
        results["table_iii"][lam_key] = {}

        metrics_store: Dict[str, Dict[str, List[float]]] = {
            s: {"Overall": [], "S4": [], "S3": [], "S2": [], "S1": []}
            for s in schedulers_factory
        }
        lam_run_log: List[Dict[str, Any]] = []  # per-run seed + raw metrics

        for r in range(num_runs):
            seed = 6000 + int(lam * 10) + r
            tg = TraceGenerator(seed=seed)
            # Generate trace with 4 criticality levels {1, 2, 3, 4} in equal proportion
            trace = tg.generate_adsuite_trace(
                num_dags=num_dags,
                arrival_rate_lambda=lam,
                multi_criticality=True
            )

            for s_name, factory in schedulers_factory.items():
                platform = create_adsuite_platform()
                sim = ACPSSimulator(platform, factory())
                m = sim.run(trace)

                metrics_store[s_name]["Overall"].append(m.dsr)
                metrics_store[s_name]["S4"].append(m.dsr_levels.get(4, 0.0))
                metrics_store[s_name]["S3"].append(m.dsr_levels.get(3, 0.0))
                metrics_store[s_name]["S2"].append(m.dsr_levels.get(2, 0.0))
                metrics_store[s_name]["S1"].append(m.dsr_levels.get(1, 0.0))

            lam_run_log.append({
                "run_idx": r, "seed": seed,
                "schedulers": {s: {"Overall": metrics_store[s]["Overall"][-1],
                                   "S4": metrics_store[s]["S4"][-1],
                                   "S3": metrics_store[s]["S3"][-1],
                                   "S2": metrics_store[s]["S2"][-1],
                                   "S1": metrics_store[s]["S1"][-1]}
                               for s in schedulers_factory}
            })

        for s_name in schedulers_factory:
            n = len(metrics_store[s_name]["Overall"])
            results["table_iii"][lam_key][s_name] = {
                "Overall": round(float(np.mean(metrics_store[s_name]["Overall"])), 2),
                "Overall_std": round(_std(metrics_store[s_name]["Overall"]), 2),
                "S4": round(float(np.mean(metrics_store[s_name]["S4"])), 2),
                "S3": round(float(np.mean(metrics_store[s_name]["S3"])), 2),
                "S2": round(float(np.mean(metrics_store[s_name]["S2"])), 2),
                "S1": round(float(np.mean(metrics_store[s_name]["S1"])), 2),
                "variance_note": _variance_note(n),
            }

        results["table_iii"][lam_key]["_runs"] = lam_run_log

        print(f"Table III (lambda={lam}):")
        for s_name in schedulers_factory:
            d = results["table_iii"][lam_key][s_name]
            print(f"  {s_name:<9}: Overall={d['Overall']}% | S4={d['S4']}% | S3={d['S3']}% | S2={d['S2']}% | S1={d['S1']}%")

    elapsed = time.time() - start_t
    output_path = os.path.join(output_dir, "exp5_scalability_study.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Experiment 5 complete in {elapsed:.1f}s. Saved to {output_path}")
    return results


if __name__ == "__main__":
    run_experiment_5(num_dags=80, num_runs=2)
