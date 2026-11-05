"""
Experiment 6: Real Hardware SelfPilot Experiment (Section V & Figure 10).
Evaluates DSR(S), DSR(S_crit), and Effective Throughput ET(S)
on the real vehicle hardware platform (TZTEK GEACX2 + Advantech MIC-75G30)
with SelfPilot autonomous driving application DAG across:
- Arrival rate lambda in {5, 10, 20, 30, 40} DAGs/s
- Critical ratio R_crit in {30%, 60%, 90%}
- Schedulers: ADS, ASDYS, HetSched, TD_ADS, CASDS
"""

import json
import os
import time
from typing import Dict, List, Any
import numpy as np

from ..benchmarks.selfpilot_benchmark import create_selfpilot_platform
from ..benchmarks.trace_generator import TraceGenerator
from ..schedulers.casds_scheduler import CASDSScheduler
from ..schedulers.baselines import (
    ADSScheduler, ASDYSScheduler, HetSchedScheduler, TD_ADSScheduler
)
from ..simulation.simulator import ACPSSimulator


def _std(values):
    """Sample std (ddof=1) when N>=2; population std (ddof=0) for N==1 (variance not estimable)."""
    return float(np.std(values, ddof=1)) if len(values) >= 2 else float(np.std(values, ddof=0))


def _variance_note(n: int) -> str:
    return "" if n >= 2 else "SINGLE_RUN_VARIANCE_NOT_ESTIMABLE"


def run_experiment_6(
    num_dags: int = 150,
    num_runs: int = 3,
    output_dir: str = "results"
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    lambdas = [5.0, 10.0, 20.0, 30.0, 40.0]
    critical_ratios = [0.3, 0.6, 0.9]

    schedulers_factory = {
        "ADS": lambda: ADSScheduler(),
        "ASDYS": lambda: ASDYSScheduler(),
        "HetSched": lambda: HetSchedScheduler(),
        "TD_ADS": lambda: TD_ADSScheduler(),
        "CASDS": lambda: CASDSScheduler(),
    }

    results: Dict[str, Any] = {
        "metadata": {
            "num_dags": num_dags,
            "num_runs": num_runs,
            "lambdas": lambdas,
            "critical_ratios": critical_ratios,
            "schedulers": list(schedulers_factory.keys())
        },
        "data": {}
    }

    print(f"=== Starting Experiment 6 (Real Hardware SelfPilot - Figure 10) ===")
    start_t = time.time()

    for rcrit in critical_ratios:
        rcrit_key = f"rcrit_{int(rcrit*100)}"
        results["data"][rcrit_key] = {}

        for lam in lambdas:
            lam_key = f"lambda_{int(lam)}"
            results["data"][rcrit_key][lam_key] = {}

            metrics_store: Dict[str, Dict[str, List[float]]] = {
                s: {"DSR": [], "DSR_crit": [], "ET": []} for s in schedulers_factory
            }
            lam_run_log: List[Dict[str, Any]] = []  # per-run seed + raw metrics

            for r in range(num_runs):
                seed = 3000 + int(rcrit * 100) + int(lam * 10) + r
                tg = TraceGenerator(seed=seed)
                trace = tg.generate_selfpilot_trace(
                    num_dags=num_dags,
                    arrival_rate_lambda=lam,
                    critical_ratio=rcrit
                )

                for s_name, factory in schedulers_factory.items():
                    platform = create_selfpilot_platform()
                    sim = ACPSSimulator(platform, factory())
                    m = sim.run(trace)
                    metrics_store[s_name]["DSR"].append(m.dsr)
                    metrics_store[s_name]["DSR_crit"].append(m.dsr_crit)
                    metrics_store[s_name]["ET"].append(m.effective_throughput)

                lam_run_log.append({
                    "run_idx": r, "seed": seed,
                    "schedulers": {s: {"DSR": metrics_store[s]["DSR"][-1],
                                       "DSR_crit": metrics_store[s]["DSR_crit"][-1],
                                       "ET": metrics_store[s]["ET"][-1]}
                                   for s in schedulers_factory}
                })

            for s_name in schedulers_factory:
                n = len(metrics_store[s_name]["DSR"])
                results["data"][rcrit_key][lam_key][s_name] = {
                    "DSR_mean": float(np.mean(metrics_store[s_name]["DSR"])),
                    "DSR_std": _std(metrics_store[s_name]["DSR"]),
                    "DSR_crit_mean": float(np.mean(metrics_store[s_name]["DSR_crit"])),
                    "DSR_crit_std": _std(metrics_store[s_name]["DSR_crit"]),
                    "ET_mean": float(np.mean(metrics_store[s_name]["ET"])),
                    "ET_std": _std(metrics_store[s_name]["ET"]),
                    "variance_note": _variance_note(n),
                }

            results["data"][rcrit_key][lam_key]["_runs"] = lam_run_log

            print(f"SelfPilot: Rcrit={int(rcrit*100)}%, lambda={lam} -> "
                  f"CASDS DSR={results['data'][rcrit_key][lam_key]['CASDS']['DSR_mean']:.2f}%, "
                  f"ET={results['data'][rcrit_key][lam_key]['CASDS']['ET_mean']:.2f} DAG/s | "
                  f"TD_ADS DSR={results['data'][rcrit_key][lam_key]['TD_ADS']['DSR_mean']:.2f}%")

    elapsed = time.time() - start_t
    output_path = os.path.join(output_dir, "exp6_selfpilot_hardware.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Experiment 6 complete in {elapsed:.1f}s. Saved to {output_path}")
    return results


if __name__ == "__main__":
    run_experiment_6(num_dags=80, num_runs=2)
