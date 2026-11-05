"""
Experiment 1: Simulation Benchmark (Section IV-B & Figure 5).
Compares DSR(S) and DSR(S_crit) across:
- Arrival rate lambda in {20, 30, 40} DAGs/s
- Critical ratio R_crit in {30%, 60%, 90%}
- Schedulers: ADS, ASDYS, HetSched, TD_ADS, ULS, CASDS
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


def run_experiment_1(
    num_dags: int = 200,
    num_runs: int = 5,
    output_dir: str = "results"
) -> Dict[str, Any]:
    """
    Runs Experiment 1 over grid of lambda and R_crit.
    """
    os.makedirs(output_dir, exist_ok=True)
    lambdas = [20.0, 30.0, 40.0]
    critical_ratios = [0.3, 0.6, 0.9]

    schedulers_factory = {
        "ADS": lambda: ADSScheduler(),
        "ASDYS": lambda: ASDYSScheduler(),
        "HetSched": lambda: HetSchedScheduler(),
        "TD_ADS": lambda: TD_ADSScheduler(),
        "ULS": lambda: ULSScheduler(),
        "CASDS": lambda: CASDSScheduler(),
    }

    results: Dict[str, Any] = {
        "metadata": {
            "num_dags": num_dags,
            "num_runs": num_runs,
            "lambdas": lambdas,
            "critical_ratios": critical_ratios,
            "schedulers": list(schedulers_factory.keys()),
            "std_method": "sample_ddof1_when_N_ge_2_else_population_ddof0"
        },
        "data": {}
    }

    total_runs = len(lambdas) * len(critical_ratios) * num_runs * len(schedulers_factory)
    print(f"=== Starting Experiment 1 (Simulation Benchmark) ===")
    print(f"Total simulation runs: {total_runs} (dags per trace: {num_dags})")

    start_t = time.time()
    run_idx = 0

    for lam in lambdas:
        for rcrit in critical_ratios:
            scenario_key = f"lambda_{int(lam)}_rcrit_{int(rcrit*100)}"
            results["data"][scenario_key] = {}

            # Storage for runs
            run_metrics: Dict[str, Dict[str, List[float]]] = {
                s_name: {"DSR": [], "DSR_crit": [], "Makespan": [], "Utilization": [], "ET": []}
                for s_name in schedulers_factory
            }
            run_log: List[Dict[str, Any]] = []  # per-run seed + raw metrics

            for r in range(num_runs):
                seed = 1000 + int(lam * 100) + int(rcrit * 10) + r
                tg = TraceGenerator(seed=seed)
                # Generate exact same trace for all schedulers to compare fairly
                base_trace = tg.generate_adsuite_trace(
                    num_dags=num_dags,
                    arrival_rate_lambda=lam,
                    critical_ratio=rcrit,
                    relative_deadline=400.0
                )

                run_entry: Dict[str, Any] = {"run_idx": r, "seed": seed, "schedulers": {}}

                for s_name, factory in schedulers_factory.items():
                    run_idx += 1
                    platform = create_adsuite_platform()
                    scheduler = factory()
                    sim = ACPSSimulator(platform, scheduler)
                    m = sim.run(base_trace)

                    run_metrics[s_name]["DSR"].append(m.dsr)
                    run_metrics[s_name]["DSR_crit"].append(m.dsr_crit)
                    run_metrics[s_name]["Makespan"].append(m.makespan_ms)
                    run_metrics[s_name]["Utilization"].append(m.utilization)
                    run_metrics[s_name]["ET"].append(m.effective_throughput)

                    run_entry["schedulers"][s_name] = {
                        "DSR": m.dsr,
                        "DSR_crit": m.dsr_crit,
                        "Makespan_ms": m.makespan_ms,
                        "Utilization": m.utilization,
                        "ET": m.effective_throughput,
                    }

                run_log.append(run_entry)

            # Compute means and sample stds (ddof=1 when N>=2)
            for s_name in schedulers_factory:
                n = len(run_metrics[s_name]["DSR"])
                results["data"][scenario_key][s_name] = {
                    "DSR_mean": float(np.mean(run_metrics[s_name]["DSR"])),
                    "DSR_std": _std(run_metrics[s_name]["DSR"]),
                    "DSR_crit_mean": float(np.mean(run_metrics[s_name]["DSR_crit"])),
                    "DSR_crit_std": _std(run_metrics[s_name]["DSR_crit"]),
                    "Makespan_mean": float(np.mean(run_metrics[s_name]["Makespan"])),
                    "Makespan_std": _std(run_metrics[s_name]["Makespan"]),
                    "Utilization_mean": float(np.mean(run_metrics[s_name]["Utilization"])),
                    "Utilization_std": _std(run_metrics[s_name]["Utilization"]),
                    "ET_mean": float(np.mean(run_metrics[s_name]["ET"])),
                    "ET_std": _std(run_metrics[s_name]["ET"]),
                    "variance_note": _variance_note(n),
                }

            results["data"][scenario_key]["_runs"] = run_log

            print(f"Completed scenario: lambda={lam}, Rcrit={int(rcrit*100)}% -> "
                  f"CASDS DSR={results['data'][scenario_key]['CASDS']['DSR_mean']:.2f}% | "
                  f"ULS DSR={results['data'][scenario_key]['ULS']['DSR_mean']:.2f}% | "
                  f"HetSched DSR={results['data'][scenario_key]['HetSched']['DSR_mean']:.2f}%")

    elapsed = time.time() - start_t
    results["metadata"]["elapsed_seconds"] = round(elapsed, 2)

    output_path = os.path.join(output_dir, "exp1_simulation_benchmark.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Experiment 1 complete in {elapsed:.1f}s. Saved to {output_path}")
    return results


if __name__ == "__main__":
    run_experiment_1(num_dags=100, num_runs=3)
