"""
Experiment 2: System Efficiency Comparison (Section IV-B & Figure 6).
Evaluates Makespan MS(S), Utilization U(S, P), and Effective Throughput ET(S)
across arrival rates lambda in {20, 25, 30, 35, 40} DAGs/s.
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


def run_experiment_2(
    num_dags: int = 150,
    num_runs: int = 3,
    output_dir: str = "results"
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    lambdas = [20.0, 25.0, 30.0, 35.0, 40.0]
    critical_ratio = 0.6  # Standard baseline scenario

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
            "critical_ratio": critical_ratio,
            "schedulers": list(schedulers_factory.keys()),
            "std_method": "sample_ddof1_when_N_ge_2_else_population_ddof0"
        },
        "by_lambda": {},
        "overall_summary": {}
    }

    print(f"=== Starting Experiment 2 (System Efficiency & Effective Throughput) ===")
    start_t = time.time()

    all_metrics: Dict[str, Dict[str, List[float]]] = {
        s: {"Makespan": [], "Utilization": [], "ET": [], "DSR": []}
        for s in schedulers_factory
    }

    for lam in lambdas:
        lam_key = f"lambda_{int(lam)}"
        results["by_lambda"][lam_key] = {}

        lam_metrics: Dict[str, Dict[str, List[float]]] = {
            s: {"Makespan": [], "Utilization": [], "ET": []}
            for s in schedulers_factory
        }
        lam_run_log: List[Dict[str, Any]] = []  # per-run seed + raw metrics

        for r in range(num_runs):
            seed = 5000 + int(lam * 10) + r
            tg = TraceGenerator(seed=seed)
            trace = tg.generate_adsuite_trace(
                num_dags=num_dags,
                arrival_rate_lambda=lam,
                critical_ratio=critical_ratio,
                relative_deadline=400.0
            )

            run_entry: Dict[str, Any] = {"run_idx": r, "seed": seed, "schedulers": {}}

            for s_name, factory in schedulers_factory.items():
                platform = create_adsuite_platform()
                sim = ACPSSimulator(platform, factory())
                m = sim.run(trace)

                lam_metrics[s_name]["Makespan"].append(m.makespan_s)
                lam_metrics[s_name]["Utilization"].append(m.utilization)
                lam_metrics[s_name]["ET"].append(m.effective_throughput)

                all_metrics[s_name]["Makespan"].append(m.makespan_s)
                all_metrics[s_name]["Utilization"].append(m.utilization)
                all_metrics[s_name]["ET"].append(m.effective_throughput)
                all_metrics[s_name]["DSR"].append(m.dsr)

                run_entry["schedulers"][s_name] = {
                    "DSR": m.dsr,
                    "Makespan_s": m.makespan_s,
                    "Utilization": m.utilization,
                    "ET": m.effective_throughput,
                }

            lam_run_log.append(run_entry)

        for s_name in schedulers_factory:
            n = len(lam_metrics[s_name]["Makespan"])
            results["by_lambda"][lam_key][s_name] = {
                "Makespan_mean": float(np.mean(lam_metrics[s_name]["Makespan"])),
                "Makespan_std": _std(lam_metrics[s_name]["Makespan"]),
                "Utilization_mean": float(np.mean(lam_metrics[s_name]["Utilization"])),
                "Utilization_std": _std(lam_metrics[s_name]["Utilization"]),
                "ET_mean": float(np.mean(lam_metrics[s_name]["ET"])),
                "ET_std": _std(lam_metrics[s_name]["ET"]),
                "variance_note": _variance_note(n),
            }

        results["by_lambda"][lam_key]["_runs"] = lam_run_log

        print(f"Lambda={lam} -> CASDS ET={results['by_lambda'][lam_key]['CASDS']['ET_mean']:.2f} DAG/s | "
              f"HetSched ET={results['by_lambda'][lam_key]['HetSched']['ET_mean']:.2f} DAG/s")

    for s_name in schedulers_factory:
        n_all = len(all_metrics[s_name]["Makespan"])
        results["overall_summary"][s_name] = {
            "Makespan_mean": float(np.mean(all_metrics[s_name]["Makespan"])),
            "Makespan_std": _std(all_metrics[s_name]["Makespan"]),
            "Utilization_mean": float(np.mean(all_metrics[s_name]["Utilization"])),
            "Utilization_std": _std(all_metrics[s_name]["Utilization"]),
            "ET_mean": float(np.mean(all_metrics[s_name]["ET"])),
            "ET_std": _std(all_metrics[s_name]["ET"]),
            "variance_note": _variance_note(n_all),
        }

    elapsed = time.time() - start_t
    results["metadata"]["elapsed_seconds"] = round(elapsed, 2)
    output_path = os.path.join(output_dir, "exp2_system_efficiency.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Experiment 2 complete in {elapsed:.1f}s. Saved to {output_path}")
    return results


if __name__ == "__main__":
    run_experiment_2(num_dags=100, num_runs=2)
