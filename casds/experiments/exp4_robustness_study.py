"""
Experiment 4: Robustness Analysis (Section IV-D, Figures 7 & 8).
Evaluates scheduling performance under:
1. Execution time delays (mu_d in [0.0, 1.0]) (Figure 7)
2. Transient PE failures (lambda_f in [0, 10] failures/s) (Figure 8)
Fixed environment: lambda = 30 DAGs/s, R_crit = 60%.
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


def run_experiment_4(
    num_dags: int = 150,
    num_runs: int = 3,
    output_dir: str = "results"
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    fixed_lambda = 30.0
    fixed_rcrit = 0.6

    schedulers_factory = {
        "ADS": lambda: ADSScheduler(),
        "ASDYS": lambda: ASDYSScheduler(),
        "HetSched": lambda: HetSchedScheduler(),
        "TD_ADS": lambda: TD_ADSScheduler(),
        "ULS": lambda: ULSScheduler(),
        "CASDS": lambda: CASDSScheduler(),
    }

    results: Dict[str, Any] = {
        "execution_delay": {},
        "pe_failure": {}
    }

    print(f"=== Starting Experiment 4 (Robustness Analysis - Figures 7 & 8) ===")
    start_t = time.time()

    # Part 1: Execution Time Delays (Figure 7)
    mu_delays = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    print("--- Evaluating Execution Time Delays (mu_d in [0.0, 1.0]) ---")
    for mu in mu_delays:
        mu_key = f"mu_{int(mu*10)}"
        results["execution_delay"][mu_key] = {}

        metrics_store: Dict[str, Dict[str, List[float]]] = {
            s: {"DSR": [], "DSR_crit": []} for s in schedulers_factory
        }
        mu_run_log: List[Dict[str, Any]] = []  # per-run seed + raw metrics

        for r in range(num_runs):
            seed = 8000 + int(mu * 100) + r
            tg = TraceGenerator(seed=seed)
            trace = tg.generate_adsuite_trace(
                num_dags=num_dags,
                arrival_rate_lambda=fixed_lambda,
                critical_ratio=fixed_rcrit,
                exec_delay_mu=mu
            )

            for s_name, factory in schedulers_factory.items():
                platform = create_adsuite_platform()
                sim = ACPSSimulator(platform, factory())
                m = sim.run(trace)
                metrics_store[s_name]["DSR"].append(m.dsr)
                metrics_store[s_name]["DSR_crit"].append(m.dsr_crit)

            mu_run_log.append({
                "run_idx": r, "seed": seed, "mu_d": mu,
                "schedulers": {s: {"DSR": metrics_store[s]["DSR"][-1],
                                   "DSR_crit": metrics_store[s]["DSR_crit"][-1]}
                               for s in schedulers_factory}
            })

        for s_name in schedulers_factory:
            n = len(metrics_store[s_name]["DSR"])
            results["execution_delay"][mu_key][s_name] = {
                "DSR_mean": float(np.mean(metrics_store[s_name]["DSR"])),
                "DSR_std": _std(metrics_store[s_name]["DSR"]),
                "DSR_crit_mean": float(np.mean(metrics_store[s_name]["DSR_crit"])),
                "DSR_crit_std": _std(metrics_store[s_name]["DSR_crit"]),
                "variance_note": _variance_note(n),
            }

        results["execution_delay"][mu_key]["_runs"] = mu_run_log

        print(f"Delay mu={mu:.1f} -> CASDS DSR={results['execution_delay'][mu_key]['CASDS']['DSR_mean']:.2f}% | "
              f"TD_ADS DSR={results['execution_delay'][mu_key]['TD_ADS']['DSR_mean']:.2f}%")

    # Part 2: PE Failure Rates (Figure 8)
    lambda_failures = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    print("--- Evaluating PE Failure Rates (lambda_f in [0, 10]) ---")
    for lam_f in lambda_failures:
        fail_key = f"lambda_f_{int(lam_f)}"
        results["pe_failure"][fail_key] = {}

        metrics_store: Dict[str, Dict[str, List[float]]] = {
            s: {"DSR": [], "DSR_crit": []} for s in schedulers_factory
        }
        fail_run_log: List[Dict[str, Any]] = []  # per-run seed + raw metrics

        for r in range(num_runs):
            seed = 9000 + int(lam_f * 10) + r
            tg = TraceGenerator(seed=seed)
            trace = tg.generate_adsuite_trace(
                num_dags=num_dags,
                arrival_rate_lambda=fixed_lambda,
                critical_ratio=fixed_rcrit
            )
            # Estimate total simulation time approx num_dags / lambda * 1000 ms
            est_total_ms = (num_dags / fixed_lambda) * 1000.0 + 1000.0
            failures = tg.generate_pe_failure_events(
                total_simulation_time=est_total_ms,
                num_pes=13,
                failure_rate_lambda_f=lam_f
            )

            for s_name, factory in schedulers_factory.items():
                platform = create_adsuite_platform()
                sim = ACPSSimulator(platform, factory())
                m = sim.run(trace, pe_failures=failures)
                metrics_store[s_name]["DSR"].append(m.dsr)
                metrics_store[s_name]["DSR_crit"].append(m.dsr_crit)

            fail_run_log.append({
                "run_idx": r, "seed": seed, "lambda_f": lam_f,
                "schedulers": {s: {"DSR": metrics_store[s]["DSR"][-1],
                                   "DSR_crit": metrics_store[s]["DSR_crit"][-1]}
                               for s in schedulers_factory}
            })

        for s_name in schedulers_factory:
            n = len(metrics_store[s_name]["DSR"])
            results["pe_failure"][fail_key][s_name] = {
                "DSR_mean": float(np.mean(metrics_store[s_name]["DSR"])),
                "DSR_std": _std(metrics_store[s_name]["DSR"]),
                "DSR_crit_mean": float(np.mean(metrics_store[s_name]["DSR_crit"])),
                "DSR_crit_std": _std(metrics_store[s_name]["DSR_crit"]),
                "variance_note": _variance_note(n),
            }

        results["pe_failure"][fail_key]["_runs"] = fail_run_log

        print(f"Fail rate lambda_f={lam_f:.1f} -> CASDS DSR={results['pe_failure'][fail_key]['CASDS']['DSR_mean']:.2f}% | "
              f"TD_ADS DSR={results['pe_failure'][fail_key]['TD_ADS']['DSR_mean']:.2f}%")

    elapsed = time.time() - start_t
    output_path = os.path.join(output_dir, "exp4_robustness_study.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Experiment 4 complete in {elapsed:.1f}s. Saved to {output_path}")
    return results


if __name__ == "__main__":
    run_experiment_4(num_dags=80, num_runs=2)
