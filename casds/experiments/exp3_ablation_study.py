"""
Experiment 3: Ablation Study (Section IV-C & Table II).
Evaluates contributions of CAS-driven urgency U_i^a, Adaptive Priority P_i^a,
and Intra-application ranking strategies (HEFT vs AEFT).
"""

import json
import os
import time
from typing import Dict, List, Any
import numpy as np

from ..benchmarks.adsuite_benchmark import create_adsuite_platform
from ..benchmarks.trace_generator import TraceGenerator
from ..schedulers.ablation_schedulers import AblationScheduler
from ..simulation.simulator import ACPSSimulator


def _std(values):
    """Sample std (ddof=1) when N>=2; population std (ddof=0) for N==1 (variance not estimable)."""
    return float(np.std(values, ddof=1)) if len(values) >= 2 else float(np.std(values, ddof=0))


def _variance_note(n: int) -> str:
    return "" if n >= 2 else "SINGLE_RUN_VARIANCE_NOT_ESTIMABLE"


def run_experiment_3(
    num_dags: int = 150,
    num_runs: int = 3,
    output_dir: str = "results"
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)

    # Configurations listed in Table II:
    configurations = {
        "T_rem in (8)": lambda: AblationScheduler(urgency_mode="Trem", priority_mode="Adaptive", intra_ranking="HEFT"),
        "T-Ddl in [19], [21]": lambda: AblationScheduler(urgency_mode="T-Ddl", priority_mode="Adaptive", intra_ranking="HEFT"),
        "Urgency in [20]": lambda: AblationScheduler(urgency_mode="HetSched", priority_mode="Adaptive", intra_ranking="HEFT"),
        "CAS in (10)": lambda: AblationScheduler(urgency_mode="CAS", priority_mode="Adaptive", intra_ranking="HEFT"),
        "w/o C_i^a": lambda: AblationScheduler(urgency_mode="CAS", priority_mode="w/o_Crit", intra_ranking="HEFT"),
        "Multi-Key Sorting": lambda: AblationScheduler(urgency_mode="CAS", priority_mode="Multikey", intra_ranking="HEFT"),
        "Adaptive Priority": lambda: AblationScheduler(urgency_mode="CAS", priority_mode="Adaptive", intra_ranking="HEFT"),
        "with AEFT": lambda: AblationScheduler(urgency_mode="CAS", priority_mode="Adaptive", intra_ranking="AEFT"),
        "with HEFT": lambda: AblationScheduler(urgency_mode="CAS", priority_mode="Adaptive", intra_ranking="HEFT"),
    }

    results: Dict[str, Any] = {
        "table_ii": {}
    }

    print(f"=== Starting Experiment 3 (Ablation Study - Table II) ===")
    start_t = time.time()

    # Define standard scenarios for average calculation and challenging scenario
    all_scenarios = [
        (20.0, 0.3), (20.0, 0.6), (20.0, 0.9),
        (30.0, 0.3), (30.0, 0.6), (30.0, 0.9),
        (40.0, 0.3), (40.0, 0.6), (40.0, 0.9)
    ]
    challenging_scenario = (40.0, 0.9)

    for cfg_name, factory in configurations.items():
        print(f"Evaluating ablation configuration: {cfg_name}...")
        avg_dsr_list: List[float] = []
        avg_dsr_crit_list: List[float] = []
        chal_dsr_list: List[float] = []
        chal_dsr_crit_list: List[float] = []
        run_log: List[Dict[str, Any]] = []  # per-run seed + raw metrics

        for r in range(num_runs):
            # 1. Run all scenarios for Average
            for lam, rcrit in all_scenarios:
                seed = 7000 + int(lam * 10) + int(rcrit * 10) + r
                tg = TraceGenerator(seed=seed)
                trace = tg.generate_adsuite_trace(num_dags=num_dags, arrival_rate_lambda=lam, critical_ratio=rcrit)

                platform = create_adsuite_platform()
                sim = ACPSSimulator(platform, factory())
                m = sim.run(trace)

                avg_dsr_list.append(m.dsr)
                avg_dsr_crit_list.append(m.dsr_crit)

                run_log.append({"run_idx": r, "seed": seed, "lam": lam, "rcrit": rcrit,
                                 "DSR": m.dsr, "DSR_crit": m.dsr_crit})

                if (lam, rcrit) == challenging_scenario:
                    chal_dsr_list.append(m.dsr)
                    chal_dsr_crit_list.append(m.dsr_crit)

        n_avg = len(avg_dsr_list)
        n_chal = len(chal_dsr_list)
        results["table_ii"][cfg_name] = {
            "Avg_DSR": round(float(np.mean(avg_dsr_list)), 2),
            "Avg_DSR_std": round(_std(avg_dsr_list), 2),
            "Avg_DSR_crit": round(float(np.mean(avg_dsr_crit_list)), 2),
            "Avg_DSR_crit_std": round(_std(avg_dsr_crit_list), 2),
            "Challenging_DSR": round(float(np.mean(chal_dsr_list)), 2),
            "Challenging_DSR_std": round(_std(chal_dsr_list), 2),
            "Challenging_DSR_crit": round(float(np.mean(chal_dsr_crit_list)), 2),
            "Challenging_DSR_crit_std": round(_std(chal_dsr_crit_list), 2),
            "variance_note": _variance_note(n_chal),
            "_runs": run_log,
        }
        print(f"  -> Avg: DSR={results['table_ii'][cfg_name]['Avg_DSR']}%, "
              f"DSR_crit={results['table_ii'][cfg_name]['Avg_DSR_crit']}% | "
              f"Challenging: DSR={results['table_ii'][cfg_name]['Challenging_DSR']}%, "
              f"DSR_crit={results['table_ii'][cfg_name]['Challenging_DSR_crit']}%")

    elapsed = time.time() - start_t
    output_path = os.path.join(output_dir, "exp3_ablation_study.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Experiment 3 complete in {elapsed:.1f}s. Saved to {output_path}")
    return results


if __name__ == "__main__":
    run_experiment_3(num_dags=80, num_runs=2)
