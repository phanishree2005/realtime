"""
Experiment Runner for reproducing the Research Paper Results:
1. Varying Arrival Rate lambda in {20, 30, 40} DAGs/s
2. Varying Critical Ratio Rcrit in {30%, 60%, 90%}
3. Noise / Execution Time Delay robustness testing
4. Multi-Criticality scalability testing (ISO 26262 ASIL levels 1..4)
5. Ablation study
"""

import json
import os
from typing import Dict, List, Optional
import numpy as np

from src.casds.simulator import ACPSSimulator, SimulationResults
from src.casds.schedulers import (
    CASDSScheduler,
    HetSchedScheduler,
    ADSScheduler,
    ASDYSScheduler,
    TD_ADSScheduler,
    ULSScheduler
)


def run_arrival_criticality_experiments(
    num_dags: int = 500,
    lambdas: List[float] = [20.0, 30.0, 40.0],
    rcrits: List[float] = [0.30, 0.60, 0.90],
    seed: int = 42
) -> Dict:
    """
    Runs experiments corresponding to Figure 5 & Figure 6 of the paper.
    """
    sim = ACPSSimulator()
    schedulers = [
        CASDSScheduler(),
        HetSchedScheduler(),
        ADSScheduler(),
        ASDYSScheduler(),
        TD_ADSScheduler(),
        ULSScheduler()
    ]

    all_results = []

    for lam in lambdas:
        for rcrit in rcrits:
            print(f"\n---> Running Scenario: Arrival Rate lambda={lam} DAG/s, Critical Ratio Rcrit={int(rcrit*100)}%")
            # Generate trace with exact parameters
            trace = sim.generate_poisson_trace(
                num_dags=num_dags,
                arrival_rate_lambda=lam,
                rcrit=rcrit,
                deadline=400.0,
                random_seed=seed
            )

            scenario_records = {}
            for sched in schedulers:
                res = sim.run_simulation(sched, trace)
                print(f"  [{sched.name:<8}] DSR: {res.dsr_total:6.2f}% | DSR(Crit): {res.dsr_critical:6.2f}% | MS: {res.makespan/1000:6.2f}s | ET: {res.effective_throughput:5.2f}")
                scenario_records[sched.name] = {
                    'dsr_total': res.dsr_total,
                    'dsr_critical': res.dsr_critical,
                    'dsr_noncritical': res.dsr_noncritical,
                    'makespan': res.makespan,
                    'utilization': res.pe_utilization,
                    'effective_throughput': res.effective_throughput
                }

            all_results.append({
                'lambda': lam,
                'rcrit': rcrit,
                'policies': scenario_records
            })

    output_path = "results_experiments.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll benchmark results saved to {output_path}")
    return all_results


def run_robustness_experiment(
    num_dags: int = 300,
    noise_levels: List[float] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    seed: int = 42
) -> Dict:
    """
    Runs robustness experiments corresponding to Figure 7 (execution time delay noise).
    """
    sim = ACPSSimulator()
    schedulers = [
        CASDSScheduler(),
        HetSchedScheduler(),
        TD_ADSScheduler(),
        ULSScheduler()
    ]

    # Fixed scenario from paper: lambda = 30 DAG/s, Rcrit = 60%
    trace = sim.generate_poisson_trace(
        num_dags=num_dags,
        arrival_rate_lambda=30.0,
        rcrit=0.60,
        deadline=400.0,
        random_seed=seed
    )

    robustness_results = []
    print("\n---> Running Robustness Experiment with Execution Time Delay Noise (Fig. 7)")
    for noise in noise_levels:
        noise_record = {'noise_level': noise, 'policies': {}}
        for sched in schedulers:
            res = sim.run_simulation(sched, trace, noise_delay_factor=noise)
            noise_record['policies'][sched.name] = {
                'dsr_total': res.dsr_total,
                'dsr_critical': res.dsr_critical
            }
        robustness_results.append(noise_record)
        print(f"  Delay Noise {noise*100:3.0f}% -> CASDS DSR: {noise_record['policies']['CASDS']['dsr_total']:.2f}% | DSR(Crit): {noise_record['policies']['CASDS']['dsr_critical']:.2f}%")

    with open("results_robustness.json", 'w') as f:
        json.dump(robustness_results, f, indent=2)
    return robustness_results


if __name__ == '__main__':
    print("=" * 60)
    print("  CASDS Benchmark Suite — Reproducing Paper Results")
    print("=" * 60)

    # Experiment 1: Arrival Rate x Critical Ratio (Fig 5 & Fig 6)
    print("\n[1/2] Running Arrival Rate & Criticality Experiments...")
    results_ac = run_arrival_criticality_experiments(
        num_dags=200,
        lambdas=[20.0, 30.0, 40.0],
        rcrits=[0.30, 0.60, 0.90]
    )
    print(f"  -> Saved {len(results_ac)} scenario results to results_experiments.json")

    # Experiment 2: Robustness under Execution Time Delay Noise (Fig 7)
    print("\n[2/2] Running Robustness / Noise Delay Experiments...")
    results_rob = run_robustness_experiment(
        num_dags=200,
        noise_levels=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    )
    print(f"  -> Saved {len(results_rob)} noise-level results to results_robustness.json")

    print("\n" + "=" * 60)
    print("  All experiments complete! Check:")
    print("  - results_experiments.json  (Fig 5 & 6 data)")
    print("  - results_robustness.json   (Fig 7 data)")
    print("=" * 60)
