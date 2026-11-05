"""
Master Runner for All CASDS Experiments & Plots.
Executes Experiments 1-6, saves benchmark datasets, generates all paper figures (Fig 5, 6, 7, 8, 10),
and prints formatted Markdown & console tables (Table I, II, III, IV).
"""

import os
import sys
import time

from .exp1_simulation_benchmark import run_experiment_1
from .exp2_system_efficiency import run_experiment_2
from .exp3_ablation_study import run_experiment_3
from .exp4_robustness_study import run_experiment_4
from .exp5_scalability_study import run_experiment_5
from .exp6_real_hardware_selfpilot import run_experiment_6
from ..visualization.plot_generator import (
    plot_fig5_dsr,
    plot_fig6_efficiency,
    plot_fig7_fig8_robustness,
    plot_fig10_selfpilot
)


def print_table_i():
    print("\n" + "="*70)
    print("TABLE I: TIME PROFILE OF TASK KERNELS ON EACH PE-TYPE (ADSuite Benchmark)")
    print("="*70)
    print(f"{'Task Kernel':<24} | {'CPU [ms]':<10} | {'GPU [ms]':<10} | {'ASIC [ms]':<10}")
    print("-" * 70)
    print(f"{'Object Detection (DET)':<24} | {'3531':<10} | {'156':<10} | {'96':<10}")
    print(f"{'Object Tracking (TRA)':<24} | {'1825':<10} | {'17':<10} | {'2':<10}")
    print(f"{'Localization (LOC)':<24} | {'165':<10} | {'95':<10} | {'10':<10}")
    print(f"{'Fusion (FUS)':<24} | {'0.1':<10} | {'-':<10} | {'-':<10}")
    print(f"{'Mission Planning (MiP)':<24} | {'1':<10} | {'-':<10} | {'-':<10}")
    print(f"{'Motion Planning (MoP)':<24} | {'8':<10} | {'-':<10} | {'-':<10}")
    print("="*70 + "\n")


def print_table_iv():
    print("\n" + "="*70)
    print("TABLE IV: TIME PROFILE OF SELFPILOT TASK KERNELS ON EACH PE-TYPE")
    print("="*70)
    print(f"{'Task Kernel':<30} | {'CPU [ms]':<10} | {'GPU1 [ms]':<10} | {'GPU2 [ms]':<10}")
    print("-" * 70)
    print(f"{'LiDAR Object Detection (LOD)':<30} | {'60':<10} | {'17':<10} | {'-':<10}")
    print(f"{'Camera Object Detection (COD)':<30} | {'62':<10} | {'25':<10} | {'-':<10}")
    print(f"{'Object Detection Fusion (ODF)':<30} | {'7':<10} | {'-':<10} | {'-':<10}")
    print(f"{'Lane Detection (LD)':<30} | {'22':<10} | {'9':<10} | {'-':<10}")
    print(f"{'Multiple Object Tracking (MOT)':<30} | {'29':<10} | {'-':<10} | {'-':<10}")
    print(f"{'Localization (LOC)':<30} | {'60':<10} | {'23':<10} | {'8':<10}")
    print(f"{'Mission Planning (MiP)':<30} | {'1':<10} | {'-':<10} | {'-':<10}")
    print(f"{'Motion Planning (MoP)':<30} | {'69':<10} | {'-':<10} | {'-':<10}")
    print(f"{'Control (CON)':<30} | {'11':<10} | {'-':<10} | {'-':<10}")
    print("="*70 + "\n")


def run_full_pipeline(num_dags: int = 100, num_runs: int = 2, results_dir: str = "results"):
    """
    Executes complete paper reproduction pipeline:
    - Prints benchmark profiles (Table I, Table IV)
    - Runs Experiments 1 to 6
    - Plots Figures 5, 6, 7, 8, 10
    """
    total_start = time.time()
    figures_dir = os.path.join(results_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    print("="*80)
    print(" REPRODUCING PAPER: CONTEXT-AWARE-SLACK-DRIVEN DYNAMIC SCHEDULING (CASDS)")
    print("="*80)

    # 1. Platform & Kernel profiles
    print_table_i()
    print_table_iv()

    # 2. Experiment 1: Simulation Benchmark (Fig 5)
    print("\n>>> Running Experiment 1: Simulation Benchmark (Fig 5)...")
    exp1_res = run_experiment_1(num_dags=num_dags, num_runs=num_runs, output_dir=results_dir)
    plot_fig5_dsr(os.path.join(results_dir, "exp1_simulation_benchmark.json"), os.path.join(figures_dir, "fig5_dsr_comparison.png"))

    # 3. Experiment 2: System Efficiency (Fig 6)
    print("\n>>> Running Experiment 2: System Efficiency (Fig 6)...")
    exp2_res = run_experiment_2(num_dags=num_dags, num_runs=num_runs, output_dir=results_dir)
    plot_fig6_efficiency(os.path.join(results_dir, "exp2_system_efficiency.json"), os.path.join(figures_dir, "fig6_system_efficiency.png"))

    # 4. Experiment 3: Ablation Study (Table II)
    print("\n>>> Running Experiment 3: Ablation Study (Table II)...")
    exp3_res = run_experiment_3(num_dags=max(60, num_dags // 2), num_runs=num_runs, output_dir=results_dir)

    # 5. Experiment 4: Robustness Analysis (Fig 7 & 8)
    print("\n>>> Running Experiment 4: Robustness Analysis (Fig 7 & 8)...")
    exp4_res = run_experiment_4(num_dags=max(60, num_dags // 2), num_runs=num_runs, output_dir=results_dir)
    plot_fig7_fig8_robustness(
        os.path.join(results_dir, "exp4_robustness_study.json"),
        os.path.join(figures_dir, "fig7_execution_delay_robustness.png"),
        os.path.join(figures_dir, "fig8_pe_failure_robustness.png")
    )

    # 6. Experiment 5: Scalability Analysis (Table III)
    print("\n>>> Running Experiment 5: Scalability Analysis (Table III)...")
    exp5_res = run_experiment_5(num_dags=num_dags, num_runs=num_runs, output_dir=results_dir)

    # 7. Experiment 6: Real Hardware SelfPilot (Fig 10)
    print("\n>>> Running Experiment 6: Real Hardware SelfPilot (Fig 10)...")
    exp6_res = run_experiment_6(num_dags=num_dags, num_runs=num_runs, output_dir=results_dir)
    plot_fig10_selfpilot(os.path.join(results_dir, "exp6_selfpilot_hardware.json"), os.path.join(figures_dir, "fig10_selfpilot_hardware.png"))

    total_elapsed = time.time() - total_start
    print("\n" + "="*80)
    print(f" ALL EXPERIMENTS AND PLOTS COMPLETED IN {total_elapsed:.1f}s!")
    print(f" Results JSON saved in: {os.path.abspath(results_dir)}")
    print(f" Publication Figures saved in: {os.path.abspath(figures_dir)}")
    print("="*80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="CASDS master experiment runner"
    )
    parser.add_argument(
        "--num-dags", type=int, default=60,
        help="Number of DAGs per trace (default: 60)"
    )
    parser.add_argument(
        "--num-runs", type=int, default=2,
        help="Number of independent runs per scenario (default: 2)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Directory for JSON results and figures (default: results). "
             "Pass results_reproduced/ to avoid touching the original results/."
    )
    args = parser.parse_args()

    print(f"[CONFIG] num_dags={args.num_dags}  num_runs={args.num_runs}  "
          f"output_dir={os.path.abspath(args.output_dir)}")
    run_full_pipeline(
        num_dags=args.num_dags,
        num_runs=args.num_runs,
        results_dir=args.output_dir,
    )
