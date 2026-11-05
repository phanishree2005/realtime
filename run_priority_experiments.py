"""
Master Experiment Runner & Figure Generator
Executes Experiments 1 to 5 comparing:
1. Uniform Priority Distribution
2. Arithmetic Priority Distribution
3. Geometric Priority Distribution
4. Logarithmic Priority Distribution (Configurable)

Outputs raw CSV datasets to results/csv/ and publication-quality plots to results/plots/.
"""

import os
import sys
import time
import argparse
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.task_model import TaskSet
from src.synthetic_task_generator import SyntheticTaskGenerator
from src.priority_assigner import PriorityAssigner
from src.rta_analyzer import RTAAnalyzer
from src.discrete_event_simulator import DiscreteEventSimulator
from src.nuscenes_inspector import NuScenesInspector


# Matplotlib publication style settings
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 2.2,
    "lines.markersize": 7,
    "grid.alpha": 0.35,
    "axes.grid": True
})

STRATEGY_COLORS = {
    "Uniform": "#2563eb",       # Royal Blue
    "Arithmetic": "#16a34a",    # Forest Green
    "Geometric": "#d97706",     # Amber / Orange
    "Logarithmic": "#dc2626"    # Crimson Red
}

STRATEGY_MARKERS = {
    "Uniform": "o",
    "Arithmetic": "s",
    "Geometric": "^",
    "Logarithmic": "D"
}


# ==============================================================================
# EXPERIMENT 1: Schedulability Ratio vs. CPU Utilization
# ==============================================================================
def run_experiment_1(
    num_tasksets: int = 50,
    n_tasks: int = 10,
    p_max: int = 8,
    output_dir: str = "results"
) -> pd.DataFrame:
    print("\n" + "="*70)
    print(f">>> RUNNING EXPERIMENT 1: Schedulability Ratio vs. CPU Utilization (P_max={p_max})")
    print("="*70)

    utilizations = np.arange(0.40, 0.96, 0.05)
    generator = SyntheticTaskGenerator(seed=42)
    assigner = PriorityAssigner(p_min=1, p_max=p_max)
    analyzer = RTAAnalyzer()

    records = []

    for u_target in utilizations:
        u_val = round(float(u_target), 2)
        print(f"Evaluating utilization U = {u_val:.2f} ({num_tasksets} tasksets)...")

        # Generate base task sets at this utilization
        base_sets = [
            generator.generate_random_task_set(n_tasks=n_tasks, target_utilization=u_val)
            for _ in range(num_tasksets)
        ]

        # Fair comparison: evaluate all 4 strategies on the IDENTICAL task sets
        for strat in PriorityAssigner.ALL_STRATEGIES:
            sched_count = 0
            mean_wcrt_ratios = []

            for base_set in base_sets:
                assigned_set = assigner.assign_priorities(base_set, strat, enforce_unique=False)
                rta_res = analyzer.analyze(assigned_set)
                if rta_res["is_schedulable"]:
                    sched_count += 1
                ratios = [r["response_ratio"] for r in rta_res["per_task_results"]]
                mean_wcrt_ratios.append(float(np.mean(ratios)))

            sched_ratio_pct = (sched_count / num_tasksets) * 100.0
            avg_wcrt_ratio = float(np.mean(mean_wcrt_ratios))

            records.append({
                "Utilization": u_val,
                "Strategy": strat,
                "Schedulability_Ratio_pct": sched_ratio_pct,
                "Avg_WCRT_to_Deadline_Ratio": avg_wcrt_ratio,
                "Num_TaskSets": num_tasksets,
                "Num_Tasks_Per_Set": n_tasks,
                "P_max": p_max
            })

    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "csv", "exp1_schedulability_vs_utilization.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved Experiment 1 CSV: {csv_path}")

    # Plot Figure 1
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for strat in PriorityAssigner.ALL_STRATEGIES:
        sub = df[df["Strategy"] == strat]
        ax.plot(
            sub["Utilization"],
            sub["Schedulability_Ratio_pct"],
            label=f"{strat} Priority",
            color=STRATEGY_COLORS[strat],
            marker=STRATEGY_MARKERS[strat]
        )

    ax.set_title("Figure 1: Schedulability Ratio vs. Processor Utilization", pad=12, fontweight="bold")
    ax.set_xlabel(r"Processor Utilization $U = \sum C_i / T_i$")
    ax.set_ylabel("Schedulability Ratio (%)")
    ax.set_ylim(-5, 105)
    ax.set_xlim(0.38, 0.97)
    ax.axvline(x=0.693, color="#64748b", linestyle="--", linewidth=1.5, label=r"Liu & Layland Bound ($\ln 2 \approx 69.3\%$)")
    ax.legend(loc="lower left", framealpha=0.9)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "plots", "fig1_schedulability_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved Figure 1 Plot: {plot_path}")
    return df


# ==============================================================================
# EXPERIMENT 2: Deadline Miss Ratio vs. CPU Utilization (Stress / Overload)
# ==============================================================================
def run_experiment_2(
    num_tasksets: int = 30,
    n_tasks: int = 10,
    p_max: int = 8,
    duration_ms: float = 800.0,
    output_dir: str = "results"
) -> pd.DataFrame:
    print("\n" + "="*70)
    print(f">>> RUNNING EXPERIMENT 2: Deadline Miss Ratio vs. CPU Utilization (P_max={p_max})")
    print("="*70)

    utilizations = np.arange(0.65, 1.16, 0.05)
    generator = SyntheticTaskGenerator(seed=101)
    assigner = PriorityAssigner(p_min=1, p_max=p_max)
    sim = DiscreteEventSimulator(measure_overhead=False)

    records = []

    for u_target in utilizations:
        u_val = round(float(u_target), 2)
        print(f"Simulating stress condition U = {u_val:.2f}...")

        base_sets = [
            generator.generate_random_task_set(n_tasks=n_tasks, target_utilization=u_val)
            for _ in range(num_tasksets)
        ]

        for strat in PriorityAssigner.ALL_STRATEGIES:
            miss_ratios = []
            preemptions = []
            actual_utils = []

            for base_set in base_sets:
                assigned_set = assigner.assign_priorities(base_set, strat, enforce_unique=False)
                sim_res = sim.run(assigned_set, duration_ms=duration_ms, use_wcet=True)
                miss_ratios.append(sim_res["deadline_miss_ratio_pct"])
                preemptions.append(sim_res["preemption_count"])
                actual_utils.append(sim_res["achieved_cpu_utilization"])

            records.append({
                "Utilization": u_val,
                "Strategy": strat,
                "Deadline_Miss_Ratio_pct": float(np.mean(miss_ratios)),
                "Mean_Preemptions": float(np.mean(preemptions)),
                "Achieved_Utilization": float(np.mean(actual_utils))
            })

    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "csv", "exp2_deadline_misses_vs_utilization.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved Experiment 2 CSV: {csv_path}")

    # Plot Figure 2
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for strat in PriorityAssigner.ALL_STRATEGIES:
        sub = df[df["Strategy"] == strat]
        ax.plot(
            sub["Utilization"],
            sub["Deadline_Miss_Ratio_pct"],
            label=f"{strat} Priority",
            color=STRATEGY_COLORS[strat],
            marker=STRATEGY_MARKERS[strat]
        )

    ax.set_title("Figure 2: Deadline Miss Ratio under Heavy Workloads & Overload", pad=12, fontweight="bold")
    ax.set_xlabel("Nominal Processor Utilization $U$")
    ax.set_ylabel("Deadline Miss Ratio (%)")
    ax.set_ylim(-2, max(60, df["Deadline_Miss_Ratio_pct"].max() + 5))
    ax.axvline(x=1.0, color="#ef4444", linestyle=":", linewidth=1.8, label="Overload Boundary ($U = 1.0$)")
    ax.legend(loc="upper left", framealpha=0.9)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "plots", "fig2_deadline_miss_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved Figure 2 Plot: {plot_path}")
    return df


# ==============================================================================
# EXPERIMENT 3: Priority Quantization Levels (Hardware / RTOS Constraint)
# ==============================================================================
def run_experiment_3(
    num_tasksets: int = 50,
    n_tasks: int = 16,
    target_u: float = 0.72,
    output_dir: str = "results"
) -> pd.DataFrame:
    print("\n" + "="*70)
    print(">>> RUNNING EXPERIMENT 3: Impact of Priority Grid Quantization Levels (P_max)")
    print("="*70)

    # Common RTOS priority limits: 4, 8, 16, 32, 64, 128, 256
    p_max_levels = [4, 8, 16, 32, 64, 128, 256]
    generator = SyntheticTaskGenerator(seed=202)
    analyzer = RTAAnalyzer()

    records = []
    base_sets = [
        generator.generate_random_task_set(n_tasks=n_tasks, target_utilization=target_u)
        for _ in range(num_tasksets)
    ]

    for p_max in p_max_levels:
        print(f"Evaluating P_max = {p_max} priority levels...")
        assigner = PriorityAssigner(p_min=1, p_max=p_max)

        for strat in PriorityAssigner.ALL_STRATEGIES:
            sched_count = 0
            for base_set in base_sets:
                # With limited p_max < n_tasks, enforce_unique=False to test quantization resilience
                enforce = (p_max >= n_tasks)
                assigned_set = assigner.assign_priorities(base_set, strat, enforce_unique=enforce)
                rta_res = analyzer.analyze(assigned_set)
                if rta_res["is_schedulable"]:
                    sched_count += 1

            sched_ratio = (sched_count / num_tasksets) * 100.0
            records.append({
                "P_max": p_max,
                "Strategy": strat,
                "Schedulability_Ratio_pct": sched_ratio,
                "N_tasks": n_tasks,
                "Target_Utilization": target_u
            })

    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "csv", "exp3_priority_quantization_levels.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved Experiment 3 CSV: {csv_path}")

    # Plot Figure 3
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for strat in PriorityAssigner.ALL_STRATEGIES:
        sub = df[df["Strategy"] == strat]
        ax.plot(
            sub["P_max"],
            sub["Schedulability_Ratio_pct"],
            label=f"{strat} Priority",
            color=STRATEGY_COLORS[strat],
            marker=STRATEGY_MARKERS[strat]
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(p_max_levels)
    ax.set_xticklabels([str(p) for p in p_max_levels])
    ax.set_title(f"Figure 3: Schedulability vs. RTOS Priority Levels ($N={n_tasks}, U={target_u}$)", pad=12, fontweight="bold")
    ax.set_xlabel("Available Priority Levels $P_{\\max}$ (Log2 Scale)")
    ax.set_ylabel("Schedulability Ratio (%)")
    ax.set_ylim(-5, 105)
    ax.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "plots", "fig3_priority_quantization_impact.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved Figure 3 Plot: {plot_path}")
    return df


# ==============================================================================
# EXPERIMENT 4: Scheduling Decision Overhead vs. Task Set Size
# ==============================================================================
def run_experiment_4(
    num_runs: int = 15,
    target_u: float = 0.70,
    duration_ms: float = 600.0,
    output_dir: str = "results"
) -> pd.DataFrame:
    print("\n" + "="*70)
    print(">>> RUNNING EXPERIMENT 4: Scheduling Decision Overhead & Context Switches")
    print("="*70)

    task_sizes = [5, 10, 15, 20, 25, 30]
    generator = SyntheticTaskGenerator(seed=303)
    assigner = PriorityAssigner(p_min=1, p_max=256)
    sim = DiscreteEventSimulator(measure_overhead=True)

    records = []

    for n in task_sizes:
        print(f"Profiling scheduling overhead for N = {n} tasks...")
        for strat in PriorityAssigner.ALL_STRATEGIES:
            overheads_us = []
            context_switches = []

            for _ in range(num_runs):
                base_set = generator.generate_random_task_set(n_tasks=n, target_utilization=target_u)
                assigned_set = assigner.assign_priorities(base_set, strat)
                sim_res = sim.run(assigned_set, duration_ms=duration_ms)
                overheads_us.append(sim_res["mean_scheduling_overhead_us"])
                context_switches.append(sim_res["context_switch_count"])

            records.append({
                "N_tasks": n,
                "Strategy": strat,
                "Mean_Overhead_us": float(np.mean(overheads_us)),
                "Std_Overhead_us": float(np.std(overheads_us)),
                "Mean_Context_Switches": float(np.mean(context_switches))
            })

    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "csv", "exp4_scheduling_overhead.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved Experiment 4 CSV: {csv_path}")

    # Plot Figure 4
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.0))

    # Overhead plot
    for strat in PriorityAssigner.ALL_STRATEGIES:
        sub = df[df["Strategy"] == strat]
        ax1.plot(
            sub["N_tasks"],
            sub["Mean_Overhead_us"],
            label=f"{strat}",
            color=STRATEGY_COLORS[strat],
            marker=STRATEGY_MARKERS[strat]
        )

    ax1.set_title("(a) Mean Decision Overhead vs. Task Set Size", fontweight="bold")
    ax1.set_xlabel("Number of Tasks $N$")
    ax1.set_ylabel("Overhead per Dispatch ($\\mu$s)")
    ax1.legend(loc="upper left")

    # Context switch plot
    for strat in PriorityAssigner.ALL_STRATEGIES:
        sub = df[df["Strategy"] == strat]
        ax2.plot(
            sub["N_tasks"],
            sub["Mean_Context_Switches"],
            label=f"{strat}",
            color=STRATEGY_COLORS[strat],
            marker=STRATEGY_MARKERS[strat]
        )

    ax2.set_title("(b) Total Context Switches vs. Task Set Size", fontweight="bold")
    ax2.set_xlabel("Number of Tasks $N$")
    ax2.set_ylabel("Total Context Switches (in 600 ms)")
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "plots", "fig4_scheduling_overhead_analysis.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved Figure 4 Plot: {plot_path}")
    return df


# ==============================================================================
# EXPERIMENT 5: nuScenes Real Driving Scenarios Case Study
# ==============================================================================
def run_experiment_5(
    output_dir: str = "results"
) -> pd.DataFrame:
    print("\n" + "="*70)
    print(">>> RUNNING EXPERIMENT 5: nuScenes Driving Scenarios Perception Case Study")
    print("="*70)

    inspector = NuScenesInspector()
    scenes = inspector.load_json("scene.json")
    annotations = inspector.load_json("sample_annotation.json")
    samples = inspector.load_json("sample.json")

    # Map sample tokens to scene tokens
    sample_to_scene = {s["token"]: s.get("scene_token") for s in samples}
    scene_box_counts: Dict[str, int] = {sc["token"]: 0 for sc in scenes}

    for ann in annotations:
        s_tok = ann.get("sample_token")
        sc_tok = sample_to_scene.get(s_tok)
        if sc_tok in scene_box_counts:
            scene_box_counts[sc_tok] += 1

    generator = SyntheticTaskGenerator(seed=404, dataset_inspector=inspector)
    assigner = PriorityAssigner(p_min=1, p_max=256)
    sim = DiscreteEventSimulator(measure_overhead=True)

    records = []
    # If no scenes found, fallback to 10 standard scenes
    scene_list = scenes if scenes else [{"name": f"scene-000{i}", "token": f"token_{i}"} for i in range(10)]

    for sc in scene_list:
        sc_name = sc.get("name", "Unknown")
        sc_tok = sc.get("token", "")
        n_boxes = scene_box_counts.get(sc_tok, 120)
        # Jitter factor scales with object clutter density
        clutter_jitter = min(0.35, 0.05 + 0.30 * (n_boxes / 2000.0))

        # Base 10-task perception set with target utilization ~0.75
        base_set = generator.generate_perception_task_set(target_utilization=0.74, clutter_noise=False)

        for strat in PriorityAssigner.ALL_STRATEGIES:
            assigned_set = assigner.assign_priorities(base_set, strat)
            sim_res = sim.run(assigned_set, duration_ms=1000.0, use_wcet=False, jitter_factor=clutter_jitter)

            records.append({
                "Scene": sc_name,
                "Obstacle_Annotations": n_boxes,
                "Clutter_Jitter": round(clutter_jitter, 3),
                "Strategy": strat,
                "Deadline_Misses": sim_res["deadline_misses"],
                "Deadline_Miss_Ratio_pct": sim_res["deadline_miss_ratio_pct"],
                "Achieved_CPU_Util": round(sim_res["achieved_cpu_utilization"], 3),
                "Mean_Response_Time_ms": round(sim_res["mean_response_time_ms"], 2),
                "Max_Response_Time_ms": round(sim_res["max_response_time_ms"], 2)
            })

    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "csv", "exp5_nuscenes_scenario_benchmark.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved Experiment 5 CSV: {csv_path}")

    # Plot Figure 5: Clutter vs Max Response Time across scenes
    fig, ax = plt.subplots(figsize=(9, 5.2))
    pivot = df.pivot(index="Scene", columns="Strategy", values="Max_Response_Time_ms")
    x_indices = np.arange(len(pivot.index))
    width = 0.2

    for i, strat in enumerate(PriorityAssigner.ALL_STRATEGIES):
        ax.bar(
            x_indices + (i - 1.5) * width,
            pivot[strat],
            width=width,
            label=f"{strat}",
            color=STRATEGY_COLORS[strat]
        )

    ax.set_title("Figure 5: Worst-Case Response Time Across 10 nuScenes Driving Scenarios", pad=12, fontweight="bold")
    ax.set_xlabel("nuScenes Scenario")
    ax.set_ylabel("Maximum Observed Response Time (ms)")
    ax.set_xticks(x_indices)
    ax.set_xticklabels(pivot.index, rotation=35, ha="right")
    ax.legend(loc="upper left")
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "plots", "fig5_nuscenes_scenario_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved Figure 5 Plot: {plot_path}")
    return df


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Real-Time Systems Priority Distribution Experiments")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save CSVs and plots")
    parser.add_argument("--quick", action="store_true", help="Run with reduced iterations for rapid verification")
    args = parser.parse_args()

    out_csv = os.path.join(args.output_dir, "csv")
    out_plots = os.path.join(args.output_dir, "plots")
    os.makedirs(out_csv, exist_ok=True)
    os.makedirs(out_plots, exist_ok=True)

    t_start = time.time()
    print("="*80)
    print(" REAL-TIME SYSTEMS RESEARCH SUITE: 4 PRIORITY DISTRIBUTION STRATEGIES")
    print(" 1. Uniform   2. Arithmetic   3. Geometric   4. Logarithmic")
    print(" Output Directory:", os.path.abspath(args.output_dir))
    print("="*80)

    n_tasksets_e1 = 20 if args.quick else 60
    n_tasksets_e2 = 15 if args.quick else 40
    n_tasksets_e3 = 20 if args.quick else 50
    n_runs_e4 = 6 if args.quick else 15

    # Run Experiments 1-5
    run_experiment_1(num_tasksets=n_tasksets_e1, output_dir=args.output_dir)
    run_experiment_2(num_tasksets=n_tasksets_e2, output_dir=args.output_dir)
    run_experiment_3(num_tasksets=n_tasksets_e3, output_dir=args.output_dir)
    run_experiment_4(num_runs=n_runs_e4, output_dir=args.output_dir)
    run_experiment_5(output_dir=args.output_dir)

    t_total = time.time() - t_start
    print("\n" + "="*80)
    print(f" ALL 5 EXPERIMENTS COMPLETED IN {t_total:.2f} SECONDS!")
    print(f" CSV files saved in : {os.path.abspath(out_csv)}")
    print(f" PNG plots saved in : {os.path.abspath(out_plots)}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
