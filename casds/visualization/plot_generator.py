"""
Publication-Quality Plot Generator for ACPS Dynamic Scheduling.
Generates figures matching Figures 5, 6, 7, 8, 10 and Tables I, II, III, IV of the IEEE TII paper.
"""

import json
import os
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np


# Palette matching IEEE Transactions publications
SCHEDULER_COLORS = {
    "ADS": "#4A90E2",       # Soft Blue
    "ASDYS": "#50E3C2",     # Mint/Teal
    "HetSched": "#F5A623",  # Amber/Orange
    "TD_ADS": "#9013FE",    # Purple
    "ULS": "#7ED321",       # Lime Green
    "CASDS": "#D0021B",     # Crimson Red (Proposed)
}


def setup_plot_style():
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["grid.color"] = "#E0E0E0"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.alpha"] = 0.7


def plot_fig5_dsr(exp1_json_path: str, output_path: str):
    """Generates Figure 5: DSR(S) and DSR(S_crit) comparison under lambda and R_crit."""
    if not os.path.exists(exp1_json_path):
        print(f"File {exp1_json_path} does not exist. Skipping Fig 5.")
        return

    with open(exp1_json_path, "r") as f:
        res = json.load(f)

    data = res["data"]
    schedulers = ["ADS", "ASDYS", "HetSched", "TD_ADS", "ULS", "CASDS"]
    lambdas = [20.0, 30.0, 40.0]
    rcrits = [0.3, 0.6, 0.9]

    setup_plot_style()
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    subtitles = [
        (r"(a) DSR(S) under $\lambda = 20$ DAG/s", r"(b) DSR(S_{crit}) under $\lambda = 20$ DAG/s"),
        (r"(c) DSR(S) under $\lambda = 30$ DAG/s", r"(d) DSR(S_{crit}) under $\lambda = 30$ DAG/s"),
        (r"(e) DSR(S) under $\lambda = 40$ DAG/s", r"(f) DSR(S_{crit}) under $\lambda = 40$ DAG/s"),
    ]

    for row_idx, lam in enumerate(lambdas):
        ax_dsr = axes[row_idx, 0]
        ax_crit = axes[row_idx, 1]

        x = np.arange(len(rcrits))
        bar_width = 0.13

        for s_idx, s_name in enumerate(schedulers):
            dsr_means = []
            dsr_stds = []
            crit_means = []
            crit_stds = []

            for rcrit in rcrits:
                k = f"lambda_{int(lam)}_rcrit_{int(rcrit*100)}"
                s_data = data.get(k, {}).get(s_name, {})
                dsr_means.append(s_data.get("DSR_mean", 0.0))
                dsr_stds.append(s_data.get("DSR_std", 0.0))
                crit_means.append(s_data.get("DSR_crit_mean", 0.0))
                crit_stds.append(s_data.get("DSR_crit_std", 0.0))

            offset = (s_idx - len(schedulers)/2 + 0.5) * bar_width
            color = SCHEDULER_COLORS.get(s_name, "#888888")

            # DSR plot
            ax_dsr.bar(
                x + offset, dsr_means, yerr=dsr_stds, width=bar_width,
                label=s_name if row_idx == 0 else "", color=color,
                capsize=2, edgecolor="#222222", linewidth=0.5
            )

            # DSR_crit plot
            ax_crit.bar(
                x + offset, crit_means, yerr=crit_stds, width=bar_width,
                label=s_name if row_idx == 0 else "", color=color,
                capsize=2, edgecolor="#222222", linewidth=0.5
            )

        for ax, title, ylabel in [(ax_dsr, subtitles[row_idx][0], "DSR(S) [%]"),
                                   (ax_crit, subtitles[row_idx][1], "DSR(S_crit) [%]")]:
            ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_xlabel("Critical DAG Ratio $R_{crit}$ [%]", fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels(["30%", "60%", "90%"])
            ax.set_ylim(0, 105)
            ax.grid(True, axis="y")

    fig.legend(schedulers, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 0.99), frameon=True, fontsize=10)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated Figure 5 at {output_path}")


def plot_fig6_efficiency(exp2_json_path: str, output_path: str):
    """Generates Figure 6: Makespan, Utilization, and Effective Throughput."""
    if not os.path.exists(exp2_json_path):
        print(f"File {exp2_json_path} does not exist. Skipping Fig 6.")
        return

    with open(exp2_json_path, "r") as f:
        res = json.load(f)

    setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    plt.subplots_adjust(wspace=0.3)

    schedulers = ["ADS", "ASDYS", "HetSched", "TD_ADS", "ULS", "CASDS"]
    summary = res.get("overall_summary", {})

    # (a) MS(S) and U(S, P)
    x = np.arange(len(schedulers))
    ms_vals = [summary.get(s, {}).get("Makespan_mean", 0.0) for s in schedulers]
    ms_errs = [summary.get(s, {}).get("Makespan_std", 0.0) for s in schedulers]
    util_vals = [summary.get(s, {}).get("Utilization_mean", 0.0) for s in schedulers]
    util_errs = [summary.get(s, {}).get("Utilization_std", 0.0) for s in schedulers]

    ax1_twin = ax1.twinx()
    # Bar for makespan
    bars = ax1.bar(x - 0.15, ms_vals, yerr=ms_errs, width=0.3, color="#4A90E2", label="MS(S)", capsize=3)
    # Line for utilization
    line = ax1_twin.plot(x + 0.15, util_vals, 'o-', color="#D0021B", linewidth=2, markersize=6, label="U(S, P)")
    ax1_twin.errorbar(x + 0.15, util_vals, yerr=util_errs, fmt='none', ecolor="#D0021B", capsize=3)

    ax1.set_title("(a) MS(S) and U(S, P)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Scheduling Methods", fontsize=10)
    ax1.set_ylabel("Makespan MS(S) [s]", fontsize=10, color="#4A90E2")
    ax1_twin.set_ylabel("Utilization U(S, P) [%]", fontsize=10, color="#D0021B")
    ax1.set_xticks(x)
    ax1.set_xticklabels(schedulers, rotation=25)
    ax1.grid(True, axis="y")

    # (b) ET(S) vs lambda
    lambdas = [20, 25, 30, 35, 40]
    by_lam = res.get("by_lambda", {})

    for s in schedulers:
        et_vals = []
        for l in lambdas:
            k = f"lambda_{l}"
            et_vals.append(by_lam.get(k, {}).get(s, {}).get("ET_mean", 0.0))
        color = SCHEDULER_COLORS.get(s, "#888888")
        marker = "o" if s != "CASDS" else "s"
        lw = 2.5 if s == "CASDS" else 1.5
        ax2.plot(lambdas, et_vals, marker=marker, linewidth=lw, label=s, color=color)

    ax2.set_title("(b) Effective Throughput ET(S)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Arrival Rate $\lambda$ [DAG/s]", fontsize=10)
    ax2.set_ylabel("ET(S) [DAG/s]", fontsize=10)
    ax2.set_xticks(lambdas)
    ax2.grid(True)
    ax2.legend(loc="upper right", frameon=True, fontsize=9)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated Figure 6 at {output_path}")


def plot_fig7_fig8_robustness(exp4_json_path: str, output_fig7: str, output_fig8: str):
    """Generates Figure 7 (Delay Robustness) and Figure 8 (PE Failure Robustness)."""
    if not os.path.exists(exp4_json_path):
        print(f"File {exp4_json_path} does not exist. Skipping Fig 7 & 8.")
        return

    with open(exp4_json_path, "r") as f:
        res = json.load(f)

    setup_plot_style()
    schedulers = ["ADS", "ASDYS", "HetSched", "TD_ADS", "ULS", "CASDS"]

    # Figure 7: Execution Delay
    exec_delays = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    fig7, (ax7_1, ax7_2) = plt.subplots(1, 2, figsize=(12, 4.8))
    plt.subplots_adjust(wspace=0.25)

    delay_data = res.get("execution_delay", {})
    for s in schedulers:
        dsr_means = []
        crit_means = []
        for mu in exec_delays:
            k = f"mu_{int(mu*10)}"
            dsr_means.append(delay_data.get(k, {}).get(s, {}).get("DSR_mean", 0.0))
            crit_means.append(delay_data.get(k, {}).get(s, {}).get("DSR_crit_mean", 0.0))

        color = SCHEDULER_COLORS.get(s, "#888888")
        ax7_1.plot(exec_delays, dsr_means, 'o-', label=s, color=color, linewidth=2 if s == "CASDS" else 1.2)
        ax7_2.plot(exec_delays, crit_means, 's-', label=s, color=color, linewidth=2 if s == "CASDS" else 1.2)

    ax7_1.set_title("(a) DSR(S) under Execution Time Delay", fontsize=11, fontweight="bold")
    ax7_1.set_xlabel("Delay Parameter $\mu_d$", fontsize=10)
    ax7_1.set_ylabel("DSR(S) [%]", fontsize=10)
    ax7_1.grid(True)
    ax7_1.legend(loc="lower left", frameon=True, fontsize=8)

    ax7_2.set_title("(b) DSR(S_crit) under Execution Time Delay", fontsize=11, fontweight="bold")
    ax7_2.set_xlabel("Delay Parameter $\mu_d$", fontsize=10)
    ax7_2.set_ylabel("DSR(S_crit) [%]", fontsize=10)
    ax7_2.grid(True)

    plt.savefig(output_fig7, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated Figure 7 at {output_fig7}")

    # Figure 8: PE Failures
    fail_rates = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    fig8, (ax8_1, ax8_2) = plt.subplots(1, 2, figsize=(12, 4.8))
    plt.subplots_adjust(wspace=0.25)

    fail_data = res.get("pe_failure", {})
    for s in schedulers:
        dsr_means = []
        crit_means = []
        for lam_f in fail_rates:
            k = f"lambda_f_{int(lam_f)}"
            dsr_means.append(fail_data.get(k, {}).get(s, {}).get("DSR_mean", 0.0))
            crit_means.append(fail_data.get(k, {}).get(s, {}).get("DSR_crit_mean", 0.0))

        color = SCHEDULER_COLORS.get(s, "#888888")
        ax8_1.plot(fail_rates, dsr_means, 'o-', label=s, color=color, linewidth=2 if s == "CASDS" else 1.2)
        ax8_2.plot(fail_rates, crit_means, 's-', label=s, color=color, linewidth=2 if s == "CASDS" else 1.2)

    ax8_1.set_title("(a) DSR(S) under PE Failures", fontsize=11, fontweight="bold")
    ax8_1.set_xlabel("Failure Rate $\lambda_f$ [failures/s]", fontsize=10)
    ax8_1.set_ylabel("DSR(S) [%]", fontsize=10)
    ax8_1.grid(True)
    ax8_1.legend(loc="lower left", frameon=True, fontsize=8)

    ax8_2.set_title("(b) DSR(S_crit) under PE Failures", fontsize=11, fontweight="bold")
    ax8_2.set_xlabel("Failure Rate $\lambda_f$ [failures/s]", fontsize=10)
    ax8_2.set_ylabel("DSR(S_crit) [%]", fontsize=10)
    ax8_2.grid(True)

    plt.savefig(output_fig8, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated Figure 8 at {output_fig8}")


def plot_fig10_selfpilot(exp6_json_path: str, output_path: str):
    """Generates Figure 10: 9 subplots for SelfPilot real hardware benchmark."""
    if not os.path.exists(exp6_json_path):
        print(f"File {exp6_json_path} does not exist. Skipping Fig 10.")
        return

    with open(exp6_json_path, "r") as f:
        res = json.load(f)

    data = res["data"]
    schedulers = ["ADS", "ASDYS", "HetSched", "TD_ADS", "CASDS"]
    lambdas = [5, 10, 20, 30, 40]
    rcrits = [0.3, 0.6, 0.9]

    setup_plot_style()
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    plt.subplots_adjust(hspace=0.35, wspace=0.28)

    subtitles = [
        [r"(a) DSR(S) under $R_{crit}=30\%$", r"(b) DSR(S_{crit}) under $R_{crit}=30\%$", r"(c) ET(S) under $R_{crit}=30\%$"],
        [r"(d) DSR(S) under $R_{crit}=60\%$", r"(e) DSR(S_{crit}) under $R_{crit}=60\%$", r"(f) ET(S) under $R_{crit}=60\%$"],
        [r"(g) DSR(S) under $R_{crit}=90\%$", r"(h) DSR(S_{crit}) under $R_{crit}=90\%$", r"(i) ET(S) under $R_{crit}=90\%$"],
    ]

    for row_idx, rcrit in enumerate(rcrits):
        rcrit_key = f"rcrit_{int(rcrit*100)}"

        for col_idx, metric_key in enumerate(["DSR_mean", "DSR_crit_mean", "ET_mean"]):
            ax = axes[row_idx, col_idx]

            for s_name in schedulers:
                vals = []
                for lam in lambdas:
                    lam_key = f"lambda_{lam}"
                    vals.append(data.get(rcrit_key, {}).get(lam_key, {}).get(s_name, {}).get(metric_key, 0.0))

                color = SCHEDULER_COLORS.get(s_name, "#888888")
                lw = 2.5 if s_name == "CASDS" else 1.3
                ax.plot(lambdas, vals, 'o-', label=s_name if (row_idx == 0 and col_idx == 0) else "",
                        color=color, linewidth=lw)

            title = subtitles[row_idx][col_idx]
            ylabel = "DSR [%]" if "DSR" in metric_key else "ET [DAG/s]"
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.set_xlabel(r"$\lambda$ [DAG/s]", fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_xticks(lambdas)
            ax.grid(True)

    fig.legend(schedulers, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.99), frameon=True, fontsize=10)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated Figure 10 at {output_path}")
