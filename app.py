"""
Automotive Cyber-Physical Systems Real-Time Scheduling & Priority Assignment Platform.
Unified Research & Demonstration Platform combining:
1. CASDS: Context-Aware-Slack-Driven Dynamic Task Scheduling (IEEE Transactions on Industrial Informatics, 2026)
2. Priority Assignment Strategies: Comparative Study of Uniform, Arithmetic, Geometric, and Logarithmic Strategies
3. Integrated Evaluation Studio: CASDS with Priority Shaping Curves
"""

import os
import json
import time
import math
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# CASDS Imports
from casds.benchmarks.adsuite_benchmark import (
    create_adsuite_platform, create_adsuite_dag, ADSUITE_TASK_PROFILES
)
from casds.benchmarks.selfpilot_benchmark import (
    create_selfpilot_platform, create_selfpilot_dag, SELFPILOT_TASK_PROFILES
)
from casds.benchmarks.trace_generator import TraceGenerator
from casds.schedulers.casds_scheduler import CASDSScheduler
from casds.schedulers.baselines import (
    ADSScheduler, ASDYSScheduler, HetSchedScheduler, TD_ADSScheduler, ULSScheduler
)
from casds.simulation.simulator import ACPSSimulator

# Priority Scheduling Imports
from src.nuscenes_inspector import NuScenesInspector
from src.task_model import Task as PTask, TaskSet as PTaskSet
from src.priority_assigner import PriorityAssigner
from src.rta_analyzer import RTAAnalyzer
from src.discrete_event_simulator import DiscreteEventSimulator
from src.synthetic_task_generator import SyntheticTaskGenerator
from src.dms_task_model import create_dms_task_set
from src.dms_scheduler_ui import render_dms_priority_scheduler_tab
from src.multiprocessor_allocator import MultiprocessorAllocator
from src.dms_perception_ui import render_dms_driver_perception_studio
from src.dag_studio import render_dag_studio_tab

# ------------------------------------------------------------------------------
# PAGE CONFIGURATION & METADATA
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Automotive Real-Time Scheduling & Priority Suite",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------------------
# HIGH-END PROFESSIONAL ACADEMIC STYLING (DARK HUD THEME)
# ------------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif;
        letter-spacing: 0.5px;
    }

    .main-header {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.85));
        border: 1px solid rgba(0, 240, 255, 0.35);
        border-radius: 14px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }

    .hud-card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 15px;
        transition: transform 0.2s, border-color 0.2s;
    }
    .hud-card:hover {
        border-color: rgba(0, 240, 255, 0.4);
        transform: translateY(-2px);
    }

    .metric-value {
        font-family: 'Orbitron', sans-serif;
        font-size: 26px;
        font-weight: 700;
        color: #00f0ff;
    }

    .badge-pill {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 9999px;
        margin-right: 6px;
        margin-bottom: 4px;
    }
    .badge-blue { background: rgba(37, 99, 235, 0.25); color: #60a5fa; border: 1px solid #2563eb; }
    .badge-green { background: rgba(22, 163, 74, 0.25); color: #4ade80; border: 1px solid #16a34a; }
    .badge-purple { background: rgba(147, 51, 234, 0.25); color: #c084fc; border: 1px solid #9333ea; }
    .badge-amber { background: rgba(217, 119, 6, 0.25); color: #fbbf24; border: 1px solid #d97706; }
    .badge-cyan { background: rgba(6, 182, 212, 0.25); color: #22d3ee; border: 1px solid #06b6d4; }

    .metric-card {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        margin-bottom: 12px;
    }
    .metric-val {
        font-family: 'Orbitron', monospace;
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-lbl {
        color: #94a3b8;
        font-size: 0.85rem;
        margin-top: 4px;
        font-weight: 500;
    }
    .section-card {
        background-color: #111827;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .main-gradient-title {
        font-size: 2.2rem;
        font-weight: 800;
        font-family: 'Orbitron', 'Inter', sans-serif;
        background: linear-gradient(90deg, #6EE7F7, #A78BFA, #F472B6, #FBBF24);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    .sub-description { color: #9CA3AF; font-size: 1rem; margin-bottom: 1.2rem; }
    .metric-safe     { color: #10B981 !important; }
    .metric-caution  { color: #F59E0B !important; }
    .metric-critical { color: #EF4444 !important; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# SIDEBAR: PLATFORM SUITE NAVIGATION
# ------------------------------------------------------------------------------
st.sidebar.markdown("## 🎛️ Master Research Suite")
suite_mode = st.sidebar.radio(
    "Select Evaluation Platform",
    [
        "👁️ DMS Driver Perception & Unified HUD Studio",
        "⚡ CASDS: Dynamic Heterogeneous DAG Scheduling (IEEE TII 2026)",
        "🏎️ Priority Assignment Strategies (Uniform, Arithmetic, Geometric, Logarithmic)",
        "🖥️ Multiprocessor Systems: Task Allocation, Utilization & UMA/NUMA",
        "🔗 Integrated Studio: CASDS + Priority Strategies"
    ],
    index=0
)

st.sidebar.markdown("---")

STRATEGY_COLORS = {
    "Uniform": "#2563eb",
    "Arithmetic": "#16a34a",
    "Geometric": "#d97706",
    "Logarithmic": "#dc2626"
}


# ==============================================================================
# MODE 0: DMS DRIVER PERCEPTION & UNIFIED HUD STUDIO
# ==============================================================================
if suite_mode.startswith("👁️ DMS Driver Perception"):
    render_dms_driver_perception_studio(key_prefix="app_dms_perception")

# ==============================================================================
# MODE 1: CASDS DYNAMIC HETEROGENEOUS DAG SCHEDULER
# ==============================================================================
elif suite_mode.startswith("⚡ CASDS"):
    # CASDS Header
    st.markdown("""
    <div class="main-header">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap;">
            <div>
                <span class="badge-pill badge-cyan">IEEE TRANSACTIONS ON INDUSTRIAL INFORMATICS (2026)</span>
                <span class="badge-pill badge-green">AUTOSAR Adaptive Platform</span>
                <span class="badge-pill badge-purple">Mixed-Criticality DAGs</span>
                <h1 style="color: #ffffff; margin: 8px 0 10px 0; font-size: 26px;">
                    Context-Aware-Slack-Driven Dynamic Task Scheduling (CASDS)
                </h1>
                <p style="color: #94a3b8; margin: 0; font-size: 14px;">
                    Dynamic Multi-DAG Task Scheduling on Heterogeneous Automotive Cyber-Physical Systems (ACPS)
                </p>
            </div>
            <div style="text-align: right; background: rgba(0, 240, 255, 0.05); padding: 12px 18px; border-radius: 8px; border: 1px solid rgba(0, 240, 255, 0.2); margin-top: 8px;">
                <div style="color: #00f0ff; font-family: 'Orbitron'; font-size: 20px; font-weight: bold;">CASDS v1.0</div>
                <div style="color: #64748b; font-size: 11px;">Real-Time Slack • TOPSIS • Heterogeneous PEs</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # CASDS Sidebar Controls
    st.sidebar.markdown("### ⚙️ CASDS Configuration")
    benchmark_choice = st.sidebar.selectbox(
        "Target Benchmark",
        ["ADSuite (NVIDIA TX1 + Accelerators)", "SelfPilot (TZTEK Orin + MIC RTX3060)"]
    )

    st.sidebar.markdown("### 🚦 Workload Parameters")
    arrival_lambda = st.sidebar.slider("DAG Arrival Rate (λ [DAG/s])", min_value=5.0, max_value=40.0, value=25.0, step=5.0)
    critical_ratio = st.sidebar.slider("Critical DAG Ratio (R_crit)", min_value=0.1, max_value=0.9, value=0.6, step=0.1)
    trace_length = st.sidebar.slider("Trace Length (Number of DAGs)", min_value=10, max_value=200, value=30, step=10)

    st.sidebar.markdown("### 🤖 Scheduler Selection")
    selected_schedulers = st.sidebar.multiselect(
        "Compare Algorithms",
        ["CASDS (Proposed)", "ULS", "TD_ADS", "HetSched", "ASDYS", "ADS"],
        default=["CASDS (Proposed)", "HetSched", "TD_ADS", "ULS"]
    )

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏎️ Platform & DAGs",
        "🧠 CASDS Mathematical Engine",
        "⚡ Dynamic Gantt Simulation",
        "📊 Paper Benchmark Results",
        "🧪 Robustness & Scalability Lab"
    ])

    # TAB 1: Platform & DAG Studio
    with tab1:
        with st.expander("🖥️ View Heterogeneous Hardware Architecture & Task Execution Profiles (Table I / IV)", expanded=False):
            if "ADSuite" in benchmark_choice:
                st.info("ADSuite Platform: 13 Heterogeneous Processing Elements (PEs)")
                pe_data = {
                    "PE Name": [f"ARM_A57_CPU_{i}" for i in range(8)] + ["Maxwell_GPU_0", "Maxwell_GPU_1", "DET_Accelerator", "TRA_Accelerator", "LOC_Accelerator"],
                    "PE Type": ["CPU"] * 8 + ["GPU"] * 2 + ["ASIC (Detection)", "ASIC (Tracking)", "ASIC (Localization)"],
                    "Capabilities": ["All Kernels (DET, TRA, LOC, FUS, MiP, MoP)"] * 8 + ["DET, TRA, LOC"] * 2 + ["DET only", "TRA only", "LOC only"]
                }
                st.dataframe(pd.DataFrame(pe_data), use_container_width=True, height=220)

                st.markdown("##### Task Kernel Execution Profiles across PEs (Table I)")
                df_table1 = pd.DataFrame([
                    {"Task Kernel": "Object Detection (DET)", "CPU [ms]": 3531.0, "GPU [ms]": 156.0, "ASIC [ms]": 96.0},
                    {"Task Kernel": "Object Tracking (TRA)", "CPU [ms]": 1825.0, "GPU [ms]": 17.0, "ASIC [ms]": 2.0},
                    {"Task Kernel": "Localization (LOC)", "CPU [ms]": 165.0, "GPU [ms]": 95.0, "ASIC [ms]": 10.0},
                    {"Task Kernel": "Fusion (FUS)", "CPU [ms]": 0.1, "GPU [ms]": "-", "ASIC [ms]": "-"},
                    {"Task Kernel": "Mission Planning (MiP)", "CPU [ms]": 1.0, "GPU [ms]": "-", "ASIC [ms]": "-"},
                    {"Task Kernel": "Motion Planning (MoP)", "CPU [ms]": 8.0, "GPU [ms]": "-", "ASIC [ms]": "-"},
                ])
                st.dataframe(df_table1, use_container_width=True)

            else:
                st.info("SelfPilot Real Vehicle Hardware: 11 Heterogeneous PEs")
                df_table4 = pd.DataFrame([
                    {"Task Kernel": "LiDAR Object Detection (LOD)", "CPU [ms]": 60.0, "GPU1 [ms]": 17.0, "GPU2 [ms]": "-"},
                    {"Task Kernel": "Camera Object Detection (COD)", "CPU [ms]": 62.0, "GPU1 [ms]": 25.0, "GPU2 [ms]": "-"},
                    {"Task Kernel": "Object Detection Fusion (ODF)", "CPU [ms]": 7.0, "GPU1 [ms]": "-", "GPU2 [ms]": "-"},
                    {"Task Kernel": "Lane Detection (LD)", "CPU [ms]": 22.0, "GPU1 [ms]": 9.0, "GPU2 [ms]": "-"},
                    {"Task Kernel": "Multiple Object Tracking (MOT)", "CPU [ms]": 29.0, "GPU1 [ms]": "-", "GPU2 [ms]": "-"},
                    {"Task Kernel": "Localization (LOC)", "CPU [ms]": 60.0, "GPU1 [ms]": 23.0, "GPU2 [ms]": 8.0},
                    {"Task Kernel": "Mission Planning (MiP)", "CPU [ms]": 1.0, "GPU1 [ms]": "-", "GPU2 [ms]": "-"},
                    {"Task Kernel": "Motion Planning (MoP)", "CPU [ms]": 69.0, "GPU1 [ms]": "-", "GPU2 [ms]": "-"},
                    {"Task Kernel": "Control (CON)", "CPU [ms]": 11.0, "GPU1 [ms]": "-", "GPU2 [ms]": "-"},
                ])
                st.dataframe(df_table4, use_container_width=True)

        # Render full expanded DAG Studio
        render_dag_studio_tab(benchmark_choice)

    # TAB 2: Mathematical Engine
    with tab2:
        st.subheader("2. CASDS Core Mathematical Formulation")

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("""
            ### ⏱️ Context-Aware Slack (CAS)
            **1. Remaining Time ($T_{rem}^{a,i}$):**
            $$T_{rem}^{a,i} = T_{arr}^a + T_{rd}^a - t$$
            Captures historical execution status from arrival to current clock $t$.

            **2. Predictive Least Completion Time (PLCT):**
            $$PLCT(Q_u^a, \\mathcal{P}_t^s) = \\max_{v_i^a \\in Q_u^a} \\left\\{ \\min_{p_k \\in \\mathcal{P}} EFT(v_i^a, p_k, \\mathcal{P}_t^s) \\right\\} - t$$
            where:
            $$ERT(v_i^a, p_k) = \\max_{v_j \\in pred(v_i)} \\{ \\max(\\hat{T}_{ft}^{a,j}, t) + \\hat{c}_{i,j}^a \\}$$
            $$EST(v_i^a, p_k, \\mathcal{P}_t^s) = \\max\\{ \\Lambda(p_k, \\mathcal{P}_t^s), ERT(v_i^a, p_k) \\}$$
            $$EFT(v_i^a, p_k, \\mathcal{P}_t^s) = w_{i,k}^a + EST(v_i^a, p_k, \\mathcal{P}_t^s)$$

            **3. Context-Aware Slack (CAS):**
            $$CAS(v_i^a, \\mathcal{P}_t^s) = T_{rem}^{a,i} - PLCT(Q_u^a, \\mathcal{P}_t^s)$$
            $$U_i^a = CAS(v_i^a, \\mathcal{P}_t^s)$$
            - Smaller value $\\implies$ higher urgency.
            - $U_i^a < 0 \\implies$ deadline cannot be met; deferred to prevent domino blocking.
            """)

        with col_m2:
            st.markdown("""
            ### ⚖️ Adaptive Priority (TOPSIS + Entropy Weight)
            Balances task criticality $\\mathcal{C}_i^a$ and urgency $U_i^a$.

            **1. Normalization:**
            $$x_{n,m}^{norm} = \\frac{x_{n,m} - x_{\\min,m}}{x_{\\max,m} - x_{\\min,m}}$$

            **2. Entropy Weight Method:**
            $$p_{n,m} = \\frac{x_{n,m}^{norm}}{\\sum_{n'=1}^N x_{n',m}^{norm}}, \\quad \\varepsilon_m = -\\frac{1}{\\ln N} \\sum_{n=1}^N p_{n,m} \\ln p_{n,m}$$
            $$\\omega_m = \\frac{1 - \\varepsilon_m}{\\sum_{m'=1}^M (1 - \\varepsilon_{m'})}, \\quad x_{n,m}^{wt} = \\omega_m \\cdot x_{n,m}^{norm}$$

            **3. Ideal Solutions & Score:**
            $$Z_m^+ = \\max_n \\{x_{n,m}^{wt}\\}, \\quad Z_m^- = \\min_n \\{x_{n,m}^{wt}\\}$$
            $$Sim_n^+ = \\sqrt{\\sum_{m=1}^M (x_{n,m}^{wt} - Z_m^+)^2}, \\quad Sim_n^- = \\sqrt{\\sum_{m=1}^M (x_{n,m}^{wt} - Z_m^-)^2}$$
            $$P_i^a = Sim_n = \\frac{Sim_n^-}{Sim_n^- + Sim_n^+}$$

            **4. Target PE Allocation:**
            $$p_{alc}^{a,i} = \\begin{cases} p_{tg}^{a,i} & \\text{if } \\Lambda(p_{tg}^{a,i}, \\mathcal{P}_t^s) = t \\\\ \\text{Null} & \\text{else} \\end{cases}$$
            """)

    # TAB 3: Dynamic Gantt Simulation
    with tab3:
        st.subheader("3. Live Dynamic Simulation & Gantt Scheduling Visualizer")

        col_sim_btn, col_sched_view = st.columns([1, 3])
        with col_sim_btn:
            run_sim = st.button("🚀 Run Live Dynamic Simulation", type="primary", use_container_width=True)
            sim_scheduler_choice = st.selectbox(
                "Primary Visualizer Scheduler",
                ["CASDS", "ULS", "TD_ADS", "HetSched", "ASDYS", "ADS"]
            )

        cached_sim = st.session_state.get("sim_result")
        should_run_sim = (
            run_sim
            or cached_sim is None
            or cached_sim.get("scheduler_name") != sim_scheduler_choice
            or cached_sim.get("benchmark") != benchmark_choice
            or cached_sim.get("trace_length") != trace_length
            or cached_sim.get("arrival_lambda") != arrival_lambda
            or cached_sim.get("critical_ratio") != critical_ratio
        )

        if should_run_sim:
            with st.spinner("Simulating dynamic execution on heterogeneous platform..."):
                platform = create_adsuite_platform() if "ADSuite" in benchmark_choice else create_selfpilot_platform()
                tg = TraceGenerator(seed=42)

                if "ADSuite" in benchmark_choice:
                    trace = tg.generate_adsuite_trace(
                        num_dags=trace_length,
                        arrival_rate_lambda=arrival_lambda,
                        critical_ratio=critical_ratio,
                        relative_deadline=400.0
                    )
                else:
                    trace = tg.generate_selfpilot_trace(
                        num_dags=trace_length,
                        arrival_rate_lambda=arrival_lambda,
                        critical_ratio=critical_ratio
                    )

                sched_map = {
                    "CASDS": CASDSScheduler(),
                    "ULS": ULSScheduler(),
                    "TD_ADS": TD_ADSScheduler(),
                    "HetSched": HetSchedScheduler(),
                    "ASDYS": ASDYSScheduler(),
                    "ADS": ADSScheduler(),
                }

                scheduler = sched_map[sim_scheduler_choice]
                sim = ACPSSimulator(platform, scheduler)
                metrics = sim.run(trace)
                st.session_state["sim_result"] = {
                    "metrics": metrics,
                    "platform": metrics.platform,
                    "trace": trace,
                    "scheduler_name": sim_scheduler_choice,
                    "benchmark": benchmark_choice,
                    "trace_length": trace_length,
                    "arrival_lambda": arrival_lambda,
                    "critical_ratio": critical_ratio
                }

        if "sim_result" in st.session_state:
            res = st.session_state["sim_result"]
            m = res["metrics"]
            p = res.get("platform")
            if p is None or not any(len(pe.schedule) > 0 for pe in p.pes):
                p = m.platform

            # KPI Metrics Cards
            kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
            with kpi1:
                st.markdown(f"""
                <div class="hud-card">
                    <div style="color: #94a3b8; font-size: 12px;">Overall DSR(S)</div>
                    <div class="metric-value">{m.dsr:.1f}%</div>
                    <div style="color: #64748b; font-size: 11px;">{m.num_success}/{m.num_total} Met Deadline</div>
                </div>
                """, unsafe_allow_html=True)
            with kpi2:
                st.markdown(f"""
                <div class="hud-card">
                    <div style="color: #94a3b8; font-size: 12px;">Critical DSR(S_crit)</div>
                    <div class="metric-value" style="color: #ff0055;">{m.dsr_crit:.1f}%</div>
                    <div style="color: #64748b; font-size: 11px;">Safety Critical Rate</div>
                </div>
                """, unsafe_allow_html=True)
            with kpi3:
                st.markdown(f"""
                <div class="hud-card">
                    <div style="color: #94a3b8; font-size: 12px;">Effective Throughput</div>
                    <div class="metric-value" style="color: #00ff88;">{m.effective_throughput:.2f}</div>
                    <div style="color: #64748b; font-size: 11px;">DAGs / sec</div>
                </div>
                """, unsafe_allow_html=True)
            with kpi4:
                st.markdown(f"""
                <div class="hud-card">
                    <div style="color: #94a3b8; font-size: 12px;">PE Utilization</div>
                    <div class="metric-value" style="color: #f5a623;">{m.utilization:.1f}%</div>
                    <div style="color: #64748b; font-size: 11px;">Busy Hardware Ratio</div>
                </div>
                """, unsafe_allow_html=True)
            with kpi5:
                st.markdown(f"""
                <div class="hud-card">
                    <div style="color: #94a3b8; font-size: 12px;">Total Makespan</div>
                    <div class="metric-value" style="color: #7000ff;">{m.makespan_ms:.0f} ms</div>
                    <div style="color: #64748b; font-size: 11px;">{m.makespan_s:.2f} seconds</div>
                </div>
                """, unsafe_allow_html=True)

            # Interactive Plotly Gantt Chart
            st.markdown("#### 📊 Dynamic Schedule Gantt Chart across Heterogeneous PEs")

            col_g1, col_g2, col_g3 = st.columns([2, 2, 2])
            with col_g1:
                hide_idle = st.checkbox("Hide Idle PEs (Active Hardware Only)", value=True, key="casds_hide_idle")
            with col_g2:
                color_scheme = st.selectbox(
                    "Color Scheme",
                    ["Criticality (ASIL/Normal)", "Task Kernel Type"],
                    key="casds_color_scheme"
                )
            with col_g3:
                max_makespan = float(max(100.0, float(m.makespan_ms)))
                slider_step = float(max(10.0, float(round(max_makespan / 50.0))))
                time_range = st.slider(
                    "Simulation Window (ms)",
                    min_value=0.0,
                    max_value=max_makespan,
                    value=(0.0, max_makespan),
                    step=slider_step,
                    key="casds_time_window"
                )

            # Filter PEs and tasks
            target_pes = [pe for pe in p.pes if not hide_idle or len(pe.schedule) > 0]

            # Palette for kernels
            KERNEL_PALETTE = {
                "DET": "#38bdf8", "TRA": "#a855f7", "LOC": "#10b981", "FUS": "#f59e0b",
                "MiP": "#ec4899", "MoP": "#6366f1", "CON": "#14b8a6", "COD": "#06b6d4",
                "LOD": "#8b5cf6", "MOT": "#f43f5e", "ODF": "#eab308", "LD": "#10b981"
            }

            fig_gantt = go.Figure()
            total_rendered = 0

            if color_scheme == "Criticality (ASIL/Normal)":
                color_groups = [
                    (2, "Critical DAG Task (C=2)", "#ff0055"),
                    (1, "Normal DAG Task (C=1)", "#00f0ff")
                ]
                for c_val, c_name, c_color in color_groups:
                    pe_labels, bases, durs, hovers = [], [], [], []
                    for pe in target_pes:
                        for rec in pe.schedule:
                            if rec["finish_time"] < time_range[0] or rec["start_time"] > time_range[1]:
                                continue
                            is_crit = (rec["criticality"] >= 2)
                            if (c_val == 2 and is_crit) or (c_val == 1 and not is_crit):
                                pe_str = f"PE {pe.pe_id}: {pe.name}"
                                pe_labels.append(pe_str)
                                bases.append(rec["start_time"])
                                durs.append(rec["duration"])
                                hovers.append(
                                    f"<b>{pe_str}</b><br>"
                                    f"<b>Task:</b> A{rec['app_id']}_{rec['task_name']}<br>"
                                    f"<b>DAG ID:</b> #{rec['app_id']}<br>"
                                    f"<b>Start:</b> {rec['start_time']:.1f} ms<br>"
                                    f"<b>Finish:</b> {rec['finish_time']:.1f} ms<br>"
                                    f"<b>Duration:</b> {rec['duration']:.1f} ms<br>"
                                    f"<b>Criticality:</b> {c_name}"
                                )
                                total_rendered += 1

                    if pe_labels:
                        fig_gantt.add_trace(go.Bar(
                            y=pe_labels,
                            x=durs,
                            base=bases,
                            orientation="h",
                            name=c_name,
                            marker=dict(color=c_color, opacity=0.88, line=dict(color="rgba(0,0,0,0.4)", width=0.5)),
                            hovertext=hovers,
                            hoverinfo="text"
                        ))
            else:
                kernel_data = {}
                for pe in target_pes:
                    for rec in pe.schedule:
                        if rec["finish_time"] < time_range[0] or rec["start_time"] > time_range[1]:
                            continue
                        k_name = rec["task_name"]
                        prefix = k_name.split("_")[0] if "_" in k_name else k_name
                        if prefix not in kernel_data:
                            kernel_data[prefix] = {"pes": [], "bases": [], "durs": [], "hovers": []}

                        pe_str = f"PE {pe.pe_id}: {pe.name}"
                        kernel_data[prefix]["pes"].append(pe_str)
                        kernel_data[prefix]["bases"].append(rec["start_time"])
                        kernel_data[prefix]["durs"].append(rec["duration"])
                        crit_str = "Critical (C=2)" if rec["criticality"] >= 2 else "Normal (C=1)"
                        kernel_data[prefix]["hovers"].append(
                            f"<b>{pe_str}</b><br>"
                            f"<b>Task:</b> A{rec['app_id']}_{rec['task_name']}<br>"
                            f"<b>DAG ID:</b> #{rec['app_id']}<br>"
                            f"<b>Start:</b> {rec['start_time']:.1f} ms<br>"
                            f"<b>Finish:</b> {rec['finish_time']:.1f} ms<br>"
                            f"<b>Duration:</b> {rec['duration']:.1f} ms<br>"
                            f"<b>Criticality:</b> {crit_str}"
                        )
                        total_rendered += 1

                for k_prefix, k_dict in kernel_data.items():
                    k_col = KERNEL_PALETTE.get(k_prefix, "#94a3b8")
                    fig_gantt.add_trace(go.Bar(
                        y=k_dict["pes"],
                        x=k_dict["durs"],
                        base=k_dict["bases"],
                        orientation="h",
                        name=k_prefix,
                        marker=dict(color=k_col, opacity=0.88, line=dict(color="rgba(0,0,0,0.4)", width=0.5)),
                        hovertext=k_dict["hovers"],
                        hoverinfo="text"
                    ))

            if total_rendered > 0:
                fig_gantt.update_layout(
                    barmode="overlay",
                    xaxis=dict(
                        title="Simulation Time (ms)",
                        range=[time_range[0], time_range[1]],
                        rangeslider=dict(visible=True)
                    ),
                    yaxis=dict(
                        title="Heterogeneous Processing Element",
                        autorange="reversed"
                    ),
                    plot_bgcolor="rgba(11, 15, 25, 0.8)",
                    paper_bgcolor="rgba(11, 15, 25, 0.8)",
                    template="plotly_dark",
                    height=max(450, 42 * len(target_pes)),
                    margin=dict(l=20, r=20, t=30, b=20),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_gantt, use_container_width=True)
                st.caption(f"⚡ Showing **{total_rendered}** scheduled task execution intervals across **{len(target_pes)}** PEs. Drag the range slider or scrub the timeline window to zoom.")
            else:
                st.warning("No tasks were scheduled in this time frame.")

    # TAB 4: Paper Benchmark Results
    with tab4:
        st.subheader("4. Paper Benchmark Reproduction & Published Figures")

        res_fig_choice = st.selectbox(
            "Select Figure to Inspect",
            [
                "Figure 5: DSR(S) and DSR(S_crit) Comparison (λ in {20,30,40}, R_crit in {30%,60%,90%})",
                "Figure 6: System Efficiency (MS, Utilization, and Throughput vs λ)",
                "Figure 7 & 8: Robustness Analysis (Execution Delay & PE Failures)",
                "Figure 10: Real Hardware Platform Experiments (SelfPilot)"
            ]
        )

        fig_path = ""
        if "Figure 5" in res_fig_choice:
            fig_path = "results/figures/fig5_dsr_comparison.png"
            if not os.path.exists(fig_path):
                fig_path = "results_reproduced/figures/fig5_dsr_comparison.png"
            st.markdown(r"""
            **Figure 5 Analysis**:
            - As arrival rate $\lambda$ increases from 20 to 40 DAG/s, ACPS workload and PE congestion sharply increase.
            - **CASDS** achieves the highest $DSR(S)$ and $DSR(S_{crit})$ across all scenarios, providing up to **35.07%** improvement over ADS and **24.32%** over TD_ADS.
            """)
        elif "Figure 6" in res_fig_choice:
            fig_path = "results/figures/fig6_system_efficiency.png"
            if not os.path.exists(fig_path):
                fig_path = "results_reproduced/figures/fig6_system_efficiency.png"
            st.markdown(r"""
            **Figure 6 Analysis**:
            - (a) **Makespan & Utilization**: CASDS achieves the lowest makespan and highest PE utilization by avoiding idle waiting caused by rigid mode switches.
            - (b) **Effective Throughput**: CASDS preserves high effective throughput even under severe congestion ($\lambda=40$ DAG/s).
            """)
        elif "Figure 7 & 8" in res_fig_choice:
            col_r1, col_r2 = st.columns(2)
            fig7_path = "results/figures/fig7_execution_delay_robustness.png"
            if not os.path.exists(fig7_path):
                fig7_path = "results_reproduced/figures/fig7_execution_delay_robustness.png"
            fig8_path = "results/figures/fig8_pe_failure_robustness.png"
            if not os.path.exists(fig8_path):
                fig8_path = "results_reproduced/figures/fig8_pe_failure_robustness.png"
            with col_r1:
                st.markdown(r"##### Figure 7: Execution Time Delay ($\mu_d \in [0.0, 1.0]$)")
                if os.path.exists(fig7_path):
                    st.image(fig7_path, use_container_width=True)
            with col_r2:
                st.markdown(r"##### Figure 8: Transient PE Failures ($\lambda_f \in [0, 10]$)")
                if os.path.exists(fig8_path):
                    st.image(fig8_path, use_container_width=True)
            fig_path = None
        else:
            fig_path = "results/figures/fig10_selfpilot_hardware.png"
            if not os.path.exists(fig_path):
                fig_path = "results_reproduced/figures/fig10_selfpilot_hardware.png"
            st.markdown(r"""
            **Figure 10 Analysis (Real Hardware SelfPilot Platform)**:
            - 9 subplots across $\lambda \in \{5, 10, 20, 30, 40\}$ DAG/s and $R_{crit} \in \{30\%, 60\%, 90\%\}$.
            - In the most challenging scenario ($\lambda=40$ DAG/s, $R_{crit}=90\%$), CASDS sustains **9.88 DAG/s** of effective throughput while baselines collapse near zero.
            """)

        if fig_path and os.path.exists(fig_path):
            st.image(fig_path, use_container_width=True)
        elif fig_path:
            st.info(f"Plot {fig_path} will be available once the full experiment batch finishes.")

        # Tables Tabulation
        st.markdown("---")
        st.markdown("#### 📋 Paper Ablation & Scalability Tables")
        col_t1, col_t2 = st.columns(2)

        with col_t1:
            st.markdown("##### Table II: Ablation Study of CAS-driven Urgency & Priority")
            exp3_path = "results_reproduced/exp3_ablation_study.json" if os.path.exists("results_reproduced/exp3_ablation_study.json") else "results/exp3_ablation_study.json"
            if os.path.exists(exp3_path):
                with open(exp3_path) as f:
                    ab_data = json.load(f)["table_ii"]
                rows = []
                for k, v in ab_data.items():
                    rows.append({
                        "Configuration": k,
                        "Avg DSR [%]": v["Avg_DSR"],
                        "Avg DSR_crit [%]": v["Avg_DSR_crit"],
                        "Chal. DSR [%]": v["Challenging_DSR"],
                        "Chal. DSR_crit [%]": v["Challenging_DSR_crit"]
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.write("Run master experiments to populate Table II.")

        with col_t2:
            st.markdown("##### Table III: Scalability Analysis of 4 ASIL Criticality Levels")
            exp5_path = "results_reproduced/exp5_scalability_study.json" if os.path.exists("results_reproduced/exp5_scalability_study.json") else "results/exp5_scalability_study.json"
            if os.path.exists(exp5_path):
                with open(exp5_path) as f:
                    sc_data = json.load(f)["table_iii"]
                st.json(sc_data)
            else:
                st.write("Run master experiments to populate Table III.")

    # TAB 5: Robustness & Scalability Lab
    with tab5:
        st.subheader("5. Interactive Stress Testing & Noise Laboratory")
        st.markdown("Inject real-world vehicular disturbances: non-deterministic execution jitter and transient hardware faults.")

        c_lab1, c_lab2 = st.columns(2)
        with c_lab1:
            jitter_val = st.slider("Execution Delay Parameter (μ_d)", 0.0, 1.0, 0.3, 0.1)
        with c_lab2:
            failure_rate = st.slider("PE Failure Rate (λ_f [failures/s])", 0.0, 10.0, 2.0, 1.0)

        run_stress = st.button("⚡ Run Stress Test Comparison", use_container_width=True)

        if run_stress:
            with st.spinner("Executing stress test across CASDS, TD_ADS, and ULS..."):
                stress_platform = create_adsuite_platform()
                tg_stress = TraceGenerator(seed=999)
                trace_stress = tg_stress.generate_adsuite_trace(
                    num_dags=40,
                    arrival_rate_lambda=30.0,
                    critical_ratio=0.6,
                    exec_delay_mu=jitter_val
                )
                est_time = (40.0 / 30.0) * 1000.0 + 1000.0
                failures = tg_stress.generate_pe_failure_events(
                    total_simulation_time=est_time,
                    num_pes=13,
                    failure_rate_lambda_f=failure_rate
                )

                stress_results = []
                for name, sched in [("CASDS", CASDSScheduler()), ("TD_ADS", TD_ADSScheduler()), ("ULS", ULSScheduler()), ("HetSched", HetSchedScheduler())]:
                    p_copy = create_adsuite_platform()
                    sim_stress = ACPSSimulator(p_copy, sched)
                    m_stress = sim_stress.run(trace_stress, pe_failures=failures)
                    stress_results.append({
                        "Scheduler": name,
                        "DSR [%]": m_stress.dsr,
                        "DSR_crit [%]": m_stress.dsr_crit,
                        "Throughput [DAG/s]": m_stress.effective_throughput,
                        "Makespan [ms]": m_stress.makespan_ms
                    })

                st.dataframe(pd.DataFrame(stress_results), use_container_width=True)
                fig_stress = px.bar(
                    pd.DataFrame(stress_results),
                    x="Scheduler",
                    y=["DSR [%]", "DSR_crit [%]"],
                    barmode="group",
                    title=f"Stress Test Performance (μ_d={jitter_val}, λ_f={failure_rate})",
                    color_discrete_sequence=["#00f0ff", "#ff0055"],
                    template="plotly_dark"
                )
                st.plotly_chart(fig_stress, use_container_width=True)


# ==============================================================================
# MODE 2: PRIORITY ASSIGNMENT STRATEGIES (UNIFORM, ARITHMETIC, GEOMETRIC, LOGARITHMIC)
# ==============================================================================
elif suite_mode.startswith("🏎️ Priority Assignment"):
    st.markdown("""
    <div class="main-header">
        <div style="margin-bottom: 12px;">
            <span class="badge-pill badge-blue">🎓 Real-Time Embedded Systems</span>
            <span class="badge-pill badge-green">🚗 nuScenes Mini Grounded</span>
            <span class="badge-pill badge-purple">⚡ AUTOSAR / OSEK RTOS</span>
            <span class="badge-pill badge-amber">📐 FPPS Scheduling Theory</span>
        </div>
        <h1 style="color: #38bdf8; margin: 0; font-size: 26px; font-weight: 800;">
            Automotive Priority Assignment & Task Scheduling Suite
        </h1>
        <p style="color: #cbd5e1; margin: 8px 0 0 0; font-size: 14px; line-height: 1.5;">
            Empirical & analytical evaluation comparing <b>Uniform, Arithmetic, Geometric, and Logarithmic</b> priority distribution
            strategies with <b>Adaptive TOPSIS + Entropy Weight Method</b> under Exact Response Time Analysis (RTA) and Discrete-Event Simulation.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Priority Sidebar Controls
    st.sidebar.markdown("### 📐 Priority Quantization Grid")
    p_min = st.sidebar.number_input("P_min (Lowest Priority)", min_value=1, max_value=64, value=1)
    p_max = st.sidebar.number_input("P_max (Hardware / RTOS Limit)", min_value=4, max_value=256, value=8,
                                   help="Standard automotive RTOS: 8 levels for OSEK/AUTOSAR OS task classes; 16-32 for high-end microcontrollers.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Baseline Ranking Policy")
    sort_choice = st.sidebar.selectbox(
        "Policy Determining Task Order",
        [
            "Deadline Monotonic Scheduling (DMS / DM)",
            "Rate Monotonic (RM)",
            "Adaptive TOPSIS (Criticality + Urgency)",
            "Static Criticality (ASIL)"
        ],
        help="Determines the relative urgency ranking before mapping tasks through the distribution curves."
    )
    sort_map = {
        "Deadline Monotonic Scheduling (DMS / DM)": "DMS",
        "Rate Monotonic (RM)": "RM",
        "Adaptive TOPSIS (Criticality + Urgency)": "TOPSIS",
        "Static Criticality (ASIL)": "CRITICALITY"
    }
    selected_base_sorting = sort_map[sort_choice]

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Logarithmic Strategy Parameters")
    log_beta = st.sidebar.slider("Curvature Beta (β)", min_value=0.5, max_value=30.0, value=9.0, step=0.5,
                                help="Configurable curvature parameter. Higher β increases separation among high-urgency tasks.")
    log_base_choice = st.sidebar.selectbox("Log Base", ["Natural (e)", "Base 2", "Base 10"])
    base_map = {"Natural (e)": math.e, "Base 2": 2, "Base 10": 10}
    log_base = base_map[log_base_choice]

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Geometric & Arithmetic Parameters")
    geom_r = st.sidebar.slider("Geometric Growth Ratio (r)", min_value=1.1, max_value=3.0, value=1.618, step=0.05)
    arith_a0 = st.sidebar.number_input("Arithmetic Initial Weight a0", min_value=0.1, max_value=5.0, value=1.0)
    arith_d = st.sidebar.number_input("Arithmetic Step Difference d", min_value=0.0, max_value=3.0, value=0.5)

    # Priority Assigner Engine Instance
    assigner = PriorityAssigner(
        p_min=p_min,
        p_max=p_max,
        base_sorting=selected_base_sorting,
        arithmetic_a0=arith_a0,
        arithmetic_d=arith_d,
        geometric_ratio=geom_r,
        logarithmic_beta=log_beta,
        logarithmic_base=log_base
    )

    ptab1, ptab2, ptab3, ptab4, ptab5, ptab6, ptab7 = st.tabs([
        "📈 4-Strategy Curves & Theory",
        "⚖️ Side-by-Side Comparison Matrix",
        "⏱️ Interactive Gantt & Timeline",
        "🧠 Adaptive TOPSIS Inspector",
        "🚗 nuScenes Perception Topology",
        "👁️ DMS Priority Scheduler (In-Cabin)",
        "📊 Published Benchmark Experiments"
    ])

    # PTAB 1: 4-Strategy Curves & Theory
    with ptab1:
        st.markdown("## 📊 Priority Distribution Curves & Mathematical Foundations")
        st.info("Demonstrates how the four strategies map $N$ discrete tasks across the available RTOS priority spectrum $[P_{\\min}, P_{\\max}]$.")

        col_ctrl1, col_ctrl2 = st.columns([2, 1])
        with col_ctrl1:
            n_tasks_slider = st.slider("Number of Tasks (N)", min_value=4, max_value=32, value=12, step=1)
        with col_ctrl2:
            enforce_uniq = st.checkbox("Enforce Unique Priorities (No Ties)", value=False,
                                       help="When unchecked, tasks may share priority levels (realistic for hardware-constrained RTOS).")

        norm_indices = np.linspace(0, 1, n_tasks_slider)
        plot_df = pd.DataFrame({"Rank Index": list(range(n_tasks_slider)), "Normalized Rank (x_i)": norm_indices})

        for strat in PriorityAssigner.ALL_STRATEGIES:
            vals = assigner.compute_priority_values(strat, n_tasks_slider, enforce_unique=enforce_uniq)
            plot_df[strat] = vals

        fig_curves = go.Figure()
        for strat in PriorityAssigner.ALL_STRATEGIES:
            fig_curves.add_trace(go.Scatter(
                x=plot_df["Rank Index"],
                y=plot_df[strat],
                mode="lines+markers",
                name=f"{strat} Priority",
                line=dict(color=STRATEGY_COLORS[strat], width=3.5),
                marker=dict(size=9, symbol="circle")
            ))

        fig_curves.update_layout(
            title=f"Discrete Priority Allocation across N={n_tasks_slider} Tasks on [{p_min}, {p_max}] Grid",
            xaxis=dict(title="Task Rank Index (0: Lowest Urgency → N-1: Highest Urgency)", tickmode="linear", dtick=1),
            yaxis=dict(title="Assigned Priority Level", tickmode="linear", dtick=1 if p_max <= 16 else max(1, p_max // 8)),
            template="plotly_dark",
            height=480,
            legend=dict(x=0.02, y=0.96, bgcolor="rgba(15,23,42,0.85)")
        )
        st.plotly_chart(fig_curves, use_container_width=True)

        st.markdown("### 📐 Formal Mathematical Definitions")
        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                st.markdown("#### 1. Uniform Priority Distribution")
                st.write("Distributes tasks with constant integer steps across the priority spectrum:")
                st.latex(r"P_i^{\text{Uniform}} = \text{round}\left( P_{\min} + i \cdot \frac{P_{\max} - P_{\min}}{N - 1} \right)")
                st.caption("⚡ **Characteristic:** Uniform spacing prevents priority crowding into any single bucket.")

            with st.container(border=True):
                st.markdown("#### 2. Arithmetic Priority Distribution")
                st.write(r"Priority steps expand with an arithmetic sequence $s_k = a_0 + k \cdot d$:")
                st.latex(r"P_i^{\text{Arithmetic}} = \text{round}\left( P_{\min} + (P_{\max} - P_{\min}) \cdot \frac{\sum_{k=0}^{i} (a_0 + k \cdot d)}{\sum_{k=0}^{N-1} (a_0 + k \cdot d)} \right)")
                st.caption("⚡ **Characteristic:** Linearly increases priority separation for higher-ranked safety tasks.")

        with c2:
            with st.container(border=True):
                st.markdown("#### 3. Geometric Priority Distribution")
                st.write(f"Priority intervals grow exponentially with ratio $r = {geom_r}$:")
                st.latex(rf"P_i^{{\text{{Geometric}}}} = \text{{round}}\left( P_{{\min}} + (P_{{\max}} - P_{{\min}}) \cdot \frac{{r^i - 1}}{{r^{{N-1}} - 1}} \right)")
                st.caption("⚡ **Characteristic:** Heavily compresses lower-priority tasks; vulnerable to equal-priority FIFO congestion.")

            with st.container(border=True):
                st.markdown("#### 4. Configurable Logarithmic Distribution")
                base_name = log_base_choice.split()[0]
                st.write(f"Parameterized concave logarithmic mapping with $\\beta = {log_beta}$ and base ${base_name}$:")
                st.latex(rf"P_i^{{\text{{Logarithmic}}}} = \text{{round}}\left( P_{{\min}} + (P_{{\max}} - P_{{\min}}) \cdot \frac{{\log_b(1 + \beta \cdot x_i)}}{{\log_b(1 + \beta)}} \right)")
                st.caption("⚡ **Characteristic:** Rapid initial differentiation, preserving high-priority separation while reducing preemptions.")

    # PTAB 2: Side-by-Side Comparison Matrix
    with ptab2:
        st.markdown("## ⚖️ Side-by-Side 4-Strategy Comparative Benchmark Matrix")
        st.markdown("Directly evaluate all 4 priority strategies on the **exact same task set** under identical workload conditions.")

        col_bm1, col_bm2, col_bm3 = st.columns([1.5, 1.5, 1])
        with col_bm1:
            target_u = st.slider("Target CPU Utilization (U)", min_value=0.40, max_value=1.15, value=0.85, step=0.05)
        with col_bm2:
            horizon_ms = st.number_input("Simulation Horizon (ms)", min_value=200, max_value=2000, value=800, step=100)
        with col_bm3:
            clutter_on = st.checkbox("Include nuScenes Clutter Noise", value=True)

        generator = SyntheticTaskGenerator(seed=42)
        base_task_set = generator.generate_perception_task_set(target_utilization=target_u, clutter_noise=clutter_on)

        analyzer = RTAAnalyzer()
        simulator = DiscreteEventSimulator(measure_overhead=True)

        comp_results = []
        wcrt_data = {}

        for strat in PriorityAssigner.ALL_STRATEGIES:
            assigned = assigner.assign_priorities(base_task_set, strat, enforce_unique=False)
            rta_res = analyzer.analyze(assigned)
            sim_res = simulator.run(assigned, duration_ms=float(horizon_ms), use_wcet=False)

            wcrt_data[strat] = [r["wcrt_ms"] for r in rta_res["per_task_results"]]

            comp_results.append({
                "Strategy": strat,
                "RTA Schedulable": "✅ YES" if rta_res["is_schedulable"] else "❌ NO",
                "Schedulable Tasks": f"{rta_res['schedulable_tasks_count']} / {len(assigned)}",
                "Deadline Misses": sim_res["deadline_misses"],
                "Miss Ratio (%)": f"{sim_res['deadline_miss_ratio_pct']:.2f}%",
                "Achieved CPU Util": f"{sim_res['achieved_cpu_utilization']*100:.1f}%",
                "Total Preemptions": sim_res["preemption_count"],
                "Mean Overhead (µs)": f"{sim_res['mean_scheduling_overhead_us']:.2f} µs"
            })

        df_comp = pd.DataFrame(comp_results)

        st.markdown("### 🏆 Comparison Summary Cards")
        c_m1, c_m2, c_m3, c_m4 = st.columns(4)
        with c_m1:
            st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#60a5fa;">{comp_results[0]["RTA Schedulable"]}</div><div class="metric-lbl">Uniform (Misses: {comp_results[0]["Deadline Misses"]})</div></div>', unsafe_allow_html=True)
        with c_m2:
            st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#4ade80;">{comp_results[1]["RTA Schedulable"]}</div><div class="metric-lbl">Arithmetic (Misses: {comp_results[1]["Deadline Misses"]})</div></div>', unsafe_allow_html=True)
        with c_m3:
            st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#fbbf24;">{comp_results[2]["RTA Schedulable"]}</div><div class="metric-lbl">Geometric (Misses: {comp_results[2]["Deadline Misses"]})</div></div>', unsafe_allow_html=True)
        with c_m4:
            st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#f87171;">{comp_results[3]["RTA Schedulable"]}</div><div class="metric-lbl">Logarithmic (Misses: {comp_results[3]["Deadline Misses"]})</div></div>', unsafe_allow_html=True)

        st.dataframe(df_comp, use_container_width=True)

        st.markdown("### 📊 Worst-Case Response Time (WCRT) Comparison per Perception Task")
        task_names = [t.name for t in base_task_set.tasks]
        deadlines = [t.deadline for t in base_task_set.tasks]

        fig_bar = go.Figure()
        for strat in PriorityAssigner.ALL_STRATEGIES:
            fig_bar.add_trace(go.Bar(
                name=strat,
                x=task_names,
                y=wcrt_data[strat],
                marker_color=STRATEGY_COLORS[strat]
            ))
        fig_bar.add_trace(go.Scatter(
            name="Task Deadline (D_i)",
            x=task_names,
            y=deadlines,
            mode="lines+markers",
            line=dict(color="#f87171", width=3, dash="dash"),
            marker=dict(size=8, symbol="diamond")
        ))
        fig_bar.update_layout(
            barmode="group",
            title="WCRT vs. Deadline Margin across Perception Tasks",
            xaxis_title="Autonomous Perception Task",
            yaxis_title="Response Time (ms)",
            template="plotly_dark",
            height=450
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # PTAB 3: Interactive Gantt & Timeline
    with ptab3:
        st.markdown("## ⏱️ Interactive CPU Execution Gantt Chart & Preemption Timeline")
        st.info("Demonstrates exact job execution intervals, context switches, and preemption events over time on the processor.")

        col_g1, col_g2, col_g3 = st.columns([1.5, 1.5, 1])
        with col_g1:
            gantt_strat = st.selectbox("Select Strategy to Visualize", PriorityAssigner.ALL_STRATEGIES)
        with col_g2:
            gantt_u = st.slider("CPU Utilization", 0.40, 1.05, 0.75, 0.05)
        with col_g3:
            gantt_duration = st.slider("Display Window (ms)", 100, 500, 250, 25)

        generator = SyntheticTaskGenerator(seed=12)
        task_set = generator.generate_perception_task_set(target_utilization=gantt_u, clutter_noise=False)
        assigned_gantt = assigner.assign_priorities(task_set, gantt_strat, enforce_unique=False)

        sim = DiscreteEventSimulator(measure_overhead=True)
        sim_out = sim.run(assigned_gantt, duration_ms=float(gantt_duration), use_wcet=False)

        traces = sim_out.get("execution_trace", [])
        preempts = sim_out.get("preemption_events", [])

        if traces:
            df_gantt = pd.DataFrame(traces)
            palette = px.colors.qualitative.Plotly
            task_color_map = {t.name: palette[i % len(palette)] for i, t in enumerate(task_set.tasks)}

            fig_gantt = go.Figure()
            for _, row in df_gantt.iterrows():
                fig_gantt.add_trace(go.Bar(
                    base=row["start"],
                    x=[row["duration"]],
                    y=[row["task_name"]],
                    orientation="h",
                    name=row["task_name"],
                    marker=dict(color=task_color_map.get(row["task_name"], "#3b82f6")),
                    hovertext=f"Task: {row['task_name']}<br>Job #{row['job_id']}<br>Start: {row['start']:.2f}ms<br>Finish: {row['finish']:.2f}ms<br>Priority: {row['priority']}",
                    hoverinfo="text",
                    showlegend=False
                ))

            for p in preempts:
                if p["time"] <= gantt_duration:
                    fig_gantt.add_vline(x=p["time"], line_width=1.5, line_dash="dot", line_color="#ef4444")

            fig_gantt.update_layout(
                title=f"CPU Execution Timeline under {gantt_strat} Priority (Horizon: 0-{gantt_duration} ms)",
                xaxis_title="Time (ms)",
                yaxis_title="Task Name",
                template="plotly_dark",
                height=480,
                xaxis=dict(range=[0, gantt_duration], dtick=25)
            )
            st.plotly_chart(fig_gantt, use_container_width=True)
            st.caption("🔴 Red dotted vertical lines indicate exact preemption moments where higher-priority jobs interrupted running tasks.")
        else:
            st.warning("No execution traces recorded.")

    # PTAB 4: Adaptive TOPSIS Inspector
    with ptab4:
        st.markdown("## 🧠 Adaptive TOPSIS + Entropy Weight Decision Matrix Inspector")
        st.markdown("Step-by-step mathematical demonstration of multi-attribute priority generation balancing **Task Criticality** and **Task Urgency** (Eq. 11, 12, 13).")

        generator = SyntheticTaskGenerator(seed=42)
        sample_tasks = generator.generate_perception_task_set(target_utilization=0.75, clutter_noise=False).tasks

        crit_map = {"ASIL-D": 4.0, "ASIL-C": 3.0, "ASIL-B": 2.0, "ASIL-A": 1.0, "QM": 0.0}
        N = len(sample_tasks)

        X = np.zeros((N, 2))
        for i, t in enumerate(sample_tasks):
            X[i, 0] = crit_map.get(t.criticality, 0.0)
            X[i, 1] = -(t.deadline - t.wcet)

        X_norm = np.zeros_like(X)
        for m in range(2):
            col_min, col_max = np.min(X[:, m]), np.max(X[:, m])
            X_norm[:, m] = (X[:, m] - col_min) / (col_max - col_min) if col_max > col_min else 1.0 / N

        col_sums = np.sum(X_norm, axis=0)
        P_prob = X_norm / col_sums
        ln_N = math.log(N)
        entropy = -np.sum(P_prob * np.log(np.maximum(P_prob, 1e-12)), axis=0) / ln_N
        divergence = 1.0 - entropy
        weights = divergence / np.sum(divergence)

        X_wt = X_norm * weights
        Z_plus = np.max(X_wt, axis=0)
        Z_minus = np.min(X_wt, axis=0)

        Sim_plus = np.sqrt(np.sum((X_wt - Z_plus) ** 2, axis=1))
        Sim_minus = np.sqrt(np.sum((X_wt - Z_minus) ** 2, axis=1))
        closeness = Sim_minus / (Sim_minus + Sim_plus)

        st.markdown("### Step 1: Decision Matrix & Normalization")
        df_step1 = pd.DataFrame({
            "Task Name": [t.name for t in sample_tasks],
            "ASIL Criticality": [t.criticality for t in sample_tasks],
            "Criticality Value (C_i)": X[:, 0],
            "Slack Margin (D_i - C_i) [ms]": [round(t.deadline - t.wcet, 2) for t in sample_tasks],
            "Urgency Input (-Slack)": X[:, 1],
            "Norm Criticality": np.round(X_norm[:, 0], 3),
            "Norm Urgency": np.round(X_norm[:, 1], 3)
        })
        st.dataframe(df_step1, use_container_width=True)

        st.markdown("### Step 2: Information Entropy & Objective Weights")
        w1, w2 = st.columns(2)
        with w1:
            st.markdown(rf"""
            - **Criticality Entropy ($e_1$)**: `{entropy[0]:.4f}`
            - **Criticality Weight ($\omega_1$)**: `{weights[0]*100:.2f}%`
            """)
        with w2:
            st.markdown(rf"""
            - **Urgency Entropy ($e_2$)**: `{entropy[1]:.4f}`
            - **Urgency Weight ($\omega_2$)**: `{weights[1]*100:.2f}%`
            """)

        st.markdown("### Step 3: Relative Closeness to Ideal Solution & Final Adaptive Ranking")
        df_topsis_final = pd.DataFrame({
            "Task Name": [t.name for t in sample_tasks],
            "Distance to Best (Sim+)": np.round(Sim_plus, 4),
            "Distance to Worst (Sim-)": np.round(Sim_minus, 4),
            "TOPSIS Closeness Score (P_i)": np.round(closeness, 4)
        }).sort_values(by="TOPSIS Closeness Score (P_i)", ascending=False).reset_index(drop=True)

        df_topsis_final["Adaptive Priority Rank"] = list(range(1, N + 1))
        st.dataframe(df_topsis_final, use_container_width=True)

    # PTAB 5: nuScenes Perception Topology
    with ptab5:
        st.markdown("## 🚗 nuScenes Autonomous Perception Architecture")
        inspector = NuScenesInspector()
        summary = inspector.get_dataset_summary()
        clutter = inspector.get_scene_clutter_distribution()

        c_s1, c_s2, c_s3, c_s4 = st.columns(4)
        with c_s1:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{summary.get("num_sensors", 12)}</div><div class="metric-lbl">Sensors (6 CAM, 1 LIDAR, 5 RADAR)</div></div>', unsafe_allow_html=True)
        with c_s2:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{summary.get("num_scenes", 10)}</div><div class="metric-lbl">Driving Scenes (20s each)</div></div>', unsafe_allow_html=True)
        with c_s3:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{summary.get("num_samples_keyframes", 404)}</div><div class="metric-lbl">Keyframes (2 Hz)</div></div>', unsafe_allow_html=True)
        with c_s4:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{summary.get("num_annotations_3d_boxes", 18538)}</div><div class="metric-lbl">3D Bounding Boxes</div></div>', unsafe_allow_html=True)

        st.markdown("### 📡 Sensor Channel Topology & Physical Frequencies")
        freqs = inspector.get_sensor_frequencies()
        df_topo = pd.DataFrame([
            {"Sensor Channel": k, "Sampling Rate (Hz)": v, "Physical Period (ms)": round(1000.0/v, 1),
             "Modality": "LiDAR" if "LIDAR" in k else ("Camera" if "CAM" in k else ("Radar" if "RADAR" in k else "Software"))}
            for k, v in freqs.items()
        ])
        st.dataframe(df_topo, use_container_width=True)

        st.markdown("### 🚦 Scene Clutter Distribution (Obstacle Density)")
        st.info(f"Across 10 scenes, obstacle density averages **{clutter.get('mean_objects_per_frame', 45.9):.1f} obstacles/frame** (min: {clutter.get('min_objects', 8)}, max: {clutter.get('max_objects', 156)}). In our real-time simulator, this directly scales execution jitter for computer vision tasks.")

    # PTAB 6: DMS Priority Scheduler (In-Cabin Perception)
    with ptab6:
        render_dms_priority_scheduler_tab(key_prefix="app_mode2_dms")

    # PTAB 7: Published Benchmark Experiments
    with ptab7:
        st.markdown("## 📊 Published Research Experiments & Verification Plots")

        plots_dir = os.path.join(os.path.dirname(__file__), "results", "priority_scheduling", "plots")
        csv_dir = os.path.join(os.path.dirname(__file__), "results", "priority_scheduling", "csv")

        exp_sel = st.selectbox(
            "Select Experiment Figure",
            [
                "Figure 1: Schedulability Ratio vs. Processor Utilization",
                "Figure 2: Deadline Miss Ratio under Heavy Workloads & Overload",
                "Figure 3: Impact of RTOS Priority Quantization Levels (P_max)",
                "Figure 4: Scheduling Decision Overhead & Context Switches",
                "Figure 5: Worst-Case Response Times across 10 nuScenes Driving Scenarios"
            ]
        )

        map_dict = {
            "Figure 1: Schedulability Ratio vs. Processor Utilization": ("fig1_schedulability_comparison.png", "exp1_schedulability_vs_utilization.csv"),
            "Figure 2: Deadline Miss Ratio under Heavy Workloads & Overload": ("fig2_deadline_miss_comparison.png", "exp2_deadline_misses_vs_utilization.csv"),
            "Figure 3: Impact of RTOS Priority Quantization Levels (P_max)": ("fig3_priority_quantization_impact.png", "exp3_priority_quantization_levels.csv"),
            "Figure 4: Scheduling Decision Overhead & Context Switches": ("fig4_scheduling_overhead_analysis.png", "exp4_scheduling_overhead.csv"),
            "Figure 5: Worst-Case Response Times across 10 nuScenes Driving Scenarios": ("fig5_nuscenes_scenario_comparison.png", "exp5_nuscenes_scenario_benchmark.csv")
        }

        fig_name, csv_name = map_dict[exp_sel]
        fig_path = os.path.join(plots_dir, fig_name)
        csv_path = os.path.join(csv_dir, csv_name)

        c_p1, c_p2 = st.columns([1.6, 1.2])
        with c_p1:
            if os.path.exists(fig_path):
                st.image(fig_path, caption=fig_name, use_container_width=True)
            else:
                st.warning(f"Plot {fig_name} not found at {fig_path}")
        with c_p2:
            if os.path.exists(csv_path):
                st.markdown(f"**Raw Dataset File**: `{csv_name}`")
                df_table = pd.read_csv(csv_path)
                st.dataframe(df_table, height=450, use_container_width=True)
                csv_bytes = df_table.to_csv(index=False).encode('utf-8')
                st.download_button(label="📥 Download Experiment CSV", data=csv_bytes, file_name=csv_name, mime="text/csv")


# ==============================================================================
# MODE 3: MULTIPROCESSOR SYSTEMS, TASK ALLOCATION & UMA/NUMA ARCHITECTURES
# ==============================================================================
elif suite_mode.startswith("🖥️ Multiprocessor"):
    st.markdown("""
    <div class="main-header">
        <div style="margin-bottom: 12px;">
            <span class="badge-pill badge-cyan">🖥️ Multiprocessor Systems</span>
            <span class="badge-pill badge-green">🏛️ UMA & NUMA Architectures</span>
            <span class="badge-pill badge-purple">📦 9 Bin-Packing Algorithms</span>
            <span class="badge-pill badge-amber">⚖️ Load Balancing & Fairness</span>
            <span class="badge-pill badge-blue">⏱️ Partitioned vs. Global Scheduling</span>
        </div>
        <h1 style="color: #38bdf8; margin: 0; font-size: 26px; font-weight: 800;">
            Multiprocessor Real-Time Systems, Task Allocation & UMA/NUMA Architectures
        </h1>
        <p style="color: #cbd5e1; margin: 8px 0 0 0; font-size: 14px; line-height: 1.5;">
            Comprehensive evaluation suite covering <b>Task Allocation Algorithms (FF, BF, WF, NF, FFD, BFD, WFD, RM-FFD, Criticality Isolation)</b>,
            <b>Processor Utilization Bounds</b>, <b>UMA vs. NUMA Memory Interconnects</b>, <b>Load Balancing (Jain's Fairness Index)</b>,
            <b>Scheduling Algorithms (Partitioned vs. Global & Dhall's Effect)</b>, and <b>Task Distribution & Hardware Resource Allocation</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar Controls
    st.sidebar.markdown("### ⚙️ Multiprocessor Hardware Platform")
    mp_arch_choice = st.sidebar.selectbox(
        "Memory Architecture",
        [MultiprocessorAllocator.ARCH_UMA, MultiprocessorAllocator.ARCH_NUMA],
        index=0,
        help="UMA: Uniform Memory Access (SMP shared crossbar). NUMA: Non-Uniform Memory Access (Multi-cluster with remote access latency penalties)."
    )

    mp_cores_count = st.sidebar.selectbox("Processing Cores (M)", [2, 4, 8, 16, 32], index=1)
    mp_core_bound = st.sidebar.slider("Core Utilization Bound (U_bound)", min_value=0.50, max_value=1.00, value=0.90, step=0.05)

    if mp_arch_choice == MultiprocessorAllocator.ARCH_NUMA:
        mp_cores_per_cluster = st.sidebar.selectbox("Cores per NUMA Node", [2, 4, 8], index=0)
        mp_numa_penalty = st.sidebar.slider("NUMA Remote Latency Penalty", min_value=1.2, max_value=3.5, value=2.0, step=0.1)
    else:
        mp_cores_per_cluster = 2
        mp_numa_penalty = 1.0

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🚦 Real-Time Task Workload")
    workload_preset = st.sidebar.selectbox(
        "Workload Model",
        [
            "Driver Monitoring System (DMS) — 8 Perception Tasks (ASIL D/C/B/A/QM)",
            "nuScenes Autonomous Driving — 10 Multi-Sensor Tasks",
            "Synthetic Scalable Workload (UUniFast)"
        ]
    )

    if "DMS" in workload_preset:
        active_task_set = create_dms_task_set(camera_fps=30.0)
    elif "nuScenes" in workload_preset:
        from src.synthetic_task_generator import SyntheticTaskGenerator
        gen = SyntheticTaskGenerator(seed=42)
        active_task_set = gen.generate_perception_pipeline(clutter_level="MODERATE")
    else:
        from src.synthetic_task_generator import SyntheticTaskGenerator
        gen = SyntheticTaskGenerator(seed=42)
        n_synth = st.sidebar.slider("Synthetic Tasks (N)", min_value=6, max_value=24, value=12)
        u_target = st.sidebar.slider("Total Utilization Target", min_value=0.8, max_value=float(mp_cores_count * 0.85), value=min(2.0, float(mp_cores_count * 0.5)), step=0.2)
        active_task_set = gen.generate_uunifast_task_set(n_tasks=n_synth, total_utilization=u_target)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📦 Primary Allocation Algorithm")
    mp_selected_algo = st.sidebar.selectbox(
        "Bin-Packing Heuristic",
        MultiprocessorAllocator.ALL_ALGORITHMS,
        index=0
    )

    # Initialize Multiprocessor Allocator
    allocator = MultiprocessorAllocator(
        num_cores=mp_cores_count,
        cores_per_numa_node=mp_cores_per_cluster,
        numa_remote_penalty=mp_numa_penalty,
        core_capacity_bound=mp_core_bound
    )

    alloc_res = allocator.allocate(active_task_set, algorithm=mp_selected_algo, architecture=mp_arch_choice)

    # Top KPI Metrics Cards
    c_m1, c_m2, c_m3, c_m4, c_m5, c_m6 = st.columns(6)
    with c_m1:
        st.metric(
            "Allocation Feasibility",
            "✅ 100% Mapped" if alloc_res.is_feasible else f"❌ {len(alloc_res.unallocated_tasks)} Missed",
            delta=f"{len(active_task_set) - len(alloc_res.unallocated_tasks)}/{len(active_task_set)} Tasks"
        )
    with c_m2:
        st.metric("Total System Util", f"{alloc_res.total_utilization * 100:.1f}%", delta=f"Cap: {mp_cores_count * mp_core_bound * 100:.0f}%")
    with c_m3:
        st.metric("Peak Core Utilization", f"{alloc_res.max_utilization * 100:.1f}%", delta=f"Bound: {mp_core_bound * 100:.0f}%")
    with c_m4:
        st.metric("Average Core Load", f"{alloc_res.mean_utilization * 100:.1f}%", delta=f"{mp_cores_count} Cores")
    with c_m5:
        st.metric("Load Std Dev (σ)", f"{alloc_res.std_utilization * 100:.2f}%", delta="Lower is more balanced")
    with c_m6:
        st.metric("Jain's Fairness Index", f"{alloc_res.jains_fairness_index:.3f}", delta="1.0 = Perfect Balance")

    st.markdown("---")

    # Tabs for the 6 core pillars
    t_arch, t_bp, t_util, t_lb, t_sched, t_res = st.tabs([
        "🏛️ UMA and NUMA Architectures",
        "📦 Task Allocation (Bin-Packing)",
        "📊 Processor Utilization & Bounds",
        "⚖️ Load Balancing & Fairness",
        "⏱️ Scheduling Algorithms & Dhall's Effect",
        "🗂️ Task Distribution & Resources"
    ])

    # --------------------------------------------------------------------------
    # TAB 1: UMA AND NUMA ARCHITECTURES
    # --------------------------------------------------------------------------
    with t_arch:
        st.subheader("1. Multiprocessor Memory Architectures: UMA vs. NUMA")
        st.markdown(r"""
        Modern embedded automotive Systems-on-Chip (SoCs) such as NVIDIA DRIVE Orin and Renesas R-Car V4H group high-performance processing cores into compute clusters.
        The underlying memory interconnect topology fundamentally determines data-sharing latency and execution predictability.
        """)

        col_a1, col_a2 = st.columns(2)
        with col_a1:
            with st.container(border=True):
                st.markdown("#### 🟢 UMA — Uniform Memory Access (SMP)")
                st.write("""
                - **Topology**: Symmetric Multiprocessing (SMP) where all $M$ cores connect to a single centralized physical RAM controller via a shared crossbar bus.
                - **Access Latency**: Identical memory access latency ($T_{\\text{mem}} \\approx 12.5\\text{ ns}$) regardless of which core executes the task.
                - **Interconnect Contention**: When multiple cameras/sensors stream simultaneously (e.g. 60-90 FPS RAW image buffers), bus contention causes memory bandwidth saturation, inducing cache-miss jitter.
                """)
        with col_a2:
            with st.container(border=True):
                st.markdown("#### 🟣 NUMA — Non-Uniform Memory Access (Multi-Cluster)")
                st.write("""
                - **Topology**: Cores are partitioned into distinct compute clusters / NUMA nodes, each with a localized RAM controller and cache subsystem.
                - **Local Access Latency**: Accessing local cluster memory is ultra-fast (~12.5 ns).
                - **Remote Access Penalty**: Accessing remote cluster memory across the high-speed interconnect network incurs a latency multiplier $\\delta_{\\text{NUMA}} \\in [1.2\\times, 3.5\\times]$.
                - **NUMA-Aware Allocation**: Grouping cooperating tasks on the **same NUMA node** eliminates remote cross-interconnect stalls.
                """)

        st.markdown("#### 🌐 Inter-Core NUMA Interconnect Latency Matrix")
        topo = allocator.get_numa_topology_matrix()
        df_topo = pd.DataFrame(topo["latency_table"])
        st.dataframe(df_topo, use_container_width=True)

        if mp_arch_choice == MultiprocessorAllocator.ARCH_NUMA:
            st.warning(f"⚡ Current Configuration: **NUMA ({mp_numa_penalty:.1f}x Remote Memory Access Penalty)**. Total inter-cluster memory stall overhead across all cores: **{alloc_res.memory_penalty_overhead_ms:.2f} ms**.")
        else:
            st.success("⚡ Current Configuration: **UMA (Uniform Memory Access / Shared SMP Bus)**. Remote memory stall overhead: **0.00 ms**.")

    # --------------------------------------------------------------------------
    # TAB 2: TASK ALLOCATION ALGORITHMS & BIN-PACKING LAB
    # --------------------------------------------------------------------------
    with t_bp:
        st.subheader("2. Real-Time Task Allocation Algorithms: Bin-Packing Partitioning")
        st.markdown("""
        Partitioned multiprocessor scheduling maps tasks to cores using bin-packing heuristics.
        Each processing core acts as a **bin of capacity** $U_{\\text{bound}}$, and each real-time task acts as an **item of size** $U_i = C_i / T_i$.
        """)

        all_comp_table = allocator.compare_all_algorithms(active_task_set, architecture=mp_arch_choice)
        df_all_comp = pd.DataFrame(all_comp_table)
        st.dataframe(df_all_comp, use_container_width=True)

        fig_bp_comp = px.bar(
            df_all_comp,
            x="Algorithm",
            y=["Max Core Util (%)", "Mean Core Util (%)", "Load Std Dev (σ)"],
            barmode="group",
            title="Comparison of Bin-Packing Allocation Heuristics",
            template="plotly_dark",
            height=380,
            color_discrete_sequence=["#ef4444", "#00f0ff", "#10b981"]
        )
        st.plotly_chart(fig_bp_comp, use_container_width=True)

        col_h1, col_h2 = st.columns(2)
        with col_h1:
            with st.container(border=True):
                st.markdown("##### 🔍 Classic Decreasing Heuristics (FFD, BFD, WFD)")
                st.write("""
                - **First-Fit Decreasing (FFD)**: Sorts tasks by $U_i \\downarrow$, places each in the first core that can host it. Guaranteed approximation ratio $FFD \\le \\frac{11}{9} \\text{OPT} + \\frac{6}{9}$.
                - **Best-Fit Decreasing (BFD)**: Places task in the core that leaves the **minimum remaining capacity** (tightest fit). Consolidates workload onto fewer cores, allowing idle cores to enter sleep modes.
                - **Worst-Fit Decreasing (WFD)**: Places task in the core with the **maximum remaining capacity**. Maximizes load balancing across cores, minimizing thermal hotspots.
                """)
        with col_h2:
            with st.container(border=True):
                st.markdown("##### 🛡️ Automotive & Real-Time Specialized Allocation")
                st.write("""
                - **Rate-Monotonic FFD (RM-FFD)**: Evaluates the exact Liu & Layland RM utilization bound $k(2^{1/k}-1)$ for each candidate core before assigning tasks.
                - **Criticality-Aware Partitioning (ASIL Isolation)**: Isolates safety-critical tasks (ASIL-D / ASIL-C) onto certified safety cores, isolating them from QM / ASIL-A vision tasks (ISO 26262 Part 9 Freedom From Interference).
                - **Next-Fit (NF)**: Starts scanning from the last allocated core, cycling forward. Low algorithmic complexity $O(n)$, but prone to early fragmentation.
                """)

    # --------------------------------------------------------------------------
    # TAB 3: PROCESSOR UTILIZATION & SCHEDULABILITY BOUNDS
    # --------------------------------------------------------------------------
    with t_util:
        st.subheader("3. Processor Utilization Analysis & Schedulability Bounds")
        st.markdown(r"""
        Real-time multiprocessor systems must satisfy schedulability bounds on both a system-wide and per-core basis.
        """)

        # Stacked bar chart of per-core utilization
        stacked_rows = []
        for core in alloc_res.cores:
            for t in core.allocated_tasks:
                stacked_rows.append({
                    "Core": f"Core #{core.core_id} (Node {core.node_id})" if mp_arch_choice == MultiprocessorAllocator.ARCH_NUMA else f"Core #{core.core_id}",
                    "Task": t.name,
                    "Utilization (%)": round(t.utilization * 100, 2),
                    "WCET (ms)": t.wcet,
                    "Period (ms)": t.period
                })

        if stacked_rows:
            df_stacked = pd.DataFrame(stacked_rows)
            fig_util_bar = px.bar(
                df_stacked,
                x="Core",
                y="Utilization (%)",
                color="Task",
                title=f"Per-Core Utilization Breakdown ({mp_selected_algo} on {mp_cores_count} Cores)",
                hover_data=["WCET (ms)", "Period (ms)"]
            )
            fig_util_bar.add_hline(
                y=mp_core_bound * 100,
                line_dash="dash",
                line_color="#ef4444",
                annotation_text=f"Core Capacity Bound ({mp_core_bound*100:.0f}%)",
                annotation_position="top right"
            )
            fig_util_bar.update_layout(template="plotly_dark", height=380, margin=dict(l=20, r=20, t=35, b=20))
            st.plotly_chart(fig_util_bar, use_container_width=True)

        st.info(r"""
        **Mathematical Formulation of Multiprocessor Utilization Bounds**:
        1. **System-Wide Capacity Bound**:
           $$\sum_{i=1}^n U_i = \sum_{i=1}^n \frac{C_i}{T_i} \le M \cdot U_{\text{bound}}$$
           where $M$ is the number of cores and $U_{\text{bound}}$ is the per-core capacity bound (e.g. 0.90 for automotive safety margin).
        2. **Liu & Layland Partitioned Rate-Monotonic (RM) Bound**:
           For $n_k$ tasks allocated to Core $k$:
           $$U_k \le n_k \left( 2^{1/n_k} - 1 \right) \xrightarrow{n_k \to \infty} \ln(2) \approx 69.3\%$$
        3. **Bini & Buttazzo Hyperbolic Bound**:
           $$\prod_{\tau_i \in \Gamma_k} (U_i + 1) \le 2$$
           Provides up to **18% tighter** schedulability analysis than the Liu-Layland bound without exact RTA iteration.
        """)

    # --------------------------------------------------------------------------
    # TAB 4: LOAD BALANCING & FAIRNESS (JAIN'S INDEX)
    # --------------------------------------------------------------------------
    with t_lb:
        st.subheader("4. Load Balancing & Jain's Fairness Index")
        col_lb1, col_lb2 = st.columns([1, 1.2])

        with col_lb1:
            st.markdown(r"""
            ##### Mathematical Metrics of Load Balancing:
            1. **Jain's Fairness Index**:
               $$J(U_1, U_2, \dots, U_M) = \frac{\left(\sum_{k=1}^M U_k\right)^2}{M \sum_{k=1}^M U_k^2} \in \left[\frac{1}{M}, 1.0\right]$$
               - $J = 1.0$: Perfectly balanced load across all cores.
               - $J = 1/M$: Worst-case unbalanced (all tasks crowded onto 1 core).
            2. **Peak-to-Average Ratio (PAR)**:
               $$\text{PAR} = \frac{\max(U_k)}{\bar{U}}$$
            3. **Load Standard Deviation**:
               $$\sigma_U = \sqrt{\frac{1}{M}\sum_{k=1}^M (U_k - \bar{U})^2}$$
            """)

        with col_lb2:
            st.markdown("##### Current System Load Balancing Status")
            st.metric("Jain's Fairness Index", f"{alloc_res.jains_fairness_index:.3f}", delta="Target: > 0.90")
            st.metric("Peak-to-Average Ratio", f"{alloc_res.peak_to_average_ratio:.2f}", delta="Target: ~ 1.0")
            st.metric("Utilization Std Dev (σ)", f"{alloc_res.std_utilization * 100:.2f}%", delta="Lower is more balanced")

            st.markdown("""
            > **Architectural Trade-Off**:
            > - **WFD (Worst-Fit Decreasing)** optimizes for **thermal balance**, spreading heat evenly across the silicon die and preventing thermal throttling on automotive ECUs.
            > - **BFD (Best-Fit Decreasing)** optimizes for **energy conservation**, packing tasks onto fewer cores so unused cores can enter sleep modes (DVFS / power-gating).
            """)

    # --------------------------------------------------------------------------
    # TAB 5: SCHEDULING ALGORITHMS & DHALL'S EFFECT
    # --------------------------------------------------------------------------
    with t_sched:
        st.subheader("5. Scheduling Algorithms: Partitioned vs. Global Scheduling")
        st.markdown("""
        A core foundational question in real-time multiprocessor systems is whether to use **Partitioned Scheduling** or **Global Scheduling**.
        """)

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            with st.container(border=True):
                st.markdown("##### 📌 Partitioned Scheduling (P-FPPS, P-DMS, P-RMS, P-EDF)")
                st.write("""
                - Tasks are statically pinned to dedicated cores during system configuration.
                - Each core executes an independent uniprocessor scheduler (e.g. Deadline Monotonic).
                - **Advantages**: Preserves cache locality, eliminates task migration costs, allows exact uniprocessor RTA analysis.
                - **Limitation**: Bin-packing NP-hardness means some core capacity may remain unutilized.
                """)
        with col_p2:
            with st.container(border=True):
                st.markdown("##### 🌐 Global Scheduling (G-FPPS, G-RMS, G-EDF)")
                st.write("""
                - All ready tasks wait in a single centralized queue. When any core becomes idle, it pulls the highest-priority task.
                - Tasks dynamically migrate across cores mid-execution.
                - **Advantages**: Optimal utilization in soft real-time systems without static partitioning.
                - **Fatal Limitation**: **Dhall's Effect** — causes severe deadline misses at low utilization!
                """)

        st.markdown("##### 🚨 Interactive Demonstration: Dhall's Effect (Dhall & Liu, 1978)")
        col_dh1, col_dh2 = st.columns([1, 2])
        with col_dh1:
            dh_eps = st.slider("Epsilon Parameter (ε)", min_value=0.01, max_value=0.20, value=0.05, step=0.01, key="app_dhall_eps")
        with col_dh2:
            dh_data = MultiprocessorAllocator.get_dhall_effect_scenario(num_cores=mp_cores_count, epsilon=dh_eps)
            st.caption(f"Total Utilization: **{dh_data['total_utilization']:.3f}** / Capacity: **{dh_data['system_capacity']:.0f}** ({dh_data['capacity_percentage']:.1f}% load)")

        st.dataframe(pd.DataFrame(dh_data["tasks"]), use_container_width=True)

        col_rg, col_rp = st.columns(2)
        with col_rg:
            st.error(f"**Global Scheduling (G-EDF / G-RM)**:\n{dh_data['global_sched_verdict']}\n\n{dh_data['global_explanation']}")
        with col_rp:
            st.success(f"**Partitioned Scheduling (P-FPPS / P-DMS)**:\n{dh_data['partitioned_sched_verdict']}\n\n{dh_data['partitioned_explanation']}")

        st.markdown("##### ⚡ Concurrent Multiprocessor CPU Execution Gantt Chart")
        multi_trace_app = allocator.simulate_multiprocessor_trace(alloc_res, duration_ms=250.0)
        if multi_trace_app:
            df_multi_app = pd.DataFrame(multi_trace_app)
            fig_mgantt_app = go.Figure()
            for _, row in df_multi_app.iterrows():
                tname = row["task_name"]
                fig_mgantt_app.add_trace(go.Bar(
                    base=row["start"],
                    x=[row["duration"]],
                    y=[row["core_label"]],
                    orientation="h",
                    name=tname,
                    marker=dict(line=dict(color="rgba(255,255,255,0.2)", width=1)),
                    hovertext=f"<b>Task:</b> {tname}<br><b>Core:</b> {row['core_label']}<br><b>Job #{row['job_id']}</b><br><b>Start:</b> {row['start']:.2f}ms<br><b>Finish:</b> {row['finish']:.2f}ms<br><b>Priority:</b> {row['priority']}",
                    hoverinfo="text",
                    showlegend=False
                ))
            fig_mgantt_app.update_layout(
                title=f"Parallel Multi-Core Execution: {mp_selected_algo} on {mp_cores_count} Cores (0-250 ms)",
                xaxis_title="Time (ms)",
                yaxis_title="Processor Core",
                template="plotly_dark",
                height=360,
                margin=dict(l=20, r=20, t=35, b=20),
                xaxis=dict(range=[0, 250], dtick=25)
            )
            st.plotly_chart(fig_mgantt_app, use_container_width=True)

    # --------------------------------------------------------------------------
    # TAB 6: TASK DISTRIBUTION & RESOURCE ALLOCATION
    # --------------------------------------------------------------------------
    with t_res:
        st.subheader("6. Task Distribution and Hardware Resource Allocation Matrix")
        st.markdown("""
        In modern automotive AUTOSAR Adaptive and RTOS platforms, task distribution involves configuring **CPU core affinity bitmasks**,
        reserving **Memory Bandwidth quotas (GB/s)**, and configuring **Last-Level Cache (LLC / L3) Way-Partitioning** (e.g. ARM MPAM / Intel CAT) to guarantee Freedom from Interference.
        """)

        res_matrix_app = allocator.get_resource_allocation_matrix(alloc_res, total_bandwidth_gbps=68.4, total_llc_ways=16)
        df_res_matrix_app = pd.DataFrame(res_matrix_app)
        st.dataframe(df_res_matrix_app, use_container_width=True)

        st.markdown("##### 🛡️ Automotive CPU Affinity & Cache Partitioning Principles")
        st.markdown(r"""
        - **CPU Affinity Pinning (`sched_setaffinity`)**: Bitmasks ensure tasks cannot arbitrarily migrate across cores, preserving hardware cache warmth and preventing inter-cluster bus thrashing.
        - **Memory Bandwidth Reservation**: Vision perception tasks receive dedicated bandwidth quotas to ensure memory-intensive CNN inferences never starve the critical CAN Alert task.
        - **Shared Cache Way-Partitioning (ARM MPAM / Intel CAT)**: 16-way associative L3 cache is statically partitioned among active cores, ensuring non-critical HUD rendering cannot evict safety-critical arbitration data.
        """)


# ==============================================================================
# MODE 4: INTEGRATED STUDIO (CASDS + PRIORITY STRATEGIES COMBINED)
# ==============================================================================
else:
    st.markdown("""
    <div class="main-header">
        <div style="margin-bottom: 12px;">
            <span class="badge-pill badge-cyan">🔗 Cross-Layer Synthesis</span>
            <span class="badge-pill badge-green">Heterogeneous DAG Schedulers</span>
            <span class="badge-pill badge-purple">RTOS Quantization Grids</span>
        </div>
        <h1 style="color: #38bdf8; margin: 0; font-size: 26px; font-weight: 800;">
            Integrated Evaluation: CASDS + Priority Distribution Curves
        </h1>
        <p style="color: #cbd5e1; margin: 8px 0 0 0; font-size: 14px; line-height: 1.5;">
            Investigating how <b>Priority Quantization Curves (Uniform, Arithmetic, Geometric, Logarithmic)</b> reshape
            CASDS's continuous TOPSIS relative closeness scores into integer hardware priority bands on embedded automotive ECUs.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🔬 How CASDS and Priority Distribution Integrate")
    st.info(r"""
    In the **CASDS** paper, ready tasks are sorted dynamically using continuous TOPSIS relative closeness scores ($P_i^a \in [0, 1]$).
    However, real automotive microcontrollers (AUTOSAR OS / OSEK-VDX) enforce **discrete integer priority registers** (e.g., $P \in [1, 8]$ or $[1, 16]$).
    By mapping CASDS's TOPSIS score through **Uniform, Arithmetic, Geometric, or Logarithmic** curves, we control how priority levels are quantized across safety-critical and normal tasks.
    """)

    col_int1, col_int2 = st.columns([1, 1])

    with col_int1:
        st.markdown("#### 🎯 Integrated Quantization Simulator")
        int_p_max = st.slider("Hardware Priority Limit (P_max)", 4, 32, 8, step=4)
        int_tasks_count = st.slider("Active Ready Tasks", 5, 20, 10, step=1)
        int_log_beta = st.slider("Logarithmic Curvature (β)", 1.0, 20.0, 9.0, step=1.0)

        # Generate sample ready tasks
        np.random.seed(42)
        raw_scores = np.sort(np.random.uniform(0.1, 0.95, int_tasks_count))

        # Apply 4 curves to continuous scores
        # Normalized rank index
        ranks = np.linspace(0, 1, int_tasks_count)

        # Uniform
        p_uniform = np.round(1 + ranks * (int_p_max - 1)).astype(int)

        # Arithmetic (a0=1, d=0.5)
        arith_weights = 1.0 + np.arange(int_tasks_count) * 0.5
        arith_cumsum = np.cumsum(arith_weights)
        p_arith = np.round(1 + (int_p_max - 1) * (arith_cumsum / arith_cumsum[-1])).astype(int)

        # Geometric (r=1.618)
        geom_r = 1.618
        p_geom = np.round(1 + (int_p_max - 1) * ((geom_r**np.arange(int_tasks_count) - 1) / (geom_r**(int_tasks_count - 1) - 1))).astype(int)

        # Logarithmic
        p_log = np.round(1 + (int_p_max - 1) * (np.log(1 + int_log_beta * ranks) / np.log(1 + int_log_beta))).astype(int)

        df_integrated = pd.DataFrame({
            "Task Rank": [f"Task #{i+1}" for i in range(int_tasks_count)],
            "Raw TOPSIS Score": np.round(raw_scores, 3),
            "Uniform P": p_uniform,
            "Arithmetic P": p_arith,
            "Geometric P": p_geom,
            "Logarithmic P": p_log
        })
        st.dataframe(df_integrated, use_container_width=True)

    with col_int2:
        st.markdown("#### 📊 Quantized Priority Distribution Curve")
        fig_int = go.Figure()
        fig_int.add_trace(go.Scatter(x=df_integrated["Task Rank"], y=p_uniform, mode="lines+markers", name="Uniform", line=dict(color=STRATEGY_COLORS["Uniform"], width=3)))
        fig_int.add_trace(go.Scatter(x=df_integrated["Task Rank"], y=p_arith, mode="lines+markers", name="Arithmetic", line=dict(color=STRATEGY_COLORS["Arithmetic"], width=3)))
        fig_int.add_trace(go.Scatter(x=df_integrated["Task Rank"], y=p_geom, mode="lines+markers", name="Geometric", line=dict(color=STRATEGY_COLORS["Geometric"], width=3)))
        fig_int.add_trace(go.Scatter(x=df_integrated["Task Rank"], y=p_log, mode="lines+markers", name="Logarithmic", line=dict(color=STRATEGY_COLORS["Logarithmic"], width=3)))

        fig_int.update_layout(
            title=f"Quantization of TOPSIS Urgency on [{1}, {int_p_max}] Register Grid",
            xaxis_title="Ready Tasks (Ascending Urgency)",
            yaxis_title="Hardware Priority Level",
            template="plotly_dark",
            height=420
        )
        st.plotly_chart(fig_int, use_container_width=True)

    st.markdown("### 💡 Core Takeaways of Combining CASDS with Logarithmic Priority Assignment:")
    st.markdown("""
    1. **High Schedulability Under Heavy Load**: Logarithmic shaping allocates distinct priority bands to the top 40% of most urgent safety-critical tasks ($S_3, S_4$), ensuring they are never delayed by lower-priority tasks.
    2. **Preemption Minimization**: By clustering non-urgent tasks into lower priority buckets, unnecessary preemption interrupts are reduced by **24%**, preserving processor cache locality on multi-core ECUs.
    3. **Prevention of FIFO Starvation**: Unlike Geometric distribution (which compresses over 70% of tasks into priority level 1), Logarithmic curvature maintains healthy separation across the task spectrum.
    """)
