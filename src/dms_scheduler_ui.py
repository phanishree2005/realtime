"""
Interactive Streamlit UI Module: Real-Time Priority Scheduler for Driver Monitoring System (DMS).
Implements:
1. DMS Task Architecture & ISO 26262 ASIL Configuration
2. Priority Scheduling Strategies: Deadline Monotonic (DMS), Rate Monotonic (RMS),
   Adaptive TOPSIS (Urgency + Criticality), Uniform, Arithmetic, Geometric, and Logarithmic
3. Exact Response Time Analysis (RTA) Engine & Schedulability Inspector
4. High-Resolution Interactive Plotly Gantt Chart of Discrete-Event CPU Execution
5. Priority Distribution Curves & Overload Resilience Lab
"""

import math
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.dms_task_model import create_dms_task_set, get_dms_task_metadata
from src.priority_assigner import PriorityAssigner
from src.rta_analyzer import RTAAnalyzer
from src.discrete_event_simulator import DiscreteEventSimulator
from src.multiprocessor_allocator import MultiprocessorAllocator


ASIL_COLORS = {
    "ASIL-D": "#ef4444",  # Red
    "ASIL-C": "#f97316",  # Orange
    "ASIL-B": "#3b82f6",  # Blue
    "ASIL-A": "#10b981",  # Green
    "QM":     "#64748b"   # Slate Gray
}

TASK_COLORS = {
    "T1_CAM_ISP":   "#06b6d4",  # Cyan
    "T2_FACE_LOC":  "#8b5cf6",  # Purple
    "T3_GAZE_9Z":   "#3b82f6",  # Blue
    "T4_PERCLOS":   "#f59e0b",  # Amber
    "T5_DISTRACT":  "#ec4899",  # Pink
    "T6_CASDS_ARB": "#10b981",  # Emerald Green
    "T7_CAN_ALERT": "#ef4444",  # Crimson Red
    "T8_HUD_RENDER":"#64748b"   # Slate
}


def render_dms_priority_scheduler_tab(key_prefix: str = "dms_sched"):
    """
    Renders the complete Real-Time Priority Scheduler tab for Driver Monitoring System.
    Can be seamlessly embedded in any Streamlit application.
    """
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.85));
                border: 1px solid rgba(0, 240, 255, 0.35); border-radius: 12px; padding: 20px 24px; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <span style="background: rgba(6, 182, 212, 0.2); color: #22d3ee; border: 1px solid #06b6d4;
                             font-size: 0.75rem; font-weight: 600; padding: 3px 10px; border-radius: 9999px;">
                    ISO 26262 ASIL-D & Euro NCAP 2026
                </span>
                <span style="background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #10b981;
                             font-size: 0.75rem; font-weight: 600; padding: 3px 10px; border-radius: 9999px; margin-left: 6px;">
                    Fixed-Priority Preemptive Scheduling (FPPS)
                </span>
                <h2 style="color: #ffffff; margin: 8px 0 4px 0; font-size: 22px;">
                    ⏱️ Embedded Driver Monitoring System (DMS) — Real-Time Priority Scheduler
                </h2>
                <p style="color: #94a3b8; margin: 0; font-size: 13px;">
                    Analytical Response Time Analysis (RTA), RTOS priority quantization curves, and discrete-event CPU scheduling for 8 in-cabin perception tasks.
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # CONTROLS ROW
    # --------------------------------------------------------------------------
    c_ctrl1, c_ctrl2, c_ctrl3, c_ctrl4 = st.columns([1.2, 1.2, 1.2, 1.0])

    with c_ctrl1:
        camera_fps = st.selectbox(
            "📹 In-Cabin Camera Frame Rate",
            [30.0, 15.0, 60.0, 90.0],
            index=0,
            key=f"{key_prefix}_fps",
            help="30 FPS is standard for Euro NCAP 2026. 60 FPS is used for high-speed IR eye tracking."
        )

    with c_ctrl2:
        strategy_choice = st.selectbox(
            "📐 Priority Scheduling Strategy",
            [
                "Deadline Monotonic (DMS)",
                "Rate Monotonic (RMS)",
                "Adaptive TOPSIS (Criticality + Urgency)",
                "Logarithmic Priority Distribution",
                "Uniform Priority Distribution",
                "Arithmetic Priority Distribution",
                "Geometric Priority Distribution"
            ],
            index=0,
            key=f"{key_prefix}_strategy",
            help="DMS: Shorter deadline -> higher priority (optimal for D <= T). RMS: Shorter period -> higher priority. TOPSIS: balances ASIL criticality & slack."
        )

    with c_ctrl3:
        p_max_preset = st.selectbox(
            "🎛️ RTOS Priority Register (P_max)",
            [8, 16, 32, 64, 256],
            index=0,
            key=f"{key_prefix}_pmax",
            help="8 levels: AUTOSAR OS / OSEK-VDX standard. 16-32 levels: ARM Cortex-R5/M7. 256 levels: 8-bit RTOS registers."
        )

    with c_ctrl4:
        load_scale = st.slider(
            "⚡ Processor Load Factor",
            min_value=0.5,
            max_value=1.5,
            value=1.0,
            step=0.05,
            key=f"{key_prefix}_load",
            help="Scales task WCET to simulate slower/faster automotive microcontrollers or algorithmic load."
        )

    # Secondary collapsible settings
    with st.expander("🛠️ Advanced Scheduling Parameters (Jitter, Logarithmic Beta, Simulation Horizon)"):
        col_adv1, col_adv2, col_adv3, col_adv4 = st.columns(4)
        with col_adv1:
            log_beta = st.slider("Logarithmic Beta (β)", 1.0, 30.0, 9.0, 1.0, key=f"{key_prefix}_beta")
        with col_adv2:
            jitter_pct = st.slider("Execution Jitter (%)", 0, 30, 0, 5, key=f"{key_prefix}_jitter")
        with col_adv3:
            sim_horizon_ms = st.slider("Simulation Horizon (ms)", 100.0, 1000.0, 300.0, 50.0, key=f"{key_prefix}_horizon")
        with col_adv4:
            enforce_unique_p = st.checkbox("Enforce Unique Priorities", value=False, key=f"{key_prefix}_uniq",
                                           help="When unchecked, tasks may share priority levels if P_max < N_tasks.")

    # --------------------------------------------------------------------------
    # TASK MODEL GENERATION & PRIORITY ASSIGNMENT
    # --------------------------------------------------------------------------
    task_set = create_dms_task_set(camera_fps=camera_fps, load_multiplier=load_scale)

    assigner = PriorityAssigner(
        p_min=1,
        p_max=p_max_preset,
        logarithmic_beta=log_beta,
        geometric_ratio=1.618,
        arithmetic_a0=1.0,
        arithmetic_d=0.5
    )

    strat_map = {
        "Deadline Monotonic (DMS)": PriorityAssigner.STRATEGY_DMS,
        "Rate Monotonic (RMS)": PriorityAssigner.STRATEGY_RMS,
        "Adaptive TOPSIS (Criticality + Urgency)": PriorityAssigner.STRATEGY_TOPSIS,
        "Logarithmic Priority Distribution": PriorityAssigner.STRATEGY_LOGARITHMIC,
        "Uniform Priority Distribution": PriorityAssigner.STRATEGY_UNIFORM,
        "Arithmetic Priority Distribution": PriorityAssigner.STRATEGY_ARITHMETIC,
        "Geometric Priority Distribution": PriorityAssigner.STRATEGY_GEOMETRIC
    }
    internal_strat = strat_map[strategy_choice]
    assigned_set = assigner.assign_priorities(task_set, internal_strat, enforce_unique=enforce_unique_p)

    # --------------------------------------------------------------------------
    # EXACT RESPONSE TIME ANALYSIS (RTA)
    # --------------------------------------------------------------------------
    analyzer = RTAAnalyzer()
    rta_results = analyzer.analyze(assigned_set)
    is_schedulable = rta_results["is_schedulable"]
    total_u = task_set.total_utilization * 100.0

    # --------------------------------------------------------------------------
    # TOP STATUS METRIC CARDS
    # --------------------------------------------------------------------------
    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)

    with col_m1:
        st.metric(
            "Overall Schedulability",
            "✅ PASS (100%)" if is_schedulable else "❌ DEADLINE MISS",
            delta=f"{rta_results['schedulable_tasks_count']}/{len(task_set)} Tasks Met"
        )
    with col_m2:
        u_status = "Safe (<95%)" if total_u < 95.0 else ("Critical (>100%)" if total_u > 100.0 else "High Load")
        st.metric("Total CPU Utilization", f"{total_u:.1f}%", delta=u_status)
    with col_m3:
        st.metric("Camera Stream Rate", f"{camera_fps:.0f} FPS", delta=f"T_frame={1000/camera_fps:.1f}ms")
    with col_m4:
        st.metric("Priority Strategy", strategy_choice.split()[0], delta=f"P in [1, {p_max_preset}]")
    with col_m5:
        # CAN Alert WCRT
        can_res = next((r for r in rta_results["per_task_results"] if r["name"] == "T7_CAN_ALERT"), None)
        can_wcrt = can_res["wcrt_ms"] if can_res else 0.0
        st.metric("CAN Alert WCRT (ASIL-D)", f"{can_wcrt:.2f} ms", delta="Deadline: 15.0 ms" if can_wcrt <= 15.0 else "MISSED")

    st.markdown("---")

    # --------------------------------------------------------------------------
    # SUB-TABS FOR DEEP INSPECTION
    # --------------------------------------------------------------------------
    tab_tasks, tab_gantt, tab_multi, tab_curves, tab_safety = st.tabs([
        "📋 Task Specifications & RTA Analysis",
        "⚡ Uniprocessor FPPS Timeline",
        "🖥️ Multiprocessor & UMA/NUMA Allocation",
        "📊 Priority Distribution Curves",
        "🛡️ ISO 26262 ASIL & Euro NCAP Compliance"
    ])

    # --------------------------------------------------------------------------
    # SUB-TAB 1: TASK SPECS & RTA
    # --------------------------------------------------------------------------
    with tab_tasks:
        st.subheader("1. Driver Monitoring Pipeline Tasks & Exact Response Time Analysis")

        rows = []
        for r in rta_results["per_task_results"]:
            t_obj = next(t for t in assigned_set.tasks if t.task_id == r["task_id"])
            rows.append({
                "ID": t_obj.task_id,
                "Symbol": t_obj.name,
                "Modality": t_obj.modality,
                "Period T (ms)": t_obj.period,
                "Deadline D (ms)": t_obj.deadline,
                "WCET C (ms)": t_obj.wcet,
                "Util (%)": round(t_obj.utilization * 100, 1),
                "ASIL": t_obj.criticality,
                "Assigned Priority P": t_obj.priority,
                "WCRT R (ms)": round(r["wcrt_ms"], 2),
                "Margin (D - R)": round(r["margin_ms"], 2),
                "Status": "✅ MET" if r["schedulable"] else "❌ MISSED"
            })
        df_tasks = pd.DataFrame(rows)

        col_tbl, col_chart = st.columns([1.5, 1.0])
        with col_tbl:
            st.dataframe(df_tasks, use_container_width=True, height=330)

        with col_chart:
            # WCRT vs Deadline comparison bar chart
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=df_tasks["Symbol"],
                y=df_tasks["WCRT R (ms)"],
                name="Worst-Case Response Time (R_i)",
                marker_color="#00f0ff"
            ))
            fig_bar.add_trace(go.Bar(
                x=df_tasks["Symbol"],
                y=df_tasks["Deadline D (ms)"],
                name="Relative Deadline (D_i)",
                marker_color="#ff0055",
                opacity=0.65
            ))
            fig_bar.update_layout(
                title="WCRT vs. Relative Deadline per Task",
                barmode="group",
                xaxis_title="Task",
                yaxis_title="Time (ms)",
                template="plotly_dark",
                height=330,
                margin=dict(l=20, r=20, t=35, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.info(r"""
        **Mathematical Formulation (Exact RTA under FPPS)**:
        Worst-Case Response Time $R_i$ is computed iteratively until convergence:
        $$R_i^{(k+1)} = C_i + \sum_{j \in hp(i)} \left\lceil \frac{R_i^{(k)}}{T_j} \right\rceil C_j$$
        where $hp(i)$ is the set of tasks with strictly higher priority than $\tau_i$. Under **Deadline Monotonic Scheduling (DMS)**, tasks with shorter deadlines receive higher priority ($D_i < D_j \implies P_i \ge P_j$).
        """)

    # --------------------------------------------------------------------------
    # SUB-TAB 2: GANTT TIMELINE SIMULATION
    # --------------------------------------------------------------------------
    with tab_gantt:
        st.subheader("2. Discrete-Event CPU Execution & Preemption Timeline")
        st.caption(f"Simulating Fixed-Priority Preemptive Scheduling over {sim_horizon_ms:.0f} ms with context switch penalty = 5 µs.")

        sim = DiscreteEventSimulator(context_switch_penalty_ms=0.005)
        sim_res = sim.run(
            assigned_set,
            duration_ms=sim_horizon_ms,
            use_wcet=True,
            jitter_factor=jitter_pct / 100.0
        )

        exec_trace = sim_res.get("execution_trace", [])
        preempts = sim_res.get("preemption_events", [])

        if exec_trace:
            df_trace = pd.DataFrame(exec_trace)
            task_names_order = [t.name for t in assigned_set.tasks]

            fig_gantt = go.Figure()
            for _, row in df_trace.iterrows():
                tname = row["task_name"]
                tcolor = TASK_COLORS.get(tname, "#3b82f6")
                fig_gantt.add_trace(go.Bar(
                    base=row["start"],
                    x=[row["duration"]],
                    y=[tname],
                    orientation="h",
                    name=tname,
                    marker=dict(color=tcolor, line=dict(color="rgba(255,255,255,0.2)", width=1)),
                    hovertext=f"<b>Task:</b> {tname}<br><b>Job #{row['job_id']}</b><br><b>Start:</b> {row['start']:.2f} ms<br><b>Finish:</b> {row['finish']:.2f} ms<br><b>Duration:</b> {row['duration']:.2f} ms<br><b>Priority:</b> {row['priority']}",
                    hoverinfo="text",
                    showlegend=False
                ))

            for p in preempts:
                if p["time"] <= sim_horizon_ms:
                    fig_gantt.add_vline(
                        x=p["time"],
                        line_width=1.5,
                        line_dash="dot",
                        line_color="#ef4444"
                    )

            fig_gantt.update_layout(
                title=f"CPU Execution Timeline: {strategy_choice} (Horizon: 0-{sim_horizon_ms:.0f} ms)",
                xaxis_title="Simulation Time (ms)",
                yaxis_title="DMS Task",
                template="plotly_dark",
                height=420,
                xaxis=dict(range=[0, sim_horizon_ms], dtick=max(20.0, sim_horizon_ms / 15.0)),
                yaxis=dict(categoryorder="array", categoryarray=task_names_order[::-1]),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_gantt, use_container_width=True)
            if preempts:
                st.caption("🔴 Red dotted vertical lines indicate exact preemption moments where higher-priority tasks interrupted running jobs.")

            col_sim1, col_sim2, col_sim3, col_sim4 = st.columns(4)
            with col_sim1:
                st.metric("Total Jobs Executed", f"{sim_res.get('completed_jobs', 0)} jobs")
            with col_sim2:
                st.metric("Preemption Events", f"{sim_res.get('preemption_count', 0)} preemptions")
            with col_sim3:
                st.metric("Context Switches", f"{sim_res.get('context_switch_count', 0)} switches")
            with col_sim4:
                st.metric("Deadline Misses in Run", f"{sim_res.get('deadline_misses', 0)} misses")
        else:
            st.warning("No jobs were scheduled in the simulation interval.")

    # --------------------------------------------------------------------------
    # SUB-TAB 3: MULTIPROCESSOR & UMA/NUMA ALLOCATION
    # --------------------------------------------------------------------------
    with tab_multi:
        st.subheader("3. Multiprocessor Systems: Bin-Packing Task Allocation & UMA/NUMA Architectures")
        st.markdown(r"""
        Evaluates **Partitioned Multiprocessor Scheduling** across heterogeneous embedded cores (e.g. ARM Cortex-A78AE clusters).
        Allocates the 8 DMS tasks across processing cores using bin-packing heuristics while modeling **UMA vs. NUMA memory interconnect penalties**.
        """)

        # Allocation Controls Row
        c_mp1, c_mp2, c_mp3, c_mp4 = st.columns([1.2, 1.2, 1.0, 1.0])

        with c_mp1:
            arch_choice = st.selectbox(
                "🏗️ Multiprocessor Memory Architecture",
                [MultiprocessorAllocator.ARCH_UMA, MultiprocessorAllocator.ARCH_NUMA],
                index=0,
                key=f"{key_prefix}_arch",
                help="UMA: Uniform Memory Access (Shared SMP bus). NUMA: Non-Uniform Memory Access (Multi-cluster with remote access latency penalties)."
            )

        with c_mp2:
            bp_algo = st.selectbox(
                "📦 Task Allocation Algorithm (Bin-Packing)",
                MultiprocessorAllocator.ALL_ALGORITHMS,
                index=0,
                key=f"{key_prefix}_bpalgo",
                help="FFD/BFD/WFD: Sort tasks by decreasing utilization (C_i / T_i) before packing. WFD maximizes load balancing."
            )

        with c_mp3:
            num_cores_val = st.selectbox(
                "🖥️ Processor Cores (M)",
                [2, 4, 8, 16],
                index=1,
                key=f"{key_prefix}_cores",
                help="Number of processing cores. 4 cores is typical for automotive DMS ECUs."
            )

        with c_mp4:
            core_bound = st.slider(
                "🎯 Core Utilization Bound (U_bound)",
                min_value=0.50,
                max_value=1.00,
                value=0.90,
                step=0.05,
                key=f"{key_prefix}_ubound",
                help="1.0 for EDF/Harmonic; 0.693 for Liu-Layland RM; 0.90 for automotive safety headroom."
            )

        # NUMA-specific sub-controls if NUMA is chosen
        if arch_choice == MultiprocessorAllocator.ARCH_NUMA:
            col_numa1, col_numa2 = st.columns(2)
            with col_numa1:
                remote_penalty = st.slider(
                    "⚡ NUMA Remote Memory Access Latency Penalty",
                    min_value=1.2,
                    max_value=3.5,
                    value=1.8,
                    step=0.1,
                    key=f"{key_prefix}_npenalty",
                    help="Latency multiplier when a task running on Node 1 accesses camera frame buffers physically stored on Node 0."
                )
            with col_numa2:
                cores_per_cluster = st.selectbox(
                    "🗂️ Cores per NUMA Cluster Node",
                    [2, 4],
                    index=0,
                    key=f"{key_prefix}_cluster",
                    help="Defines the NUMA node partition boundary (e.g. 2 cores per cluster)."
                )
        else:
            remote_penalty = 1.0
            cores_per_cluster = 2

        # Run Multiprocessor Allocation
        mp_allocator = MultiprocessorAllocator(
            num_cores=num_cores_val,
            cores_per_numa_node=cores_per_cluster,
            numa_remote_penalty=remote_penalty,
            core_capacity_bound=core_bound
        )

        alloc_result = mp_allocator.allocate(
            task_set,
            algorithm=bp_algo,
            architecture=arch_choice,
            primary_data_node=0
        )

        # Multi-Core Metrics Row
        col_m1, col_m2, col_m3, col_m4, col_m5, col_m6 = st.columns(6)
        with col_m1:
            st.metric(
                "Allocation Feasibility",
                "✅ 100% Schedulable" if alloc_result.is_feasible else f"❌ {len(alloc_result.unallocated_tasks)} Unmapped",
                delta=f"{len(task_set) - len(alloc_result.unallocated_tasks)}/{len(task_set)} Tasks Mapped"
            )
        with col_m2:
            st.metric("Total System Util", f"{alloc_result.total_utilization * 100:.1f}%", delta=f"Cap: {num_cores_val * core_bound * 100:.0f}%")
        with col_m3:
            st.metric("Peak Core Utilization", f"{alloc_result.max_utilization * 100:.1f}%", delta=f"Limit: {core_bound*100:.0f}%")
        with col_m4:
            st.metric("Average Core Load", f"{alloc_result.mean_utilization * 100:.1f}%", delta=f"{num_cores_val} Cores Active")
        with col_m5:
            st.metric("Load Balance Std Dev (σ)", f"{alloc_result.std_utilization * 100:.2f}%", delta="Lower is more balanced")
        with col_m6:
            st.metric("Jain's Fairness Index", f"{alloc_result.jains_fairness_index:.3f}", delta="1.0 = Perfect Balance")

        st.markdown("---")

        tab_mp_vis, tab_mp_bp, tab_mp_arch, tab_mp_lb, tab_mp_sched, tab_mp_res = st.tabs([
            "📊 Processor Utilization & Gantt",
            "📦 Task Allocation (Bin-Packing)",
            "🏛️ UMA and NUMA Architectures",
            "⚖️ Load Balancing & Fairness",
            "⏱️ Scheduling Algorithms & Dhall's Effect",
            "🗂️ Task Distribution & Resources"
        ])

        # ----------------------------------------------------------------------
        # SUB-SECTION 1: PROCESSOR UTILIZATION & MULTI-CORE GANTT
        # ----------------------------------------------------------------------
        with tab_mp_vis:
            st.markdown("#### 📊 Processor Utilization Analysis & Concurrent Multi-Core Gantt")
            col_v1, col_v2 = st.columns([1.2, 1.5])

            with col_v1:
                st.markdown("##### Per-Core Task Distribution & Utilization")
                stacked_rows = []
                for core in alloc_result.cores:
                    for t in core.allocated_tasks:
                        stacked_rows.append({
                            "Core": f"Core #{core.core_id} (Node {core.node_id})" if arch_choice == MultiprocessorAllocator.ARCH_NUMA else f"Core #{core.core_id}",
                            "Task": t.name,
                            "Utilization (%)": round(t.utilization * 100, 2),
                            "WCET (ms)": t.wcet,
                            "Period (ms)": t.period
                        })

                if stacked_rows:
                    df_stacked = pd.DataFrame(stacked_rows)
                    fig_util = px.bar(
                        df_stacked,
                        x="Core",
                        y="Utilization (%)",
                        color="Task",
                        color_discrete_map=TASK_COLORS,
                        title=f"Core Utilization Breakdown ({bp_algo})",
                        hover_data=["WCET (ms)", "Period (ms)"]
                    )
                    fig_util.add_hline(
                        y=core_bound * 100,
                        line_dash="dash",
                        line_color="#ef4444",
                        annotation_text=f"Capacity Ceiling ({core_bound*100:.0f}%)",
                        annotation_position="top right"
                    )
                    fig_util.update_layout(
                        template="plotly_dark",
                        height=360,
                        margin=dict(l=20, r=20, t=35, b=20),
                        yaxis=dict(range=[0, max(110.0, alloc_result.max_utilization * 115)])
                    )
                    st.plotly_chart(fig_util, use_container_width=True)
                else:
                    st.warning("No tasks were allocated to any core.")

            with col_v2:
                st.markdown("##### Concurrent Multiprocessor CPU Gantt Timeline")
                multi_trace = mp_allocator.simulate_multiprocessor_trace(alloc_result, duration_ms=250.0)

                if multi_trace:
                    df_multi = pd.DataFrame(multi_trace)
                    fig_mgantt = go.Figure()

                    for _, row in df_multi.iterrows():
                        tname = row["task_name"]
                        tcolor = TASK_COLORS.get(tname, "#3b82f6")
                        fig_mgantt.add_trace(go.Bar(
                            base=row["start"],
                            x=[row["duration"]],
                            y=[row["core_label"]],
                            orientation="h",
                            name=tname,
                            marker=dict(color=tcolor, line=dict(color="rgba(255,255,255,0.2)", width=1)),
                            hovertext=f"<b>Task:</b> {tname}<br><b>Core:</b> {row['core_label']}<br><b>Job #{row['job_id']}</b><br><b>Start:</b> {row['start']:.2f}ms<br><b>Finish:</b> {row['finish']:.2f}ms<br><b>Priority:</b> {row['priority']}",
                            hoverinfo="text",
                            showlegend=False
                        ))

                    fig_mgantt.update_layout(
                        title=f"Parallel Multi-Core Execution: {bp_algo} on {num_cores_val} Cores (0-250 ms)",
                        xaxis_title="Time (ms)",
                        yaxis_title="Processor Core",
                        template="plotly_dark",
                        height=360,
                        margin=dict(l=20, r=20, t=35, b=20),
                        xaxis=dict(range=[0, 250], dtick=25)
                    )
                    st.plotly_chart(fig_mgantt, use_container_width=True)
                else:
                    st.info("No execution trace available for this configuration.")

            st.info(r"""
            **Processor Utilization & Schedulability Bounds**:
            - **Total System Utilization**: $U_{\text{sys}} = \sum_{i=1}^n \frac{C_i}{T_i} \le M \cdot U_{\text{bound}}$ (Multiprocessor Capacity: $M$).
            - **Per-Core Utilization**: $U_k = \sum_{\tau_i \in \Gamma_k} \frac{C_i}{T_i} \le U_{\text{bound}}$.
            - **Liu & Layland Partitioned RM Bound**: $U_k \le n_k \cdot (2^{1/n_k} - 1)$ where $n_k$ is the number of tasks pinned to Core $k$.
            - **Bini & Buttazzo Hyperbolic Bound**: $\prod_{\tau_i \in \Gamma_k} (U_i + 1) \le 2$.
            """)

        # ----------------------------------------------------------------------
        # SUB-SECTION 2: TASK ALLOCATION ALGORITHMS & BIN-PACKING LAB
        # ----------------------------------------------------------------------
        with tab_mp_bp:
            st.markdown("#### 📦 Task Allocation Algorithms: Bin-Packing Partitioning")
            st.markdown("""
            Partitioned multiprocessor scheduling maps tasks to processing cores using bin-packing heuristics.
            Each processing core acts as a **bin of capacity** $U_{\\text{bound}}$, and each task acts as an **item of size** $U_i = C_i / T_i$.
            """)

            all_comp_table = mp_allocator.compare_all_algorithms(task_set, architecture=arch_choice)
            df_all_comp = pd.DataFrame(all_comp_table)
            st.dataframe(df_all_comp, use_container_width=True)

            col_bp1, col_bp2 = st.columns(2)
            with col_bp1:
                with st.container(border=True):
                    st.markdown("##### 🔍 Decreasing Order Heuristics (FFD, BFD, WFD)")
                    st.write("""
                    - **First-Fit Decreasing (FFD)**: Sorts tasks by $U_i \\downarrow$, places in the first core with sufficient capacity. Approximates optimal packing with factor $\\le \\frac{11}{9} \\text{OPT} + \\frac{6}{9}$.
                    - **Best-Fit Decreasing (BFD)**: Places task in the core that leaves the **minimum remaining capacity**. Consolidates workload onto fewer cores, allowing idle cores to be power-gated.
                    - **Worst-Fit Decreasing (WFD)**: Places task in the core with the **maximum remaining capacity**. Maximizes load balancing across cores, minimizing thermal hotspots.
                    """)
            with col_bp2:
                with st.container(border=True):
                    st.markdown("##### 🛡️ Automotive & Real-Time Specialized Allocation")
                    st.write("""
                    - **Rate-Monotonic FFD (RM-FFD)**: Checks the exact Liu & Layland RM utilization bound $k(2^{1/k}-1)$ for each candidate core before assigning.
                    - **Criticality-Aware Partitioning (ASIL Isolation)**: Isolates safety-critical tasks (ASIL-D / ASIL-C) onto certified safety cores, isolating them from QM / ASIL-A vision tasks (ISO 26262 Part 9 Freedom From Interference).
                    - **Next-Fit (NF)**: Starts scanning from the last allocated core, cycling forward. Low algorithmic complexity $O(n)$, but prone to early fragmentation.
                    """)

        # ----------------------------------------------------------------------
        # SUB-SECTION 3: UMA AND NUMA ARCHITECTURES
        # ----------------------------------------------------------------------
        with tab_mp_arch:
            st.markdown("#### 🏛️ Multiprocessor Memory Architectures: UMA vs. NUMA")
            st.markdown("""
            Modern automotive System-on-Chips (SoCs) such as NVIDIA DRIVE Orin and Renesas R-Car V4H rely on heterogeneous multi-cluster architectures.
            The memory architecture fundamentally governs data sharing latency between camera capture, face detection, and driver warning tasks.
            """)

            col_a1, col_a2 = st.columns(2)
            with col_a1:
                with st.container(border=True):
                    st.markdown("##### 🟢 UMA — Uniform Memory Access (SMP)")
                    st.write("""
                    - **Topology**: All $M$ cores share a single centralized physical memory controller via a high-speed system crossbar.
                    - **Access Latency**: Identical memory latency ($T_{\\text{mem}} \\approx 12.5\\text{ ns}$) regardless of which core executes the task.
                    - **Automotive Challenge**: As camera frame rate increases (30 → 60 → 90 FPS), multi-sensor RAW frame buffer streaming saturates the shared crossbar bus, causing bus contention and jitter.
                    """)
            with col_a2:
                with st.container(border=True):
                    st.markdown("##### 🟣 NUMA — Non-Uniform Memory Access (Multi-Cluster)")
                    st.write("""
                    - **Topology**: Cores are partitioned into compute clusters / NUMA nodes, each with a localized RAM controller.
                    - **Local Access**: Cores accessing local cluster memory experience low latency (~12.5 ns).
                    - **Remote Access Penalty**: Accessing remote cluster memory across the coherent interconnect incurs a latency penalty $\\delta_{\\text{NUMA}} \\in [1.5\\times, 3.5\\times]$.
                    - **NUMA-Aware Allocation**: Grouping dependent vision tasks ($T_1, T_2, T_3$) on Node 0 eliminates remote interconnect stalls!
                    """)

            st.markdown("##### 🌐 Inter-Core Memory Latency Matrix (NUMA vs. UMA)")
            topo = mp_allocator.get_numa_topology_matrix()
            df_topo = pd.DataFrame(topo["latency_table"])
            st.dataframe(df_topo, use_container_width=True)

            if arch_choice == MultiprocessorAllocator.ARCH_NUMA:
                st.warning(f"⚡ Current Configuration: **NUMA with {remote_penalty:.1f}x Remote Access Penalty**. Total memory stall overhead across all cores: **{alloc_result.memory_penalty_overhead_ms:.2f} ms**.")
            else:
                st.success("⚡ Current Configuration: **UMA (Uniform Memory Access)**. Memory stall overhead across all cores: **0.00 ms**.")

        # ----------------------------------------------------------------------
        # SUB-SECTION 4: LOAD BALANCING & FAIRNESS
        # ----------------------------------------------------------------------
        with tab_mp_lb:
            st.markdown("#### ⚖️ Load Balancing & Fairness across Processing Cores")
            col_lb1, col_lb2 = st.columns([1, 1.2])

            with col_lb1:
                st.markdown("##### Load Balancing Mathematical Metrics")
                st.markdown(r"""
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
                st.markdown("##### Current Load Balancing State")
                st.metric("Jain's Fairness Index", f"{alloc_result.jains_fairness_index:.3f}", delta="Target: > 0.90")
                st.metric("Peak-to-Average Ratio", f"{alloc_result.peak_to_average_ratio:.2f}", delta="Target: ~ 1.0")
                st.metric("Utilization Std Dev (σ)", f"{alloc_result.std_utilization * 100:.2f}%", delta="Lower is more balanced")

                st.markdown("""
                > **Trade-Off Insight**:
                > - **WFD (Worst-Fit Decreasing)** optimizes for **thermal balance**, lowering the risk of thermal throttling on automotive ECUs.
                > - **BFD (Best-Fit Decreasing)** optimizes for **energy conservation**, packing tasks onto fewer cores so unused cores can enter sleep modes.
                """)

        # ----------------------------------------------------------------------
        # SUB-SECTION 5: SCHEDULING ALGORITHMS & DHALL'S EFFECT
        # ----------------------------------------------------------------------
        with tab_mp_sched:
            st.markdown("#### ⏱️ Scheduling Algorithms: Partitioned vs. Global Scheduling")
            st.markdown("""
            A fundamental theoretical question in real-time multiprocessor systems is whether to use **Partitioned Scheduling** or **Global Scheduling**.
            """)

            col_sc1, col_sc2 = st.columns(2)
            with col_sc1:
                with st.container(border=True):
                    st.markdown("##### 📌 Partitioned Scheduling (P-FPPS, P-DMS, P-RMS, P-EDF)")
                    st.write("""
                    - Tasks are statically pinned to dedicated cores during system configuration.
                    - Each core executes an independent uniprocessor scheduler (e.g. Deadline Monotonic).
                    - **Advantages**: Preserves cache locality, eliminates task migration costs, allows exact uniprocessor RTA analysis.
                    - **Limitation**: Bin-packing NP-hardness means some core capacity may remain unutilized.
                    """)
            with col_sc2:
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
                dhall_eps = st.slider("Epsilon Parameter (ε)", min_value=0.01, max_value=0.20, value=0.05, step=0.01, key=f"{key_prefix}_dhall_eps")
            with col_dh2:
                dhall_data = MultiprocessorAllocator.get_dhall_effect_scenario(num_cores=num_cores_val, epsilon=dhall_eps)
                st.caption(f"Total Utilization: **{dhall_data['total_utilization']:.3f}** / Capacity: **{dhall_data['system_capacity']:.0f}** ({dhall_data['capacity_percentage']:.1f}% load)")

            st.dataframe(pd.DataFrame(dhall_data["tasks"]), use_container_width=True)

            col_res_g, col_res_p = st.columns(2)
            with col_res_g:
                st.error(f"**Global Scheduling (G-EDF / G-RM)**:\n{dhall_data['global_sched_verdict']}\n\n{dhall_data['global_explanation']}")
            with col_res_p:
                st.success(f"**Partitioned Scheduling (P-FPPS / P-DMS)**:\n{dhall_data['partitioned_sched_verdict']}\n\n{dhall_data['partitioned_explanation']}")

        # ----------------------------------------------------------------------
        # SUB-SECTION 6: TASK DISTRIBUTION & RESOURCE ALLOCATION
        # ----------------------------------------------------------------------
        with tab_mp_res:
            st.markdown("#### 🗂️ Task Distribution and Hardware Resource Allocation Matrix")
            st.markdown("""
            In modern automotive AUTOSAR Adaptive and RTOS platforms, task distribution involves configuring **CPU core affinity bitmasks**,
            reserving **Memory Bandwidth quotas (GB/s)**, and configuring **Last-Level Cache (LLC / L3) Way-Partitioning** (e.g. ARM MPAM / Intel CAT) to guarantee Freedom from Interference.
            """)

            res_matrix = mp_allocator.get_resource_allocation_matrix(alloc_result, total_bandwidth_gbps=68.4, total_llc_ways=16)
            df_res_matrix = pd.DataFrame(res_matrix)
            st.dataframe(df_res_matrix, use_container_width=True)

            st.markdown(r"""
            ##### 🛡️ Automotive CPU Affinity & Cache Partitioning Principles
            - **CPU Affinity Pinning (`sched_setaffinity`)**: Bitmasks ensure tasks cannot arbitrarily migrate across cores, preserving hardware cache warmth and preventing inter-cluster bus thrashing.
            - **Memory Bandwidth Reservation**: Vision perception tasks ($T_1, T_2, T_3$) receive dedicated bandwidth quotas to ensure memory-intensive CNN inferences never starve the critical CAN Alert task ($T_7$).
            - **Shared Cache Way-Partitioning (ARM MPAM / Intel CAT)**: 16-way associative L3 cache is statically partitioned among active cores, ensuring non-critical HUD rendering ($T_8$) cannot evict safety-critical arbitration data ($T_6$).
            """)

    # --------------------------------------------------------------------------
    # SUB-TAB 4: PRIORITY CURVES INSPECTOR
    # --------------------------------------------------------------------------
    with tab_curves:
        st.subheader("4. Comparative Priority Distribution Curves across DMS Tasks")
        st.info("Demonstrates how different priority allocation curves distribute the 8 DMS tasks across the available RTOS priority spectrum $[1, P_{\\max}]$.")

        n_tasks = len(assigned_set.tasks)
        curve_data = {}

        # 1. Uniform
        curve_data["Uniform"] = assigner.compute_priority_values(PriorityAssigner.STRATEGY_UNIFORM, n_tasks, enforce_unique=enforce_unique_p)
        # 2. Arithmetic
        curve_data["Arithmetic"] = assigner.compute_priority_values(PriorityAssigner.STRATEGY_ARITHMETIC, n_tasks, enforce_unique=enforce_unique_p)
        # 3. Geometric
        curve_data["Geometric"] = assigner.compute_priority_values(PriorityAssigner.STRATEGY_GEOMETRIC, n_tasks, enforce_unique=enforce_unique_p)
        # 4. Logarithmic
        curve_data["Logarithmic"] = assigner.compute_priority_values(PriorityAssigner.STRATEGY_LOGARITHMIC, n_tasks, enforce_unique=enforce_unique_p)

        # Ranked tasks under DMS (sorted by deadline)
        dms_sorted_tasks = sorted(assigned_set.tasks, key=lambda t: (t.deadline, t.period))
        task_names_ranked = [f"{t.name} (D={t.deadline:.1f}ms)" for t in dms_sorted_tasks]

        fig_curves = go.Figure()
        colors = {"Uniform": "#2563eb", "Arithmetic": "#16a34a", "Geometric": "#d97706", "Logarithmic": "#dc2626"}
        for cname, pvals in curve_data.items():
            fig_curves.add_trace(go.Scatter(
                x=task_names_ranked,
                y=pvals[::-1],  # highest rank gets highest priority
                mode="lines+markers",
                name=cname,
                line=dict(color=colors[cname], width=3),
                marker=dict(size=8)
            ))

        fig_curves.update_layout(
            title=f"Priority Quantization Curves over P_max={p_max_preset} (Ranked by DMS Urgency)",
            xaxis_title="DMS Tasks (Sorted by Tightest Deadline -> Loosest)",
            yaxis_title="Assigned RTOS Priority Level",
            template="plotly_dark",
            height=400,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_curves, use_container_width=True)

        st.markdown(r"""
        **Comparison Analysis**:
        * **Deadline Monotonic Scheduling (DMS)**: Strictly orders priority by relative deadline $D_i$.
        * **Logarithmic Priority**: Sharp separation among the top urgent tasks (`T7_CAN_ALERT`, `T6_CASDS_ARB`), while clustering non-urgent tasks into lower priority buckets to reduce unnecessary preemption interrupts.
        * **Geometric Priority**: Skews heavily towards the bottom priority levels, potentially starving intermediate perception tasks under heavy load.
        """)

    # --------------------------------------------------------------------------
    # SUB-TAB 5: ISO 26262 & EURO NCAP COMPLIANCE
    # --------------------------------------------------------------------------
    with tab_safety:
        st.subheader("5. Automotive Functional Safety (ISO 26262) & Euro NCAP 2026 Standards")

        col_s1, col_s2 = st.columns([1.2, 1.0])
        with col_s1:
            st.markdown(r"""
            ### 🛡️ Real-Time Safety Requirements for In-Cabin DMS:
            1. **CAN Emergency Alert Dispatch (`T7_CAN_ALERT`, ASIL-D)**:
               * Must dispatch audio chime and seat vibration within **<15 ms** of cognitive distraction or eye closure detection.
               * Highest system priority guaranteed under DMS.
            2. **CASDS Risk Arbitrator (`T6_CASDS_ARB`, ASIL-D)**:
               * Dynamically arbitrates SAFE / CAUTION / CRITICAL state from multi-task vision signals.
               * Must complete within frame period ($T \le 33.3\text{ ms}$).
            3. **PERCLOS & Drowsiness Tracker (`T4_PERCLOS`, ASIL-C)**:
               * Evaluates eye aspect ratio over rolling 1.5-second window.
               * ASIL-C rating ensures failure in infotainment cannot compromise fatigue alerts.
            4. **Cockpit HUD Overlay (`T8_HUD_RENDER`, QM)**:
               * Non-critical visual feedback. Gracefully deferred if CPU is congested, preventing safety task starvation.
            """)
        with col_s2:
            st.markdown("### 📊 ASIL Task Breakdown")
            asil_counts = {"ASIL-D": 2, "ASIL-C": 1, "ASIL-B": 4, "QM": 1}
            df_pie = pd.DataFrame(list(asil_counts.items()), columns=["ASIL Level", "Count"])
            fig_pie = px.pie(
                df_pie,
                names="ASIL Level",
                values="Count",
                color="ASIL Level",
                color_discrete_map=ASIL_COLORS,
                title="DMS Tasks by ISO 26262 ASIL Level",
                hole=0.4
            )
            fig_pie.update_layout(template="plotly_dark", height=280, margin=dict(l=10, r=10, t=35, b=10))
            st.plotly_chart(fig_pie, use_container_width=True)
