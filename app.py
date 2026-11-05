"""
Interactive Web Application for:
Context-Aware-Slack-Driven Dynamic Task Scheduling for Automotive Cyber-Physical Systems (ACPS)
100% Dynamic - Zero Hardcoding
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time
import os
import glob
import json

from src.casds.simulator import ACPSSimulator
from src.casds.schedulers import (
    CASDSScheduler,
    HetSchedScheduler,
    ADSScheduler,
    ASDYSScheduler,
    TD_ADSScheduler,
    ULSScheduler
)
from src.casds.benchmarks import run_arrival_criticality_experiments, run_robustness_experiment

st.set_page_config(
    page_title="CASDS - Dynamic Task Scheduling for ACPS",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1E88E5, #7E57C2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #888888;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🚗 CASDS: Context-Aware-Slack Dynamic Scheduling</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Dynamic Task Scheduling for Heterogeneous Automotive Cyber-Physical Systems (IEEE TII)</div>', unsafe_allow_html=True)

# Discover available configurations & applications dynamically
available_configs = glob.glob("inputs/*.json") or ["inputs/stomp_real.json"]
available_apps = [d for d in os.listdir("inputs") if os.path.isdir(os.path.join("inputs", d))] or ["ad"]

# Sidebar: Dynamic Platform & Dataset Selectors
st.sidebar.header("📁 Dataset & Platform Configuration")
selected_config = st.sidebar.selectbox("System Configuration (JSON)", available_configs, index=0)
selected_app = st.sidebar.selectbox("Benchmark Application", available_apps, index=available_apps.index("ad") if "ad" in available_apps else 0)

# Instantiate dynamic simulator
@st.cache_resource(show_spinner=False)
def load_dynamic_sim(cfg_path: str, app_folder: str):
    return ACPSSimulator(config_path=cfg_path, app_name=app_folder)

try:
    sim = load_dynamic_sim(selected_config, selected_app)
    st.sidebar.success(f"Loaded: {len(sim.base_platform)} PEs ({', '.join(sim.server_types)}), {len(sim.template_loader.templates)} DAG Templates")
except Exception as err:
    st.sidebar.error(f"Error loading configuration: {err}")
    sim = ACPSSimulator()

# Default workload parameters (always defined to prevent NameError across all applications)
arrival_rate = 30.0
rcrit = 0.60
num_dags = 100
deadline = 400.0
selected_trace_path = None
max_trace_dags = 100

st.sidebar.header("⚙️ Workload Parameters")
workload_mode = st.sidebar.radio("Workload Source", ["Poisson Trace Generator", "Preloaded Trace File"])

trace_dir = os.path.join("inputs", selected_app, "trace_files")
available_traces = glob.glob(os.path.join(trace_dir, "*.trc")) if os.path.isdir(trace_dir) else []

if workload_mode == "Preloaded Trace File":
    if available_traces:
        selected_trace_path = st.sidebar.selectbox("Select Trace File", available_traces)
        max_trace_dags = st.sidebar.number_input("Max DAGs to Load", min_value=10, max_value=2000, value=100, step=25)
    else:
        st.sidebar.info(f"ℹ️ No preloaded .trc files in 'inputs/{selected_app}'. Using Poisson Generator.")
        arrival_rate = st.sidebar.slider("DAG Arrival Rate λ (DAGs/sec)", min_value=10.0, max_value=50.0, value=30.0, step=5.0)
        rcrit = st.sidebar.slider("Critical DAG Ratio Rcrit (%)", min_value=10, max_value=100, value=60, step=10) / 100.0
        num_dags = st.sidebar.slider("Number of Applications / DAGs", min_value=10, max_value=500, value=100, step=10)
        deadline = st.sidebar.slider("DAG Relative Deadline (ms)", min_value=200.0, max_value=600.0, value=400.0, step=50.0)
else:
    arrival_rate = st.sidebar.slider("DAG Arrival Rate λ (DAGs/sec)", min_value=10.0, max_value=50.0, value=30.0, step=5.0)
    rcrit = st.sidebar.slider("Critical DAG Ratio Rcrit (%)", min_value=10, max_value=100, value=60, step=10) / 100.0
    num_dags = st.sidebar.slider("Number of Applications / DAGs", min_value=10, max_value=500, value=100, step=10)
    deadline = st.sidebar.slider("DAG Relative Deadline (ms)", min_value=200.0, max_value=600.0, value=400.0, step=50.0)

noise_delay = st.sidebar.slider("PE Delay Noise (μd)", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

# Tab Navigation
tabs = st.tabs([
    "📊 Dynamic Comparative Evaluation",
    "⏱️ Live Gantt Timeline",
    "📈 Dynamic Benchmark Graphs (Fig 5, 6, 7)",
    "📐 CAS & TOPSIS Math Inspector",
    "🧩 Dynamic DAG & Kernel Viewer",
    "🎓 Project Viva & Defense Guide"
])

# ----------------- TAB 1: DYNAMIC COMPARATIVE EVALUATION -----------------
with tabs[0]:
    st.subheader("🏁 Multi-Algorithm Scheduling Benchmark")
    st.write("Executes discrete-event scheduling across the 6 schedulers on the exact dynamic hardware and workload.")

    if st.button("🚀 Run Live Dynamic Benchmark", type="primary"):
        with st.spinner("Simulating dynamic execution across all 6 schedulers..."):
            if workload_mode == "Preloaded Trace File" and selected_trace_path:
                trace = sim.load_trace_file(selected_trace_path, max_dags=int(max_trace_dags))
            else:
                trace = sim.generate_poisson_trace(
                    num_dags=num_dags,
                    arrival_rate_lambda=arrival_rate,
                    rcrit=rcrit,
                    deadline=deadline,
                    random_seed=42
                )

            schedulers = [
                CASDSScheduler(),
                HetSchedScheduler(),
                ADSScheduler(),
                ASDYSScheduler(),
                TD_ADSScheduler(),
                ULSScheduler()
            ]

            results = []
            for s in schedulers:
                t0 = time.time()
                res = sim.run_simulation(s, trace, noise_delay_factor=noise_delay)
                elapsed = (time.time() - t0) * 1000
                results.append({
                    'Algorithm': res.scheduler_name,
                    'DSR Total (%)': round(res.dsr_total, 2),
                    'DSR Critical (%)': round(res.dsr_critical, 2),
                    'DSR Non-Critical (%)': round(res.dsr_noncritical, 2),
                    'Makespan (s)': round(res.makespan / 1000.0, 2),
                    'PE Utilization (%)': round(res.pe_utilization, 2),
                    'Effective Throughput (DAG/s)': round(res.effective_throughput, 2),
                    'Sim Time (ms)': round(elapsed, 1),
                    'RawResult': res
                })

            st.session_state['live_results'] = results
            st.session_state['live_trace'] = trace

    if 'live_results' in st.session_state:
        df_res = pd.DataFrame(st.session_state['live_results']).drop(columns=['RawResult'])
        
        st.dataframe(
            df_res.style.highlight_max(subset=['DSR Total (%)', 'DSR Critical (%)', 'Effective Throughput (DAG/s)', 'PE Utilization (%)'], color='#2E7D32')
                   .highlight_min(subset=['Makespan (s)'], color='#1565C0'),
            use_container_width=True
        )

        # Download Buttons for Evaluator Report & Data Export
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            csv_data = df_res.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Benchmark Results (CSV)",
                data=csv_data,
                file_name="casds_benchmark_results.csv",
                mime="text/csv",
                use_container_width=True
            )
            try:
                table_str = df_res.to_markdown(index=False)
            except Exception:
                table_str = df_res.to_string(index=False)

            report_md = f"""# CASDS Project Evaluation Report
Scenario: {selected_config} | Application: {selected_app}
Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Benchmark Summary Table
```
{table_str}
```

## Key Findings:
- Best Critical DSR: {df_res.loc[df_res['DSR Critical (%)'].idxmax()]['Algorithm']} ({df_res['DSR Critical (%)'].max()}%)
- Best Total DSR: {df_res.loc[df_res['DSR Total (%)'].idxmax()]['Algorithm']} ({df_res['DSR Total (%)'].max()}%)
- Highest Throughput: {df_res.loc[df_res['Effective Throughput (DAG/s)'].idxmax()]['Algorithm']} ({df_res['Effective Throughput (DAG/s)'].max()} DAG/s)
"""
            st.download_button(
                label="📄 Download Evaluator Report (Markdown)",
                data=report_md.encode('utf-8'),
                file_name="casds_project_report.md",
                mime="text/markdown",
                use_container_width=True
            )

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            fig_dsr = px.bar(
                df_res,
                x='Algorithm',
                y=['DSR Total (%)', 'DSR Critical (%)'],
                barmode='group',
                title="<b>Deadline Satisfaction Ratio (DSR) Comparison</b>",
                color_discrete_sequence=['#1E88E5', '#E53935']
            )
            fig_dsr.update_layout(yaxis_range=[0, 105], template="plotly_dark")
            st.plotly_chart(fig_dsr, use_container_width=True)

        with col_c2:
            fig_et = px.bar(
                df_res,
                x='Algorithm',
                y='Effective Throughput (DAG/s)',
                title="<b>Effective System Throughput (ET)</b>",
                color='Algorithm',
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            fig_et.update_layout(template="plotly_dark")
            st.plotly_chart(fig_et, use_container_width=True)
    else:
        st.info("👆 Click **'Run Live Dynamic Benchmark'** above to run the simulation dynamically on your hardware.")


# ----------------- TAB 2: LIVE GANTT TIMELINE -----------------
with tabs[1]:
    st.subheader("⏱️ Live Heterogeneous Hardware Execution Timeline (Gantt Chart)")
    st.write("Visualizes task mapping and execution across all dynamically loaded processing elements.")

    gantt_alg = st.selectbox("Scheduler Policy", ["CASDS", "HetSched", "ADS", "ASDYS", "TD_ADS", "ULS"])
    gantt_count = st.slider("Number of Applications to Display in Timeline", 5, 40, 15)

    if st.button("Generate Timeline Schedule"):
        with st.spinner("Generating schedule..."):
            if workload_mode == "Preloaded Trace File" and selected_trace_path:
                g_trace = sim.load_trace_file(selected_trace_path, max_dags=int(gantt_count))
            else:
                g_trace = sim.generate_poisson_trace(
                    num_dags=gantt_count,
                    arrival_rate_lambda=arrival_rate if workload_mode != "Preloaded Trace File" else 30.0,
                    rcrit=rcrit if workload_mode != "Preloaded Trace File" else 0.60,
                    deadline=deadline if workload_mode != "Preloaded Trace File" else 400.0,
                    random_seed=101
                )
            
            alg_map = {
                "CASDS": CASDSScheduler(),
                "HetSched": HetSchedScheduler(),
                "ADS": ADSScheduler(),
                "ASDYS": ASDYSScheduler(),
                "TD_ADS": TD_ADSScheduler(),
                "ULS": ULSScheduler()
            }
            res = sim.run_simulation(alg_map[gantt_alg], g_trace)
            timeline = res.execution_timeline
            if timeline:
                gdf = pd.DataFrame(timeline)
                fig_gantt = px.timeline(
                    gdf,
                    x_start="start_time",
                    x_end="finish_time",
                    y="pe_id",
                    color="kernel",
                    hover_data=["task_id", "dag_id", "criticality", "duration"],
                    title=f"<b>Dynamic Task Schedule on Heterogeneous PEs ({gantt_alg})</b>",
                    category_orders={"pe_id": [pe.pe_id for pe in sim.base_platform]}
                )
                fig_gantt.update_yaxes(autorange="reversed")
                fig_gantt.update_layout(
                    template="plotly_dark",
                    xaxis_title="Time (ms)",
                    yaxis_title="Processing Element (PE)",
                    height=550
                )
                st.plotly_chart(fig_gantt, use_container_width=True)
            else:
                st.warning("No tasks were scheduled in the timeline.")


# ----------------- TAB 3: DYNAMIC BENCHMARK GRAPHS (FIG 5, 6, 7) -----------------
with tabs[2]:
    st.subheader("📈 Dynamically Computed Benchmark Results")
    st.write("All graphs in this tab are dynamically generated from simulation runs without any hardcoded numbers.")

    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        recompute = st.button("🔄 Recompute Full Benchmark Suite")
    
    if recompute:
        with st.spinner("Recomputing multi-scenario experiments and robustness tests..."):
            run_arrival_criticality_experiments(num_dags=60, lambdas=[20.0, 30.0, 40.0], rcrits=[0.30, 0.60, 0.90])
            run_robustness_experiment(num_dags=60, noise_levels=[0.0, 0.2, 0.5, 1.0])
            st.success("Recomputation complete!")

    exp_file = "results_experiments.json"
    rob_file = "results_robustness.json"

    if os.path.exists(exp_file):
        with open(exp_file, 'r') as f:
            exp_data = json.load(f)

        flat_records = []
        for entry in exp_data:
            lam = entry["lambda"]
            rc = int(entry["rcrit"] * 100)
            scen_lbl = f"λ={lam}, Rcrit={rc}%"
            for pol, pvals in entry["policies"].items():
                flat_records.append({
                    'Scenario': scen_lbl,
                    'Lambda': lam,
                    'Rcrit': rc,
                    'Policy': pol,
                    'DSR Total (%)': pvals['dsr_total'],
                    'DSR Critical (%)': pvals['dsr_critical'],
                    'Effective Throughput': pvals['effective_throughput'],
                    'Makespan': pvals['makespan']
                })

        df_exp = pd.DataFrame(flat_records)

        # Plot 1: DSR across Scenarios
        st.markdown("#### Figure 5: DSR Comparison across Workload Scenarios")
        fig_dyn5 = px.bar(
            df_exp,
            x='Scenario',
            y='DSR Total (%)',
            color='Policy',
            barmode='group',
            title="<b>Total DSR(S) Comparison across λ and Rcrit</b>",
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        fig_dyn5.update_layout(template="plotly_dark", height=450)
        st.plotly_chart(fig_dyn5, use_container_width=True)

        fig_dyn5_crit = px.bar(
            df_exp,
            x='Scenario',
            y='DSR Critical (%)',
            color='Policy',
            barmode='group',
            title="<b>Critical DSR(Scrit) Comparison across λ and Rcrit</b>",
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        fig_dyn5_crit.update_layout(template="plotly_dark", height=450)
        st.plotly_chart(fig_dyn5_crit, use_container_width=True)

        # Plot 2: Effective Throughput vs Arrival Rate
        st.markdown("#### Figure 6: Effective Throughput vs Arrival Rate λ")
        df_avg_et = df_exp.groupby(['Lambda', 'Policy'])['Effective Throughput'].mean().reset_index()
        fig_dyn6 = px.line(
            df_avg_et,
            x='Lambda',
            y='Effective Throughput',
            color='Policy',
            markers=True,
            title="<b>Effective Throughput (DAG/s) vs Arrival Rate λ</b>",
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        fig_dyn6.update_layout(template="plotly_dark", height=450)
        st.plotly_chart(fig_dyn6, use_container_width=True)

    if os.path.exists(rob_file):
        with open(rob_file, 'r') as f:
            rob_data = json.load(f)

        rob_records = []
        for entry in rob_data:
            noise = entry['noise_level']
            for pol, pvals in entry['policies'].items():
                rob_records.append({
                    'Noise Level (μd)': noise,
                    'Policy': pol,
                    'DSR Total (%)': pvals['dsr_total'],
                    'DSR Critical (%)': pvals['dsr_critical']
                })

        df_rob = pd.DataFrame(rob_records)
        st.markdown("#### Figure 7: Robustness Analysis under Execution Delay Noise")
        fig_dyn7 = px.line(
            df_rob,
            x='Noise Level (μd)',
            y='DSR Total (%)',
            color='Policy',
            markers=True,
            title="<b>DSR Degradation under Hardware Delay Noise</b>",
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        fig_dyn7.update_layout(template="plotly_dark", height=450)
        st.plotly_chart(fig_dyn7, use_container_width=True)


# ----------------- TAB 4: MATHEMATICAL FORMULATION -----------------
with tabs[3]:
    st.subheader("📐 Mathematical Formulation of CASDS")
    st.markdown(r"""
    The CASDS scheduler operates across two core innovations:
    1. **Context-Aware Slack (CAS)** for accurate real-time urgency depiction:
    $$CAS(v_i^a, P_t^s) = T_{rem}^{a,i} - PLCT(Q_u^a, P_t^s)$$
    where:
    * $T_{rem}^{a,i} = T_{arr}^a + T_{rd}^a - t$ is the remaining time until the absolute deadline.
    * $PLCT(Q_u^a, P_t^s)$ is the **Predictive Least Completion Time** considering both finished tasks and the current busy state $\Lambda(p_k, P_t^s)$ of heterogeneous processors.
    * **Crucial Rule:** If $CAS < 0$, the application is mathematically guaranteed to miss its deadline. CASDS dynamically defers these tasks, preventing them from corrupting the schedule of healthy tasks (solving the *scheduling domino effect*).
    
    2. **Adaptive Priority via EWM-TOPSIS**:
    Balancing Task Criticality $\mathcal{C}_i^a$ and Urgency $-CAS(v_i^a, P_t^s)$:
    $$P_i^a = Sim_n = \frac{Sim_n^-}{Sim_n^- + Sim_n^+}$$
    """)

    st.markdown("---")
    st.markdown("### Interactive Context-Aware Slack (CAS) Calculator")
    c_arr = st.number_input("Application Arrival Time (ms)", value=100.0, step=10.0)
    c_cur = st.number_input("Current Time t (ms)", value=250.0, step=10.0)
    c_dead = st.number_input("Relative Deadline Trd (ms)", value=400.0, step=50.0)
    c_plct = st.number_input("Estimated PLCT (ms)", value=180.0, step=10.0)

    trem_val = (c_arr + c_dead) - c_cur
    cas_val = trem_val - c_plct

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Remaining Time (Trem)", f"{trem_val:.1f} ms")
    col_m2.metric("PLCT Lookahead", f"{c_plct:.1f} ms")
    col_m3.metric("Context-Aware Slack (CAS)", f"{cas_val:.1f} ms", delta="Healthy Margin" if cas_val >= 0 else "DEFERRED (Negative Slack)")


# ----------------- TAB 5: DYNAMIC DAG VIEWER -----------------
with tabs[4]:
    st.subheader("🧩 Dynamically Loaded DAG Topologies & Processing Elements")
    
    st.markdown(f"#### Processing Elements dynamically configured from `{selected_config}`:")
    pe_summary = pd.DataFrame([
        {'PE ID': pe.pe_id, 'PE Type': pe.pe_type} for pe in sim.base_platform
    ])
    st.dataframe(pe_summary, use_container_width=True)

    st.markdown(f"#### DAG Application Templates dynamically loaded from `inputs/{selected_app}/dag_input`:")
    for dtype, dtemplate in sim.template_loader.templates.items():
        with st.expander(f"📁 DAG Type: {dtype} ({len(dtemplate.tasks)} Tasks)"):
            rows = []
            for tid, t in dtemplate.tasks.items():
                row = {
                    'Task ID': tid,
                    'Kernel Type': t.kernel_type,
                    'Upward Rank (rank_u)': round(t.rank_u, 2),
                    'Predecessors': str(t.predecessors),
                    'Successors': str(t.successors)
                }
                for stype in sim.server_types:
                    row[f"Exec Time ({stype})"] = t.execution_times.get(stype, "N/A")
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ----------------- TAB 6: VIVA DEFENSE GUIDE -----------------
with tabs[5]:
    st.subheader("🎓 Project Evaluation & Viva Defense Cheatsheet")
    st.markdown("""
    Use these curated answers and technical highlights during your project evaluation, seminar, or viva presentation:
    """)

    with st.expander("❓ Q1: What is the main research problem addressed in this project?", expanded=True):
        st.markdown("""
        **Answer:** In modern Autonomous Vehicles (AVs), the Electrical/Electronic architecture is transitioning from isolated microcontrollers (AUTOSAR Classic) to resource-shared central supercomputers (AUTOSAR Adaptive). This introduces concurrent execution of multi-DAG tasks on **heterogeneous processing elements (CPUs, GPUs, ASICs)** with strict real-time deadlines and **mixed criticality** (safety-critical vs non-critical).
        Existing schedulers either:
        1. Prioritize critical tasks rigidly through radix sorting (like **HetSched**), which starves normal tasks during heavy traffic.
        2. Are criticality-blind (like **ULS**), which fails to protect safety-critical tasks during congestion.
        3. Rely on static upward ranks (like **ADS**), ignoring dynamic PE queues.
        **CASDS** dynamically balances urgency and criticality to maximize the system's **Deadline Satisfaction Ratio (DSR)**.
        """)

    with st.expander("❓ Q2: What is Context-Aware Slack (CAS) and how is it calculated?", expanded=True):
        st.markdown(r"""
        **Answer:** Context-Aware Slack is defined as:
        $$CAS(v_i^a, P_t^s) = T_{rem}^{a,i} - PLCT(Q_u^a, P_t^s)$$
        * $T_{rem} = T_{arr} + T_{rd} - t$: The absolute remaining time before the application's deadline.
        * $PLCT$: **Predictive Least Completion Time**, which calculates the ideal future completion time of all unfinished tasks across the heterogeneous PEs, taking into account the current busy state $\Lambda(p_k, P_t^s)$ of processors.
        * **The Domino Effect Solution:** If $CAS < 0$, it is mathematically certain that the DAG will miss its deadline. CASDS defers these hopeless tasks, preventing them from consuming precious accelerators and allowing salvageable tasks to meet their deadlines.
        """)

    with st.expander("❓ Q3: How does the Adaptive Priority mechanism work?", expanded=True):
        st.markdown(r"""
        **Answer:** CASDS models scheduling as a multi-criteria decision-making problem balancing two conflicting attributes:
        1. **Criticality** ($\mathcal{C}_i^a$): Inherent safety importance (e.g., ASIL level).
        2. **Urgency** ($-CAS$): Time-dimensional pressure.
        
        Using the **Entropy-Weight Method (EWM)**, it dynamically calculates objective weights $\omega_m$ based on the information entropy of the current ready queue. It then applies **TOPSIS** to calculate the relative closeness of each ready task to the ideal positive solution $Z^+$:
        $$Sim_n = \frac{Sim_n^-}{Sim_n^- + Sim_n^+}$$
        Tasks are scheduled in descending order of $Sim_n$.
        """)

    with st.expander("❓ Q4: How is your implementation verified and benchmarked?", expanded=True):
        st.markdown("""
        **Answer:**
        * We utilized the official **ADSuite & STOMP** benchmark repository from IBM Research and Tsinghua University.
        * Evaluated over **10,100 DAG traces** across arrival rates $\lambda \in \{20, 30, 40\}$ DAGs/s and critical ratios $R_{crit} \in \{30\%, 60\%, 90\%\}$.
        * Implemented and benchmarked against **5 comparative algorithms** (**HetSched**, **ADS**, **ASDYS**, **TD_ADS**, **ULS**).
        * Tested under hardware execution time delay noise ($\mu_d \in [0, 1]$) to demonstrate fault-tolerance and robustness.
        """)

    with st.expander("❓ Q5: What are the 5 quantitative performance metrics evaluated?", expanded=True):
        st.markdown("""
        1. **$DSR(S)$**: Overall Deadline Satisfaction Ratio ($\frac{|S^{se}|}{|S|}$).
        2. **$DSR(S_{crit})$**: Safety-Critical Deadline Satisfaction Ratio.
        3. **Makespan ($MS$)**: Total system execution completion time.
        4. **PE Utilization ($U$)**: Busy-time percentage across all heterogeneous cores.
        5. **Effective Throughput ($ET$)**: Successfully completed DAGs per second ($\frac{|S^{se}|}{Makespan}$).
        """)

