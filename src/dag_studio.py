"""
DAG Studio & Advanced Graph Visualizer for Automotive Real-Time Systems.
Implements:
1. Interactive Visual DAG Network Diagrams (Plotly + Graphviz SVG) with Directed Precedence Arrows
2. Automated Critical Path Analysis & Latency Computation
3. Graph Topology & Structural Metrics (Depth, Max Parallelism, CCR, Speedup Bound)
4. HEFT Upward Rank (rank_u) Priority Heatmaps
5. Multi-DAG Poisson Arrival & Traffic Stream Timeline Visualizer
6. Automotive Autonomous Driving DAGs (ADSuite Types 1-4, SelfPilot) + DMS Perception DAG (ISO 26262 ASIL-D)
7. Custom / Synthetic Layered DAG Generator Sandbox
"""

import math
import random
from typing import Dict, List, Tuple, Any, Optional
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from casds.models.task import Task
from casds.models.dag import DAGApplication
from casds.models.platform import HeterogeneousPlatform
from casds.benchmarks.adsuite_benchmark import (
    create_adsuite_platform, create_adsuite_dag, ADSUITE_TASK_PROFILES
)
from casds.benchmarks.selfpilot_benchmark import (
    create_selfpilot_platform, create_selfpilot_dag, SELFPILOT_TASK_PROFILES
)


# ------------------------------------------------------------------------------
# COLOR PALETTES FOR DAG VISUALIZATION
# ------------------------------------------------------------------------------
KERNEL_COLORS: Dict[str, str] = {
    # ADSuite & General
    "DET": "#06b6d4",   # Cyan
    "TRA": "#8b5cf6",   # Purple
    "LOC": "#10b981",   # Emerald Green
    "FUS": "#f59e0b",   # Amber
    "MiP": "#ec4899",   # Pink
    "MoP": "#6366f1",   # Indigo
    "CON": "#14b8a6",   # Teal
    # SelfPilot
    "COD": "#06b6d4",
    "LOD": "#8b5cf6",
    "MOT": "#f43f5e",   # Rose
    "ODF": "#eab308",   # Yellow
    "LD":  "#10b981",
    # DMS In-Cabin
    "ISP": "#0284c7",   # Blue
    "FACE": "#8b5cf6",  # Violet
    "GAZE": "#3b82f6",  # Electric Blue
    "PERC": "#f59e0b",  # Amber
    "DIST": "#ec4899",  # Pink
    "ARB":  "#10b981",  # Green
    "CAN":  "#ef4444",  # Crimson Red
    "HUD":  "#64748b"   # Slate
}

CRITICALITY_COLORS: Dict[int, str] = {
    2: "#ff0055",  # Critical (Safety Critical ASIL-D)
    1: "#00f0ff"   # Normal (QM / Standard)
}


# ------------------------------------------------------------------------------
# DMS COCKPIT PERCEPTION DAG CREATOR
# ------------------------------------------------------------------------------
def create_dms_dag(
    app_id: int = 0,
    criticality: int = 2,
    arrival_time: float = 0.0,
    relative_deadline: float = 60.0
) -> DAGApplication:
    """
    Creates the Driver Monitoring System (DMS) 8-Task DAG:
    T0: Camera Frame Acquisition & ISP Preprocessing
    T1: Face ROI Localization & Landmark Extraction
    T2: Cockpit 9-Zone Gaze Tracking Head
    T3: PERCLOS Eye Closure & Micro-Sleep Head
    T4: Distraction Activity Classification (10 Classes)
    T5: CASDS Real-Time Risk Arbitrator
    T6: CAN Bus Emergency Alert & Braking Dispatcher
    T7: Cockpit HUD Telemetry & Gaze Visualization
    """
    dms_task_specs = [
        (0, "T0_CAM_ISP", "ISP", {"CPU": 4.8, "GPU": 2.0, "DET_ASIC": 1.2, "TRA_ASIC": float("inf"), "LOC_ASIC": float("inf")}),
        (1, "T1_FACE_LOC", "FACE", {"CPU": 6.2, "GPU": 2.1, "DET_ASIC": 1.5, "TRA_ASIC": float("inf"), "LOC_ASIC": float("inf")}),
        (2, "T2_GAZE_9Z", "GAZE", {"CPU": 7.0, "GPU": 2.5, "DET_ASIC": 1.8, "TRA_ASIC": float("inf"), "LOC_ASIC": float("inf")}),
        (3, "T3_PERCLOS", "PERC", {"CPU": 4.0, "GPU": 1.5, "DET_ASIC": float("inf"), "TRA_ASIC": float("inf"), "LOC_ASIC": float("inf")}),
        (4, "T4_DISTRACT", "DIST", {"CPU": 11.5, "GPU": 3.8, "DET_ASIC": float("inf"), "TRA_ASIC": float("inf"), "LOC_ASIC": float("inf")}),
        (5, "T5_CASDS_ARB", "ARB", {"CPU": 1.4, "GPU": float("inf"), "DET_ASIC": float("inf"), "TRA_ASIC": float("inf"), "LOC_ASIC": float("inf")}),
        (6, "T6_CAN_ALERT", "CAN", {"CPU": 0.8, "GPU": float("inf"), "DET_ASIC": float("inf"), "TRA_ASIC": float("inf"), "LOC_ASIC": float("inf")}),
        (7, "T7_HUD_RENDER", "HUD", {"CPU": 3.2, "GPU": 1.0, "DET_ASIC": float("inf"), "TRA_ASIC": float("inf"), "LOC_ASIC": float("inf")})
    ]

    tasks: Dict[int, Task] = {}
    for tid, name, ktype, profiles in dms_task_specs:
        tasks[tid] = Task(tid, name, ktype, profiles, app_id)

    edges = [
        (0, 1),        # ISP -> Face Loc
        (1, 2),        # Face Loc -> Gaze 9-Zone
        (1, 3),        # Face Loc -> PERCLOS
        (1, 4),        # Face Loc -> Distraction
        (2, 5),        # Gaze -> CASDS Arbiter
        (3, 5),        # PERCLOS -> CASDS Arbiter
        (4, 5),        # Distraction -> CASDS Arbiter
        (5, 6),        # Arbiter -> CAN Alert
        (5, 7)         # Arbiter -> HUD Render
    ]

    edge_costs = {
        (0, 1): 4.0,   # Raw frame to Face detector
        (1, 2): 2.5,   # Face crop to Gaze
        (1, 3): 2.0,   # Eye landmarks to PERCLOS
        (1, 4): 3.0,   # Face ROI to Distraction
        (2, 5): 1.0,   # Gaze vector to Arbiter
        (3, 5): 1.0,   # Fatigue score to Arbiter
        (4, 5): 1.5,   # Distraction class to Arbiter
        (5, 6): 0.5,   # Critical alert to CAN bus
        (5, 7): 1.2    # Telemetry to Cockpit HUD
    }

    return DAGApplication(
        app_id=app_id,
        dag_type="DMS Cockpit (8 tasks)",
        tasks=tasks,
        edges=edges,
        edge_costs=edge_costs,
        criticality=criticality,
        arrival_time=arrival_time,
        relative_deadline=relative_deadline
    )


# ------------------------------------------------------------------------------
# SYNTHETIC LAYERED DAG GENERATOR
# ------------------------------------------------------------------------------
def create_synthetic_dag(
    num_tasks: int = 10,
    num_layers: int = 4,
    edge_density: float = 0.5,
    app_id: int = 99,
    criticality: int = 2,
    relative_deadline: float = 300.0,
    seed: int = 42
) -> DAGApplication:
    """
    Generates a guaranteed Directed Acyclic Graph (DAG) with layer structure:
    Tasks are distributed across sequential stages; edges only flow strictly from layer i to layer j (j > i).
    """
    random.seed(seed)
    np.random.seed(seed)

    num_layers = max(2, min(num_layers, num_tasks - 1))
    tasks: Dict[int, Task] = {}

    # Assign tasks to layers
    layer_nodes: List[List[int]] = [[] for _ in range(num_layers)]
    layer_nodes[0].append(0)  # At least 1 entry node
    layer_nodes[-1].append(num_tasks - 1)  # At least 1 exit node

    # Distribute intermediate tasks
    for tid in range(1, num_tasks - 1):
        target_layer = random.randint(0, num_layers - 1)
        layer_nodes[target_layer].append(tid)

    # Ensure every layer has at least one task
    for l_idx in range(num_layers):
        if not layer_nodes[l_idx]:
            # Borrow a node from a layer with >1 nodes
            for other_l in range(num_layers):
                if len(layer_nodes[other_l]) > 1:
                    borrowed = layer_nodes[other_l].pop()
                    layer_nodes[l_idx].append(borrowed)
                    break

    # Build tasks with realistic heterogeneous profiles
    available_kernels = ["DET", "TRA", "LOC", "FUS", "MiP", "MoP"]
    for l_idx, nodes in enumerate(layer_nodes):
        for tid in nodes:
            k = available_kernels[min(l_idx, len(available_kernels) - 1)]
            name = f"T{tid}_{k}"
            profile = {
                "CPU": round(random.uniform(50.0, 300.0), 1),
                "GPU": round(random.uniform(10.0, 45.0), 1),
                "DET_ASIC": round(random.uniform(5.0, 20.0), 1) if k in ["DET", "TRA"] else float("inf"),
                "TRA_ASIC": round(random.uniform(2.0, 10.0), 1) if k == "TRA" else float("inf"),
                "LOC_ASIC": round(random.uniform(4.0, 15.0), 1) if k == "LOC" else float("inf"),
            }
            tasks[tid] = Task(tid, name, k, profile, app_id)

    # Connect edges forward between layers
    edges: List[Tuple[int, int]] = []
    edge_costs: Dict[Tuple[int, int], float] = {}

    for l_idx in range(num_layers - 1):
        src_nodes = layer_nodes[l_idx]
        dst_nodes = layer_nodes[l_idx + 1]

        # Guarantee at least 1 edge out of every node
        for src in src_nodes:
            dst = random.choice(dst_nodes)
            edges.append((src, dst))
            edge_costs[(src, dst)] = round(random.uniform(5.0, 30.0), 1)

        # Guarantee at least 1 edge into every destination node
        for dst in dst_nodes:
            if not any(e[1] == dst for e in edges):
                src = random.choice(src_nodes)
                edges.append((src, dst))
                edge_costs[(src, dst)] = round(random.uniform(5.0, 30.0), 1)

        # Additional random forward cross-edges based on edge density
        if edge_density > 0.3:
            for src in src_nodes:
                for dst in dst_nodes:
                    if (src, dst) not in edges and random.random() < (edge_density - 0.2):
                        edges.append((src, dst))
                        edge_costs[(src, dst)] = round(random.uniform(5.0, 25.0), 1)

    return DAGApplication(
        app_id=app_id,
        dag_type=f"Synthetic Layered ({num_tasks} tasks)",
        tasks=tasks,
        edges=edges,
        edge_costs=edge_costs,
        criticality=criticality,
        arrival_time=0.0,
        relative_deadline=relative_deadline
    )


# ------------------------------------------------------------------------------
# GRAPH TOPOLOGY & STRUCTURAL METRICS ENGINE
# ------------------------------------------------------------------------------
def compute_dag_metrics(dag: DAGApplication, platform: HeterogeneousPlatform) -> Dict[str, Any]:
    """
    Computes rigorous graph theory and real-time scheduling metrics:
    - Node Count (|V|), Edge Count (|E|)
    - Exact Critical Path (Longest Path in DAG) and its minimum execution cost
    - DAG Depth / Number of Pipeline Stages
    - Maximum Degree of Parallelism
    - Communication-to-Computation Ratio (CCR)
    - Theoretical Speedup Upper Bound (Amdahl's / Brent's Theorem)
    - In-degree & Out-degree distribution
    """
    G = nx.DiGraph()
    fastest_wcet: Dict[int, float] = {}

    for tid, t in dag.tasks.items():
        valid_costs = [t.get_wcet(pe.pe_type) for pe in platform.pes if t.get_wcet(pe.pe_type) < float("inf")]
        min_w = min(valid_costs) if valid_costs else 10.0
        fastest_wcet[tid] = min_w
        G.add_node(tid, name=t.name, kernel=t.task_type, min_wcet=min_w)

    for u, v in dag.edges:
        comm = dag.edge_costs.get((u, v), platform.default_inter_pe_comm)
        G.add_edge(u, v, weight=comm)

    # Weighted Graph for Critical Path
    G_weighted = nx.DiGraph()
    for tid in G.nodes():
        G_weighted.add_node(tid)
    for u, v, data in G.edges(data=True):
        G_weighted.add_edge(u, v, weight=fastest_wcet[u] + data["weight"])

    # Longest Path (Critical Path)
    try:
        longest_path = nx.dag_longest_path(G_weighted, weight="weight")
    except Exception:
        longest_path = list(dag.tasks.keys())[:1]

    cp_latency = 0.0
    for i in range(len(longest_path) - 1):
        u, v = longest_path[i], longest_path[i+1]
        cp_latency += fastest_wcet[u] + dag.edge_costs.get((u, v), platform.default_inter_pe_comm)
    if longest_path:
        cp_latency += fastest_wcet[longest_path[-1]]

    # Layer & Parallelism Analysis
    try:
        generations = list(nx.topological_generations(G))
        depth = len(generations)
        max_parallelism = max([len(gen) for gen in generations]) if generations else 1
    except Exception:
        depth = 1
        max_parallelism = len(dag.tasks)

    total_fastest_workload = sum(fastest_wcet.values())
    total_comm = sum(data.get("weight", 0.0) for _, _, data in G.edges(data=True))
    ccr = total_comm / total_fastest_workload if total_fastest_workload > 0 else 0.0
    speedup_bound = total_fastest_workload / cp_latency if cp_latency > 0 else 1.0

    in_degrees = [d for _, d in G.in_degree()]
    out_degrees = [d for _, d in G.out_degree()]

    return {
        "num_nodes": len(dag.tasks),
        "num_edges": len(dag.edges),
        "critical_path_nodes": longest_path,
        "critical_path_latency_ms": cp_latency,
        "depth": depth,
        "max_parallelism": max_parallelism,
        "total_workload_ms": total_fastest_workload,
        "total_comm_ms": total_comm,
        "ccr": ccr,
        "speedup_bound": speedup_bound,
        "max_in_degree": max(in_degrees) if in_degrees else 0,
        "max_out_degree": max(out_degrees) if out_degrees else 0,
    }


# ------------------------------------------------------------------------------
# INTERACTIVE 2D PLOTLY NETWORK GRAPH VISUALIZER
# ------------------------------------------------------------------------------
def create_dag_network_plotly_figure(
    dag: DAGApplication,
    platform: HeterogeneousPlatform,
    critical_path_nodes: List[int],
    color_mode: str = "Kernel Category",
    show_comm_labels: bool = True
) -> go.Figure:
    """
    Renders an interactive 2D Directed Acyclic Graph with:
    - Hierarchical topological layered coordinates (left-to-right flow)
    - Glowing amber highlight for Critical Path nodes & edges
    - Directed arrows with communication latencies (ms)
    - Rich hover tooltips with multi-PE execution profiles
    """
    G = nx.DiGraph()
    for tid, t in dag.tasks.items():
        G.add_node(tid, name=t.name, kernel=t.task_type)
    for u, v in dag.edges:
        G.add_edge(u, v)

    # Compute topological layer layout
    pos: Dict[int, Tuple[float, float]] = {}
    try:
        generations = list(nx.topological_generations(G))
    except Exception:
        generations = [[tid] for tid in dag.tasks.keys()]

    num_layers = len(generations)
    for layer_idx, nodes in enumerate(generations):
        x = layer_idx * 2.2
        n_nodes = len(nodes)
        for i, node in enumerate(nodes):
            y = (i - (n_nodes - 1) / 2.0) * 1.8
            pos[node] = (x, y)

    # Critical path edge pairs
    cp_edges = set((critical_path_nodes[i], critical_path_nodes[i+1]) for i in range(len(critical_path_nodes) - 1))
    cp_node_set = set(critical_path_nodes)

    fig = go.Figure()

    # 1. Draw Edges & Directed Arrows
    for u, v in dag.edges:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        is_cp = (u, v) in cp_edges
        edge_color = "#f59e0b" if is_cp else "rgba(148, 163, 184, 0.45)"
        edge_width = 3.5 if is_cp else 1.8
        comm = dag.edge_costs.get((u, v), platform.default_inter_pe_comm)

        # Base edge line
        fig.add_trace(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode="lines",
            line=dict(color=edge_color, width=edge_width, dash="solid" if is_cp else "dot"),
            hoverinfo="none",
            showlegend=False
        ))

        # Arrowhead annotation
        ax = x0 + 0.72 * (x1 - x0)
        ay = y0 + 0.72 * (y1 - y0)
        fig.add_annotation(
            x=ax, y=ay,
            ax=ax - 0.1 * (x1 - x0),
            ay=ay - 0.1 * (y1 - y0),
            xref="x", yref="y",
            axref="x", ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.4,
            arrowwidth=2.0 if is_cp else 1.2,
            arrowcolor=edge_color
        )

        # Edge communication latency badge
        if show_comm_labels:
            mid_x = (x0 + x1) / 2.0
            mid_y = (y0 + y1) / 2.0 + 0.14
            fig.add_trace(go.Scatter(
                x=[mid_x],
                y=[mid_y],
                mode="text",
                text=[f"{comm:.0f}ms"],
                textfont=dict(size=9, color="#f59e0b" if is_cp else "#94a3b8"),
                hoverinfo="none",
                showlegend=False
            ))

    # 2. Draw Nodes
    node_x, node_y, node_text, node_color, node_hover = [], [], [], [], []
    node_line_color, node_line_width = [], []

    for tid, (x, y) in pos.items():
        task = dag.tasks[tid]
        is_cp = tid in cp_node_set
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"<b>{task.name}</b><br><span style='font-size:9px'>T{tid}</span>")

        # Color mode
        if color_mode == "Critical Path":
            col = "#f59e0b" if is_cp else "#38bdf8"
        elif color_mode == "ASIL Criticality":
            col = "#ff0055" if dag.criticality >= 2 else "#00f0ff"
        else:  # Kernel Category
            k_prefix = task.task_type.split("_")[0]
            col = KERNEL_COLORS.get(k_prefix, "#38bdf8")

        node_color.append(col)

        if is_cp:
            node_line_color.append("#f59e0b")
            node_line_width.append(3.5)
        else:
            node_line_color.append("rgba(255, 255, 255, 0.7)")
            node_line_width.append(1.5)

        # Hover Profile Card
        valid_pe_costs = []
        for pe in platform.pes:
            c = task.get_wcet(pe.pe_type)
            if c < float("inf"):
                valid_pe_costs.append(f"• <b>{pe.name}</b> ({pe.pe_type}): <code>{c:.1f} ms</code>")

        hover_card = (
            f"<b>Task T{tid}: {task.name}</b><br>"
            f"<b>Kernel:</b> {task.task_type}<br>"
            f"<b>Critical Path:</b> {'🔥 YES (Critical Delay)' if is_cp else 'No'}<br>"
            f"<b>HEFT Upward Rank (rank_u):</b> {task.upward_rank:.2f}<br>"
            f"<b>Predecessors:</b> {task.predecessors}<br>"
            f"<b>Successors:</b> {task.successors}<br>"
            f"<b>Heterogeneous Execution Times:</b><br>" +
            "<br>".join(valid_pe_costs[:5])
        )
        node_hover.append(hover_card)

    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        marker=dict(
            size=52,
            color=node_color,
            line=dict(color=node_line_color, width=node_line_width),
            opacity=0.95
        ),
        text=node_text,
        textposition="middle center",
        textfont=dict(color="#ffffff", size=10, family="Inter, Arial, sans-serif"),
        hovertext=node_hover,
        hoverinfo="text",
        showlegend=False
    ))

    fig.update_layout(
        title=f"🕸️ {dag.dag_type} — Directed Precedence Graph & Critical Path Flow",
        plot_bgcolor="rgba(11, 15, 25, 0.9)",
        paper_bgcolor="rgba(11, 15, 25, 0.9)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        template="plotly_dark",
        height=500,
        margin=dict(l=30, r=30, t=50, b=30)
    )

    return fig


# ------------------------------------------------------------------------------
# GRAPHVIZ DOT FORMAT EXPORTER (FOR CRISP SVG EMBEDDING)
# ------------------------------------------------------------------------------
def generate_dag_graphviz_dot(
    dag: DAGApplication,
    platform: HeterogeneousPlatform,
    critical_path_nodes: List[int]
) -> str:
    """Generates pure Graphviz DOT notation for native Streamlit SVG rendering."""
    cp_set = set(critical_path_nodes)
    cp_edges = set((critical_path_nodes[i], critical_path_nodes[i+1]) for i in range(len(critical_path_nodes) - 1))

    dot_lines = [
        "digraph DAG {",
        "  rankdir=LR;",
        "  bgcolor=\"transparent\";",
        "  node [style=\"filled,rounded\", shape=box, fontname=\"Inter, Arial\", fontcolor=\"white\", height=0.6, width=1.4];",
        "  edge [fontname=\"Inter, Arial\", fontsize=10, arrowsize=0.8];"
    ]

    for tid, t in dag.tasks.items():
        is_cp = tid in cp_set
        fill = "#b45309" if is_cp else "#0f766e"
        border = "#f59e0b" if is_cp else "#14b8a6"
        penwidth = 2.8 if is_cp else 1.2
        label = f"{t.name}\\nT{tid} | rank={t.upward_rank:.1f}"
        dot_lines.append(f'  node_{tid} [label="{label}", fillcolor="{fill}", color="{border}", penwidth={penwidth}];')

    for u, v in dag.edges:
        is_cp = (u, v) in cp_edges
        comm = dag.edge_costs.get((u, v), platform.default_inter_pe_comm)
        ecol = "#f59e0b" if is_cp else "#64748b"
        pwidth = 3.0 if is_cp else 1.2
        style = "solid" if is_cp else "dashed"
        dot_lines.append(f'  node_{u} -> node_{v} [label="{comm:.0f}ms", color="{ecol}", fontcolor="{ecol}", penwidth={pwidth}, style="{style}"];')

    dot_lines.append("}")
    return "\n".join(dot_lines)


# ------------------------------------------------------------------------------
# MULTI-DAG TRAFFIC STREAM & POISSON ARRIVAL VISUALIZER
# ------------------------------------------------------------------------------
def create_multi_dag_stream_plotly_figure(
    trace: List[DAGApplication],
    platform: HeterogeneousPlatform,
    max_display: int = 15,
    show_slack: bool = True
) -> go.Figure:
    """
    Visualizes concurrent multi-DAG application streams arriving on the clock timeline:
    - Solid Bar: Minimum critical path compute workload
    - Hatched/Translucent Bar: Context-aware slack cushion margin up to deadline
    - Inverted Triangle: Absolute deadline
    """
    sub_trace = trace[:max_display]
    fig = go.Figure()

    for idx, app in enumerate(sub_trace):
        is_crit = (app.criticality >= 2)
        base_color = "#ff0055" if is_crit else "#00f0ff"
        slack_color = "rgba(255, 0, 85, 0.22)" if is_crit else "rgba(0, 240, 255, 0.22)"
        crit_label = "Safety-Critical (ASIL-D, C=2)" if is_crit else "Normal (QM, C=1)"

        m = compute_dag_metrics(app, platform)
        cp_time = round(m["critical_path_latency_ms"], 1)
        arr = round(app.arrival_time, 1)
        rel_ddl = round(app.relative_deadline, 1)
        abs_dl = round(arr + rel_ddl, 1)
        slack_time = round(max(0.0, rel_ddl - cp_time), 1)

        base_type = app.dag_type.split("(")[0].strip()
        app_label = f"DAG #{app.app_id}: {base_type} ({len(app.tasks)} tasks)"

        if show_slack:
            # 1. Compute execution segment
            fig.add_trace(go.Bar(
                y=[app_label],
                x=[cp_time],
                base=[arr],
                orientation="h",
                name="Critical Path Execution (Compute)" if idx == 0 else None,
                legendgroup="Execution",
                marker=dict(color=base_color, opacity=0.9, line=dict(color="rgba(255,255,255,0.4)", width=1)),
                hovertext=(
                    f"<b>{app_label}</b><br>"
                    f"<b>Criticality:</b> {crit_label}<br>"
                    f"<b>Arrival Clock (T_arr):</b> {arr:.1f} ms<br>"
                    f"<b>Min Execution Workload:</b> {cp_time:.1f} ms<br>"
                    f"<b>Available Slack Margin:</b> {slack_time:.1f} ms<br>"
                    f"<b>Absolute Deadline:</b> {abs_dl:.1f} ms<br>"
                    f"<b>Internal Tasks:</b> {len(app.tasks)} nodes"
                ),
                hoverinfo="text",
                showlegend=(idx == 0)
            ))

            # 2. Slack cushion segment
            if slack_time > 0:
                fig.add_trace(go.Bar(
                    y=[app_label],
                    x=[slack_time],
                    base=[arr + cp_time],
                    orientation="h",
                    name="Available Slack Cushion" if idx == 0 else None,
                    legendgroup="Slack",
                    marker=dict(color=slack_color, line=dict(color=base_color, width=1.2)),
                    hovertext=(
                        f"<b>{app_label} (Slack Cushion)</b><br>"
                        f"<b>Available Slack Margin:</b> {slack_time:.1f} ms<br>"
                        f"<b>Window:</b> {arr + cp_time:.1f} ms ➔ {abs_dl:.1f} ms"
                    ),
                    hoverinfo="text",
                    showlegend=(idx == 0)
                ))
        else:
            fig.add_trace(go.Bar(
                y=[app_label],
                x=[rel_ddl],
                base=[arr],
                orientation="h",
                name=crit_label if idx == 0 else None,
                marker=dict(color=base_color, opacity=0.85, line=dict(color="rgba(255,255,255,0.3)", width=1)),
                hovertext=(
                    f"<b>{app_label}</b><br>"
                    f"<b>Criticality:</b> {crit_label}<br>"
                    f"<b>Arrival Clock:</b> {arr:.1f} ms<br>"
                    f"<b>Relative Deadline:</b> {rel_ddl:.1f} ms<br>"
                    f"<b>Absolute Deadline:</b> {abs_dl:.1f} ms"
                ),
                hoverinfo="text",
                showlegend=(idx == 0)
            ))

        # Arrival trigger marker
        fig.add_trace(go.Scatter(
            x=[arr],
            y=[app_label],
            mode="markers",
            marker=dict(size=9, symbol="triangle-right", color="#ffffff"),
            hoverinfo="none",
            showlegend=False
        ))

        # Absolute deadline marker
        fig.add_trace(go.Scatter(
            x=[abs_dl],
            y=[app_label],
            mode="markers",
            marker=dict(size=10, symbol="triangle-left", color=base_color, line=dict(color="#ffffff", width=1)),
            hovertext=f"<b>Absolute Deadline:</b> {abs_dl:.1f} ms",
            hoverinfo="text",
            showlegend=False
        ))

    fig.update_layout(
        title=r"🚦 Multi-DAG Dynamic Arrival & Real-Time Horizon Stream (Poisson $\lambda$)",
        xaxis=dict(title="Simulation Clock Time (ms)", rangeslider=dict(visible=True)),
        yaxis=dict(title="Active Arriving DAGs", autorange="reversed"),
        template="plotly_dark",
        barmode="overlay",
        height=460,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


# ------------------------------------------------------------------------------
# COMPLETE DAG STUDIO USER INTERFACE
# ------------------------------------------------------------------------------
def render_dag_studio_tab(current_benchmark: str):
    """
    Renders the expanded, interactive DAG Studio inside Tab 1 of the platform.
    """
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(15, 23, 42, 0.9), rgba(30, 41, 59, 0.8));
                border: 1px solid rgba(0, 240, 255, 0.3); border-radius: 12px; padding: 18px 22px; margin-bottom: 22px;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <span style="background: rgba(0, 240, 255, 0.15); color: #00f0ff; border: 1px solid #00f0ff;
                             font-size: 0.75rem; font-weight: 700; padding: 3px 10px; border-radius: 9999px;">
                    DAG Topology & Precedence Studio
                </span>
                <h2 style="color: #ffffff; margin: 8px 0 4px 0; font-size: 22px;">
                    Autonomous Driving & In-Cabin Multi-Task Directed Acyclic Graphs
                </h2>
                <p style="color: #94a3b8; margin: 0; font-size: 13px;">
                    Interactive 2D graph visualizer, critical path analyzer, topological metrics, and synthetic DAG generator.
                </p>
            </div>
            <div style="text-align: right; background: rgba(0, 240, 255, 0.05); padding: 8px 14px; border-radius: 8px; border: 1px solid rgba(0, 240, 255, 0.2); margin-top: 6px;">
                <div style="color: #00f0ff; font-family: 'Orbitron'; font-size: 16px; font-weight: bold;">IEEE TII 2026</div>
                <div style="color: #64748b; font-size: 11px;">Section III & IV-A Modeling</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Mode Selector
    studio_subtab = st.radio(
        "DAG Studio Mode",
        [
            "🕸️ Interactive Graph Visualizer & Critical Path",
            "📊 Topology & Structural Metrics",
            "🧠 HEFT Upward Rank (rank_u) Breakdown",
            "🚦 Multi-DAG Dynamic Traffic Stream",
            "🧪 Custom / Synthetic DAG Generator Sandbox"
        ],
        horizontal=True
    )

    st.markdown("---")

    # DAG Selection Dropdown
    default_dag_idx = 4 if ("SelfPilot" in current_benchmark) else 1
    col_sel1, col_sel2 = st.columns([2, 1])
    with col_sel1:
        dag_choice = st.selectbox(
            "Select Target Application DAG",
            [
                "ADSuite Type 1: Single Camera Perception Pipeline (6 tasks)",
                "ADSuite Type 2: Dual Camera + LiDAR Sensor Fusion (8 tasks)",
                "ADSuite Type 3: Cross Perception & Control (7 tasks)",
                "ADSuite Type 4: Dual Perception & Split Planning (9 tasks)",
                "SelfPilot: Full Real-World Autonomous Vehicle Pipeline (9 tasks)",
                "DMS Cockpit: ISO 26262 ASIL-D In-Cabin Perception (8 tasks)"
            ],
            index=default_dag_idx,
            key="dag_studio_target_dag_select"
        )
    with col_sel2:
        graph_engine = st.selectbox(
            "Graph Rendering Engine",
            ["Interactive Plotly 2D (Interactive)", "Graphviz SVG (Vector)"],
            key="dag_studio_engine_select"
        )

    # Instantiate chosen DAG and pair with matching hardware platform
    if "Type 1" in dag_choice:
        active_dag = create_adsuite_dag(0, "Type 1")
        platform_ref = create_adsuite_platform()
    elif "Type 2" in dag_choice:
        active_dag = create_adsuite_dag(0, "Type 2")
        platform_ref = create_adsuite_platform()
    elif "Type 3" in dag_choice:
        active_dag = create_adsuite_dag(0, "Type 3")
        platform_ref = create_adsuite_platform()
    elif "Type 4" in dag_choice:
        active_dag = create_adsuite_dag(0, "Type 4")
        platform_ref = create_adsuite_platform()
    elif "SelfPilot" in dag_choice:
        active_dag = create_selfpilot_dag(0)
        platform_ref = create_selfpilot_platform()
    else:
        active_dag = create_dms_dag(0)
        platform_ref = create_adsuite_platform()

    # Compute ranks & metrics
    active_dag.compute_heft_ranks(platform_ref)
    metrics = compute_dag_metrics(active_dag, platform_ref)
    cp_nodes = metrics["critical_path_nodes"]

    # --------------------------------------------------------------------------
    # SUBTAB 1: INTERACTIVE GRAPH VISUALIZER & CRITICAL PATH
    # --------------------------------------------------------------------------
    if studio_subtab.startswith("🕸️ Interactive Graph"):
        col_ctrl1, col_ctrl2 = st.columns([1, 1])
        with col_ctrl1:
            color_mode = st.radio("Node Color Palette", ["Kernel Category", "Critical Path", "ASIL Criticality"], horizontal=True)
        with col_ctrl2:
            show_comm = st.checkbox("Show Edge Latencies (ms)", value=True)

        if graph_engine.startswith("Interactive Plotly"):
            fig_dag = create_dag_network_plotly_figure(
                active_dag, platform_ref, cp_nodes, color_mode=color_mode, show_comm_labels=show_comm
            )
            st.plotly_chart(fig_dag, use_container_width=True)
        else:
            dot_str = generate_dag_graphviz_dot(active_dag, platform_ref, cp_nodes)
            st.graphviz_chart(dot_str, use_container_width=True)

        # Critical path banner
        cp_task_names = [f"T{tid} ({active_dag.tasks[tid].name})" for tid in cp_nodes]
        st.markdown(f"""
        <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid #f59e0b; border-radius: 8px; padding: 12px 16px; margin-top: 10px;">
            <span style="color: #f59e0b; font-weight: bold;">🔥 Critical Path Sequence (Bottleneck Delay):</span>
            <span style="color: #ffffff; margin-left: 8px;">{' ➔ '.join(cp_task_names)}</span>
            <div style="color: #cbd5e1; font-size: 12px; margin-top: 4px;">
                Total Minimum Critical Path Latency: <b>{metrics['critical_path_latency_ms']:.1f} ms</b> •
                Any preemption or slowdown along this chain immediately delays the application makespan.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # SUBTAB 2: TOPOLOGY & STRUCTURAL METRICS
    # --------------------------------------------------------------------------
    elif studio_subtab.startswith("📊 Topology"):
        st.markdown("#### 📐 Mathematical Topology & Scheduling Characteristics")

        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Total Nodes (|V|)", metrics["num_nodes"])
        with m2:
            st.metric("Total Edges (|E|)", metrics["num_edges"])
        with m3:
            st.metric("Graph Depth (Stages)", f"{metrics['depth']} levels")
        with m4:
            st.metric("Max Parallelism", f"{metrics['max_parallelism']} tasks")
        with m5:
            st.metric("Critical Path Length", f"{metrics['critical_path_latency_ms']:.1f} ms")

        st.markdown("---")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown(
                "##### ⚡ Speedup & Concurrency Limits\n"
                f"- **Total Workload Volume (Fastest Execution)**: `{metrics['total_workload_ms']:.1f} ms`\n"
                f"- **Total Edge Communication Delay**: `{metrics['total_comm_ms']:.1f} ms`\n"
                f"- **Communication-to-Computation Ratio (CCR)**: `{metrics['ccr']:.3f}`\n"
                f"- **Theoretical Maximum Speedup (Brent's Theorem)**: `{metrics['speedup_bound']:.2f}x`\n"
                f"- **Max Fan-In (Predecessors)**: `{metrics['max_in_degree']}`\n"
                f"- **Max Fan-Out (Successors)**: `{metrics['max_out_degree']}`"
            )
        with col_t2:
            # Parallelism Profile Chart
            G_temp = nx.DiGraph()
            for u, v in active_dag.edges:
                G_temp.add_edge(u, v)
            try:
                generations = list(nx.topological_generations(G_temp))
                stage_df = pd.DataFrame([
                    {"Pipeline Stage": f"Stage {idx + 1}", "Concurrent Tasks": len(gen), "Tasks": ", ".join(active_dag.tasks[t].name for t in gen)}
                    for idx, gen in enumerate(generations)
                ])
                fig_stages = px.bar(
                    stage_df,
                    x="Pipeline Stage",
                    y="Concurrent Tasks",
                    hover_data=["Tasks"],
                    title="Parallelism Profile by Pipeline Stage",
                    color_discrete_sequence=["#06b6d4"],
                    template="plotly_dark",
                    height=280
                )
                st.plotly_chart(fig_stages, use_container_width=True)
            except Exception:
                pass

    # --------------------------------------------------------------------------
    # SUBTAB 3: HEFT UPWARD RANK BREAKDOWN
    # --------------------------------------------------------------------------
    elif studio_subtab.startswith("🧠 HEFT Upward"):
        st.markdown("""
        #### 🔢 HEFT Upward Rank Formulation (Section III-B & Eq. 7)
        Tasks are dynamically sorted within each application descending by **Upward Rank ($rank_u$)**:
        $$rank_u(v_i) = \\bar{w}_i + \\max_{v_j \\in succ(v_i)} \\left( rank_u(v_j) + \\bar{c}_{i,j} \\right)$$
        Higher upward rank indicates that the task has longer downstream dependent workloads and must be scheduled earlier.
        """)

        rank_table = []
        for tid in active_dag.ordered_tasks:
            t = active_dag.tasks[tid]
            valid_costs = [t.get_wcet(pe.pe_type) for pe in platform_ref.pes if t.get_wcet(pe.pe_type) < float("inf")]
            avg_w = sum(valid_costs) / len(valid_costs) if valid_costs else 0.0
            is_cp = tid in cp_nodes

            rank_table.append({
                "Task ID": tid,
                "Task Name": t.name,
                "Kernel": t.task_type,
                "Upward Rank (rank_u)": round(t.upward_rank, 2),
                "Average WCET (w_bar) [ms]": round(avg_w, 2),
                "Critical Path": "🔥 YES" if is_cp else "No",
                "Predecessors": str(t.predecessors),
                "Successors": str(t.successors)
            })

        df_rank = pd.DataFrame(rank_table)
        st.dataframe(df_rank, use_container_width=True)

        fig_rank = px.bar(
            df_rank,
            x="Task Name",
            y="Upward Rank (rank_u)",
            color="Critical Path",
            color_discrete_map={"🔥 YES": "#f59e0b", "No": "#38bdf8"},
            title="HEFT Upward Rank Priority Spectrum (Descending Order)",
            template="plotly_dark",
            height=320
        )
        st.plotly_chart(fig_rank, use_container_width=True)

    # --------------------------------------------------------------------------
    # SUBTAB 4: MULTI-DAG TRAFFIC STREAM
    # --------------------------------------------------------------------------
    elif studio_subtab.startswith("🚦 Multi-DAG"):
        st.markdown("#### 🚦 Concurrent Multi-DAG Arrival Stream Visualizer")
        st.markdown(r"Demonstrates dynamic application traffic arrivals generated under Poisson arrival process ($\lambda$).")

        col_str1, col_str2, col_str3 = st.columns([2, 1, 1])
        with col_str1:
            stream_profile = st.selectbox(
                "Workload Traffic Pattern",
                [
                    "Synchronized with Dropdown Selection Above",
                    "Mixed Autonomous Driving (Random Types 1, 2, 3, 4)",
                    "Pure SelfPilot Real-Vehicle Stream (9 Tasks)",
                    "Pure Driver Monitoring System (DMS) Stream (8 Tasks)"
                ],
                key="dag_stream_profile"
            )
        with col_str2:
            stream_lambda = st.slider("Arrival Rate (λ [DAG/s])", min_value=5.0, max_value=40.0, value=20.0, step=5.0, key="st_lam")
        with col_str3:
            stream_count = st.slider("Displayed Arrivals", min_value=5, max_value=20, value=10, step=1, key="st_cnt")

        # Generate dynamic stream with realistic, varying deadlines
        random.seed(42)
        np.random.seed(42)
        trace_stream: List[DAGApplication] = []
        clock = 0.0

        if "SelfPilot" in stream_profile or ("Synchronized" in stream_profile and "SelfPilot" in dag_choice):
            st_platform = create_selfpilot_platform()
            for i in range(stream_count):
                inter_arr = np.random.exponential(1000.0 / stream_lambda)
                clock += inter_arr
                crit = 2 if random.random() < 0.6 else 1
                rel_ddl = round(max(80.0, float(np.random.normal(200.0, 30.0))), 1)
                trace_stream.append(create_selfpilot_dag(i, criticality=crit, arrival_time=clock, relative_deadline=rel_ddl))
        elif "DMS" in stream_profile or ("Synchronized" in stream_profile and "DMS" in dag_choice):
            st_platform = create_adsuite_platform()
            for i in range(stream_count):
                inter_arr = np.random.exponential(1000.0 / stream_lambda)
                clock += inter_arr
                crit = 2 if random.random() < 0.6 else 1
                rel_ddl = round(float(random.choice([33.3, 50.0, 75.0, 100.0])), 1)
                trace_stream.append(create_dms_dag(i, criticality=crit, arrival_time=clock, relative_deadline=rel_ddl))
        else:
            st_platform = create_adsuite_platform()
            ddl_bases = {"Type 1": 240.0, "Type 2": 360.0, "Type 3": 300.0, "Type 4": 420.0}
            dag_types = ["Type 1", "Type 2", "Type 3", "Type 4"]

            for i in range(stream_count):
                inter_arr = np.random.exponential(1000.0 / stream_lambda)
                clock += inter_arr
                crit = 2 if random.random() < 0.6 else 1

                if "Synchronized" in stream_profile:
                    if "Type 1" in dag_choice:
                        dt = "Type 1"
                    elif "Type 2" in dag_choice:
                        dt = "Type 2"
                    elif "Type 3" in dag_choice:
                        dt = "Type 3"
                    elif "Type 4" in dag_choice:
                        dt = "Type 4"
                    else:
                        dt = random.choice(dag_types)
                else:
                    dt = random.choice(dag_types)

                base_d = ddl_bases.get(dt, 320.0)
                rel_ddl = round(max(100.0, float(np.random.normal(base_d, 25.0))), 1)
                trace_stream.append(create_adsuite_dag(i, dag_type=dt, criticality=crit, arrival_time=clock, relative_deadline=rel_ddl))

        fig_stream = create_multi_dag_stream_plotly_figure(trace_stream, st_platform, max_display=stream_count, show_slack=True)
        st.plotly_chart(fig_stream, use_container_width=True)
        st.caption(
            "🔴 Solid Colored: Minimum Critical Path Compute Execution Time • "
            "🏁 Hatched/Translucent: Available Context-Aware Slack (CAS) Margin Cushion • "
            "◀ Inverted Triangle: Absolute End-to-End Deadline"
        )

    # --------------------------------------------------------------------------
    # SUBTAB 5: CUSTOM / SYNTHETIC DAG GENERATOR SANDBOX
    # --------------------------------------------------------------------------
    else:
        st.markdown("#### 🧪 Synthetic Layered DAG Generator Sandbox")
        st.markdown("Interactively configure and synthesize novel automotive DAG topologies to inspect their properties.")

        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            syn_nodes = st.slider("Number of Tasks (|V|)", min_value=4, max_value=20, value=9, step=1)
        with col_s2:
            syn_layers = st.slider("Pipeline Depth (Layers)", min_value=2, max_value=6, value=4, step=1)
        with col_s3:
            syn_density = st.slider("Cross-Branch Edge Density", min_value=0.1, max_value=0.9, value=0.4, step=0.1)

        custom_dag = create_synthetic_dag(
            num_tasks=syn_nodes,
            num_layers=syn_layers,
            edge_density=syn_density,
            app_id=99,
            criticality=2,
            relative_deadline=300.0,
            seed=random.randint(1, 9999) if st.button("🎲 Regenerate New Random Seed") else 42
        )
        custom_dag.compute_heft_ranks(platform_ref)
        syn_metrics = compute_dag_metrics(custom_dag, platform_ref)
        syn_cp = syn_metrics["critical_path_nodes"]

        # Display Metrics
        sm1, sm2, sm3, sm4 = st.columns(4)
        with sm1:
            st.metric("Synthesized Nodes", syn_metrics["num_nodes"])
        with sm2:
            st.metric("Synthesized Edges", syn_metrics["num_edges"])
        with sm3:
            st.metric("Critical Path", f"{syn_metrics['critical_path_latency_ms']:.1f} ms")
        with sm4:
            st.metric("Max Parallelism", f"{syn_metrics['max_parallelism']} tasks")

        fig_syn = create_dag_network_plotly_figure(custom_dag, platform_ref, syn_cp)
        st.plotly_chart(fig_syn, use_container_width=True)
