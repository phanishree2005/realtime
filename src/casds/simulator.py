"""
Dynamic Discrete-Event Simulation Engine for ACPS Task Scheduling
100% Dynamic: Automatically loads platforms, DAG topologies, and task profiles
directly from JSON configuration and graph files without hardcoding.
"""

import copy
import glob
import heapq
import json
import os
import random
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.casds.core import (
    DAGApplication,
    ProcessingElement,
    TaskNode,
    UpwardRankCalculator,
)
from src.casds.schedulers import BaseScheduler


def load_platform_from_json(config_path: str = "inputs/stomp_real.json") -> Tuple[List[ProcessingElement], List[str]]:
    """
    Dynamically constructs Processing Elements directly from JSON configuration.
    Extracts server types and counts without any hardcoding.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    servers_dict = config.get("simulation", {}).get("servers", {})
    if not servers_dict:
        raise ValueError(f"No 'servers' block found in {config_path}")

    pes: List[ProcessingElement] = []
    server_types_order: List[str] = []

    # Sort by server ID if available to ensure exact column alignment
    sorted_servers = sorted(
        servers_dict.items(),
        key=lambda item: item[1].get("id", 0) if isinstance(item[1], dict) else 0
    )

    for stype, sdata in sorted_servers:
        server_types_order.append(stype)
        count = sdata.get("count", 1) if isinstance(sdata, dict) else 1
        for i in range(count):
            pes.append(ProcessingElement(pe_id=f"{stype}_{i}", pe_type=stype))

    return pes, server_types_order


class DynamicDAGLoader:
    """
    Dynamically loads DAG topologies (.graphml) and execution matrices (.txt)
    from any application folder based on the server types defined in the configuration.
    """
    def __init__(self, dag_dir: str = "inputs/ad/dag_input", server_types_order: Optional[List[str]] = None):
        self.dag_dir = dag_dir
        self.server_types = server_types_order or []
        self.templates: Dict[str, DAGApplication] = {}
        self._discover_and_load_templates()

    def _discover_and_load_templates(self):
        # Look for all .graphml files in the target directory
        graphml_files = glob.glob(os.path.join(self.dag_dir, "dag*.graphml"))
        
        for gpath in graphml_files:
            fname = os.path.basename(gpath)
            # Match pattern dag<id>.graphml
            match = re.search(r"dag([0-9a-zA-Z_]+)\.graphml", fname)
            if not match:
                continue
            dtype = match.group(1)
            comp_path = os.path.join(self.dag_dir, f"dag_{dtype}.txt")
            if not os.path.exists(comp_path):
                # Try fallback naming
                comp_path = os.path.join(self.dag_dir, f"dag{dtype}.txt")
                if not os.path.exists(comp_path):
                    continue

            # Dynamically parse task execution matrix
            task_profiles: List[Tuple[str, Dict[str, float]]] = []
            with open(comp_path, 'r') as f:
                raw_lines = [line.strip() for line in f if line.strip()]
                if not raw_lines:
                    continue

                for row_idx, line in enumerate(raw_lines[1:]):  # skip header row
                    cols = [c.strip() for c in line.split(',')]
                    ktype = cols[0]
                    exec_times: Dict[str, float] = {}

                    # STOMP convention: Column 0 is task type, Column 1 is weight/slack factor.
                    # Column 2+ correspond to the server types in order.
                    for col_idx, val in enumerate(cols[2:]):
                        if col_idx < len(self.server_types):
                            stype = self.server_types[col_idx]
                            if val != "None":
                                try:
                                    exec_times[stype] = float(val)
                                except ValueError:
                                    pass

                    task_profiles.append((ktype, exec_times))

            # Dynamically parse graph structure from .graphml
            try:
                tree = ET.parse(gpath)
                root = tree.getroot()
                ns = {'g': 'http://graphml.graphdrawing.org/xmlns'}
                graph_elem = root.find('g:graph', ns) or root.find('graph')
                if graph_elem is None:
                    continue

                nodes_elem = graph_elem.findall('g:node', ns) or graph_elem.findall('node')
                edges_elem = graph_elem.findall('g:edge', ns) or graph_elem.findall('edge')

                tasks: Dict[int, TaskNode] = {}
                for idx, node in enumerate(nodes_elem):
                    nid = int(node.get('id', idx))
                    ktype, edict = task_profiles[idx] if idx < len(task_profiles) else (f"task_{nid}", {})
                    tasks[nid] = TaskNode(
                        task_id=nid,
                        dag_id=0,
                        kernel_type=ktype,
                        execution_times=edict,
                        predecessors=[],
                        successors=[]
                    )

                for edge in edges_elem:
                    src = int(edge.get('source'))
                    tgt = int(edge.get('target'))
                    if src in tasks and tgt in tasks:
                        tasks[tgt].predecessors.append(src)
                        tasks[src].successors.append(tgt)

                template_dag = DAGApplication(
                    dag_id=0,
                    dag_type=str(dtype),
                    criticality=1,
                    arrival_time=0.0,
                    relative_deadline=400.0,
                    tasks=tasks
                )
                UpwardRankCalculator.compute_ranks(template_dag)
                self.templates[str(dtype)] = template_dag
            except Exception as e:
                pass

    def instantiate_dag(self, dag_id: int, dag_type: str, criticality: int, arrival_time: float, deadline: float = 400.0) -> DAGApplication:
        key = str(dag_type)
        if key not in self.templates:
            # Fallback to first available template if requested type is absent
            key = next(iter(self.templates.keys()))
        template = self.templates[key]

        new_tasks: Dict[int, TaskNode] = {}
        for tid, t in template.tasks.items():
            new_tasks[tid] = TaskNode(
                task_id=tid,
                dag_id=dag_id,
                kernel_type=t.kernel_type,
                execution_times=dict(t.execution_times),
                predecessors=list(t.predecessors),
                successors=list(t.successors),
                rank_u=t.rank_u,
                status='pending'
            )

        return DAGApplication(
            dag_id=dag_id,
            dag_type=str(dag_type),
            criticality=criticality,
            arrival_time=arrival_time,
            relative_deadline=deadline,
            tasks=new_tasks
        )


@dataclass
class SimulationResults:
    scheduler_name: str
    total_dags: int
    successful_dags: int
    dsr_total: float
    critical_dags: int
    successful_critical_dags: int
    dsr_critical: float
    noncritical_dags: int
    successful_noncritical_dags: int
    dsr_noncritical: float
    makespan: float
    pe_utilization: float
    effective_throughput: float
    execution_timeline: List[dict]


class ACPSSimulator:
    """
    Dynamic Simulator for heterogeneous ACPS.
    Initializes entirely from dynamic config and files.
    """
    def __init__(self, config_path: str = "inputs/stomp_real.json", app_name: str = "ad"):
        self.config_path = config_path
        self.app_name = app_name
        self.base_platform, self.server_types = load_platform_from_json(config_path)
        dag_dir = os.path.join("inputs", app_name, "dag_input")
        self.template_loader = DynamicDAGLoader(dag_dir=dag_dir, server_types_order=self.server_types)

    def load_trace_file(self, trace_path: str, max_dags: Optional[int] = None) -> List[DAGApplication]:
        dags: List[DAGApplication] = []
        with open(trace_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = [p.strip() for p in line.split(',')]
                if len(parts) < 5:
                    continue
                arr_time = float(parts[0])
                did = int(parts[1])
                dtype = parts[2]
                crit = int(parts[3])
                deadline = float(parts[4])
                dag = self.template_loader.instantiate_dag(
                    dag_id=did,
                    dag_type=dtype,
                    criticality=crit,
                    arrival_time=arr_time,
                    deadline=deadline
                )
                dags.append(dag)
                if max_dags and len(dags) >= max_dags:
                    break
        return dags

    def generate_poisson_trace(
        self,
        num_dags: int = 500,
        arrival_rate_lambda: float = 30.0,
        rcrit: float = 0.60,
        deadline: float = 400.0,
        random_seed: int = 42
    ) -> List[DAGApplication]:
        rng = random.Random(random_seed)
        dags: List[DAGApplication] = []
        curr_time = 0.0
        available_types = list(self.template_loader.templates.keys())
        if not available_types:
            raise RuntimeError("No DAG templates found in target directory.")

        for did in range(num_dags):
            inter_arrival_s = rng.expovariate(arrival_rate_lambda)
            curr_time += inter_arrival_s * 1000.0
            dtype = rng.choice(available_types)
            crit = 2 if rng.random() < rcrit else 1
            dag = self.template_loader.instantiate_dag(
                dag_id=did,
                dag_type=dtype,
                criticality=crit,
                arrival_time=curr_time,
                deadline=deadline
            )
            dags.append(dag)

        return dags

    def run_simulation(
        self,
        scheduler: BaseScheduler,
        dag_trace: List[DAGApplication],
        noise_delay_factor: float = 0.0
    ) -> SimulationResults:
        dags = copy.deepcopy(dag_trace)
        pes = copy.deepcopy(self.base_platform)

        event_queue: List[Tuple[float, int, int, object]] = []
        event_counter = 0

        for dag in dags:
            heapq.heappush(event_queue, (dag.arrival_time, 0, event_counter, dag))
            event_counter += 1

        active_dags: Dict[int, DAGApplication] = {}
        all_dags: Dict[int, DAGApplication] = {d.dag_id: d for d in dags}
        current_time = 0.0
        timeline_log: List[dict] = []

        while event_queue:
            evt_time, evt_type, _, payload = heapq.heappop(event_queue)
            current_time = evt_time

            if evt_type == 0:
                new_dag: DAGApplication = payload
                active_dags[new_dag.dag_id] = new_dag

            elif evt_type == 1:
                finished_task: TaskNode = payload
                dag = all_dags[finished_task.dag_id]
                finished_task.status = 'completed'
                finished_task.finish_time = current_time

                if dag.is_finished() and not dag.completed:
                    dag.completed = True
                    dag.completion_time = current_time
                    dag.missed_deadline = (current_time > dag.absolute_deadline)
                    if dag.dag_id in active_dags:
                        del active_dags[dag.dag_id]

            while event_queue and event_queue[0][0] <= current_time:
                sub_time, sub_type, _, sub_payload = heapq.heappop(event_queue)
                if sub_type == 0:
                    active_dags[sub_payload.dag_id] = sub_payload
                elif sub_type == 1:
                    ft = sub_payload
                    d = all_dags[ft.dag_id]
                    ft.status = 'completed'
                    ft.finish_time = sub_time
                    if d.is_finished() and not d.completed:
                        d.completed = True
                        d.completion_time = sub_time
                        d.missed_deadline = (sub_time > d.absolute_deadline)
                        if d.dag_id in active_dags:
                            del active_dags[d.dag_id]

            if active_dags:
                allocations = scheduler.schedule_step(current_time, active_dags, pes)
                for task, pe in allocations:
                    task.status = 'running'
                    task.allocated_pe = pe.pe_id
                    task.start_time = current_time

                    base_duration = task.execution_times.get(pe.pe_type, 10.0)
                    if noise_delay_factor > 0:
                        delay = max(0.0, np.random.normal(noise_delay_factor * base_duration, 0.1 * noise_delay_factor * base_duration))
                        duration = base_duration + delay
                    else:
                        duration = base_duration

                    pe.busy_until = current_time + duration
                    pe.total_busy_time += duration
                    finish_time = current_time + duration

                    timeline_entry = {
                        'task_id': f"{task.dag_id}.{task.task_id}",
                        'dag_id': task.dag_id,
                        'dag_type': all_dags[task.dag_id].dag_type,
                        'kernel': task.kernel_type,
                        'criticality': all_dags[task.dag_id].criticality,
                        'pe_id': pe.pe_id,
                        'pe_type': pe.pe_type,
                        'start_time': current_time,
                        'finish_time': finish_time,
                        'duration': duration
                    }
                    timeline_log.append(timeline_entry)
                    pe.execution_log.append(timeline_entry)

                    heapq.heappush(event_queue, (finish_time, 1, event_counter, task))
                    event_counter += 1

        total_dags = len(dags)
        successful_dags = sum(1 for d in dags if d.completed and not d.missed_deadline)
        dsr_total = (successful_dags / total_dags) * 100.0 if total_dags else 0.0

        crit_dags = [d for d in dags if d.criticality > 1]
        crit_count = len(crit_dags)
        crit_success = sum(1 for d in crit_dags if d.completed and not d.missed_deadline)
        dsr_critical = (crit_success / crit_count) * 100.0 if crit_count else 0.0

        noncrit_dags = [d for d in dags if d.criticality <= 1]
        noncrit_count = len(noncrit_dags)
        noncrit_success = sum(1 for d in noncrit_dags if d.completed and not d.missed_deadline)
        dsr_noncrit = (noncrit_success / noncrit_count) * 100.0 if noncrit_count else 0.0

        makespan = current_time
        total_pe_busy = sum(pe.total_busy_time for pe in pes)
        total_pe_capacity = len(pes) * makespan if makespan > 0 else 1.0
        pe_utilization = (total_pe_busy / total_pe_capacity) * 100.0

        makespan_seconds = makespan / 1000.0 if makespan > 0 else 1.0
        effective_throughput = successful_dags / makespan_seconds if makespan_seconds > 0 else 0.0

        return SimulationResults(
            scheduler_name=scheduler.name,
            total_dags=total_dags,
            successful_dags=successful_dags,
            dsr_total=dsr_total,
            critical_dags=crit_count,
            successful_critical_dags=crit_success,
            dsr_critical=dsr_critical,
            noncritical_dags=noncrit_count,
            successful_noncritical_dags=noncrit_success,
            dsr_noncritical=dsr_noncrit,
            makespan=makespan,
            pe_utilization=pe_utilization,
            effective_throughput=effective_throughput,
            execution_timeline=timeline_log
        )
