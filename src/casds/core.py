"""
Core Mathematical Models for Dynamic Task Scheduling on ACPS:
- TaskNode, DAGApplication, ProcessingElement
- HEFT Upward Rank (rank_u, Eq. 7)
- Context-Aware Slack (CAS, Eq. 8-10)
- Entropy-Weight Method (EWM) + TOPSIS (Eq. 11-13)
All structures are generic and dynamically populated from input configuration files.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class TaskNode:
    task_id: int
    dag_id: int
    kernel_type: str
    execution_times: Dict[str, float]  # server_type -> execution time in ms (dynamically parsed)
    predecessors: List[int] = field(default_factory=list)
    successors: List[int] = field(default_factory=list)
    
    # Dynamic runtime tracking
    rank_u: float = 0.0
    status: str = 'pending'  # pending, ready, running, completed, deferred
    allocated_pe: Optional[str] = None
    start_time: Optional[float] = None
    finish_time: Optional[float] = None
    predicted_finish_time: Optional[float] = None


@dataclass
class DAGApplication:
    dag_id: int
    dag_type: str
    criticality: int
    arrival_time: float
    relative_deadline: float
    tasks: Dict[int, TaskNode] = field(default_factory=dict)
    
    # State tracking
    completed: bool = False
    missed_deadline: bool = False
    completion_time: Optional[float] = None
    
    @property
    def absolute_deadline(self) -> float:
        return self.arrival_time + self.relative_deadline

    def is_finished(self) -> bool:
        return all(t.status == 'completed' for t in self.tasks.values())

    def get_unfinished_tasks(self) -> List[TaskNode]:
        return [t for t in self.tasks.values() if t.status != 'completed']

    def get_ready_tasks(self) -> List[TaskNode]:
        ready = []
        for t in self.tasks.values():
            if t.status in ('pending', 'deferred'):
                preds_done = all(self.tasks[p].status == 'completed' for p in t.predecessors)
                if preds_done:
                    ready.append(t)
        return ready


@dataclass
class ProcessingElement:
    pe_id: str
    pe_type: str
    busy_until: float = 0.0
    current_task: Optional[TaskNode] = None
    total_busy_time: float = 0.0
    execution_log: List[dict] = field(default_factory=list)

    def is_available(self, current_time: float) -> bool:
        return self.busy_until <= current_time

    def earliest_available_time(self, current_time: float) -> float:
        return max(self.busy_until, current_time)


class UpwardRankCalculator:
    """Calculates upward rank rank_u(v_i^a) for tasks within a DAG (Eq. 7)."""

    @staticmethod
    def compute_ranks(dag: DAGApplication, avg_comm_cost: float = 5.0):
        # Helper to compute average execution time across compatible PEs
        def avg_exec_time(task: TaskNode) -> float:
            valid_times = [t for t in task.execution_times.values() if t is not None and t > 0]
            return float(np.mean(valid_times)) if valid_times else 10.0

        memo: Dict[int, float] = {}

        def get_rank(task_id: int) -> float:
            if task_id in memo:
                return memo[task_id]
            task = dag.tasks[task_id]
            w_avg = avg_exec_time(task)
            if not task.successors:
                memo[task_id] = w_avg
                return w_avg
            max_succ = max(get_rank(succ_id) + avg_comm_cost for succ_id in task.successors)
            memo[task_id] = w_avg + max_succ
            return memo[task_id]

        for t_id in dag.tasks:
            dag.tasks[t_id].rank_u = get_rank(t_id)


class ContextAwareSlack:
    """
    Computes Context-Aware Slack (CAS) and task urgency U_i^a (Eq. 8, 9, 10).
    Integrates historical execution, real-time workload, and forward-predictive
    least completion time (PLCT).
    """

    @staticmethod
    def calculate_plct(dag: DAGApplication, PEs: List[ProcessingElement], current_time: float, avg_comm: float = 5.0) -> float:
        unfinished = dag.get_unfinished_tasks()
        if not unfinished:
            return 0.0

        sorted_tasks = sorted(unfinished, key=lambda t: t.rank_u, reverse=True)
        pred_finish: Dict[int, float] = {}
        for t in dag.tasks.values():
            if t.status == 'completed' and t.finish_time is not None:
                pred_finish[t.task_id] = t.finish_time

        pe_virt_avail = {pe.pe_id: pe.earliest_available_time(current_time) for pe in PEs}

        eft_values = []
        for task in sorted_tasks:
            ert = current_time
            if task.predecessors:
                pred_times = [pred_finish.get(p, current_time) + avg_comm for p in task.predecessors]
                ert = max(max(pred_times), current_time)

            min_eft = float('inf')
            best_pe_id = None
            for pe in PEs:
                if pe.pe_type in task.execution_times:
                    w_ik = task.execution_times[pe.pe_type]
                    est = max(pe_virt_avail[pe.pe_id], ert)
                    eft = est + w_ik
                    if eft < min_eft:
                        min_eft = eft
                        best_pe_id = pe.pe_id

            if best_pe_id is not None:
                pe_virt_avail[best_pe_id] = min_eft
                pred_finish[task.task_id] = min_eft
                eft_values.append(min_eft)
            else:
                pred_finish[task.task_id] = current_time + 100.0
                eft_values.append(current_time + 100.0)

        max_eft = max(eft_values) if eft_values else current_time
        plct = max_eft - current_time
        return max(0.0, plct)

    @classmethod
    def compute_cas(cls, dag: DAGApplication, task: TaskNode, PEs: List[ProcessingElement], current_time: float) -> Tuple[float, float]:
        trem = dag.absolute_deadline - current_time
        plct = cls.calculate_plct(dag, PEs, current_time)
        cas = trem - plct
        return cas, trem


class TopsisAdaptivePriority:
    """
    Entropy-Weight-Method (EWM) + TOPSIS for balancing Task Criticality and Urgency (Eq. 11, 12, 13).
    """

    @staticmethod
    def calculate_priorities(ready_tasks: List[Tuple[TaskNode, DAGApplication, float]]) -> List[Tuple[TaskNode, float]]:
        N = len(ready_tasks)
        if N == 0:
            return []
        if N == 1:
            return [(ready_tasks[0][0], 1.0)]

        raw_matrix = np.zeros((N, 2), dtype=float)
        for i, (task, dag, cas_val) in enumerate(ready_tasks):
            raw_matrix[i, 0] = float(dag.criticality)
            raw_matrix[i, 1] = -float(cas_val)

        norm_matrix = np.zeros_like(raw_matrix)
        for m in range(2):
            col_min = np.min(raw_matrix[:, m])
            col_max = np.max(raw_matrix[:, m])
            diff = col_max - col_min
            if diff > 1e-9:
                norm_matrix[:, m] = (raw_matrix[:, m] - col_min) / diff
            else:
                norm_matrix[:, m] = 1.0

        eps = 1e-12
        weights = np.array([0.5, 0.5])
        if N > 1:
            entropies = np.zeros(2)
            col_sums = np.sum(norm_matrix, axis=0)
            for m in range(2):
                if col_sums[m] > 1e-9:
                    p = norm_matrix[:, m] / col_sums[m]
                    p = np.where(p > eps, p, eps)
                    entropies[m] = -np.sum(p * np.log(p)) / np.log(N)
                else:
                    entropies[m] = 1.0

            d = 1.0 - entropies
            sum_d = np.sum(d)
            if sum_d > 1e-9:
                weights = d / sum_d
            else:
                weights = np.array([0.5, 0.5])

        weighted_matrix = norm_matrix * weights
        Z_plus = np.max(weighted_matrix, axis=0)
        Z_minus = np.min(weighted_matrix, axis=0)

        sim_scores = []
        for i in range(N):
            dist_plus = math.sqrt(np.sum((weighted_matrix[i] - Z_plus) ** 2))
            dist_minus = math.sqrt(np.sum((weighted_matrix[i] - Z_minus) ** 2))
            total_dist = dist_plus + dist_minus
            sim = dist_minus / total_dist if total_dist > 1e-9 else 0.5
            sim_scores.append((ready_tasks[i][0], float(sim)))

        sim_scores.sort(key=lambda x: x[1], reverse=True)
        return sim_scores
