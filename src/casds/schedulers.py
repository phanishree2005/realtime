"""
Dynamic Scheduling Algorithms for Automotive Cyber-Physical Systems (ACPS)
Implements:
1. CASDS (Proposed - Context-Aware-Slack-Driven Dynamic Scheduling)
2. HetSched (Amarnath et al., IEEE CAL 2021)
3. ADS (Xie et al., IEEE TVT 2017)
4. ASDYS (Bai et al., IEEE TII 2021)
5. TD_ADS (Zhao et al., IEEE Access 2024)
6. ULS (Cai et al., IEEE TSC 2025)
"""

from typing import List, Dict, Optional, Tuple, Set
import numpy as np
from src.casds.core import TaskNode, DAGApplication, ProcessingElement, ContextAwareSlack, TopsisAdaptivePriority


class BaseScheduler:
    def __init__(self, name: str):
        self.name = name

    def schedule_step(
        self,
        current_time: float,
        active_dags: Dict[int, DAGApplication],
        pes: List[ProcessingElement],
        avg_comm: float = 5.0
    ) -> List[Tuple[TaskNode, ProcessingElement]]:
        """
        Takes current active DAGs and PEs, returns a list of (task, pe) allocations to dispatch at current_time.
        """
        raise NotImplementedError


class CASDSScheduler(BaseScheduler):
    """
    Proposed CASDS Algorithm:
    - Context-Aware Slack (CAS = Trem - PLCT)
    - Deferral of negative slack tasks to prevent domino effect
    - Entropy-Weight Method + TOPSIS for adaptive criticality/urgency balancing
    - Real-time Earliest Finish Time (EFT) allocation
    """
    def __init__(self):
        super().__init__("CASDS")

    def schedule_step(
        self,
        current_time: float,
        active_dags: Dict[int, DAGApplication],
        pes: List[ProcessingElement],
        avg_comm: float = 5.0
    ) -> List[Tuple[TaskNode, ProcessingElement]]:
        # Step 1: Collect ready tasks across all active DAGs
        ready_tasks_with_meta: List[Tuple[TaskNode, DAGApplication, float]] = []
        deferred_tasks: List[Tuple[TaskNode, DAGApplication, float]] = []

        for dag_id, dag in active_dags.items():
            if dag.is_finished():
                continue
            ready = dag.get_ready_tasks()
            for task in ready:
                cas_val, trem = ContextAwareSlack.compute_cas(dag, task, pes, current_time)
                if cas_val < 0:
                    # Urgent failure condition: cannot meet deadline, defer execution to save other tasks (Section III-C1)
                    deferred_tasks.append((task, dag, cas_val))
                else:
                    ready_tasks_with_meta.append((task, dag, cas_val))

        # Step 2 & 3: Global task sorting via EWM-TOPSIS
        prioritized_tasks: List[TaskNode] = []
        if ready_tasks_with_meta:
            sorted_topsis = TopsisAdaptivePriority.calculate_priorities(ready_tasks_with_meta)
            prioritized_tasks.extend([t for t, _ in sorted_topsis])

        # Deferred tasks placed at the end ordered by smallest negative slack
        if deferred_tasks:
            deferred_tasks.sort(key=lambda x: x[2], reverse=True)
            prioritized_tasks.extend([t for t, _, _ in deferred_tasks])

        # Step 4: Task-PE Allocation (Eq. 14 - 15)
        # Assign sequentially to PE giving earliest finish time
        dispatched: List[Tuple[TaskNode, ProcessingElement]] = []
        allocated_pes_this_step: Set[str] = set()

        for task in prioritized_tasks:
            dag = active_dags[task.dag_id]
            best_pe: Optional[ProcessingElement] = None
            min_eft = float('inf')

            for pe in pes:
                if pe.pe_id in allocated_pes_this_step:
                    continue  # Nonpreemptive: already assigned this exact timestep
                if pe.pe_type in task.execution_times:
                    w_ik = task.execution_times[pe.pe_type]
                    # Check predecessor communication
                    ert = current_time
                    if task.predecessors:
                        pred_finish_times = [
                            dag.tasks[p].finish_time if dag.tasks[p].status == 'completed' else current_time
                            for p in task.predecessors
                        ]
                        # If predecessor on same PE, comm = 0, else avg_comm
                        pred_comms = [
                            0.0 if dag.tasks[p].allocated_pe == pe.pe_id else avg_comm
                            for p in task.predecessors
                        ]
                        ert = max([f + c for f, c in zip(pred_finish_times, pred_comms)] + [current_time])

                    est = max(pe.earliest_available_time(current_time), ert)
                    eft = est + w_ik
                    if eft < min_eft:
                        min_eft = eft
                        best_pe = pe

            # Nonpreemptive Constraint (Eq. 15): only dispatch if PE is currently free
            if best_pe is not None and best_pe.is_available(current_time):
                dispatched.append((task, best_pe))
                allocated_pes_this_step.add(best_pe.pe_id)

        return dispatched


class HetSchedScheduler(BaseScheduler):
    """
    Heterogeneity-Aware Scheduler (Amarnath et al. [20]):
    - Radix priority: Critical tasks strictly prioritized over non-critical tasks
    - Secondary sort: Static/approximate slack
    - Greedy EFT allocation
    """
    def __init__(self):
        super().__init__("HetSched")

    def schedule_step(
        self,
        current_time: float,
        active_dags: Dict[int, DAGApplication],
        pes: List[ProcessingElement],
        avg_comm: float = 5.0
    ) -> List[Tuple[TaskNode, ProcessingElement]]:
        ready_tasks: List[Tuple[TaskNode, DAGApplication]] = []
        for dag in active_dags.values():
            if not dag.is_finished():
                for t in dag.get_ready_tasks():
                    ready_tasks.append((t, dag))

        if not ready_tasks:
            return []

        # Sort: Primary = Criticality (descending), Secondary = Slack (ascending)
        def sort_key(item):
            task, dag = item
            # approx slack = remaining time - min execution time
            min_exec = min(task.execution_times.values()) if task.execution_times else 10.0
            slack = (dag.absolute_deadline - current_time) - min_exec
            return (dag.criticality, -slack)

        ready_tasks.sort(key=sort_key, reverse=True)

        dispatched = []
        allocated_pes_this_step: Set[str] = set()

        for task, dag in ready_tasks:
            best_pe = None
            min_eft = float('inf')
            for pe in pes:
                if pe.pe_id in allocated_pes_this_step:
                    continue
                if pe.pe_type in task.execution_times:
                    w = task.execution_times[pe.pe_type]
                    est = pe.earliest_available_time(current_time)
                    eft = est + w
                    if eft < min_eft:
                        min_eft = eft
                        best_pe = pe

            if best_pe is not None and best_pe.is_available(current_time):
                dispatched.append((task, best_pe))
                allocated_pes_this_step.add(best_pe.pe_id)

        return dispatched


class ADSScheduler(BaseScheduler):
    """
    Adaptive Dynamic Scheduling (Xie et al. [17]):
    - Uses upward rank value to sort tasks
    - Mode switching: when high-criticality task has impending deadline miss, suppresses low-criticality tasks
    """
    def __init__(self, mode_switch_threshold: float = 50.0):
        super().__init__("ADS")
        self.mode_switch_threshold = mode_switch_threshold

    def schedule_step(
        self,
        current_time: float,
        active_dags: Dict[int, DAGApplication],
        pes: List[ProcessingElement],
        avg_comm: float = 5.0
    ) -> List[Tuple[TaskNode, ProcessingElement]]:
        ready_tasks = []
        critical_in_danger = False

        for dag in active_dags.values():
            if not dag.is_finished():
                rem = dag.absolute_deadline - current_time
                if dag.criticality > 1 and rem < self.mode_switch_threshold:
                    critical_in_danger = True
                for t in dag.get_ready_tasks():
                    ready_tasks.append((t, dag))

        # Mode switch: if critical in danger, filter out low criticality tasks
        if critical_in_danger:
            ready_tasks = [item for item in ready_tasks if item[1].criticality > 1] or ready_tasks

        # Sort by upward rank descending
        ready_tasks.sort(key=lambda item: (item[1].criticality, item[0].rank_u), reverse=True)

        dispatched = []
        allocated_pes_this_step: Set[str] = set()

        for task, dag in ready_tasks:
            best_pe = None
            min_eft = float('inf')
            for pe in pes:
                if pe.pe_id in allocated_pes_this_step:
                    continue
                if pe.pe_type in task.execution_times:
                    eft = pe.earliest_available_time(current_time) + task.execution_times[pe.pe_type]
                    if eft < min_eft:
                        min_eft = eft
                        best_pe = pe

            if best_pe is not None and best_pe.is_available(current_time):
                dispatched.append((task, best_pe))
                allocated_pes_this_step.add(best_pe.pe_id)

        return dispatched


class ASDYSScheduler(BaseScheduler):
    """
    Active Strategy Dynamic Scheduling (Bai et al. [19]):
    - Traffic shaping admitting tasks proportionally to criticality
    """
    def __init__(self):
        super().__init__("ASDYS")

    def schedule_step(
        self,
        current_time: float,
        active_dags: Dict[int, DAGApplication],
        pes: List[ProcessingElement],
        avg_comm: float = 5.0
    ) -> List[Tuple[TaskNode, ProcessingElement]]:
        ready_tasks = []
        for dag in active_dags.values():
            if not dag.is_finished():
                for t in dag.get_ready_tasks():
                    ready_tasks.append((t, dag))

        # Traffic shaping: critical tasks get top priority, non-critical admitted if load allows
        busy_pes = sum(1 for p in pes if not p.is_available(current_time))
        load_ratio = busy_pes / len(pes) if pes else 0.0

        if load_ratio > 0.75:
            # Under high load, prioritize critical tasks exclusively
            filtered = [item for item in ready_tasks if item[1].criticality > 1]
            if filtered:
                ready_tasks = filtered

        ready_tasks.sort(key=lambda item: (item[1].criticality, item[0].rank_u), reverse=True)

        dispatched = []
        allocated_pes_this_step: Set[str] = set()
        for task, dag in ready_tasks:
            best_pe = None
            min_eft = float('inf')
            for pe in pes:
                if pe.pe_id in allocated_pes_this_step:
                    continue
                if pe.pe_type in task.execution_times:
                    eft = pe.earliest_available_time(current_time) + task.execution_times[pe.pe_type]
                    if eft < min_eft:
                        min_eft = eft
                        best_pe = pe
            if best_pe is not None and best_pe.is_available(current_time):
                dispatched.append((task, best_pe))
                allocated_pes_this_step.add(best_pe.pe_id)
        return dispatched


class TD_ADSScheduler(BaseScheduler):
    """
    Task-Degradation Aware ADS (Zhao et al. [21]):
    - Subdeadline based on task depth in DAG
    - Sorts tasks by dynamic subdeadline urgency
    """
    def __init__(self):
        super().__init__("TD_ADS")

    def schedule_step(
        self,
        current_time: float,
        active_dags: Dict[int, DAGApplication],
        pes: List[ProcessingElement],
        avg_comm: float = 5.0
    ) -> List[Tuple[TaskNode, ProcessingElement]]:
        ready_tasks = []
        for dag in active_dags.values():
            if not dag.is_finished():
                rem = dag.absolute_deadline - current_time
                for t in dag.get_ready_tasks():
                    # Subdeadline proportional to rank_u ratio
                    total_rank = max(t.rank_u for t in dag.tasks.values()) or 1.0
                    subdeadline = dag.arrival_time + dag.relative_deadline * (1.0 - t.rank_u / total_rank)
                    urgency = subdeadline - current_time
                    ready_tasks.append((t, dag, urgency))

        # Sort primarily by criticality, secondarily by subdeadline urgency (smaller is more urgent)
        ready_tasks.sort(key=lambda x: (x[1].criticality, -x[2]), reverse=True)

        dispatched = []
        allocated_pes_this_step: Set[str] = set()
        for task, dag, _ in ready_tasks:
            best_pe = None
            min_eft = float('inf')
            for pe in pes:
                if pe.pe_id in allocated_pes_this_step:
                    continue
                if pe.pe_type in task.execution_times:
                    eft = pe.earliest_available_time(current_time) + task.execution_times[pe.pe_type]
                    if eft < min_eft:
                        min_eft = eft
                        best_pe = pe
            if best_pe is not None and best_pe.is_available(current_time):
                dispatched.append((task, best_pe))
                allocated_pes_this_step.add(best_pe.pe_id)
        return dispatched


class ULSScheduler(BaseScheduler):
    """
    Urgency-Based List Scheduling (Cai et al. [22]):
    - Criticality-agnostic
    - Uses runtime urgency metric combining distributed subdeadline and topological level
    """
    def __init__(self):
        super().__init__("ULS")

    def schedule_step(
        self,
        current_time: float,
        active_dags: Dict[int, DAGApplication],
        pes: List[ProcessingElement],
        avg_comm: float = 5.0
    ) -> List[Tuple[TaskNode, ProcessingElement]]:
        ready_tasks = []
        for dag in active_dags.values():
            if not dag.is_finished():
                for t in dag.get_ready_tasks():
                    rem = dag.absolute_deadline - current_time
                    # ULS urgency metric
                    urgency_score = rem - (t.rank_u * 0.8)
                    ready_tasks.append((t, dag, urgency_score))

        # Criticality-agnostic sort: strictly lowest urgency_score first
        ready_tasks.sort(key=lambda x: x[2])

        dispatched = []
        allocated_pes_this_step: Set[str] = set()
        for task, dag, _ in ready_tasks:
            best_pe = None
            min_eft = float('inf')
            for pe in pes:
                if pe.pe_id in allocated_pes_this_step:
                    continue
                if pe.pe_type in task.execution_times:
                    eft = pe.earliest_available_time(current_time) + task.execution_times[pe.pe_type]
                    if eft < min_eft:
                        min_eft = eft
                        best_pe = pe
            if best_pe is not None and best_pe.is_available(current_time):
                dispatched.append((task, best_pe))
                allocated_pes_this_step.add(best_pe.pe_id)
        return dispatched
