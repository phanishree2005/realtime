"""
DAG Application Model for Automotive Cyber-Physical Systems.
Represents an intelligent application DAG G^a = (V^a, E^a) with mixed criticality,
arrival times, deadlines, and inter-task dependencies.
"""

from typing import Dict, List, Optional, Tuple, Set
import copy
from .task import Task, TaskStatus
from .platform import HeterogeneousPlatform


class DAGApplication:
    """
    Application model: A_a = (G^a, C^a, W^a, C_comm^a, T_arr^a, T_rd^a)
    """
    def __init__(
        self,
        app_id: int,
        dag_type: str,
        tasks: Dict[int, Task],
        edges: List[Tuple[int, int]],
        edge_costs: Optional[Dict[Tuple[int, int], float]] = None,
        criticality: int = 1,
        arrival_time: float = 0.0,
        relative_deadline: float = 400.0
    ):
        self.app_id = app_id
        self.dag_type = dag_type
        self.tasks: Dict[int, Task] = tasks
        self.edges: List[Tuple[int, int]] = edges
        self.edge_costs: Dict[Tuple[int, int], float] = edge_costs if edge_costs is not None else {}
        self.criticality: int = criticality
        self.arrival_time: float = arrival_time
        self.relative_deadline: float = relative_deadline
        self.absolute_deadline: float = arrival_time + relative_deadline

        # Connect predecessors and successors in tasks
        for src, dst in edges:
            if dst not in self.tasks[src].successors:
                self.tasks[src].successors.append(dst)
            if src not in self.tasks[dst].predecessors:
                self.tasks[dst].predecessors.append(src)

        # Queues
        self.ordered_tasks: List[int] = []    # Q_o^a
        self.unfinished_tasks: List[int] = [] # Q_u^a
        self.ready_tasks: List[int] = []      # Q_r^a
        self.finished_tasks: List[int] = []   # Q_f^a

        # Completion tracking
        self.completion_time: Optional[float] = None
        self.is_completed: bool = False
        self.met_deadline: Optional[bool] = None

    def get_task(self, task_id: int) -> Task:
        return self.tasks[task_id]

    def get_entry_tasks(self) -> List[int]:
        return [tid for tid, t in self.tasks.items() if len(t.predecessors) == 0]

    def get_exit_tasks(self) -> List[int]:
        return [tid for tid, t in self.tasks.items() if len(t.successors) == 0]

    def topological_sort(self) -> List[int]:
        """Returns topological ordering of task IDs."""
        in_degree = {tid: len(t.predecessors) for tid, t in self.tasks.items()}
        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        order = []
        while queue:
            curr = queue.pop(0)
            order.append(curr)
            for succ in self.tasks[curr].successors:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)
        return order

    def compute_heft_ranks(self, platform: HeterogeneousPlatform) -> List[int]:
        """
        Step 2: Task sorting within application using HEFT upward rank (Eq. 7):
        rank_u(v_i) = bar{w}_i + max_{v_j in succ(v_i)} (rank_u(v_j) + bar{c}_{i,j})
        """
        # 1. Compute average execution cost bar{w}_i for each task
        avg_w: Dict[int, float] = {}
        for tid, task in self.tasks.items():
            valid_costs = []
            for pe in platform.pes:
                cost = task.get_wcet(pe.pe_type)
                if cost < float("inf"):
                    valid_costs.append(cost)
            avg_w[tid] = sum(valid_costs) / len(valid_costs) if valid_costs else 100.0

        # 2. Compute upward rank in reverse topological order
        topo_order = self.topological_sort()
        rev_topo = list(reversed(topo_order))

        for tid in rev_topo:
            task = self.tasks[tid]
            if not task.successors:
                task.upward_rank = avg_w[tid]
            else:
                max_succ_rank = 0.0
                for succ_id in task.successors:
                    comm_cost = self.edge_costs.get((tid, succ_id), platform.default_inter_pe_comm)
                    succ_rank = self.tasks[succ_id].upward_rank + comm_cost
                    if succ_rank > max_succ_rank:
                        max_succ_rank = succ_rank
                task.upward_rank = avg_w[tid] + max_succ_rank

        # Sort tasks descending by upward rank -> Q_o^a
        self.ordered_tasks = sorted(
            topo_order,
            key=lambda tid: self.tasks[tid].upward_rank,
            reverse=True
        )
        self.unfinished_tasks = list(self.ordered_tasks)
        return self.ordered_tasks

    def compute_aeft_ranks(self, platform: HeterogeneousPlatform) -> List[int]:
        """
        Ablation: Intra-application ranking using AEFT's Improved Optimistic Cost Table (IOCT).
        Computes predictive optimistic execution cost considering fastest PE path.
        """
        rev_topo = list(reversed(self.topological_sort()))
        ioct: Dict[Tuple[int, int], float] = {}  # (task_id, pe_id) -> cost

        for tid in rev_topo:
            task = self.tasks[tid]
            for pe in platform.pes:
                w_cost = task.get_wcet(pe.pe_type)
                if w_cost == float("inf"):
                    ioct[(tid, pe.pe_id)] = float("inf")
                    continue

                if not task.successors:
                    ioct[(tid, pe.pe_id)] = w_cost
                else:
                    max_succ_cost = 0.0
                    for succ_id in task.successors:
                        edge_comm = self.edge_costs.get((tid, succ_id), platform.default_inter_pe_comm)
                        # Best PE for successor considering communication
                        min_succ = float("inf")
                        for succ_pe in platform.pes:
                            succ_w = self.tasks[succ_id].get_wcet(succ_pe.pe_type)
                            if succ_w == float("inf"):
                                continue
                            c_cost = 0.0 if pe.pe_id == succ_pe.pe_id else edge_comm
                            cost = ioct.get((succ_id, succ_pe.pe_id), 0.0) + c_cost
                            if cost < min_succ:
                                min_succ = cost
                        if min_succ != float("inf") and min_succ > max_succ_cost:
                            max_succ_cost = min_succ
                    ioct[(tid, pe.pe_id)] = w_cost + max_succ_cost

        # Rank by average valid IOCT
        for tid in self.tasks:
            valid_vals = [ioct[(tid, pe.pe_id)] for pe in platform.pes if ioct[(tid, pe.pe_id)] < float("inf")]
            self.tasks[tid].upward_rank = sum(valid_vals) / len(valid_vals) if valid_vals else 0.0

        topo_order = self.topological_sort()
        self.ordered_tasks = sorted(
            topo_order,
            key=lambda tid: self.tasks[tid].upward_rank,
            reverse=True
        )
        self.unfinished_tasks = list(self.ordered_tasks)
        return self.ordered_tasks

    def compute_subdeadlines(self, platform: HeterogeneousPlatform):
        """
        Computes task subdeadlines based on DAG critical path and topological level.
        Used by TD_ADS, ULS, and ablation studies.
        """
        # Compute avg execution times
        avg_w: Dict[int, float] = {}
        for tid, task in self.tasks.items():
            valid_costs = [
                task.get_wcet(pe.pe_type) for pe in platform.pes
                if task.get_wcet(pe.pe_type) < float("inf")
            ]
            avg_w[tid] = sum(valid_costs) / len(valid_costs) if valid_costs else 10.0

        topo = self.topological_sort()
        # Top-level (earliest possible start time along critical path)
        top_level: Dict[int, float] = {tid: 0.0 for tid in topo}
        for tid in topo:
            for succ in self.tasks[tid].successors:
                comm = self.edge_costs.get((tid, succ), platform.default_inter_pe_comm)
                candidate = top_level[tid] + avg_w[tid] + comm
                if candidate > top_level[succ]:
                    top_level[succ] = candidate

        # Critical path length
        critical_path = max(top_level[tid] + avg_w[tid] for tid in topo)
        critical_path = max(1.0, critical_path)

        # Distribute deadline proportionally
        for tid, task in self.tasks.items():
            task.topological_level = top_level[tid]
            fraction = (top_level[tid] + avg_w[tid]) / critical_path
            task.subdeadline = self.arrival_time + fraction * self.relative_deadline

    def update_ready_tasks(self) -> List[int]:
        """
        Identifies ready tasks in Q_u^a whose predecessors have all finished.
        """
        finished_set = set(self.finished_tasks)
        ready = []
        for tid in self.unfinished_tasks:
            task = self.tasks[tid]
            if task.status in (TaskStatus.RUNNING, TaskStatus.FINISHED):
                continue
            # Check if all predecessors are finished
            if all(pred in finished_set for pred in task.predecessors):
                task.status = TaskStatus.READY
                ready.append(tid)
        self.ready_tasks = ready
        return ready

    def mark_task_finished(self, task_id: int, finish_time: float):
        """Record task completion and check application completion."""
        task = self.tasks[task_id]
        task.status = TaskStatus.FINISHED
        task.finish_time = finish_time
        if task_id in self.unfinished_tasks:
            self.unfinished_tasks.remove(task_id)
        if task_id not in self.finished_tasks:
            self.finished_tasks.append(task_id)
        if task_id in self.ready_tasks:
            self.ready_tasks.remove(task_id)

        # Check if entire application is finished
        if len(self.unfinished_tasks) == 0:
            self.is_completed = True
            self.completion_time = max(
                (t.finish_time for t in self.tasks.values() if t.finish_time is not None),
                default=finish_time
            )
            self.met_deadline = self.completion_time <= self.absolute_deadline

    def clone(self) -> "DAGApplication":
        """Deep clone application with fresh dynamic state."""
        new_tasks = {tid: t.clone() for tid, t in self.tasks.items()}
        app = DAGApplication(
            app_id=self.app_id,
            dag_type=self.dag_type,
            tasks=new_tasks,
            edges=list(self.edges),
            edge_costs=dict(self.edge_costs),
            criticality=self.criticality,
            arrival_time=self.arrival_time,
            relative_deadline=self.relative_deadline
        )
        app.ordered_tasks = list(self.ordered_tasks)
        app.unfinished_tasks = list(self.ordered_tasks)
        return app
