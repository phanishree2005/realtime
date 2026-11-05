"""
Context-Aware-Slack-Driven Dynamic Scheduling (CASDS) Algorithm.
Implements the novel scheduling method proposed in Section III:
- Step 1: Ready tasks searching
- Step 2: Task sorting within application (HEFT upward rank)
- Step 3: Global tasks sorting (Context-Aware Slack + TOPSIS Entropy Weight Adaptive Priority)
- Step 4: Task-PE Allocation (Earliest Finish Time with immediate availability check)
"""

import math
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from .base_scheduler import BaseScheduler
from ..models.dag import DAGApplication
from ..models.task import Task, TaskStatus
from ..models.platform import HeterogeneousPlatform, ProcessingElement


class CASDSScheduler(BaseScheduler):
    """
    Proposed CASDS Dynamic Scheduler.
    """
    def __init__(self, intra_ranking: str = "HEFT"):
        super().__init__("CASDS")
        self.intra_ranking = intra_ranking  # "HEFT" or "AEFT" (for ablation study)

    def sort_within_application(self, app: DAGApplication, platform: HeterogeneousPlatform):
        """Step 2: Tasks sorting within application."""
        if self.intra_ranking == "AEFT":
            app.compute_aeft_ranks(platform)
        else:
            app.compute_heft_ranks(platform)

    def compute_plct(
        self,
        app: DAGApplication,
        current_time: float,
        platform: HeterogeneousPlatform
    ) -> float:
        """
        Calculates Predictive Least Completion Time PLCT(Q_u^a, P_t^s) (Eq. 9).
        Predicts future execution status of unfinished tasks Q_u^a taking into account
        current PE availability Lambda(p_k, P_t^s) and predecessor dependencies.
        """
        if not app.unfinished_tasks:
            return 0.0

        # Map of predicted finish times for tasks in this application
        pred_ft: Dict[int, float] = {}

        # 1. Finished tasks have known actual finish time
        for tid in app.finished_tasks:
            pred_ft[tid] = app.tasks[tid].finish_time if app.tasks[tid].finish_time is not None else current_time

        # 2. Running tasks have scheduled finish time
        for tid, task in app.tasks.items():
            if task.status == TaskStatus.RUNNING and task.allocated_pe is not None:
                pe = platform.get_pe(task.allocated_pe)
                pred_ft[tid] = pe.busy_until

        # 3. Predict completion time forward through unfinished tasks in topological order
        topo_order = app.topological_sort()
        unfinished_set = set(app.unfinished_tasks)

        # Virtual PE availability state for forward prediction
        virtual_avail = {pe.pe_id: platform.get_available_time(pe.pe_id, current_time) for pe in platform.pes}

        max_app_eft = current_time

        for tid in topo_order:
            if tid not in unfinished_set:
                continue

            task = app.tasks[tid]
            if task.status == TaskStatus.RUNNING:
                continue

            # Find best capable PE for task
            best_eft = float("inf")
            best_pe_id = -1

            for pe in platform.pes:
                wcet = task.get_wcet(pe.pe_type)
                if wcet == float("inf"):
                    continue

                # ERT
                ert = current_time
                for pred_id in task.predecessors:
                    p_ft = pred_ft.get(pred_id, current_time)
                    pred_task = app.tasks[pred_id]
                    comm = 0.0 if pred_task.allocated_pe == pe.pe_id else app.edge_costs.get((pred_id, tid), platform.default_inter_pe_comm)
                    candidate = max(p_ft, current_time) + comm
                    if candidate > ert:
                        ert = candidate

                est = max(virtual_avail[pe.pe_id], ert)
                eft = est + wcet

                if eft < best_eft:
                    best_eft = eft
                    best_pe_id = pe.pe_id

            if best_pe_id != -1 and best_eft < float("inf"):
                pred_ft[tid] = best_eft
                # Update virtual availability of chosen PE
                virtual_avail[best_pe_id] = best_eft
                if best_eft > max_app_eft:
                    max_app_eft = best_eft
            else:
                pred_ft[tid] = current_time + 100.0
                if pred_ft[tid] > max_app_eft:
                    max_app_eft = pred_ft[tid]

        # PLCT is predictive completion span from current time t
        plct = max(0.0, max_app_eft - current_time)
        return plct

    def compute_context_aware_slack(
        self,
        app: DAGApplication,
        task: Task,
        current_time: float,
        platform: HeterogeneousPlatform
    ) -> Tuple[float, float]:
        """
        Computes Context-Aware Slack CAS(v_i^a, P_t^s) (Eq. 8, 9, 10).
        Returns (CAS, T_rem).
        """
        # Eq. (8): T_rem = T_arr + T_rd - t
        t_rem = app.arrival_time + app.relative_deadline - current_time

        # Eq. (9): PLCT
        plct = self.compute_plct(app, current_time, platform)

        # Eq. (10): CAS = T_rem - PLCT
        cas = t_rem - plct
        return cas, t_rem

    def compute_topsis_priorities(
        self,
        ready_tasks: List[Tuple[DAGApplication, Task, float, float]]
    ) -> List[float]:
        """
        Step 3.2: Adaptive Priority Combining Criticality and Urgency via TOPSIS & Entropy (Eq. 11-13).
        ready_tasks is a list of (app, task, CAS, T_rem).
        Returns priority score P_i^a for each task.
        """
        N = len(ready_tasks)
        if N == 0:
            return []
        if N == 1:
            return [1.0]

        # Attributes: m=0: Criticality C_i^a, m=1: Urgency -U_i^a
        # Note: smaller CAS means more urgent, so -CAS means higher is more urgent.
        # If CAS < 0, deadline is hopeless, handled with penalty.
        X = np.zeros((N, 2), dtype=np.float64)
        for i, (app, task, cas, _) in enumerate(ready_tasks):
            X[i, 0] = float(app.criticality)
            # Opposite number for urgency to ensure aligned trend
            # Hopeless tasks (cas < 0) get lower base score
            if cas >= 0:
                X[i, 1] = -cas
            else:
                X[i, 1] = -1000.0 + cas

        # Min-Max Normalization (Eq. 11)
        X_norm = np.zeros_like(X)
        for m in range(2):
            col_min = np.min(X[:, m])
            col_max = np.max(X[:, m])
            if col_max - col_min > 1e-9:
                X_norm[:, m] = (X[:, m] - col_min) / (col_max - col_min)
            else:
                X_norm[:, m] = 1.0 / N

        # Entropy Weight Method (Eq. 12)
        # p_{n,m} = x_{n,m}^{norm} / sum(x_{n,m}^{norm})
        col_sums = np.sum(X_norm, axis=0)
        col_sums[col_sums == 0] = 1e-12
        P_prob = X_norm / col_sums

        # Entropy e_m = -1/ln(N) * sum(p * ln(p))
        ln_N = math.log(N) if N > 1 else 1.0
        entropy = np.zeros(2)
        for m in range(2):
            p = P_prob[:, m]
            p_pos = p[p > 0]
            if len(p_pos) > 0 and ln_N > 0:
                entropy[m] = -np.sum(p_pos * np.log(p_pos)) / ln_N
            else:
                entropy[m] = 1.0
            entropy[m] = min(1.0, max(0.0, entropy[m]))

        # Weight omega_m = (1 - e_m) / sum(1 - e_k)
        divergence = 1.0 - entropy
        div_sum = np.sum(divergence)
        if div_sum > 1e-9:
            weights = divergence / div_sum
        else:
            weights = np.array([0.5, 0.5])

        # Weighted attribute matrix X_wt = omega_m * x_{n,m}^{norm}
        X_wt = X_norm * weights

        # Ideal Best and Worst Solutions (Eq. 13)
        Z_plus = np.max(X_wt, axis=0)
        Z_minus = np.min(X_wt, axis=0)

        # Distances to Best and Worst
        Sim_plus = np.sqrt(np.sum((X_wt - Z_plus) ** 2, axis=1))
        Sim_minus = np.sqrt(np.sum((X_wt - Z_minus) ** 2, axis=1))

        # Relative Closeness Score Sim_n = Sim_minus / (Sim_minus + Sim_plus)
        denom = Sim_minus + Sim_plus
        denom[denom == 0] = 1e-12
        Sim = Sim_minus / denom

        # Penalize hopeless tasks (cas < 0) so they don't precede feasible tasks
        for i, (_, _, cas, _) in enumerate(ready_tasks):
            if cas < 0:
                Sim[i] = Sim[i] * 0.1 - 1.0

        return Sim.tolist()

    def schedule(
        self,
        current_time: float,
        platform: HeterogeneousPlatform
    ) -> List[Tuple[int, int, int, float, float]]:
        """
        Executes Steps 1, 3, 4 of CASDS algorithm.
        Returns list of (app_id, task_id, pe_id, start_time, duration).
        """
        # Step 1: Ready tasks searching across active applications
        ready_candidates: List[Tuple[DAGApplication, Task, float, float]] = []
        for app in list(self.active_applications.values()):
            if app.is_completed:
                continue
            ready_tids = app.update_ready_tasks()
            for tid in ready_tids:
                task = app.tasks[tid]
                cas, t_rem = self.compute_context_aware_slack(app, task, current_time, platform)
                ready_candidates.append((app, task, cas, t_rem))

        if not ready_candidates:
            return []

        # Step 3: Global tasks sorting (TOPSIS Entropy Weight Adaptive Priority)
        priorities = self.compute_topsis_priorities(ready_candidates)
        # Pair with index and sort descending
        sorted_indices = sorted(range(len(ready_candidates)), key=lambda i: priorities[i], reverse=True)
        priority_queue = [ready_candidates[i] for i in sorted_indices]

        # Step 4: Task-PE Allocation
        scheduled_decisions = []
        # Track currently occupied PEs in this time step
        scheduled_decisions = []
        for app, task, _, _ in priority_queue:
            decision = self.allocate_task_eft(app, task, current_time, platform)
            if decision is not None:
                scheduled_decisions.append(decision)

        return scheduled_decisions
