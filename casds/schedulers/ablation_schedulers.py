"""
Ablation Schedulers (Section IV-C & Table II).
Implements various configurations for the ablation study:
- CAS-driven Urgency variations:
  1. T_rem only (Eq. 8)
  2. T-Ddl (subdeadline in [19], [21])
  3. Urgency in [20] (HetSched urgency)
  4. Context-Aware Slack (CAS)
- Priority variations:
  1. w/o Criticality C_i^a (urgency only)
  2. Multi-key sorting (radix: criticality first, then urgency)
  3. Adaptive Priority (TOPSIS + Entropy)
- Intra-application ranking variations:
  1. with HEFT
  2. with AEFT (IOCT)
"""

from typing import List, Tuple, Optional
import numpy as np

from .casds_scheduler import CASDSScheduler
from ..models.dag import DAGApplication
from ..models.task import Task, TaskStatus
from ..models.platform import HeterogeneousPlatform, ProcessingElement


class AblationScheduler(CASDSScheduler):
    """
    Parametric scheduler allowing isolated toggles of urgency metric,
    priority mechanism, and intra-app ranking.
    """
    def __init__(
        self,
        urgency_mode: str = "CAS",       # "Trem", "T-Ddl", "HetSched", "CAS"
        priority_mode: str = "Adaptive", # "w/o_Crit", "Multikey", "Adaptive"
        intra_ranking: str = "HEFT"      # "HEFT", "AEFT"
    ):
        super().__init__(intra_ranking=intra_ranking)
        self.urgency_mode = urgency_mode
        self.priority_mode = priority_mode
        self.name = f"Ablation_{urgency_mode}_{priority_mode}_{intra_ranking}"

    def sort_within_application(self, app: DAGApplication, platform: HeterogeneousPlatform):
        if self.intra_ranking == "AEFT":
            app.compute_aeft_ranks(platform)
        else:
            app.compute_heft_ranks(platform)

        if self.urgency_mode in ("T-Ddl", "HetSched"):
            app.compute_subdeadlines(platform)

    def calculate_urgency(
        self,
        app: DAGApplication,
        task: Task,
        current_time: float,
        platform: HeterogeneousPlatform
    ) -> float:
        """Calculates urgency based on selected ablation mode."""
        t_rem = app.arrival_time + app.relative_deadline - current_time

        if self.urgency_mode == "Trem":
            # 1. T_rem in (8): historical remaining time only
            return t_rem

        elif self.urgency_mode == "T-Ddl":
            # 2. T-Ddl in [19], [21]: task subdeadline margin
            return task.subdeadline - current_time

        elif self.urgency_mode == "HetSched":
            # 3. Urgency in [20]: remaining time normalized by static execution cost
            valid_costs = [
                task.get_wcet(pe.pe_type) for pe in platform.pes
                if task.get_wcet(pe.pe_type) < float("inf")
            ]
            avg_w = sum(valid_costs) / len(valid_costs) if valid_costs else 10.0
            return t_rem / max(1.0, avg_w)

        else:
            # 4. Context-Aware Slack (CAS, Eq. 10)
            cas, _ = self.compute_context_aware_slack(app, task, current_time, platform)
            return cas

    def schedule(
        self,
        current_time: float,
        platform: HeterogeneousPlatform
    ) -> List[Tuple[int, int, int, float, float]]:
        ready_candidates: List[Tuple[DAGApplication, Task, float, float]] = []
        for app in list(self.active_applications.values()):
            if app.is_completed:
                continue
            ready_tids = app.update_ready_tasks()
            for tid in ready_tids:
                task = app.tasks[tid]
                urgency = self.calculate_urgency(app, task, current_time, platform)
                t_rem = app.arrival_time + app.relative_deadline - current_time
                ready_candidates.append((app, task, urgency, t_rem))

        if not ready_candidates:
            return []

        # Priority calculation based on priority_mode
        if self.priority_mode == "w/o_Crit":
            # Sort solely by urgency ascending (smaller urgency value first)
            ready_candidates.sort(key=lambda x: x[2])
            priority_queue = ready_candidates

        elif self.priority_mode == "Multikey":
            # Radix: Criticality descending, then urgency ascending
            ready_candidates.sort(key=lambda x: (-x[0].criticality, x[2]))
            priority_queue = ready_candidates

        else:
            # Adaptive Priority via TOPSIS + Entropy
            priorities = self.compute_topsis_priorities(ready_candidates)
            sorted_indices = sorted(range(len(ready_candidates)), key=lambda i: priorities[i], reverse=True)
            priority_queue = [ready_candidates[i] for i in sorted_indices]

        # Allocation step
        scheduled = []
        for app, task, _, _ in priority_queue:
            decision = self.allocate_task_eft(app, task, current_time, platform)
            if decision is not None:
                scheduled.append(decision)

        return scheduled

