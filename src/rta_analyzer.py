"""
Exact Response Time Analysis (RTA) Engine for Fixed-Priority Preemptive Scheduling (FPPS).
Calculates exact worst-case response times (WCRT) and theoretical schedulability ratios.
"""

import math
from typing import Dict, List, Any, Tuple, Optional
from .task_model import Task, TaskSet


class RTAAnalyzer:
    """
    Computes exact Response Time Analysis for independent periodic tasks
    under Fixed-Priority Preemptive Scheduling (Joseph & Pandya / Audsley).
    """

    def __init__(self, max_iterations: int = 500):
        self.max_iterations = max_iterations

    def compute_task_wcrt(
        self,
        task: Task,
        higher_priority_tasks: List[Task],
        same_priority_tasks: Optional[List[Task]] = None
    ) -> Tuple[float, bool]:
        """
        Calculates worst-case response time R_i for a single task.
        Accounts for higher-priority preemption and same-priority FIFO queueing interference.
        Returns: (R_i, is_schedulable)
        """
        sp_interference = sum(t.wcet for t in (same_priority_tasks or []))
        r_current = task.wcet + sp_interference
        iteration = 0

        while iteration < self.max_iterations:
            interference = sp_interference
            for hp_task in higher_priority_tasks:
                if hp_task.period > 0:
                    num_releases = math.ceil(r_current / hp_task.period)
                    interference += num_releases * hp_task.wcet

            r_next = task.wcet + interference

            if r_next > task.deadline:
                return (r_next, False)  # Deadline missed

            if abs(r_next - r_current) < 1e-9:
                return (r_next, r_next <= task.deadline)  # Converged

            r_current = r_next
            iteration += 1

        return (r_current, r_current <= task.deadline)

    def analyze(self, task_set: TaskSet) -> Dict[str, Any]:
        """
        Performs full RTA across all tasks in task_set.
        Returns detailed per-task metrics and overall schedulability status.
        """
        if len(task_set) == 0:
            return {
                "is_schedulable": True,
                "schedulable_tasks_count": 0,
                "total_tasks": 0,
                "per_task_results": []
            }

        all_schedulable = True
        per_task_results = []

        for task in task_set.tasks:
            hp_tasks = [t for t in task_set.tasks if t.priority > task.priority]
            sp_tasks = [t for t in task_set.tasks if t.priority == task.priority and t.task_id != task.task_id]

            wcrt, is_sched = self.compute_task_wcrt(task, hp_tasks, sp_tasks)
            margin = task.deadline - wcrt

            if not is_sched:
                all_schedulable = False

            per_task_results.append({
                "task_id": task.task_id,
                "name": task.name,
                "priority": task.priority,
                "period_ms": task.period,
                "deadline_ms": task.deadline,
                "wcet_ms": task.wcet,
                "wcrt_ms": wcrt,
                "schedulable": is_sched,
                "margin_ms": margin,
                "response_ratio": wcrt / task.deadline if task.deadline > 0 else float("inf")
            })

        sched_count = sum(1 for r in per_task_results if r["schedulable"])
        return {
            "is_schedulable": all_schedulable,
            "schedulable_tasks_count": sched_count,
            "total_tasks": len(task_set),
            "schedulability_fraction": sched_count / len(task_set) if len(task_set) > 0 else 1.0,
            "total_utilization": task_set.total_utilization,
            "per_task_results": per_task_results
        }
