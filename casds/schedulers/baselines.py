"""
Comparative Dynamic Schedulers from Literature:
1. ADS (Xie et al., 2017) [17]: Upward rank + system mode switching (normal vs critical mode)
2. ASDYS (Bai et al., 2021) [19]: Upward rank + active traffic shaping & task degradation
3. HetSched (Amarnath et al., 2021) [20]: Radix multi-key sorting (criticality first, then urgency)
4. TD_ADS (Zhao et al., 2024) [21]: Subdeadline-based dynamic urgency sorting
5. ULS (Cai et al., 2025) [22]: Criticality-agnostic urgency list scheduling (topological + subdeadline)
"""

from typing import Dict, List, Tuple, Optional, Any
from .base_scheduler import BaseScheduler
from ..models.dag import DAGApplication
from ..models.task import Task, TaskStatus
from ..models.platform import HeterogeneousPlatform, ProcessingElement


class ADSScheduler(BaseScheduler):
    """
    Adaptive Dynamic Scheduling (ADS) [17].
    Uses upward rank value to define task priority, and dynamically switches system
    mode to save high-criticality applications when impending deadline misses are observed.
    """
    def __init__(self):
        super().__init__("ADS")
        self.critical_mode: bool = False

    def sort_within_application(self, app: DAGApplication, platform: HeterogeneousPlatform):
        app.compute_heft_ranks(platform)

    def schedule(
        self,
        current_time: float,
        platform: HeterogeneousPlatform
    ) -> List[Tuple[int, int, int, float, float]]:
        # 1. Check for impending deadline misses among critical applications
        self.critical_mode = False
        for app in self.active_applications.values():
            if app.criticality >= 2 and not app.is_completed:
                t_rem = app.arrival_time + app.relative_deadline - current_time
                if 0 < t_rem < 150.0 and len(app.unfinished_tasks) > 0:
                    self.critical_mode = True
                    break

        # 2. Collect ready tasks
        ready_tasks: List[Tuple[DAGApplication, Task]] = []
        for app in list(self.active_applications.values()):
            if app.is_completed:
                continue
            # In critical mode, suspend low-criticality tasks
            if self.critical_mode and app.criticality < 2:
                continue

            ready_tids = app.update_ready_tasks()
            for tid in ready_tids:
                ready_tasks.append((app, app.tasks[tid]))

        if not ready_tasks:
            return []

        # 3. Sort globally by upward rank value (traditional approach)
        ready_tasks.sort(key=lambda item: item[1].upward_rank, reverse=True)

        # 4. Task-PE Allocation via EFT
        scheduled = []
        for app, task in ready_tasks:
            decision = self.allocate_task_eft(app, task, current_time, platform)
            if decision is not None:
                scheduled.append(decision)

        return scheduled


class ASDYSScheduler(BaseScheduler):
    """
    Active Strategy Dynamic Scheduling (ASDYS) [19].
    Uses traffic shaping and active degradation of low-criticality tasks under high workload.
    """
    def __init__(self):
        super().__init__("ASDYS")

    def sort_within_application(self, app: DAGApplication, platform: HeterogeneousPlatform):
        app.compute_heft_ranks(platform)

    def schedule(
        self,
        current_time: float,
        platform: HeterogeneousPlatform
    ) -> List[Tuple[int, int, int, float, float]]:
        active_critical = sum(1 for a in self.active_applications.values() if a.criticality >= 2 and not a.is_completed)
        total_active = len(self.active_applications)

        ready_tasks: List[Tuple[DAGApplication, Task]] = []
        for app in list(self.active_applications.values()):
            if app.is_completed:
                continue
            # Traffic shaping / active degradation under heavy load
            if app.criticality < 2 and active_critical >= 3 and total_active >= 8:
                continue

            ready_tids = app.update_ready_tasks()
            for tid in ready_tids:
                ready_tasks.append((app, app.tasks[tid]))

        if not ready_tasks:
            return []

        # Sort by upward rank
        ready_tasks.sort(key=lambda item: item[1].upward_rank, reverse=True)

        scheduled = []
        for app, task in ready_tasks:
            decision = self.allocate_task_eft(app, task, current_time, platform)
            if decision is not None:
                scheduled.append(decision)

        return scheduled


class HetSchedScheduler(BaseScheduler):
    """
    Heterogeneity-Aware Scheduler (HetSched) [20].
    Uses hybrid heterogeneous ranking with radix multi-key sorting:
    Primary key: Criticality (descending)
    Secondary key: Static-based urgency (subdeadline and execution time)
    """
    def __init__(self):
        super().__init__("HetSched")

    def sort_within_application(self, app: DAGApplication, platform: HeterogeneousPlatform):
        app.compute_heft_ranks(platform)
        app.compute_subdeadlines(platform)

    def schedule(
        self,
        current_time: float,
        platform: HeterogeneousPlatform
    ) -> List[Tuple[int, int, int, float, float]]:
        ready_tasks: List[Tuple[DAGApplication, Task, float]] = []
        for app in list(self.active_applications.values()):
            if app.is_completed:
                continue
            ready_tids = app.update_ready_tasks()
            for tid in ready_tids:
                task = app.tasks[tid]
                urgency = task.subdeadline - current_time
                ready_tasks.append((app, task, urgency))

        if not ready_tasks:
            return []

        # Radix multi-key sorting:
        # Key 1: Criticality descending (-crit)
        # Key 2: Urgency ascending (smaller urgency value first)
        ready_tasks.sort(key=lambda x: (-x[0].criticality, x[2]))

        scheduled = []
        for app, task, _ in ready_tasks:
            decision = self.allocate_task_eft(app, task, current_time, platform)
            if decision is not None:
                scheduled.append(decision)

        return scheduled


class TD_ADSScheduler(BaseScheduler):
    """
    Task-Degradation Aware ADS (TD_ADS) [21].
    Utilizes urgency based on task subdeadlines to sort tasks dynamically.
    """
    def __init__(self):
        super().__init__("TD_ADS")

    def sort_within_application(self, app: DAGApplication, platform: HeterogeneousPlatform):
        app.compute_heft_ranks(platform)
        app.compute_subdeadlines(platform)

    def schedule(
        self,
        current_time: float,
        platform: HeterogeneousPlatform
    ) -> List[Tuple[int, int, int, float, float]]:
        ready_tasks: List[Tuple[DAGApplication, Task, float]] = []
        for app in list(self.active_applications.values()):
            if app.is_completed:
                continue
            ready_tids = app.update_ready_tasks()
            for tid in ready_tids:
                task = app.tasks[tid]
                # Subdeadline-based urgency with criticality priority boost
                urgency = task.subdeadline - current_time
                if app.criticality >= 2:
                    urgency -= 30.0
                ready_tasks.append((app, task, urgency))

        if not ready_tasks:
            return []

        # Sort tasks by urgency (ascending: smaller urgency margin first)
        ready_tasks.sort(key=lambda x: x[2])

        scheduled = []
        for app, task, _ in ready_tasks:
            decision = self.allocate_task_eft(app, task, current_time, platform)
            if decision is not None:
                scheduled.append(decision)

        return scheduled


class ULSScheduler(BaseScheduler):
    """
    Urgency-based List Scheduling (ULS) [22].
    Criticality-agnostic list scheduling using dynamically distributed subdeadline
    and static topological top-level.
    """
    def __init__(self):
        super().__init__("ULS")

    def sort_within_application(self, app: DAGApplication, platform: HeterogeneousPlatform):
        app.compute_heft_ranks(platform)
        app.compute_subdeadlines(platform)

    def schedule(
        self,
        current_time: float,
        platform: HeterogeneousPlatform
    ) -> List[Tuple[int, int, int, float, float]]:
        ready_tasks: List[Tuple[DAGApplication, Task, float]] = []
        for app in list(self.active_applications.values()):
            if app.is_completed:
                continue
            ready_tids = app.update_ready_tasks()
            for tid in ready_tids:
                task = app.tasks[tid]
                # ULS urgency metric combining subdeadline margin and top-level
                urgency = (task.subdeadline - current_time) - 0.1 * task.topological_level
                ready_tasks.append((app, task, urgency))

        if not ready_tasks:
            return []

        # Sort purely by urgency (smaller urgency first)
        ready_tasks.sort(key=lambda x: x[2])

        scheduled = []
        for app, task, _ in ready_tasks:
            decision = self.allocate_task_eft(app, task, current_time, platform)
            if decision is not None:
                scheduled.append(decision)

        return scheduled
