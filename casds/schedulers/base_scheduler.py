"""
Base Dynamic Scheduler Interface for ACPS.
Provides common methods for tracking active applications, ready tasks,
and computing Earliest Ready Time (ERT), Earliest Start Time (EST), and Earliest Finish Time (EFT).
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
from ..models.dag import DAGApplication
from ..models.task import Task, TaskStatus
from ..models.platform import HeterogeneousPlatform, ProcessingElement


class BaseScheduler(ABC):
    """
    Abstract dynamic scheduler running in ACPS event-driven framework.
    """
    def __init__(self, name: str):
        self.name = name
        self.active_applications: Dict[int, DAGApplication] = {}

    def add_application(self, app: DAGApplication, platform: HeterogeneousPlatform):
        """Called when a new application arrives at runtime."""
        self.active_applications[app.app_id] = app
        # Trigger Step 2: Task sorting within application
        self.sort_within_application(app, platform)

    @abstractmethod
    def sort_within_application(self, app: DAGApplication, platform: HeterogeneousPlatform):
        """Step 2: Intra-application task sorting."""
        pass

    @abstractmethod
    def schedule(
        self,
        current_time: float,
        platform: HeterogeneousPlatform
    ) -> List[Tuple[int, int, int, float, float]]:
        """
        Main scheduling invocation triggered by arrival or task finish event.
        Returns list of (app_id, task_id, pe_id, start_time, duration).
        """
        pass

    def compute_ert(
        self,
        app: DAGApplication,
        task: Task,
        pe: ProcessingElement,
        current_time: float,
        platform: HeterogeneousPlatform
    ) -> float:
        """
        Earliest Ready Time (ERT) of task v_i^a on PE p_k (Eq. 5 & 9):
        ERT(v_i^a, p_k) = max_{v_j in pred(v_i)} ( max(T_ft^{a,j}, t) + c_{i,j,k,l}^a )
        """
        if not task.predecessors:
            return max(app.arrival_time, current_time)

        max_ready = current_time
        for pred_id in task.predecessors:
            pred_task = app.get_task(pred_id)
            # Finish time of predecessor
            pred_finish = pred_task.finish_time if pred_task.finish_time is not None else current_time
            # Communication delay: 0 if same PE, else inter-PE cost
            pred_pe = pred_task.allocated_pe
            if pred_pe is not None and pred_pe == pe.pe_id:
                comm_cost = 0.0
            else:
                comm_cost = app.edge_costs.get((pred_id, task.task_id), platform.default_inter_pe_comm)
            
            ready_candidate = max(pred_finish, current_time) + comm_cost
            if ready_candidate > max_ready:
                max_ready = ready_candidate

        return max_ready

    def compute_est(
        self,
        app: DAGApplication,
        task: Task,
        pe: ProcessingElement,
        current_time: float,
        platform: HeterogeneousPlatform
    ) -> float:
        """
        Earliest Start Time (EST) on PE p_k (Eq. 9):
        EST = max{ Lambda(p_k, P_t^s), ERT(v_i^a, p_k) }
        """
        lambda_pe = platform.get_available_time(pe.pe_id, current_time)
        ert = self.compute_ert(app, task, pe, current_time, platform)
        return max(lambda_pe, ert)

    def compute_eft(
        self,
        app: DAGApplication,
        task: Task,
        pe: ProcessingElement,
        current_time: float,
        platform: HeterogeneousPlatform
    ) -> float:
        """
        Earliest Finish Time (EFT) on PE p_k (Eq. 9):
        EFT = w_{i,k}^a + EST
        """
        wcet = task.get_wcet(pe.pe_type)
        if wcet == float("inf"):
            return float("inf")
        est = self.compute_est(app, task, pe, current_time, platform)
        return est + wcet

    def allocate_task_eft(
        self,
        app: DAGApplication,
        task: Task,
        current_time: float,
        platform: HeterogeneousPlatform
    ) -> Optional[Tuple[int, int, int, float, float]]:
        """
        Step 4: Finds target PE with minimum EFT (Eq. 14) and allocates if PE is idle (Eq. 15).
        Returns (app_id, task_id, pe_id, start_time, duration) or None if target PE is busy.
        """
        best_pe: Optional[ProcessingElement] = None
        min_eft = float("inf")
        best_est = float("inf")

        for pe in platform.pes:
            if pe.is_failed:
                continue
            wcet = task.get_wcet(pe.pe_type)
            if wcet == float("inf"):
                continue
            est = self.compute_est(app, task, pe, current_time, platform)
            eft = est + wcet
            if eft < min_eft:
                min_eft = eft
                best_est = est
                best_pe = pe

        if best_pe is not None and platform.is_idle(best_pe.pe_id, current_time):
            wcet = task.get_wcet(best_pe.pe_type)
            task.status = TaskStatus.RUNNING
            task.allocated_pe = best_pe.pe_id
            task.start_time = best_est
            task.actual_execution_time = wcet

            best_pe.allocate(
                task_id=task.task_id,
                app_id=app.app_id,
                task_name=task.name,
                start_time=best_est,
                duration=wcet,
                criticality=app.criticality
            )
            return (app.app_id, task.task_id, best_pe.pe_id, best_est, wcet)

        return None

    def clean_completed_applications(self):
        """Remove fully executed applications from active set."""
        completed_ids = [aid for aid, app in self.active_applications.items() if app.is_completed]
        for aid in completed_ids:
            del self.active_applications[aid]

    def reset(self):
        self.active_applications.clear()

