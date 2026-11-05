"""
Real-Time Task, Job, and TaskSet Models
Data structures representing periodic/sporadic automotive tasks,
job execution lifecycles, and task set collections.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import math
import copy


@dataclass
class Task:
    """
    Periodic real-time task model: tau_i = (T_i, D_i, C_i, P_i).
    Time units are milliseconds (ms).
    """
    task_id: int
    name: str
    period: float            # T_i (ms)
    deadline: float          # D_i (ms)
    wcet: float              # Worst-case execution time C_i (ms)
    bcet: float = 0.0        # Best-case execution time (ms)
    priority: int = 0        # Priority level (higher integer = higher priority)
    criticality: str = "ASIL-D"  # ISO 26262 ASIL level: ASIL-D, ASIL-C, ASIL-B, ASIL-A, QM
    modality: str = "Generic"    # Sensor / pipeline modality

    def __post_init__(self):
        if self.bcet <= 0.0:
            self.bcet = max(0.1, self.wcet * 0.7)
        if self.deadline <= 0.0:
            self.deadline = self.period

    @property
    def utilization(self) -> float:
        """Task processor utilization U_i = C_i / T_i."""
        return self.wcet / self.period if self.period > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "period_ms": self.period,
            "deadline_ms": self.deadline,
            "wcet_ms": self.wcet,
            "bcet_ms": self.bcet,
            "priority": self.priority,
            "utilization": self.utilization,
            "criticality": self.criticality,
            "modality": self.modality
        }


@dataclass
class Job:
    """
    Instance (invocation) of a Task released at runtime.
    """
    job_id: int
    task_id: int
    task_name: str
    priority: int
    release_time: float
    absolute_deadline: float
    execution_time: float
    remaining_time: float
    start_time: Optional[float] = None
    finish_time: Optional[float] = None
    preemption_count: int = 0
    is_completed: bool = False
    missed_deadline: bool = False

    @property
    def response_time(self) -> Optional[float]:
        """Response time R = finish_time - release_time."""
        if self.finish_time is not None:
            return self.finish_time - self.release_time
        return None

    @property
    def lateness(self) -> Optional[float]:
        """Lateness L = finish_time - absolute_deadline."""
        if self.finish_time is not None:
            return self.finish_time - self.absolute_deadline
        return None


class TaskSet:
    """
    Collection of periodic real-time tasks.
    Supports cloning to guarantee fair comparison across priority strategies.
    """
    def __init__(self, tasks: Optional[List[Task]] = None, name: str = "TaskSet"):
        self.tasks: List[Task] = list(tasks) if tasks else []
        self.name: str = name

    def add_task(self, task: Task) -> None:
        self.tasks.append(task)

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self):
        return iter(self.tasks)

    def __getitem__(self, idx: int) -> Task:
        return self.tasks[idx]

    @property
    def total_utilization(self) -> float:
        """Total processor utilization U = sum(C_i / T_i)."""
        return sum(t.utilization for t in self.tasks)

    def get_hyper_period(self, max_limit_ms: float = 2000.0) -> float:
        """
        Calculates the hyper-period H = LCM(T_1, ..., T_n).
        Clamped to max_limit_ms for simulation tractability.
        """
        if not self.tasks:
            return 0.0

        periods_int = [int(round(t.period * 10)) for t in self.tasks]  # 0.1 ms granularity
        lcm_val = periods_int[0]
        for p in periods_int[1:]:
            lcm_val = (lcm_val * p) // math.gcd(lcm_val, p)
            if (lcm_val / 10.0) > max_limit_ms:
                return max_limit_ms
        return min(max_limit_ms, lcm_val / 10.0)

    def clone(self) -> 'TaskSet':
        """Deep copy task set to ensure fair comparison under identical task attributes."""
        cloned_tasks = [copy.deepcopy(t) for t in self.tasks]
        return TaskSet(cloned_tasks, name=self.name)

    def to_list(self) -> List[Dict[str, Any]]:
        return [t.to_dict() for t in self.tasks]
