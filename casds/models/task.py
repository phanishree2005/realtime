"""
Task Model for Automotive Cyber-Physical Systems (ACPS)
"""

from enum import Enum
from typing import Dict, List, Optional


class TaskStatus(Enum):
    UNFINISHED = "UNFINISHED"
    READY = "READY"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    DEFERRED = "DEFERRED"
    MISSED = "MISSED"


class Task:
    """
    Represents an individual computation task v_i^a in an application DAG G^a.
    """
    def __init__(
        self,
        task_id: int,
        name: str,
        task_type: str,
        wcet: Dict[str, float],
        app_id: int = 0,
        predecessors: Optional[List[int]] = None,
        successors: Optional[List[int]] = None
    ):
        self.task_id = task_id
        self.name = name
        self.task_type = task_type
        # wcet maps pe_type or pe_id to execution time in ms (inf if unsupported)
        self.wcet = wcet
        self.app_id = app_id
        self.predecessors: List[int] = predecessors if predecessors is not None else []
        self.successors: List[int] = successors if successors is not None else []

        # Runtime dynamic state
        self.status = TaskStatus.UNFINISHED
        self.allocated_pe: Optional[int] = None
        self.start_time: Optional[float] = None
        self.finish_time: Optional[float] = None
        self.actual_execution_time: Optional[float] = None

        # Static / dynamic ranking attributes
        self.upward_rank: float = 0.0
        self.subdeadline: float = 0.0
        self.topological_level: int = 0
        self.t_level: float = 0.0
        self.b_level: float = 0.0

    @property
    def full_id(self) -> str:
        return f"A{self.app_id}_T{self.task_id}_{self.name}"

    def can_run_on(self, pe_type: str) -> bool:
        """Returns True if the task can execute on this PE type."""
        time_cost = self.wcet.get(pe_type, float("inf"))
        return time_cost is not None and time_cost != float("inf")

    def get_wcet(self, pe_type: str) -> float:
        """Get execution cost for a given PE type; returns float('inf') if unsupported."""
        return self.wcet.get(pe_type, float("inf"))

    def reset(self):
        """Reset dynamic execution state."""
        self.status = TaskStatus.UNFINISHED
        self.allocated_pe = None
        self.start_time = None
        self.finish_time = None
        self.actual_execution_time = None

    def clone(self) -> "Task":
        """Create an identical copy of the task with clean dynamic state."""
        t = Task(
            task_id=self.task_id,
            name=self.name,
            task_type=self.task_type,
            wcet=dict(self.wcet),
            app_id=self.app_id,
            predecessors=list(self.predecessors),
            successors=list(self.successors)
        )
        t.upward_rank = self.upward_rank
        t.subdeadline = self.subdeadline
        t.topological_level = self.topological_level
        t.t_level = self.t_level
        t.b_level = self.b_level
        return t

    def __repr__(self) -> str:
        return f"<Task {self.full_id} type={self.task_type} status={self.status.value}>"
