"""
Heterogeneous Processing Platform and Cost Model for ACPS.
Models multi-core CPUs, GPUs, and ASICs with communication topologies and failure states.
"""

from typing import Dict, List, Optional, Tuple, Any
import copy


class ProcessingElement:
    """
    Represents an individual heterogeneous processing element (PE) p_k in P.
    """
    def __init__(self, pe_id: int, name: str, pe_type: str):
        self.pe_id = pe_id
        self.name = name
        self.pe_type = pe_type  # e.g., 'CPU', 'GPU', 'ASIC', 'Ampere_GPU', etc.

        # Dynamic state
        self.busy_until: float = 0.0
        self.current_task: Optional[Any] = None
        self.schedule: List[Dict[str, Any]] = []

        # Fault-tolerance / noise injection state
        self.is_failed: bool = False
        self.failure_end_time: float = 0.0

    def available_time(self, current_time: float) -> float:
        """Returns earliest time PE becomes free: Lambda(p_k, P_t^s)."""
        earliest = max(current_time, self.busy_until)
        if self.is_failed:
            earliest = max(earliest, self.failure_end_time)
        return earliest

    def is_idle(self, current_time: float) -> bool:
        """PE is idle if available at current_time with no waiting time."""
        return self.available_time(current_time) <= current_time

    def allocate(
        self,
        task_id: int,
        app_id: int,
        task_name: str,
        start_time: float,
        duration: float,
        criticality: int = 1
    ):
        """Schedule a task on this PE."""
        finish_time = start_time + duration
        self.busy_until = finish_time
        record = {
            "task_id": task_id,
            "app_id": app_id,
            "task_name": task_name,
            "start_time": start_time,
            "duration": duration,
            "finish_time": finish_time,
            "criticality": criticality
        }
        self.schedule.append(record)

    def total_busy_time(self, until_time: Optional[float] = None) -> float:
        """Calculate cumulative busy time of this PE."""
        total = 0.0
        for rec in self.schedule:
            st = rec["start_time"]
            ft = rec["finish_time"]
            if until_time is not None:
                if st >= until_time:
                    continue
                ft = min(ft, until_time)
            total += max(0.0, ft - st)
        return total

    def reset(self):
        """Reset PE dynamic state."""
        self.busy_until = 0.0
        self.current_task = None
        self.schedule = []
        self.is_failed = False
        self.failure_end_time = 0.0

    def __repr__(self) -> str:
        return f"<PE {self.pe_id}: {self.name} ({self.pe_type}) busy_until={self.busy_until:.2f}>"


class HeterogeneousPlatform:
    """
    Heterogeneous Processing Platform P = {p_1, ..., p_|P|}.
    Contains PEs and Inter-PE communication topology matrix C.
    """
    def __init__(
        self,
        pes: List[ProcessingElement],
        comm_matrix: Optional[Dict[Tuple[int, int], float]] = None,
        default_inter_pe_comm: float = 10.0
    ):
        self.pes: List[ProcessingElement] = pes
        self.pe_map: Dict[int, ProcessingElement] = {p.pe_id: p for p in pes}
        self.num_pes = len(pes)
        self.comm_matrix: Dict[Tuple[int, int], float] = comm_matrix if comm_matrix is not None else {}
        self.default_inter_pe_comm = default_inter_pe_comm

    def get_pe(self, pe_id: int) -> ProcessingElement:
        return self.pe_map[pe_id]

    def get_available_time(self, pe_id: int, current_time: float) -> float:
        """Returns Lambda(p_k, P_t^s)."""
        return self.pe_map[pe_id].available_time(current_time)

    def is_idle(self, pe_id: int, current_time: float) -> bool:
        """Returns True if PE is idle at current_time."""
        return self.pe_map[pe_id].is_idle(current_time)

    def get_comm_cost(self, src_pe: int, dst_pe: int, edge_cost: Optional[float] = None) -> float:
        """
        Communication cost c_{i,j,k,l}.
        If source and destination tasks run on same PE, cost is 0 (intra-PE).
        Otherwise inter-PE communication delay is applied.
        """
        if src_pe == dst_pe:
            return 0.0
        if edge_cost is not None and edge_cost > 0:
            return edge_cost
        return self.comm_matrix.get((src_pe, dst_pe), self.default_inter_pe_comm)

    def avg_comm_cost(self, edge_cost: Optional[float] = None) -> float:
        """
        Calculates average communication cost bar{c}_{i,j} between distinct PEs.
        Used in HEFT rank calculation: 1/(|P|^2 - |P|) sum_{k != l} c_{i,j,k,l}.
        """
        if edge_cost is not None and edge_cost > 0:
            return edge_cost
        if self.num_pes <= 1:
            return 0.0
        total = 0.0
        count = 0
        for i in range(self.num_pes):
            for j in range(self.num_pes):
                if i != j:
                    total += self.get_comm_cost(i, j, edge_cost)
                    count += 1
        return total / count if count > 0 else 0.0

    def get_pes_for_type(self, pe_type: str) -> List[ProcessingElement]:
        """Get all PEs matching a specific type."""
        return [p for p in self.pes if p.pe_type == pe_type]

    def reset(self):
        """Reset all PEs to initial clean state."""
        for p in self.pes:
            p.reset()

    def clone(self) -> "HeterogeneousPlatform":
        """Create a deep copy of the platform."""
        new_pes = [
            ProcessingElement(p.pe_id, p.name, p.pe_type)
            for p in self.pes
        ]
        return HeterogeneousPlatform(
            pes=new_pes,
            comm_matrix=dict(self.comm_matrix),
            default_inter_pe_comm=self.default_inter_pe_comm
        )
