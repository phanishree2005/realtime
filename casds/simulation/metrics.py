"""
Evaluation Metrics for Dynamic Scheduling on ACPS (Section IV-A3).
Computes DSR(S), DSR(S_crit), Makespan MS(S), PE Utilization U(S, P), and Effective Throughput ET(S).
"""

from typing import List, Dict, Any, Optional
from ..models.dag import DAGApplication
from ..models.platform import HeterogeneousPlatform


class SchedulingMetrics:
    """
    Computes and aggregates all performance metrics defined in Section IV-A3 and IV-E.
    """
    def __init__(
        self,
        applications: List[DAGApplication],
        platform: HeterogeneousPlatform,
        total_time_ms: float,
        start_time_ms: float = 0.0,
        scheduler_runtime_s: float = 0.0
    ):
        self.applications = applications
        self.platform = platform
        self.total_time_ms = total_time_ms
        self.start_time_ms = start_time_ms
        self.scheduler_runtime_s = scheduler_runtime_s

        self.num_total = len(applications)
        self.num_success = 0
        self.num_critical_total = 0
        self.num_critical_success = 0

        # Criticality level breakdown (e.g. S1, S2, S3, S4 for ASIL)
        self.crit_counts: Dict[int, int] = {}
        self.crit_success: Dict[int, int] = {}

        for app in applications:
            c = app.criticality
            self.crit_counts[c] = self.crit_counts.get(c, 0) + 1
            if app.met_deadline:
                self.crit_success[c] = self.crit_success.get(c, 0) + 1

            if c >= 2:
                self.num_critical_total += 1
                if app.met_deadline:
                    self.num_critical_success += 1

            if app.met_deadline:
                self.num_success += 1

        # Calculate Makespan (in seconds for ET calculation, or ms)
        first_arrival = min((a.arrival_time for a in applications), default=0.0)
        last_completion = max(
            (a.completion_time for a in applications if a.completion_time is not None),
            default=total_time_ms
        )
        self.makespan_ms = max(1.0, last_completion - first_arrival)
        self.makespan_s = self.makespan_ms / 1000.0

        # 1. DSR(S): Deadline satisfaction ratio of all applications [%]
        self.dsr = (self.num_success / self.num_total * 100.0) if self.num_total > 0 else 0.0

        # 2. DSR(S_crit): DSR of critical DAGs [%]
        self.dsr_crit = (
            (self.num_critical_success / self.num_critical_total * 100.0)
            if self.num_critical_total > 0 else 0.0
        )

        # Level-wise DSR [%]
        self.dsr_levels: Dict[int, float] = {}
        for c, count in self.crit_counts.items():
            succ = self.crit_success.get(c, 0)
            self.dsr_levels[c] = (succ / count * 100.0) if count > 0 else 0.0

        # 4. U(S, P): Utilization of PEs on P [%]
        total_busy_ms = sum(p.total_busy_time(until_time=last_completion) for p in platform.pes)
        total_capacity_ms = platform.num_pes * self.makespan_ms
        self.utilization = (total_busy_ms / total_capacity_ms * 100.0) if total_capacity_ms > 0 else 0.0

        # 5. ET(S): Effective Throughput = |S_se| / Makespan(S) [DAGs/s]
        self.effective_throughput = (self.num_success / self.makespan_s) if self.makespan_s > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "DSR": round(self.dsr, 2),
            "DSR_crit": round(self.dsr_crit, 2),
            "DSR_levels": {f"S_{k}": round(v, 2) for k, v in self.dsr_levels.items()},
            "Makespan_ms": round(self.makespan_ms, 2),
            "Utilization": round(self.utilization, 2),
            "EffectiveThroughput": round(self.effective_throughput, 2),
            "SuccessCount": self.num_success,
            "TotalCount": self.num_total,
            "SchedulerRuntime_s": round(self.scheduler_runtime_s, 4),
        }

    def __repr__(self) -> str:
        return (
            f"SchedulingMetrics(DSR={self.dsr:.2f}%, DSR_crit={self.dsr_crit:.2f}%, "
            f"ET={self.effective_throughput:.2f} DAG/s, Util={self.utilization:.2f}%, "
            f"MS={self.makespan_ms:.1f}ms)"
        )
