"""
Trace Generator for ACPS Dynamic Workloads (Section IV-A2, IV-D, IV-E, V-B).
Generates Poisson arrival traces, mixed-criticality distributions, deadline profiles,
execution time delay noise, and PE failure injections.
"""

from typing import List, Optional, Tuple, Dict
import random
import numpy as np

from .adsuite_benchmark import create_adsuite_dag
from .selfpilot_benchmark import create_selfpilot_dag
from ..models.dag import DAGApplication


class TraceGenerator:
    """
    Generates dynamic multi-application workloads matching the paper's experimental setup.
    """
    def __init__(self, seed: Optional[int] = 42):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def generate_adsuite_trace(
        self,
        num_dags: int = 1000,
        arrival_rate_lambda: float = 20.0,  # DAGs per second
        critical_ratio: float = 0.3,        # R_crit in {0.3, 0.6, 0.9}
        relative_deadline: float = 400.0,   # ms
        multi_criticality: bool = False,    # True for 4 ASIL levels {1, 2, 3, 4}
        exec_delay_mu: float = 0.0          # mu_d in [0.0, 1.0] for robustness analysis
    ) -> List[DAGApplication]:
        """
        Generates application trace for ADSuite benchmark (Section IV-A2).
        Arrivals follow Poisson distribution with expected value lambda (inter-arrival in ms: Exp(lambda/1000)).
        """
        trace: List[DAGApplication] = []
        dag_types = ["Type 1", "Type 2", "Type 3", "Type 4"]
        current_time = 0.0

        for app_id in range(num_dags):
            # Inter-arrival time from exponential distribution (rate lambda/1000 per ms)
            inter_arrival = np.random.exponential(1000.0 / arrival_rate_lambda)
            current_time += inter_arrival

            # Select DAG type
            dtype = random.choice(dag_types)

            # Assign criticality
            if multi_criticality:
                # Section IV-E: 4 criticality levels {1, 2, 3, 4} with equal proportion
                crit = random.choice([1, 2, 3, 4])
            else:
                # 2 criticality levels {1, 2}, where 2 is critical
                crit = 2 if random.random() < critical_ratio else 1

            app = create_adsuite_dag(
                app_id=app_id,
                dag_type=dtype,
                criticality=crit,
                arrival_time=current_time,
                relative_deadline=relative_deadline
            )

            # Apply execution noise if mu_d > 0 (Section IV-D)
            if exec_delay_mu > 0.0:
                for task in app.tasks.values():
                    new_wcet = {}
                    for pe_type, wcet_val in task.wcet.items():
                        if wcet_val < float("inf"):
                            mean_delay = exec_delay_mu * wcet_val
                            std_delay = 0.1 * mean_delay if mean_delay > 0 else 0.0
                            noise = np.random.normal(mean_delay, std_delay) if std_delay > 0 else mean_delay
                            new_wcet[pe_type] = max(0.1, wcet_val + noise)
                        else:
                            new_wcet[pe_type] = float("inf")
                    task.wcet = new_wcet

            trace.append(app)

        return trace

    def generate_selfpilot_trace(
        self,
        num_dags: int = 1000,
        arrival_rate_lambda: float = 20.0,  # in {5, 10, 20, 30, 40}
        critical_ratio: float = 0.6,        # in {0.3, 0.6, 0.9}
        exec_delay_mu: float = 0.0
    ) -> List[DAGApplication]:
        """
        Generates application trace for SelfPilot benchmark (Section V-B).
        Deadlines ~ N(200, 20^2) ms.
        """
        trace: List[DAGApplication] = []
        current_time = 0.0

        for app_id in range(num_dags):
            inter_arrival = np.random.exponential(1000.0 / arrival_rate_lambda)
            current_time += inter_arrival

            # Criticality in {1, 2}
            crit = 2 if random.random() < critical_ratio else 1

            # Relative deadline ~ N(200, 20^2) ms
            rel_ddl = max(50.0, np.random.normal(200.0, 20.0))

            app = create_selfpilot_dag(
                app_id=app_id,
                criticality=crit,
                arrival_time=current_time,
                relative_deadline=rel_ddl
            )

            # Apply execution noise if specified
            if exec_delay_mu > 0.0:
                for task in app.tasks.values():
                    new_wcet = {}
                    for pe_type, wcet_val in task.wcet.items():
                        if wcet_val < float("inf"):
                            mean_delay = exec_delay_mu * wcet_val
                            std_delay = 0.1 * mean_delay if mean_delay > 0 else 0.0
                            noise = np.random.normal(mean_delay, std_delay) if std_delay > 0 else mean_delay
                            new_wcet[pe_type] = max(0.1, wcet_val + noise)
                        else:
                            new_wcet[pe_type] = float("inf")
                    task.wcet = new_wcet

            trace.append(app)

        return trace

    def generate_pe_failure_events(
        self,
        total_simulation_time: float,
        num_pes: int,
        failure_rate_lambda_f: float = 0.0,  # failures per second
        avg_recovery_time_ms: float = 15.0
    ) -> List[Tuple[float, int, float]]:
        """
        Generates transient PE failures following Poisson process (Section IV-D).
        Returns list of (failure_time_ms, pe_id, recovery_duration_ms).
        """
        if failure_rate_lambda_f <= 0.0:
            return []

        failures = []
        current_time = 0.0
        while current_time < total_simulation_time:
            inter_failure = np.random.exponential(1000.0 / failure_rate_lambda_f)
            current_time += inter_failure
            if current_time >= total_simulation_time:
                break
            pe_id = random.randint(0, num_pes - 1)
            duration = max(1.0, np.random.exponential(avg_recovery_time_ms))
            failures.append((current_time, pe_id, duration))

        return failures
