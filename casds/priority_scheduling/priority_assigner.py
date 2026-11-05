"""
Priority Assignment Strategies for Automotive Real-Time Systems.
Implements:
1. Uniform Priority Distribution
2. Arithmetic Priority Distribution
3. Geometric Priority Distribution
4. Logarithmic Priority Distribution (Configurable parameters and formula)
"""

import math
from typing import List, Callable, Optional, Dict, Any
import numpy as np

from .task_model import Task, TaskSet


class PriorityAssigner:
    """
    Assigns priorities to a TaskSet under four different distribution strategies.
    Supports configurable priority ranges [P_min, P_max], baseline ordering metrics
    (RM, DM, Criticality), and configurable parameters for arithmetic, geometric,
    and logarithmic distributions.
    """

    STRATEGY_UNIFORM = "Uniform"
    STRATEGY_ARITHMETIC = "Arithmetic"
    STRATEGY_GEOMETRIC = "Geometric"
    STRATEGY_LOGARITHMIC = "Logarithmic"

    ALL_STRATEGIES = [
        STRATEGY_UNIFORM,
        STRATEGY_ARITHMETIC,
        STRATEGY_GEOMETRIC,
        STRATEGY_LOGARITHMIC
    ]

    def __init__(
        self,
        p_min: int = 1,
        p_max: int = 256,
        base_sorting: str = "RM",
        arithmetic_a0: float = 1.0,
        arithmetic_d: float = 0.5,
        geometric_ratio: float = 1.618,
        logarithmic_beta: float = 9.0,
        logarithmic_base: float = math.e,
        custom_log_func: Optional[Callable[[float, float, float], float]] = None
    ):
        """
        Args:
            p_min: Minimum priority integer (lowest priority level).
            p_max: Maximum priority integer (highest priority level).
            base_sorting: Ordering policy to determine priority rank:
                          'RM' (Rate Monotonic: shorter T_i -> higher rank)
                          'DM' (Deadline Monotonic: shorter D_i -> higher rank)
                          'Criticality' (ASIL-D > ASIL-C > ASIL-B > ASIL-A > QM)
            arithmetic_a0: Base step weight for arithmetic progression.
            arithmetic_d: Step increment for arithmetic progression.
            geometric_ratio: Growth ratio r > 1 for geometric progression.
            logarithmic_beta: Curvature coefficient beta > 0 for logarithmic formula.
            logarithmic_base: Logarithm base (math.e, 2, 10).
            custom_log_func: Pluggable custom formula function f(x, p_min, p_max) -> priority.
        """
        self.p_min = p_min
        self.p_max = p_max
        self.base_sorting = base_sorting.upper()
        self.arithmetic_a0 = arithmetic_a0
        self.arithmetic_d = arithmetic_d
        self.geometric_ratio = geometric_ratio
        self.logarithmic_beta = logarithmic_beta
        self.logarithmic_base = logarithmic_base
        self.custom_log_func = custom_log_func

    def compute_topsis_scores(self, tasks: List[Task]) -> Dict[int, float]:
        """
        Calculates Adaptive Priority Combining Criticality (C_i) and Urgency (U_i)
        using TOPSIS and Entropy Weight Method (Eq. 11, 12, 13).

        Attributes:
          m=0: Task Criticality (ASIL-D=4, ASIL-C=3, ASIL-B=2, ASIL-A=1, QM=0)
          m=1: Task Urgency (-Slack = -(D_i - C_i), so less slack -> higher urgency)
        """
        N = len(tasks)
        if N == 0:
            return {}
        if N == 1:
            return {tasks[0].task_id: 1.0}

        crit_map = {"ASIL-D": 4.0, "ASIL-C": 3.0, "ASIL-B": 2.0, "ASIL-A": 1.0, "QM": 0.0}

        # 1. Attribute Matrix X: [N, 2]
        X = np.zeros((N, 2), dtype=np.float64)
        for i, t in enumerate(tasks):
            X[i, 0] = crit_map.get(t.criticality, 0.0)
            # Urgency: negative slack so tighter margin has higher value
            slack = t.deadline - t.wcet
            X[i, 1] = -slack

        # 2. Min-Max Normalization (Eq. 11)
        X_norm = np.zeros_like(X)
        for m in range(2):
            col_min = np.min(X[:, m])
            col_max = np.max(X[:, m])
            if col_max - col_min > 1e-9:
                X_norm[:, m] = (X[:, m] - col_min) / (col_max - col_min)
            else:
                X_norm[:, m] = 1.0 / N

        # 3. Entropy Weight Method (Eq. 12)
        col_sums = np.sum(X_norm, axis=0)
        col_sums[col_sums == 0] = 1e-12
        P_prob = X_norm / col_sums

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

        divergence = 1.0 - entropy
        div_sum = np.sum(divergence)
        weights = divergence / div_sum if div_sum > 1e-9 else np.array([0.5, 0.5])

        # Weighted attribute matrix X_wt = omega_m * x_{n,m}^{norm}
        X_wt = X_norm * weights

        # 4. Ideal Best (Z+) and Worst (Z-) Solutions (Eq. 13)
        Z_plus = np.max(X_wt, axis=0)
        Z_minus = np.min(X_wt, axis=0)

        # Distances
        Sim_plus = np.sqrt(np.sum((X_wt - Z_plus) ** 2, axis=1))
        Sim_minus = np.sqrt(np.sum((X_wt - Z_minus) ** 2, axis=1))

        # Relative Closeness P_i = Sim^- / (Sim^- + Sim^+)
        denom = Sim_minus + Sim_plus
        denom[denom == 0] = 1e-9
        closeness = Sim_minus / denom

        return {tasks[i].task_id: float(closeness[i]) for i in range(N)}

    def _get_sort_key(self, task: Task, topsis_scores: Optional[Dict[int, float]] = None) -> tuple:
        """Determines sorting tuple so lowest key maps to highest priority."""
        if self.base_sorting in ["TOPSIS", "ADAPTIVE", "ADAPTIVE_TOPSIS"]:
            score = (topsis_scores or {}).get(task.task_id, 0.0)
            # Higher TOPSIS score means higher priority, so return negative score
            return (-score, task.period, task.task_id)
        elif self.base_sorting == "DM":
            return (task.deadline, task.period, task.task_id)
        elif self.base_sorting == "CRITICALITY":
            crit_map = {"ASIL-D": 1, "ASIL-C": 2, "ASIL-B": 3, "ASIL-A": 4, "QM": 5}
            return (crit_map.get(task.criticality, 9), task.period, task.task_id)
        else:  # Default RM
            return (task.period, task.deadline, task.task_id)

    def _log_base(self, val: float) -> float:
        """Computes log with configured base."""
        if self.logarithmic_base == 2:
            return math.log2(val)
        elif self.logarithmic_base == 10:
            return math.log10(val)
        return math.log(val)

    def compute_priority_values(
        self,
        strategy: str,
        n_tasks: int,
        enforce_unique: bool = True
    ) -> List[int]:
        """
        Computes the priority values for N tasks under the chosen distribution strategy.
        Index 0 gets lowest priority, Index N-1 gets highest priority.
        """
        if n_tasks <= 0:
            return []
        if n_tasks == 1:
            return [self.p_max]

        priorities_raw: List[float] = []

        if strategy == self.STRATEGY_UNIFORM:
            # Uniform: linear progression P_i = P_min + i * (P_max - P_min) / (N - 1)
            for i in range(n_tasks):
                norm_i = i / (n_tasks - 1)
                p = self.p_min + norm_i * (self.p_max - self.p_min)
                priorities_raw.append(p)

        elif strategy == self.STRATEGY_ARITHMETIC:
            # Arithmetic: step size s_k = a_0 + k * d
            # Cumulative sums S_i = sum_{k=0}^i s_k
            steps = [self.arithmetic_a0 + k * self.arithmetic_d for k in range(n_tasks)]
            cum_steps = np.cumsum(steps)
            total_sum = cum_steps[-1]
            for i in range(n_tasks):
                ratio = cum_steps[i] / total_sum
                p = self.p_min + ratio * (self.p_max - self.p_min)
                priorities_raw.append(p)

        elif strategy == self.STRATEGY_GEOMETRIC:
            # Geometric: exponential growth curve
            r = max(1.001, self.geometric_ratio)
            denominator = (r ** (n_tasks - 1)) - 1.0
            for i in range(n_tasks):
                if denominator > 0:
                    ratio = ((r ** i) - 1.0) / denominator
                else:
                    ratio = i / (n_tasks - 1)
                p = self.p_min + ratio * (self.p_max - self.p_min)
                priorities_raw.append(p)

        elif strategy == self.STRATEGY_LOGARITHMIC:
            # Configurable Logarithmic:
            # P_i = P_min + (P_max - P_min) * (log_b(1 + beta * x_i) / log_b(1 + beta))
            beta = max(0.001, self.logarithmic_beta)
            denom = self._log_base(1.0 + beta)

            for i in range(n_tasks):
                norm_i = i / (n_tasks - 1)
                if self.custom_log_func is not None:
                    p = self.custom_log_func(norm_i, self.p_min, self.p_max)
                else:
                    ratio = self._log_base(1.0 + beta * norm_i) / denom
                    p = self.p_min + ratio * (self.p_max - self.p_min)
                priorities_raw.append(p)
        else:
            raise ValueError(f"Unknown priority distribution strategy: {strategy}")

        # Round to integers within bounds
        int_priorities = [int(round(p)) for p in priorities_raw]
        int_priorities = [max(self.p_min, min(self.p_max, p)) for p in int_priorities]

        # If priority range is wide enough and uniqueness is requested, resolve ties
        if enforce_unique and (self.p_max - self.p_min + 1) >= n_tasks:
            for i in range(1, n_tasks):
                if int_priorities[i] <= int_priorities[i - 1]:
                    int_priorities[i] = int_priorities[i - 1] + 1
            # Clamp overflow from right to left if needed
            if int_priorities[-1] > self.p_max:
                overflow = int_priorities[-1] - self.p_max
                int_priorities = [max(self.p_min, p - overflow) for p in int_priorities]

        return int_priorities

    def assign_priorities(
        self,
        task_set: TaskSet,
        strategy: str,
        enforce_unique: bool = True
    ) -> TaskSet:
        """
        Assigns priorities to the TaskSet according to the specified strategy.
        Modifies and returns a cloned TaskSet to preserve immutability of the original.
        """
        cloned_set = task_set.clone()
        n = len(cloned_set.tasks)
        if n == 0:
            return cloned_set

        # Sort tasks based on baseline policy (RM/DM/Criticality/TOPSIS)
        topsis_scores = self.compute_topsis_scores(cloned_set.tasks) if self.base_sorting in ["TOPSIS", "ADAPTIVE", "ADAPTIVE_TOPSIS"] else None
        sorted_tasks = sorted(cloned_set.tasks, key=lambda t: self._get_sort_key(t, topsis_scores))

        # Generate priority spectrum for N tasks
        priority_levels = self.compute_priority_values(strategy, n, enforce_unique=enforce_unique)

        # Map highest priority levels to top-ranked tasks (lowest periods)
        # reversed so rank 0 gets highest priority level (e.g. priority_levels[-1])
        for idx, task in enumerate(sorted_tasks):
            assigned_p = priority_levels[n - 1 - idx]
            task.priority = assigned_p

        return cloned_set

    def get_strategy_formulas_doc(self) -> Dict[str, str]:
        """Returns documented mathematical formulas for all strategies."""
        return {
            self.STRATEGY_UNIFORM: "P_i = round(P_min + i * (P_max - P_min) / (N - 1))",
            self.STRATEGY_ARITHMETIC: f"P_i = round(P_min + (P_max - P_min) * (sum_{{k=0}}^i ({self.arithmetic_a0} + k*{self.arithmetic_d})) / TotalSum)",
            self.STRATEGY_GEOMETRIC: f"P_i = round(P_min + (P_max - P_min) * (({self.geometric_ratio}^i - 1) / ({self.geometric_ratio}^(N-1) - 1)))",
            self.STRATEGY_LOGARITHMIC: f"P_i = round(P_min + (P_max - P_min) * (log_{self.logarithmic_base}(1 + {self.logarithmic_beta} * x_i) / log_{self.logarithmic_base}(1 + {self.logarithmic_beta})))"
        }
