"""
Multiprocessor Systems & Task Allocation Engine for Real-Time Embedded Cyber-Physical Systems.
Implements:
1. Multiprocessor Memory Architectures:
   - UMA (Uniform Memory Access): Symmetric Multiprocessing (SMP) with uniform bus latency
   - NUMA (Non-Uniform Memory Access): Distributed multi-cluster memory nodes with remote access penalties
2. Real-Time Task Allocation Algorithms (Bin-Packing Partitioning):
   - First-Fit (FF), Best-Fit (BF), Worst-Fit (WF), Next-Fit (NF)
   - First-Fit Decreasing (FFD), Best-Fit Decreasing (BFD), Worst-Fit Decreasing (WFD)
3. Processor Utilization & Load Balancing Metrics:
   - Per-core utilization, utilization variance / standard deviation, peak core load
4. Partitioned Multiprocessor Scheduling (Partitioned FPPS / DMS / RMS)
"""

import math
import copy
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from src.task_model import Task, TaskSet
from src.rta_analyzer import RTAAnalyzer


class ProcessorCore:
    """Represents a single processing core in a Multiprocessor System."""

    def __init__(self, core_id: int, node_id: int = 0, capacity: float = 1.0):
        self.core_id = core_id
        self.node_id = node_id  # NUMA node identifier (e.g. 0 or 1)
        self.capacity = capacity
        self.allocated_tasks: List[Task] = []

    @property
    def current_utilization(self) -> float:
        """Sum of utilizations of all tasks assigned to this core."""
        return sum(t.utilization for t in self.allocated_tasks)

    @property
    def remaining_capacity(self) -> float:
        """Available headroom on this core."""
        return max(0.0, self.capacity - self.current_utilization)

    def can_accommodate(self, task: Task) -> bool:
        """Checks if task utilization fits within remaining capacity."""
        return (self.current_utilization + task.utilization) <= (self.capacity + 1e-9)

    def assign_task(self, task: Task):
        self.allocated_tasks.append(task)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "core_id": self.core_id,
            "node_id": self.node_id,
            "tasks_count": len(self.allocated_tasks),
            "task_names": [t.name for t in self.allocated_tasks],
            "utilization": round(self.current_utilization, 4),
            "utilization_pct": round(self.current_utilization * 100.0, 2),
            "remaining_capacity": round(self.remaining_capacity, 4)
        }


class AllocationResult:
    """Encapsulates the complete result of a multiprocessor task allocation run."""

    def __init__(
        self,
        algorithm: str,
        architecture: str,
        num_cores: int,
        cores: List[ProcessorCore],
        unallocated_tasks: List[Task],
        memory_penalty_overhead_ms: float = 0.0
    ):
        self.algorithm = algorithm
        self.architecture = architecture
        self.num_cores = num_cores
        self.cores = cores
        self.unallocated_tasks = unallocated_tasks
        self.is_feasible = (len(unallocated_tasks) == 0)
        self.memory_penalty_overhead_ms = memory_penalty_overhead_ms

        utils = [c.current_utilization for c in cores]
        self.utilizations = utils
        self.total_utilization = float(sum(utils))
        self.mean_utilization = float(np.mean(utils)) if utils else 0.0
        self.std_utilization = float(np.std(utils)) if utils else 0.0
        self.max_utilization = float(np.max(utils)) if utils else 0.0
        self.min_utilization = float(np.min(utils)) if utils else 0.0
        self.peak_to_average_ratio = (self.max_utilization / self.mean_utilization) if self.mean_utilization > 0 else 1.0

        # Jain's Fairness Index: J(U) = (sum(U_i))^2 / (M * sum(U_i^2))
        sum_u = sum(utils)
        sum_u_sq = sum(u ** 2 for u in utils)
        if sum_u_sq > 0 and len(cores) > 0:
            self.jains_fairness_index = float((sum_u ** 2) / (len(cores) * sum_u_sq))
        else:
            self.jains_fairness_index = 1.0

        # CPU Core Affinity Masks (e.g. Core 0 -> 0x0001, Core 1 -> 0x0002, etc.)
        self.affinity_masks: Dict[str, str] = {}
        for core in cores:
            mask_hex = f"0x{1 << core.core_id:04X}"
            for t in core.allocated_tasks:
                self.affinity_masks[t.name] = mask_hex

        # Schedulability status per core under local Fixed-Priority Preemptive Scheduling
        self.core_schedulable: Dict[int, bool] = {}
        for core in cores:
            if not core.allocated_tasks:
                self.core_schedulable[core.core_id] = True
                continue
            # Assign local Deadline Monotonic priorities for RTA analysis
            sorted_tasks = sorted(core.allocated_tasks, key=lambda t: (t.deadline, t.period))
            for p_idx, t in enumerate(sorted_tasks):
                t.priority = len(sorted_tasks) - p_idx
            c_set = TaskSet(tasks=sorted_tasks, name=f"Core_{core.core_id}")
            rta = RTAAnalyzer()
            rta_res = rta.analyze(c_set)
            self.core_schedulable[core.core_id] = rta_res.get("is_schedulable", True)


class MultiprocessorAllocator:
    """
    Allocates real-time tasks to multiprocessor cores using bin-packing heuristics
    under UMA and NUMA memory architectures.
    """

    ALGO_FF = "First-Fit (FF)"
    ALGO_BF = "Best-Fit (BF)"
    ALGO_WF = "Worst-Fit (WF)"
    ALGO_NF = "Next-Fit (NF)"
    ALGO_FFD = "First-Fit Decreasing (FFD)"
    ALGO_BFD = "Best-Fit Decreasing (BFD)"
    ALGO_WFD = "Worst-Fit Decreasing (WFD)"
    ALGO_RM_FFD = "Rate-Monotonic FFD (RM-FFD)"
    ALGO_CRIT_ISOLATION = "Criticality-Aware Partitioning (ASIL-D Isolation)"

    ALL_ALGORITHMS = [
        ALGO_FFD,
        ALGO_BFD,
        ALGO_WFD,
        ALGO_FF,
        ALGO_BF,
        ALGO_WF,
        ALGO_NF,
        ALGO_RM_FFD,
        ALGO_CRIT_ISOLATION
    ]

    ARCH_UMA = "UMA (Uniform Memory Access)"
    ARCH_NUMA = "NUMA (Non-Uniform Memory Access)"

    def __init__(
        self,
        num_cores: int = 4,
        cores_per_numa_node: int = 2,
        numa_remote_penalty: float = 1.8,
        core_capacity_bound: float = 1.0
    ):
        """
        Args:
            num_cores: Total number of processing cores in the system (M).
            cores_per_numa_node: Number of cores grouped per NUMA cluster (Node 0, Node 1, etc.).
            numa_remote_penalty: Latency multiplier for remote memory access across interconnect (e.g. 1.8x - 2.5x).
            core_capacity_bound: Schedulable utilization limit per core (1.0 for EDF, 0.693 for RM, or custom).
        """
        self.num_cores = max(1, num_cores)
        self.cores_per_numa_node = max(1, cores_per_numa_node)
        self.numa_remote_penalty = max(1.0, numa_remote_penalty)
        self.core_capacity_bound = core_capacity_bound

    def _init_cores(self, num_cores: int) -> List[ProcessorCore]:
        cores = []
        for i in range(num_cores):
            node_id = i // self.cores_per_numa_node
            cores.append(ProcessorCore(core_id=i, node_id=node_id, capacity=self.core_capacity_bound))
        return cores

    def _adjust_task_for_numa(self, task: Task, target_core: ProcessorCore, primary_data_node: int = 0) -> Tuple[Task, float]:
        """
        Under NUMA, if task executes on a core belonging to a different NUMA node than
        its primary frame buffer, remote memory stalls expand WCET by memory intensity fraction.
        """
        cloned = copy.deepcopy(task)
        penalty_overhead_ms = 0.0

        if target_core.node_id != primary_data_node:
            # Memory access stalls: vision/AI tasks have high memory intensity (alpha = 0.35)
            memory_intensity = 0.35 if any(k in task.name for k in ["CAM", "FACE", "GAZE", "HUD"]) else 0.15
            remote_stall_factor = 1.0 + memory_intensity * (self.numa_remote_penalty - 1.0)
            orig_wcet = cloned.wcet
            cloned.wcet = round(orig_wcet * remote_stall_factor, 2)
            cloned.bcet = round(cloned.wcet * 0.7, 2)
            penalty_overhead_ms = cloned.wcet - orig_wcet

        return cloned, penalty_overhead_ms

    def allocate(
        self,
        task_set: TaskSet,
        algorithm: str = ALGO_FFD,
        architecture: str = ARCH_UMA,
        primary_data_node: int = 0
    ) -> AllocationResult:
        """
        Allocates tasks from task_set across multiprocessor cores using the specified algorithm and memory architecture.
        """
        cores = self._init_cores(self.num_cores)
        tasks_to_allocate = [copy.deepcopy(t) for t in task_set.tasks]

        # Sorting for "Decreasing" variants (Decreasing order of task utilization C_i / T_i)
        is_decreasing = "Decreasing" in algorithm or "D" in algorithm.split()[-1]
        if is_decreasing:
            tasks_to_allocate.sort(key=lambda t: t.utilization, reverse=True)

        unallocated: List[Task] = []
        total_memory_overhead = 0.0
        next_fit_index = 0

        for task in tasks_to_allocate:
            assigned = False

            if algorithm in [self.ALGO_FF, self.ALGO_FFD]:
                # First-Fit: First core with enough remaining capacity
                for core in cores:
                    adj_task, mem_ovh = self._adjust_task_for_numa(task, core, primary_data_node) if architecture == self.ARCH_NUMA else (task, 0.0)
                    if core.can_accommodate(adj_task):
                        core.assign_task(adj_task)
                        total_memory_overhead += mem_ovh
                        assigned = True
                        break

            elif algorithm in [self.ALGO_BF, self.ALGO_BFD]:
                # Best-Fit: Core that leaves the MINIMUM remaining capacity (tightest fit)
                best_core = None
                min_remaining = float("inf")
                best_adj_task = None
                best_mem_ovh = 0.0

                for core in cores:
                    adj_task, mem_ovh = self._adjust_task_for_numa(task, core, primary_data_node) if architecture == self.ARCH_NUMA else (task, 0.0)
                    if core.can_accommodate(adj_task):
                        rem = core.remaining_capacity - adj_task.utilization
                        if rem < min_remaining:
                            min_remaining = rem
                            best_core = core
                            best_adj_task = adj_task
                            best_mem_ovh = mem_ovh

                if best_core is not None:
                    best_core.assign_task(best_adj_task)
                    total_memory_overhead += best_mem_ovh
                    assigned = True

            elif algorithm in [self.ALGO_WF, self.ALGO_WFD]:
                # Worst-Fit: Core with the MAXIMUM remaining capacity (maximizes load balancing)
                worst_core = None
                max_remaining = -float("inf")
                worst_adj_task = None
                worst_mem_ovh = 0.0

                for core in cores:
                    adj_task, mem_ovh = self._adjust_task_for_numa(task, core, primary_data_node) if architecture == self.ARCH_NUMA else (task, 0.0)
                    if core.can_accommodate(adj_task):
                        rem = core.remaining_capacity - adj_task.utilization
                        if rem > max_remaining:
                            max_remaining = rem
                            worst_core = core
                            worst_adj_task = adj_task
                            worst_mem_ovh = mem_ovh

                if worst_core is not None:
                    worst_core.assign_task(worst_adj_task)
                    total_memory_overhead += worst_mem_ovh
                    assigned = True

            elif algorithm == self.ALGO_NF:
                # Next-Fit: Scan starts from the last filled core, without cycling back until full
                start_idx = next_fit_index
                for offset in range(len(cores)):
                    curr_idx = (start_idx + offset) % len(cores)
                    core = cores[curr_idx]
                    adj_task, mem_ovh = self._adjust_task_for_numa(task, core, primary_data_node) if architecture == self.ARCH_NUMA else (task, 0.0)
                    if core.can_accommodate(adj_task):
                        core.assign_task(adj_task)
                        total_memory_overhead += mem_ovh
                        next_fit_index = curr_idx
                        assigned = True
                        break

            elif algorithm == self.ALGO_RM_FFD:
                # Rate-Monotonic FFD: Sorts by decreasing utilization, assigns to first core
                # satisfying the exact Liu & Layland RM schedulability bound k*(2^(1/k) - 1)
                for core in cores:
                    adj_task, mem_ovh = self._adjust_task_for_numa(task, core, primary_data_node) if architecture == self.ARCH_NUMA else (task, 0.0)
                    k = len(core.allocated_tasks) + 1
                    rm_bound = k * (math.pow(2.0, 1.0 / k) - 1.0)
                    prospective_u = core.current_utilization + adj_task.utilization
                    if prospective_u <= min(core.capacity, rm_bound + 1e-5):
                        core.assign_task(adj_task)
                        total_memory_overhead += mem_ovh
                        assigned = True
                        break
                # Fallback: if RM bound is tight, fit in first core with standard capacity headroom
                if not assigned:
                    for core in cores:
                        adj_task, mem_ovh = self._adjust_task_for_numa(task, core, primary_data_node) if architecture == self.ARCH_NUMA else (task, 0.0)
                        if core.can_accommodate(adj_task):
                            core.assign_task(adj_task)
                            total_memory_overhead += mem_ovh
                            assigned = True
                            break

            elif algorithm == self.ALGO_CRIT_ISOLATION:
                # Criticality-Aware Allocation (ISO 26262 Freedom From Interference - FFI)
                # Maps safety-critical tasks (ASIL-D, ASIL-C) to dedicated Safety Cores (e.g. Cores 0..half)
                # Maps non-safety tasks (QM, ASIL-A, ASIL-B) to Performance Cores
                is_safety_task = any(tag in task.name for tag in ["CAN_ALERT", "CASDS_ARB", "ALERT"])
                safety_cores = cores[:max(1, len(cores) // 2)]
                perf_cores = cores[max(1, len(cores) // 2):] if len(cores) > 1 else cores

                preferred_cores = safety_cores if is_safety_task else (perf_cores if perf_cores else safety_cores)
                fallback_cores = perf_cores if is_safety_task else safety_cores

                for core in preferred_cores:
                    adj_task, mem_ovh = self._adjust_task_for_numa(task, core, primary_data_node) if architecture == self.ARCH_NUMA else (task, 0.0)
                    if core.can_accommodate(adj_task):
                        core.assign_task(adj_task)
                        total_memory_overhead += mem_ovh
                        assigned = True
                        break

                if not assigned:
                    for core in fallback_cores:
                        adj_task, mem_ovh = self._adjust_task_for_numa(task, core, primary_data_node) if architecture == self.ARCH_NUMA else (task, 0.0)
                        if core.can_accommodate(adj_task):
                            core.assign_task(adj_task)
                            total_memory_overhead += mem_ovh
                            assigned = True
                            break

            if not assigned:
                unallocated.append(task)

        return AllocationResult(
            algorithm=algorithm,
            architecture=architecture,
            num_cores=self.num_cores,
            cores=cores,
            unallocated_tasks=unallocated,
            memory_penalty_overhead_ms=total_memory_overhead
        )

    def compare_all_algorithms(
        self,
        task_set: TaskSet,
        architecture: str = ARCH_UMA
    ) -> List[Dict[str, Any]]:
        """Runs and compares all bin-packing allocation algorithms side-by-side."""
        records = []
        for algo in self.ALL_ALGORITHMS:
            res = self.allocate(task_set, algorithm=algo, architecture=architecture)
            records.append({
                "Algorithm": algo,
                "Feasible": "✅ YES" if res.is_feasible else f"❌ {len(res.unallocated_tasks)} Missed",
                "Max Core Util (%)": round(res.max_utilization * 100, 2),
                "Mean Core Util (%)": round(res.mean_utilization * 100, 2),
                "Load Std Dev (σ)": round(res.std_utilization * 100, 2),
                "Jain's Fairness": round(res.jains_fairness_index, 3),
                "Peak-to-Avg Ratio": round(res.peak_to_average_ratio, 2),
                "NUMA Memory Stall (ms)": round(res.memory_penalty_overhead_ms, 2)
            })
        return records

    def simulate_multiprocessor_trace(
        self,
        allocation_result: AllocationResult,
        duration_ms: float = 300.0
    ) -> List[Dict[str, Any]]:
        """
        Generates simulated multi-core Gantt execution slices for all partitioned cores concurrently.
        """
        multi_trace = []
        for core in allocation_result.cores:
            if not core.allocated_tasks:
                continue

            # Sort tasks on this core by priority (Deadline Monotonic)
            sorted_tasks = sorted(core.allocated_tasks, key=lambda t: (t.deadline, t.period))
            for p_idx, t in enumerate(sorted_tasks):
                t.priority = len(sorted_tasks) - p_idx

            # Independent simulation per core
            from src.discrete_event_simulator import DiscreteEventSimulator
            sim = DiscreteEventSimulator(context_switch_penalty_ms=0.005)
            c_set = TaskSet(tasks=sorted_tasks, name=f"Core_{core.core_id}")
            sim_out = sim.run(c_set, duration_ms=duration_ms, use_wcet=True)

            for slice_data in sim_out.get("execution_trace", []):
                multi_trace.append({
                    "core_label": f"Core #{core.core_id} (Node {core.node_id})",
                    "core_id": core.core_id,
                    "node_id": core.node_id,
                    "task_name": slice_data["task_name"],
                    "job_id": slice_data["job_id"],
                    "start": slice_data["start"],
                    "finish": slice_data["finish"],
                    "duration": slice_data["duration"],
                    "priority": slice_data["priority"]
                })

        return multi_trace

    @staticmethod
    def get_dhall_effect_scenario(num_cores: int = 4, epsilon: float = 0.05) -> Dict[str, Any]:
        """
        Demonstrates Dhall's Effect (Dhall & Liu, 1978):
        Shows how Global Earliest Deadline First (G-EDF) and Global Rate Monotonic (G-RM)
        can fail at arbitrarily low utilization (U ≈ 1.0 + epsilon), far below system capacity M,
        whereas Partitioned Scheduling (P-EDF / P-RMS) easily schedules the workload.
        """
        # M light tasks: C = 2*epsilon, T = 1.0, D = 1.0 -> U = 2*epsilon
        # 1 heavy task: C = 1.0, T = 1.0 + epsilon, D = 1.0 + epsilon -> U = 1.0 / (1.0 + epsilon)
        c_light = 2.0 * epsilon * 100.0  # scaled to ms
        t_light = 100.0
        c_heavy = 100.0
        t_heavy = 100.0 * (1.0 + epsilon)

        total_u = (num_cores * (c_light / t_light)) + (c_heavy / t_heavy)
        capacity = float(num_cores)

        tasks = []
        for i in range(num_cores):
            tasks.append({
                "name": f"Light_Task_τ{i+1}",
                "wcet_ms": round(c_light, 2),
                "period_ms": round(t_light, 2),
                "deadline_ms": round(t_light, 2),
                "utilization": round(c_light / t_light, 4),
                "type": "Short High-Rate Stream"
            })
        tasks.append({
            "name": f"Heavy_Task_τ{num_cores+1}",
            "wcet_ms": round(c_heavy, 2),
            "period_ms": round(t_heavy, 2),
            "deadline_ms": round(t_heavy, 2),
            "utilization": round(c_heavy / t_heavy, 4),
            "type": "Heavy Compute Kernel"
        })

        return {
            "num_cores": num_cores,
            "epsilon": epsilon,
            "total_utilization": round(total_u, 4),
            "system_capacity": capacity,
            "capacity_percentage": round((total_u / capacity) * 100.0, 2),
            "tasks": tasks,
            "global_sched_verdict": "❌ MISSED DEADLINE (Dhall's Effect)",
            "global_explanation": f"Under Global RM/EDF, all {num_cores} light tasks release simultaneously and occupy all {num_cores} cores from t=0 to t={c_light:.1f}ms. The heavy task (requiring {c_heavy:.1f}ms) is delayed until t={c_light:.1f}ms, finishing at t={c_light + c_heavy:.1f}ms, which violates its deadline D={t_heavy:.1f}ms!",
            "partitioned_sched_verdict": "✅ 100% FEASIBLE (Partitioned FPPS / RMS)",
            "partitioned_explanation": f"Under Partitioned Scheduling, the heavy task is pinned to Core 0 (U={c_heavy/t_heavy*100:.1f}%), and the {num_cores} light tasks share the remaining {num_cores-1} cores. Every task comfortably meets all deadlines!"
        }

    def get_numa_topology_matrix(self) -> Dict[str, Any]:
        """
        Computes the inter-core memory latency matrix under NUMA vs UMA.
        Local memory access: ~12.5 ns.
        Remote memory access: 12.5 ns * numa_remote_penalty.
        """
        base_latency_ns = 12.5
        nodes_count = max(1, math.ceil(self.num_cores / self.cores_per_numa_node))
        matrix = []

        for core_id in range(self.num_cores):
            core_node = core_id // self.cores_per_numa_node
            row = {"Core": f"Core #{core_id}", "NUMA Node": f"Node {core_node}"}
            for n in range(nodes_count):
                if n == core_node:
                    row[f"Node {n} Latency (ns)"] = f"{base_latency_ns:.1f} (Local)"
                else:
                    remote_lat = base_latency_ns * self.numa_remote_penalty
                    row[f"Node {n} Latency (ns)"] = f"{remote_lat:.1f} (Remote {self.numa_remote_penalty}x)"
            matrix.append(row)

        return {
            "base_latency_ns": base_latency_ns,
            "nodes_count": nodes_count,
            "latency_table": matrix
        }

    def get_resource_allocation_matrix(
        self,
        alloc_result: AllocationResult,
        total_bandwidth_gbps: float = 68.4,
        total_llc_ways: int = 16
    ) -> List[Dict[str, Any]]:
        """
        Computes hardware resource allocations across cores:
        - CPU Utilization Quota (%)
        - Memory Bandwidth Reservation (GB/s) based on memory intensity
        - Last-Level Cache (LLC / L3) Way-Partitioning (Intel CAT / ARM MPAM)
        """
        records = []
        tot_u = max(1e-5, alloc_result.total_utilization)

        for core in alloc_result.cores:
            c_u = core.current_utilization
            u_share = c_u / tot_u

            # Memory intensity weight (vision tasks have high memory traffic)
            mem_weight = sum(
                (2.5 if any(k in t.name for k in ["CAM", "FACE", "GAZE", "HUD"]) else 1.0) * t.utilization
                for t in core.allocated_tasks
            )
            # Memory bandwidth allocated proportionally
            bw_gbps = round((mem_weight / max(1e-5, tot_u * 2.0)) * total_bandwidth_gbps, 2)
            bw_gbps = min(total_bandwidth_gbps, max(2.0, bw_gbps))

            # Cache ways (minimum 2 ways, max 16 ways)
            allocated_ways = max(2, min(total_llc_ways, round(u_share * total_llc_ways)))

            records.append({
                "Core ID": f"Core #{core.core_id}",
                "NUMA Node": f"Node {core.node_id}",
                "Assigned Tasks": ", ".join(t.name for t in core.allocated_tasks) if core.allocated_tasks else "Idle",
                "Tasks Count": len(core.allocated_tasks),
                "CPU Utilization (%)": round(c_u * 100.0, 1),
                "Affinity Mask": f"0x{1 << core.core_id:04X}",
                "Memory Bandwidth (GB/s)": bw_gbps,
                "LLC Cache Ways (CAT/MPAM)": f"{allocated_ways} / {total_llc_ways} Ways",
                "Schedulable (Local RTA)": "✅ Pass" if alloc_result.core_schedulable.get(core.core_id, True) else "⚠️ Tight"
            })

        return records
