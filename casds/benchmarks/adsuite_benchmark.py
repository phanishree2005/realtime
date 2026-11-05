"""
ADSuite Autonomous Driving Benchmark (Section IV-A & Table I).
Defines heterogeneous platform (13 PEs) and 4 types of DAG structures.
"""

from typing import Dict, List, Tuple
from ..models.task import Task
from ..models.platform import ProcessingElement, HeterogeneousPlatform
from ..models.dag import DAGApplication


# Table I: Execution time cost (ms) of task kernels on each PE-type
ADSUITE_TASK_PROFILES = {
    "DET": {
        "CPU": 3531.0,
        "GPU": 156.0,
        "DET_ASIC": 96.0,
        "TRA_ASIC": float("inf"),
        "LOC_ASIC": float("inf"),
    },
    "TRA": {
        "CPU": 1825.0,
        "GPU": 17.0,
        "DET_ASIC": float("inf"),
        "TRA_ASIC": 2.0,
        "LOC_ASIC": float("inf"),
    },
    "LOC": {
        "CPU": 165.0,
        "GPU": 95.0,
        "DET_ASIC": float("inf"),
        "TRA_ASIC": float("inf"),
        "LOC_ASIC": 10.0,
    },
    "FUS": {
        "CPU": 0.1,
        "GPU": float("inf"),
        "DET_ASIC": float("inf"),
        "TRA_ASIC": float("inf"),
        "LOC_ASIC": float("inf"),
    },
    "MiP": {
        "CPU": 1.0,
        "GPU": float("inf"),
        "DET_ASIC": float("inf"),
        "TRA_ASIC": float("inf"),
        "LOC_ASIC": float("inf"),
    },
    "MoP": {
        "CPU": 8.0,
        "GPU": float("inf"),
        "DET_ASIC": float("inf"),
        "TRA_ASIC": float("inf"),
        "LOC_ASIC": float("inf"),
    },
}


def create_adsuite_platform() -> HeterogeneousPlatform:
    """
    Creates the 13-PE heterogeneous processing platform P (Table I):
    - 8 ARM A57 CPU cores
    - 2 Maxwell GPU cores
    - 1 DET Accelerator (ASIC)
    - 1 TRA Accelerator (ASIC)
    - 1 LOC Accelerator (ASIC)
    """
    pes: List[ProcessingElement] = []
    pe_id = 0

    # 8 CPUs
    for i in range(8):
        pes.append(ProcessingElement(pe_id, f"ARM_A57_CPU_{i}", "CPU"))
        pe_id += 1

    # 2 GPUs
    for i in range(2):
        pes.append(ProcessingElement(pe_id, f"Maxwell_GPU_{i}", "GPU"))
        pe_id += 1

    # 3 Specialized ASICs
    pes.append(ProcessingElement(pe_id, "DET_Accelerator", "DET_ASIC"))
    pe_id += 1
    pes.append(ProcessingElement(pe_id, "TRA_Accelerator", "TRA_ASIC"))
    pe_id += 1
    pes.append(ProcessingElement(pe_id, "LOC_Accelerator", "LOC_ASIC"))
    pe_id += 1

    # Communication delays between different PE types (0 to 136 ms)
    # CPU-CPU: ~2 ms, CPU-GPU (PCIe): ~15 ms, CPU-ASIC: ~8 ms, GPU-ASIC: ~25 ms
    comm_matrix: Dict[Tuple[int, int], float] = {}
    for p1 in pes:
        for p2 in pes:
            if p1.pe_id == p2.pe_id:
                comm_matrix[(p1.pe_id, p2.pe_id)] = 0.0
            else:
                if p1.pe_type == "CPU" and p2.pe_type == "CPU":
                    comm_matrix[(p1.pe_id, p2.pe_id)] = 2.0
                elif (p1.pe_type == "CPU" and p2.pe_type == "GPU") or (p1.pe_type == "GPU" and p2.pe_type == "CPU"):
                    comm_matrix[(p1.pe_id, p2.pe_id)] = 15.0
                elif (p1.pe_type == "CPU" and "ASIC" in p2.pe_type) or ("ASIC" in p1.pe_type and p2.pe_type == "CPU"):
                    comm_matrix[(p1.pe_id, p2.pe_id)] = 8.0
                elif (p1.pe_type == "GPU" and "ASIC" in p2.pe_type) or ("ASIC" in p1.pe_type and p2.pe_type == "GPU"):
                    comm_matrix[(p1.pe_id, p2.pe_id)] = 25.0
                else:
                    comm_matrix[(p1.pe_id, p2.pe_id)] = 10.0

    return HeterogeneousPlatform(pes, comm_matrix, default_inter_pe_comm=15.0)


def create_adsuite_dag(
    app_id: int,
    dag_type: str = "Type 1",
    criticality: int = 1,
    arrival_time: float = 0.0,
    relative_deadline: float = 400.0
) -> DAGApplication:
    """
    Constructs one of the 4 DAG structures from ADSuite benchmark (Fig. 4).
    """
    tasks: Dict[int, Task] = {}
    edges: List[Tuple[int, int]] = []
    edge_costs: Dict[Tuple[int, int], float] = {}

    if dag_type == "Type 1":
        # 6 tasks: DET -> TRA -> FUS -> MiP -> MoP; LOC -> FUS
        kernel_names = ["DET", "TRA", "LOC", "FUS", "MiP", "MoP"]
        for idx, k in enumerate(kernel_names):
            tasks[idx] = Task(idx, k, k, dict(ADSUITE_TASK_PROFILES[k]), app_id)
        edges = [(0, 1), (1, 3), (2, 3), (3, 4), (4, 5)]
        # Inter-task data transmission costs in [0, 136] ms (Section IV-A1)
        edge_costs = {
            (0, 1): 25.0,  # Detection to Tracking (bounding boxes + features)
            (1, 3): 10.0,  # Tracking to Fusion
            (2, 3): 5.0,   # Localization to Fusion
            (3, 4): 2.0,   # Fusion to Mission Planning
            (4, 5): 2.0    # Mission Planning to Motion Planning
        }

    elif dag_type == "Type 2":
        # 8 tasks: Dual camera & lidar perception
        # 0: DET_cam, 1: DET_lidar, 2: TRA_cam, 3: TRA_lidar, 4: LOC, 5: FUS, 6: MiP, 7: MoP
        task_specs = [
            (0, "DET_cam", "DET"),
            (1, "DET_lidar", "DET"),
            (2, "TRA_cam", "TRA"),
            (3, "TRA_lidar", "TRA"),
            (4, "LOC", "LOC"),
            (5, "FUS", "FUS"),
            (6, "MiP", "MiP"),
            (7, "MoP", "MoP"),
        ]
        for tid, name, k in task_specs:
            tasks[tid] = Task(tid, name, k, dict(ADSUITE_TASK_PROFILES[k]), app_id)
        edges = [(0, 2), (1, 3), (2, 5), (3, 5), (4, 5), (5, 6), (6, 7)]
        edge_costs = {
            (0, 2): 25.0, (1, 3): 35.0,
            (2, 5): 10.0, (3, 5): 15.0, (4, 5): 5.0,
            (5, 6): 2.0, (6, 7): 2.0
        }

    elif dag_type == "Type 3":
        # 7 tasks: Cross perception and dual planning
        # 0: DET, 1: TRA, 2: LOC, 3: FUS, 4: MiP, 5: MoP, 6: CON (Control)
        task_specs = [
            (0, "DET", "DET"),
            (1, "TRA", "TRA"),
            (2, "LOC", "LOC"),
            (3, "FUS", "FUS"),
            (4, "MiP", "MiP"),
            (5, "MoP", "MoP"),
            (6, "CON", "MoP"),  # Uses MoP kernel profile for low latency control
        ]
        for tid, name, k in task_specs:
            tasks[tid] = Task(tid, name, k, dict(ADSUITE_TASK_PROFILES[k]), app_id)
        edges = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (3, 5), (4, 5), (5, 6)]
        edge_costs = {
            (0, 1): 20.0, (0, 2): 15.0, (1, 3): 10.0, (2, 3): 5.0,
            (3, 4): 2.0, (3, 5): 2.0, (4, 5): 2.0, (5, 6): 1.0
        }

    else:  # Type 4
        # 9 tasks: Dual perception, fusion, split planning and arbitration
        task_specs = [
            (0, "DET_1", "DET"),
            (1, "DET_2", "DET"),
            (2, "TRA_1", "TRA"),
            (3, "TRA_2", "TRA"),
            (4, "LOC", "LOC"),
            (5, "FUS", "FUS"),
            (6, "MiP_Global", "MiP"),
            (7, "MiP_Local", "MiP"),
            (8, "MoP", "MoP"),
        ]
        for tid, name, k in task_specs:
            tasks[tid] = Task(tid, name, k, dict(ADSUITE_TASK_PROFILES[k]), app_id)
        edges = [(0, 2), (1, 3), (2, 5), (3, 5), (4, 5), (5, 6), (5, 7), (6, 8), (7, 8)]
        edge_costs = {
            (0, 2): 25.0, (1, 3): 25.0,
            (2, 5): 10.0, (3, 5): 10.0, (4, 5): 5.0,
            (5, 6): 2.0, (5, 7): 2.0, (6, 8): 2.0, (7, 8): 2.0
        }

    return DAGApplication(
        app_id=app_id,
        dag_type=dag_type,
        tasks=tasks,
        edges=edges,
        edge_costs=edge_costs,
        criticality=criticality,
        arrival_time=arrival_time,
        relative_deadline=relative_deadline
    )
