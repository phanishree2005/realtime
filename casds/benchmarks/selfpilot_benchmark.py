"""
SelfPilot Autonomous Driving Benchmark (Section V & Table IV).
Defines real vehicle hardware platform (TZTEK GEACX2 + Advantech MIC-75G30)
and the full SelfPilot autonomous driving DAG architecture.
"""

from typing import Dict, List, Tuple
from ..models.task import Task
from ..models.platform import ProcessingElement, HeterogeneousPlatform
from ..models.dag import DAGApplication


# Table IV: Time profile of SelfPilot task kernels on each PE-type (ms)
SELFPILOT_TASK_PROFILES = {
    "LOD": {  # LiDAR Object Detection
        "CPU": 60.0,
        "GPU1": 17.0,
        "GPU2": float("inf"),
    },
    "COD": {  # Camera Object Detection
        "CPU": 62.0,
        "GPU1": 25.0,
        "GPU2": float("inf"),
    },
    "ODF": {  # Object Detection Fusion
        "CPU": 7.0,
        "GPU1": float("inf"),
        "GPU2": float("inf"),
    },
    "LD": {   # Lane Detection
        "CPU": 22.0,
        "GPU1": 9.0,
        "GPU2": float("inf"),
    },
    "MOT": {  # Multiple Object Tracking
        "CPU": 29.0,
        "GPU1": float("inf"),
        "GPU2": float("inf"),
    },
    "LOC": {  # Localization
        "CPU": 60.0,
        "GPU1": 23.0,
        "GPU2": 8.0,
    },
    "MiP": {  # Mission Planning
        "CPU": 1.0,
        "GPU1": float("inf"),
        "GPU2": float("inf"),
    },
    "MoP": {  # Motion Planning
        "CPU": 69.0,
        "GPU1": float("inf"),
        "GPU2": float("inf"),
    },
    "CON": {  # Control
        "CPU": 11.0,
        "GPU1": float("inf"),
        "GPU2": float("inf"),
    },
}


def create_selfpilot_platform() -> HeterogeneousPlatform:
    """
    Creates real hardware platform (Table IV & Section V-A):
    - 8 Arm A78 CPUs
    - 2 Ampere GPUs (GPU1 on TZTEK GEACX2)
    - 1 RTX 3060 GPU (GPU2 on MIC-75G30 connected via ETH)
    """
    pes: List[ProcessingElement] = []
    pe_id = 0

    # 8 Arm A78 CPUs
    for i in range(8):
        pes.append(ProcessingElement(pe_id, f"Arm_A78_CPU_{i}", "CPU"))
        pe_id += 1

    # 2 Ampere GPUs (Jetson AGX Orin)
    for i in range(2):
        pes.append(ProcessingElement(pe_id, f"Ampere_GPU1_{i}", "GPU1"))
        pe_id += 1

    # 1 RTX 3060 GPU (IPC connected via ETH)
    pes.append(ProcessingElement(pe_id, "RTX3060_GPU2", "GPU2"))
    pe_id += 1

    # Communication delays in [0, 12] ms (Section V-B)
    # Intra-board (CPU-GPU1 PCIe): ~2.5 ms, Inter-unit (ETH to GPU2): ~8-12 ms
    comm_matrix: Dict[Tuple[int, int], float] = {}
    for p1 in pes:
        for p2 in pes:
            if p1.pe_id == p2.pe_id:
                comm_matrix[(p1.pe_id, p2.pe_id)] = 0.0
            else:
                if p1.pe_type == "CPU" and p2.pe_type == "CPU":
                    comm_matrix[(p1.pe_id, p2.pe_id)] = 1.0
                elif (p1.pe_type == "CPU" and p2.pe_type == "GPU1") or (p1.pe_type == "GPU1" and p2.pe_type == "CPU"):
                    comm_matrix[(p1.pe_id, p2.pe_id)] = 2.5
                elif (p1.pe_type == "GPU1" and p2.pe_type == "GPU1"):
                    comm_matrix[(p1.pe_id, p2.pe_id)] = 3.0
                elif p1.pe_type == "GPU2" or p2.pe_type == "GPU2":
                    comm_matrix[(p1.pe_id, p2.pe_id)] = 10.0  # ETH link to external GPU
                else:
                    comm_matrix[(p1.pe_id, p2.pe_id)] = 4.0

    return HeterogeneousPlatform(pes, comm_matrix, default_inter_pe_comm=3.0)


def create_selfpilot_dag(
    app_id: int,
    criticality: int = 1,
    arrival_time: float = 0.0,
    relative_deadline: float = 200.0
) -> DAGApplication:
    """
    Creates SelfPilot application DAG (Fig. 9(d)):
    9 tasks:
    0: COD (Camera Object Detection)
    1: LOD (LiDAR Object Detection)
    2: MOT (Multi-Object Tracking)
    3: ODF (Object Detection Fusion)
    4: LD  (Lane Detection)
    5: LOC (Localization)
    6: MiP (Mission Planning)
    7: MoP (Motion Planning)
    8: CON (Control)

    Edges:
    COD -> MOT, LOD -> MOT, MOT -> ODF
    ODF -> MoP, LD -> MoP, LOC -> MoP
    LOC -> MiP, MiP -> MoP, MoP -> CON
    """
    tasks: Dict[int, Task] = {}
    kernel_names = ["COD", "LOD", "MOT", "ODF", "LD", "LOC", "MiP", "MoP", "CON"]
    for idx, k in enumerate(kernel_names):
        tasks[idx] = Task(idx, k, k, dict(SELFPILOT_TASK_PROFILES[k]), app_id)

    edges = [
        (0, 2),  # COD -> MOT
        (1, 2),  # LOD -> MOT
        (2, 3),  # MOT -> ODF
        (3, 7),  # ODF -> MoP
        (4, 7),  # LD -> MoP
        (5, 7),  # LOC -> MoP
        (5, 6),  # LOC -> MiP
        (6, 7),  # MiP -> MoP
        (7, 8),  # MoP -> CON
    ]

    edge_costs = {
        (0, 2): 4.0,  # camera detections to MOT
        (1, 2): 6.0,  # point clouds to MOT
        (2, 3): 2.0,  # tracks to ODF
        (3, 7): 1.5,  # fused objects to MoP
        (4, 7): 1.0,  # lane boundaries to MoP
        (5, 7): 0.5,  # pose to MoP
        (5, 6): 0.5,  # pose to MiP
        (6, 7): 0.5,  # route waypoints to MoP
        (7, 8): 0.5,  # trajectory to CON
    }

    return DAGApplication(
        app_id=app_id,
        dag_type="SelfPilot",
        tasks=tasks,
        edges=edges,
        edge_costs=edge_costs,
        criticality=criticality,
        arrival_time=arrival_time,
        relative_deadline=relative_deadline
    )
