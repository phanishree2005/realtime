"""
Synthetic Automotive Perception Task-Set Generator.
Implements UUniFast algorithm (Bini & Buttazzo, 2005) for unbiased utilization generation,
realistic perception task templates grounded in nuScenes sensor rates, and clutter-based jitter scaling.
"""

import random
from typing import List, Optional, Dict, Any
import numpy as np

from .task_model import Task, TaskSet
from .nuscenes_inspector import NuScenesInspector


class SyntheticTaskGenerator:
    """
    Generates synthetic task sets for real-time automotive perception systems.
    Clearly labelled and separate from dataset data.
    """

    # Baseline automotive perception task taxonomy with nuScenes-grounded physical rates
    PERCEPTION_TASK_TEMPLATES = [
        {"name": "LIDAR_3D_DET",     "period": 50.0,  "deadline": 50.0,  "base_wcet": 12.0, "criticality": "ASIL-D", "modality": "LiDAR"},
        {"name": "CAM_FRONT_DET",    "period": 83.3,  "deadline": 83.3,  "base_wcet": 20.0, "criticality": "ASIL-D", "modality": "Camera"},
        {"name": "RADAR_FRONT_PROC", "period": 75.0,  "deadline": 75.0,  "base_wcet": 5.0,  "criticality": "ASIL-D", "modality": "Radar"},
        {"name": "CAM_SURROUND_DET", "period": 100.0, "deadline": 100.0, "base_wcet": 24.0, "criticality": "ASIL-B", "modality": "Camera"},
        {"name": "RADAR_SURR_PROC",  "period": 100.0, "deadline": 100.0, "base_wcet": 6.0,  "criticality": "ASIL-B", "modality": "Radar"},
        {"name": "SENSOR_FUSION",    "period": 50.0,  "deadline": 45.0,  "base_wcet": 6.0,  "criticality": "ASIL-D", "modality": "Fusion"},
        {"name": "EGO_LOCALIZATION", "period": 20.0,  "deadline": 20.0,  "base_wcet": 3.0,  "criticality": "ASIL-D", "modality": "Pose"},
        {"name": "OBJECT_TRACKING",  "period": 50.0,  "deadline": 50.0,  "base_wcet": 4.0,  "criticality": "ASIL-B", "modality": "Tracking"},
        {"name": "OCCUPANCY_GRID",   "period": 100.0, "deadline": 100.0, "base_wcet": 15.0, "criticality": "QM",     "modality": "Mapping"},
        {"name": "TRAJ_PLANNING",    "period": 100.0, "deadline": 90.0,  "base_wcet": 10.0, "criticality": "ASIL-D", "modality": "Planning"},
    ]

    # Typical automotive period options (ms)
    AUTOMOTIVE_PERIODS = [10.0, 20.0, 25.0, 40.0, 50.0, 75.0, 80.0, 100.0, 200.0]

    def __init__(self, seed: Optional[int] = 42, dataset_inspector: Optional[NuScenesInspector] = None):
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.inspector = dataset_inspector or NuScenesInspector()
        self.clutter_stats = self.inspector.get_scene_clutter_distribution()

    def uunifast(self, n: int, total_u: float) -> List[float]:
        """
        Bini & Buttazzo (2005) UUniFast algorithm:
        Generates an unbiased uniform distribution of utilization across n tasks with sum = total_u.
        """
        if n == 1:
            return [total_u]

        sum_u = total_u
        utilizations: List[float] = []

        for i in range(1, n):
            next_sum_u = sum_u * (random.random() ** (1.0 / (n - i)))
            utilizations.append(sum_u - next_sum_u)
            sum_u = next_sum_u

        utilizations.append(sum_u)
        return utilizations

    def generate_perception_task_set(
        self,
        target_utilization: float = 0.70,
        constrained_deadlines: bool = False,
        clutter_noise: bool = True
    ) -> TaskSet:
        """
        Generates a standard 10-task perception TaskSet grounded in nuScenes sensor rates,
        scaled to exactly target_utilization.
        """
        n = len(self.PERCEPTION_TASK_TEMPLATES)
        utils = self.uunifast(n, target_utilization)

        tasks: List[Task] = []
        for idx, tmpl in enumerate(self.PERCEPTION_TASK_TEMPLATES):
            u_i = utils[idx]
            period = tmpl["period"]
            wcet = max(0.1, round(u_i * period, 2))

            if constrained_deadlines:
                deadline = round(tmpl["deadline"] * random.uniform(0.85, 1.0), 2)
            else:
                deadline = period

            # Add minor WCET expansion if scene clutter noise is requested
            if clutter_noise:
                mean_objs = self.clutter_stats.get("mean_objects_per_frame", 45.0)
                # Jitter scales slightly with scene clutter
                clutter_factor = 1.0 + 0.05 * (mean_objs / 100.0)
                wcet = round(wcet * clutter_factor, 2)

            task = Task(
                task_id=idx + 1,
                name=tmpl["name"],
                period=period,
                deadline=deadline,
                wcet=wcet,
                bcet=round(wcet * 0.7, 2),
                criticality=tmpl["criticality"],
                modality=tmpl["modality"]
            )
            tasks.append(task)

        return TaskSet(tasks, name=f"PerceptionSet_U{target_utilization:.2f}")

    def generate_random_task_set(
        self,
        n_tasks: int = 10,
        target_utilization: float = 0.70,
        constrained_deadlines: bool = False
    ) -> TaskSet:
        """
        Generates an arbitrary-sized synthetic TaskSet with UUniFast utilization
        and automotive period distributions.
        """
        utils = self.uunifast(n_tasks, target_utilization)
        tasks: List[Task] = []

        criticality_levels = ["ASIL-D", "ASIL-C", "ASIL-B", "ASIL-A", "QM"]

        for idx in range(n_tasks):
            period = random.choice(self.AUTOMOTIVE_PERIODS)
            wcet = max(0.1, round(utils[idx] * period, 2))

            if constrained_deadlines:
                deadline = round(period * random.uniform(0.80, 1.0), 2)
            else:
                deadline = period

            task = Task(
                task_id=idx + 1,
                name=f"Task_{idx+1}",
                period=period,
                deadline=deadline,
                wcet=wcet,
                bcet=round(wcet * 0.7, 2),
                criticality=random.choice(criticality_levels),
                modality="Synthetic"
            )
            tasks.append(task)

        return TaskSet(tasks, name=f"Synthetic_N{n_tasks}_U{target_utilization:.2f}")

    def generate_batch_task_sets(
        self,
        n_sets: int = 50,
        n_tasks: int = 10,
        target_utilization: float = 0.70,
        constrained_deadlines: bool = False
    ) -> List[TaskSet]:
        """Generates a batch of distinct task sets with the same target utilization."""
        batch = []
        for _ in range(n_sets):
            batch.append(self.generate_random_task_set(
                n_tasks=n_tasks,
                target_utilization=target_utilization,
                constrained_deadlines=constrained_deadlines
            ))
        return batch
