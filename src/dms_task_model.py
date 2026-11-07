"""
Real-Time Task Model for Low-Latency Embedded Driver Monitoring System (DMS).
Compliant with ISO 26262 (ASIL A-D) and Euro NCAP 2026 In-Cabin Safety Requirements.

Defines the 8 core tasks in the in-vehicle DMS processing pipeline:
1. Camera Frame Acquisition & ISP Preprocessing (ASIL-B)
2. Face ROI Localization & Landmark Tracking (ASIL-B)
3. Cockpit 9-Zone Gaze Tracking Head (ASIL-B)
4. Drowsiness, Eye Closure (PERCLOS) & Blink Head (ASIL-C)
5. Distraction Activity Classification (10 Classes) (ASIL-B)
6. CASDS Dynamic Safety Arbitrator & Risk Engine (ASIL-D)
7. CAN Bus Emergency Warning & HMI Dispatcher (ASIL-D)
8. Cockpit HUD Telemetry & Gaze Visualization (QM)
"""

from typing import List, Dict, Any, Optional
from src.task_model import Task, TaskSet


DMS_TASK_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "task_id": 1,
        "name": "T1_CAM_ISP",
        "label": "Camera Ingestion & ISP Preprocessing",
        "description": "Raw RGB/IR frame capture, demosaicing, auto-exposure, and lens distortion correction.",
        "base_period_ms": 33.33,   # 30 FPS
        "base_deadline_ms": 33.33,
        "base_wcet_ms": 4.80,
        "criticality": "ASIL-B",
        "modality": "Vision Sensor",
        "is_vision_rate": True
    },
    {
        "task_id": 2,
        "name": "T2_FACE_LOC",
        "label": "Face Detection & Landmark Extraction",
        "description": "Driver face bounding box localization and 68-point facial landmark alignment.",
        "base_period_ms": 33.33,
        "base_deadline_ms": 33.33,
        "base_wcet_ms": 6.20,
        "criticality": "ASIL-B",
        "modality": "Vision Preprocessing",
        "is_vision_rate": True
    },
    {
        "task_id": 3,
        "name": "T3_GAZE_9Z",
        "label": "Cockpit Gaze Tracking Head (9 Zones)",
        "description": "Multi-task MobileNetV3 head classifying driver gaze across 9 cockpit zones.",
        "base_period_ms": 33.33,
        "base_deadline_ms": 33.33,
        "base_wcet_ms": 7.00,
        "criticality": "ASIL-B",
        "modality": "Vision AI",
        "is_vision_rate": True
    },
    {
        "task_id": 4,
        "name": "T4_PERCLOS",
        "label": "Drowsiness & Blink Detection Head",
        "description": "Calculates PERCLOS (Percentage of Eye Closure), micro-sleeps, and yawning intervals.",
        "base_period_ms": 50.00,   # 20 Hz
        "base_deadline_ms": 50.00,
        "base_wcet_ms": 4.00,
        "criticality": "ASIL-C",
        "modality": "Safety AI",
        "is_vision_rate": False
    },
    {
        "task_id": 5,
        "name": "T5_DISTRACT",
        "label": "Distraction Activity Classifier (10 Classes)",
        "description": "Deep classifier recognizing texting, phone call, drinking, reaching, and mirror adjustments.",
        "base_period_ms": 100.00,  # 10 Hz
        "base_deadline_ms": 100.00,
        "base_wcet_ms": 11.50,
        "criticality": "ASIL-B",
        "modality": "Behavior AI",
        "is_vision_rate": False
    },
    {
        "task_id": 6,
        "name": "T6_CASDS_ARB",
        "label": "CASDS Dynamic Safety Arbitrator",
        "description": "Context-Aware Slack arbitrator fusing gaze, drowsiness, and distraction into SAFE/CAUTION/CRITICAL.",
        "base_period_ms": 33.33,
        "base_deadline_ms": 33.33,
        "base_wcet_ms": 1.40,
        "criticality": "ASIL-D",
        "modality": "Real-Time Arbitration",
        "is_vision_rate": True
    },
    {
        "task_id": 7,
        "name": "T7_CAN_ALERT",
        "label": "CAN Bus Emergency Warning Dispatcher",
        "description": "Transmits high-priority CAN/FlexRay frames triggering cabin acoustic alarms and seat haptics.",
        "base_period_ms": 20.00,   # 50 Hz
        "base_deadline_ms": 15.00,  # Tight deadline (D < T) for critical alert responsiveness
        "base_wcet_ms": 0.70,
        "criticality": "ASIL-D",
        "modality": "Vehicle Bus HMI",
        "is_vision_rate": False
    },
    {
        "task_id": 8,
        "name": "T8_HUD_RENDER",
        "label": "Cockpit HUD Overlay & Telemetry",
        "description": "Renders real-time HUD graphics, 2D gaze map, eye bounding boxes, and system latency telemetry.",
        "base_period_ms": 33.33,
        "base_deadline_ms": 33.33,
        "base_wcet_ms": 3.80,
        "criticality": "QM",
        "modality": "Cockpit Graphics",
        "is_vision_rate": True
    }
]


def create_dms_task_set(
    camera_fps: float = 30.0,
    load_multiplier: float = 1.0,
    name: str = "DMS_RealTime_TaskSet"
) -> TaskSet:
    """
    Constructs a TaskSet representing the 8-task Driver Monitoring System pipeline.

    Args:
        camera_fps: In-cabin camera frame rate (e.g. 15, 30, 60 FPS).
                    Adjusts periods for vision-rate tasks (T = 1000 / camera_fps).
        load_multiplier: Scaling factor for WCET to simulate processor speed or algorithmic complexity.
        name: Name tag for the TaskSet.

    Returns:
        TaskSet containing the 8 parameterized Task objects.
    """
    vision_period = round(1000.0 / max(5.0, camera_fps), 2)
    tasks: List[Task] = []

    for defn in DMS_TASK_DEFINITIONS:
        if defn["is_vision_rate"]:
            period = vision_period
            deadline = vision_period
        else:
            period = defn["base_period_ms"]
            deadline = defn["base_deadline_ms"]

        wcet = round(defn["base_wcet_ms"] * load_multiplier, 2)
        bcet = round(wcet * 0.7, 2)

        task = Task(
            task_id=defn["task_id"],
            name=defn["name"],
            period=period,
            deadline=deadline,
            wcet=wcet,
            bcet=bcet,
            priority=0,
            criticality=defn["criticality"],
            modality=defn["modality"]
        )
        tasks.append(task)

    return TaskSet(tasks=tasks, name=name)


def get_dms_task_metadata() -> List[Dict[str, Any]]:
    """Returns copy of static metadata for UI documentation and table presentation."""
    return [dict(d) for d in DMS_TASK_DEFINITIONS]
