"""
DMDUnifiedNet: Unified 4-Task Deep Neural Network for Real-Time Driver Monitoring
Backbone: MobileNetV3-Small (Low-latency, embedded-ready)

Multi-Task Prediction Heads:
1. Gaze Zone          -> 9 cockpit zones (gaze_zone/*)
2. Drowsiness         -> 2 classes: Eyes Open vs Blinking/Closed
3. Distraction Class  -> 10 classes: safe_drive, phonecall_right, phonecall_left,
                         texting_right, texting_left, radio, drinking, hair_and_makeup,
                         reach_side, reach_backseat
4. CASDS Safety Level -> 3 classes: SAFE (Green), CAUTION (Yellow), CRITICAL (Red)
                         (Derived from gaze + distraction + drowsiness jointly)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# --- Label Definitions ---

GAZE_ZONES = [
    "left_mirror",      # 0
    "left",             # 1
    "front",            # 2 (Road ahead)
    "center_mirror",    # 3
    "front_right",      # 4
    "right_mirror",     # 5
    "right",            # 6
    "infotainment",     # 7
    "steering_wheel"    # 8
]

DISTRACTION_CLASSES = [
    "safe_drive",           # 0
    "phonecall_right",      # 1
    "phonecall_left",       # 2
    "texting_right",        # 3
    "texting_left",         # 4
    "radio",                # 5
    "drinking",             # 6
    "hair_and_makeup",      # 7
    "reach_side",           # 8
    "reach_backseat",       # 9
]

SAFETY_LEVELS = [
    "SAFE",       # 0 - Normal driving
    "CAUTION",    # 1 - Minor distraction / drowsiness
    "CRITICAL",   # 2 - High-risk: phone use, eyes closed, gaze off road
]

# Distraction classes that are CRITICAL
CRITICAL_DISTRACTIONS = {"phonecall_right", "phonecall_left", "texting_right", "texting_left"}
CAUTION_DISTRACTIONS  = {"radio", "drinking", "hair_and_makeup", "reach_side", "reach_backseat"}


def compute_safety_level(gaze_zone: str, distraction: str, is_blinking: bool) -> int:
    """
    Rule-based CASDS safety level:
    CRITICAL (2) -> Phone/texting OR eyes closed
    CAUTION  (1) -> Road gaze ok but minor distraction or blink
    SAFE     (0) -> Looking road, safe_drive
    """
    if is_blinking or distraction in CRITICAL_DISTRACTIONS:
        return 2  # CRITICAL
    elif distraction in CAUTION_DISTRACTIONS:
        return 1  # CAUTION
    elif gaze_zone not in {"front", "front_right", "center_mirror", "left_mirror", "right_mirror"}:
        return 1  # CAUTION (off-road gaze)
    return 0  # SAFE


class DMDUnifiedNet(nn.Module):
    """
    Unified 4-task Driver Monitoring Network.
    Shared MobileNetV3 backbone + 4 multi-task prediction heads.
    """

    def __init__(
        self,
        num_gaze_zones: int = 9,
        num_distraction_classes: int = 10,
        num_safety_levels: int = 3,
        pretrained: bool = True
    ):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)

        # Shared backbone (MobileNetV3-Small)
        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        in_features = 576  # MobileNetV3-Small final channels

        # Shared representation layer
        self.shared_fc = nn.Sequential(
            nn.Linear(in_features, 384),
            nn.BatchNorm1d(384),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.35)
        )

        # ---------- Task Head 1: Gaze Zone (9 classes) ----------
        self.gaze_head = nn.Sequential(
            nn.Linear(384, 192),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(192, num_gaze_zones)
        )

        # ---------- Task Head 2: Drowsiness / Blink (2 classes) ----------
        self.drowsy_head = nn.Sequential(
            nn.Linear(384, 96),
            nn.ReLU(inplace=True),
            nn.Linear(96, 2)
        )

        # ---------- Task Head 3: Distraction Activity (10 classes) ----------
        self.distraction_head = nn.Sequential(
            nn.Linear(384, 192),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(192, num_distraction_classes)
        )

        # ---------- Task Head 4: CASDS Safety Level (3 classes: Safe/Caution/Critical) ----------
        self.safety_head = nn.Sequential(
            nn.Linear(384, 96),
            nn.ReLU(inplace=True),
            nn.Linear(96, num_safety_levels)
        )

    def forward(self, x):
        feat = self.features(x)
        feat = self.avgpool(feat)
        feat = torch.flatten(feat, 1)
        emb  = self.shared_fc(feat)

        gaze_logits        = self.gaze_head(emb)
        drowsy_logits      = self.drowsy_head(emb)
        distraction_logits = self.distraction_head(emb)
        safety_logits      = self.safety_head(emb)

        return gaze_logits, drowsy_logits, distraction_logits, safety_logits
