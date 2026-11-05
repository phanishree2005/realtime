"""
Unified Inference Engine for 4-Task DMD Driver Monitoring System
Predicts: Gaze Zone | Drowsiness | Distraction Class | CASDS Safety Level
"""

import os
import sys
import time
from typing import Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from dmd_project.models.dmd_unified_net import (
    DISTRACTION_CLASSES,
    GAZE_ZONES,
    SAFETY_LEVELS,
    DMDUnifiedNet,
    compute_safety_level,
)

# Color palette for safety levels
SAFETY_COLORS_BGR = {
    0: (0, 200, 80),    # SAFE - Green
    1: (0, 165, 255),   # CAUTION - Orange
    2: (0, 40, 220),    # CRITICAL - Red
}
SAFETY_ICONS = {0: "🟢", 1: "🟡", 2: "🔴"}

DISTRACTION_FRIENDLY = {
    "safe_drive":      "✅ Safe Drive",
    "phonecall_right": "📱 Phone Call (Right)",
    "phonecall_left":  "📱 Phone Call (Left)",
    "texting_right":   "💬 Texting (Right)",
    "texting_left":    "💬 Texting (Left)",
    "radio":           "📻 Radio Adjustment",
    "drinking":        "🥤 Drinking",
    "hair_and_makeup": "💄 Hair / Makeup",
    "reach_side":      "🤚 Reaching Side",
    "reach_backseat":  "🤚 Reaching Backseat",
}


class UnifiedDriverMonitor:
    """
    Real-time inference engine for 4-task driver monitoring.
    Loads the trained DMDUnifiedNet checkpoint and processes frames.
    """

    def __init__(
        self,
        checkpoint_path: str = "dmd_project/checkpoints/best_unified_model.pth",
        fallback_checkpoint: str = "dmd_project/checkpoints/best_dmd_gazenet.pth",
        device: str = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DMDUnifiedNet(
            num_gaze_zones=len(GAZE_ZONES),
            num_distraction_classes=len(DISTRACTION_CLASSES),
            num_safety_levels=len(SAFETY_LEVELS),
            pretrained=False
        ).to(self.device)

        loaded = False
        if os.path.exists(checkpoint_path):
            try:
                ckpt = torch.load(checkpoint_path, map_location=self.device)
                state = ckpt.get('model_state_dict', ckpt)
                self.model.load_state_dict(state)
                print(f"[OK] Loaded Unified 4-Task model from: {checkpoint_path}")
                loaded = True
            except Exception as e:
                print(f"[WARN] Could not load unified checkpoint: {e}")

        if not loaded:
            print(f"[INFO] Unified checkpoint not found, using random weights. Train the model first.")

        self.model.eval()
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def process_frame(self, frame_bgr: np.ndarray, fps: float = 30.0) -> Tuple[np.ndarray, Dict]:
        """Run 4-task inference on a single BGR frame. Returns HUD frame + metadata dict."""
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized  = cv2.resize(img_rgb, (224, 224))

        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        tensor = self.norm(tensor).unsqueeze(0).to(self.device)

        with torch.no_grad():
            g_logits, drow_logits, dist_logits, safe_logits = self.model(tensor)

            # Gaze
            gaze_probs   = F.softmax(g_logits,    dim=1).squeeze(0)
            zone_id      = int(torch.argmax(gaze_probs).item())
            zone_conf    = float(gaze_probs[zone_id].item())
            gaze_zone    = GAZE_ZONES[zone_id]

            # Drowsiness
            drow_probs   = F.softmax(drow_logits, dim=1).squeeze(0)
            is_blinking  = bool(torch.argmax(drow_probs).item() == 1)
            drow_conf    = float(drow_probs[1].item())

            # Distraction
            dist_probs   = F.softmax(dist_logits, dim=1).squeeze(0)
            dist_id      = int(torch.argmax(dist_probs).item())
            dist_conf    = float(dist_probs[dist_id].item())
            distraction  = DISTRACTION_CLASSES[dist_id]

            # CASDS Safety Level
            safe_probs   = F.softmax(safe_logits, dim=1).squeeze(0)
            safety_id    = int(torch.argmax(safe_probs).item())
            safety_conf  = float(safe_probs[safety_id].item())
            safety_level = SAFETY_LEVELS[safety_id]

        is_attentive = gaze_zone in {"front", "front_right", "center_mirror", "left_mirror", "right_mirror"}

        # Draw unified HUD
        hud_frame = self._draw_unified_hud(
            frame_bgr, gaze_zone, zone_conf, is_attentive,
            is_blinking, drow_conf, distraction, dist_conf,
            safety_id, safety_level, safety_conf, fps
        )

        meta = {
            # Gaze
            'gaze_zone':             gaze_zone,
            'predicted_zone':        gaze_zone,
            'zone_id':               zone_id,
            'zone_confidence':       zone_conf,
            'gaze_confidence':       zone_conf,
            'all_zone_probabilities': {GAZE_ZONES[i]: float(gaze_probs[i]) for i in range(len(GAZE_ZONES))},
            # Attention
            'is_attentive':          is_attentive,
            # Drowsiness
            'drowsy':                is_blinking,
            'is_blinking':           is_blinking,
            'drowsiness_confidence': drow_conf,
            # Distraction
            'distraction_class':     distraction,
            'distraction_id':        dist_id,
            'distraction_confidence': dist_conf,
            'all_distraction_probs': {DISTRACTION_CLASSES[i]: float(dist_probs[i]) for i in range(len(DISTRACTION_CLASSES))},
            # Safety
            'safety_level':          safety_level,
            'safety_id':             safety_id,
            'safety_confidence':     safety_conf,
        }
        return hud_frame, meta

    def _draw_unified_hud(
        self, frame, gaze_zone, zone_conf, is_attentive,
        is_blinking, drow_conf, distraction, dist_conf,
        safety_id, safety_level, safety_conf, fps
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        canvas = frame.copy()
        safety_color = SAFETY_COLORS_BGR[safety_id]

        # Top HUD bar
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (10, 10, 18), -1)
        cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)

        # Safety level header  
        safe_label = f"SAFETY: {safety_level}  ({safety_conf*100:.0f}%)"
        cv2.putText(canvas, safe_label, (15, 33), cv2.FONT_HERSHEY_DUPLEX, 0.85, safety_color, 2, cv2.LINE_AA)

        # Critical border alert
        if safety_id == 2:
            cv2.rectangle(canvas, (0, 0), (w-1, h-1), (0, 30, 220), 5)
        elif safety_id == 1:
            cv2.rectangle(canvas, (0, 0), (w-1, h-1), (0, 140, 255), 3)

        # Gaze zone
        gaze_label = f"Gaze: {gaze_zone.replace('_',' ').upper()} ({zone_conf*100:.0f}%)"
        cv2.putText(canvas, gaze_label, (15, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 2, cv2.LINE_AA)

        # Drowsiness (top right)
        eye_color = (0, 165, 255) if is_blinking else (0, 210, 80)
        eye_text  = f"DROWSY: {drow_conf*100:.0f}%" if is_blinking else "EYES OPEN"
        cv2.putText(canvas, eye_text, (w - 280, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.7, eye_color, 2, cv2.LINE_AA)

        # Distraction (top right)
        dist_friendly = DISTRACTION_CLASSES[DISTRACTION_CLASSES.index(distraction)] if distraction in DISTRACTION_CLASSES else distraction
        dist_color    = (0, 200, 80) if distraction == 'safe_drive' else (0, 100, 255)
        dist_text     = f"{dist_friendly.replace('_',' ').upper()[:22]}"
        cv2.putText(canvas, dist_text, (w - 280, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.6, dist_color, 2, cv2.LINE_AA)

        # FPS
        fps_text = f"{fps:.1f} FPS | {1000/max(1,fps):.0f}ms"
        cv2.putText(canvas, fps_text, (w - 280, 93), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

        # Bottom cockpit mini-map
        map_h, map_w = 115, 215
        mx, my = w - map_w - 15, h - map_h - 15
        ov2 = canvas.copy()
        cv2.rectangle(ov2, (mx, my), (mx+map_w, my+map_h), (8, 8, 14), -1)
        cv2.addWeighted(ov2, 0.72, canvas, 0.28, 0, canvas)
        cv2.rectangle(canvas, (mx, my), (mx+map_w, my+map_h), (80, 80, 80), 1)
        cv2.putText(canvas, "COCKPIT GAZE MAP", (mx+25, my+17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 190, 190), 1)

        zone_layout = {
            "left_mirror":  (mx+22,  my+42),
            "left":         (mx+50,  my+48),
            "front":        (mx+107, my+36),
            "center_mirror":(mx+107, my+58),
            "steering_wheel":(mx+107,my+88),
            "infotainment": (mx+150, my+75),
            "front_right":  (mx+162, my+44),
            "right_mirror": (mx+192, my+38),
            "right":        (mx+182, my+58),
        }
        for zname, (zx, zy) in zone_layout.items():
            color = (0, 255, 255) if zname == gaze_zone else (60, 60, 60)
            cv2.circle(canvas, (zx, zy), 7 if zname == gaze_zone else 4, color, -1)
            if zname == gaze_zone:
                cv2.circle(canvas, (zx, zy), 10, (0, 180, 200), 2)

        return canvas
