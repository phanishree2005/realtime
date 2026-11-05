"""
Cockpit Head-Up Display (HUD) Visualizer for Driver Monitoring System
Overlays real-time AI gaze zone, distraction warning, and drowsiness alerts onto the video frame.
"""

import cv2
import numpy as np
from dmd_project.utils.openlabel_parser import GAZE_ZONES


# Cockpit layout coordinate map for visual HUD display
COCKPIT_ZONES = {
    "left_mirror": (60, 100),
    "left": (160, 100),
    "front": (320, 70),
    "center_mirror": (320, 130),
    "front_right": (480, 100),
    "right_mirror": (580, 100),
    "right": (520, 150),
    "infotainment": (400, 180),
    "steering_wheel": (240, 180)
}


def draw_hud(
    frame: np.ndarray,
    predicted_zone: str,
    zone_confidence: float,
    is_attentive: bool,
    is_blinking: bool,
    fps: float = 30.0
) -> np.ndarray:
    """
    Renders an in-vehicle Driver Monitoring HUD overlay on the video frame.
    """
    h, w = frame.shape[:2]
    canvas = frame.copy()

    # Top Status Bar (Semi-transparent HUD banner)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, 85), (20, 20, 25), -1)
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

    # Status 1: Driver Attention Banner
    if is_attentive:
        status_color = (0, 220, 100)  # Green
        status_text = "STATUS: ATTENTIVE"
    else:
        status_color = (0, 60, 255)   # Red
        status_text = "WARNING: DISTRACTED!"
        # Distraction red border alert around entire frame
        cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), status_color, 4)

    cv2.putText(canvas, status_text, (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.9, status_color, 2, cv2.LINE_AA)

    # Status 2: Gaze Zone
    zone_label = f"Gaze Zone: {predicted_zone.upper()} ({zone_confidence*100:.1f}%)"
    cv2.putText(canvas, zone_label, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)

    # Status 3: Eye State / Drowsiness
    if is_blinking:
        eye_color = (0, 165, 255)  # Orange
        eye_text = "EYES: BLINK / CLOSED"
    else:
        eye_color = (0, 255, 120)  # Green
        eye_text = "EYES: OPEN"

    cv2.putText(canvas, eye_text, (w - 320, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, eye_color, 2, cv2.LINE_AA)

    # Performance FPS
    fps_text = f"Latency: {1000.0/max(1.0, fps):.1f}ms | {fps:.1f} FPS"
    cv2.putText(canvas, fps_text, (w - 320, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)

    # Bottom Mini-Cockpit Map
    map_h, map_w = 120, 220
    map_x, map_y = w - map_w - 20, h - map_h - 20
    overlay_map = canvas.copy()
    cv2.rectangle(overlay_map, (map_x, map_y), (map_x + map_w, map_y + map_h), (10, 10, 15), -1)
    cv2.addWeighted(overlay_map, 0.7, canvas, 0.3, 0, canvas)
    cv2.rectangle(canvas, (map_x, map_y), (map_x + map_w, map_y + map_h), (100, 100, 100), 1)
    cv2.putText(canvas, "COCKPIT GAZE MAP", (map_x + 30, map_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # Draw mini zones on cockpit map
    zone_layout = {
        "left_mirror": (map_x + 25, map_y + 40),
        "left": (map_x + 55, map_y + 45),
        "front": (map_x + 110, map_y + 35),
        "center_mirror": (map_x + 110, map_y + 60),
        "steering_wheel": (map_x + 110, map_y + 90),
        "infotainment": (map_x + 155, map_y + 75),
        "front_right": (map_x + 165, map_y + 45),
        "right_mirror": (map_x + 195, map_y + 40),
        "right": (map_x + 185, map_y + 60)
    }

    for zname, (zx, zy) in zone_layout.items():
        if zname == predicted_zone:
            cv2.circle(canvas, (zx, zy), 7, (0, 255, 255), -1)
            cv2.circle(canvas, (zx, zy), 9, (0, 180, 255), 2)
        else:
            cv2.circle(canvas, (zx, zy), 4, (80, 80, 80), -1)

    return canvas
