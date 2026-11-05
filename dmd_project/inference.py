"""
Inference Engine for DMD Driver Monitoring System
Performs real-time gaze zone tracking, distraction warning, and drowsiness detection with cockpit HUD.
"""

import argparse
import os
import sys
import time
from typing import Dict, Tuple

# Ensure project root in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from dmd_project.models.dmd_gazenet import DMDGazeNet
from dmd_project.utils.hud_visualizer import draw_hud
from dmd_project.utils.openlabel_parser import GAZE_ZONES


class DriverMonitor:
    def __init__(self, checkpoint_path: str = "dmd_project/checkpoints/best_dmd_gazenet.pth", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DMDGazeNet(num_gaze_zones=len(GAZE_ZONES), pretrained=False).to(self.device)

        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            self.model.load_state_dict(state_dict)
            print(f"Loaded trained DMDGazeNet weights from {checkpoint_path}")
        else:
            print(f"Checkpoint not found at {checkpoint_path}. Using initialized model.")

        self.model.eval()
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def process_frame(self, frame_bgr: np.ndarray, fps: float = 30.0) -> Tuple[np.ndarray, Dict]:
        """
        Processes a single video frame and renders cockpit HUD.
        """
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, (224, 224))

        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        tensor = self.norm(tensor).unsqueeze(0).to(self.device)

        with torch.no_grad():
            gaze_logits, att_logits, drow_logits = self.model(tensor)

            gaze_probs = F.softmax(gaze_logits, dim=1).squeeze(0)
            zone_id = int(torch.argmax(gaze_probs).item())
            zone_conf = float(gaze_probs[zone_id].item())
            predicted_zone = GAZE_ZONES[zone_id]

            att_probs = F.softmax(att_logits, dim=1).squeeze(0)
            is_attentive = bool(torch.argmax(att_probs).item() == 1)

            drow_probs = F.softmax(drow_logits, dim=1).squeeze(0)
            is_blinking = bool(torch.argmax(drow_probs).item() == 1)

        hud_frame = draw_hud(
            frame=frame_bgr,
            predicted_zone=predicted_zone,
            zone_confidence=zone_conf,
            is_attentive=is_attentive,
            is_blinking=is_blinking,
            fps=fps
        )

        meta = {
            'predicted_zone': predicted_zone,
            'zone_id': zone_id,
            'zone_confidence': zone_conf,
            'is_attentive': is_attentive,
            'is_blinking': is_blinking,
            'all_zone_probabilities': {GAZE_ZONES[i]: float(gaze_probs[i].item()) for i in range(len(GAZE_ZONES))}
        }

        return hud_frame, meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Driver Monitoring Inference")
    parser.add_argument("--video", type=str, default=r"C:\Users\Phanishree N\Downloads\dmd-dataset-gaze-gA-1\dmd\gA\1\s6\gA_1_s6_2019-03-08T09;15;15+01;00_rgb_face.mp4")
    parser.add_argument("--checkpoint", type=str, default="dmd_project/checkpoints/best_dmd_gazenet.pth")
    parser.add_argument("--max_frames", type=int, default=150)
    args = parser.parse_args()

    monitor = DriverMonitor(checkpoint_path=args.checkpoint)
    cap = cv2.VideoCapture(args.video)

    out_path = "dmd_project/output_hud_demo.avi"
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(out_path, fourcc, 25.0, (w, h))

    count = 0
    t0 = time.time()
    print("Processing video frames with real-time HUD...")

    while count < args.max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        hud_frame, meta = monitor.process_frame(frame)
        writer.write(hud_frame)
        count += 1

    cap.release()
    writer.release()
    print(f"Processed {count} frames in {time.time()-t0:.2f}s. Saved output video to: {out_path}")
