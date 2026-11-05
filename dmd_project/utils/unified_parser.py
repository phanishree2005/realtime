"""
Unified Multi-Dataset Parser for DMD Driver Monitoring System
Parses annotations from:
  - Gaze Dataset     (gA-*) -> gaze_zone annotations
  - Drowsiness Dataset (gZ-*) -> blinks/blinking annotations
  - Distraction Dataset (gC-*) -> driver_actions/* + gaze_on_road/* annotations
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from dmd_project.models.dmd_unified_net import (
    DISTRACTION_CLASSES,
    GAZE_ZONES,
    SAFETY_LEVELS,
    compute_safety_level,
)

ZONE_TO_ID = {z: i for i, z in enumerate(GAZE_ZONES)}
DISTRACTION_TO_ID = {d: i for i, d in enumerate(DISTRACTION_CLASSES)}

# All known distraction label mappings from OpenLabel
DRIVER_ACTION_MAP = {
    "driver_actions/safe_drive":           "safe_drive",
    "driver_actions/phonecall_right":      "phonecall_right",
    "driver_actions/phonecall_left":       "phonecall_left",
    "driver_actions/texting_right":        "texting_right",
    "driver_actions/texting_left":         "texting_left",
    "driver_actions/radio":                "radio",
    "driver_actions/drinking":             "drinking",
    "driver_actions/hair_and_makeup":      "hair_and_makeup",
    "driver_actions/reach_side":           "reach_side",
    "driver_actions/reach_backseat":       "reach_backseat",
    "driver_actions/talking_to_passenger": "safe_drive",  # similar to safe
    "driver_actions/unclassified":         "safe_drive",
}


def parse_gaze_json(json_path: str) -> Tuple[Dict[int, str], List[Tuple[int, int]]]:
    """Parse gaze_zone/* annotations and blink intervals."""
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)
    actions = data.get('openlabel', {}).get('actions', {})
    frame_to_zone: Dict[int, str] = {}
    blink_intervals: List[Tuple[int, int]] = []

    for _, info in actions.items():
        atype = info.get('type', '')
        intervals = info.get('frame_intervals', [])
        if atype.startswith('gaze_zone/'):
            zone = atype.replace('gaze_zone/', '').strip()
            if zone in ZONE_TO_ID:
                for iv in intervals:
                    for fid in range(iv['frame_start'], iv['frame_end'] + 1):
                        frame_to_zone[fid] = zone
        elif atype == 'blinks/blinking':
            for iv in intervals:
                blink_intervals.append((iv['frame_start'], iv['frame_end']))
    return frame_to_zone, blink_intervals


def parse_drowsiness_json(json_path: str) -> Tuple[Dict[int, str], List[Tuple[int, int]]]:
    """
    Parse drowsiness annotations from gZ dataset (driver 33).
    Identifies blink, eye closure, and yawn intervals.
    Gaze zone defaults to 'front' since driver is facing road.
    """
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)
    actions = data.get('openlabel', {}).get('actions', {})
    drowsy_intervals: List[Tuple[int, int]] = []
    max_frame = 0

    for _, info in actions.items():
        atype = info.get('type', '')
        intervals = info.get('frame_intervals', [])
        # Drowsiness indicators: blinks, closed eyes, closing eyes, yawning
        if (atype == 'blinks/blinking' or 
            atype in ('eyes_state/close', 'eyes_state/closing') or 
            atype.startswith('yawning/')):
            for iv in intervals:
                drowsy_intervals.append((iv['frame_start'], iv['frame_end']))
                if iv['frame_end'] > max_frame:
                    max_frame = iv['frame_end']
        for iv in intervals:
            if iv['frame_end'] > max_frame:
                max_frame = iv['frame_end']

    # Assign default 'front' gaze for all valid frame indices in interval
    frame_to_zone: Dict[int, str] = {fid: 'front' for fid in range(0, max_frame + 1, 5)}
    return frame_to_zone, drowsy_intervals


def parse_distraction_json(json_path: str) -> Tuple[Dict[int, str], Dict[int, str], List[Tuple[int, int]]]:
    """
    Parse distraction annotation:
    Returns:
        frame_to_distraction: frame_id -> distraction class name
        frame_to_gaze_on_road: frame_id -> 'looking_road' | 'not_looking_road'
        blink_intervals: list of (start, end)
    """
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)
    actions = data.get('openlabel', {}).get('actions', {})
    frame_to_distraction: Dict[int, str] = {}
    frame_to_gaze_on_road: Dict[int, str] = {}
    blink_intervals: List[Tuple[int, int]] = []

    for _, info in actions.items():
        atype = info.get('type', '')
        intervals = info.get('frame_intervals', [])
        if atype in DRIVER_ACTION_MAP:
            mapped = DRIVER_ACTION_MAP[atype]
            for iv in intervals:
                for fid in range(iv['frame_start'], iv['frame_end'] + 1):
                    frame_to_distraction[fid] = mapped
        elif atype.startswith('gaze_on_road/'):
            val = atype.replace('gaze_on_road/', '').strip()
            for iv in intervals:
                for fid in range(iv['frame_start'], iv['frame_end'] + 1):
                    frame_to_gaze_on_road[fid] = val
        elif atype == 'blinks/blinking':
            for iv in intervals:
                blink_intervals.append((iv['frame_start'], iv['frame_end']))
    return frame_to_distraction, frame_to_gaze_on_road, blink_intervals


def _is_blinking(frame_idx: int, blink_intervals: List[Tuple[int, int]]) -> bool:
    for start, end in blink_intervals:
        if start <= frame_idx <= end:
            return True
    return False


def extract_unified_frames(
    video_path: str,
    json_path: str,
    dataset_type: str,   # 'gaze', 'drowsiness', or 'distraction'
    output_dir: str = "dmd_project/data/unified_frames",
    sample_stride: int = 5,
    max_frames_per_class: int = 100,
    session_tag: str = ""
) -> pd.DataFrame:
    """
    Extracts frames from a DMD video with multi-task labels.
    Supports gaze, drowsiness, and distraction dataset types.
    Returns a DataFrame with all 4 task labels.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"metadata_{session_tag}_{dataset_type}.csv")

    if os.path.exists(csv_path):
        print(f"Found cached metadata: {csv_path}")
        return pd.read_csv(csv_path)

    print(f"Extracting {dataset_type} frames from {os.path.basename(video_path)} ...")

    # --- Parse annotations ---
    frame_to_zone: Dict[int, str] = {}
    frame_to_distraction: Dict[int, str] = {}
    blink_intervals: List[Tuple[int, int]] = []
    frame_to_gaze_on_road: Dict[int, str] = {}

    if dataset_type == 'gaze':
        frame_to_zone, blink_intervals = parse_gaze_json(json_path)
        # Default distraction for gaze dataset
        for fid in frame_to_zone:
            frame_to_distraction[fid] = 'safe_drive'
    elif dataset_type == 'drowsiness':
        frame_to_zone, blink_intervals = parse_drowsiness_json(json_path)
        for fid in frame_to_zone:
            frame_to_distraction[fid] = 'safe_drive'
    elif dataset_type == 'distraction':
        frame_to_distraction, frame_to_gaze_on_road, blink_intervals = parse_distraction_json(json_path)
        # Assign synthetic gaze zone from road gaze annotation
        for fid, gaze_status in frame_to_gaze_on_road.items():
            if gaze_status == 'looking_road':
                frame_to_zone[fid] = 'front'
            else:
                frame_to_zone[fid] = 'infotainment'
        # Fill distraction for any frame with gaze but no distraction
        for fid in frame_to_zone:
            if fid not in frame_to_distraction:
                frame_to_distraction[fid] = 'safe_drive'

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    class_counts_distraction = {d: 0 for d in DISTRACTION_CLASSES}
    drowsy_count = 0
    alert_count = 0
    records = []
    current_frame = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fid = current_frame
        current_frame += 1

        if fid % sample_stride != 0:
            continue

        # Need at minimum zone + distraction label
        if fid not in frame_to_zone and fid not in frame_to_distraction:
            continue

        zone = frame_to_zone.get(fid, 'front')
        distraction = frame_to_distraction.get(fid, 'safe_drive')

        if distraction not in DISTRACTION_TO_ID:
            continue

        blinking = _is_blinking(fid, blink_intervals)

        # Dataset-specific balancing
        if dataset_type == 'drowsiness':
            if blinking:
                if drowsy_count >= max_frames_per_class:
                    continue
                drowsy_count += 1
            else:
                if alert_count >= max_frames_per_class:
                    continue
                alert_count += 1
        else:
            if class_counts_distraction[distraction] >= max_frames_per_class:
                continue
            class_counts_distraction[distraction] += 1

        is_attentive = zone in {"front", "front_right", "center_mirror", "left_mirror", "right_mirror"}
        safety_level = compute_safety_level(zone, distraction, blinking)

        img_name = f"{session_tag}_{dataset_type}_frame_{fid:06d}_{distraction}.jpg"
        img_path = os.path.join(output_dir, img_name)

        if not os.path.exists(img_path):
            resized = cv2.resize(frame, (224, 224))
            cv2.imwrite(img_path, resized)

        records.append({
            'frame_id':        fid,
            'session_tag':     session_tag,
            'dataset_type':    dataset_type,
            'file_path':       img_path,
            'gaze_zone':       zone,
            'zone_id':         ZONE_TO_ID.get(zone, 2),
            'distraction':     distraction,
            'distraction_id':  DISTRACTION_TO_ID[distraction],
            'blinking':        int(blinking),
            'attentive':       int(is_attentive),
            'safety_level':    safety_level,   # 0=SAFE, 1=CAUTION, 2=CRITICAL
        })

    cap.release()
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)

    print(f"  -> Extracted {len(df)} frames for {session_tag}/{dataset_type}")
    if len(df) > 0:
        print(f"  -> Distraction distribution:\n{df['distraction'].value_counts().to_string()}")
    return df


def build_unified_dataset(output_dir: str = "dmd_project/data/unified_frames") -> pd.DataFrame:
    """
    Builds a combined DataFrame from all 3 DMD datasets:
    - gaze (gA-1): 5340 frames
    - drowsiness (gZ-33 s5): 2 sessions
    - distraction (gC-13): 3 sessions
    """
    os.makedirs(output_dir, exist_ok=True)
    merged_csv = os.path.join(output_dir, "unified_dataset.csv")

    if os.path.exists(merged_csv):
        print(f"Loaded cached unified dataset: {merged_csv}")
        return pd.read_csv(merged_csv)

    all_frames = []

    # --- 1. Gaze Dataset (gA-1) ---
    gaze_video  = r"C:\Users\Phanishree N\Downloads\dmd-dataset-gaze-gA-1\dmd\gA\1\s6\gA_1_s6_2019-03-08T09;15;15+01;00_rgb_face.mp4"
    gaze_json   = r"C:\Users\Phanishree N\Downloads\dmd-dataset-gaze-gA-1\dmd\gA\1\s6\gA_1_s6_2019-03-08T09;15;15+01;00_rgb_ann_gaze.json"
    if os.path.exists(gaze_video):
        df = extract_unified_frames(gaze_video, gaze_json, 'gaze', output_dir, session_tag="gA1_s6", max_frames_per_class=80)
        all_frames.append(df)
    else:
        print("WARNING: Gaze dataset not found, skipping.")

    # --- 2. Drowsiness Dataset (gZ-33) ---
    drow_base = r"C:\Users\Phanishree N\Downloads\dmd-dataset-drowsiness-gZ-33\dmd\gZ\33\s5"
    drow_sessions = [
        ("gZ_33_s5_2019-04-04T09;29;07+02;00_rgb_face.mp4", "gZ_33_s5_2019-04-04T09;29;07+02;00_rgb_ann_drowsiness.json", "gZ33_morning"),
        ("gZ_33_s5_2019-04-04T15;02;29+02;00_rgb_face.mp4", "gZ_33_s5_2019-04-04T15;02;29+02;00_rgb_ann_drowsiness.json", "gZ33_afternoon"),
    ]
    for vid_name, ann_name, tag in drow_sessions:
        vid_path = os.path.join(drow_base, vid_name)
        ann_path = os.path.join(drow_base, ann_name)
        if os.path.exists(vid_path):
            df = extract_unified_frames(vid_path, ann_path, 'drowsiness', output_dir, session_tag=tag, max_frames_per_class=80)
            all_frames.append(df)
        else:
            print(f"WARNING: Drowsiness session {tag} not found.")

    # --- 3. Distraction Dataset (gC-13) ---
    dist_base = r"C:\Users\Phanishree N\Downloads\dmd-dataset-distraction-gC-13\dmd\gC\13"
    dist_sessions = [
        ("s1", "gC_13_s1_2019-03-04T10;26;12+01;00_rgb_face.mp4", "gC_13_s1_2019-03-04T10;26;12+01;00_rgb_ann_distraction.json", "gC13_s1"),
        ("s2", "gC_13_s2_2019-03-04T10;11;37+01;00_rgb_face.mp4", "gC_13_s2_2019-03-04T10;11;37+01;00_rgb_ann_distraction.json", "gC13_s2"),
        ("s3", "gC_13_s3_2019-03-12T10;06;34+01;00_rgb_face.mp4", "gC_13_s3_2019-03-12T10;06;34+01;00_rgb_ann_distraction.json", "gC13_s3"),
    ]
    for sess_folder, vid_name, ann_name, tag in dist_sessions:
        vid_path = os.path.join(dist_base, sess_folder, vid_name)
        ann_path = os.path.join(dist_base, sess_folder, ann_name)
        if os.path.exists(vid_path):
            df = extract_unified_frames(vid_path, ann_path, 'distraction', output_dir, session_tag=tag, max_frames_per_class=100)
            all_frames.append(df)
        else:
            print(f"WARNING: Distraction session {tag} not found.")

    if not all_frames:
        raise RuntimeError("No datasets found! Check that DMD data is in Downloads.")

    unified_df = pd.concat(all_frames, ignore_index=True)
    unified_df.to_csv(merged_csv, index=False)
    print(f"\n=== Unified Dataset Built ===")
    print(f"Total samples : {len(unified_df)}")
    print(f"Distraction   :\n{unified_df['distraction'].value_counts().to_string()}")
    print(f"Gaze zones    :\n{unified_df['gaze_zone'].value_counts().to_string()}")
    print(f"Safety levels : {unified_df['safety_level'].value_counts().to_dict()}")
    return unified_df
