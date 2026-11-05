"""
OpenLabel Parser for DMD (Driver Monitoring Dataset)
Parses frame-level gaze zone annotations and blink/drowsiness intervals.
"""

import glob
import json
import os
import tarfile
import zipfile
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


GAZE_ZONES = [
    "left_mirror",      # 0
    "left",             # 1
    "front",            # 2 (Looking at road)
    "center_mirror",    # 3
    "front_right",      # 4
    "right_mirror",     # 5
    "right",            # 6
    "infotainment",     # 7
    "steering_wheel"    # 8
]

ZONE_TO_ID = {z: idx for idx, z in enumerate(GAZE_ZONES)}


def resolve_dmd_source(
    source_path: str,
    extract_dir: str = "dmd_project/data/archive_extracted"
) -> Tuple[str, str]:
    """
    Given a dataset path (which can be a .tar.gz, .tar, .zip archive or a folder),
    resolves and returns the paths to:
    (face_video_path, gaze_json_path)
    Extracts only the required video and annotation members if needed.
    """
    source_path = os.path.abspath(source_path.strip().strip('"').strip("'"))
    
    # 1. If it's a direct tar.gz or tar archive
    if os.path.isfile(source_path) and (source_path.endswith('.tar.gz') or source_path.endswith('.tar') or source_path.endswith('.tgz')):
        # Check if sibling uncompressed directory exists
        stem = source_path
        for ext in ['.tar.gz', '.tgz', '.tar']:
            if stem.endswith(ext):
                stem = stem[:-len(ext)]
                break
        
        if os.path.isdir(stem):
            # Recursively find inside sibling directory
            res = _search_dmd_directory(stem)
            if res:
                return res

        # Otherwise extract required members to extract_dir
        os.makedirs(extract_dir, exist_ok=True)
        # Check if already extracted in extract_dir
        res = _search_dmd_directory(extract_dir)
        if res:
            return res
            
        print(f"Extracting face video and gaze annotations from archive: {source_path}...")
        with tarfile.open(source_path, 'r:*') as tar:
            members_to_extract = []
            for member in tar.getmembers():
                name_lower = member.name.lower()
                if ('rgb_face' in name_lower and name_lower.endswith('.mp4')) or \
                   ('ann_gaze' in name_lower and name_lower.endswith('.json')):
                    members_to_extract.append(member)
            
            if not members_to_extract:
                raise ValueError(f"No face video or gaze annotations found inside archive {source_path}")
            
            tar.extractall(path=extract_dir, members=members_to_extract)
            print(f"Successfully extracted {len(members_to_extract)} files into {extract_dir}")
            
        res = _search_dmd_directory(extract_dir)
        if res:
            return res
        raise FileNotFoundError(f"Extracted files not found in {extract_dir}")

    # 2. If it's a zip file
    if os.path.isfile(source_path) and source_path.endswith('.zip'):
        os.makedirs(extract_dir, exist_ok=True)
        res = _search_dmd_directory(extract_dir)
        if res:
            return res
        with zipfile.ZipFile(source_path, 'r') as zf:
            for item in zf.namelist():
                item_lower = item.lower()
                if ('rgb_face' in item_lower and item_lower.endswith('.mp4')) or \
                   ('ann_gaze' in item_lower and item_lower.endswith('.json')):
                    zf.extract(item, path=extract_dir)
        res = _search_dmd_directory(extract_dir)
        if res:
            return res
        raise FileNotFoundError(f"Could not find video and json inside zip: {source_path}")

    # 3. If it's a directory
    if os.path.isdir(source_path):
        res = _search_dmd_directory(source_path)
        if res:
            return res
        raise FileNotFoundError(f"Could not find rgb_face.mp4 and ann_gaze.json inside {source_path}")

    # 4. If it's a direct file (e.g. video)
    if os.path.isfile(source_path):
        folder = os.path.dirname(source_path)
        res = _search_dmd_directory(folder)
        if res:
            return res

    raise FileNotFoundError(f"Invalid DMD dataset source: {source_path}")


def _search_dmd_directory(directory: str) -> Optional[Tuple[str, str]]:
    """Helper to locate rgb_face video and ann_gaze json in a directory tree."""
    video_path = None
    json_path = None

    for root, _, files in os.walk(directory):
        for f in files:
            f_lower = f.lower()
            if 'rgb_face' in f_lower and f_lower.endswith('.mp4'):
                video_path = os.path.join(root, f)
            elif 'ann_gaze' in f_lower and f_lower.endswith('.json'):
                json_path = os.path.join(root, f)

    # Fallback to any mp4 and any json if specific names not matched
    if not video_path or not json_path:
        for root, _, files in os.walk(directory):
            for f in files:
                f_lower = f.lower()
                if not video_path and f_lower.endswith('.mp4'):
                    video_path = os.path.join(root, f)
                if not json_path and f_lower.endswith('.json') and 'openlabel' in f_lower or 'gaze' in f_lower:
                    json_path = os.path.join(root, f)

    if video_path and json_path:
        return (os.path.abspath(video_path), os.path.abspath(json_path))
    return None


def parse_dmd_gaze_json(json_path: str) -> Tuple[Dict[int, str], List[Tuple[int, int]]]:
    """
    Parses OpenLabel JSON file to extract:
    1. frame_id -> gaze_zone label
    2. list of (frame_start, frame_end) for blink intervals
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    ol = data.get('openlabel', {})
    actions = ol.get('actions', {})

    frame_to_zone: Dict[int, str] = {}
    blink_intervals: List[Tuple[int, int]] = []

    for action_id, action_info in actions.items():
        atype = action_info.get('type', '')
        intervals = action_info.get('frame_intervals', [])

        if atype.startswith('gaze_zone/'):
            zone_name = atype.replace('gaze_zone/', '').strip()
            if zone_name in ZONE_TO_ID:
                for interval in intervals:
                    f_start = interval.get('frame_start', 0)
                    f_end = interval.get('frame_end', 0)
                    for fid in range(f_start, f_end + 1):
                        frame_to_zone[fid] = zone_name

        elif atype == 'blinks/blinking':
            for interval in intervals:
                f_start = interval.get('frame_start', 0)
                f_end = interval.get('frame_end', 0)
                blink_intervals.append((f_start, f_end))

    return frame_to_zone, blink_intervals


def extract_labeled_frames(
    video_path: str,
    json_path: str,
    output_dir: str = "dmd_project/data/extracted_frames",
    sample_stride: int = 5,
    max_frames_per_class: int = 80
) -> pd.DataFrame:
    """
    Extracts frames from the DMD face video aligned with gaze annotations.
    Balances classes and creates a training metadata CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "dataset_metadata.csv")

    if os.path.exists(csv_path):
        print(f"Pre-extracted dataset metadata found at {csv_path}. Loading existing...")
        return pd.read_csv(csv_path)

    frame_to_zone, blink_intervals = parse_dmd_gaze_json(json_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    class_counts = {z: 0 for z in GAZE_ZONES}
    records = []

    def is_blinking(frame_idx: int) -> bool:
        for start, end in blink_intervals:
            if start <= frame_idx <= end:
                return True
        return False

    current_frame = 0
    print(f"Extracting frames from {os.path.basename(video_path)} with stride {sample_stride}...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame in frame_to_zone and current_frame % sample_stride == 0:
            zone = frame_to_zone[current_frame]
            if class_counts[zone] < max_frames_per_class:
                img_name = f"frame_{current_frame:05d}_{zone}.jpg"
                img_path = os.path.join(output_dir, img_name)
                
                # Resize for fast deep learning inference & training (e.g. 224x224)
                resized = cv2.resize(frame, (224, 224))
                cv2.imwrite(img_path, resized)

                blinking = is_blinking(current_frame)
                is_attentive = (zone in ["front", "front_right", "center_mirror", "left_mirror", "right_mirror"])

                records.append({
                    'frame_id': current_frame,
                    'file_path': img_path,
                    'gaze_zone': zone,
                    'zone_id': ZONE_TO_ID[zone],
                    'blinking': int(blinking),
                    'attentive': int(is_attentive)
                })
                class_counts[zone] += 1

        current_frame += 1

    cap.release()

    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    print(f"Successfully extracted {len(df)} balanced frames to {output_dir}")
    print(f"Class distribution:\n{df['gaze_zone'].value_counts()}")
    return df
