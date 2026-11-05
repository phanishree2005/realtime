"""
PyTorch Dataset for Unified 4-Task DMD Driver Monitoring
Supports all 4 labels: gaze_zone, drowsiness, distraction_class, safety_level
"""

import os
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class UnifiedDMDDataset(Dataset):
    """
    Unified dataset for the 4-task DMDUnifiedNet model.
    Each sample returns:
        (image_tensor, gaze_label, drowsy_label, distraction_label, safety_label)
    """

    def __init__(self, df: pd.DataFrame, is_train: bool = True, img_size: int = 224):
        self.df       = df.reset_index(drop=True)
        self.is_train = is_train
        self.img_size = img_size

        self.norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # Augmentation pipeline (train only)
        self.aug = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomRotation(degrees=8),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple:
        row      = self.df.iloc[idx]
        img_path = row['file_path']

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[0] != self.img_size or img.shape[1] != self.img_size:
            img = cv2.resize(img, (self.img_size, self.img_size))

        if self.is_train:
            img_tensor = self.aug(img)          # Returns CHW float [0,1]
        else:
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        img_tensor = self.norm(img_tensor)

        gaze_label        = torch.tensor(int(row['zone_id']),          dtype=torch.long)
        drowsy_label      = torch.tensor(int(row['blinking']),         dtype=torch.long)
        distraction_label = torch.tensor(int(row['distraction_id']),   dtype=torch.long)
        safety_label      = torch.tensor(int(row['safety_level']),     dtype=torch.long)

        return img_tensor, gaze_label, drowsy_label, distraction_label, safety_label
