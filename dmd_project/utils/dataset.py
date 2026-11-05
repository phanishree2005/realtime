"""
PyTorch Dataset for DMD Driver Monitoring
Loads extracted face frames and multi-task target labels.
"""

import os
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class DMDDataset(Dataset):
    def __init__(self, df: pd.DataFrame, is_train: bool = True, img_size: int = 224):
        self.df = df.reset_index(drop=True)
        self.is_train = is_train
        self.img_size = img_size

        self.norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img_path = row['file_path']

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[0] != self.img_size or img.shape[1] != self.img_size:
            img = cv2.resize(img, (self.img_size, self.img_size))

        # Basic data augmentation
        if self.is_train:
            # Color jittering (brightness / contrast)
            if np.random.rand() > 0.5:
                alpha = 1.0 + np.random.uniform(-0.15, 0.15)
                beta = np.random.uniform(-10, 10)
                img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)

        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_tensor = self.norm(img_tensor)

        gaze_label = torch.tensor(row['zone_id'], dtype=torch.long)
        attention_label = torch.tensor(row['attentive'], dtype=torch.long)
        blink_label = torch.tensor(row['blinking'], dtype=torch.long)

        return img_tensor, gaze_label, attention_label, blink_label
