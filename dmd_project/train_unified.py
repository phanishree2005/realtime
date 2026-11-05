"""
Unified 4-Task Training Pipeline for DMD Driver Monitoring System
Trains DMDUnifiedNet on Gaze + Drowsiness + Distraction datasets simultaneously.

Tasks:
  1. Gaze Zone Classification      (9 classes)
  2. Drowsiness / Blink Detection  (2 classes)
  3. Distraction Activity Class    (10 classes)
  4. CASDS Safety Level            (3 classes: SAFE / CAUTION / CRITICAL)
"""

import argparse
import json
import os
import sys
import time
from typing import Callable, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from dmd_project.models.dmd_unified_net import (
    DISTRACTION_CLASSES,
    GAZE_ZONES,
    SAFETY_LEVELS,
    DMDUnifiedNet,
)
from dmd_project.utils.unified_dataset import UnifiedDMDDataset
from dmd_project.utils.unified_parser import build_unified_dataset


def train_unified_model(
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 1e-3,
    save_dir: str = "dmd_project/checkpoints",
    output_frames_dir: str = "dmd_project/data/unified_frames",
    device: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, Dict], None]] = None,
):
    os.makedirs(save_dir, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # ---------------------------------------------------------------
    # Build unified dataset from all 3 DMD sources
    # ---------------------------------------------------------------
    print("\n=== Building Unified Multi-Dataset ===")
    df = build_unified_dataset(output_dir=output_frames_dir)
    print(f"Total samples available: {len(df)}")

    if len(df) == 0:
        raise ValueError("No samples extracted. Check dataset paths.")

    # Stratify split by distraction_id (most imbalanced label)
    try:
        train_df, val_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df['distraction_id']
        )
    except ValueError:
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    print(f"\nDataset split: {len(train_df)} train | {len(val_df)} validation")

    train_ds = UnifiedDMDDataset(train_df, is_train=True)
    val_ds   = UnifiedDMDDataset(val_df,   is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    # ---------------------------------------------------------------
    # Model, loss, optimizer
    # ---------------------------------------------------------------
    model = DMDUnifiedNet(
        num_gaze_zones=len(GAZE_ZONES),
        num_distraction_classes=len(DISTRACTION_CLASSES),
        num_safety_levels=len(SAFETY_LEVELS),
        pretrained=True
    ).to(device)

    ce_gaze        = nn.CrossEntropyLoss()
    ce_drowsy      = nn.CrossEntropyLoss()
    ce_distraction = nn.CrossEntropyLoss()
    ce_safety      = nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.05)

    best_val_gaze_acc = 0.0
    history: List[Dict] = []

    # Task loss weights
    W_GAZE, W_DROWSY, W_DIST, W_SAFETY = 1.5, 1.0, 2.0, 1.5

    print(f"\n{'='*70}")
    print(" Starting Unified 4-Task Training: Gaze | Drowsy | Distraction | Safety")
    print(f"{'='*70}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            imgs, gaze_t, drowsy_t, dist_t, safety_t = [x.to(device) for x in batch]
            optimizer.zero_grad()

            g_logits, drow_logits, dist_logits, safe_logits = model(imgs)

            loss = (
                W_GAZE   * ce_gaze(g_logits, gaze_t) +
                W_DROWSY * ce_drowsy(drow_logits, drowsy_t) +
                W_DIST   * ce_distraction(dist_logits, dist_t) +
                W_SAFETY * ce_safety(safe_logits, safety_t)
            )
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        correct_gaze = correct_drowsy = correct_dist = correct_safety = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                imgs, gaze_t, drowsy_t, dist_t, safety_t = [x.to(device) for x in batch]
                g_logits, drow_logits, dist_logits, safe_logits = model(imgs)

                loss = (
                    W_GAZE   * ce_gaze(g_logits, gaze_t) +
                    W_DROWSY * ce_drowsy(drow_logits, drowsy_t) +
                    W_DIST   * ce_distraction(dist_logits, dist_t) +
                    W_SAFETY * ce_safety(safe_logits, safety_t)
                )
                val_loss += loss.item()

                correct_gaze   += (torch.argmax(g_logits,    dim=1) == gaze_t).sum().item()
                correct_drowsy += (torch.argmax(drow_logits, dim=1) == drowsy_t).sum().item()
                correct_dist   += (torch.argmax(dist_logits, dim=1) == dist_t).sum().item()
                correct_safety += (torch.argmax(safe_logits, dim=1) == safety_t).sum().item()
                total += imgs.size(0)

        val_loss /= max(1, len(val_loader))
        n = max(1, total)
        gaze_acc    = correct_gaze   / n * 100
        drowsy_acc  = correct_drowsy / n * 100
        dist_acc    = correct_dist   / n * 100
        safety_acc  = correct_safety / n * 100
        duration    = time.time() - t0

        print(
            f"Epoch [{epoch:2d}/{epochs}] ({duration:.1f}s) | "
            f"Loss: {train_loss:.4f} | "
            f"Gaze: {gaze_acc:.1f}% | "
            f"Drowsy: {drowsy_acc:.1f}% | "
            f"Distraction: {dist_acc:.1f}% | "
            f"Safety: {safety_acc:.1f}%"
        )

        record = {
            'epoch': epoch,
            'train_loss': round(train_loss, 4),
            'val_loss':   round(val_loss, 4),
            'gaze_acc':        round(gaze_acc, 2),
            'drowsiness_acc':  round(drowsy_acc, 2),
            'distraction_acc': round(dist_acc, 2),
            'safety_acc':      round(safety_acc, 2),
        }
        history.append(record)

        if progress_callback:
            progress_callback(epoch, epochs, record)

        # Save best checkpoint by gaze accuracy
        if gaze_acc > best_val_gaze_acc:
            best_val_gaze_acc = gaze_acc
            ckpt = os.path.join(save_dir, "best_unified_model.pth")
            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'gaze_acc':         best_val_gaze_acc,
                'distraction_acc':  dist_acc,
                'safety_acc':       safety_acc,
                'drowsiness_acc':   drowsy_acc,
                'gaze_zones':       GAZE_ZONES,
                'distraction_classes': DISTRACTION_CLASSES,
                'safety_levels':    SAFETY_LEVELS,
            }, ckpt)
            print(f"  --> New best checkpoint saved (Gaze {best_val_gaze_acc:.1f}%)")

    log_path = os.path.join(save_dir, "unified_training_history.json")
    with open(log_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Training Complete! Best Gaze Accuracy: {best_val_gaze_acc:.2f}%")
    print(f"{'='*70}")
    return model, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Unified 4-Task DMD Model")
    parser.add_argument("--epochs",     type=int,   default=5,    help="Training epochs")
    parser.add_argument("--batch_size", type=int,   default=8,    help="Batch size")
    parser.add_argument("--lr",         type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    train_unified_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
