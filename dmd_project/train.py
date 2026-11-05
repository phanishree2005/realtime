"""
Training Pipeline for DMD Driver Monitoring System
Multi-task training for Gaze Zone, Attention, and Drowsiness prediction.
"""

import argparse
import json
import os
import sys
import time
from typing import Callable, Dict, Optional

# Ensure project root in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from dmd_project.models.dmd_gazenet import DMDGazeNet
from dmd_project.utils.dataset import DMDDataset
from dmd_project.utils.openlabel_parser import (
    GAZE_ZONES,
    extract_labeled_frames,
    resolve_dmd_source,
)


def train_dmd_model(
    source_path: Optional[str] = None,
    video_path: Optional[str] = None,
    json_path: Optional[str] = None,
    output_frames_dir: str = "dmd_project/data/extracted_frames",
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 1e-3,
    save_dir: str = "dmd_project/checkpoints",
    device: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, Dict[str, float]], None]] = None,
    force_extract: bool = False
):
    os.makedirs(save_dir, exist_ok=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using compute device: {device}")

    # Resolve dataset paths from archive (.tar.gz) or directory
    if source_path:
        video_path, json_path = resolve_dmd_source(source_path)
    elif not video_path or not json_path:
        default_archive = r"C:\Users\Phanishree N\Downloads\dmd-dataset-gaze-gA-1.tar.gz"
        default_dir = r"C:\Users\Phanishree N\Downloads\dmd-dataset-gaze-gA-1"
        if os.path.exists(default_archive):
            video_path, json_path = resolve_dmd_source(default_archive)
        elif os.path.exists(default_dir):
            video_path, json_path = resolve_dmd_source(default_dir)
        else:
            raise FileNotFoundError("Could not find DMD dataset source or archive.")

    print(f"Resolved Video Source: {video_path}")
    print(f"Resolved Annotations: {json_path}")

    # Extract or load dataset metadata
    df = extract_labeled_frames(
        video_path=video_path,
        json_path=json_path,
        output_dir=output_frames_dir,
        sample_stride=5,
        max_frames_per_class=80
    )

    if len(df) == 0:
        raise ValueError("No frames extracted from video!")

    # Train / Val Split stratified by gaze zone
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['zone_id'])
    print(f"Dataset split: {len(train_df)} training samples, {len(val_df)} validation samples.")

    train_dataset = DMDDataset(train_df, is_train=True)
    val_dataset = DMDDataset(val_df, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss criteria, optimizer, scheduler
    model = DMDGazeNet(num_gaze_zones=len(GAZE_ZONES), pretrained=True).to(device)

    ce_gaze = nn.CrossEntropyLoss()
    ce_attention = nn.CrossEntropyLoss()
    ce_drowsy = nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_acc = 0.0
    history = []

    print("\n" + "="*75)
    print("Starting DMD Driver Monitoring Network Training...")
    print("="*75)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        correct_gaze = 0
        total_samples = 0

        for images, gaze_targets, att_targets, drow_targets in train_loader:
            images = images.to(device)
            gaze_targets = gaze_targets.to(device)
            att_targets = att_targets.to(device)
            drow_targets = drow_targets.to(device)

            optimizer.zero_grad()
            gaze_logits, att_logits, drow_logits = model(images)

            loss_g = ce_gaze(gaze_logits, gaze_targets)
            loss_a = ce_attention(att_logits, att_targets)
            loss_d = ce_drowsy(drow_logits, drow_targets)

            # Combined multi-task loss
            total_loss = loss_g + 0.5 * loss_a + 0.5 * loss_d
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()
            preds_g = torch.argmax(gaze_logits, dim=1)
            correct_gaze += (preds_g == gaze_targets).sum().item()
            total_samples += images.size(0)

        scheduler.step()
        train_loss /= max(1, len(train_loader))
        train_acc = (correct_gaze / total_samples) * 100.0 if total_samples else 0.0

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct_gaze = 0
        val_correct_att = 0
        val_correct_drow = 0
        val_total = 0

        with torch.no_grad():
            for images, gaze_targets, att_targets, drow_targets in val_loader:
                images = images.to(device)
                gaze_targets = gaze_targets.to(device)
                att_targets = att_targets.to(device)
                drow_targets = drow_targets.to(device)

                gaze_logits, att_logits, drow_logits = model(images)
                loss_g = ce_gaze(gaze_logits, gaze_targets)
                loss_a = ce_attention(att_logits, att_targets)
                loss_d = ce_drowsy(drow_logits, drow_targets)

                val_loss += (loss_g + 0.5 * loss_a + 0.5 * loss_d).item()

                val_correct_gaze += (torch.argmax(gaze_logits, dim=1) == gaze_targets).sum().item()
                val_correct_att += (torch.argmax(att_logits, dim=1) == att_targets).sum().item()
                val_correct_drow += (torch.argmax(drow_logits, dim=1) == drow_targets).sum().item()
                val_total += images.size(0)

        val_loss /= max(1, len(val_loader))
        gaze_acc = (val_correct_gaze / val_total) * 100.0 if val_total else 0.0
        att_acc = (val_correct_att / val_total) * 100.0 if val_total else 0.0
        drow_acc = (val_correct_drow / val_total) * 100.0 if val_total else 0.0
        duration = time.time() - t0

        print(f"Epoch [{epoch:2d}/{epochs:2d}] ({duration:.1f}s) | Train Loss: {train_loss:.4f} | Gaze Acc: {gaze_acc:5.2f}% | Attn Acc: {att_acc:5.2f}% | Drowsy Acc: {drow_acc:5.2f}%")

        record = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'gaze_acc': gaze_acc,
            'attention_acc': att_acc,
            'drowsiness_acc': drow_acc
        }
        history.append(record)

        if progress_callback:
            progress_callback(epoch, epochs, record)

        if gaze_acc > best_val_acc:
            best_val_acc = gaze_acc
            ckpt_path = os.path.join(save_dir, "best_dmd_gazenet.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'gaze_acc': best_val_acc,
                'classes': GAZE_ZONES
            }, ckpt_path)
            print(f"  --> Saved new best model checkpoint (Gaze Acc: {best_val_acc:.2f}%) to {ckpt_path}")

    # Save training history
    log_path = os.path.join(save_dir, "training_history.json")
    with open(log_path, 'w') as f:
        json.dump(history, f, indent=2)

    print("\nTraining completed successfully!")
    print(f"Best Validation Gaze Accuracy: {best_val_acc:.2f}%")
    return model, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train DMD Driver Monitoring Model")
    parser.add_argument("--source", "--archive", dest="source", type=str,
                        default=r"C:\Users\Phanishree N\Downloads\dmd-dataset-gaze-gA-1.tar.gz",
                        help="Path to DMD archive (.tar.gz) or dataset folder")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    train_dmd_model(
        source_path=args.source,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
