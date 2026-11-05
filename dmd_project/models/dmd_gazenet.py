"""
DMDGazeNet: Multi-Task Deep Neural Network for Real-Time Driver Monitoring
Predicts:
1. Gaze Zone (9 classes)
2. Attention State (Attentive vs Distracted)
3. Blink / Drowsiness State (Eyes Open vs Closed/Blink)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class DMDGazeNet(nn.Module):
    def __init__(self, num_gaze_zones: int = 9, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)

        # Feature extractor
        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Shared representation layer (576 channels for MobileNetV3-small)
        in_features = 576
        self.shared_fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

        # Multi-task heads:
        # 1. Gaze Zone Head (9 classes)
        self.gaze_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_gaze_zones)
        )

        # 2. Attention / Distraction Head (2 classes: Attentive vs Distracted)
        self.attention_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)
        )

        # 3. Drowsiness / Blink Head (2 classes: Eyes Open vs Blinking)
        self.drowsy_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        feat = self.features(x)
        feat = self.avgpool(feat)
        feat = torch.flatten(feat, 1)

        embedding = self.shared_fc(feat)

        gaze_logits = self.gaze_head(embedding)
        attention_logits = self.attention_head(embedding)
        drowsy_logits = self.drowsy_head(embedding)

        return gaze_logits, attention_logits, drowsy_logits
