"""Classifier and Confidence heads per spec Section 6.3."""

import torch
import torch.nn as nn


class GestureHead(nn.Module):
    """Linear(256, 128) -> ReLU -> Linear(128, N) -> logits."""

    def __init__(self, in_features: int = 256, num_classes: int = 15):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 128)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(f)))


class ConfidenceHead(nn.Module):
    """Linear(256, 64) -> ReLU -> Linear(64, 1) -> conf_logit.

    Receives f.detach() so it cannot degrade the classifier.
    """

    def __init__(self, in_features: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 64)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        # Detach feature f to prevent gradients backpropagating into backbone
        return self.fc2(self.relu(self.fc1(f.detach())))
