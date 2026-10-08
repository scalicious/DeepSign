"""Learned correction module (C1) per spec Section 6.1."""

import torch
import torch.nn as nn


class CorrectionModule(nn.Module):
    """Learned correction module with per-channel gain and residual conv.

    Input x in [0,1], shape (B,3,H,W). Output x_hat in [0,1], same shape.
    Zero-initialized so that at init, correction(x) == x exactly.
    """

    def __init__(self):
        super().__init__()
        # Gain branch: GAP -> Linear(3,16) -> ReLU -> Linear(16,3) -> g = 1 + 0.5*tanh(.)
        self.gain_gap = nn.AdaptiveAvgPool2d(1)
        self.gain_fc1 = nn.Linear(3, 16)
        self.gain_relu = nn.ReLU(inplace=True)
        self.gain_fc2 = nn.Linear(16, 3)

        # Residual branch: Conv3x3(3,16) -> ReLU -> Conv3x3(16,16) -> ReLU -> Conv3x3(16,3)
        self.res_conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.res_relu1 = nn.ReLU(inplace=True)
        self.res_conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.res_relu2 = nn.ReLU(inplace=True)
        self.res_conv3 = nn.Conv2d(16, 3, kernel_size=3, padding=1)

        self._init_weights()

    def _init_weights(self):
        # Zero-initialize the last linear in gain branch
        nn.init.zeros_(self.gain_fc2.weight)
        nn.init.zeros_(self.gain_fc2.bias)

        # Zero-initialize the last conv in residual branch
        nn.init.zeros_(self.res_conv3.weight)
        nn.init.zeros_(self.res_conv3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # Gain
        pooled = self.gain_gap(x).view(b, c)
        h_gain = self.gain_relu(self.gain_fc1(pooled))
        raw_gain = self.gain_fc2(h_gain)
        g = (1.0 + 0.5 * torch.tanh(raw_gain)).view(b, c, 1, 1)

        # Residual
        h_res = self.res_relu1(self.res_conv1(x))
        h_res = self.res_relu2(self.res_conv2(h_res))
        delta = self.res_conv3(h_res)

        x_hat = torch.clamp(g * x + delta, 0.0, 1.0)
        return x_hat
