"""AquaSign 4-block CNN backbone per spec Section 6.2."""

import torch
import torch.nn as nn


class ConvBnRelu(nn.Module):
    """Conv3x3 -> BN -> ReLU block."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class BackboneBlock(nn.Module):
    """Two ConvBnRelu layers followed by MaxPool2d(2)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layer1 = ConvBnRelu(in_channels, out_channels)
        self.layer2 = ConvBnRelu(out_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x)
        return x


class AquaSignBackbone(nn.Module):
    """Four-stage backbone producing 256-d feature vector f.

    Channels: 3 -> 32 -> 64 -> 128 -> 256
    """

    def __init__(self, dropout_p: float = 0.3):
        super().__init__()
        self.block1 = BackboneBlock(3, 32)
        self.block2 = BackboneBlock(32, 64)
        self.block3 = BackboneBlock(64, 128)
        self.block4 = BackboneBlock(128, 256)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout_p)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input x: (B, 3, H, W) normalized.

        Returns feature vector f: (B, 256).
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        f = self.gap(x).flatten(1)
        f = self.dropout(f)
        return f
