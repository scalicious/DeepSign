"""Train-time augmentation pipeline per spec Section 5.5."""

import random
from typing import Optional, Tuple
import torch
import torchvision.transforms.v2 as T
from torchvision.transforms import functional as TF


class AquaSignAugment:
    """Train-time augmentation: RandomResizedCrop + Brightness/Contrast Jitter + Optional Haze Augment.

    - No horizontal flip (gestures are handedness-dependent).
    - No hue/saturation jitter (would fight correction module).
    """

    def __init__(
        self,
        input_hw: Tuple[int, int] = (144, 192),
        crop_scale: Tuple[float, float] = (0.85, 1.0),
        crop_ratio: Tuple[float, float] = (0.9, 1.1),
        jitter_val: float = 0.1,
        p_haze: float = 0.5,
        haze_sim: Optional[object] = None,
    ):
        self.input_hw = tuple(input_hw)
        self.crop_scale = crop_scale
        self.crop_ratio = crop_ratio
        self.jitter_val = jitter_val
        self.p_haze = p_haze
        self.haze_sim = haze_sim

        self.crop_transform = T.RandomResizedCrop(
            size=self.input_hw,
            scale=self.crop_scale,
            ratio=self.crop_ratio,
            antialias=True,
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply augmentations to float32 tensor x in [0, 1] of shape (3, H, W) or (B, 3, H, W)."""
        is_batched = x.ndim == 4
        if not is_batched:
            x = x.unsqueeze(0)

        # 1. Random Resized Crop
        x = self.crop_transform(x)

        # 2. Brightness and Contrast Jitter (0.1)
        b_factor = 1.0 + random.uniform(-self.jitter_val, self.jitter_val)
        c_factor = 1.0 + random.uniform(-self.jitter_val, self.jitter_val)
        x = TF.adjust_brightness(x, b_factor)
        x = TF.adjust_contrast(x, c_factor)
        x = torch.clamp(x, 0.0, 1.0)

        # 3. Haze augmentation if simulator provided
        if self.haze_sim is not None and random.random() < self.p_haze:
            severity = random.choice([0, 1, 2, 3])
            if severity > 0:
                x = self.haze_sim(x, severity=severity)

        if not is_batched:
            x = x.squeeze(0)
        return x
