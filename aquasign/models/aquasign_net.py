"""AquaSign full model assembling correction, normalization buffer, backbone, and heads."""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from aquasign.models.backbone import AquaSignBackbone
from aquasign.models.correction import CorrectionModule
from aquasign.models.heads import ConfidenceHead, GestureHead


class AquaSignNet(nn.Module):
    """AquaSign complete architecture:

    Raw image x in [0, 1]
      -> Correction module (or Identity)
      -> Internal normalization (buffers mean, std)
      -> Backbone
      -> Gesture head & Confidence head (with f.detach())
    """

    def __init__(
        self,
        num_classes: int = 15,
        use_correction: bool = False,
        norm_mean: Optional[Tuple[float, float, float]] = None,
        norm_std: Optional[Tuple[float, float, float]] = None,
        dropout_p: float = 0.3,
    ):
        super().__init__()
        self.use_correction = use_correction

        # Correction module
        if use_correction:
            self.correction = CorrectionModule()
        else:
            self.correction = nn.Identity()

        # Non-trainable normalization buffers (per-channel mean/std)
        if norm_mean is None:
            norm_mean = (0.485, 0.456, 0.406)
        if norm_std is None:
            norm_std = (0.229, 0.224, 0.225)

        self.register_buffer("norm_mean", torch.tensor(norm_mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("norm_std", torch.tensor(norm_std, dtype=torch.float32).view(1, 3, 1, 1))

        # Backbone
        self.backbone = AquaSignBackbone(dropout_p=dropout_p)

        # Heads
        self.gesture_head = GestureHead(in_features=256, num_classes=num_classes)
        self.confidence_head = ConfidenceHead(in_features=256)

    def set_norm_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Update registered normalization statistics."""
        self.norm_mean.copy_(mean.view(1, 3, 1, 1))
        self.norm_std.copy_(std.view(1, 3, 1, 1))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply per-channel normalization inside model."""
        return (x - self.norm_mean) / (self.norm_std + 1e-6)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Input:
            x: (B, 3, H, W) raw image tensor in [0, 1]
            return_features: bool
        Output:
            dict containing:
              - 'logits': (B, num_classes)
              - 'conf_logit': (B, 1)
              - 'x_corrected': (B, 3, H, W)
              - 'features': (B, 256) [if return_features=True]
        """
        x_hat = self.correction(x)
        x_norm = self.normalize(x_hat)
        f = self.backbone(x_norm)
        logits = self.gesture_head(f)
        conf_logit = self.confidence_head(f)

        out = {
            "logits": logits,
            "conf_logit": conf_logit,
            "x_corrected": x_hat,
        }
        if return_features:
            out["features"] = f

        return out
