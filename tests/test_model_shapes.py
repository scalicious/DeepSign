"""Test model output tensor shapes per spec Section 13."""

import torch
import pytest
from aquasign.models.aquasign_net import AquaSignNet


def test_model_shapes():
    num_classes = 15
    model = AquaSignNet(num_classes=num_classes, use_correction=True)
    model.eval()

    x = torch.rand(2, 3, 144, 192)
    with torch.no_grad():
        out = model(x, return_features=True)

    assert "logits" in out, "Missing logits in model output"
    assert "conf_logit" in out, "Missing conf_logit in model output"
    assert "features" in out, "Missing features in model output"
    assert "x_corrected" in out, "Missing x_corrected in model output"

    assert out["logits"].shape == (2, num_classes), f"Logits shape was {out['logits'].shape}, expected (2, {num_classes})"
    assert out["conf_logit"].shape == (2, 1), f"Conf logit shape was {out['conf_logit'].shape}, expected (2, 1)"
    assert out["features"].shape == (2, 256), f"Features shape was {out['features'].shape}, expected (2, 256)"
    assert out["x_corrected"].shape == (2, 3, 144, 192), f"x_corrected shape was {out['x_corrected'].shape}, expected (2, 3, 144, 192)"
