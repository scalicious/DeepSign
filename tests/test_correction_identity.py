"""Test that correction module is exactly identity at initialization per spec Section 13."""

import torch
import pytest
from aquasign.models.correction import CorrectionModule


def test_correction_identity():
    corr = CorrectionModule()
    corr.eval()

    # Generate random images in [0, 1]
    x = torch.rand(4, 3, 144, 192)

    with torch.no_grad():
        x_hat = corr(x)

    max_diff = torch.max(torch.abs(x_hat - x)).item()
    assert max_diff <= 1e-6, f"Correction module at init deviates from identity by {max_diff} > 1e-6"
