"""Unit tests for alignment losses per spec Section 13."""

import torch
import pytest
from aquasign.losses import coral_loss, mean_match_loss, domain_alignment_loss


def test_losses():
    torch.manual_seed(42)
    f = torch.randn(32, 256)
    f2 = torch.randn(32, 256) + 2.0  # Differently distributed

    # 1. coral(f, f) == 0 and mean_match(f, f) == 0
    c_zero = coral_loss(f, f)
    m_zero = mean_match_loss(f, f)
    assert torch.isclose(c_zero, torch.tensor(0.0), atol=1e-7), f"coral(f, f) = {c_zero.item()} != 0"
    assert torch.isclose(m_zero, torch.tensor(0.0), atol=1e-7), f"mean_match(f, f) = {m_zero.item()} != 0"

    # 2. Both > 0 for differently distributed inputs
    c_diff = coral_loss(f, f2)
    m_diff = mean_match_loss(f, f2)
    assert c_diff.item() > 0, "coral(f, f2) <= 0 for distinct inputs"
    assert m_diff.item() > 0, "mean_match(f, f2) <= 0 for distinct inputs"

    # 3. Symmetric
    c_sym = coral_loss(f2, f)
    m_sym = mean_match_loss(f2, f)
    assert torch.isclose(c_diff, c_sym, atol=1e-7), "CORAL loss is not symmetric"
    assert torch.isclose(m_diff, m_sym, atol=1e-7), "Mean match loss is not symmetric"
