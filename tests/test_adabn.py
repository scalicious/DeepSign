"""Unit test for AdaBN per spec Section 13."""

import hashlib
import torch
import torch.nn as nn
import pytest
from aquasign.adabn import adapt_bn
from aquasign.models.aquasign_net import AquaSignNet


def compute_param_hash(model: nn.Module) -> str:
    hasher = hashlib.sha256()
    for name, param in model.named_parameters():
        hasher.update(name.encode("utf-8"))
        hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


def test_adabn():
    model = AquaSignNet(num_classes=15, use_correction=False)
    model.eval()

    # Pre-populate BN statistics with initial non-trivial data
    x_init = torch.rand(4, 3, 144, 192)
    model.train()
    model(x_init)
    model.eval()

    # Get BN layers
    bns = [m for m in model.backbone.modules() if isinstance(m, nn.BatchNorm2d)]
    assert len(bns) > 0, "No BatchNorm2d layers found in backbone"
    initial_running_means = [m.running_mean.clone() for m in bns]

    # Compute parameter hash before AdaBN
    hash_before = compute_param_hash(model)

    # Dummy target-domain unlabeled loader
    # Using distinct distribution (mean shifted)
    target_batches = [
        torch.rand(8, 3, 144, 192) * 0.5 + 0.5,
        torch.rand(8, 3, 144, 192) * 0.5 + 0.5,
    ]

    # Run adapt_bn
    device = torch.device("cpu")
    adapted_model = adapt_bn(model, target_batches, device=device)

    # 1. Parameter hash identical before and after
    hash_after = compute_param_hash(adapted_model)
    assert hash_before == hash_after, "AdaBN modified trainable model parameters!"

    # 2. Running mean of BN changes
    means_changed = False
    for initial_mean, bn in zip(initial_running_means, bns):
        if not torch.allclose(initial_mean, bn.running_mean, atol=1e-4):
            means_changed = True
            break
    assert means_changed, "AdaBN failed to update running_mean statistics!"

    # 3. Dropout modules remain in eval mode after call
    dropouts = [m for m in model.modules() if isinstance(m, nn.Dropout)]
    for drop in dropouts:
        assert not drop.training, "Dropout module is in training mode after AdaBN!"

    # 4. Entire model is in eval mode
    assert not model.training, "Model is in training mode after AdaBN!"
