"""Adaptive Batch Normalization (AdaBN) implementation per spec Section 8."""

import copy
from typing import Any, Iterable, Optional
import torch
import torch.nn as nn


@torch.no_grad()
def adapt_bn(
    model: nn.Module,
    unlabeled_loader: Iterable[Any],
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Adapt model BatchNorm statistics to target domain images without updating weights.

    Args:
        model: AquaSignNet model
        unlabeled_loader: DataLoader yielding either tensors x or tuples (x, ...)
        device: torch device to evaluate on

    Returns:
        adapted model (same instance, or modified in-place)
    """
    if device is None:
        device = next(model.parameters()).device

    bns = [m for m in model.backbone.modules() if isinstance(m, nn.BatchNorm2d)]
    old_momentum = [m.momentum for m in bns]

    # Ensure whole model is in eval mode first (keeps Dropout and heads frozen/deterministic)
    model.eval()

    # Set only BN layers into train mode with cumulative averaging (momentum = None)
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
        m.train()

    # Forward target domain batches
    for batch in unlabeled_loader:
        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch
        model(x.to(device))

    # Restore momentum and set back to eval
    for m, mom in zip(bns, old_momentum):
        m.momentum = mom

    model.eval()
    return model
