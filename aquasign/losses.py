"""Domain alignment losses: CORAL and Mean Matching per spec Section 7.1 and 13."""

import torch


def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute Correlation Alignment (CORAL) loss between source and target features.

    Loss = 1 / (4 * d^2) * || C_s - C_t ||_F^2
    """
    d = source.size(1)
    ns = source.size(0)
    nt = target.size(0)

    # Source covariance
    mean_s = torch.mean(source, dim=0, keepdim=True)
    xm_s = source - mean_s
    cov_s = torch.mm(xm_s.t(), xm_s) / max(ns - 1, 1)

    # Target covariance
    mean_t = torch.mean(target, dim=0, keepdim=True)
    xm_t = target - mean_t
    cov_t = torch.mm(xm_t.t(), xm_t) / max(nt - 1, 1)

    loss = torch.sum((cov_s - cov_t) ** 2) / (4.0 * (d ** 2))
    return loss


def mean_match_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute mean matching loss between source and target features.

    Loss = 1 / d * || mu_s - mu_t ||_2^2
    """
    d = source.size(1)
    mean_s = torch.mean(source, dim=0)
    mean_t = torch.mean(target, dim=0)
    loss = torch.sum((mean_s - mean_t) ** 2) / d
    return loss


def domain_alignment_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Combined domain alignment loss: CORAL + Mean Matching."""
    return coral_loss(source, target) + mean_match_loss(source, target)
