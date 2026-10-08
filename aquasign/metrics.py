"""Metrics computation per spec Section 11: macro-F1, selective accuracy, AURC, AUROC."""

from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[str, float]:
    """Compute accuracy and macro-F1 over all classes."""
    acc = float(np.mean(y_true == y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
    }


def compute_domain_gap(src_macro_f1: float, tgt_macro_f1: float) -> float:
    """Domain gap = F1_src - F1_tgt."""
    return float(src_macro_f1 - tgt_macro_f1)


def selective_accuracy(y_true: np.ndarray, y_pred: np.ndarray, conf: np.ndarray, coverage: float) -> float:
    """Selective accuracy at coverage fraction c in (0, 1].

    Keeps top c fraction of samples with highest confidence.
    """
    assert 0.0 < coverage <= 1.0, f"Coverage must be in (0, 1], got {coverage}"
    n = len(y_true)
    if n == 0:
        return 0.0
    k = max(1, int(np.round(n * coverage)))

    # Sort descending by confidence
    order = np.argsort(-conf)
    top_indices = order[:k]

    return float(np.mean(y_true[top_indices] == y_pred[top_indices]))


def compute_risk_coverage_curve(
    y_true: np.ndarray, y_pred: np.ndarray, conf: np.ndarray, coverages: Optional[List[float]] = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute risk (1 - selective_accuracy) across coverages and trapezoidal AURC."""
    if coverages is None:
        coverages = np.linspace(0.1, 1.0, 10)
    else:
        coverages = np.array(sorted(coverages))

    risks = []
    for c in coverages:
        acc_c = selective_accuracy(y_true, y_pred, conf, c)
        risks.append(1.0 - acc_c)

    risks = np.array(risks)
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    aurc = float(trapz_fn(risks, coverages))
    return coverages, risks, aurc


def compute_confidence_auroc(y_true: np.ndarray, y_pred: np.ndarray, conf: np.ndarray) -> float:
    """Compute AUROC of confidence predicting whether classification was correct."""
    correct = (y_true == y_pred).astype(int)
    # If all correct or all incorrect, AUROC is 0.5 by definition
    if len(np.unique(correct)) < 2:
        return 0.5
    return float(roc_auc_score(correct, conf))


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_npy_path: str,
    save_png_path: str,
) -> None:
    """Compute and save confusion matrix as .npy and PNG."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    np.save(save_npy_path, cm)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Confusion Matrix",
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.tight_layout()
    plt.savefig(save_png_path, dpi=200)
    plt.close()
