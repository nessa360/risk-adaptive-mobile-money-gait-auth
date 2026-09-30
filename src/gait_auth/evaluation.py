"""Evaluation metrics for closed-set classification and biometric verification."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_curve


def eer(genuine_scores: np.ndarray, impostor_scores: np.ndarray) -> float:
    """Compute EER from genuine and impostor similarity scores in [0, 1]."""
    genuine_scores = np.asarray(genuine_scores, dtype=float)
    impostor_scores = np.asarray(impostor_scores, dtype=float)
    if len(genuine_scores) == 0 or len(impostor_scores) == 0:
        return float("nan")
    scores = np.concatenate([genuine_scores, impostor_scores])
    labels = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    index = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[index] + fnr[index]) / 2.0)


def far_frr(genuine_scores: np.ndarray, impostor_scores: np.ndarray, threshold: float) -> tuple[float, float]:
    genuine_scores = np.asarray(genuine_scores, dtype=float)
    impostor_scores = np.asarray(impostor_scores, dtype=float)
    far = float(np.mean(impostor_scores >= threshold)) if len(impostor_scores) else float("nan")
    frr = float(np.mean(genuine_scores < threshold)) if len(genuine_scores) else float("nan")
    return far, frr


def classifier_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def verification_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    """Compute binary verification metrics where y_true=1 means genuine."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    genuine = scores[y_true == 1]
    impostor = scores[y_true == 0]
    far, frr = far_frr(genuine, impostor, threshold)
    predictions = (scores >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "eer": eer(genuine, impostor),
        "far": far,
        "frr": frr,
        "threshold": float(threshold),
    }
