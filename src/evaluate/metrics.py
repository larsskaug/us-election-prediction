"""Evaluation metrics for election forecasts.

Includes Brier score, log-loss, calibration, and district-level accuracy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def brier_score(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    """Compute the Brier score (mean squared probability error).

    Parameters
    ----------
    y_true : np.ndarray, shape (n,)
        Binary outcomes (1 = Dem win, 0 = Rep win).
    p_pred : np.ndarray, shape (n,)
        Predicted probability of Dem win.

    Returns
    -------
    float
        Brier score; lower is better (0 = perfect, 1 = worst).
    """
    y_true = np.asarray(y_true, dtype=float)
    p_pred = np.asarray(p_pred, dtype=float)
    return float(np.mean((p_pred - y_true) ** 2))


def log_loss(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    eps: float = 1e-15,
) -> float:
    """Compute binary cross-entropy log-loss.

    Parameters
    ----------
    y_true : np.ndarray, shape (n,)
        Binary outcomes.
    p_pred : np.ndarray, shape (n,)
        Predicted probability of Dem win.
    eps : float
        Clipping value to avoid log(0).

    Returns
    -------
    float
        Mean log-loss; lower is better.
    """
    y_true = np.asarray(y_true, dtype=float)
    p_pred = np.clip(np.asarray(p_pred, dtype=float), eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p_pred) + (1 - y_true) * np.log(1 - p_pred)))


def accuracy(y_true: np.ndarray, p_pred: np.ndarray, threshold: float = 0.5) -> float:
    """Compute binary classification accuracy at a given probability threshold.

    Parameters
    ----------
    y_true : np.ndarray, shape (n,)
        Binary outcomes.
    p_pred : np.ndarray, shape (n,)
        Predicted probability of Dem win.
    threshold : float
        Decision boundary (default 0.5).

    Returns
    -------
    float
        Fraction of correctly predicted districts.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = (np.asarray(p_pred, dtype=float) >= threshold).astype(int)
    return float(np.mean(y_true == y_pred))


def calibration_curve(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute a reliability (calibration) curve.

    Parameters
    ----------
    y_true : np.ndarray, shape (n,)
        Binary outcomes.
    p_pred : np.ndarray, shape (n,)
        Predicted probability of Dem win.
    n_bins : int
        Number of equal-width probability bins.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``bin_center``, ``mean_predicted``,
        ``fraction_positive``, ``count``.
    """
    y_true = np.asarray(y_true, dtype=float)
    p_pred = np.asarray(p_pred, dtype=float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p_pred >= lo) & (p_pred < hi)
        count = int(mask.sum())
        if count == 0:
            continue
        rows.append(
            {
                "bin_center": float((lo + hi) / 2),
                "mean_predicted": float(p_pred[mask].mean()),
                "fraction_positive": float(y_true[mask].mean()),
                "count": count,
            }
        )
    return pd.DataFrame(rows)


def evaluate_forecast(
    y_true: np.ndarray,
    p_pred: np.ndarray,
) -> dict[str, float]:
    """Compute all standard forecast evaluation metrics.

    Parameters
    ----------
    y_true : np.ndarray, shape (n,)
        Binary outcomes.
    p_pred : np.ndarray, shape (n,)
        Predicted probability of Dem win.

    Returns
    -------
    dict[str, float]
        Dictionary with keys ``brier_score``, ``log_loss``, ``accuracy``.
    """
    return {
        "brier_score": brier_score(y_true, p_pred),
        "log_loss": log_loss(y_true, p_pred),
        "accuracy": accuracy(y_true, p_pred),
    }
