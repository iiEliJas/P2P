#
#   Evaluation metrics
#
#   all functions accept array-like inputs (lists, Series, numpy arrays)
#

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


ArrayLike = Sequence[float] | np.ndarray


#
# Convert inputs to float64 numpy arrays and validate shapes
#
def _to_arrays(actual: ArrayLike, predicted: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(actual, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    if a.shape != p.shape:
        raise ValueError(
            f"Shape mismatch: actual {a.shape} vs predicted {p.shape}."
        )
    if len(a) == 0:
        raise ValueError("Empty arrays passed to metric function.")
    return a, p


#
# Mean Absolute Error
# returns a positive float
#
def mae(actual: ArrayLike, predicted: ArrayLike) -> float:
    a, p = _to_arrays(actual, predicted)
    return float(np.mean(np.abs(a - p)))



#
# Root Mean Squarred Error
# returns a positive float
#
def rmse(actual: ArrayLike, predicted: ArrayLike) -> float:
    a, p = _to_arrays(actual, predicted)
    return float(np.sqrt(np.mean((a - p) ** 2)))



#
# Symmetric Mean Absolute Percentage Error
# sMAPE = 100 * mean( |a - p| / ((|a| + |p|) / 2) )
# returns a float in [0, 200]
#
def smape(actual: ArrayLike, predicted: ArrayLike) -> float:
    a, p = _to_arrays(actual, predicted)
    denominator = (np.abs(a) + np.abs(p)) / 2.0

    with np.errstate(invalid="ignore", divide="ignore"):
        terms = np.where(denominator == 0, 0.0, np.abs(a - p) / denominator)
    return float(100.0 * np.mean(terms))



#
# Mean Absolute Percentage Error
# returns a positive float in % (e.g. 5.2 means 5.2%)
#
def mape(actual: ArrayLike, predicted: ArrayLike) -> float:
    a, p = _to_arrays(actual, predicted)
    mask = a != 0
    if not mask.any():
        return float("nan")
    return float(100.0 * np.mean(np.abs((a[mask] - p[mask]) / a[mask])))



#
# Compute all metrics at once and returns them as a dictionary (Keys: mae, rmse, smape, mape)
#
def evaluate_all(actual: ArrayLike, predicted: ArrayLike) -> dict[str, float]:
    return {
        "mae": mae(actual, predicted),
        "rmse": rmse(actual, predicted),
        "smape": smape(actual, predicted),
        "mape": mape(actual, predicted),
    }
