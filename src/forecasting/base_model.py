#########################################################
#       Abstract base class for all models
#########################################################

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from src.utils.metrics import evaluate_all

logger = logging.getLogger(__name__)



# Abstract base class for all forecasting models
#   All models must implement fit() and predict() methods
#
#   Parameters: 
#       horizon : int
#           number of future periods to forecast
#      lookback : int
#           context window
#
class BaseForecastingModel(ABC):

    def __init__(self, horizon: int = 4, lookback: int = 8) -> None:
        self.horizon = horizon
        self.lookback = lookback
        self.is_fitted: bool = False
        self._model: Any = None

    #################################################
    # Abstract methods
    # 

    #
    # Train the model on train data
    #
    @abstractmethod
    def fit(self, train_data: pd.Series) -> None:
        pass

    #
    # Generate forecasts
    #   Returns pd.Series with a DatetimeIndex continuing from the end of the training data
    #
    @abstractmethod
    def predict(self, steps: int | None = None) -> pd.Series:
        pass


    #################################################
    # Helpers
    #

    #
    # Compute evaluation metrics for a pair of actual / predicted series
    # Returns dict with keys:
    #       mae, rmse, smape
    #
    def evaluate(self, actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
        metrics = evaluate_all(actual.to_numpy(dtype=float), predicted.to_numpy(dtype=float))
        logger.info(
            "%s evaluation:  MAE: %.2f | RMSE: %.2f | sMAPE: %.2f%%",
            self.__class__.__name__,
            metrics["mae"],
            metrics["rmse"],
            metrics["smape"],
        )
        return metrics


    #
    # Check if fit has been called before predict
    #
    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} must call fit() before predict()."
            )


    #
    # Representation for debugging
    #
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"horizon={self.horizon}, lookback={self.lookback}, "
            f"fitted={self.is_fitted})"
        )
