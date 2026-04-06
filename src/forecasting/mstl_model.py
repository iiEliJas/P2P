#########################################################
#               MSTL model
#
# How it works:
#   1. Decompose the training series with statsmodels MSTL
#   2. Forecast the trend component using ETS
#   3. Project the seasonal components forward
#   4. Sum trend forecast + seasonal projections
#
#########################################################

from __future__ import annotations

import logging
from typing import cast

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.seasonal import DecomposeResult, MSTL

from src.config import (
    FORECAST_HORIZON,
    LOOKBACK_WINDOW,
    MSTL_SEASONAL_PERIODS,
    MSTL_TREND_WINDOW,
)
from src.forecasting.base_model import BaseForecastingModel

logger = logging.getLogger(__name__)



# MSTL-based forecasting model
# 
# Parameters:
#   horizon : number of future periods to forecast
#   lookback : context window (not used by MSTL)
#   seasonal_periods : list of seasonal periods to decompose
#   trend_window : loess window length passed to MSTL (odd number)
#
# Usage:
#   model = MSTLModel(horizon=4)
#   model.fit(train_series)
#   forecast = model.predict()
#   metrics  = model.evaluate(test_series, forecast)
#
class MSTLModel(BaseForecastingModel):

    def __init__(self, horizon: int = FORECAST_HORIZON, lookback: int = LOOKBACK_WINDOW, 
                 seasonal_periods: list[int] | None = None, trend_window: int = MSTL_TREND_WINDOW) -> None:
        super().__init__(horizon=horizon, lookback=lookback)
        self.seasonal_periods = seasonal_periods or MSTL_SEASONAL_PERIODS
        self.trend_window = trend_window

        # after fit()
        self._decomp: DecomposeResult | None = None
        self._trend_model = None
        self._seasonal_components: dict[int, np.ndarray] = {}
        self._last_date: pd.Timestamp | None = None
        self._freq: str | None = None


    #
    # Decomposes train data with MSTL and fit a trend forecaster
    #
    def fit(self, train_data: pd.Series) -> None:
        if len(train_data) < max(self.seasonal_periods) * 2:
            raise ValueError(
                f"Training series ({len(train_data)} obs) is too short for "
                f"seasonal periods {self.seasonal_periods}. Must be at least "
                f"{max(self.seasonal_periods) * 2} observations."
            )

        self._last_date = train_data.index[-1]
        self._freq = pd.infer_freq(cast(pd.DatetimeIndex, train_data.index)) or "W"

        ############################
        # Decomposition
         
        # check trend_window is odd
        _odd = lambda n: n if n % 2 == 1 else n + 1
        windows = [_odd(max(self.trend_window, p + 1)) for p in self.seasonal_periods]

        logger.info(
            "Fitting MSTL (seasonal_periods=%s, trend_window=%d) on %d obs ...",
            self.seasonal_periods,
            windows,
            len(train_data),
        )
        mstl = MSTL(train_data, periods=self.seasonal_periods, windows=windows)
        self._decomp = mstl.fit()

        ############################
        # Fit ETS on the trend component
        trend_series = pd.Series(
            self._decomp.trend, index=train_data.index
        ).dropna()

        self._trend_model = Holt(trend_series, exponential=False, damped_trend=True)
        self._trend_fit = self._trend_model.fit(optimized=True)

        ############################
        # Store seasonal cycles
        seasonal_df = self._decomp.seasonal
        if isinstance(seasonal_df, pd.Series):
            seasonal_df = seasonal_df.to_frame(name=f"seasonal_{self.seasonal_periods[0]}")

        seasonal_df = cast(pd.DataFrame, seasonal_df)

        for period in self.seasonal_periods:
            col = f"seasonal_{period}"
            if col in seasonal_df.columns:
                cycle = np.asarray(seasonal_df[col].values[-period:])   # last complete cycle
                self._seasonal_components[period] = cycle

        self.is_fitted = True
        logger.info("MSTL model fitted successfully.")


    #
    # Forecast [steps] periods ahead
    # Returns pd.Series with DatetimeIndex continuing from train data
    #
    def predict(self, steps: int | None = None) -> pd.Series:
        self._require_fitted()
        steps = steps or self.horizon

        assert self._last_date is not None
        assert self._decomp is not None

        ############################
        # Trend forecast
        trend_forecast = self._trend_fit.forecast(steps)

        ############################
        # Seasonal projections
        seasonal_sum = np.zeros(steps)
        for period, cycle in self._seasonal_components.items():
            projection = np.array(
                [cycle[i % period] for i in range(steps)]
            )
            seasonal_sum += projection

        ############################
        # Combine
        forecast_values = trend_forecast.values + seasonal_sum
        forecast_values = np.maximum(forecast_values, 0)    # cant be negative

        # DatetimeIndex for the forecast horizon
        future_index = pd.date_range(
            start=self._last_date,
            periods=steps + 1,
            freq=self._freq,
        )[1:]

        forecast = pd.Series(forecast_values, index=future_index, name="forecast", dtype=float)

        logger.info("MSTL forecast generated: %s", forecast.values.tolist())
        return forecast


    #
    # Return fitted values for the training period (trend + seasonal sum)
    #
    def get_fitted_values(self) -> pd.Series:
        self._require_fitted()
        assert self._decomp is not None
        seasonal_df = self._decomp.seasonal
        if isinstance(seasonal_df, pd.Series):
            seasonal_total = seasonal_df
        else:
            seasonal_total = seasonal_df.sum(axis=1)

        fitted = self._decomp.trend + seasonal_total
        return fitted.rename("fitted").dropna()
