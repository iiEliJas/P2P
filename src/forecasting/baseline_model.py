import pandas as pd
import logging
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.forecasting.base_model import BaseForecastingModel

logger = logging.getLogger(__name__)


# Exponential Smoothing baseline model
class BaselineModel(BaseForecastingModel):

    def __init__(
        self,
        horizon: int = 8,
        lookback: int = 8,
        trend: str = "add",
        seasonal: str = "add",
        seasonal_periods: int = 52,
    ):
        super().__init__(horizon, lookback)
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.fitted_model = None
        self._last_date: pd.Timestamp | None = None


    #
    # Decomposes train data with MSTL and fit a trend forecaster
    #
    def fit(self, train_data: pd.Series) -> None:
        try:
            self.model = ExponentialSmoothing(
                train_data,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods,
                initialization_method="estimated",
            )
            self.fitted_model = self.model.fit(optimized=True)
            self.is_fitted = True
            self._last_date = train_data.index[-1]
            logger.info("Baseline (Exponential Smoothing) model fitted successfully")
        except Exception as e:
            logger.error(f"Error fitting Baseline model: {e}")
            raise
    
    #
    # Forecast [steps] periods ahead
    # Returns pd.Series with DatetimeIndex continuing from train data
    #
    def predict(self, steps: int | None = None) -> pd.Series:
        self._require_fitted()
        steps = steps or self.horizon

        assert self._last_date is not None

        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        try:
            forecast = self.fitted_model.forecast(steps=self.horizon)   # type: ignore
            forecast_index = pd.date_range(
                start=self._last_date + pd.Timedelta(weeks=1),
                periods=self.horizon,
                freq="W",
            )
            return pd.Series(forecast, index=forecast_index, name="forecast")
        except Exception as e:
            logger.error(f"Error with Baseline model: {e}")
            raise
