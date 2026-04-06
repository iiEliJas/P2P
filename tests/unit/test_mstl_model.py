#
# Unit tests for src/forecasting/mstl_model.py
#

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.forecasting.mstl_model import MSTLModel


@pytest.fixture
def fitted_model(train_series) -> MSTLModel:
    model = MSTLModel(horizon=4, seasonal_periods=[52, 4])
    model.fit(train_series)
    return model


class TestMSTLFit:
    def test_fit_sets_is_fitted(self, train_series):
        model = MSTLModel()
        model.fit(train_series)
        assert model.is_fitted is True

    def test_fit_too_short_raises(self):
        short = pd.Series(np.ones(20),
            index=pd.date_range("2020-01-01", periods=20, freq="W"))
        model = MSTLModel(seasonal_periods=[52])
        with pytest.raises(ValueError, match="too short"):
            model.fit(short)


class TestMSTLPredict:
    def test_returns_series(self, fitted_model):
        forecast = fitted_model.predict()
        assert isinstance(forecast, pd.Series)

    def test_forecast_length_matches_horizon(self, fitted_model):
        horizon = 4
        fitted_model.horizon = horizon
        forecast = fitted_model.predict()
        assert len(forecast) == horizon

    def test_forecast_length_override(self, fitted_model):
        forecast = fitted_model.predict(steps=6)
        assert len(forecast) == 6

    def test_forecast_has_datetime_index(self, fitted_model):
        forecast = fitted_model.predict()
        assert isinstance(forecast.index, pd.DatetimeIndex)

    def test_forecast_is_future(self, fitted_model, train_series):
        forecast = fitted_model.predict()
        assert forecast.index.min() > train_series.index.max()

    def test_forecast_no_negatives(self, fitted_model):
        """Demand forecasts should not be negative."""
        forecast = fitted_model.predict()
        assert (forecast.values >= 0).all()

    def test_forecast_dtype(self, fitted_model):
        forecast = fitted_model.predict()
        assert np.issubdtype(forecast.dtype, np.floating)

    def test_predict_before_fit_raises(self):
        model = MSTLModel()
        with pytest.raises(RuntimeError, match="fit"):
            model.predict()


class TestMSTLEvaluation:
    def test_evaluate_returns_dict(self, fitted_model, test_series):
        forecast = fitted_model.predict()
        metrics = fitted_model.evaluate(test_series, forecast)
        assert isinstance(metrics, dict)

    def test_evaluate_keys(self, fitted_model, test_series):
        forecast = fitted_model.predict()
        metrics = fitted_model.evaluate(test_series, forecast)
        assert "mae" in metrics
        assert "smape" in metrics
        assert "rmse" in metrics

    def test_evaluate_non_negative(self, fitted_model, test_series):
        forecast = fitted_model.predict()
        metrics = fitted_model.evaluate(test_series, forecast)
        assert metrics["mae"] >= 0
        assert metrics["rmse"] >= 0
        assert 0 <= metrics["smape"] <= 200


class TestMSTLFittedValues:
    def test_fitted_values_length(self, fitted_model, train_series):
        fitted = fitted_model.get_fitted_values()
        # results should be close to train length
        assert len(fitted) > len(train_series) * 0.8

    def test_fitted_values_dtype(self, fitted_model):
        fitted = fitted_model.get_fitted_values()
        assert np.issubdtype(fitted.dtype, np.floating)
