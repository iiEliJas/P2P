import pytest
import pandas as pd
import numpy as np
from src.forecasting.baseline_model import BaselineModel


@pytest.fixture
def sample_series():
    dates = pd.date_range(start="2015-01-01", periods=150, freq="W")
    trend = np.linspace(10000, 9000, 150)
    seasonal = 500 * np.sin(np.arange(150) * 2 * np.pi / 52)
    noise = np.random.normal(0, 200, 150)
    demand = trend + seasonal + noise
    return pd.Series(demand, index=dates, name="demand")


def test_baseline_initialization():
    model = BaselineModel(horizon=8, lookback=8)
    assert model.horizon == 8
    assert model.lookback == 8
    assert model.is_fitted is False


def test_baseline_fit(sample_series):
    model = BaselineModel()
    train = sample_series[:140]
    model.fit(train)
    assert model.is_fitted is True
    assert model.fitted_model is not None


def test_baseline_predict(sample_series):
    model = BaselineModel()
    train = sample_series[:140]
    model.fit(train)

    forecast = model.predict()
    assert len(forecast) == 8
    assert forecast.notna().all()
    assert isinstance(forecast, pd.Series)


def test_baseline_predict_without_fit(sample_series):
    model = BaselineModel()
    with pytest.raises(RuntimeError, match="must call fit"):
        model.predict()


def test_baseline_evaluation(sample_series):
    model = BaselineModel()
    train = sample_series[:140]
    test = sample_series[140:148]

    model.fit(train)
    forecast = model.predict()

    metrics = model.evaluate(test, forecast)
    assert "mae" in metrics
    assert "smape" in metrics
    assert "rmse" in metrics
    assert all(v >= 0 for v in metrics.values())


def test_baseline_forecast_shape(sample_series):
    model = BaselineModel(horizon=4)
    train = sample_series[:140]
    model.fit(train)

    forecast = model.predict()
    assert len(forecast) == 4


def test_baseline_custom_parameters():
    model = BaselineModel(
        horizon=6,
        lookback=10,
        trend="mul",
        seasonal="mul",
        seasonal_periods=13,
    )
    assert model.horizon == 6
    assert model.lookback == 10
    assert model.trend == "mul"
    assert model.seasonal == "mul"
    assert model.seasonal_periods == 13
