import pytest
import pandas as pd
import numpy as np
from src.forecasting.nbeats_model import NBeatsModel


@pytest.fixture
def sample_series():
    dates = pd.date_range(start="2015-01-01", periods=150, freq="W")
    trend = np.linspace(10000, 9000, 150)
    seasonal = 500 * np.sin(np.arange(150) * 2 * np.pi / 52)
    noise = np.random.normal(0, 200, 150)
    demand = trend + seasonal + noise
    return pd.Series(demand, index=dates, name="demand")


def test_nbeats_initialization():
    model = NBeatsModel(horizon=8, lookback=8)
    assert model.horizon == 8
    assert model.lookback == 8
    assert model.is_fitted is False
    assert model.device in ["cpu", "cuda"]


def test_nbeats_fit(sample_series):
    model = NBeatsModel(epochs=1, batch_size=32)
    train = sample_series[:140]
    model.fit(train)
    assert model.is_fitted is True
    assert model.fitted_model is not None
    assert model.scaler_mean is not None
    assert model.scaler_std is not None


def test_nbeats_predict(sample_series):
    model = NBeatsModel(epochs=1, batch_size=32)
    train = sample_series[:140]
    model.fit(train)

    forecast = model.predict()
    assert len(forecast) == 8
    assert forecast.notna().all()
    assert isinstance(forecast, pd.Series)


def test_nbeats_predict_without_fit(sample_series):
    model = NBeatsModel()
    with pytest.raises(RuntimeError, match="must call fit"):
        model.predict()


def test_nbeats_evaluation(sample_series):
    model = NBeatsModel(epochs=1, batch_size=32)
    train = sample_series[:140]
    test = sample_series[140:148]

    model.fit(train)
    forecast = model.predict()

    metrics = model.evaluate(test, forecast)
    assert "mae" in metrics
    assert "smape" in metrics
    assert "rmse" in metrics
    assert all(v >= 0 for v in metrics.values())


def test_nbeats_forecast_shape(sample_series):
    model = NBeatsModel(horizon=4, epochs=1)
    train = sample_series[:140]
    model.fit(train)

    forecast = model.predict()
    assert len(forecast) == 4


def test_nbeats_custom_parameters():
    model = NBeatsModel(
        horizon=6,
        lookback=10,
        num_stacks=20,
        num_blocks=2,
        hidden_size=64,
        learning_rate=0.002,
    )
    assert model.horizon == 6
    assert model.lookback == 10
    assert model.num_stacks == 20
    assert model.num_blocks == 2
    assert model.hidden_size == 64
    assert model.learning_rate == 0.002


def test_nbeats_normalization(sample_series):
    model = NBeatsModel(epochs=1)
    train = sample_series[:140]
    model.fit(train)

    assert model.scaler_mean == train.mean()
    assert model.scaler_std == train.std()

    forecast = model.predict()
    assert forecast.min() > 0 or forecast.max() < 0
