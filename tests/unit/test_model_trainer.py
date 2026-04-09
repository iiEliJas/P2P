import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from src.forecasting.model_trainer import ModelTrainer
from src.forecasting.baseline_model import BaselineModel


@pytest.fixture
def sample_series():
    dates = pd.date_range(start="2015-01-01", periods=150, freq="W")
    trend = np.linspace(10000, 9000, 150)
    seasonal = 500 * np.sin(np.arange(150) * 2 * np.pi / 52)
    noise = np.random.normal(0, 200, 150)
    demand = trend + seasonal + noise
    return pd.Series(demand, index=dates, name="demand")


@pytest.fixture
def model_and_data(sample_series):
    train = sample_series[:140]
    test = sample_series[140:148]
    model = BaselineModel()
    return model, train, test


def test_trainer_initialization(model_and_data):
    model, train, test = model_and_data
    trainer = ModelTrainer(model, train, test)

    assert trainer.model == model
    assert trainer.training_time is None
    assert trainer.metrics is None
    assert trainer.predictions is None


def test_trainer_train(model_and_data):
    model, train, test = model_and_data
    trainer = ModelTrainer(model, train, test)

    metrics = trainer.train()

    assert metrics is not None
    assert "mae" in metrics
    assert "smape" in metrics
    assert "rmse" in metrics
    assert trainer.training_time is not None
    assert trainer.training_time > 0


def test_trainer_get_metrics(model_and_data):
    model, train, test = model_and_data
    trainer = ModelTrainer(model, train, test)
    trainer.train()

    metrics = trainer.get_metrics()
    assert isinstance(metrics, dict)
    assert "mae" in metrics


def test_trainer_get_metrics_without_training(model_and_data):
    model, train, test = model_and_data
    trainer = ModelTrainer(model, train, test)

    with pytest.raises(ValueError, match="not been trained"):
        trainer.get_metrics()


def test_trainer_get_predictions(model_and_data):
    model, train, test = model_and_data
    trainer = ModelTrainer(model, train, test)
    trainer.train()

    predictions = trainer.get_predictions()
    assert isinstance(predictions, pd.Series)
    assert len(predictions) == model.horizon


def test_trainer_get_training_time(model_and_data):
    model, train, test = model_and_data
    trainer = ModelTrainer(model, train, test)
    trainer.train()

    training_time = trainer.get_training_time()
    assert isinstance(training_time, float)
    assert training_time > 0


def test_trainer_save_and_load_model(model_and_data):
    model, train, test = model_and_data
    trainer = ModelTrainer(model, train, test)
    trainer.train()

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.pkl"
        trainer.save_model(model_path)
        assert model_path.exists()

        # Create new trainer and load model
        new_model = BaselineModel()
        new_trainer = ModelTrainer(new_model, train, test)
        new_trainer.load_model(model_path)

        assert new_trainer.model.is_fitted is True


def test_trainer_handles_different_model_types(sample_series):
    from src.forecasting.baseline_model import BaselineModel

    train = sample_series[:140]
    test = sample_series[140:148]

    # Test with baseline model
    baseline = BaselineModel()
    trainer_baseline = ModelTrainer(baseline, train, test)
    metrics_baseline = trainer_baseline.train()
    assert metrics_baseline is not None


def test_trainer_get_summary(model_and_data):
    model, train, test = model_and_data
    trainer = ModelTrainer(model, train, test)
    trainer.train()

    summary = trainer.get_summary()
    assert "model_name" in summary
    assert "horizon" in summary
    assert "lookback" in summary
    assert "training_time" in summary
    assert "metrics" in summary
    assert "is_fitted" in summary
    assert summary["is_fitted"] is True
