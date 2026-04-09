import pytest
import pandas as pd
import numpy as np

from src.forecasting.baseline_model import BaselineModel
from src.forecasting.model_trainer import ModelTrainer
from src.forecasting.model_evaluator import ModelEvaluator


@pytest.fixture
def sample_series():
    dates = pd.date_range(start="2015-01-01", periods=150, freq="W")
    trend = np.linspace(10000, 9000, 150)
    seasonal = 500 * np.sin(np.arange(150) * 2 * np.pi / 52)
    noise = np.random.normal(0, 200, 150)
    demand = trend + seasonal + noise
    return pd.Series(demand, index=dates, name="demand")


def test_full_forecasting_pipeline(sample_series):
    train = sample_series[:140]
    test = sample_series[140:148]

    # Initialize models
    models = {
        "Baseline": BaselineModel(horizon=8, lookback=8),
    }

    # Train all models
    trainers = {}
    for name, model in models.items():
        trainer = ModelTrainer(model, train, test)
        trainer.train()
        trainers[name] = trainer

    # Evaluate all models
    evaluator = ModelEvaluator(models, test)
    results = evaluator.evaluate_all()

    # Assertions
    assert len(results) == 1
    assert "smape" in results.columns
    assert "mae" in results.columns
    assert "rmse" in results.columns
    assert results["smape"].notna().all()
    assert results["mae"].notna().all()
    assert results["rmse"].notna().all()


def test_pipeline_model_comparison(sample_series):
    train = sample_series[:140]
    test = sample_series[140:148]

    # Initialize models
    models = {
        "Baseline": BaselineModel(horizon=8, lookback=8),
    }

    # Train
    for model in models.values():
        trainer = ModelTrainer(model, train, test)
        trainer.train()

    # Evaluate
    evaluator = ModelEvaluator(models, test)
    results = evaluator.evaluate_all()

    # Test comparison methods
    best_model = evaluator.get_best_model()
    assert best_model is not None
    assert best_model in models.keys()

    summary = evaluator.get_summary()
    assert "best_model" in summary
    assert "best_smape" in summary
    assert "improvement" in summary


def test_pipeline_predictions_alignment(sample_series):
    train = sample_series[:140]
    test = sample_series[140:148]

    models = {
        "Baseline": BaselineModel(horizon=8, lookback=8),
    }

    for model in models.values():
        trainer = ModelTrainer(model, train, test)
        trainer.train()

    evaluator = ModelEvaluator(models, test)
    evaluator.evaluate_all()

    # Verify predictions exist and align with test data
    for model_name, forecast in evaluator.predictions.items():  # type: ignore
        assert len(forecast) == len(test)
        assert forecast.index[0] > train.index[-1]
