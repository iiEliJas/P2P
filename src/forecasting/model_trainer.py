import time
import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# training pipeline for all forecasting models
class ModelTrainer:

    def __init__(
        self,
        model: Any,
        train_data: pd.Series,
        test_data: pd.Series,
        config: Optional[Dict] = None,
    ):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.config = config or {}
        self.training_time = None
        self.metrics = None
        self.predictions = None



    def train(self) -> Dict[str, float]:
        start_time = time.time()

        try:
            logger.info(f"Training {self.model.__class__.__name__}...")
            self.model.fit(self.train_data)

            # Generate predictions on test data
            self.predictions = self.model.predict()

            # Evaluate
            self.metrics = self.model.evaluate(self.test_data, self.predictions)
            self.training_time = time.time() - start_time

            logger.info(
                f"{self.model.__class__.__name__} training completed in {self.training_time:.2f}s"
            )
            return self.metrics

        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise



    def get_metrics(self) -> Dict[str, float]:
        if self.metrics is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        return self.metrics.copy()

    def get_predictions(self) -> pd.Series:
        if self.predictions is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        return self.predictions.copy()

    def get_training_time(self) -> float:
        if self.training_time is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        return self.training_time



    def save_model(self, filepath: str | Path) -> None:
        filepath = Path(filepath)
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "wb") as f:
                pickle.dump(self.model, f)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise



    def load_model(self, filepath: str | Path) -> None:
        filepath = Path(filepath)
        try:
            with open(filepath, "rb") as f:
                self.model = pickle.load(f)
            self.model.is_fitted = True
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise



    def get_summary(self) -> Dict[str, Any]:
        return {
            "model_name": self.model.__class__.__name__,
            "horizon": self.model.horizon,
            "lookback": self.model.lookback,
            "training_time": self.training_time,
            "metrics": self.metrics,
            "is_fitted": self.model.is_fitted,
        }
