import pandas as pd
import numpy as np
import logging
import warnings
from pytorch_forecasting import NBeats, TimeSeriesDataSet
from lightning.pytorch import Trainer
import torch

from src.forecasting.base_model import BaseForecastingModel

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


# N-BEATS neural network model using pytorch_forecasting
class NBeatsModel(BaseForecastingModel):

    def __init__(
        self,
        horizon: int = 8,
        lookback: int = 8,
        stack_types: list | None = None,
        num_stacks: int = 30,
        num_blocks: int = 3,
        num_layers: int = 4,
        hidden_size: int = 128,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        device: str = "cpu",
    ):
        super().__init__(horizon, lookback)
        self.stack_types = stack_types or ["trend", "seasonality"]
        self.num_stacks = num_stacks
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device if torch.cuda.is_available() else "cpu"
        self.fitted_model = None
        self.scaler_mean = None
        self.scaler_std = None
        self._last_date: pd.Timestamp | None = None
        self._last_lookback_data = None

    def fit(self, train_data: pd.Series) -> None:
        try:
            # Suppress warnings from NBeats hyperparameters
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Attribute 'loss' is an instance of")
                warnings.filterwarnings("ignore", message="Attribute 'logging_metrics' is an instance of")

                # Normalize the data
                self.scaler_mean = train_data.mean()
                self.scaler_std = train_data.std()
                train_normalized = (train_data - self.scaler_mean) / (self.scaler_std + 1e-8)
                self._last_lookback_data = train_normalized.iloc[-self.lookback:]

                # Prepare data in format required by pytorch_forecasting
                df = self._prepare_dataframe(train_normalized)

                # Create dataset
                training_cutoff = len(df) - self.horizon
                dataset = TimeSeriesDataSet(
                    data=df,
                    time_idx="time_idx",
                    target="value",
                    group_ids=["group"],
                    time_varying_unknown_reals=["value"],
                    add_relative_time_idx=False,
                    add_target_scales=False,
                    add_encoder_length=False,
                    target_normalizer=None,
                    min_encoder_length=self.lookback,
                    max_encoder_length=self.lookback,
                    min_prediction_length=self.horizon,
                    max_prediction_length=self.horizon,
                    allow_missing_timesteps=False,
                )

                train_dataloader = dataset.to_dataloader(
                    train=True, batch_size=self.batch_size, num_workers=0
                )

                # Create and train model
                self.fitted_model = NBeats.from_dataset(
                    dataset,
                    learning_rate=self.learning_rate,
                    stack_types=self.stack_types,
                    num_blocks=[self.num_blocks] * len(self.stack_types),
                    num_block_layers=[self.num_layers] * len(self.stack_types),
                    widths=[self.hidden_size] * len(self.stack_types),
                )

                trainer = Trainer(
                    max_epochs=self.epochs,
                    accelerator="auto",
                    enable_progress_bar=False,
                    enable_model_summary=False,
                    logger=False,
                    enable_checkpointing=False,
                    limit_val_batches=0,
                )

                trainer.fit(self.fitted_model, train_dataloaders=train_dataloader)
                self.is_fitted = True
                self._last_date = train_data.index[-1]
                logger.info("N-BEATS model trained successfully")

        except Exception as e:
            logger.error(f"Error fitting N-BEATS model: {e}")
            raise

    def predict(self, steps: int | None = None) -> pd.Series:
        self._require_fitted()
        if steps is None:
            steps = self.horizon
        elif not isinstance(steps, int):
            raise TypeError("steps must be an int or None")

        assert self._last_date is not None

        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        try:
            # Use the last lookback data for prediction
            data_normalized = self._last_lookback_data
            assert data_normalized is not None

            # Prepare encoder data and append dummy decoder rows for prediction
            encoder_df = self._prepare_dataframe(data_normalized)   # type: ignore
            future_df = pd.DataFrame({
                "time_idx": range(len(encoder_df), len(encoder_df) + steps),
                "value": np.zeros(steps, dtype=float),
                "group": "demand",
            })
            df = pd.concat([encoder_df, future_df], ignore_index=True)

            # Create dataset for prediction
            dataset = TimeSeriesDataSet(
                data=df,
                time_idx="time_idx",
                target="value",
                group_ids=["group"],
                time_varying_unknown_reals=["value"],
                add_relative_time_idx=False,
                add_target_scales=False,
                add_encoder_length=False,
                target_normalizer=None,
                min_encoder_length=self.lookback,
                max_encoder_length=self.lookback,
                min_prediction_length=steps,
                max_prediction_length=steps,
                allow_missing_timesteps=False,
                predict_mode=True,
            )

            # Get predictions
            predictions_list = []
            with torch.no_grad():
                for x, _ in dataset.to_dataloader(train=False, batch_size=1):
                    y_hat = self.fitted_model(x)    # type: ignore
                    predictions_list.append(y_hat.prediction.cpu().numpy())

            # Average predictions if multiple batches
            if predictions_list:
                forecast_normalized = np.concatenate(predictions_list, axis=0)[-1, :, 0]
            else:
                # Fallback: use last lookback window
                forecast_normalized = np.zeros(steps)

            # Denormalize
            forecast = forecast_normalized * (self.scaler_std + 1e-8) + self.scaler_mean    # type: ignore

            forecast_index = pd.date_range(
                start=self._last_date + pd.Timedelta(weeks=1),
                periods=steps,
                freq="W",
            )
            return pd.Series(forecast, index=forecast_index, name="forecast")

        except Exception as e:
            logger.error(f"Error predicting with N-BEATS model: {e}")
            raise

    # Helper to convert series to dataframe format for pytorch_forecasting
    def _prepare_dataframe(self, data: pd.Series) -> pd.DataFrame:
        df = pd.DataFrame({
            "time_idx": range(len(data)),
            "value": data.values,
            "group": "demand",
        })
        return df
