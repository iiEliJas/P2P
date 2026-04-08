######################################
#           DataCleaner
######################################

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

from src.config import (COL_ORDER_DATE, COL_QUANTITY,FREQUENCY, TEST_WEEKS, TRAIN_WEEKS)

logger = logging.getLogger(__name__)


# Cleaner for the DataCo Supply Chain dataset
#
#   data : raw DataFrame returned by DataLoader
#   frequency : pandas offset alias for aggregation (the default is 'W' for weekly)
#
#   Usage:
#       cleaner = DataCleaner(raw_file)
#       weekly = cleaner.aggregate_by_frequency()
#       weekly = cleaner.handle_missing_values()
#       weekly = cleaner.remove_outliers()
#       train, test = cleaner.train_test_split()    
#
class DataCleaner:

    def __init__(self, data: pd.DataFrame, frequency: str = FREQUENCY, drop_last_weeks: int = 0) -> None:
        self.frequency = frequency
        self._raw = data.copy()
        self.drop_last_weeks = drop_last_weeks
        self._series: pd.Series | None = None


    # summarize raw data by frequency and sum the quantity
    # Returns a series with a DatetimeIndex
    def aggregate_by_frequency(self) -> pd.Series:
        df = self._raw[[COL_ORDER_DATE, COL_QUANTITY]].copy()
        df = df.dropna(subset=[COL_ORDER_DATE])
        df = df.set_index(COL_ORDER_DATE)
        df.index = pd.to_datetime(df.index)

        # numeric quantity
        df[COL_QUANTITY] = pd.to_numeric(df[COL_QUANTITY], errors="coerce")

        weekly = (
            df[COL_QUANTITY]
            .resample(self.frequency)
            .sum()
            .rename("demand")
        )

        logger.info(
            "Aggregated to %d %s periods (%.0f–%.0f).",
            len(weekly),
            self.frequency,
            weekly.index[0].year,
            weekly.index[-1].year,
        )

        if self.drop_last_weeks > 0:
            weekly = weekly.iloc[:-self.drop_last_weeks]
            logger.info("Trimmed %d incomplete tail week(s).", self.drop_last_weeks)

        self._series = weekly
        return self._series.copy()


    # Fill gaps created by weeks with zero transactions.
    def handle_missing_values(self) -> pd.Series:
        series = self._require_aggregated()
        before = int(series.isna().sum())

        # missing week means no orders were placed
        self._series = series.fillna(0)

        # Check if there are still NaNs
        remaining = int(self._series.isna().sum())
        if remaining:
            self._series = self._series.interpolate(method="time")
            logger.warning("Interpolated %d remaining NaN(s).", remaining)

        logger.info("Missing-value handling: %d NaN(s) → filled with 0.", before)
        return self._series.copy()


    # Remove outliers using the specified method
    def remove_outliers(self, method: str = "iqr") -> pd.Series:
        series = self._require_aggregated()

        if method == "iqr":
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            upper = q3 + 1.5 * iqr

            # Only clip upward spikes — never clip low values upward
            n_clipped = (series > upper).sum()
            self._series = series.clip(upper=upper)
            logger.info("IQR outlier removal: %d upper spikes clipped to %.1f.", n_clipped, upper)
        else:
            raise ValueError(f"Unknown outlier method: '{method}'. Use 'iqr'.")

        return self._series.copy()


    # Load, clean, and validate the raw data from the CSV file
    # Returns the cleaned DataFrame as a Tuple
    def train_test_split(
        self,
        train_weeks: int = TRAIN_WEEKS,
        test_weeks: int = TEST_WEEKS,
    ) -> Tuple[pd.Series, pd.Series]:
        series = self._require_aggregated()

        required = train_weeks + test_weeks
        available = len(series)
        if available < required:
            raise ValueError(
                f"Series has {available} weeks but split requires {required} -> too short"
            )

        train = series.iloc[:train_weeks]
        test = series.iloc[train_weeks:train_weeks + test_weeks]

        logger.info(
            "Train/test split — train: %d weeks (%s - %s), test: %d weeks (%s - %s).",
            len(train),
            train.index[0].date(),
            train.index[-1].date(),
            len(test),
            test.index[0].date(),
            test.index[-1].date(),
        )
        return train, test


    # return the current state of the cleaned series
    def get_cleaned_series(self) -> pd.Series:
        series = self._require_aggregated()
        return series.copy()


    # ensure aggregate_by_frequency() was called
    def _require_aggregated(self) -> pd.Series:
        if self._series is None:
            raise RuntimeError(
                "Call aggregate_by_frequency() before this method."
            )
        return self._series
