#
# Unit tests for src/utils/data_cleaner.py
#

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.utils.data_cleaner import DataCleaner


@pytest.fixture
def cleaner(raw_dataframe) -> DataCleaner:
    return DataCleaner(raw_dataframe)



class TestAggregation:
    def test_returns_series(self, cleaner):
        result = cleaner.aggregate_by_frequency()
        assert isinstance(result, pd.Series)

    def test_datetime_index(self, cleaner):
        result = cleaner.aggregate_by_frequency()
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_weekly_frequency(self, cleaner):
        result = cleaner.aggregate_by_frequency()
        # checks that the index is regularly spaced by the expected frequency
        diffs = result.index[1:] - result.index[:-1]
        assert all(d.days == 7 for d in diffs)

    def test_sum_is_positive(self, cleaner):
        result = cleaner.aggregate_by_frequency()
        assert (result >= 0).all()

    def test_without_aggregation_raises(self, raw_dataframe):
        c = DataCleaner(raw_dataframe)
        with pytest.raises(RuntimeError):
            c.handle_missing_values()



class TestMissingValues:
    def test_no_nans_after_handling(self, cleaner):
        cleaner.aggregate_by_frequency()
        result = cleaner.handle_missing_values()
        assert result.isna().sum() == 0

    def test_length_preserved(self, cleaner):
        agg = cleaner.aggregate_by_frequency()
        cleaned = cleaner.handle_missing_values()
        assert len(cleaned) == len(agg)



class TestOutlierRemoval:
    def test_clips_extreme_values(self, cleaner):
        cleaner.aggregate_by_frequency()
        cleaner.handle_missing_values()
        series = cleaner.get_cleaned_series()

        # add a outlier directly
        cleaner._series.iloc[0] = 1000000
        result = cleaner.remove_outliers()

        assert result.max() < 1000000

        q1 = result.quantile(0.25)
        q3 = result.quantile(0.75)
        upper = q3 + 1.5 * (q3 - q1)
        assert result.max() <= upper + 1  # small tolerance

    def test_unknown_method_raises(self, cleaner):
        cleaner.aggregate_by_frequency()
        with pytest.raises(ValueError):
            cleaner.remove_outliers(method="zscore")

    def test_no_negatives_after_removal(self, cleaner):
        cleaner.aggregate_by_frequency()
        cleaner.handle_missing_values()
        result = cleaner.remove_outliers()
        assert (result >= 0).all()



class TestTrainTestSplit:
    def _prepare(self, cleaner):
        cleaner.aggregate_by_frequency()
        cleaner.handle_missing_values()
        cleaner.remove_outliers()

    def test_returns_two_series(self, cleaner):
        self._prepare(cleaner)
        train, test = cleaner.train_test_split()
        assert isinstance(train, pd.Series)
        assert isinstance(test, pd.Series)

    def test_no_overlap(self, cleaner):
        self._prepare(cleaner)
        train, test = cleaner.train_test_split()
        assert train.index[-1] < test.index[0]

    def test_no_future_leakage(self, cleaner):
        # check training dates < testing dates
        self._prepare(cleaner)
        train, test = cleaner.train_test_split()
        assert train.index.max() < test.index.min()

    def test_too_short_raises(self, cleaner):
        cleaner.aggregate_by_frequency()
        with pytest.raises(ValueError, match="too short"):
            cleaner.train_test_split(train_weeks=1000, test_weeks=1000)
