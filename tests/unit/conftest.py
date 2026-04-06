#
#   pytest fixtures for unit tests.
#

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


##########################################################
#           pd.Series fixtures
#

@pytest.fixture
def weekly_dates() -> pd.DatetimeIndex:
    # 162 weekly timestamps
    return pd.date_range(start="2015-01-04", periods=162, freq="W")


@pytest.fixture
def weekly_demand(weekly_dates) -> pd.Series:
    # Realistic weekly demand via seasonal and noise
    rng = np.random.default_rng(42)
    n = len(weekly_dates)
    trend = np.linspace(12_000, 9_500, n)
    seasonal = 500 * np.sin(np.arange(n) * 2 * np.pi / 52)
    noise = rng.normal(0, 200, n)
    values = np.maximum(trend + seasonal + noise, 0)
    return pd.Series(values, index=weekly_dates, name="demand")


@pytest.fixture
def train_series(weekly_demand) -> pd.Series:
    # first 158 weeks for training
    return weekly_demand.iloc[:158]


@pytest.fixture
def test_series(weekly_demand) -> pd.Series:
    # last 4 weeks for testing
    return weekly_demand.iloc[158:162]



##########################################################
#           raw DataFrame
# 

@pytest.fixture
def raw_dataframe() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    days = 1200
    perday = 10     # transactions per day
    n = days * perday

    base_dates = pd.date_range("2015-01-01", periods=days, freq="D")
    order_dates = pd.to_datetime(np.repeat(base_dates.to_numpy(), perday))

    return pd.DataFrame(
        {
            "order date (DateOrders)": order_dates,
            "Order Item Quantity": rng.integers(1, 20, n),
            "shipping date (DateOrders)": order_dates + pd.Timedelta(days=3),
        }
    )


@pytest.fixture
def sample_csv(tmp_path, raw_dataframe) -> str:
    # write raw_dataframe into a CSV and return its path
    path = tmp_path / "sample.csv"
    raw_dataframe.to_csv(path, index=False)
    return str(path)
