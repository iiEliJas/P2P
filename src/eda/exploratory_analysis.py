#
#   ExploratoryAnalysis
#       statistically summarise a weekly demand series
#       produce all EDA charts.
#

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)


#
# Class for running EDA on a weekly demand series and trend and seasonality analysis, distribution stats, and visualizations
#
#   Usage:
#       eda = ExploratoryAnalysis(weekly_series)
#       trend_info   = eda.analyze_trends()
#       season_info  = eda.detect_seasonality()
#       dist_info    = eda.get_distribution_stats()
#       eda.plot_all_eda_charts()
#
class ExploratoryAnalysis:

    def __init__(self, data: pd.Series) -> None:
        if not isinstance(data.index, pd.DatetimeIndex):
            raise TypeError("data must have a DatetimeIndex.")
        self.data = data.copy()


    # Trend analysis using linear regression and stationarity tests
    # Returns dictionary with keys:
    #       slope       (units/week)
    #       intercept
    #       r_squared
    #       p_value     (significance of the trend)
    #       direction   (upward | downward | flat)
    #       first_value 
    #       last_value
    #       pct_change  (% change across the full series)
    #       rolling_mean    (pd.Series of 8-week rolling mean)
    #       is_stationary   (ADF test result (True if stationary at 5 %))
    #
    def analyze_trends(self) -> dict[str, Any]:
        y = self.data.values.astype(float)
        x = np.arange(len(y))

        slope, intercept, r, p, _ = stats.linregress(x, y)  # type: ignore

        if abs(slope) < 1:  # type: ignore
            direction = "flat"
        elif slope > 0:  # type: ignore
            direction = "upward"
        else:
            direction = "downward"

        adf_result = adfuller(y, autolag="AIC")
        is_stationary = adf_result[1] < 0.05

        first, last = float(y[0]), float(y[-1])
        pct = 100.0 * (last - first) / first if first != 0 else float("nan")

        result = {
            "slope": float(slope),  # type: ignore
            "intercept": float(intercept),  # type: ignore
            "r_squared": float(r ** 2),  # type: ignore
            "p_value": float(p),  # type: ignore
            "direction": direction,
            "first_value": first,
            "last_value": last,
            "pct_change": pct,
            "rolling_mean": self.data.rolling(8, center=True).mean(),
            "is_stationary": is_stationary,
        }

        logger.info(
            "Trend: %s (slope=%.2f, R²=%.3f, p=%.4f, stationary=%s)",
            direction,
            slope,  # type: ignore
            r ** 2,  # type: ignore
            p,  # type: ignore
            is_stationary,
        )
        return result

   
    # Seasonality detection using STL decomposition and ACF analysis
    # Returns dictionary with keys:
    #       period             (seasonal period tested)
    #       seasonal_strength  (fraction of variance explained by seasonal component)
    #       acf_at_period      (ACF value at period lag)
    #       has_seasonality    (True if seasonal_strength > 0.4 or acf_at_period > 0.3)
    #       decomposition      (statsmodels DecomposeResult object)
    #
    def detect_seasonality(self, period: int = 52) -> dict[str, Any]:
        from statsmodels.tsa.seasonal import seasonal_decompose
        from statsmodels.tsa.stattools import acf

        decomp = seasonal_decompose(
            self.data.dropna(), model="additive", period=period
        )

        seasonal_var = np.var(decomp.seasonal.dropna())
        total_var = np.var(self.data.dropna())
        strength = seasonal_var / total_var if total_var > 0 else 0.0

        acf_vals = acf(self.data.dropna(), nlags=max(period + 5, 60))
        acf_at_period = float(acf_vals[period]) if period < len(acf_vals) else float("nan")

        has_seasonality = strength > 0.4 or acf_at_period > 0.3

        result = {
            "period": period,
            "seasonal_strength": float(strength),
            "acf_at_period": acf_at_period,
            "has_seasonality": has_seasonality,
            "decomposition": decomp,
        }

        logger.info(
            "Seasonality: strength=%.3f, ACF[%d]=%.3f, detected=%s",
            strength,
            period,
            acf_at_period,
            has_seasonality,
        )
        return result

   
    # Compute distributional statistics including monthly breakdown
    # Returns dictionary with keys:
    #       mean, median, std, min, max
    #       skewness, kurtosis
    #       monthly_means       (pd.Series indexed 1–12)
    #       monthly_stds        (pd.Series indexed 1–12)
    #       cv                  (coefficient of variation)
    #       outlier_count       (number of IQR-based outliers)
    #
    def get_distribution_stats(self) -> dict[str, Any]:
        s = self.data.dropna()

        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        outliers = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()

        df = s.to_frame("demand")
        df.index = pd.to_datetime(df.index)  # Ensure DatetimeIndex
        df["month"] = df.index.month  # type: ignore
        monthly_means = df.groupby("month")["demand"].mean()
        monthly_stds = df.groupby("month")["demand"].std()

        result = {
            "mean": float(s.mean()),
            "median": float(s.median()),
            "std": float(s.std()),
            "min": float(s.min()),
            "max": float(s.max()),
            "skewness": float(s.skew()),  # type: ignore
            "kurtosis": float(s.kurtosis()),  # type: ignore
            "cv": float(s.std() / s.mean()) if s.mean() != 0 else float("nan"),
            "monthly_means": monthly_means,
            "monthly_stds": monthly_stds,
            "outlier_count": int(outliers),
        }

        logger.info(
            "Distribution: mean=%.1f, std=%.1f, cv=%.2f, outliers=%d",
            result["mean"],
            result["std"],
            result["cv"],
            result["outlier_count"],
        )
        return result

   

    #
    # Generate and save all EDA visualizations using src.utils.visualization functions
    #
    def plot_all_eda_charts(self) -> None:
        """Generate and save all Phase 1 EDA visualisations."""
        from src.utils.visualization import (  # type: ignore
            plot_autocorrelation,
            plot_monthly_distribution,
            plot_seasonality,
            plot_time_series,
        )

        logger.info("Generating EDA charts …")
        plot_time_series(self.data, title="Weekly Demand — Full History")
        plot_seasonality(self.data, period=52)
        plot_autocorrelation(self.data, lags=60)
        plot_monthly_distribution(self.data)
        logger.info("EDA charts saved to results/visualizations/.")
