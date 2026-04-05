#
#  Interactive visualization utils.
#
# saves an HTML files to results/visualizations/ for each plot, which can be opened in a browser
#

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import statsmodels.api as sm

from src.config import FIGURES_DIR, FIGURE_HEIGHT, FIGURE_WIDTH, PLOTLY_TEMPLATE

from statsmodels.tsa.stattools import acf, pacf


#
# Write figure to an HTML file in FIGURES_DIR with the given filename
#
def _save(fig: go.Figure, filename: str) -> None:
    path = FIGURES_DIR / f"{filename}.html"
    fig.write_html(str(path))


##################################################################
#           Raw and cleaned time series
#


#
# line chart of the raw or cleaned time series, with optional rolling-average overlay
#
def plot_time_series(data: pd.Series, title: str = "Weekly Demand", filename: str = "time_series", rolling_window: int = 8) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data.values,
            mode="lines",
            name="Weekly demand",
            line=dict(color="#1b71ae", width=1.5),
        )
    )

    if rolling_window > 1:
        rolling = data.rolling(rolling_window, center=True).mean()
        fig.add_trace(
            go.Scatter(
                x=rolling.index,
                y=rolling.values,
                mode="lines",
                name=f"{rolling_window}-week MA",
                line=dict(color="#ff7f0e", width=2.5, dash="dot"),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Units",
        template=PLOTLY_TEMPLATE,
        width=FIGURE_WIDTH,
        height=FIGURE_HEIGHT,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )

    _save(fig, filename)
    return fig



##################################################################
#           Seasonal decomposition
#


#
# Seasonal decomposition using statsmodels
#
def plot_seasonality(data: pd.Series, period: int = 52, filename: str = "seasonality_decomposition") -> go.Figure:
    decomp = sm.tsa.seasonal_decompose(data.dropna(), model="additive", period=period)

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        subplot_titles=["Observed", "Trend", "Seasonal", "Residual"],
        vertical_spacing=0.07,
    )
    traces = [
        (decomp.observed, "#1f77b4"),
        (decomp.trend, "#ff7f0e"),
        (decomp.seasonal, "#2ca02c"),
        (decomp.resid, "#711b1b"),
    ]
    for row, (series, color) in enumerate(traces, start=1):
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                line=dict(color=color, width=1.5),
                showlegend=False,
            ),
            row=row,
            col=1,
        )

    fig.update_layout(
        title=f"Seasonal Decomposition (period = {period} weeks)",
        height=900,
        width=FIGURE_WIDTH,
        template=PLOTLY_TEMPLATE,
    )

    _save(fig, filename)
    return fig




##################################################################
#           Forecast vs actual

#
# Overlay line chart of forecast vs actual, optionally showing training history in the background
#
def plot_forecast_comparison( actual: pd.Series, forecast: pd.Series, title: str = "Forecast vs Actual",
                             filename: str = "forecast_comparison", train: pd.Series | None = None,) -> go.Figure:
    fig = go.Figure()

    if train is not None:
        fig.add_trace(
            go.Scatter(
                x=train.index,
                y=train.values,
                mode="lines",
                name="Training history",
                line=dict(color="#aec7e8", width=1),
                opacity=0.6,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=actual.index,
            y=actual.values,
            mode="lines+markers",
            name="Actual",
            line=dict(color="#1f77b4", width=2.5),
            marker=dict(size=7),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast.index,
            y=forecast.values,
            mode="lines+markers",
            name="Forecast",
            line=dict(color="#d62728", width=2.5, dash="dash"),
            marker=dict(size=7, symbol="x"),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Units",
        template=PLOTLY_TEMPLATE,
        width=FIGURE_WIDTH,
        height=FIGURE_HEIGHT,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )

    _save(fig, filename)
    return fig




##################################################################
#           ACF / PACF
# ---------------------------------------------------------------------------

#
# ACF and PACF charts
#
def plot_autocorrelation( data: pd.Series, lags: int = 60, filename: str = "autocorrelation",) -> go.Figure:
    acf_vals, acf_conf, *_ = acf(data.dropna(), nlags=lags, alpha=0.05)
    pacf_vals, pacf_conf, *_ = pacf(data.dropna(), nlags=lags, alpha=0.05)
    lag_range = list(range(lags + 1))

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["ACF", "PACF"],
        horizontal_spacing=0.12,
    )

    for col, (vals, conf) in enumerate(
        [(acf_vals, acf_conf), (pacf_vals, pacf_conf)], start=1
    ):
        upper = conf[:, 1] - vals
        lower = vals - conf[:, 0]

        fig.add_trace(
            go.Bar(
                x=lag_range,
                y=vals,
                error_y=dict(type="data", array=upper, arrayminus=lower, visible=True),
                name="ACF" if col == 1 else "PACF",
                marker_color="#1f77b4",
            ),
            row=1,
            col=col,
        )

    fig.update_layout(
        title="Autocorrelation functions",
        template=PLOTLY_TEMPLATE,
        width=FIGURE_WIDTH,
        height=450,
        showlegend=False,
    )

    _save(fig, filename)
    return fig




##################################################################
#           Distribution by month
#

#
# Box plots of weekly demand by calendar month
#
def plot_monthly_distribution( data: pd.Series, filename: str = "monthly_distribution",) -> go.Figure:
    df = data.to_frame("demand")
    df["month"] = pd.DatetimeIndex(df.index).month
    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]

    fig = go.Figure()
    for m in range(1, 13):
        subset = df.loc[df["month"] == m, "demand"]
        fig.add_trace(
            go.Box(y=subset.values, name=month_names[m - 1], showlegend=False)
        )

    fig.update_layout(
        title="Weekly Demand Distribution by Month",
        xaxis_title="Month",
        yaxis_title="Units",
        template=PLOTLY_TEMPLATE,
        width=FIGURE_WIDTH,
        height=FIGURE_HEIGHT,
    )

    _save(fig, filename)
    return fig
