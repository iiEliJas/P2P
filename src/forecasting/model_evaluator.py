import pandas as pd
import logging
from typing import Dict, List, Any

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.config import FIGURES_DIR, FIGURE_WIDTH, FIGURE_HEIGHT, PLOTLY_TEMPLATE

logger = logging.getLogger(__name__)


# Compare and evaluate forecasting models
class ModelEvaluator:

    def __init__(self, models_dict: Dict[str, Any], test_data: pd.Series):
        self.models = models_dict
        self.test_data = test_data
        self.results = None
        self.predictions = None



    def evaluate_all(self) -> pd.DataFrame:
        results_list = []

        for model_name, model in self.models.items():
            try:
                logger.info(f"Evaluating {model_name}...")

                # Generate predictions
                if model.is_fitted:
                    forecast = model.predict()
                else:
                    raise ValueError(f"{model_name} is not fitted")

                # Store predictions
                if self.predictions is None:
                    self.predictions = {}
                self.predictions[model_name] = forecast

                # Evaluate
                metrics = model.evaluate(self.test_data, forecast)
                metrics["Model"] = model_name
                results_list.append(metrics)

            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
                continue

        self.results = pd.DataFrame(results_list)
        self.results = self.results.sort_values("smape").reset_index(drop=True)
        return self.results.copy()



    def get_comparison_table(self) -> pd.DataFrame:
        if self.results is None:
            raise ValueError("Call evaluate_all() first")

        table = self.results.copy()
        table["Rank"] = range(1, len(table) + 1)
        table["SMAPE %"] = table["smape"].apply(lambda x: f"{x:.2f}%")
        table["MAE"] = table["mae"].apply(lambda x: f"{x:.2f}")
        table["RMSE"] = table["rmse"].apply(lambda x: f"{x:.2f}")

        return table[["Rank", "Model", "SMAPE %", "MAE", "RMSE"]]



    def get_best_model(self) -> str:
        if self.results is None:
            raise ValueError("Call evaluate_all() first")
        return self.results.iloc[0]["Model"]



    def get_summary(self) -> Dict[str, Any]:
        if self.results is None:
            raise ValueError("Call evaluate_all() first")

        best_idx = 0
        worst_idx = len(self.results) - 1

        best_model = self.results.iloc[best_idx]
        worst_model = self.results.iloc[worst_idx]

        improvement = (
            (worst_model["smape"] - best_model["smape"]) / worst_model["smape"] * 100
        )

        return {
            "best_model": best_model["Model"],
            "best_smape": f"{best_model['smape']:.2f}%",
            "best_mae": f"{best_model['mae']:.2f}",
            "worst_model": worst_model["Model"],
            "worst_smape": f"{worst_model['smape']:.2f}%",
            "improvement": f"{improvement:.1f}%",
            "total_models": len(self.results),
        }



    def plot_smape_comparison(self, filename: str = "smape_comparison") -> go.Figure:
        if self.results is None:
            raise ValueError("Call evaluate_all() first")

        sorted_results = self.results.sort_values("smape")

        fig = go.Figure(
            data=[
                go.Bar(
                    x=sorted_results["Model"],
                    y=sorted_results["smape"],
                    marker_color="#1f77b4",
                    text=sorted_results["smape"].apply(lambda x: f"{x:.1f}%"),
                    textposition="auto",
                )
            ]
        )

        fig.update_layout(
            title="SMAPE (%) Comparison Across Models",
            xaxis_title="Model",
            yaxis_title="SMAPE (%)",
            template=PLOTLY_TEMPLATE,
            width=FIGURE_WIDTH,
            height=FIGURE_HEIGHT,
            showlegend=False,
        )

        fig.write_html(FIGURES_DIR / f"{filename}.html")
        return fig



    def plot_mae_comparison(self, filename: str = "mae_comparison") -> go.Figure:
        if self.results is None:
            raise ValueError("Call evaluate_all() first")

        sorted_results = self.results.sort_values("mae")

        fig = go.Figure(
            data=[
                go.Bar(
                    x=sorted_results["Model"],
                    y=sorted_results["mae"],
                    marker_color="#ff7f0e",
                    text=sorted_results["mae"].apply(lambda x: f"{x:.1f}"),
                    textposition="auto",
                )
            ]
        )

        fig.update_layout(
            title="MAE Comparison Across Models",
            xaxis_title="Model",
            yaxis_title="Mean Absolute Error",
            template=PLOTLY_TEMPLATE,
            width=FIGURE_WIDTH,
            height=FIGURE_HEIGHT,
            showlegend=False,
        )

        fig.write_html(FIGURES_DIR / f"{filename}.html")
        return fig



    def plot_forecast_comparison(
        self,
        filename: str = "forecast_comparison_all",
        train_data: pd.Series | None = None,
    ) -> go.Figure:
        if self.predictions is None:
            raise ValueError("Call evaluate_all() first")

        fig = go.Figure()

        # Add training data if provided
        if train_data is not None:
            fig.add_trace(
                go.Scatter(
                    x=train_data.index,
                    y=train_data.values,
                    mode="lines",
                    name="Training history",
                    line=dict(color="#aec7e8", width=1),
                    opacity=0.6,
                )
            )

        # Add test data
        fig.add_trace(
            go.Scatter(
                x=self.test_data.index,
                y=self.test_data.values,
                mode="lines+markers",
                name="Actual",
                line=dict(color="#1f77b4", width=2.5),
                marker=dict(size=8),
            )
        )

        # Add forecasts from all models
        colors = ["#d62728", "#2ca02c", "#9467bd", "#8c564b"]
        for idx, (model_name, forecast) in enumerate(self.predictions.items()):
            fig.add_trace(
                go.Scatter(
                    x=forecast.index,
                    y=forecast.values,
                    mode="lines+markers",
                    name=f"{model_name} Forecast",
                    line=dict(color=colors[idx % len(colors)], width=2.5, dash="dash"),
                    marker=dict(size=7, symbol="x"),
                )
            )

        fig.update_layout(
            title="Forecast Comparison: All Models vs Actual",
            xaxis_title="Date",
            yaxis_title="Units",
            template=PLOTLY_TEMPLATE,
            width=FIGURE_WIDTH,
            height=FIGURE_HEIGHT,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            hovermode="x unified",
        )

        fig.write_html(FIGURES_DIR / f"{filename}.html")
        return fig



    def plot_metrics_heatmap(self, filename: str = "metrics_heatmap") -> go.Figure:
        if self.results is None:
            raise ValueError("Call evaluate_all() first")

        # Prepare data for heatmap
        heatmap_data = self.results.set_index("Model")[["mae", "smape", "rmse"]]
        heatmap_data.columns = ["MAE", "SMAPE", "RMSE"]

        fig = go.Figure(
            data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale="Viridis",
                text=np.round(heatmap_data.values, 2),
                texttemplate="%{text}",
                textfont={"size": 12},
            )
        )

        fig.update_layout(
            title="Model Metrics Heatmap",
            xaxis_title="Metrics",
            yaxis_title="Models",
            template=PLOTLY_TEMPLATE,
            width=FIGURE_WIDTH,
            height=400,
        )

        fig.write_html(FIGURES_DIR / f"{filename}.html")
        return fig


import numpy as np
