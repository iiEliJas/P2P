#
# Runs phase 1
#
# Usage
#    python scripts/run_phase1.py
#    python scripts/run_phase1.py --config other_config.yaml
#    python scripts/run_phase1.py --data other_data.csv
#

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATA_RAW,
    DATAFILE,
    FORECASTS_DIR,
)
from src.eda.exploratory_analysis import ExploratoryAnalysis
from src.forecasting.mstl_model import MSTLModel
from src.utils.data_cleaner import DataCleaner
from src.utils.data_loader import DataLoader
from src.utils.visualization import plot_forecast_comparison


# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_phase1")



#####################################################
#           Helpers
#

def _load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _separator(title: str) -> None:
    logger.info("\n\n")
    logger.info("=" * 60)
    logger.info("  %s", title)
    logger.info("=" * 60)



#####################################################
#           Main pipeline
#

def run(data_path: Path, config: dict) -> None:

    split_cfg = config["split"]
    fc_cfg    = config["forecasting"]
    out_cfg   = config["outputs"]

    #################################
    # Load data
    #
    _separator("STEP 1 Load data")
    loader = DataLoader(data_path)
    raw_df = loader.load()
    summary = loader.get_summary()
    logger.info("Shape          : %s", summary["shape"])
    logger.info("Date range     : %s → %s", *summary["date_range"])
    logger.info("Dupes dropped  : %d", summary["duplicate_rows_dropped"])

    #################################
    # Clean data
    #
    _separator("STEP 2 Clean data")
    cleaner = DataCleaner(raw_df, frequency=config["data"]["frequency"])
    cleaner.aggregate_by_frequency()
    cleaner.handle_missing_values()
    cleaner.remove_outliers(method=config["cleaning"]["outlier_method"])
    train, test = cleaner.train_test_split(
        train_weeks=split_cfg["train_weeks"],
        test_weeks=split_cfg["test_weeks"],
    )
    logger.info("Train: %d weeks  |  Test: %d weeks", len(train), len(test))

    #################################
    # EDA
    #
    _separator("STEP 3 EDA")
    eda = ExploratoryAnalysis(cleaner.get_cleaned_series())

    trend_info  = eda.analyze_trends()
    season_info = eda.detect_seasonality()
    dist_info   = eda.get_distribution_stats()

    logger.info("Trend direction  : %s  (slope=%.2f, R²=%.3f)",
                trend_info["direction"], trend_info["slope"], trend_info["r_squared"])
    logger.info("Seasonality      : detected=%s  (strength=%.3f)",
                season_info["has_seasonality"], season_info["seasonal_strength"])
    logger.info("Demand mean/std  : %.1f / %.1f  (CV=%.2f)",
                dist_info["mean"], dist_info["std"], dist_info["cv"])
    logger.info("Outliers found   : %d", dist_info["outlier_count"])

    if out_cfg["save_plots"]:
        eda.plot_all_eda_charts()
        logger.info("EDA charts saved to results/visualizations/")

    #################################
    # MSTL forecast
    #
    _separator("STEP 4 MSTL Forecast")
    model = MSTLModel(
        horizon=fc_cfg["horizon"],
        lookback=fc_cfg["lookback"],
        seasonal_periods=fc_cfg["seasonal_periods"],
        trend_window=fc_cfg["trend_window"],
    )
    model.fit(train)
    forecast = model.predict()
    logger.info("Forecast values  : %s", forecast.round(1).tolist())

    #################################
    # Evaluate
    #
    _separator("STEP 5 Evaluation")
    metrics = model.evaluate(test, forecast)
    logger.info("MAE    : %.2f",   metrics["mae"])
    logger.info("RMSE   : %.2f",   metrics["rmse"])
    logger.info("sMAPE  : %.2f%%", metrics["smape"])

    #################################
    # Save outputs
    #
    _separator("STEP 6 Save outputs")

    if out_cfg["save_forecast_csv"]:
        forecast_df = pd.DataFrame({
            "date":     forecast.index,
            "forecast": forecast.values,
            "actual":   test.values,
        })
        csv_path = FORECASTS_DIR / "phase1_forecast.csv"
        forecast_df.to_csv(csv_path, index=False)
        logger.info("Forecast CSV     → %s", csv_path)

    if out_cfg["save_metrics_json"]:
        metrics_path = FORECASTS_DIR / "phase1_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Metrics JSON     → %s", metrics_path)

    if out_cfg["save_plots"]:
        plot_forecast_comparison(
            actual=test,
            forecast=forecast,
            title="MSTL Forecast vs Actual — Phase 1",
            filename="phase1_forecast_vs_actual",
            train=train,
        )
        logger.info("Forecast chart saved to results/visualizations/phase1_forecast_vs_actual.html")

    _separator("PHASE 1 COMPLETE")
    logger.info("MAE=%.2f | RMSE=%.2f | sMAPE=%.2f%%",
                metrics["mae"], metrics["rmse"], metrics["smape"])


######################################################
# CLI
#

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Phase 1 supply-chain analytics pipeline."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "phase1_config.yaml"),
        help="Path to phase1_config.yaml (default: configs/phase1_config.yaml)",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Override path to raw CSV (default: value in config + data/raw/)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args   = _parse_args()
    config = _load_config(args.config)

    data_path = (
        Path(args.data)
        if args.data
        else DATA_RAW / config["data"]["filename"]
    )

    run(data_path, config)