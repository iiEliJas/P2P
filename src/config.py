from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
#
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
FIGURES_DIR = RESULTS_DIR / "visualizations"
FORECASTS_DIR = RESULTS_DIR / "forecasts"

for _dir in [DATA_PROCESSED, MODELS_DIR, FIGURES_DIR, FORECASTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)



# ---------------------------------------------------------------------------
# Dataset
#
DATAFILE = "DataCoSupplyChainDataset.csv"
DATAFILE_ENCODING = "latin-1"

# column names must be like in the csv file
COL_ORDER_DATE = "order date (DateOrders)"
COL_QUANTITY = "Order Item Quantity"
COL_SHIP_DATE = "shipping date (DateOrders)"



# ---------------------------------------------------------------------------
# Data split
# 
FREQUENCY = "W"
TRAIN_WEEKS = 134
TEST_WEEKS = 8



# ---------------------------------------------------------------------------
# Forecasting
# 
FORECAST_HORIZON = 8
LOOKBACK_WINDOW = 8

MSTL_SEASONAL_PERIODS = [52, 13]
MSTL_TREND_WINDOW = 35



# ---------------------------------------------------------------------------
# Visualization
#
PLOTLY_TEMPLATE = "plotly_white"
FIGURE_WIDTH = 1100
FIGURE_HEIGHT = 500