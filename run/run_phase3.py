#
# Run phase 3
#

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_loader import DataLoader
from src.utils.data_cleaner import DataCleaner
from src.forecasting.nbeats_model import NBeatsModel
from src.forecasting.model_trainer import ModelTrainer
from src.config import (
    DATA_RAW,
    DATAFILE,
    TRAIN_WEEKS,
    TEST_WEEKS,
    FORECAST_HORIZON,
    RESULTS_DIR,
)
from src.optimization.ilp_model import ILPShippingModel
from src.optimization.solver import Solver
from src.optimization.baselines import all_standard_baseline
from src.optimization.optimization_comparison import OptimizationComparison
from src.utils.visualization import (
    plot_allocation_comparison,
    plot_delivery_time_comparison,
    plot_cost_breakdown,
    plot_service_level,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_phase3")


#####################################################
#           Helpers
#

def _separator(title: str) -> None:
    logger.info("\n\n")
    logger.info("=" * 60)
    logger.info("  %s", title)
    logger.info("=" * 60)


#####################################################
#           Main pipeline
#

def main():
    _separator("PHASE 3: OPTIMIZATION")


    #################################
    # Load
    #
    _separator("STEP 1 Load data")
    try:
        loader = DataLoader(DATA_RAW / DATAFILE)
        raw_data = loader.load()
        logger.info(f"Loaded {len(raw_data)} records")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return False



    #################################
    # Clean
    #
    _separator("STEP 2 Clean data")
    try:
        cleaner = DataCleaner(raw_data, frequency="W")
        weekly = cleaner.aggregate_by_frequency()
        weekly = cleaner.handle_missing_values()
        weekly = cleaner.remove_outliers(method="iqr")

        train, test = cleaner.train_test_split(
            train_weeks=TRAIN_WEEKS,
            test_weeks=TEST_WEEKS,
        )
        logger.info(f"Train: {len(train)} weeks | Test: {len(test)} weeks")
    except Exception as e:
        logger.error(f"Failed to prepare data: {e}")
        return False



    #################################
    # Train
    #
    _separator("STEP 3 Train N-BEATS")
    try:
        nbeats = NBeatsModel(
            horizon=FORECAST_HORIZON,
            lookback=16,
            epochs=50,
            batch_size=32,
            hidden_size=64,
            num_stacks=15,
            num_blocks=2,
        )
        trainer = ModelTrainer(nbeats, train, test)
        metrics = trainer.train()

        logger.info("N-BEATS trained successfully")
        logger.info(f"  SMAPE: {metrics['smape']:.2f}%")
        logger.info(f"  MAE: {metrics['mae']:.2f}")

        forecast = nbeats.predict()
        forecast_demand = int(forecast.sum())
        logger.info(f"  Demand Forecast: {forecast_demand} units")
    except Exception as e:
        logger.error(f"Failed to train model: {e}")
        return False



    #################################
    # Setup optimization
    #
    _separator("STEP 4 Setup optimization")
    shipping_modes = ["First Class", "Same Day", "Second Class", "Standard Class"]
    delivery_times = {
        "First Class": 2.0,
        "Same Day": 1.0,
        "Second Class": 3.0,
        "Standard Class": 4.0,
    }
    costs = {
        "First Class": 1.5,
        "Same Day": 2.5,
        "Second Class": 1.0,
        "Standard Class": 0.8,
    }
    capacities = {
        "First Class": 2500.0,
        "Same Day": 1500.0,
        "Second Class": 9000.0,
        "Standard Class": 9000.0,
    }
    budget = 20058
    fast_service_ratio = 0.10
    logger.info("Optimization parameters loaded")



    #################################
    # Build and solve ILP
    #
    _separator("STEP 5 Build and solve ILP")
    try:
        ilp_model = ILPShippingModel(
            demand_forecast=forecast_demand,
            shipping_modes=shipping_modes,
            delivery_times=delivery_times,
            costs=costs,
            capacities=capacities,
            budget=budget,
            fast_service_ratio=fast_service_ratio,
        )

        ilp_model.build_model()
        logger.info("ILP model built")

        solver = Solver(time_limit=60)
        ilp_solution = solver.solve(ilp_model)

        if not ilp_solution:
            logger.error("Failed to find optimal solution")
            return False

        logger.info("Optimal solution found")

        ilp_metrics = solver.get_metrics(ilp_model, ilp_solution)
        logger.info(f"  Total Days: {ilp_metrics['total_delivery_days']}")
        logger.info(f"  Total Cost: ${ilp_metrics['total_cost']}")
        logger.info(f"  Fast Ratio: {ilp_metrics['fast_ratio']}%")
    except Exception as e:
        logger.error(f"Failed to solve optimization: {e}")
        return False

    #################################
    # Compare and visualize
    #
    _separator("STEP 6 Compare and visualize")
    try:
        baseline_solution = all_standard_baseline(
            forecast_demand,
            shipping_modes,
            capacities,
            costs,
            budget,
        )
        if baseline_solution is None:
            logger.error("Failed to compute baseline solution")
            return False

        baseline_metrics = solver.get_metrics(ilp_model, baseline_solution)

        comparison = OptimizationComparison()
        results = comparison.compare(
            ilp_solution,
            {"Baseline (All Standard)": baseline_solution},
            ilp_model,
            solver,
        )

        improvement = comparison.get_improvement(
            ilp_metrics["total_delivery_days"],
            baseline_metrics["total_delivery_days"],
        )

        _separator("OPTIMIZATION COMPARISON RESULTS")
        logger.info("\n" + results.to_string())
        logger.info("=" * 70)
        logger.info(f"\nImprovement over baseline: {improvement:.1f}%")

        allocations = {
            "ILP Optimal": ilp_solution,
            "Baseline All Standard": baseline_solution,
        }
        
        plot_allocation_comparison(allocations, shipping_modes)
        logger.info("  Allocation comparison")

        plot_delivery_time_comparison(results)
        logger.info("  Delivery time comparison")

        plot_cost_breakdown(ilp_solution, costs)
        logger.info("  Cost breakdown")

        plot_service_level(results)
        logger.info("  Service level compliance") 
    except Exception as e:
        logger.error(f"Failed to generate comparisons: {e}")
        return False


    _separator("PHASE 3 COMPLETE")
    logger.info(f"Summary: Demand={forecast_demand}, Days={ilp_metrics['total_delivery_days']}, Cost=${ilp_metrics['total_cost']}, Fast Ratio={ilp_metrics['fast_ratio']}%, Improvement={improvement:.1f}%")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
