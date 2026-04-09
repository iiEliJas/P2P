#
# Runs phase 2
#
# Usage
#    python run/run_phase2.py
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
from src.config import (
    DATA_RAW, DATAFILE, TRAIN_WEEKS, TEST_WEEKS, 
    FORECAST_HORIZON, LOOKBACK_WINDOW, RESULTS_DIR
)
from src.forecasting.mstl_model import MSTLModel
from src.forecasting.baseline_model import BaselineModel
from src.forecasting.nbeats_model import NBeatsModel
from src.forecasting.model_trainer import ModelTrainer
from src.forecasting.model_evaluator import ModelEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_phase2")



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
    _separator("PHASE 2")
    
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
        cleaner = DataCleaner(raw_data, frequency='W')
        weekly = cleaner.aggregate_by_frequency()
        weekly = cleaner.handle_missing_values()
        weekly = cleaner.remove_outliers(method='iqr')
        
        train, test = cleaner.train_test_split(
            train_weeks=TRAIN_WEEKS,
            test_weeks=TEST_WEEKS
        )
        logger.info(f"Train: {len(train)} weeks | Test: {len(test)} weeks")
    except Exception as e:
        logger.error(f"Failed to prepare data: {e}")
        return False
    


    #################################
    # Init models
    #
    _separator("STEP 3 Init models")
    try:
        models = {
            'MSTL': MSTLModel(
                horizon=FORECAST_HORIZON,
                lookback=LOOKBACK_WINDOW
            ),
            'Baseline': BaselineModel(
                horizon=FORECAST_HORIZON,
                lookback=LOOKBACK_WINDOW
            ),
            'N-BEATS': NBeatsModel(
                horizon=FORECAST_HORIZON,
                lookback=LOOKBACK_WINDOW,
                epochs=100,
                batch_size=32,
                hidden_size=128,
                num_stacks=30,
                num_blocks=3,
            )
        }
        for name in models.keys():
            logger.info(f"  {name} init")
    except Exception as e:
        logger.error(f"Failed to init models: {e}")
        return False
    


    #################################
    # Train all models
    #
    _separator("STEP 4 Train all models")
    training_results = []
    
    for name, model in models.items():
        try:
            logger.info(f"\n  Training {name}...")
            trainer = ModelTrainer(model, train, test)
            metrics = trainer.train()
            
            result = {
                'Model': name,
                'MAE': metrics['mae'],
                'SMAPE': metrics['smape'],
                'RMSE': metrics['rmse'],
                'Training Time (s)': trainer.get_training_time(),
            }
            training_results.append(result)
            
            logger.info(f"      {name} completed in {result['Training Time (s)']:.2f}s")
            logger.info(f"      MAE: {result['MAE']:.2f}")
            logger.info(f"      SMAPE: {result['SMAPE']:.2f}%")
            logger.info(f"      RMSE: {result['RMSE']:.2f}")
            
        except Exception as e:
            logger.error(f"  {name} training failed: {e}")
            continue
    


    #################################
    # Evaluate
    #
    _separator("STEP 5 Evaluation")
    try:
        evaluator = ModelEvaluator(models, test)
        results_df = evaluator.evaluate_all()
        
        _separator("MODEL COMPARISON")
        
        comparison_table = evaluator.get_comparison_table()
        logger.info("\n" + comparison_table.to_string(index=False))
        
        summary = evaluator.get_summary()
        logger.info("\n" + "="*70)
        logger.info(f"   BEST MODEL: {summary['best_model']}")
        logger.info(f"   SMAPE: {summary['best_smape']}")
        logger.info(f"   MAE: {summary['best_mae']}")
        logger.info(f"   Improvement over worst: {summary['improvement']}")
        logger.info("="*70)
        
        # Generate visualizations
        logger.info("\nGenerating visualizations...")
        evaluator.plot_smape_comparison()
        logger.info("  SMAPE comparison")
        
        evaluator.plot_mae_comparison()
        logger.info("  MAE comparison")
        
        evaluator.plot_forecast_comparison(train_data=train)
        logger.info("  Forecast comparison")
        
        evaluator.plot_metrics_heatmap()
        logger.info("  Metrics heatmap")
        
    except Exception as e:
        logger.error(f"Failed to evaluate models: {e}")
        return False
    


    _separator("PHASE 2 COMPLETE")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
