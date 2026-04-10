import pandas as pd
import logging
from typing import Dict, Any, Optional
from src.optimization.ilp_model import ILPShippingModel
from src.optimization.solver import Solver

logger = logging.getLogger(__name__)


# Compare optimization solutions with baselines
class OptimizationComparison:

    def __init__(self):
        self.results = None

    def compare(
        self,
        ilp_solution: Dict[str, int],
        baseline_solutions: Dict[str, Dict[str, int]],
        ilp_model: "ILPShippingModel",
        solver: "Solver",
    ) -> pd.DataFrame:
        results_list = []

        # ILP solution
        if ilp_solution:
            ilp_metrics = solver.get_metrics(ilp_model, ilp_solution)
            results_list.append(
                {
                    "Strategy": "ILP Optimal",
                    "Total Days": ilp_metrics["total_delivery_days"],
                    "Avg Days/Unit": ilp_metrics["avg_delivery_days"],
                    "Total Cost": f"${ilp_metrics['total_cost']}",
                    "Budget Used %": f"{ilp_metrics['budget_used']}%",
                    "Fast Units": ilp_metrics["fast_units"],
                    "Fast %": f"{ilp_metrics['fast_ratio']}%",
                    "Feasible": "Yes",
                }
            )

        # baseline solutions
        for baseline_name, allocation in baseline_solutions.items():
            if allocation is None:
                results_list.append(
                    {
                        "Strategy": baseline_name,
                        "Total Days": "-",
                        "Avg Days/Unit": "-",
                        "Total Cost": "-",
                        "Budget Used %": "-",
                        "Fast Units": "-",
                        "Fast %": "-",
                        "Feasible": "No",
                    }
                )
            else:
                baseline_metrics = solver.get_metrics(ilp_model, allocation)
                is_valid = solver.validate_solution(ilp_model, allocation)
                results_list.append(
                    {
                        "Strategy": baseline_name,
                        "Total Days": baseline_metrics["total_delivery_days"],
                        "Avg Days/Unit": baseline_metrics["avg_delivery_days"],
                        "Total Cost": f"${baseline_metrics['total_cost']}",
                        "Budget Used %": f"{baseline_metrics['budget_used']}%",
                        "Fast Units": baseline_metrics["fast_units"],
                        "Fast %": f"{baseline_metrics['fast_ratio']}%",
                        "Feasible": "Yes" if is_valid else "No",
                    }
                )

        self.results = pd.DataFrame(results_list)
        return self.results



    def get_comparison_table(self) -> pd.DataFrame:
        if self.results is None:
            raise ValueError("Call compare() first")
        return self.results



    def get_improvement(self, ilp_days: int, baseline_days: int) -> float:
        if baseline_days == 0:
            return 0.0
        return ((baseline_days - ilp_days) / baseline_days) * 100



    def get_summary(self) -> Optional[Dict[str, Any]]:
        if self.results is None:
            raise ValueError("Call compare() first")

        ilp_row = self.results[self.results["Strategy"] == "ILP Optimal"]
        if ilp_row.empty:
            return None

        ilp_days = ilp_row.iloc[0]["Total Days"]
        ilp_cost = ilp_row.iloc[0]["Total Cost"]

        return {
            "best_strategy": "ILP Optimal",
            "total_days": ilp_days,
            "total_cost": ilp_cost,
            "results_df": self.results,
        }
