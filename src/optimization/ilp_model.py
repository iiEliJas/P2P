import pulp
import pandas as pd
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# Integer Linear Programming model for optimal shipping allocation
class ILPShippingModel:

    def __init__(
        self,
        demand_forecast: float,
        shipping_modes: List[str],
        delivery_times: Dict[str, float],
        costs: Dict[str, float],
        capacities: Dict[str, float],
        budget: float,
        fast_service_ratio: float = 0.10,
    ):
        self.demand = demand_forecast
        self.modes = shipping_modes
        self.delivery_times = delivery_times
        self.costs = costs
        self.capacities = capacities
        self.budget = budget
        self.fast_service_ratio = fast_service_ratio
        self.fast_modes = ["First Class", "Same Day"]

        self.prob = None
        self.x = None
        self.solution = None
        self.status = None



    def build_model(self) -> None:
        self.prob = pulp.LpProblem("Shipping_Allocation", pulp.LpMinimize)

        self.x = {}
        for mode in self.modes:
            self.x[mode] = pulp.LpVariable(
                f"units_{mode.replace(' ', '_')}",
                lowBound=0,
                cat="Integer",
            )

        self.prob += pulp.lpSum(
            [self.delivery_times[mode] * self.x[mode] for mode in self.modes]
        ), "Total_Delivery_Days"

        self._add_constraints()

        logger.info("ILP model built successfully")



    def _add_constraints(self) -> None:
        self.prob += (
            pulp.lpSum([self.x[mode] for mode in self.modes]) == self.demand,   # type: ignore
            "Demand_Fulfillment",
        )

        # Budget constraint
        self.prob += (
            pulp.lpSum(
                [self.costs[mode] * self.x[mode] for mode in self.modes]    # type: ignore
            ) <= self.budget,
            "Budget_Limit",
        )

        # capacity constraints
        for mode in self.modes:
            self.prob += (
                self.x[mode] <= self.capacities[mode],  # type: ignore
                f"Capacity_{mode.replace(' ', '_')}",
            )

        min_fast_units = self.fast_service_ratio * self.demand
        self.prob += (
            pulp.lpSum([self.x[mode] for mode in self.fast_modes]) >= min_fast_units,   # type: ignore
            "Service_Level",
        )



    def get_decision_variables(self) -> Dict[str, pulp.LpVariable]:
        if self.x is None:
            raise ValueError("Model not built - Call build_model() first")
        return self.x.copy()



    def to_dict(self) -> Dict:
        return {
            "demand": self.demand,
            "modes": self.modes,
            "delivery_times": self.delivery_times,
            "costs": self.costs,
            "capacities": self.capacities,
            "budget": self.budget,
            "fast_service_ratio": self.fast_service_ratio,
        }
