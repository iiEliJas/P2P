import pulp
import logging
from typing import Dict, Optional
from src.optimization.ilp_model import ILPShippingModel

logger = logging.getLogger(__name__)


# Solver wrapper for ILP models with result extraction and validation
class Solver:

    def __init__(self, time_limit: int = 60):
        self.time_limit = time_limit
        self.solution = None
        self.status = None

    def solve(self, ilp_model: ILPShippingModel) -> Optional[Dict]:
        self._require_fitted(ilp_model)

        ilp_model.prob.solve(   #type: ignore
            pulp.PULP_CBC_CMD(msg=False, timeLimit=self.time_limit, threads=4)
        )

        self.status = ilp_model.prob.status #type: ignore

        if self.status != pulp.LpStatusOptimal:
            logger.warning(
                f"Solver status: {pulp.LpStatus[self.status]} (not optimal)"
            )
            return None

        self.solution = self._extract_solution(ilp_model)

        logger.info(f"Optimal solution found. Status: {pulp.LpStatus[self.status]}")
        return self.solution



    def _extract_solution(self, ilp_model: ILPShippingModel) -> Dict:
        allocation = {}

        for mode in ilp_model.modes:
            value = int(ilp_model.x[mode].varValue) #type: ignore
            allocation[mode] = value
            logger.debug(f"  {mode}: {value} units")

        return allocation



    def get_metrics(self, ilp_model: ILPShippingModel, allocation: Dict) -> Dict:
        total_cost = sum(
            ilp_model.costs[mode] * allocation[mode] for mode in ilp_model.modes
        )

        total_delivery_days = sum(
            ilp_model.delivery_times[mode] * allocation[mode]
            for mode in ilp_model.modes
        )

        avg_delivery_days = total_delivery_days / ilp_model.demand

        fast_units = sum(
            allocation[mode]
            for mode in ilp_model.fast_modes
            if mode in allocation
        )
        fast_ratio = (fast_units / ilp_model.demand) * 100

        return {
            "total_cost": round(total_cost, 2),
            "total_delivery_days": int(total_delivery_days),
            "avg_delivery_days": round(avg_delivery_days, 2),
            "fast_units": int(fast_units),
            "fast_ratio": round(fast_ratio, 2),
            "budget_used": round((total_cost / ilp_model.budget) * 100, 2),
        }
    


    def validate_solution(self, ilp_model: ILPShippingModel, allocation: Dict) -> bool:
        # check demand
        total_units = sum(allocation.values())
        if total_units != ilp_model.demand:
            logger.error(
                f"Demand not met: {total_units} vs {ilp_model.demand} required"
            )
            return False

        # check budget
        total_cost = sum(
            ilp_model.costs[mode] * allocation[mode] for mode in ilp_model.modes
        )
        if total_cost > ilp_model.budget:
            logger.error(
                f"Budget exceeded: ${total_cost} vs ${ilp_model.budget} limit"
            )
            return False

        # check capacities
        for mode in ilp_model.modes:
            if allocation[mode] > ilp_model.capacities[mode]:
                logger.error(
                    f"Capacity exceeded for {mode}: "
                    f"{allocation[mode]} vs {ilp_model.capacities[mode]} max"
                )
                return False

        # check service level
        fast_units = sum(
            allocation[mode] for mode in ilp_model.fast_modes if mode in allocation
        )
        min_fast = ilp_model.fast_service_ratio * ilp_model.demand
        if fast_units < min_fast:
            logger.error(
                f"Service level not met: "
                f"{fast_units} vs {min_fast} required fast units"
            )
            return False

        return True



    def get_status(self) -> str:
        if self.status is None:
            return "Not solved"
        return pulp.LpStatus[self.status]
    

    #
    # Check if build has been called
    #
    def _require_fitted(self, ilp_model: ILPShippingModel) -> None:
        if ilp_model.prob is None:
            raise RuntimeError(
                f"{self.__class__.__name__} must call build_model() before solve()."
            )
