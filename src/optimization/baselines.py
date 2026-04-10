import pandas as pd
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# Standard baseline
def all_standard_baseline(
    demand: float,
    shipping_modes: list,
    capacities: Dict[str, float],
    costs: Dict[str, float],
    budget: float,
) -> Optional[Dict[str, int]]:
    allocation = {mode: 0 for mode in shipping_modes}

    # Try to use Standard Class for everything
    if demand <= capacities.get("Standard Class", 0):
        allocation["Standard Class"] = int(demand)
        logger.info("All Standard baseline: Feasible ✓")
        return allocation

    # If Standard Class exceeds capacity, use other modes
    standard_capacity = capacities.get("Standard Class", 0)
    allocation["Standard Class"] = int(standard_capacity)
    remaining = demand - standard_capacity

    # Fill remaining with whatever capacity is available
    other_modes = [m for m in shipping_modes if m != "Standard Class"]
    for mode in other_modes:
        if remaining <= 0:
            break
        available = min(remaining, capacities.get(mode, 0))
        allocation[mode] = int(available)
        remaining -= available

    if remaining > 0:
        logger.warning("All Standard baseline: INFEASIBLE")
        return None

    logger.info("All Standard baseline: Feasible")
    return allocation
