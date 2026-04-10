import pytest
import pandas as pd
from src.optimization.ilp_model import ILPShippingModel
from src.optimization.solver import Solver
from src.optimization.baselines import all_standard_baseline
from src.optimization.optimization_comparison import OptimizationComparison


@pytest.fixture
def complete_optimization_setup():
    demand = 1918
    modes = ["First Class", "Same Day", "Second Class", "Standard Class"]
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
        "First Class": 560.0,
        "Same Day": 350.0,
        "Second Class": 1000.0,
        "Standard Class": 2000.0,
    }
    budget = 5500

    return demand, modes, delivery_times, costs, capacities, budget


def test_full_optimization_pipeline(complete_optimization_setup):
    demand, modes, delivery_times, costs, capacities, budget = complete_optimization_setup

    # build ILP model
    ilp_model = ILPShippingModel(
        demand, modes, delivery_times, costs, capacities, budget
    )
    ilp_model.build_model()
    assert ilp_model.prob is not None

    # Solver
    solver = Solver()
    ilp_solution = solver.solve(ilp_model)
    assert ilp_solution is not None

    is_valid = solver.validate_solution(ilp_model, ilp_solution)
    assert is_valid is True

    # get metrics
    metrics = solver.get_metrics(ilp_model, ilp_solution)
    assert metrics["total_cost"] > 0
    assert metrics["total_delivery_days"] > 0


def test_baseline_comparison_pipeline(complete_optimization_setup):
    demand, modes, delivery_times, costs, capacities, budget = complete_optimization_setup

    # build and solve ILP
    ilp_model = ILPShippingModel(
        demand, modes, delivery_times, costs, capacities, budget
    )
    ilp_model.build_model()

    solver = Solver()
    ilp_solution = solver.solve(ilp_model)

    # generate baseline
    baseline_solution = all_standard_baseline(demand, modes, capacities, costs, budget)

    # compare
    comparison = OptimizationComparison()
    results = comparison.compare(
        ilp_solution,   #type: ignore
        {"Baseline All Standard": baseline_solution}, #type: ignore
        ilp_model,
        solver,
    )

    assert results is not None
    assert len(results) == 2
    assert "ILP Optimal" in results["Strategy"].values


def test_optimization_improves_over_baseline(complete_optimization_setup):
    demand, modes, delivery_times, costs, capacities, budget = complete_optimization_setup

    # Build and solve ILP
    ilp_model = ILPShippingModel(
        demand, modes, delivery_times, costs, capacities, budget
    )
    ilp_model.build_model()

    solver = Solver()
    ilp_solution = solver.solve(ilp_model)

    assert ilp_solution is not None
    # ILP metrics
    ilp_metrics = solver.get_metrics(ilp_model, ilp_solution)

    # Baseline metrics
    baseline_solution = all_standard_baseline(demand, modes, capacities, costs, budget)
    assert baseline_solution is not None
    baseline_metrics = solver.get_metrics(ilp_model, baseline_solution)

    assert ilp_metrics["total_delivery_days"] <= baseline_metrics["total_delivery_days"]


def test_solution_feasibility_throughout_pipeline(complete_optimization_setup):
    demand, modes, delivery_times, costs, capacities, budget = complete_optimization_setup

    ilp_model = ILPShippingModel(
        demand, modes, delivery_times, costs, capacities, budget
    )
    ilp_model.build_model()

    solver = Solver()
    solution = solver.solve(ilp_model)
    assert solution is not None

    assert solver.validate_solution(ilp_model, solution) is True

    assert sum(solution.values()) == demand

    # Budget
    total_cost = sum(
        costs[mode] * solution[mode] for mode in modes
    )
    assert total_cost <= budget

    # Capacities
    for mode in modes:
        assert solution[mode] <= capacities[mode]
