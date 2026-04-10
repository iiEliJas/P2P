import pytest
import pandas as pd
from src.optimization.optimization_comparison import OptimizationComparison
from src.optimization.ilp_model import ILPShippingModel
from src.optimization.solver import Solver


@pytest.fixture
def comparison_setup():
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

    ilp_model = ILPShippingModel(
        demand, modes, delivery_times, costs, capacities, budget
    )
    ilp_model.build_model()

    solver = Solver()
    ilp_solution = solver.solve(ilp_model)

    baseline_solutions = {
        "Baseline All Standard": {
            "First Class": 0,
            "Same Day": 0,
            "Second Class": 1000,
            "Standard Class": 918,
        }
    }

    return ilp_model, solver, ilp_solution, baseline_solutions


def test_comparison_initialization():
    comparison = OptimizationComparison()
    assert comparison.results is None


def test_comparison_compare(comparison_setup):
    ilp_model, solver, ilp_solution, baseline_solutions = comparison_setup
    comparison = OptimizationComparison()

    results = comparison.compare(ilp_solution, baseline_solutions, ilp_model, solver)

    assert results is not None
    assert isinstance(results, pd.DataFrame)
    assert len(results) >= 1


def test_comparison_has_required_columns(comparison_setup):
    ilp_model, solver, ilp_solution, baseline_solutions = comparison_setup
    comparison = OptimizationComparison()

    results = comparison.compare(ilp_solution, baseline_solutions, ilp_model, solver)

    required_columns = [
        "Strategy",
        "Total Days",
        "Avg Days/Unit",
        "Total Cost",
        "Fast %",
        "Feasible",
    ]
    for col in required_columns:
        assert col in results.columns


def test_comparison_ilp_optimal_row(comparison_setup):
    ilp_model, solver, ilp_solution, baseline_solutions = comparison_setup
    comparison = OptimizationComparison()

    results = comparison.compare(ilp_solution, baseline_solutions, ilp_model, solver)

    ilp_row = results[results["Strategy"] == "ILP Optimal"]
    assert not ilp_row.empty
    assert ilp_row.iloc[0]["Feasible"] == "Yes"


def test_comparison_get_comparison_table(comparison_setup):
    ilp_model, solver, ilp_solution, baseline_solutions = comparison_setup
    comparison = OptimizationComparison()

    comparison.compare(ilp_solution, baseline_solutions, ilp_model, solver)
    table = comparison.get_comparison_table()

    assert table is not None
    assert len(table) > 0


def test_comparison_get_summary(comparison_setup):
    ilp_model, solver, ilp_solution, baseline_solutions = comparison_setup
    comparison = OptimizationComparison()

    comparison.compare(ilp_solution, baseline_solutions, ilp_model, solver)
    summary = comparison.get_summary()

    assert summary is not None
    assert "best_strategy" in summary
    assert summary["best_strategy"] == "ILP Optimal"


def test_comparison_get_improvement():
    comparison = OptimizationComparison()

    improvement = comparison.get_improvement(ilp_days=5000, baseline_days=6000)
    assert improvement > 0
    assert improvement == pytest.approx(16.67, rel=0.01)


def test_comparison_handles_infeasible_baselines(comparison_setup):
    ilp_model, solver, ilp_solution, _ = comparison_setup
    comparison = OptimizationComparison()

    baseline_solutions = {
        "Infeasible Baseline": None,
        "Feasible Baseline": {
            "First Class": 100,
            "Same Day": 50,
            "Second Class": 500,
            "Standard Class": 1268,
        },
    }

    results = comparison.compare(ilp_solution, baseline_solutions, ilp_model, solver)

    assert len(results) >= 2
    infeasible_row = results[results["Strategy"] == "Infeasible Baseline"]
    assert not infeasible_row.empty
    assert infeasible_row.iloc[0]["Feasible"] == "No"
