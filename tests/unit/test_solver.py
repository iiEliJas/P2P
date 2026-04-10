import pytest
from src.optimization.solver import Solver
from src.optimization.ilp_model import ILPShippingModel


@pytest.fixture
def ilp_model_setup():
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

    model = ILPShippingModel(
        demand, modes, delivery_times, costs, capacities, budget
    )
    model.build_model()

    return model


def test_solver_initialization():
    solver = Solver(time_limit=60)
    assert solver.time_limit == 60
    assert solver.solution is None
    assert solver.status is None


def test_solver_solve(ilp_model_setup):
    solver = Solver()
    solution = solver.solve(ilp_model_setup)

    assert solution is not None
    assert isinstance(solution, dict)
    assert len(solution) == 4


def test_solver_solution_is_integer(ilp_model_setup):
    solver = Solver()
    solution = solver.solve(ilp_model_setup)

    for mode, units in solution.items():    # type: ignore
        assert isinstance(units, int)
        assert units >= 0


def test_solver_get_metrics(ilp_model_setup):
    solver = Solver()
    solution = solver.solve(ilp_model_setup)

    metrics = solver.get_metrics(ilp_model_setup, solution)    # type: ignore

    assert "total_cost" in metrics
    assert "total_delivery_days" in metrics
    assert "avg_delivery_days" in metrics
    assert "fast_units" in metrics
    assert "fast_ratio" in metrics
    assert "budget_used" in metrics


def test_solver_validate_solution(ilp_model_setup):
    solver = Solver()
    solution = solver.solve(ilp_model_setup)

    is_valid = solver.validate_solution(ilp_model_setup, solution)  # type: ignore
    assert is_valid is True


def test_solver_validate_infeasible_solution(ilp_model_setup):
    solver = Solver()
    bad_solution = {
        "First Class": 0,
        "Same Day": 0,
        "Second Class": 0,
        "Standard Class": 0,
    }

    is_valid = solver.validate_solution(ilp_model_setup, bad_solution)
    assert is_valid is False


def test_solver_get_status(ilp_model_setup):
    solver = Solver()
    assert solver.get_status() == "Not solved"

    solver.solve(ilp_model_setup)
    status = solver.get_status()
    assert status in ["Optimal", "Not Solved"]


def test_solver_solution_meets_demand(ilp_model_setup):
    solver = Solver()
    solution = solver.solve(ilp_model_setup)

    total_units = sum(solution.values())    # type: ignore
    assert total_units == ilp_model_setup.demand


def test_solver_solution_respects_budget(ilp_model_setup):
    solver = Solver()
    solution = solver.solve(ilp_model_setup)

    total_cost = sum(
        ilp_model_setup.costs[mode] * units
        for mode, units in solution.items()    # type: ignore
    )
    assert total_cost <= ilp_model_setup.budget
