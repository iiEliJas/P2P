import pytest
from src.optimization.ilp_model import ILPShippingModel


@pytest.fixture
def ilp_setup():
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
        "First Class": 560,
        "Same Day": 350,
        "Second Class": 1000,
        "Standard Class": 2000,
    }
    budget = 5500

    return demand, modes, delivery_times, costs, capacities, budget


def test_ilp_initialization(ilp_setup):
    demand, modes, delivery_times, costs, capacities, budget = ilp_setup
    model = ILPShippingModel(
        demand, modes, delivery_times, costs, capacities, budget
    )

    assert model.demand == demand
    assert model.modes == modes
    assert model.delivery_times == delivery_times
    assert model.costs == costs
    assert model.capacities == capacities
    assert model.budget == budget
    assert model.prob is None


def test_ilp_build_model(ilp_setup):
    demand, modes, delivery_times, costs, capacities, budget = ilp_setup
    model = ILPShippingModel(
        demand, modes, delivery_times, costs, capacities, budget
    )

    model.build_model()

    assert model.prob is not None
    assert model.x is not None
    assert len(model.x) == 4


def test_ilp_has_correct_variables(ilp_setup):
    demand, modes, delivery_times, costs, capacities, budget = ilp_setup
    model = ILPShippingModel(
        demand, modes, delivery_times, costs, capacities, budget
    )

    model.build_model()
    variables = model.get_decision_variables()

    assert set(variables.keys()) == set(modes)
    for mode in modes:
        assert variables[mode] is not None


def test_ilp_has_all_constraints(ilp_setup):
    demand, modes, delivery_times, costs, capacities, budget = ilp_setup
    model = ILPShippingModel(
        demand, modes, delivery_times, costs, capacities, budget
    )

    model.build_model()

    # Check that constraints are added
    assert model.prob.constraints is not None   # type: ignore
    constraint_names = [str(c) for c in model.prob.constraints] # type: ignore

    assert any("Demand" in str(c) for c in model.prob.constraints)  # type: ignore
    assert any("Budget" in str(c) for c in model.prob.constraints)  # type: ignore
    assert any("Capacity" in str(c) for c in model.prob.constraints) # type: ignore
    assert any("Service" in str(c) for c in model.prob.constraints) # type: ignore


def test_ilp_parameters_stored_correctly(ilp_setup):
    demand, modes, delivery_times, costs, capacities, budget = ilp_setup
    model = ILPShippingModel(
        demand, modes, delivery_times, costs, capacities, budget, fast_service_ratio=0.15
    )

    assert model.fast_service_ratio == 0.15
    assert model.fast_modes == ["First Class", "Same Day"]


def test_ilp_to_dict(ilp_setup):
    demand, modes, delivery_times, costs, capacities, budget = ilp_setup
    model = ILPShippingModel(
        demand, modes, delivery_times, costs, capacities, budget
    )

    model_dict = model.to_dict()

    assert model_dict["demand"] == demand
    assert model_dict["modes"] == modes
    assert model_dict["budget"] == budget
    assert model_dict["fast_service_ratio"] == 0.10
