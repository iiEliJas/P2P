import pytest
from src.optimization.baselines import all_standard_baseline


@pytest.fixture
def baseline_setup():
    demand = 1918
    modes = ["First Class", "Same Day", "Second Class", "Standard Class"]
    capacities = {
        "First Class": 560,
        "Same Day": 350,
        "Second Class": 1000,
        "Standard Class": 2000,
    }
    costs = {
        "First Class": 1.5,
        "Same Day": 2.5,
        "Second Class": 1.0,
        "Standard Class": 0.8,
    }
    budget = 5500

    return demand, modes, capacities, costs, budget


def test_all_standard_baseline_feasible(baseline_setup):
    demand, modes, capacities, costs, budget = baseline_setup
    allocation = all_standard_baseline(demand, modes, capacities, costs, budget)

    assert allocation is not None
    assert isinstance(allocation, dict)


def test_all_standard_baseline_is_integer(baseline_setup):
    demand, modes, capacities, costs, budget = baseline_setup
    allocation = all_standard_baseline(demand, modes, capacities, costs, budget)

    for mode, units in allocation.items():  #type: ignore
        assert isinstance(units, int)
        assert units >= 0


def test_all_standard_baseline_covers_demand(baseline_setup):
    demand, modes, capacities, costs, budget = baseline_setup
    allocation = all_standard_baseline(demand, modes, capacities, costs, budget)

    if allocation is not None:
        total_units = sum(allocation.values())
        assert total_units == demand or total_units >= demand


def test_all_standard_baseline_respects_capacity(baseline_setup):
    demand, modes, capacities, costs, budget = baseline_setup
    allocation = all_standard_baseline(demand, modes, capacities, costs, budget)

    if allocation is not None:
        for mode, units in allocation.items():
            assert units <= capacities[mode]


def test_all_standard_baseline_prefers_standard(baseline_setup):
    demand, modes, capacities, costs, budget = baseline_setup
    allocation = all_standard_baseline(demand, modes, capacities, costs, budget)

    if allocation is not None:
        standard_class_units = allocation.get("Standard Class", 0)
        assert standard_class_units > 0
