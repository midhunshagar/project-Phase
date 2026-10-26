from app.simulation.economy import EconomicState
from app.simulation.policies import PolicyScenario
from app.simulation.engine import simulate

def test_simulation_returns_requested_horizon():
    state = EconomicState()
    policy = PolicyScenario("income_tax", 20, 25, 5)
    history = simulate(state, policy)
    assert len(history) == 6
    assert history[-1]["year"] == state.year + 5

def test_tax_policy_changes_tax_rate():
    state = EconomicState()
    policy = PolicyScenario("income_tax", 20, 30, 1)
    history = simulate(state, policy)
    assert history[-1]["tax_rate"] == 30

def test_invalid_duration():
    state = EconomicState()
    policy = PolicyScenario("income_tax", 20, 25, 0)
    try:
        simulate(state, policy)
        assert False
    except ValueError:
        assert True
