from dataclasses import asdict
from app.simulation.economy import EconomicState
from app.simulation.policies import PolicyScenario
from app.simulation.engine import simulate, final_state

def run_baseline_and_counterfactual(initial_state: EconomicState, policy: PolicyScenario):
    baseline_policy = PolicyScenario(
        name=policy.name,
        baseline_value=policy.baseline_value,
        counterfactual_value=policy.baseline_value,
        duration_years=policy.duration_years
    )
    baseline_history = simulate(initial_state, baseline_policy)
    counterfactual_history = simulate(initial_state, policy)
    return baseline_history, counterfactual_history
