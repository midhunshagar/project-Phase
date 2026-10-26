from typing import List, Dict, Any
from .economy import EconomicState
from .policies import PolicyScenario
from .equations import step_economy

INDICATORS = [
    "gdp", "inflation", "unemployment", "consumption", "investment",
    "government_spending", "tax_revenue", "emissions", "inequality"
]

def simulate(initial_state: EconomicState, policy: PolicyScenario) -> List[Dict[str, Any]]:
    policy.validate()
    state = initial_state.copy()
    history = [state.to_dict()]
    for _ in range(policy.duration_years):
        state = step_economy(state, policy)
        history.append(state.to_dict())
    return history

def final_state(history: List[Dict[str, Any]]) -> EconomicState:
    return EconomicState.from_dict(history[-1])
