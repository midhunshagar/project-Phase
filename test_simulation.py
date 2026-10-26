from app.simulation.economy import EconomicState
from app.simulation.policies import PolicyScenario
from app.counterfactual.scenario import run_baseline_and_counterfactual
from app.counterfactual.comparison import compare


initial_state = EconomicState(
    year=2025,
    gdp=1000.0,
    inflation=5.0,
    unemployment=6.0,
    consumption=600.0,
    investment=200.0,
    government_spending=250.0,
    tax_rate=20.0,
    interest_rate=6.0,
    tax_revenue=200.0,
    emissions=500.0,
    inequality=0.35,
)


policies = [
    PolicyScenario(
        name="income_tax",
        baseline_value=20.0,
        counterfactual_value=25.0,
        duration_years=5,
    ),

    PolicyScenario(
        name="interest_rate",
        baseline_value=6.0,
        counterfactual_value=8.0,
        duration_years=5,
    ),

    PolicyScenario(
        name="government_spending",
        baseline_value=250.0,
        counterfactual_value=300.0,
        duration_years=5,
    ),

    PolicyScenario(
        name="subsidy",
        baseline_value=0.0,
        counterfactual_value=10.0,
        duration_years=5,
    ),
]


print("\n===================================")
print("ECONOSIM-AI POLICY TEST")
print("===================================")


for policy in policies:

    print("\n-----------------------------------")
    print(f"Policy: {policy.name}")
    print(
        f"{policy.baseline_value} -> "
        f"{policy.counterfactual_value}"
    )

    baseline_history, counterfactual_history = (
        run_baseline_and_counterfactual(
            initial_state,
            policy,
        )
    )

    baseline_final = baseline_history[-1]
    counterfactual_final = counterfactual_history[-1]

    comparison = compare(
        baseline_final,
        counterfactual_final,
    )

    print("\nGDP:")
    print(f"Baseline:       {baseline_final['gdp']:.2f}")
    print(f"Counterfactual: {counterfactual_final['gdp']:.2f}")

    print("\nFull impact:")
    print(comparison)