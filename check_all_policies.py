import pandas as pd

from app.data.loader import load_economic_data
from app.models.train import FEATURES, load_model
from app.services.pipeline import run_scenario


df = load_economic_data()
model = load_model()


policies = [
    ("income_tax", 20, 25),
    ("interest_rate", 6, 8),
    ("government_spending", 4350, 4700),
    ("subsidy", 0, 5),
]


print("\n========== ML VALIDATION: ALL POLICIES ==========")


for policy_name, baseline_value, counterfactual_value in policies:

    print("\n----------------------------------------")
    print(f"POLICY: {policy_name}")
    print(
        f"Baseline: {baseline_value} | "
        f"Counterfactual: {counterfactual_value}"
    )
    print("----------------------------------------")

    result = run_scenario(
        policy_name,
        baseline_value,
        counterfactual_value,
        5,
    )

    baseline = result["baseline_history"][-1]
    counterfactual = result["counterfactual_history"][-1]

    baseline_X = pd.DataFrame(
        [[baseline[f] for f in FEATURES]],
        columns=FEATURES,
    )

    counterfactual_X = pd.DataFrame(
        [[counterfactual[f] for f in FEATURES]],
        columns=FEATURES,
    )

    baseline_ml = model.predict(baseline_X)[0]
    counterfactual_ml = model.predict(counterfactual_X)[0]

    baseline_error = (
        abs(baseline_ml - baseline["gdp"])
        / baseline["gdp"]
        * 100
    )

    counterfactual_error = (
        abs(counterfactual_ml - counterfactual["gdp"])
        / counterfactual["gdp"]
        * 100
    )

    print(
        f"\nBaseline simulation GDP: "
        f"{baseline['gdp']:.2f}"
    )

    print(
        f"Baseline ML forecast: "
        f"{baseline_ml:.2f}"
    )

    print(
        f"Baseline difference: "
        f"{baseline_error:.2f}%"
    )

    print(
        f"\nCounterfactual simulation GDP: "
        f"{counterfactual['gdp']:.2f}"
    )

    print(
        f"Counterfactual ML forecast: "
        f"{counterfactual_ml:.2f}"
    )

    print(
        f"Counterfactual difference: "
        f"{counterfactual_error:.2f}%"
    )
