import pandas as pd

from app.data.loader import load_economic_data
from app.models.train import FEATURES, load_model
from app.services.pipeline import run_scenario


df = load_economic_data()
model = load_model()

result = run_scenario(
    "income_tax",
    20,
    25,
    5,
)

baseline = result["baseline_history"][-1]
counterfactual = result["counterfactual_history"][-1]


print("\n========== DATASET GDP RANGE ==========")

print("Min GDP:", df["gdp"].min())
print("Max GDP:", df["gdp"].max())
print("Last GDP:", df.iloc[-1]["gdp"])


print("\n========== FINAL SIMULATION STATES ==========")


for name, state in [
    ("BASELINE", baseline),
    ("COUNTERFACTUAL", counterfactual),
]:

    print(f"\n--- {name} ---")

    for feature in FEATURES:

        train_min = df[feature].min()
        train_max = df[feature].max()
        value = state[feature]

        outside = (
            value < train_min
            or value > train_max
        )

        print(
            f"{feature:20s} "
            f"value={value:12.4f} "
            f"train_range=({train_min:.4f}, {train_max:.4f}) "
            f"{'OUTSIDE RANGE' if outside else 'OK'}"
        )


print("\n========== ML PREDICTIONS ==========")


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


print(
    "Baseline simulation GDP:",
    baseline["gdp"],
)

print(
    "Baseline ML GDP:",
    baseline_ml,
)

print(
    "Counterfactual simulation GDP:",
    counterfactual["gdp"],
)

print(
    "Counterfactual ML GDP:",
    counterfactual_ml,
)


baseline_difference = (
    abs(baseline_ml - baseline["gdp"])
    / baseline["gdp"]
    * 100
)

counterfactual_difference = (
    abs(counterfactual_ml - counterfactual["gdp"])
    / counterfactual["gdp"]
    * 100
)


print("\n========== ML VS SIMULATION ERROR ==========")

print(
    f"Baseline difference: "
    f"{baseline_difference:.2f}%"
)

print(
    f"Counterfactual difference: "
    f"{counterfactual_difference:.2f}%"
)