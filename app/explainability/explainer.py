import pandas as pd
from sklearn.inspection import permutation_importance

from app.models.train import FEATURES, load_model


def explain_model(
    df: pd.DataFrame,
    model_dir: str = "artifacts/models",
) -> pd.DataFrame:

    model = load_model(model_dir)

    X = df[FEATURES]
    y = df["gdp"]

    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=8,
        random_state=42,
        scoring="neg_mean_absolute_error",
    )

    return (
        pd.DataFrame(
            {
                "feature": FEATURES,
                "importance": result.importances_mean,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def generate_explanation(comparison: pd.DataFrame) -> list[str]:

    messages = []

    row = comparison.set_index("indicator")

    # --------------------------------------------------
    # GDP
    # --------------------------------------------------

    if "gdp" in row.index:

        impact = float(row.loc["gdp", "percentage_impact"])

        if impact > 0:
            messages.append(
                f"Projected GDP increased by {impact:.2f}% "
                "relative to the baseline scenario."
            )

        elif impact < 0:
            messages.append(
                f"Projected GDP decreased by {abs(impact):.2f}% "
                "relative to the baseline scenario."
            )

        else:
            messages.append(
                "Projected GDP is unchanged relative to the baseline scenario."
            )

    # --------------------------------------------------
    # Household consumption
    # --------------------------------------------------

    if "consumption" in row.index:

        impact = float(row.loc["consumption", "percentage_impact"])

        if impact > 0:
            messages.append(
                f"Household consumption is projected to be "
                f"{impact:.2f}% higher."
            )

        elif impact < 0:
            messages.append(
                f"Household consumption is projected to be "
                f"{abs(impact):.2f}% lower."
            )

        else:
            messages.append(
                "Household consumption is projected to remain unchanged."
            )

    # --------------------------------------------------
    # Government tax revenue
    # --------------------------------------------------

    if "tax_revenue" in row.index:

        impact = float(row.loc["tax_revenue", "percentage_impact"])

        if impact > 0:
            messages.append(
                f"Government tax revenue is projected to be "
                f"{impact:.2f}% higher."
            )

        elif impact < 0:
            messages.append(
                f"Government tax revenue is projected to be "
                f"{abs(impact):.2f}% lower."
            )

        else:
            messages.append(
                "Government tax revenue is projected to remain unchanged."
            )

    # --------------------------------------------------
    # Unemployment
    # --------------------------------------------------

    if "unemployment" in row.index:

        impact = float(row.loc["unemployment", "absolute_impact"])

        if impact > 0:
            messages.append(
                f"Unemployment increases by {impact:.2f} "
                "percentage points relative to the baseline."
            )

        elif impact < 0:
            messages.append(
                f"Unemployment decreases by {abs(impact):.2f} "
                "percentage points relative to the baseline."
            )

        else:
            messages.append(
                "Unemployment is unchanged from the baseline scenario."
            )

    return messages