from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = [
    "inflation",
    "unemployment",
    "consumption",
    "investment",
    "government_spending",
    "tax_rate",
    "interest_rate",
    "tax_revenue",
    "emissions",
    "inequality",
]

TARGETS = ["gdp"]


def generate_synthetic_data(
    path: str = "datasets/processed/economic_sample.csv",
    n_rows: int = 120,
    seed: int = 42,
) -> pd.DataFrame:

    rng = np.random.default_rng(seed)

    rows = []

    # Starting GDP
    gdp = 1000.0

    for year in range(1980, 1980 + n_rows):

        # -----------------------------
        # Economic conditions
        # -----------------------------

        inflation = float(
            np.clip(
                rng.normal(5.0, 1.4),
                1.0,
                12.0,
            )
        )

        unemployment = float(
            np.clip(
                rng.normal(6.0, 1.2),
                2.5,
                15.0,
            )
        )

        tax_rate = float(
            np.clip(
                rng.normal(20.0, 3.0),
                8.0,
                35.0,
            )
        )

        interest_rate = float(
            np.clip(
                2.0
                + 0.7 * inflation
                + rng.normal(0, 1.2),
                1.0,
                14.0,
            )
        )

        # -----------------------------
        # Economic components
        # -----------------------------

        government_spending = float(
            max(
                100.0,
                0.25 * gdp + rng.normal(0, 20),
            )
        )

        consumption = float(
            max(
                100.0,
                0.60
                * gdp
                * (1 - 0.015 * (tax_rate - 20))
                + rng.normal(0, 15),
            )
        )

        investment = float(
            max(
                50.0,
                0.20
                * gdp
                * (1 - 0.02 * (interest_rate - 6))
                + rng.normal(0, 12),
            )
        )

        tax_revenue = float(
            max(
                20.0,
                gdp * tax_rate / 100 * 0.95
                + rng.normal(0, 8),
            )
        )

        emissions = float(
            max(
                30.0,
                0.50 * gdp + rng.normal(0, 20),
            )
        )

        inequality = float(
            np.clip(
                0.35
                + 0.002 * (tax_rate - 20)
                - 0.001
                * (government_spending / gdp - 0.25)
                * 100
                + rng.normal(0, 0.015),
                0.18,
                0.65,
            )
        )

        # -----------------------------
        # GDP growth
        # -----------------------------

        # Government spending is expressed
        # relative to GDP instead of using
        # its absolute value.
        spending_effect = (
            government_spending / gdp - 0.25
        )

        growth = (
            0.025
            - 0.0015 * (interest_rate - 6)
            - 0.0010 * (tax_rate - 20)
            - 0.0015 * (unemployment - 6)
            + 0.02 * spending_effect
            + rng.normal(0, 0.012)
        )

        # Prevent unrealistic explosive growth.
        growth = float(
            np.clip(
                growth,
                -0.05,
                0.05,
            )
        )

        # Calculate next year's GDP.
        next_gdp = max(
            150.0,
            gdp * (1 + growth),
        )

        rows.append(
            {
                "year": year,
                "gdp": next_gdp,
                "inflation": inflation,
                "unemployment": unemployment,
                "consumption": consumption,
                "investment": investment,
                "government_spending": government_spending,
                "tax_rate": tax_rate,
                "interest_rate": interest_rate,
                "tax_revenue": tax_revenue,
                "emissions": emissions,
                "inequality": inequality,
            }
        )

        gdp = next_gdp

    df = pd.DataFrame(rows)

    # Make sure the output directory exists.
    Path(path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save dataset.
    df.to_csv(
        path,
        index=False,
    )

    return df