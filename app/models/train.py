from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


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

TARGET = "gdp"


def train_models(
    df: pd.DataFrame,
    model_dir: str = "artifacts/models",
):

    Path(model_dir).mkdir(
        parents=True,
        exist_ok=True,
    )

    # Work on a copy so we don't modify
    # the original dataset.
    df = df.copy()

    # Make sure the data is ordered by year.
    df = df.sort_values("year").reset_index(drop=True)

    # --------------------------------------------------
    # Prepare forecasting dataset
    # --------------------------------------------------

    # The synthetic dataset is already constructed as:
    #
    # current economic indicators -> next-year GDP
    #
    # Therefore we do not need additional lag features
    # here.

    df = df.dropna(
        subset=FEATURES + [TARGET]
    ).reset_index(drop=True)

    # --------------------------------------------------
    # Select features and target
    # --------------------------------------------------

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    # --------------------------------------------------
    # Time-based train/test split
    # --------------------------------------------------

    split = int(len(df) * 0.8)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    # --------------------------------------------------
    # Candidate models
    # --------------------------------------------------

    candidates = {

        "linear_regression":
            LinearRegression(),

        "gradient_boosting":
            GradientBoostingRegressor(
                random_state=42
            ),

        "random_forest":
            RandomForestRegressor(
                n_estimators=250,
                random_state=42,
                n_jobs=-1,
            ),
    }

    # --------------------------------------------------
    # Train and evaluate
    # --------------------------------------------------

    results = {}

    best_name = None
    best_model = None
    best_rmse = float("inf")

    for name, model in candidates.items():

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_test
        )

        mae = mean_absolute_error(
            y_test,
            predictions,
        )

        rmse = mean_squared_error(
            y_test,
            predictions,
        ) ** 0.5

        r2 = r2_score(
            y_test,
            predictions,
        )

        results[name] = {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
        }

        # Lower RMSE = better model.
        if rmse < best_rmse:

            best_rmse = rmse
            best_name = name
            best_model = model

    # --------------------------------------------------
    # Save best model
    # --------------------------------------------------

    joblib.dump(
        best_model,
        Path(model_dir) / "gdp_model.joblib",
    )

    # Save metrics
    pd.DataFrame(results).T.to_csv(
        Path(model_dir) / "metrics.csv"
    )

    return best_name, results


def load_model(
    model_dir: str = "artifacts/models",
):

    return joblib.load(
        Path(model_dir) / "gdp_model.joblib"
    )