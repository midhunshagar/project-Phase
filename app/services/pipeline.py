from pathlib import Path
import json
import pandas as pd
from app.data.loader import load_economic_data
from app.models.train import train_models
from app.models.predict import predict_gdp
from app.simulation.economy import EconomicState
from app.simulation.policies import PolicyScenario
from app.counterfactual.scenario import run_baseline_and_counterfactual
from app.counterfactual.comparison import compare
from app.explainability.explainer import explain_model, generate_explanation

def train_if_needed():
    model_path = Path("artifacts/models/gdp_model.joblib")
    df = load_economic_data()
    if not model_path.exists():
        best, metrics = train_models(df)
    else:
        metrics_path = Path("artifacts/models/metrics.csv")
        best = "saved_model"
        metrics = pd.read_csv(metrics_path, index_col=0).to_dict(orient="index") if metrics_path.exists() else {}
    return df, best, metrics

def run_scenario(policy_name: str, baseline_value: float, counterfactual_value: float, duration_years: int):
    df, best_model, metrics = train_if_needed()

    initial = EconomicState(
        year=int(df.iloc[-1]["year"]),
        gdp=float(df.iloc[-1]["gdp"]),
        inflation=float(df.iloc[-1]["inflation"]),
        unemployment=float(df.iloc[-1]["unemployment"]),
        consumption=float(df.iloc[-1]["consumption"]),
        investment=float(df.iloc[-1]["investment"]),
        government_spending=float(df.iloc[-1]["government_spending"]),
        tax_rate=float(df.iloc[-1]["tax_rate"]),
        interest_rate=float(df.iloc[-1]["interest_rate"]),
        tax_revenue=float(df.iloc[-1]["tax_revenue"]),
        emissions=float(df.iloc[-1]["emissions"]),
        inequality=float(df.iloc[-1]["inequality"]),
    )

    policy = PolicyScenario(policy_name, baseline_value, counterfactual_value, duration_years)
    baseline_history, counterfactual_history = run_baseline_and_counterfactual(initial, policy)

    baseline_final = baseline_history[-1]
    counterfactual_final = counterfactual_history[-1]
    comparison = compare(baseline_final, counterfactual_final)

    predicted_baseline_gdp = predict_gdp(baseline_final)
    predicted_counterfactual_gdp = predict_gdp(counterfactual_final)

    explanation = generate_explanation(comparison)
    importance = explain_model(df)

    return {
        "policy": {
            "name": policy_name,
            "baseline_value": baseline_value,
            "counterfactual_value": counterfactual_value,
            "duration_years": duration_years,
        },
        "baseline_history": baseline_history,
        "counterfactual_history": counterfactual_history,
        "comparison": comparison,
        "predicted_baseline_gdp": predicted_baseline_gdp,
        "predicted_counterfactual_gdp": predicted_counterfactual_gdp,
        "explanation": explanation,
        "feature_importance": importance,
        "model": best_model,
        "metrics": metrics,
    }
