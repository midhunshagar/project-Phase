import pandas as pd
from app.simulation.engine import INDICATORS

def compare(baseline: dict, counterfactual: dict) -> pd.DataFrame:
    rows = []
    for indicator in INDICATORS:
        b = float(baseline[indicator])
        c = float(counterfactual[indicator])
        impact = c - b
        pct = (impact / b * 100.0) if b != 0 else 0.0
        rows.append({
            "indicator": indicator,
            "baseline": b,
            "counterfactual": c,
            "absolute_impact": impact,
            "percentage_impact": pct
        })
    return pd.DataFrame(rows)
