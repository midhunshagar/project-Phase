from pathlib import Path
import pandas as pd
from .generator import generate_synthetic_data

REQUIRED_COLUMNS = {
    "year", "gdp", "inflation", "unemployment", "consumption",
    "investment", "government_spending", "tax_rate", "interest_rate",
    "tax_revenue", "emissions", "inequality"
}

def load_economic_data(path: str = "datasets/processed/economic_sample.csv") -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return generate_synthetic_data(str(p))
    df = pd.read_csv(p)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    return df.sort_values("year").reset_index(drop=True)

def clean_external_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates()
    numeric = [c for c in df.columns if c != "year"]
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[numeric] = df[numeric].interpolate(limit_direction="both")
    return df.dropna(subset=["year"]).reset_index(drop=True)
