import pandas as pd
from .train import FEATURES, load_model

def predict_gdp(state: dict, model_dir: str = "artifacts/models") -> float:
    model = load_model(model_dir)
    row = pd.DataFrame([{feature: state[feature] for feature in FEATURES}])
    return float(model.predict(row)[0])
