from app.data.loader import load_economic_data
from app.models.train import train_models

if __name__ == "__main__":
    df = load_economic_data()
    best, metrics = train_models(df)
    print("ECONOSIM-AI setup complete.")
    print(f"Rows available: {len(df)}")
    print(f"Best GDP model: {best}")
    print("Metrics:")
    for name, values in metrics.items():
        print(name, values)
