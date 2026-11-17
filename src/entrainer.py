"""Train, evaluate, and serialize the house-price regression model."""

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "maisons.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_FILE = MODEL_DIR / "modele.pkl"
METADATA_FILE = MODEL_DIR / "metadata.json"
FEATURES = ["surface", "chambres"]
TARGET = "prix"


def train_model() -> dict:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_FILE}. Run: python -m src.generer_donnees"
        )
    dataset = pd.read_csv(DATA_FILE)
    required_columns = FEATURES + [TARGET]
    missing_columns = [column for column in required_columns if column not in dataset.columns]
    if missing_columns:
        raise ValueError(f"Missing required dataset columns: {missing_columns}")
    if dataset[required_columns].isna().any().any():
        raise ValueError("Dataset contains missing values in required columns.")

    X = dataset[FEATURES]
    y = dataset[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "mae": round(float(mean_absolute_error(y_test, predictions)), 2),
        "mse": round(float(mean_squared_error(y_test, predictions)), 2),
        "r2": round(float(r2_score(y_test, predictions)), 4),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with MODEL_FILE.open("wb") as model_file:
        pickle.dump(model, model_file)

    metadata = {
        "model_name": "house-price-linear-regression",
        "model_version": "1.0.0",
        "algorithm": "LinearRegression",
        "features": FEATURES,
        "target": TARGET,
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "metrics": metrics,
        "coefficients": [float(value) for value in model.coef_],
        "intercept": float(model.intercept_),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    METADATA_FILE.write_text(json.dumps(metadata, indent=4), encoding="utf-8")
    print("Training completed.")
    print(f"MAE: {metrics['mae']}")
    print(f"MSE: {metrics['mse']}")
    print(f"R2: {metrics['r2']}")
    print(f"Model saved to: {MODEL_FILE}")
    print(f"Metadata saved to: {METADATA_FILE}")
    return metadata


if __name__ == "__main__":
    train_model()