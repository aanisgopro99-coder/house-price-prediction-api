"""FastAPI application serving the saved house-price model."""

import json
import pickle
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = PROJECT_ROOT / "models" / "modele.pkl"
METADATA_FILE = PROJECT_ROOT / "models" / "metadata.json"

app = FastAPI(
    title="House Price Prediction API",
    description="Predict a house price from its surface and number of bedrooms.",
    version="1.0.0",
)


def load_artifacts():
    """Load only the model artifact created by this project."""
    if not MODEL_FILE.exists() or not METADATA_FILE.exists():
        raise RuntimeError(
            "Model files are missing. Run python -m src.generer_donnees "
            "and python -m src.entrainer before starting the API."
        )
    # Only load pickle files produced locally by this trusted project.
    with MODEL_FILE.open("rb") as model_file:
        loaded_model = pickle.load(model_file)
    loaded_metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    return loaded_model, loaded_metadata


model, metadata = load_artifacts()


class PredictionRequest(BaseModel):
    surface: Annotated[float, Field(gt=0, le=1000, description="House surface in m²")]
    chambres: Annotated[int, Field(ge=1, le=20, description="Number of bedrooms")]


class PredictionResponse(BaseModel):
    predicted_price: float


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/model-info")
def model_info():
    return metadata


@app.post("/predict", response_model=PredictionResponse)
def predict_price(request: PredictionRequest):
    try:
        input_data = pd.DataFrame(
            [[request.surface, request.chambres]],
            columns=metadata["features"],
        )
        prediction = model.predict(input_data)[0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc
    return PredictionResponse(predicted_price=round(float(prediction), 2))