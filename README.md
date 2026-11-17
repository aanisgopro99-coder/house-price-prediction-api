# House Price Prediction API

A reproducible machine-learning project that trains a Linear Regression model to estimate house prices from surface area and number of bedrooms, then serves predictions through a FastAPI REST API.

## Features

- Generate a small, reproducible demonstration dataset.
- Train and evaluate a scikit-learn Linear Regression model.
- Save the fitted model and training metadata.
- Expose `/predict`, `/health`, and `/model-info` endpoints.
- Validate prediction inputs with Pydantic.
- Run automated API tests with pytest.
- Build and run the service with Docker; CI workflow included.

## Requirements

- Python 3.11
- Git
- Docker Desktop (optional, for container execution)

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

## Generate data and train

```powershell
python -m src.generer_donnees
python -m src.entrainer
```

The generated dataset is saved in `data/maisons.csv`. The trained model and metadata are stored in `models/`.

## Run the API

```powershell
python -m uvicorn src.main:app --reload
```

- Interactive API docs: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health
- Model metadata: http://127.0.0.1:8000/model-info

Example request to `POST /predict`:

```json
{
  "surface": 100,
  "chambres": 3
}
```

## Run tests

```powershell
python -m pytest -q
```

## Docker

Generate the model first, then:

```powershell
docker build -t house-price-api .
docker run --rm -p 8000:8000 house-price-api
```

## Project scope

This is a learning project demonstrating a complete small ML-to-API workflow. The dataset is generated for demonstration and is not a real-estate market dataset. The model is educational and should not be used for real property valuation. Cloud deployment and DVC are not implemented in this version.
