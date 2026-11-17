"""Generate a reproducible synthetic house-price dataset."""

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_FILE = DATA_DIR / "maisons.csv"


def generate_dataset(n_samples: int = 500, random_state: int = 42) -> pd.DataFrame:
    """Create synthetic data using the formula described in the training guide."""
    rng = np.random.default_rng(random_state)
    surface = rng.uniform(40, 220, n_samples)
    chambres = rng.integers(1, 7, n_samples)
    noise = rng.normal(0, 15_000, n_samples)
    prix = 1_500 * surface + 8_000 * chambres + 20_000 + noise
    return pd.DataFrame({
        "surface": surface.round(2),
        "chambres": chambres,
        "prix": prix.round(2),
    })


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset = generate_dataset()
    dataset.to_csv(OUTPUT_FILE, index=False)
    print(f"Dataset generated: {OUTPUT_FILE}")
    print(f"Rows: {len(dataset)}")
    print(dataset.head())


if __name__ == "__main__":
    main()