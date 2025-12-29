from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo

from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH


def load_and_clean():
    # Ensure directories exist
    Path(RAW_DATA_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(PROCESSED_DATA_PATH).parent.mkdir(parents=True, exist_ok=True)
    # Fetch dataset from UCI
    heart_disease = fetch_ucirepo(id=45)

    # Features and target
    X = heart_disease.data.features
    y = heart_disease.data.targets

    # Combine into single dataframe
    df = pd.concat([X, y], axis=1)

    # Save raw data
    df.to_csv(RAW_DATA_PATH, index=False)

    # Basic cleaning
    df_clean = df.replace("?", pd.NA)
    df_clean = df_clean.dropna()

    # Save processed data
    df_clean.to_csv(PROCESSED_DATA_PATH, index=False)

    return df_clean


if __name__ == "__main__":
    df = load_and_clean()
    print("✅ Heart Disease dataset fetched from UCI")
    print(f"Raw data saved to: {RAW_DATA_PATH}")
    print(f"Clean data saved to: {PROCESSED_DATA_PATH}")
    print(f"Final shape: {df.shape}")
