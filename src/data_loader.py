import pandas as pd
from config import DATA_PATH, PROCESSED_DATA_PATH

def load_and_clean():
    df = pd.read_csv(DATA_PATH)
    df.replace("?", pd.NA, inplace=True)
    df = df.dropna()
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    return df
