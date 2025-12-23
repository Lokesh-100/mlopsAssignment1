import pandas as pd

def test_data_load():
    df = pd.read_csv("data/processed/heart_clean.csv")
    assert not df.empty
