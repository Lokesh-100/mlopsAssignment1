import pandas as pd
from sklearn.compose import ColumnTransformer

from src.preprocessing import get_preprocessor


def sample_dataframe():
    return pd.DataFrame({
        "age": [50, 60, 70],
        "chol": [220, 240, 260],
        "sex": ["M", "F", "M"],
        "cp": ["typical", "asymptomatic", "typical"]
    })


def test_get_preprocessor_returns_column_transformer():
    X = sample_dataframe()
    preprocessor = get_preprocessor(X)
    assert isinstance(preprocessor, ColumnTransformer)


def test_correct_columns_identified():
    X = sample_dataframe()
    preprocessor = get_preprocessor(X)
    num_transformer = preprocessor.transformers[0]
    assert list(num_transformer[2]) == ["age", "chol"]
