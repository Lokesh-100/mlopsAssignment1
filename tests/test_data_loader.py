import pandas as pd
from types import SimpleNamespace

from src import data_loader


def mock_ucimlrepo(*args, **kwargs):
    """Mocked UCI dataset object"""
    features = pd.DataFrame({
        "age": [63, 67, "?"],
        "chol": [233, 286, 250]
    })
    targets = pd.DataFrame({
        "target": [1, 0, 1]
    })

    return SimpleNamespace(
        data=SimpleNamespace(
            features=features,
            targets=targets
        )
    )


def test_load_and_clean(monkeypatch, tmp_path):
    monkeypatch.setattr(
        data_loader,
        "fetch_ucirepo",
        mock_ucimlrepo
    )
    raw_path = tmp_path / "raw.csv"
    processed_path = tmp_path / "processed.csv"

    monkeypatch.setattr(data_loader, "RAW_DATA_PATH", raw_path)
    monkeypatch.setattr(data_loader, "PROCESSED_DATA_PATH", processed_path)
    df_clean = data_loader.load_and_clean()
    assert raw_path.exists()
    assert processed_path.exists()
    assert df_clean.isna().sum().sum() == 0
    assert len(df_clean) == 2
    assert set(df_clean.columns) == {"age", "chol", "target"}
    assert isinstance(df_clean, pd.DataFrame)
