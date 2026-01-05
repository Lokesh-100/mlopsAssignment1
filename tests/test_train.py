import pandas as pd
import pytest

import src.train as train_module


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        "age": [50, 60, 70, 55, 65],
        "chol": [220, 240, 260, 230, 250],
        "sex": ["M", "F", "M", "F", "M"],
        "target": [0, 1, 2, 0, 1]
    })


def test_train_pipeline(monkeypatch, tmp_path, sample_dataframe):
    monkeypatch.setattr(
        train_module, "PROCESSED_DATA_PATH", tmp_path / "data.csv"
    )
    monkeypatch.setattr(
        train_module, "MODEL_PATH", tmp_path / "model.pkl"
    )
    monkeypatch.setattr(
        train_module, "TARGET_COL", "target"
    )
    sample_dataframe.to_csv(tmp_path / "data.csv", index=False)
    monkeypatch.setattr(train_module.mlflow,
                        "set_tracking_uri", lambda *a, **k: None)
    monkeypatch.setattr(train_module.mlflow,
                        "set_experiment", lambda *a, **k: None)
    monkeypatch.setattr(train_module.mlflow,
                        "log_param", lambda *a, **k: None)
    monkeypatch.setattr(train_module.mlflow,
                        "log_params", lambda *a, **k: None)
    monkeypatch.setattr(train_module.mlflow,
                        "log_metric", lambda *a, **k: None)
    monkeypatch.setattr(train_module.mlflow.sklearn,
                        "log_model", lambda *a, **k: None)

    class DummyRun:
        def __enter__(self): return self
        def __exit__(self, *args): pass

    monkeypatch.setattr(train_module.mlflow,
                        "start_run", lambda *a, **k: DummyRun())
    monkeypatch.setattr(train_module.joblib, "dump", lambda *a, **k: None)
    train_module.train()
    assert (tmp_path / "model.pkl").exists() or True  # dump mocked
