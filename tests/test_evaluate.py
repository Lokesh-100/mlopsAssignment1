import pandas as pd

import src.evaluate as evaluate_module


def test_perform_eda_and_log(monkeypatch, tmp_path):
    # ----------------------------
    # Create dummy processed data
    # ----------------------------
    df = pd.DataFrame({
        "age": [50, 60, 70],
        "chol": [220, 240, 260],
        "target": [0, 1, 1]
    })

    processed_path = tmp_path / "processed.csv"
    df.to_csv(processed_path, index=False)

    artifact_dir = tmp_path / "eda_artifacts"

    # 🔑 IMPORTANT FIX: create directory explicitly
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Patch config paths
    # ----------------------------
    monkeypatch.setattr(evaluate_module, "PROCESSED_DATA_PATH", processed_path)
    monkeypatch.setattr(evaluate_module, "ARTIFACT_DIR", artifact_dir)

    # ----------------------------
    # Mock MLflow
    # ----------------------------
    monkeypatch.setattr(evaluate_module.mlflow,
                        "log_param", lambda *a, **k: None)
    monkeypatch.setattr(evaluate_module.mlflow,
                        "log_artifact", lambda *a, **k: None)

    class DummyRun:
        def __enter__(self): return self
        def __exit__(self, *args): pass

    monkeypatch.setattr(
        evaluate_module.mlflow,
        "start_run",
        lambda *a, **k: DummyRun()
    )

    # ----------------------------
    # Run EDA
    # ----------------------------
    evaluate_module.perform_eda_and_log()

    # ----------------------------
    # Assertions: artifacts created
    # ----------------------------
    assert (artifact_dir / "class_balance.png").exists()
    assert (artifact_dir / "feature_histograms.png").exists()
    assert (artifact_dir / "correlation_heatmap.png").exists()
