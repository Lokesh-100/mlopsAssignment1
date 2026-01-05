import pandas as pd
import src.evaluate as evaluate_module


def test_perform_eda_and_log(monkeypatch, tmp_path):
    # ----------------------------
    # Create dummy processed data
    # ----------------------------
    df = pd.DataFrame({
        "age": [50, 60, 70],
        "chol": [220, 240, 260],
        "num": [0, 1, 2]   # ✅ MUST be `num`
    })

    processed_path = tmp_path / "processed.csv"
    df.to_csv(processed_path, index=False)

    artifact_dir = tmp_path / "eda_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Patch config paths
    # ----------------------------
    monkeypatch.setattr(evaluate_module, "PROCESSED_DATA_PATH", processed_path)
    monkeypatch.setattr(evaluate_module, "ARTIFACT_DIR", artifact_dir)

    # ----------------------------
    # Mock MLflow
    # ----------------------------
    monkeypatch.setattr(
        evaluate_module.mlflow,
        "log_param",
        lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        evaluate_module.mlflow,
        "log_artifact",
        lambda *args, **kwargs: None
    )

    class DummyRun:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            pass

    monkeypatch.setattr(
        evaluate_module.mlflow,
        "start_run",
        lambda *args, **kwargs: DummyRun()
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
