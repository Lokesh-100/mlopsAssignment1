import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from src.config import PROCESSED_DATA_PATH, TARGET_COL, MODEL_PATH
from src.preprocessing import get_preprocessor


# ------------------ MLflow configuration ------------------
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Heart Disease Prediction")


def train():
    # ------------------ Load and prepare data ------------------
    df = pd.read_csv(PROCESSED_DATA_PATH)

    # Convert multi-class target to binary (0 = no disease, 1 = disease)
    df[TARGET_COL] = (df[TARGET_COL] > 0).astype(int)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ------------------ Preprocessing ------------------
    preprocessor = get_preprocessor(X)

    # ------------------ Models ------------------
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    # ------------------ Ensure model directory exists ------------------
    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)

    # ------------------ Training & MLflow logging ------------------
    best_model = None
    best_auc = 0.0

    for name, model in models.items():
        with mlflow.start_run(run_name=name):

            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", model),
                ]
            )

            pipeline.fit(X_train, y_train)

            preds = pipeline.predict(X_test)
            probas = pipeline.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, preds)
            auc = roc_auc_score(y_test, probas)

            # Log global parameters
            mlflow.log_param("model_name", name)
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("split_random_state", 42)

            # Log model parameters safely
            model_params = model.get_params()
            model_params.pop("random_state", None)
            mlflow.log_params(model_params)

            # Log metrics
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("roc_auc", auc)

            # Log model artifact
            mlflow.sklearn.log_model(pipeline, artifact_path="model")

            if auc > best_auc:
                best_auc = auc
                best_model = pipeline

    # ------------------ Save best model ------------------
    joblib.dump(best_model, MODEL_PATH)

    print(f"✅ Best model saved to {MODEL_PATH}")
    print(f"🏆 Best ROC-AUC: {best_auc:.4f}")

    return best_model, best_auc


if __name__ == "__main__":
    train()
