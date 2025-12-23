import mlflow
import mlflow.sklearn
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from config import PROCESSED_DATA_PATH, TARGET_COL, MODEL_PATH
from preprocessing import get_preprocessor

df = pd.read_csv(PROCESSED_DATA_PATH)
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

preprocessor = get_preprocessor(X)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100)
}

mlflow.set_experiment("Heart Disease Prediction")

best_model = None
best_auc = 0

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        pipeline = Pipeline([
            ("prep", preprocessor),
            ("model", model)
        ])
        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_test)
        probas = pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probas)

        mlflow.log_param("model", name)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", auc)
        mlflow.sklearn.log_model(pipeline, "model")

        if auc > best_auc:
            best_auc = auc
            best_model = pipeline

joblib.dump(best_model, MODEL_PATH)
print("Best model saved.")
