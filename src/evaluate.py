import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from copy import deepcopy

from src.config import PROCESSED_DATA_PATH, ARTIFACT_DIR

# ----------------------------
# Configuration
# ----------------------------
os.makedirs(ARTIFACT_DIR, exist_ok=True)

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("heart-disease-experiment")


def perform_eda_and_log():
    df = pd.read_csv(PROCESSED_DATA_PATH)

    with mlflow.start_run(run_name="EDA_Analysis"):

        # ----------------------------
        # Log dataset metadata
        # ----------------------------
        mlflow.log_param("num_rows", df.shape[0])
        mlflow.log_param("num_columns", df.shape[1])

        # ----------------------------
        # Class Balance Plot
        # ----------------------------
        plt.figure(figsize=(6, 4))
        plot_df = deepcopy(df)
        plot_df.rename(columns={"num": "heart_disease_severity"}, inplace=True)
        sns.countplot(x="heart_disease_severity", data=plot_df)
        plt.title("Class Distribution")
        class_plot_path = os.path.join(ARTIFACT_DIR, "class_balance.png")
        plt.savefig(class_plot_path)
        plt.close()

        mlflow.log_artifact(class_plot_path, artifact_path="eda")

        # ----------------------------
        # Feature Histograms
        # ----------------------------
        df.hist(figsize=(15, 10), bins=20)
        plt.suptitle("Feature Distributions")

        hist_path = os.path.join(ARTIFACT_DIR, "feature_histograms.png")
        plt.savefig(hist_path)
        plt.close()

        mlflow.log_artifact(hist_path, artifact_path="eda")

        # ----------------------------
        # Correlation Heatmap
        # ----------------------------
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.corr(), cmap="coolwarm", linewidths=0.5)
        plt.title("Correlation Heatmap")

        heatmap_path = os.path.join(ARTIFACT_DIR, "correlation_heatmap.png")
        plt.savefig(heatmap_path)
        plt.close()

        mlflow.log_artifact(heatmap_path, artifact_path="eda")
        print("✅ EDA images stored locally AND logged to MLflow successfully")


if __name__ == "__main__":
    perform_eda_and_log()
