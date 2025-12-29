import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import os

from src.config import PROCESSED_DATA_PATH, ARTIFACT_DIR

# ARTIFACT_DIR = "eda_artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def perform_eda_and_log():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    target_col = df.columns[-1]

    with mlflow.start_run(run_name="EDA_Analysis"):

        # ----------------------------
        # Log basic dataset info
        # ----------------------------
        mlflow.log_param("num_rows", df.shape[0])
        mlflow.log_param("num_columns", df.shape[1])

        # ----------------------------
        # Class Balance Plot
        # ----------------------------
        plt.figure(figsize=(6, 4))
        sns.countplot(x=target_col, data=df)
        plt.title("Class Distribution")
        class_plot_path = f"{ARTIFACT_DIR}/class_balance.png"
        plt.savefig(class_plot_path)
        plt.close()

        mlflow.log_artifact(class_plot_path)

        # ----------------------------
        # Feature Histograms
        # ----------------------------
        df.hist(figsize=(15, 10), bins=20)
        plt.suptitle("Feature Distributions")
        hist_path = f"{ARTIFACT_DIR}/feature_histograms.png"
        plt.savefig(hist_path)
        plt.close()

        mlflow.log_artifact(hist_path)

        # ----------------------------
        # Correlation Heatmap
        # ----------------------------
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.corr(), cmap="coolwarm", linewidths=0.5)
        plt.title("Correlation Heatmap")
        heatmap_path = f"{ARTIFACT_DIR}/correlation_heatmap.png"
        plt.savefig(heatmap_path)
        plt.close()

        mlflow.log_artifact(heatmap_path)

        # ----------------------------
        # Statistical Summary
        # ----------------------------
        summary_path = f"{ARTIFACT_DIR}/summary_stats.csv"
        df.describe().to_csv(summary_path)
        mlflow.log_artifact(summary_path)

        print("✅ EDA logged successfully to experiment tracking")

if __name__ == "__main__":
    perform_eda_and_log()