# Heart Disease Prediction System (MLOps)

![CI Pipeline](https://github.com/Lokesh-100/mlopsAssignment1/actions/workflows/ci.yml/badge.svg)
![CD Pipeline](https://github.com/Lokesh-100/mlopsAssignment1/actions/workflows/cd.yaml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)

## 📋 Overview
This project is a complete **End-to-End MLOps implementation** for predicting heart disease. It demonstrates a production-ready machine learning pipeline that includes experiment tracking, model versioning, continuous integration/deployment (CI/CD), and real-time monitoring on a Kubernetes cluster.

The system uses a **Random Forest Classifier** selected for its high ROC-AUC score and recall, ensuring reliable predictions for healthcare use cases.

## 🚀 Live Demo & Resources
- **Live Application (Swagger UI)**: [Heart Disease Prediction API](https://heart-disease-prediction.c-158d220.kyma.ondemand.com/docs)
- **Monitoring Dashboard (Grafana)**: [Grafana Dashboard](https://grafana-mlops.c-158d220.kyma.ondemand.com/dashboards)  
  *(Credentials: `admin` / `admin`)*
- **Docker Image**: [aditya3298/mlops](https://hub.docker.com/r/aditya3298/mlops/tags)
- **Model Artifacts**: [Hugging Face Hub](https://huggingface.co/adityabhuvangiri/heart_models/tree/main)

---

## 🏗️ Architecture
The system follows a modular "Everything as Code" architecture, integrating development, operations, and monitoring.

*(Note: Insert the architecture diagram image here in your repository and reference it)*

### Components
| Component | Technology | Description |
| :--- | :--- | :--- |
| **Source Control** | Git & GitHub | Hosts code, config, and dataset versioning. |
| **Experiment Tracking** | MLflow | Tracks params, metrics, and EDA artifacts. |
| **Model Registry** | Hugging Face | Stores versioned model binaries (tagged by CI run). |
| **API Serving** | FastAPI | Exposes the model via high-performance REST endpoints. |
| **Containerization** | Docker | Packages the app for consistent deployment. |
| **Orchestration** | Kubernetes | Manages the application scaling and availability. |
| **CI/CD** | GitHub Actions | Automates testing, training, and deployment. |
| **Monitoring** | Grafana & Loki | Real-time log aggregation and system visualization. |

---

## 📂 Repository Structure
```bash
├── .github/workflows  # CI/CD pipeline definitions
├── app/               # FastAPI application code
├── deployment/        # Kubernetes manifests (deployment.yml, service.yml)
├── heart_models/      # Local model storage (synced to Hugging Face)
├── notebooks/         # EDA and experimentation notebooks
├── src/               # Source code for data processing & training
├── tests/             # Unit and integration tests
├── Dockerfile         # Container definition
├── requirements.txt   # Python dependencies
└── README.md          # Project documentation
