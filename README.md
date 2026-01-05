# Heart Disease Prediction System (MLOps)

![CI Pipeline](https://github.com/Lokesh-100/mlopsAssignment1/actions/workflows/ci.yml/badge.svg)
![CD Pipeline](https://github.com/Lokesh-100/mlopsAssignment1/actions/workflows/cd.yaml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)

## 📋 Overview
This project is a complete **End-to-End MLOps implementation** for predicting heart disease. It demonstrates a production-ready machine learning pipeline that includes experiment tracking, model versioning, continuous integration/deployment (CI/CD), and real-time monitoring on a Kubernetes cluster.

The system uses a **Random Forest Classifier** selected for its high ROC-AUC score and recall, ensuring reliable predictions for healthcare use cases.

##  Live Demo & Resources
- **Live Application (Swagger UI)**: [Heart Disease Prediction API](https://heart-disease-prediction.c-158d220.kyma.ondemand.com/docs)
- **Monitoring Dashboard (Grafana)**: [Grafana Dashboard](https://grafana-mlops.c-158d220.kyma.ondemand.com/dashboards)  
  *(Credentials: `admin` / `admin`)*
- **Docker Image**: [aditya3298/mlops](https://hub.docker.com/r/aditya3298/mlops/tags)
- **Model Artifacts**: [Hugging Face Hub](https://huggingface.co/adityabhuvangiri/heart_models/tree/main)

---

##  Architecture

<img width="2816" height="1536" alt="Block Diagram" src="https://github.com/user-attachments/assets/ecb55142-1412-45fd-8acb-0195c361e572" />


### Components
| Component | Technology | 
| :--- | :--- |
| **Source Control** | Git & GitHub  | 
| **Experiment Tracking** | MLflow |   
| **Model Registry** | Hugging Face | 
| **API Serving** | FastAPI | 
| **Containerization** | Docker | 
| **Orchestration** | Kubernetes | 
| **CI/CD** | GitHub Actions | 
| **Monitoring** | Grafana & Loki | 
