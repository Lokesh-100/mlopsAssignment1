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

## Manual Trigger & Usage Instructions

Follow the steps below to manually trigger the data pipeline, train the model, and verify the test cases.

#### 1) Prerequisites
Ensure you are in the root directory of the project, then install the local package and required dependencies:

```bash
pip install .
pip install -r requirements.txt
```
#### 2) Data Pipeline & Training
Data Loading and Preprocessing Load the data. This script saves output CSVs to the data/raw and data/processed directories.
```bash
python3 src/data_loader.py
```
#### 3) Exploratory Data Analysis (EDA)
Perform an evaluation of the data features.
```bash
python3 src/evaluate.py
```
#### 4) Model Training Train the Heart Disease detection model.
This script trains LogisticRegression and RandomForestClassifier models, selects the best one based on the ROC-AUC score, and saves it as a pickle file.
```bash
python3 src/train.py
```
#### 5) Model Inference
Run the main application to test the model's performance.
```bash
python3 app/main.py
```
#### Testing & Coverage
Verify Test Cases Install the testing frameworks and run the suite in verbose mode.
```bash
pip install pytest pytest-mock
python3 -m pytest -vv
```
#### Check Code Coverage
To verify the test coverage, install the coverage tool and run the analysis.
```bash
pip install coverage
coverage run -m pytest
coverage report -m
```
For direct usage without manual setup, please refer to the live links provided in the Live Demo and Usages section.

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
