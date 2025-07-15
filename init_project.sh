#!/bin/bash

PROJECT_NAME="house_price_agent"

echo "Creating project structure for: $PROJECT_NAME"

# Folder structure
mkdir -p $PROJECT_NAME/{agents,api,ml,data/{raw,processed},dags,monitoring,notebooks,.github/workflows,mlruns}

# Create __init__.py files
touch $PROJECT_NAME/{agents,api,ml}/__init__.py

# Agents
cat > $PROJECT_NAME/agents/data_agent.py <<EOF
class DataAgent:
    def fetch_data(self):
        pass

    def clean_data(self, raw_df):
        pass
EOF

cat > $PROJECT_NAME/agents/training_agent.py <<EOF
class TrainingAgent:
    def train_model(self, data):
        pass
EOF

cat > $PROJECT_NAME/agents/eval_agent.py <<EOF
class EvalAgent:
    def evaluate(self, model, X_val, y_val):
        pass
EOF

cat > $PROJECT_NAME/agents/deployment_agent.py <<EOF
class DeploymentAgent:
    def deploy_model(self, model_path):
        pass
EOF

cat > $PROJECT_NAME/agents/monitoring_agent.py <<EOF
class MonitoringAgent:
    def check_drift(self, new_data):
        pass
EOF

# API
cat > $PROJECT_NAME/api/app.py <<EOF
from fastapi import FastAPI
from ml.predict import predict_price

app = FastAPI()

@app.post("/predict")
def predict(input_data: dict):
    return predict_price(input_data)
EOF

# ML logic
cat > $PROJECT_NAME/ml/train.py <<EOF
def train(X_train, y_train):
    pass
EOF

cat > $PROJECT_NAME/ml/evaluate.py <<EOF
def evaluate(model, X_val, y_val):
    pass
EOF

cat > $PROJECT_NAME/ml/predict.py <<EOF
def predict_price(input_data):
    pass
EOF

cat > $PROJECT_NAME/ml/preprocess.py <<EOF
def preprocess(df):
    pass
EOF

# DAG
cat > $PROJECT_NAME/dags/data_pipeline.py <<EOF
# Placeholder for Airflow or Prefect DAG
EOF

# Monitoring
cat > $PROJECT_NAME/monitoring/drift_check.py <<EOF
def check_drift(current_data, reference_data):
    pass
EOF

cat > $PROJECT_NAME/monitoring/metrics_logger.py <<EOF
def log_metrics(metrics):
    pass
EOF

# GitHub CI/CD
cat > $PROJECT_NAME/.github/workflows/ci-cd.yml <<EOF
name: CI/CD

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: echo "No tests yet"
EOF

# Dockerfile
cat > $PROJECT_NAME/Dockerfile <<EOF
FROM python:3.10

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

# requirements.txt
cat > $PROJECT_NAME/requirements.txt <<EOF
pandas
numpy
lightgbm
xgboost
scikit-learn
mlflow
fastapi
uvicorn
dvc
evidently
prefect
langchain
openai
crewai
EOF

# config.yaml
cat > $PROJECT_NAME/config.yaml <<EOF
data:
  raw_path: "data/raw"
  processed_path: "data/processed"

model:
  type: "lightgbm"
  target: "price"
  features: ["area_m2", "num_rooms", "location", "year_built"]

monitoring:
  drift_threshold: 0.15
EOF

# README
cat > $PROJECT_NAME/README.md <<EOF
# House Price Agent 🚀

An agentic AutoML system for predicting house prices in Sweden, with MLOps features like CI/CD, drift detection, and model monitoring.

## Features
- Agent-based pipeline (data, training, eval, deployment, monitoring)
- FastAPI for model serving
- MLflow for experiment tracking
- DVC for data versioning
- GitHub Actions for CI/CD
- Docker-ready

## Getting Started
\`\`\`bash
pip install -r requirements.txt
uvicorn api.app:app --reload
\`\`\`
EOF

echo "✅ Project scaffold created at: $PROJECT_NAME/"
