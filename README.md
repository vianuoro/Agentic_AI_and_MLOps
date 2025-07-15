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
```bash
pip install -r requirements.txt
uvicorn api.app:app --reload
```
