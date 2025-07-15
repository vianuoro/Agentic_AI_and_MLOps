
# 🏡 Agentic AutoML System for Predicting House Prices in Sweden

## 🧠 Objective
Build an **Agentic AI system** that autonomously ingests Swedish housing data, retrains ML models for price prediction, evaluates performance, and deploys updated models—powered by an **MLOps pipeline** for versioning, deployment, and monitoring.

---

## 🧩 Components Overview

| Layer         | Technology                    | Description                                               |
|---------------|-------------------------------|-----------------------------------------------------------|
| Agentic AI    | LangChain / CrewAI + OpenAI   | Orchestrate autonomous agents for data prep, modeling, evaluation, and deployment |
| ML Models     | XGBoost / LightGBM / sklearn  | Regression models for predicting house prices            |
| Data Pipeline | Prefect / Airflow             | Automate data ingestion and cleaning                      |
| Serving       | FastAPI + Docker              | REST API to serve predictions                             |
| CI/CD         | GitHub Actions + MLflow + DVC | Model tracking and automated deployment                   |
| Monitoring    | EvidentlyAI / Prometheus      | Drift detection, accuracy monitoring, logging             |

---

## 🏡 Data

You can use public datasets like:
- **[Booli](https://www.booli.se/)** via API (Swedish real estate platform)
- **SCB (Statistics Sweden)**: Housing statistics
- **Kaggle datasets**: E.g., “Sweden Real Estate” or scraped data

Typical features:
- `location` (latitude, longitude)
- `area_m2`
- `num_rooms`
- `year_built`
- `property_type`
- `asking_price`

---

## 🔁 Agentic System Architecture

### Agents

- 🔹 `DataAgent`: Fetches new house listings, cleans and transforms data.
- 🔹 `TrainingAgent`: Trains models (e.g., LightGBM, XGBoost) on the latest data.
- 🔹 `EvalAgent`: Compares model versions (R², MAE) and selects best.
- 🔹 `DeploymentAgent`: Deploys improved models via API.
- 🔹 `MonitoringAgent`: Detects drift or performance drops in production.

### Agent Execution Flow

1. `DataAgent` pulls data from API or S3.
2. `TrainingAgent` uses new data to train candidate models.
3. `EvalAgent` compares models (using cross-validation & MLflow).
4. If better, `DeploymentAgent` pushes the new model.
5. `MonitoringAgent` watches for prediction drift or outliers in prod.

---

## ⚙️ MLOps Pipeline

- **Experiment Tracking**: MLflow
- **Data Versioning**: DVC
- **Model Registry**: MLflow or S3
- **CI/CD**: GitHub Actions + Docker + FastAPI
- **Monitoring**: EvidentlyAI (for data drift + model metrics)

---

## 🧪 Hands-On Task Breakdown

| Task                                  | Tool                       | Outcome                        |
|---------------------------------------|----------------------------|--------------------------------|
| ✅ Collect and clean housing data     | Pandas + API + DataAgent   | Cleaned dataset                |
| ✅ Train baseline price predictor     | LightGBM / XGBoost         | Model v1                       |
| ✅ Evaluate model with MLflow         | MLflow                     | R², MAE, plots                 |
| ✅ Wrap model in FastAPI endpoint     | FastAPI + Docker           | `/predict` API                 |
| ✅ GitHub Actions for CI/CD           | GitHub                     | Auto-deploy on main branch     |
| ✅ Setup EvidentlyAI monitoring       | Evidently + Prometheus     | Dashboard & alerts             |
| ✅ Build agent framework              | CrewAI / LangChain         | Agents coordinate training & deployment |

---

## 🖥️ Example Folder Structure

```
house_price_agent/
├── agents/
│   ├── data_agent.py
│   ├── training_agent.py
│   ├── eval_agent.py
│   ├── deployment_agent.py
│   └── monitoring_agent.py
├── api/
│   └── app.py
├── ml/
│   ├── train.py
│   ├── evaluate.py
│   └── model.pkl
├── data/
│   ├── raw/
│   └── processed/
├── dags/
│   └── data_pipeline.py  # Prefect or Airflow DAG
├── Dockerfile
├── mlruns/  # MLflow tracking
├── .github/
│   └── workflows/
│       └── ci-cd.yml
└── README.md
```

---

## 🚀 Bonus Extensions

- Add LLM summarization of trends in regions ("Prices rising in Göteborg...").
- Use a `FeatureEngineeringAgent` to generate new features (e.g., price per m², age of home).
- Add map-based dashboards using Streamlit or Leaflet.js.
- Integrate with a Slack bot to alert about model degradation or drift.

---

## 🧠 Real-World Value

- Simulates an **automated real estate ML system** in production.
- Combines **autonomous decision-making** (via agents) with **robust DevOps**.
- Offers real-time adaptation to changing housing market dynamics.
