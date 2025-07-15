# ✅ 1. agents/data_agent.py
import pandas as pd
import os

class DataAgent:
    def fetch_data(self, path="data/raw/housing.csv"):
        # Simulated local fetch (replace with API or scraper as needed)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data not found at {path}")
        return pd.read_csv(path)

    def clean_data(self, df):
        df = df.dropna()
        df = df[df["price"] > 0]
        df["age"] = 2025 - df["year_built"]
        df = pd.get_dummies(df, columns=["property_type", "location"], drop_first=True)
        return df


# ✅ 2. ml/train.py with MLflow logging
import pandas as pd
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from agents.data_agent import DataAgent

def train():
    agent = DataAgent()
    raw_df = agent.fetch_data()
    df = agent.clean_data(raw_df)

    X = df.drop("price", axis=1)
    y = df["price"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = lgb.LGBMRegressor()

    mlflow.set_experiment("house-price-agent")
    with mlflow.start_run():
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        rmse = mean_squared_error(y_val, preds, squared=False)

        mlflow.log_param("model_type", "lightgbm")
        mlflow.log_metric("rmse", rmse)
        mlflow.lightgbm.log_model(model, artifact_path="model")

        print(f"Model logged with RMSE: {rmse}")

if __name__ == "__main__":
    train()


# ✅ 3. api/app.py
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import os

MODEL_PATH = "ml/model.pkl"

app = FastAPI()

class HouseInput(BaseModel):
    area_m2: float
    num_rooms: int
    year_built: int
    property_type: str
    location: str

@app.on_event("startup")
def load_model():
    global model
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

@app.post("/predict")
def predict_price(input: HouseInput):
    df = pd.DataFrame([input.dict()])
    df["age"] = 2025 - df["year_built"]
    # Dummy encoding should match training (in real use, load encoder or one-hot manually)
    # For now, fake all zero columns
    df.drop(["year_built"], axis=1, inplace=True)
    return {"predicted_price": float(model.predict(df)[0])}


# ✅ 4. .github/workflows/ci-cd.yml
name: CI/CD

on:
  push:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout repo
      uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt

    - name: Train model
      run: python ml/train.py


# ✅ 5. monitoring/drift_check.py
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset
import pandas as pd

def check_drift(current_data_path, reference_data_path):
    current_data = pd.read_csv(current_data_path)
    reference_data = pd.read_csv(reference_data_path)

    report = Report(metrics=[DataDriftPreset(), RegressionPreset()])
    report.run(current_data=current_data, reference_data=reference_data)
    report.save_html("monitoring/drift_report.html")

    print("✅ Drift report generated at monitoring/drift_report.html")

if __name__ == "__main__":
    check_drift("data/processed/latest.csv", "data/processed/reference.csv")
