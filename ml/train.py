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
