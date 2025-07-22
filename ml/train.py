import pandas as pd
import numpy as np
import mlflow
import mlflow.tensorflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

from mlflow.models import infer_signature
from hp_agents.data_agent import DataAgent
from hp_agents.training_agent import TrainingAgent
import matplotlib.pyplot as plt

def train():
    agent = DataAgent()
    raw_df = agent.fetch_data()
    df = agent.clean_data(raw_df)

    X = df.drop("price", axis=1).values
    y = df["price"].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    trainer = TrainingAgent()

    mlflow.set_experiment("house-price-agent-tf")
    with mlflow.start_run() as run:
        model, history = trainer.train_model(X_train, y_train, X_val, y_val, epochs=50)

        val_mae = history.history['val_mae'][-1]
        val_loss = history.history['val_loss'][-1]

        mlflow.log_param("epochs", 50)
        mlflow.log_metric("val_mae", val_mae)
        mlflow.log_metric("val_loss", val_loss)

        model_path = "ml/model_tf.keras"
        model.save(model_path)

        mlflow.tensorflow.log_model(model, artifact_path="model")
        mlflow.log_artifact(model_path)

        scaler_path = "ml/scaler.joblib"
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path)

        # 🔍 Evaluate model with MLflow
        y_pred = model.predict(X_val).flatten()
        signature = infer_signature(X_val, y_pred)

        eval_results = mlflow.evaluate(
            model=model,
            data=X_val,
            targets=y_val,
            model_type="regressor",
            evaluators=["default"],
            predictions=y_pred
        )

        print(f"📊 Evaluation results:\n{eval_results.metrics}")

        # 📊 Visualizations

        # --- 1. Prediction vs Actual plot ---
        plt.figure(figsize=(8, 6))
        plt.scatter(y_val, y_pred, alpha=0.5)
        plt.xlabel("Actual Price")
        plt.ylabel("Predicted Price")
        plt.title("Actual vs Predicted House Prices")
        plot_path1 = "ml/actual_vs_pred.png"
        plt.savefig(plot_path1)
        mlflow.log_artifact(plot_path1)
        plt.close()

        # --- 2. Residuals Histogram ---
        residuals = y_val - y_pred
        plt.figure(figsize=(8, 6))
        plt.hist(residuals, bins=30, edgecolor='black')
        plt.title("Residuals Distribution")
        plt.xlabel("Residual")
        plot_path2 = "ml/residuals_hist.png"
        plt.savefig(plot_path2)
        mlflow.log_artifact(plot_path2)
        plt.close()

        # --- 3. Markdown Report ---
        report_md = f"""
        # 🏠 House Price Prediction Report

        **Val MAE:** {val_mae:,.2f}  
        **Val Loss:** {val_loss:,.2f}

        ## 📈 Visualizations
        ![Actual vs Predicted](actual_vs_pred.png)  
        ![Residuals Histogram](residuals_hist.png)
        """
        report_path = "ml/report.md"
        with open(report_path, "w") as f:
            f.write(report_md)
        mlflow.log_artifact(report_path)

        print(f"Model trained and logged. Val MAE: {val_mae:.4f}")

if __name__ == "__main__":
    train()