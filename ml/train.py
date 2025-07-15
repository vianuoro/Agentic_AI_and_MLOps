import pandas as pd
import numpy as np
import mlflow
import mlflow.tensorflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from agents.data_agent import DataAgent
from agents.training_agent import TrainingAgent

def train():
    agent = DataAgent()
    raw_df = agent.fetch_data()
    df = agent.clean_data(raw_df)

    X = df.drop("price", axis=1).values
    y = df["price"].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    trainer = TrainingAgent()

    mlflow.set_experiment("house-price-agent-tf")

    with mlflow.start_run():
        model, history = trainer.train_model(X_train, y_train, X_val, y_val, epochs=50)

        val_mae = history.history['val_mae'][-1]
        val_loss = history.history['val_loss'][-1]

        mlflow.log_param("epochs", 50)
        mlflow.log_metric("val_mae", val_mae)
        mlflow.log_metric("val_loss", val_loss)

        # Log scaler separately or save for inference
        # For simplicity, save model including preprocessing in real scenario

        mlflow.tensorflow.log_model(model, artifact_path="model")

        print(f"Model trained and logged. Val MAE: {val_mae:.4f}")

if __name__ == "__main__":
    train()
