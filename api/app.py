from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import pandas as pd

MODEL_PATH = "mlruns/<run_id>/artifacts/model"  # Replace with your actual path or load dynamically

app = FastAPI()

class HouseInput(BaseModel):
    area_m2: float
    num_rooms: int
    year_built: int
    property_type: str
    location: str

def preprocess_input(data: dict):
    # Example preprocessing to match training features
    df = pd.DataFrame([data])
    df["age"] = 2025 - df["year_built"]
    df = df.drop(columns=["year_built"])

    # Dummy encoding for property_type and location should match training
    # For demo, just drop categorical or one-hot encode here manually
    # IMPORTANT: In practice, save and reuse your preprocessing pipeline/scaler
    df = pd.get_dummies(df, columns=["property_type", "location"], drop_first=True)

    # Align columns with training set here (add missing columns with zeros)

    return df.values.astype(np.float32)

@app.on_event("startup")
def load_model():
    global model
    model = tf.keras.models.load_model(MODEL_PATH)

@app.post("/predict")
def predict_price(input: HouseInput):
    x = preprocess_input(input.dict())
    prediction = model.predict(x)
    return {"predicted_price": float(prediction[0][0])}
