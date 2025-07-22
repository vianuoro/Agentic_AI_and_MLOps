from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import joblib
import numpy as np
import pandas as pd
import os

app = FastAPI()

MODEL_PATH = "ml/model_tf"
SCALER_PATH = "ml/scaler.joblib"

class HouseInput(BaseModel):
    area_m2: float
    num_rooms: int
    year_built: int
    property_type: str
    location: str

@app.on_event("startup")
def load_resources():
    global model, scaler
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

@app.post("/predict")
def predict_price(input: HouseInput):
    data = input.dict()
    df = pd.DataFrame([data])
    df["age"] = 2025 - df["year_built"]
    df = df.drop(columns=["year_built"])

    # Dummy encoding placeholders (should match training)
    for col in ["property_type_Semi-detached", "location_Suburb"]:  # example dummy cols
        if col not in df.columns:
            df[col] = 0

    df = pd.get_dummies(df)
    df = df.reindex(columns=scaler.get_feature_names_out(), fill_value=0)

    X = scaler.transform(df.values.astype(np.float32))
    prediction = model.predict(X)
    return {"predicted_price": float(prediction[0][0])}