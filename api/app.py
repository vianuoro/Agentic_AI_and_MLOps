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