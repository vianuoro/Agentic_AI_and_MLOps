from fastapi import FastAPI
from ml.predict import predict_price

app = FastAPI()

@app.post("/predict")
def predict(input_data: dict):
    return predict_price(input_data)
