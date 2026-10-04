from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import pandas as pd

mlflow.set_tracking_uri("sqlite:///mlflow.db")

MODEL_NAME = "fraud-detector"
MODEL_STAGE = "latest"  # on affinera avec des alias plus tard

model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")

app = FastAPI(title="Fraud Detection API")

class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}

@app.post("/predict")
def predict(transaction: Transaction):
    df = pd.DataFrame([transaction.model_dump()])
    proba = model.predict_proba(df)[0, 1]
    return {
        "fraud_probability": float(proba),
        "is_fraud": bool(proba >= 0.5),
    }