"""Module 3: REST API phuc vu pipeline Breast Cancer.
Chay: uvicorn app.main:app --reload
"""
import datetime
import json
import os

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, field_validator

MODEL_PATH = "models/bc_pipeline_v1.0.0.joblib"
MODEL_VERSION = "1.0.0"
N_FEATURES = 30
LOG_PATH = "logs/predictions.log"

app = FastAPI(title="Breast Cancer Classifier API")
model = joblib.load(MODEL_PATH)


class PredictRequest(BaseModel):
    features: list[float]

    @field_validator("features")
    @classmethod
    def check_len(cls, v):
        if len(v) != N_FEATURES:
            raise ValueError(f"Can dung {N_FEATURES} dac trung, nhan {len(v)}")
        return v


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return {
        "model_version": MODEL_VERSION,
        "model_type": type(model.named_steps["clf"]).__name__,
        "pipeline_steps": list(model.named_steps.keys()),
        "n_features": N_FEATURES,
    }


@app.post("/predict")
def predict(req: PredictRequest):
    x = np.array(req.features).reshape(1, -1)
    proba = model.predict_proba(x)[0]
    label = int(np.argmax(proba))

    # Module 4 - Buoc 4: log du doan (timestamp, features, prediction,
    # confidence, model_version)
    record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "features": req.features,
        "prediction": label,
        "confidence": float(proba[label]),
        "model_version": MODEL_VERSION,
    }
    os.makedirs("logs", exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")

    return {
        "label": label,
        "label_name": ["malignant", "benign"][label],
        "confidence": float(proba[label]),
    }
