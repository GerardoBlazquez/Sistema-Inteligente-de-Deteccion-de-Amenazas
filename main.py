from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from registry import MODEL_REGISTRY

from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response

app = FastAPI(title="Threat Detection API")

REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total prediction requests",
    ["domain", "model"]
)

FRAUD_DETECTED = Counter(
    "fraud_detected_total",
    "Total fraud detections"
)

BOT_DETECTED = Counter(
    "bot_detected_total",
    "Total bot detections"
)

INFERENCE_TIME = Histogram(
    "inference_time_seconds",
    "Time spent on inference"
)


class PredictionRequest(BaseModel):
    domain: str
    model_name: str
    mode: str
    data: dict


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "Threat Detection API"
    }


@app.get("/models")
def list_models():
    response = {}

    for domain, models in MODEL_REGISTRY.items():
        response[domain] = list(models.keys())

    return response


@app.get("/models/{domain}")
def model_metadata(domain: str):

    if domain not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail="Domain not found")

    return {
        model_name: {
            "threshold_f1": model["threshold_f1"],
            "threshold_cost": model["threshold_cost"],
            "features": model["features"]
        }
        for model_name, model in MODEL_REGISTRY[domain].items()
    }

@app.post("/predict")
def predict(request: PredictionRequest):

    # Validación dominio
    if request.domain not in MODEL_REGISTRY:
        raise HTTPException(status_code=400, detail="Invalid domain")

    # Validación modelo
    if request.model_name not in MODEL_REGISTRY[request.domain]:
        raise HTTPException(status_code=400, detail="Invalid model")

    package = MODEL_REGISTRY[request.domain][request.model_name]
    model = package["model"]
    features = package["features"]

    # Construcción ordenada del vector
    try:
        X_array = np.array([[request.data[f] for f in features]])
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing feature: {str(e)}")

    # Escalado solo para el fraude (Time y Amount)
    if "scaler" in package:
        scaler = package["scaler"]

        time_idx = features.index("Time")
        amount_idx = features.index("Amount")

        scaled_values = scaler.transform(
            [[X_array[0][time_idx], X_array[0][amount_idx]]]
        )

        X_array[0][time_idx] = scaled_values[0][0]
        X_array[0][amount_idx] = scaled_values[0][1]

    X = X_array

    # Modelos supervisados
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(X)[0][1]

        if request.mode == "f1":
            threshold = package["threshold_f1"]
        elif request.mode == "cost":
            threshold = package["threshold_cost"]
        else:
            raise HTTPException(status_code=400, detail="Invalid mode")

        prediction = int(score >= threshold)

    # Modelos no supervisados
    else:
        score = -model.decision_function(X)[0]
        threshold = 0
        prediction = int(score >= threshold)

    return {
        "score": float(score),
        "classification": prediction,
        "threshold_used": threshold,
        "domain": request.domain,
        "model": request.model_name,
        "mode": request.mode
    }