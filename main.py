from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from registry import MODEL_REGISTRY
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response

app = FastAPI(title="Threat Detection API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4321",
        "http://127.0.0.1:4321",
        "http://localhost:3000",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# -----------------------------
# MÉTRICAS PROMETHEUS
# -----------------------------

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

# -----------------------------
# MODELO REQUEST
# -----------------------------

class PredictionRequest(BaseModel):
    domain: str
    model_name: str
    mode: str
    data: dict

# -----------------------------
# HEALTH CHECK
# -----------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "Threat Detection API",
        "models_available": {
            domain: {name: bool(model.get("model")) for name, model in models.items()}
            for domain, models in MODEL_REGISTRY.items()
        }
    }

# -----------------------------
# LISTA MODELOS
# -----------------------------

@app.get("/models")
def list_models():

    response = {}
    for domain, models in MODEL_REGISTRY.items():
        available_models = []

        for name, model_info in models.items():
            if model_info.get("model") is not None:
                available_models.append({
                    "name": name,
                    "threshold_f1": model_info.get("threshold_f1"),
                    "threshold_cost": model_info.get("threshold_cost"),
                    "status": "ready"
                })

        response[domain] = available_models

    return response
# -----------------------------
# METADATA MODELO
# -----------------------------

@app.get("/models/{domain}")
def model_metadata(domain: str):
    if domain not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail="Domain not found")

    response = {}

    for model_name, model_info in MODEL_REGISTRY[domain].items():
        if model_info.get("model") is None:
            continue

        response[model_name] = {
            "threshold_f1": model_info["threshold_f1"],
            "threshold_cost": model_info["threshold_cost"],
            "features": model_info["features"],
            "status": "ready"
        }

    return response

# -----------------------------
# PREDICCIÓN
# -----------------------------

@app.post("/predict")
def predict(request: PredictionRequest):

    REQUEST_COUNT.labels(
        domain=request.domain,
        model=request.model_name
    ).inc()

    # -------------------------
    # Validación dominio
    # -------------------------
    if request.domain not in MODEL_REGISTRY:
        raise HTTPException(status_code=400, detail="Invalid domain")

    # -------------------------
    # Validación modelo
    # -------------------------
    if request.model_name not in MODEL_REGISTRY[request.domain]:
        raise HTTPException(status_code=400, detail="Invalid model")

    package = MODEL_REGISTRY[request.domain][request.model_name]
    model = package["model"]

    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{request.model_name}' for domain '{request.domain}' not available."
        )

    features = package["features"]

    # -------------------------
    # Construcción vector ordenado
    # -------------------------
    try:
        X_array = np.array([[request.data.get(f, 0) for f in features]])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid data: {str(e)}")

    # -------------------------
    # Escalado (solo fraude)
    # -------------------------
    if "scaler" in package and package.get("scaler") is not None:
        try:
            scaler = package["scaler"]

            if "Time" in features and "Amount" in features:
                time_idx = features.index("Time")
                amount_idx = features.index("Amount")

                scaled_values = scaler.transform(
                    [[X_array[0][time_idx], X_array[0][amount_idx]]]
                )

                X_array[0][time_idx] = scaled_values[0][0]
                X_array[0][amount_idx] = scaled_values[0][1]

        except Exception as e:
            print(f"Scaler error: {e}")

    X = X_array

    # -------------------------
    # INFERENCIA + MÉTRICA TIEMPO
    # -------------------------
    with INFERENCE_TIME.time():

        # -------------------------
        # MODELOS SUPERVISADOS
        # -------------------------
        if hasattr(model, "predict_proba"):

            proba = model.predict_proba(X)

            if len(proba.shape) == 2:

                if proba.shape[1] == 2:
                    score = float(proba[0][1])

                elif proba.shape[1] == 1:
                    score = float(proba[0][0])

                else:
                    raise Exception(f"Invalid predict_proba shape: {proba.shape}")

            else:
                score = float(proba[0])

            threshold = (
                package["threshold_f1"]
                if request.mode == "f1"
                else package["threshold_cost"]
            )

            prediction = int(score >= threshold)

        # -------------------------
        # MODELOS NO SUPERVISADOS
        # -------------------------
        elif hasattr(model, "decision_function"):

            score = float(-model.decision_function(X)[0])
            threshold = 0.0
            prediction = int(score >= threshold)

        # -------------------------
        # FALLBACK
        # -------------------------
        elif hasattr(model, "predict"):

            prediction = int(model.predict(X)[0])
            score = float(prediction)

            threshold = (
                package.get("threshold_f1", 0.5)
                if request.mode == "f1"
                else package.get("threshold_cost", 0.5)
            )

        else:
            raise Exception("Model has no valid prediction method")

    # -------------------------
    # Métricas detección
    # -------------------------
    if prediction == 1:

        if request.domain == "fraud":
            FRAUD_DETECTED.inc()

        elif request.domain == "bots":
            BOT_DETECTED.inc()

    # -------------------------
    # Respuesta
    # -------------------------
    return {
        "score": float(score),
        "classification": int(prediction),
        "threshold_used": float(threshold),
        "domain": request.domain,
        "model": request.model_name,
        "mode": request.mode,
        "model_status": "ready"
    }

# -----------------------------
# PROMETHEUS
# -----------------------------

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")

# -----------------------------
# RUN LOCAL
# -----------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
