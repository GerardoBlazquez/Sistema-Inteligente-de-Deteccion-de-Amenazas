from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from registry import MODEL_REGISTRY
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response

app = FastAPI(title="Threat Detection API")

# ✅ CORS COMPLETO - ARREGLADO
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4321", 
        "http://127.0.0.1:4321",
        "http://localhost:3000",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # ← OPTIONS crítico
    allow_headers=["*"],
)

# Métricas Prometheus
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
        "service": "Threat Detection API",
        "models_available": {
            domain: {name: bool(model.get("model")) for name, model in models.items()}
            for domain, models in MODEL_REGISTRY.items()
        }
    }

@app.get("/models")
def list_models():
    response = {}
    for domain, models in MODEL_REGISTRY.items():
        available_models = {name: bool(model.get("model")) for name, model in models.items()}
        response[domain] = {name: status for name, status in available_models.items() if status}
    return response

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

@app.post("/predict")
def predict(request: PredictionRequest):
    REQUEST_COUNT.labels(domain=request.domain, model=request.model_name).inc()

    # Validación dominio
    if request.domain not in MODEL_REGISTRY:
        raise HTTPException(status_code=400, detail="Invalid domain")

    # Validación modelo
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

    # Construcción vector ordenado
    try:
        X_array = np.array([[request.data.get(f, 0) for f in features]])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid data: {str(e)}")

    # Escalado para fraude (Time y Amount)
    if "scaler" in package and package["scaler"] is not None:
        try:
            scaler = package["scaler"]
            time_idx = features.index("Time")
            amount_idx = features.index("Amount")
            scaled_values = scaler.transform([[X_array[0][time_idx], X_array[0][amount_idx]]])
            X_array[0][time_idx] = scaled_values[0][0]
            X_array[0][amount_idx] = scaled_values[0][1]
        except Exception as e:
            print(f"Scaler error: {e}")  # Log pero continúa

    X = X_array

    with INFERENCE_TIME.time():
        if hasattr(model, "predict_proba"):
            # Modelos supervisados
            score = model.predict_proba(X)[0][1]
            threshold = package["threshold_f1"] if request.mode == "f1" else package["threshold_cost"]
            prediction = int(score >= threshold)
        else:
            # Modelos no supervisados
            score = -model.decision_function(X)[0]
            threshold = 0
            prediction = int(score >= threshold)

    # Métricas
    if prediction == 1:
        if request.domain == "fraud":
            FRAUD_DETECTED.inc()
        elif request.domain == "bot":
            BOT_DETECTED.inc()

    return {
        "score": float(score),
        "classification": prediction,
        "threshold_used": float(threshold),
        "domain": request.domain,
        "model": request.model_name,
        "mode": request.mode,
        "model_status": "ready"
    }

@app.get("/metrics")
def metrics():
    return generate_latest()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

