from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from registry import MODEL_REGISTRY
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response

app = FastAPI(title="Threat Detection API")

# -----------------------------
# CORS
# -----------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# CONFIGURACIÓN ECONÓMICA
# -----------------------------

COSTE_FRAUDE_REAL = 250.0
COSTE_REVISION = 5.0

# Tasa real del dataset creditcard.csv
BASE_FRAUD_RATE = 0.001727  

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
# REQUEST MODELS
# -----------------------------

class PredictionRequest(BaseModel):
    domain: str
    model_name: str
    mode: str
    data: dict

class BatchEconomicRequest(BaseModel):
    scores: list[float]
    true_labels: list[int]

# -----------------------------
# HEALTH
# -----------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_available": {
            domain: {
                name: bool(model.get("model"))
                for name, model in models.items()
            }
            for domain, models in MODEL_REGISTRY.items()
        }
    }

# -----------------------------
# LIST MODELS
# -----------------------------

@app.get("/models")
def list_models():
    response = {}
    for domain, models in MODEL_REGISTRY.items():
        response[domain] = {}
        for model_name, model_info in models.items():
            if model_info.get("model") is not None:
                response[domain][model_name] = {"status": "ready"}
    return response

# -----------------------------
# PREDICT
# -----------------------------

@app.post("/predict")
def predict(request: PredictionRequest):

    REQUEST_COUNT.labels(
        domain=request.domain,
        model=request.model_name
    ).inc()

    if request.domain not in MODEL_REGISTRY:
        raise HTTPException(status_code=400, detail="Invalid domain")

    if request.model_name not in MODEL_REGISTRY[request.domain]:
        raise HTTPException(status_code=400, detail="Invalid model")

    package = MODEL_REGISTRY[request.domain][request.model_name]
    model = package["model"]

    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")

    features = package["features"]

    try:
        X_array = np.array([[request.data.get(f, 0) for f in features]])
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Escalado (solo fraude)
    if "scaler" in package and package.get("scaler") is not None:
        scaler = package["scaler"]
        if "Time" in features and "Amount" in features:
            time_idx = features.index("Time")
            amount_idx = features.index("Amount")

            scaled = scaler.transform(
                [[X_array[0][time_idx], X_array[0][amount_idx]]]
            )
            X_array[0][time_idx] = scaled[0][0]
            X_array[0][amount_idx] = scaled[0][1]

    X = X_array
    raw_score = None

    with INFERENCE_TIME.time():

        # ---------------- SUPERVISADOS ----------------
        if hasattr(model, "predict_proba"):

            proba = model.predict_proba(X)

            if proba.shape[1] == 2:
                score = float(proba[0][1])
            else:
                score = float(proba[0][0])

            # Threshold económico automático
            if request.mode == "auto_cost":
                threshold = COSTE_REVISION / COSTE_FRAUDE_REAL
            else:
                threshold = (
                    package["threshold_f1"]
                    if request.mode == "f1"
                    else package["threshold_cost"]
                )

            prediction = int(score >= threshold)

        # ---------------- NO SUPERVISADOS ----------------
        elif hasattr(model, "decision_function"):

            raw_score = float(model.decision_function(X)[0])
            normalized = 1 / (1 + np.exp(-raw_score))
            score = 1 - normalized

            threshold = 0.5
            prediction = int(score >= threshold)

        else:
            raise Exception("Model type not supported")

    if prediction == 1:
        if request.domain == "fraud":
            FRAUD_DETECTED.inc()
        elif request.domain == "bots":
            BOT_DETECTED.inc()

    # Feature importance
    feature_importance = None

    if hasattr(model, "feature_importances_"):
        feature_importance = model.feature_importances_.tolist()

    elif hasattr(model, "coef_"):
        coef = model.coef_
        feature_importance = np.abs(coef[0]).tolist()

    return {
        "score": float(score),
        "classification": int(prediction),
        "threshold_used": float(threshold),
        "domain": request.domain,
        "model": request.model_name,
        "mode": request.mode,
        "features": features,
        "feature_importance": feature_importance
    }

# -----------------------------
# ECONOMIC ANALYSIS (BATCH)
# -----------------------------

@app.post("/economic-analysis/batch")
def economic_batch_analysis(request: BatchEconomicRequest):

    if len(request.scores) != len(request.true_labels):
        raise HTTPException(status_code=400, detail="Length mismatch")

    scores = np.array(request.scores)
    labels = np.array(request.true_labels)

    thresholds = np.linspace(0, 1, 200)
    results = []

    for t in thresholds:

        predictions = (scores >= t).astype(int)

        tp = np.sum((predictions == 1) & (labels == 1))
        fp = np.sum((predictions == 1) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))
        tn = np.sum((predictions == 0) & (labels == 0))

        cost = fn * COSTE_FRAUDE_REAL + fp * COSTE_REVISION

        results.append({
            "threshold": float(t),
            "cost": float(cost),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn)
        })

    best = min(results, key=lambda x: x["cost"])

    baseline_cost = np.sum(labels == 1) * COSTE_FRAUDE_REAL

    return {
        "baseline_cost": float(baseline_cost),
        "best_threshold": best["threshold"],
        "best_cost": best["cost"],
        "roi_vs_baseline": float(baseline_cost - best["cost"]),
        "cost_curve": results
    }

# -----------------------------
# PROMETHEUS
# -----------------------------

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")

# -----------------------------
# RUN
# -----------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
