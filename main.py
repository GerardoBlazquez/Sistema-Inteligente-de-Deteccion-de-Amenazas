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
# CONFIGURACIÓN ECONÓMICA
# -----------------------------

COSTE_FRAUDE_REAL = 250.0
COSTE_REVISION = 5.0

# Probabilidad base histórica (ajústala a tu dataset real)
BASE_FRAUD_RATE = 0.0017   # ejemplo 0.17%

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
# BATCH ECONOMIC REQUEST
# -----------------------------

class BatchEconomicRequest(BaseModel):
    scores: list[float]
    true_labels: list[int]  # 0 o 1

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

        response[domain] = {}

        for model_name, model_info in models.items():

            if model_info.get("model") is not None:
                response[domain][model_name] = {
                    "status": "ready"
                }

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
    raw_score = None
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

            if request.mode == "auto_cost":
                # Threshold Bayesiano correcto
                threshold = COSTE_REVISION / COSTE_FRAUDE_REAL

            else:
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

            raw_score = float(model.decision_function(X)[0])

            normalized = 1 / (1 + np.exp(-raw_score))

            # 🔥 Invertimos para que 1 = amenaza
            score = 1 - normalized

            threshold = 0.5
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
    # FEATURE IMPORTANCE
    # -------------------------

    feature_importance = None

    # Modelos tipo RandomForest, XGBoost
    if hasattr(model, "feature_importances_"):
        feature_importance = model.feature_importances_.tolist()

    # Modelos lineales (LogisticRegression, etc.)
    elif hasattr(model, "coef_"):
        coef = model.coef_

        if hasattr(coef, "__iter__"):
            feature_importance = np.abs(coef[0]).tolist()
        else:
            feature_importance = [abs(float(coef))]

    # -------------------------
    # SIMULACIÓN ECONÓMICA
    # -------------------------

    # Threshold económicamente óptimo (Bayesiano)
    optimal_threshold = COSTE_REVISION / COSTE_FRAUDE_REAL

    # Esperanza SIN modelo (baseline poblacional coherente)
    expected_loss_without_model = BASE_FRAUD_RATE * COSTE_FRAUDE_REAL

    # Esperanza CON modelo (teoría de decisión)
    # Si bloqueamos → pagamos revisión
    # Si no bloqueamos → asumimos riesgo ponderado por probabilidad
    expected_loss_with_model = (
        score * COSTE_FRAUDE_REAL * (1 - prediction)
        + prediction * COSTE_REVISION
    )

    # ROI
    roi = expected_loss_without_model - expected_loss_with_model

    roi_percentage = (
        (roi / expected_loss_without_model) * 100
        if expected_loss_without_model > 0
        else 0
    )

    # -------------------------
    # Respuesta
    # -------------------------
    return {
        "score": float(score),
        "raw_score": float(raw_score) if raw_score is not None else None,
        "classification": int(prediction),
        "threshold_used": float(threshold),
        "domain": request.domain,
        "model": request.model_name,
        "mode": request.mode,
        "model_status": "ready",
        "features": features,
        "feature_importance": feature_importance,
        "economic_analysis": {
            "expected_loss_without_model": float(expected_loss_without_model),
            "expected_loss_with_model": float(expected_loss_with_model),
            "roi_absolute": float(roi),
            "roi_percentage": float(roi_percentage)
        }
    }


# -----------------------------
# ECONOMIC BATCH ANALYSIS
# -----------------------------

@app.post("/economic-analysis/batch")
def economic_batch_analysis(request: BatchEconomicRequest):

    if len(request.scores) != len(request.true_labels):
        raise HTTPException(status_code=400, detail="Scores and labels length mismatch")

    scores = np.array(request.scores)
    labels = np.array(request.true_labels)

    thresholds = np.linspace(0, 1, 100)

    results = []

    for t in thresholds:

        predictions = (scores >= t).astype(int)

        tp = np.sum((predictions == 1) & (labels == 1))
        fp = np.sum((predictions == 1) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))
        tn = np.sum((predictions == 0) & (labels == 0))

        cost = (
            fn * COSTE_FRAUDE_REAL +
            fp * COSTE_REVISION
        )

        results.append({
            "threshold": float(t),
            "cost": float(cost),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn)
        })

    # Mejor threshold económico
    best = min(results, key=lambda x: x["cost"])

    # Baseline sin modelo (no bloquear nada)
    baseline_cost = np.sum(labels == 1) * COSTE_FRAUDE_REAL

    profit_curve = [
        baseline_cost - r["cost"]
        for r in results
    ]

    return {
        "baseline_cost": float(baseline_cost),
        "best_threshold": best["threshold"],
        "best_cost": best["cost"],
        "roi_vs_baseline": float(baseline_cost - best["cost"]),
        "cost_curve": results,
        "profit_curve": profit_curve
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
