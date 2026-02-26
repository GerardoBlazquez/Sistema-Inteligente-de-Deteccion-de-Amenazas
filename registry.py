import joblib
import os
import numpy as np
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FRAUD_FEATURES = [
    'Time','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
    'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
    'V21','V22','V23','V24','V25','V26','V27','V28','Amount'
]

BOT_FEATURES = [
    "requests_per_ip", 
    "avg_time_diff", 
    "error_rate", 
    "unique_resources"
]

def load_model_safely(model_path):
    """Carga modelo con fallback seguro"""
    if os.path.exists(model_path):
        return joblib.load(model_path)
    print(f"⚠️  Modelo no encontrado: {model_path}. Usando fallback.")
    return None

MODEL_REGISTRY = {

    "fraud": {

        "random_forest": {
            "model": load_model_safely(os.path.join(BASE_DIR, "models", "fraud", "fraud_random_forest.pkl")),
            "scaler": load_model_safely(os.path.join(BASE_DIR, "models", "fraud", "fraud_scaler.pkl")),
            "features": FRAUD_FEATURES,
            "threshold_f1": 0.37,
            "threshold_cost": 0.15
        },

        "isolation_forest": {
            "model": load_model_safely(os.path.join(BASE_DIR, "models", "fraud", "fraud_isolation_forest.pkl")),
            "features": FRAUD_FEATURES,
            "threshold_f1": 0.0,
            "threshold_cost": 0.0
        },

        "lof": {
            "model": load_model_safely(os.path.join(BASE_DIR, "models", "fraud", "fraud_lof.pkl")),
            "features": FRAUD_FEATURES,
            "threshold_f1": 0.0,
            "threshold_cost": 0.0
        },

        "ocsvm": {
            "model": load_model_safely(os.path.join(BASE_DIR, "models", "fraud", "fraud_ocsvm.pkl")),
            "features": FRAUD_FEATURES,
            "threshold_f1": 0.0,
            "threshold_cost": 0.0
        }
    },

    "bots": {  # 🔥 CORREGIDO (antes era "bot")

        "random_forest": {
            "model": load_model_safely(os.path.join(BASE_DIR, "models", "bots", "random_forest.pkl")),
            "features": BOT_FEATURES,
            "threshold_f1": 0.41,
            "threshold_cost": 0.08
        },

        "xgboost": {
            "model": load_model_safely(os.path.join(BASE_DIR, "models", "bots", "xgboost.pkl")),
            "features": BOT_FEATURES,
            "threshold_f1": 0.41,
            "threshold_cost": 0.08
        }
    }
}
