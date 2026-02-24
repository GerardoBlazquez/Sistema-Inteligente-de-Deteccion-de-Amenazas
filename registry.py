import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FRAUD_FEATURES = [
    'Time',
    'V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
    'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
    'V21','V22','V23','V24','V25','V26','V27','V28',
    'Amount'
]

BOT_FEATURES = [
    "requests_per_ip",
    "avg_time_diff",
    "error_rate",
    "unique_resources"
]

MODEL_REGISTRY = {
    "fraud": {
        "random_forest": {
            "model": joblib.load(
                os.path.join(BASE_DIR, "models", "fraud", "fraud_random_forest.pkl")
            ),
            "scaler": joblib.load(
                os.path.join(BASE_DIR, "models", "fraud", "fraud_scaler.pkl")
            ),
            "features": FRAUD_FEATURES,
            "threshold_f1": 0.37,
            "threshold_cost": 0.15
        }
    },
    "bot": {
        "xgboost": {
            "model": joblib.load(
                os.path.join(BASE_DIR, "models", "bots", "xgboost.pkl")
            ),
            "features": BOT_FEATURES,
            "threshold_f1": 0.41,
            "threshold_cost": 0.08
        }
    }
}