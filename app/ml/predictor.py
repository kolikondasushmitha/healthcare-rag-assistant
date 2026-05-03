import pickle
import numpy as np
from pathlib import Path

MODEL_PATH = Path("models/diabetes_model.pkl")

FEATURE_NAMES = [
    "Pregnancies", "Glucose", "BloodPressure",
    "SkinThickness", "Insulin", "BMI",
    "DiabetesPedigreeFunction", "Age"
]

RISK_THRESHOLDS = {
    "low":    (0.0,  0.35),
    "medium": (0.35, 0.60),
    "high":   (0.60, 1.01),
}

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "❌ Model not found! Run: python scripts/train_model.py first"
        )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

pipeline = load_model()

def predict(features: dict) -> dict:
    values = np.array([[features[f] for f in FEATURE_NAMES]])

    prediction  = pipeline.predict(values)[0]
    probability = pipeline.predict_proba(values)[0][1]

    risk_level = next(
        label for label, (lo, hi) in RISK_THRESHOLDS.items()
        if lo <= probability < hi
    )

    importances = pipeline.named_steps["model"].feature_importances_
    top_factors = sorted(
        zip(FEATURE_NAMES, importances),
        key=lambda x: x[1],
        reverse=True
    )[:3]

    return {
        "prediction":  int(prediction),
        "probability": round(float(probability), 3),
        "risk_level":  risk_level,
        "top_factors": [f for f, _ in top_factors],
        "label":       "Diabetic" if prediction == 1 else "Non-Diabetic"
    }