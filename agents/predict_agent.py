"""
agents/predict_agent.py
Agent 2 — XGBoost Cardiovascular Risk Predictor
Predicts probability of heart attack / stroke from patient vitals.
"""

import numpy as np
import joblib
import os

FEATURES = [
    "age", "sex", "chest_pain_type", "resting_bp", "cholesterol",
    "fasting_blood_sugar", "resting_ecg", "max_heart_rate",
    "exercise_angina", "st_depression", "st_slope", "num_vessels", "thalassemia"
]

MODEL_PATH  = "models/cardio_xgb.pkl"
SCALER_PATH = "models/scaler.pkl"


class PredictAgent:
    def __init__(self):
        self.model  = None
        self.scaler = None
        self._load()

    def _load(self):
        if os.path.exists(MODEL_PATH):
            self.model  = joblib.load(MODEL_PATH)
        if os.path.exists(SCALER_PATH):
            self.scaler = joblib.load(SCALER_PATH)

    def train(self, X, y) -> None:
        from xgboost import XGBClassifier
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        X_scaled    = self.scaler.fit_transform(X)
        self.model  = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.08,
            use_label_encoder=False, eval_metric="logloss", random_state=42,
        )
        self.model.fit(X_scaled, y)
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.model,  MODEL_PATH)
        joblib.dump(self.scaler, SCALER_PATH)

    def predict(self, vitals: dict) -> dict:
        if self.model is None:
            self._auto_train()

        x        = np.array([[vitals.get(f, 0) for f in FEATURES]])
        x_scaled = self.scaler.transform(x)
        prob     = float(self.model.predict_proba(x_scaled)[0][1])
        pred     = int(self.model.predict(x_scaled)[0])

        if prob >= 0.75:
            risk_level = "Critical"
        elif prob >= 0.50:
            risk_level = "High"
        elif prob >= 0.30:
            risk_level = "Moderate"
        else:
            risk_level = "Low"

        return {
            "prediction":  pred,
            "probability": prob,
            "risk_level":  risk_level,
            "label":       "Heart Disease Detected" if pred == 1 else "No Heart Disease",
        }

    def get_report(self) -> str:
        return f"XGBoost model: {MODEL_PATH}" if self.model else "No model loaded."

    def _auto_train(self):
        import sys
        sys.path.insert(0, ".")
        from data.generate_data import generate_dataset
        df = generate_dataset(900)
        X  = df[FEATURES].values
        y  = df["target"].values
        self.train(X, y)
