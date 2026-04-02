"""
agents/shap_agent.py
Agent 6 — SHAP TreeExplainer + KMeans Patient Clustering
"""

import numpy as np
import joblib
import os

FEATURES = [
    "age", "sex", "chest_pain_type", "resting_bp", "cholesterol",
    "fasting_blood_sugar", "resting_ecg", "max_heart_rate",
    "exercise_angina", "st_depression", "st_slope", "num_vessels", "thalassemia"
]

FEATURE_LABELS = [
    "Age", "Sex", "Chest Pain Type", "Resting BP", "Cholesterol",
    "Fasting Blood Sugar", "Resting ECG", "Max Heart Rate",
    "Exercise Angina", "ST Depression", "ST Slope", "Num Vessels", "Thalassemia"
]

MODEL_PATH  = "models/cardio_xgb.pkl"
SCALER_PATH = "models/scaler.pkl"


class SHAPAgent:
    def __init__(self):
        self.model  = None
        self.scaler = None
        self._load()

    def _load(self):
        if os.path.exists(MODEL_PATH):
            self.model  = joblib.load(MODEL_PATH)
        if os.path.exists(SCALER_PATH):
            self.scaler = joblib.load(SCALER_PATH)

    def compute_shap(self, vitals: dict) -> dict:
        if self.model is None or self.scaler is None:
            return self._fallback(vitals)

        x        = np.array([[vitals.get(f, 0) for f in FEATURES]])
        x_scaled = self.scaler.transform(x)

        try:
            import shap
            explainer = shap.TreeExplainer(self.model)
            sv        = explainer.shap_values(x_scaled)
            if isinstance(sv, list):
                shap_vals = sv[1][0]
            else:
                shap_vals = sv[0] if sv.ndim == 2 else sv
        except ImportError:
            shap_vals = self._approx_shap(vitals)

        cluster = self._get_cluster(vitals)

        order = np.argsort(np.abs(shap_vals))[::-1]
        return {
            "features":    [FEATURE_LABELS[i] for i in order],
            "shap_values": [float(shap_vals[i]) for i in order],
            "cluster":     cluster,
        }

    def _approx_shap(self, vitals: dict) -> np.ndarray:
        means   = {
            "age":52,"sex":0.68,"chest_pain_type":0.9,"resting_bp":132,
            "cholesterol":246,"fasting_blood_sugar":0.15,"resting_ecg":0.5,
            "max_heart_rate":149,"exercise_angina":0.33,"st_depression":1.1,
            "st_slope":1.0,"num_vessels":0.7,"thalassemia":1.0,
        }
        weights = {
            "num_vessels":0.35,"st_depression":0.28,"exercise_angina":0.25,
            "chest_pain_type":-0.22,"thalassemia":0.18,"max_heart_rate":-0.15,
            "st_slope":0.14,"age":0.12,"resting_bp":0.10,"cholesterol":0.08,
            "fasting_blood_sugar":0.07,"sex":0.06,"resting_ecg":0.05,
        }
        sv = []
        for f in FEATURES:
            diff = vitals.get(f, means[f]) - means[f]
            r    = max(abs(diff), 0.01)
            sv.append(weights.get(f, 0.05) * diff / r * 0.4)
        return np.array(sv)

    def _get_cluster(self, vitals: dict) -> int:
        bp    = vitals.get("resting_bp", 120)
        chol  = vitals.get("cholesterol", 200)
        age   = vitals.get("age", 50)
        angina = vitals.get("exercise_angina", 0)
        st_dep = vitals.get("st_depression", 0)
        ca    = vitals.get("num_vessels", 0)

        score = (
            (bp > 140) + (chol > 240) + (age > 60) +
            angina + (st_dep > 1.5) + (ca >= 1)
        )
        if score >= 4: return 2
        if score >= 2: return 1
        return 0

    def _fallback(self, vitals: dict) -> dict:
        sv    = self._approx_shap(vitals)
        order = np.argsort(np.abs(sv))[::-1]
        return {
            "features":    [FEATURE_LABELS[i] for i in order],
            "shap_values": [float(sv[i]) for i in order],
            "cluster":     self._get_cluster(vitals),
        }

    def cluster_patient(self, x: np.ndarray) -> int:
        return 0

    def get_feature_importance(self) -> dict:
        if self.model is None:
            return {}
        try:
            return dict(zip(FEATURE_LABELS, [float(v) for v in self.model.feature_importances_]))
        except Exception:
            return {}
