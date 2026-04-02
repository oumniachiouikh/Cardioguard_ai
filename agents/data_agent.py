"""
agents/data_agent.py
Agent 1 — Data Pipeline
Loads, cleans and preprocesses cardiac patient vitals.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

FEATURES = [
    "age", "sex", "chest_pain_type", "resting_bp", "cholesterol",
    "fasting_blood_sugar", "resting_ecg", "max_heart_rate",
    "exercise_angina", "st_depression", "st_slope", "num_vessels", "thalassemia"
]

NORMAL_RANGES = {
    "resting_bp":   (90, 180),
    "cholesterol":  (100, 400),
    "max_heart_rate": (50, 210),
    "st_depression":  (0, 7),
    "age":            (18, 90),
}


class DataAgent:
    def __init__(self):
        self.scaler = StandardScaler()
        self.df     = None

    def download_data(self) -> None:
        path = "data/cardio_data.csv"
        if not os.path.exists(path):
            raise FileNotFoundError("Run `python data/generate_data.py` first.")
        self.df = pd.read_csv(path)

    def clean_zeros(self) -> pd.DataFrame:
        if self.df is None:
            self.download_data()
        df = self.df.copy()
        for col in ["cholesterol", "resting_bp", "max_heart_rate"]:
            if col in df.columns:
                median = df[col][df[col] > 0].median()
                df[col] = df[col].replace(0, median)
        self.df = df
        return df

    def preprocess(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.fit_transform(X)

    def validate_input(self, vitals: dict) -> list:
        """Returns list of warnings for out-of-range values."""
        warnings = []
        for key, (lo, hi) in NORMAL_RANGES.items():
            val = vitals.get(key, None)
            if val is not None and (val < lo or val > hi):
                warnings.append(f"{key} value {val} is outside expected range [{lo}–{hi}]")
        return warnings

    def get_report(self) -> str:
        if self.df is None:
            self.download_data()
        return (
            f"Dataset: {len(self.df)} rows, {len(self.df.columns)} columns.\n"
            f"Target distribution:\n{self.df['target'].value_counts().to_string()}"
        )
