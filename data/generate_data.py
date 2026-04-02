"""
data/generate_data.py
Generates synthetic cardiac patient data based on Cleveland Heart Disease
feature distributions and trains the XGBoost model.
Run: python data/generate_data.py
"""

import numpy as np
import pandas as pd
import os

FEATURES = [
    "age", "sex", "chest_pain_type", "resting_bp", "cholesterol",
    "fasting_blood_sugar", "resting_ecg", "max_heart_rate",
    "exercise_angina", "st_depression", "st_slope", "num_vessels", "thalassemia"
]

FEATURE_LABELS = {
    "age":               "Age (years)",
    "sex":               "Sex (1=Male, 0=Female)",
    "chest_pain_type":   "Chest Pain Type (0-3)",
    "resting_bp":        "Resting Blood Pressure (mmHg)",
    "cholesterol":       "Serum Cholesterol (mg/dL)",
    "fasting_blood_sugar":"Fasting Blood Sugar >120mg/dL (1=Yes)",
    "max_heart_rate":    "Max Heart Rate Achieved (bpm)",
    "exercise_angina":   "Exercise-Induced Angina (1=Yes)",
    "st_depression":     "ST Depression (mm)",
    "st_slope":          "ST Slope (0=Up, 1=Flat, 2=Down)",
    "num_vessels":       "Num Major Vessels (0-3)",
    "thalassemia":       "Thalassemia (0=Normal, 1=Fixed, 2=Reversible)",
    "resting_ecg":       "Resting ECG (0=Normal, 1=ST-T abnormality, 2=LVH)",
}


def generate_dataset(n: int = 900) -> pd.DataFrame:
    np.random.seed(42)

    age        = np.random.normal(55, 10, n).clip(25, 85).round().astype(int)
    sex        = np.random.binomial(1, 0.68, n)
    cp         = np.random.choice([0,1,2,3], n, p=[0.47,0.17,0.28,0.08])
    trestbps   = np.random.normal(132, 18, n).clip(90, 200).round().astype(int)
    chol       = np.random.normal(246, 52, n).clip(120, 420).round().astype(int)
    fbs        = np.random.binomial(1, 0.15, n)
    restecg    = np.random.choice([0,1,2], n, p=[0.50,0.40,0.10])
    thalach    = np.random.normal(149, 23, n).clip(70, 202).round().astype(int)
    exang      = np.random.binomial(1, 0.33, n)
    oldpeak    = np.random.exponential(1.1, n).clip(0, 6.2).round(1)
    slope      = np.random.choice([0,1,2], n, p=[0.21,0.46,0.33])
    ca         = np.random.choice([0,1,2,3], n, p=[0.58,0.22,0.13,0.07])
    thal       = np.random.choice([0,1,2], n, p=[0.54,0.06,0.40])

    # Risk score based on clinical knowledge
    risk = (
          0.04  * (age - 40)
        + 0.25  * sex
        + 0.30  * (cp == 0).astype(float)   # asymptomatic = high risk
        - 0.20  * (cp == 2).astype(float)   # non-anginal = lower
        + 0.008 * (trestbps - 120)
        + 0.003 * (chol - 200)
        + 0.20  * fbs
        + 0.30  * exang
        + 0.25  * oldpeak
        + 0.20  * (slope == 1).astype(float)
        + 0.35  * (slope == 2).astype(float)
        + 0.25  * ca
        + 0.15  * (thal == 2).astype(float)
        - 0.005 * (thalach - 100)
    )
    prob = 1 / (1 + np.exp(-risk + 1.5))
    target = (prob > 0.5).astype(int)
    # Add noise
    flip = np.random.random(n) < 0.05
    target = np.where(flip, 1 - target, target)

    df = pd.DataFrame({
        "age": age, "sex": sex, "chest_pain_type": cp,
        "resting_bp": trestbps, "cholesterol": chol,
        "fasting_blood_sugar": fbs, "resting_ecg": restecg,
        "max_heart_rate": thalach, "exercise_angina": exang,
        "st_depression": oldpeak, "st_slope": slope,
        "num_vessels": ca, "thalassemia": thal,
        "target": target,
    })
    return df


if __name__ == "__main__":
    print("Generating CardioGuard dataset...")
    df = generate_dataset(900)
    os.makedirs("data",   exist_ok=True)
    os.makedirs("models", exist_ok=True)

    df.to_csv("data/cardio_data.csv", index=False)
    print(f"  Saved {len(df)} rows → data/cardio_data.csv")
    print(f"  Target distribution:\n  {df['target'].value_counts().to_string()}")

    # Train XGBoost
    print("\nTraining XGBoost classifier...")
    from xgboost import XGBClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
    import joblib

    X = df[FEATURES].values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler  = StandardScaler()
    Xtr_sc  = scaler.fit_transform(X_train)
    Xte_sc  = scaler.transform(X_test)

    model = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="logloss",
        random_state=42,
    )
    model.fit(Xtr_sc, y_train, eval_set=[(Xte_sc, y_test)], verbose=False)

    y_pred  = model.predict(Xte_sc)
    y_proba = model.predict_proba(Xte_sc)[:,1]
    print(f"  Accuracy: {accuracy_score(y_test,y_pred):.3f}")
    print(f"  AUC-ROC:  {roc_auc_score(y_test,y_proba):.3f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No Disease','Heart Disease'])}")

    joblib.dump(model,  "models/cardio_xgb.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    print("  Model  → models/cardio_xgb.pkl")
    print("  Scaler → models/scaler.pkl")
    print("\nDone! Run: streamlit run app.py")
