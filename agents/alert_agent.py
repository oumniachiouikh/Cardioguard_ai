"""
agents/alert_agent.py
Agent 5 — Clinical Alert Engine
Generates medically-grounded alerts based on patient vitals and thresholds.
"""


class AlertAgent:
    def __init__(self):
        self.thresholds = {
            "resting_bp_stage2":  160,
            "resting_bp_stage1":  140,
            "cholesterol_high":   240,
            "cholesterol_border": 200,
            "max_hr_low":         100,
            "st_depression_high": 2.0,
            "st_depression_mod":  1.0,
        }

    def get_risk_score(self, vitals: dict) -> float:
        score = 0.0
        score += max(0, (vitals.get("resting_bp", 120) - 120) * 0.3)
        score += max(0, (vitals.get("cholesterol", 200) - 200) * 0.1)
        score += max(0, vitals.get("st_depression", 0) * 5)
        score += 15 * vitals.get("exercise_angina", 0)
        score += 10 * vitals.get("fasting_blood_sugar", 0)
        score += max(0, (vitals.get("age", 50) - 50) * 0.5)
        return round(min(score, 100), 1)

    def generate_alerts(self, vitals: dict, prediction: dict) -> list:
        alerts = []
        bp     = vitals.get("resting_bp", 120)
        chol   = vitals.get("cholesterol", 200)
        hr     = vitals.get("max_heart_rate", 150)
        angina = vitals.get("exercise_angina", 0)
        fbs    = vitals.get("fasting_blood_sugar", 0)
        st_dep = vitals.get("st_depression", 0)
        cp     = vitals.get("chest_pain_type", 1)
        age    = vitals.get("age", 50)
        thal   = vitals.get("thalassemia", 0)
        ca     = vitals.get("num_vessels", 0)

        # Blood pressure
        if bp >= self.thresholds["resting_bp_stage2"]:
            alerts.append({"level": "critical", "message": f"Stage 2 hypertension: {bp} mmHg. Immediate medical review required. Risk of hypertensive crisis."})
        elif bp >= self.thresholds["resting_bp_stage1"]:
            alerts.append({"level": "warning", "message": f"Stage 1 hypertension: {bp} mmHg. Lifestyle modification and medical assessment recommended."})

        # Cholesterol
        if chol >= self.thresholds["cholesterol_high"]:
            alerts.append({"level": "critical", "message": f"High cholesterol: {chol} mg/dL. Statin therapy and dietary intervention warranted."})
        elif chol >= self.thresholds["cholesterol_border"]:
            alerts.append({"level": "warning", "message": f"Borderline cholesterol: {chol} mg/dL. Dietary changes and lifestyle review recommended."})

        # Exercise angina
        if angina == 1:
            alerts.append({"level": "critical", "message": "Exercise-induced angina detected. This strongly suggests coronary artery narrowing. Cardiology referral is urgent."})

        # ST depression
        if st_dep >= self.thresholds["st_depression_high"]:
            alerts.append({"level": "critical", "message": f"Significant ST depression: {st_dep} mm. Indicates myocardial ischaemia. Immediate ECG and cardiology assessment needed."})
        elif st_dep >= self.thresholds["st_depression_mod"]:
            alerts.append({"level": "warning", "message": f"Moderate ST depression: {st_dep} mm. Suggests reduced cardiac blood flow. Monitor closely."})

        # Fasting blood sugar
        if fbs == 1:
            alerts.append({"level": "warning", "message": "Fasting blood sugar >120 mg/dL. Possible undiagnosed diabetes — a major cardiovascular risk amplifier. HbA1c test recommended."})

        # Max heart rate
        if hr < self.thresholds["max_hr_low"]:
            alerts.append({"level": "warning", "message": f"Low maximum heart rate: {hr} bpm. May indicate chronotropic incompetence — an independent predictor of cardiovascular mortality."})

        # Asymptomatic chest pain (type 0 = highest risk in Cleveland dataset)
        if cp == 0:
            alerts.append({"level": "warning", "message": "Asymptomatic chest pain type recorded. This pattern carries the highest cardiovascular risk in clinical studies despite absence of typical symptoms."})

        # Number of vessels
        if ca >= 2:
            alerts.append({"level": "critical", "message": f"{ca} major vessels affected. Multi-vessel coronary disease significantly elevates risk of acute event. Specialist review is essential."})
        elif ca == 1:
            alerts.append({"level": "warning", "message": "1 major vessel affected. Single-vessel coronary disease detected. Cardiology follow-up recommended."})

        # Thalassemia
        if thal == 2:
            alerts.append({"level": "warning", "message": "Reversible thalassemia defect detected. This indicates areas of myocardium with reversible ischaemia — a significant risk marker."})

        # Age
        if age >= 65:
            alerts.append({"level": "info", "message": f"Age {age}: cardiovascular risk increases significantly after 65. Annual cardiac screening is recommended."})

        # ML prediction
        risk  = prediction.get("risk_level", "Low")
        prob  = prediction.get("probability", 0)
        label = prediction.get("label", "")
        if risk == "Critical":
            alerts.append({"level": "critical", "message": f"AI predicts {label} with {prob*100:.1f}% confidence (Critical risk). Immediate cardiologist consultation is strongly advised."})
        elif risk == "High":
            alerts.append({"level": "critical", "message": f"AI predicts {label} with {prob*100:.1f}% confidence (High risk). Schedule a cardiology appointment within 48 hours."})
        elif risk == "Moderate":
            alerts.append({"level": "warning", "message": f"AI predicts moderate cardiovascular risk ({prob*100:.1f}%). Structured lifestyle intervention and GP review recommended."})

        if not alerts:
            alerts.append({"level": "info", "message": "All vitals are within acceptable clinical ranges. Maintain a heart-healthy lifestyle and attend regular check-ups."})

        return alerts
