"""
agents/explain_agent.py
Agent 4 — Mistral LLM Explainer
Generates personalized, medically-backed lifestyle interventions
based on patient vitals and prediction results.
"""


class ExplainAgent:
    def __init__(self):
        self.llm = None
        self._load_llm()

    def _load_llm(self):
        try:
            from langchain_community.llms import HuggingFacePipeline
            from transformers             import pipeline
            pipe = pipeline(
                "text-generation",
                model="mistralai/Mistral-7B-Instruct-v0.1",
                max_new_tokens=500,
                device_map="auto",
                do_sample=True,
                temperature=0.65,
            )
            self.llm = HuggingFacePipeline(pipeline=pipe)
        except Exception as e:
            print(f"[ExplainAgent] LLM unavailable: {e}. Using rule-based fallback.")
            self.llm = None

    def explain(self, query: str, vitals: dict, prediction: dict) -> str:
        vitals_str = (
            f"Age: {vitals.get('age','?')}, Sex: {'Male' if vitals.get('sex')==1 else 'Female'}, "
            f"Resting BP: {vitals.get('resting_bp','?')} mmHg, "
            f"Cholesterol: {vitals.get('cholesterol','?')} mg/dL, "
            f"Max Heart Rate: {vitals.get('max_heart_rate','?')} bpm, "
            f"ST Depression: {vitals.get('st_depression','?')} mm, "
            f"Exercise Angina: {'Yes' if vitals.get('exercise_angina')==1 else 'No'}, "
            f"Fasting Blood Sugar >120: {'Yes' if vitals.get('fasting_blood_sugar')==1 else 'No'}"
        )
        pred_str = ""
        if prediction:
            pred_str = (
                f"AI prediction: {prediction.get('label','?')} — "
                f"Risk: {prediction.get('risk_level','?')} "
                f"({prediction.get('probability',0)*100:.1f}% probability)"
            )

        if self.llm:
            prompt = f"""[INST] You are a cardiologist providing a clinical assessment.

Patient vitals: {vitals_str}
{pred_str}

Question: {query}

Provide a clear 4–5 sentence response covering:
1. Key risk factors present in this patient's profile
2. What the AI prediction means clinically
3. Specific, actionable lifestyle interventions tailored to this patient
4. When to seek immediate medical attention
[/INST]"""
            try:
                response = self.llm(prompt)
                return response.split("[/INST]")[-1].strip() if "[/INST]" in response else response.strip()
            except Exception as e:
                print(f"[ExplainAgent] Generation error: {e}")

        return self._rule_based(vitals, prediction, query)

    def _rule_based(self, vitals: dict, prediction: dict, query: str) -> str:
        risk  = prediction.get("risk_level", "Unknown") if prediction else "Unknown"
        prob  = prediction.get("probability", 0) if prediction else 0
        label = prediction.get("label", "Unknown") if prediction else "Unknown"
        age   = vitals.get("age", 50)
        bp    = vitals.get("resting_bp", 120)
        chol  = vitals.get("cholesterol", 200)
        hr    = vitals.get("max_heart_rate", 150)
        angina = vitals.get("exercise_angina", 0)
        fbs    = vitals.get("fasting_blood_sugar", 0)
        st_dep = vitals.get("st_depression", 0)

        lines = [f"**CardioGuard AI Assessment — Risk Level: {risk} ({prob*100:.1f}%)**\n"]

        # Risk factors
        factors = []
        if bp > 140:     factors.append(f"elevated blood pressure ({bp} mmHg)")
        if chol > 240:   factors.append(f"high cholesterol ({chol} mg/dL)")
        if angina == 1:  factors.append("exercise-induced angina")
        if fbs == 1:     factors.append("elevated fasting blood sugar (possible diabetes)")
        if st_dep > 2:   factors.append(f"significant ST depression ({st_dep} mm)")
        if age > 60:     factors.append(f"age ({age}) as a non-modifiable risk factor")

        if factors:
            lines.append(f"**Key risk factors identified:** {', '.join(factors)}.\n")

        # Lifestyle interventions
        lines.append("**Personalized lifestyle interventions:**")
        if bp > 140:
            lines.append("• Reduce sodium intake to <2g/day. Follow the DASH diet. Consider daily blood pressure monitoring.")
        if chol > 240:
            lines.append("• Reduce saturated fat intake. Increase soluble fiber (oats, legumes). Consider plant stanols/sterols.")
        if fbs == 1:
            lines.append("• Monitor blood glucose regularly. Reduce refined carbohydrates and sugary beverages. Consult an endocrinologist.")
        if angina == 1 or st_dep > 1:
            lines.append("• Avoid strenuous exercise until cleared by a cardiologist. Supervised cardiac rehabilitation is strongly recommended.")
        lines.append("• Aim for 150 minutes of moderate aerobic exercise per week (walking, swimming, cycling).")
        lines.append("• Follow a Mediterranean-style diet rich in fish, olive oil, vegetables, and whole grains.")
        lines.append("• If you smoke, quitting now reduces your cardiovascular risk by 50% within one year.")
        lines.append("• Manage stress through mindfulness, adequate sleep (7–9 hours), and regular relaxation.")

        if risk in ("Critical", "High"):
            lines.append("\n⚠️ **Seek immediate medical attention** if you experience chest pain, shortness of breath, arm or jaw pain, or sudden dizziness.")

        return "\n".join(lines)

    def generate_report(self) -> dict:
        return {"agent": "ExplainAgent", "model": "Mistral-7B-Instruct" if self.llm else "rule-based"}
