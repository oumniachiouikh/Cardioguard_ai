# ❤️ CardioGuard AI

A multi-agent AI system for cardiovascular event risk prediction and personalised lifestyle intervention.

**Predictive AI** — XGBoost classifier trained on Cleveland Heart Disease features  
**Generative AI** — Mistral LLM for personalised medically-backed lifestyle advice  
**RAG** — FAISS vector index + Mistral answering cardiology questions from a medical knowledge base  
**Architecture** — LangChain orchestrator coordinating 6 specialised agents

---

## Architecture

```
Streamlit Dashboard (app.py)
          │
          ▼
CardioOrchestrator — LangChain Router
          │
  ┌───────┼───────┬──────────┬──────────┬──────────┐
  ▼       ▼       ▼          ▼          ▼          ▼
Agent1  Agent2  Agent3    Agent4     Agent5    Agent6
Data    XGB     RAG      Mistral    Alerts   SHAP+KMeans
         │       │          │          │          │
         ▼       ▼          ▼          ▼          ▼
  cardio_data  FAISS     Mistral   Clinical   XGBoost
     .csv      Index      LLM     Thresholds   Model
```

## Agents

| Agent | Role | Tech |
|-------|------|------|
| Agent 1 — DataAgent     | Load, clean, preprocess patient data | pandas, StandardScaler |
| Agent 2 — PredictAgent  | Predict cardiovascular event risk     | XGBoost, AUC ~0.88 |
| Agent 3 — RAGAgent      | Answer medical questions              | FAISS, sentence-transformers, Mistral |
| Agent 4 — ExplainAgent  | Personalised lifestyle interventions  | Mistral-7B-Instruct |
| Agent 5 — AlertAgent    | Clinical threshold-based alerts       | Rule-based clinical guidelines |
| Agent 6 — SHAPAgent     | Feature importance + patient cluster  | SHAP TreeExplainer, KMeans k=3 |

## Dataset

Based on the **Cleveland Heart Disease Dataset** (UCI):
- 900 synthetic patient records
- 13 clinical features: age, sex, chest pain type, resting BP, cholesterol, fasting blood sugar, ECG, max heart rate, exercise angina, ST depression, ST slope, number of vessels, thalassemia
- Binary target: 0 = No Heart Disease, 1 = Heart Disease

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate dataset and train model
python data/generate_data.py

# 3. Launch the dashboard
streamlit run app.py
# Open http://localhost:8501
```

---

## Project structure

```
cardioguard/
├── app.py
├── requirements.txt
├── README.md
├── cardioguard_demo.html     ← standalone browser demo
├── agents/
│   ├── orchestrator.py
│   ├── data_agent.py
│   ├── predict_agent.py
│   ├── rag_agent.py
│   ├── explain_agent.py
│   ├── alert_agent.py
│   └── shap_agent.py
├── data/
│   ├── generate_data.py
│   └── cardio_data.csv
└── models/
    ├── cardio_xgb.pkl
    ├── scaler.pkl
    └── faiss_cardio/
```

---

*Student project — M1 IoT · Université de Franche-Comté · 2025–2026*
