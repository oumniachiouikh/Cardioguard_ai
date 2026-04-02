# ❤️ CardioGuard AI

A multi-agent artificial intelligence system for cardiovascular risk prediction and personalised lifestyle intervention.

---

## Overview

CardioGuard AI is designed to support early detection of cardiovascular disease (CVD) and provide actionable, personalised recommendations.

The system combines:

- A predictive machine learning model (XGBoost) for estimating cardiovascular risk
- A Retrieval-Augmented Generation (RAG) pipeline for medical question answering
- A large language model (Mistral) for generating personalised lifestyle advice
- A multi-agent architecture orchestrated using LangChain

---

## Key Features

- Cardiovascular risk prediction using XGBoost
- Model explainability using SHAP
- Clinical alert generation based on medical thresholds
- Retrieval-augmented medical question answering (FAISS + Mistral)
- Personalised lifestyle recommendations
- Patient clustering using KMeans

---

## System Architecture
```text
Streamlit Dashboard (app.py)
          │
          ▼
CardioOrchestrator — LangChain Router
          │
  ┌───────┼───────┬──────────┬──────────┬──────────┐
  ▼       ▼       ▼          ▼          ▼          ▼
Agent1  Agent2  Agent3     Agent4     Agent5    Agent6
Data    XGB     RAG       Mistral     Alerts   SHAP+KMeans
          │       │          │          │          │
          ▼       ▼          ▼          ▼          ▼
  cardio_data  FAISS      Mistral    Clinical   XGBoost
     .csv      Index       LLM      Thresholds   Model

---

## Dataset

Based on the Cleveland Heart Disease Dataset (UCI Repository):

- 900 synthetic patient records
- 13 clinical features
- Binary classification target (heart disease)

---

## Model Performance

- Accuracy: ~84%
- AUC-ROC: ~0.88
- Recall: ~86%

---

## Quickstart

```bash
pip install -r requirements.txt
python data/generate_data.py
streamlit run app.py
```

## Project Structure
```text
cardioguard/
├── app.py
├── agents/
├── data/
│   ├── generate_data.py
│   └── cardio_data.csv
└── models/
    ├── cardio_xgb.pkl
    ├── scaler.pkl
    └── faiss_cardio/

## Tech Stack
- Python
- XGBoost
- SHAP
- LangChain
- FAISS
- Mistral LLM
- Streamlit

## Contributions
Oumnia Chiouikh
- Design and implementation of the multi-agent architecture (LangChain orchestrator)
- Development of the XGBoost predictive model
- Integration of SHAP explainability and patient clustering
- Implementation of the Streamlit dashboard
- System integration and end-to-end pipeline
Cynthia Ayetolou
- Development of the RAG pipeline (FAISS + embeddings)
- Integration of the Mistral LLM for medical question answering
- Implementation of personalised lifestyle recommendation generation
- Construction and structuring of the medical knowledge base
- Contribution to testing and system validation

## Authors

Oumnia Chiouikh
Cynthia Ayetolou

M1 IoT — Université Marie et Louis Pasteur
2025–2026

