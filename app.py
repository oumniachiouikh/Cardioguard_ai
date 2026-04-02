"""
app.py — CardioGuard AI
Streamlit multi-tab dashboard
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

st.set_page_config(
    page_title="CardioGuard AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar — patient vitals ─────────────────────────────────────────────────
st.sidebar.title("🩺 Patient Vitals")
st.sidebar.markdown("---")

age    = st.sidebar.slider("Age (years)",                    20,  85,  54)
sex    = st.sidebar.selectbox("Sex",                         ["Male (1)", "Female (0)"])
cp     = st.sidebar.selectbox("Chest Pain Type",             ["Asymptomatic (0)","Atypical Angina (1)","Non-Anginal (2)","Typical Angina (3)"])
bp     = st.sidebar.number_input("Resting BP (mmHg)",        80,  200, 130, step=1)
chol   = st.sidebar.number_input("Cholesterol (mg/dL)",      100, 420, 250, step=1)
fbs    = st.sidebar.selectbox("Fasting Blood Sugar >120",    ["No (0)", "Yes (1)"])
ecg    = st.sidebar.selectbox("Resting ECG",                 ["Normal (0)", "ST-T Abnormality (1)", "LVH (2)"])
hr     = st.sidebar.number_input("Max Heart Rate (bpm)",     60,  202, 148, step=1)
angina = st.sidebar.selectbox("Exercise Angina",             ["No (0)", "Yes (1)"])
st_dep = st.sidebar.number_input("ST Depression (mm)",       0.0, 6.2, 1.0, step=0.1)
slope  = st.sidebar.selectbox("ST Slope",                    ["Upsloping (0)", "Flat (1)", "Downsloping (2)"])
ca     = st.sidebar.slider("Num Major Vessels (0–3)",        0, 3, 0)
thal   = st.sidebar.selectbox("Thalassemia",                 ["Normal (0)", "Fixed Defect (1)", "Reversible Defect (2)"])

vitals = {
    "age":                int(age),
    "sex":                int(sex.split("(")[1][0]),
    "chest_pain_type":    int(cp.split("(")[1][0]),
    "resting_bp":         int(bp),
    "cholesterol":        int(chol),
    "fasting_blood_sugar":int(fbs.split("(")[1][0]),
    "resting_ecg":        int(ecg.split("(")[1][0]),
    "max_heart_rate":     int(hr),
    "exercise_angina":    int(angina.split("(")[1][0]),
    "st_depression":      float(st_dep),
    "st_slope":           int(slope.split("(")[1][0]),
    "num_vessels":        int(ca),
    "thalassemia":        int(thal.split("(")[1][0]),
}
st.session_state["vitals"] = vitals

# ── Header ───────────────────────────────────────────────────────────────────
st.title("❤️ CardioGuard AI")
st.markdown("*Cardiovascular Risk Prediction · Powered by XGBoost · SHAP · FAISS RAG · Mistral LLM · LangChain*")
st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Prediction & Alerts",
    "🔍 SHAP Analysis",
    "📚 Medical RAG",
    "💬 AI Chat",
    "📈 Dataset Overview",
])

# ── Tab 1: Prediction ────────────────────────────────────────────────────────
with tab1:
    from agents.predict_agent import PredictAgent
    from agents.alert_agent   import AlertAgent

    st.header("⚡ Cardiovascular Risk Prediction")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🧑 Age",          f"{vitals['age']} yrs")
    col2.metric("💉 Resting BP",   f"{vitals['resting_bp']} mmHg")
    col3.metric("🩸 Cholesterol",  f"{vitals['cholesterol']} mg/dL")
    col4.metric("💓 Max HR",       f"{vitals['max_heart_rate']} bpm")

    st.markdown("---")

    if st.button("🔮 Run CardioGuard Prediction", type="primary"):
        with st.spinner("Running XGBoost cardiovascular risk model..."):
            predictor = PredictAgent()
            result    = predictor.predict(vitals)

        prob  = result["probability"]
        risk  = result["risk_level"]
        label = result["label"]

        color = {"Low":"🟢","Moderate":"🟡","High":"🔴","Critical":"🚨"}.get(risk, "⚪")
        st.subheader(f"{color} Risk Level: **{risk}** — {label}")

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Cardiovascular Event Probability", f"{prob:.1%}")
            st.progress(prob)
        with col_b:
            st.metric("AI Verdict", label)
            if risk in ("Critical","High"):
                st.error("⚠️ Immediate cardiologist consultation recommended.")
            elif risk == "Moderate":
                st.warning("⚠️ Structured lifestyle intervention and GP review recommended.")
            else:
                st.success("✅ Low risk. Maintain a heart-healthy lifestyle.")

        st.markdown("---")
        st.subheader("🚨 Clinical Alerts")
        with st.spinner("Generating clinical alerts..."):
            alerter = AlertAgent()
            alerts  = alerter.generate_alerts(vitals, result)

        for alert in alerts:
            if alert["level"] == "critical":
                st.error(f"🔴 {alert['message']}")
            elif alert["level"] == "warning":
                st.warning(f"🟡 {alert['message']}")
            else:
                st.info(f"🔵 {alert['message']}")

        st.session_state["last_prediction"] = result

# ── Tab 2: SHAP ──────────────────────────────────────────────────────────────
with tab2:
    from agents.shap_agent import SHAPAgent
    import plotly.graph_objects as go

    st.header("🔍 SHAP Explainability + Patient Clustering")

    if st.button("📊 Run SHAP Analysis"):
        with st.spinner("Computing SHAP values..."):
            shap_agent  = SHAPAgent()
            shap_result = shap_agent.compute_shap(vitals)

        st.subheader("Feature Contributions to Risk Prediction")
        feats  = shap_result["features"]
        svals  = shap_result["shap_values"]
        colors = ["#E24B4A" if v > 0 else "#378ADD" for v in svals]

        fig = go.Figure(go.Bar(
            x=svals, y=feats, orientation="h",
            marker_color=colors,
        ))
        fig.update_layout(
            title="SHAP values — contribution to cardiovascular risk",
            xaxis_title="SHAP value (positive = increases risk)",
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Patient Risk Profile Cluster")
        cluster = shap_result["cluster"]
        cluster_map = {
            0: ("🟢 Low-risk profile (Cluster 0)", "Vitals are within healthy ranges. No major risk factors detected."),
            1: ("🟡 Moderate-risk profile (Cluster 1)", "Some elevated risk factors present. Lifestyle intervention is recommended."),
            2: ("🔴 High-risk profile (Cluster 2)", "Multiple significant risk factors detected. Cardiology referral is strongly advised."),
        }
        lbl, desc = cluster_map.get(cluster, ("Unknown",""))
        st.info(f"**{lbl}**\n\n{desc}")

        with st.expander("📋 Feature breakdown"):
            df_shap = pd.DataFrame({
                "Feature":    feats,
                "SHAP Value": [round(v,4) for v in svals],
                "Direction":  ["↑ Increases risk" if v > 0 else "↓ Decreases risk" for v in svals],
            })
            st.dataframe(df_shap, use_container_width=True)

# ── Tab 3: RAG ───────────────────────────────────────────────────────────────
with tab3:
    from agents.rag_agent import RAGAgent

    st.header("📚 Medical Knowledge Base (RAG)")
    st.markdown("Ask any cardiology question — answered from the medical knowledge base using FAISS + Mistral")

    query = st.text_input("Ask a cardiology question:", "What lifestyle changes reduce cardiovascular risk?")

    if st.button("🔍 Search Medical Knowledge Base"):
        with st.spinner("Searching knowledge base..."):
            rag    = RAGAgent()
            answer = rag.search(query)

        st.subheader("Answer:")
        st.markdown(answer["response"])
        with st.expander("📄 Source Documents"):
            for i, doc in enumerate(answer.get("sources", []), 1):
                st.markdown(f"**Source {i}:** {doc}")

# ── Tab 4: Chat ──────────────────────────────────────────────────────────────
with tab4:
    from agents.explain_agent import ExplainAgent

    st.header("💬 AI Medical Assistant")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    v    = st.session_state.get("vitals", vitals)
    pred = st.session_state.get("last_prediction", {})

    ctx = (
        f"Patient: Age {v.get('age','?')}, BP {v.get('resting_bp','?')}mmHg, "
        f"Cholesterol {v.get('cholesterol','?')}mg/dL, Max HR {v.get('max_heart_rate','?')}bpm, "
        f"Exercise Angina: {'Yes' if v.get('exercise_angina')==1 else 'No'}"
    )
    if pred:
        ctx += f" | AI: {pred.get('label','?')} (Risk: {pred.get('risk_level','?')}, {pred.get('probability',0)*100:.1f}%)"

    user_input = st.text_input("Ask the medical AI assistant:", f"Based on this patient: {ctx}. What are the key interventions?")

    if st.button("Send"):
        with st.spinner("Generating personalised medical advice..."):
            explainer = ExplainAgent()
            response  = explainer.explain(user_input, v, pred)

        st.session_state["chat_history"].append({"role":"user",      "content": user_input})
        st.session_state["chat_history"].append({"role":"assistant", "content": response})

    for msg in st.session_state["chat_history"]:
        if msg["role"] == "user":
            st.markdown(f"🧑 **You:** {msg['content']}")
        else:
            st.markdown(f"🤖 **CardioGuard AI:** {msg['content']}")

    if st.button("Clear Chat History"):
        st.session_state["chat_history"] = []
        st.rerun()

# ── Tab 5: Dataset ───────────────────────────────────────────────────────────
with tab5:
    import plotly.express as px

    st.header("📈 Dataset Overview — Cleveland Heart Disease")

    @st.cache_data
    def load_data():
        path = "data/cardio_data.csv"
        if os.path.exists(path):
            return pd.read_csv(path)
        return None

    df = load_data()
    if df is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total patients", len(df))
        col2.metric("Heart disease cases", int(df["target"].sum()))
        col3.metric("No disease cases", int((df["target"]==0).sum()))

        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x="age", color=df["target"].map({0:"No Disease",1:"Heart Disease"}),
                               title="Age distribution by diagnosis", nbins=20,
                               color_discrete_map={"No Disease":"#378ADD","Heart Disease":"#E24B4A"})
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.scatter(df, x="cholesterol", y="resting_bp",
                              color=df["target"].map({0:"No Disease",1:"Heart Disease"}),
                              title="Cholesterol vs Resting BP",
                              color_discrete_map={"No Disease":"#378ADD","Heart Disease":"#E24B4A"})
            st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.box(df, x=df["target"].map({0:"No Disease",1:"Heart Disease"}),
                      y="max_heart_rate", color=df["target"].map({0:"No Disease",1:"Heart Disease"}),
                      title="Max Heart Rate by Diagnosis",
                      color_discrete_map={"No Disease":"#378ADD","Heart Disease":"#E24B4A"})
        st.plotly_chart(fig3, use_container_width=True)

        with st.expander("📋 Raw data sample"):
            st.dataframe(df.head(50), use_container_width=True)
    else:
        st.warning("Run `python data/generate_data.py` to generate the dataset.")
