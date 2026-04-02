"""
agents/rag_agent.py
Agent 3 — Medical RAG Engine
Answers cardiology questions using FAISS + Mistral.
Falls back to keyword search when LLM is unavailable.
"""

import os

FAISS_PATH = "models/faiss_cardio"

MEDICAL_KB = [
    "A heart attack (myocardial infarction) occurs when blood flow to part of the heart muscle is blocked, usually by a blood clot. The most common cause is coronary artery disease. Symptoms include chest pain, shortness of breath, pain radiating to the arm or jaw, sweating, and nausea. Immediate treatment is critical — every minute without treatment destroys more heart muscle.",
    "Stroke occurs when blood supply to part of the brain is cut off (ischemic) or when a blood vessel bursts (hemorrhagic). Risk factors include hypertension, atrial fibrillation, high cholesterol, diabetes, smoking, and obesity. The FAST acronym — Face drooping, Arm weakness, Speech difficulty, Time to call emergency — is the key recognition tool.",
    "Hypertension (high blood pressure) is defined as resting blood pressure above 130/80 mmHg. It is the single biggest modifiable risk factor for cardiovascular disease. It is often called the 'silent killer' because it produces no symptoms until a serious event occurs. Lifestyle interventions include reducing sodium intake, increasing physical activity, and losing weight.",
    "High cholesterol, particularly elevated LDL ('bad') cholesterol, contributes to atherosclerosis — the build-up of plaques in artery walls. Total cholesterol above 200 mg/dL is considered borderline high. Statins are first-line pharmacological treatment. Dietary changes including reducing saturated fat and increasing fiber are effective first-line interventions.",
    "Exercise-induced angina is chest pain caused by reduced blood flow to the heart during physical exertion. It indicates that coronary arteries are narrowed and cannot supply enough oxygenated blood during increased cardiac demand. It is a strong predictor of underlying coronary artery disease and warrants further investigation.",
    "ST depression on an ECG indicates myocardial ischaemia — the heart muscle is not receiving enough oxygen. The deeper the ST depression, the more severe the ischaemia. Combined with exercise angina and other risk factors, it significantly increases the probability of a cardiovascular event.",
    "Diabetes significantly increases cardiovascular risk by promoting atherosclerosis, hypertension, and dyslipidaemia. Patients with diabetes are 2–4 times more likely to develop coronary artery disease than non-diabetic individuals. Fasting blood sugar above 126 mg/dL on two separate tests confirms diabetes diagnosis.",
    "The Mediterranean diet — rich in olive oil, fish, fruits, vegetables, legumes, and whole grains — has the strongest evidence base for cardiovascular protection. Studies show it reduces major cardiovascular events by 30%. It reduces inflammation, improves lipid profiles, and lowers blood pressure.",
    "Aerobic exercise is one of the most powerful lifestyle interventions for cardiovascular health. The American Heart Association recommends at least 150 minutes of moderate-intensity exercise per week. Regular exercise lowers blood pressure, improves HDL cholesterol, reduces body weight, and decreases the risk of a first or subsequent cardiovascular event by up to 35%.",
    "Smoking is one of the strongest independent risk factors for cardiovascular disease. It damages the endothelium (inner lining of blood vessels), promotes platelet aggregation, raises blood pressure, and reduces HDL cholesterol. The risk begins to decrease within 24 hours of quitting and reaches near-normal levels after 10–15 years of cessation.",
    "Obesity, particularly abdominal obesity (waist circumference >102cm in men, >88cm in women), is strongly associated with cardiovascular risk through its effects on blood pressure, blood sugar, and inflammation. Even modest weight loss of 5–10% significantly reduces cardiovascular risk factors.",
    "Atrial fibrillation (AF) is an irregular heart rhythm that increases stroke risk by 5-fold due to blood clot formation in the heart. Treatment with anticoagulants reduces this risk by approximately 65%. AF affects 1–2% of the general population and up to 10% of people over 80.",
    "Thalassemia affects haemoglobin structure and can cause anaemia, which forces the heart to work harder. Patients with severe thalassemia are at increased risk of heart failure and arrhythmias due to iron overload from repeated blood transfusions.",
    "Stress management is an underappreciated cardiovascular intervention. Chronic psychological stress activates the sympathetic nervous system, raising blood pressure and promoting inflammation. Mindfulness meditation, yoga, and cognitive-behavioural therapy have all demonstrated measurable reductions in cardiovascular risk markers.",
    "Cardiac rehabilitation programs combining supervised exercise, dietary counselling, and psychological support reduce cardiovascular mortality by 20–30% in patients who have already experienced a heart attack. They are significantly underutilised despite strong evidence.",
    "Aspirin therapy in low doses (75–100mg/day) reduces platelet aggregation and is used for secondary prevention in patients with established cardiovascular disease. Its role in primary prevention is more controversial due to bleeding risk and is now generally not recommended for low-risk patients.",
    "The maximum heart rate formula (220 minus age) provides an estimate of the highest heart rate achievable during maximal exercise. A significantly reduced maximum heart rate during a stress test (chronotropic incompetence) is an independent predictor of cardiovascular mortality.",
    "Coronary artery calcium (CAC) scoring via CT scan directly measures calcified plaque in coronary arteries and is one of the most powerful predictors of future cardiovascular events, even beyond traditional risk factors. A score of zero confers a very low short-term risk.",
]


class RAGAgent:
    def __init__(self):
        self.db         = None
        self.embeddings = None
        self._load_or_build()

    def _load_or_build(self):
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_community.embeddings  import HuggingFaceEmbeddings
            from langchain.schema                import Document

            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            if os.path.exists(FAISS_PATH):
                self.db = FAISS.load_local(
                    FAISS_PATH, self.embeddings, allow_dangerous_deserialization=True
                )
            else:
                docs    = [Document(page_content=t) for t in MEDICAL_KB]
                self.db = FAISS.from_documents(docs, self.embeddings)
                os.makedirs(FAISS_PATH, exist_ok=True)
                self.db.save_local(FAISS_PATH)
        except Exception as e:
            print(f"[RAGAgent] FAISS unavailable: {e}. Using keyword fallback.")
            self.db = None

    def search(self, query: str) -> dict:
        if self.db is not None:
            try:
                results = self.db.similarity_search(query, k=3)
                context = "\n".join([r.page_content for r in results])
                sources = [r.page_content[:90] + "..." for r in results]
                return {"response": self._generate(query, context), "sources": sources}
            except Exception as e:
                print(f"[RAGAgent] Search error: {e}")

        # Keyword fallback
        q        = query.lower()
        relevant = [t for t in MEDICAL_KB if any(w in t.lower() for w in q.split() if len(w) > 4)]
        if not relevant:
            relevant = MEDICAL_KB[:3]
        context  = "\n".join(relevant[:3])
        return {
            "response": self._generate(query, context),
            "sources":  [c[:90] + "..." for c in relevant[:3]],
        }

    def _generate(self, query: str, context: str) -> str:
        try:
            from transformers import pipeline
            gen = pipeline(
                "text-generation",
                model="mistralai/Mistral-7B-Instruct-v0.1",
                max_new_tokens=300,
                device_map="auto",
            )
            prompt = f"""[INST] You are a cardiology expert. Answer the question using only the context below.
Be concise, factual, and use plain language a patient can understand.

Context:
{context}

Question: {query}
[/INST]"""
            out = gen(prompt, do_sample=False)
            raw = out[0]["generated_text"]
            return raw.split("[/INST]")[-1].strip() if "[/INST]" in raw else raw.strip()
        except Exception:
            return f"Based on the medical knowledge base:\n\n{context}"

    def build_index(self, docs=None):
        self._load_or_build()

    def load_index(self):
        return self.db
