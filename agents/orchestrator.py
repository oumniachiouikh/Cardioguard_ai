"""
agents/orchestrator.py
CardioGuard Orchestrator — LangChain Router
Routes queries to the correct specialised agent.
"""

from langchain.memory import ConversationBufferMemory
from agents.data_agent    import DataAgent
from agents.predict_agent import PredictAgent
from agents.rag_agent     import RAGAgent
from agents.explain_agent import ExplainAgent
from agents.alert_agent   import AlertAgent
from agents.shap_agent    import SHAPAgent


class CardioOrchestrator:
    def __init__(self):
        self.memory = ConversationBufferMemory(return_messages=True)
        self.agents = {
            "data":    DataAgent(),
            "predict": PredictAgent(),
            "rag":     RAGAgent(),
            "explain": ExplainAgent(),
            "alert":   AlertAgent(),
            "shap":    SHAPAgent(),
        }

    def route_query(self, query: str) -> str:
        q = query.lower()
        if any(k in q for k in ["preprocess","load","clean","dataset","data"]):
            return "data"
        if any(k in q for k in ["predict","risk","probability","likelihood","chance"]):
            return "predict"
        if any(k in q for k in ["explain","why","reason","cause","what is","interpret"]):
            return "explain"
        if any(k in q for k in ["alert","warning","danger","critical","urgent"]):
            return "alert"
        if any(k in q for k in ["shap","feature","importance","cluster","profile"]):
            return "shap"
        return "rag"

    def run(self, query: str, vitals: dict = None) -> dict:
        agent_name = self.route_query(query)
        self.memory.chat_memory.add_user_message(query)

        if agent_name == "predict" and vitals:
            result   = self.agents["predict"].predict(vitals)
            response = f"{result['label']} — Risk: {result['risk_level']} ({result['probability']*100:.1f}%)"
        elif agent_name == "explain" and vitals:
            pred     = self.agents["predict"].predict(vitals)
            response = self.agents["explain"].explain(query, vitals, pred)
        elif agent_name == "alert" and vitals:
            pred     = self.agents["predict"].predict(vitals)
            alerts   = self.agents["alert"].generate_alerts(vitals, pred)
            response = " | ".join([a["message"] for a in alerts])
        elif agent_name == "shap" and vitals:
            result   = self.agents["shap"].compute_shap(vitals)
            response = f"Top feature: {result['features'][0]} (SHAP: {result['shap_values'][0]:.3f})"
        else:
            result   = self.agents["rag"].search(query)
            response = result["response"]

        self.memory.chat_memory.add_ai_message(response)
        return {"agent": agent_name, "response": response}
