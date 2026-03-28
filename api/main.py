"""
Cognitive Conflict AI — FastAPI Backend v2
All features exposed via REST API including feedback, KG, red-team, leaderboard, audit.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import yaml
from pathlib import Path

from engine.debate_manager import DebateManager
from scoring.metrics import get_metrics, record_output
from scoring.preference_model import PreferenceModel, UserFeedback
from memory.memory_store import MemoryStore
from memory.knowledge_graph import KnowledgeGraph
from agents.red_team_agent import RedTeamAgent
from engine.audit_logger import get_logger
from agents.persona_evolver import PersonaEvolver


def load_config() -> dict:
    p = Path("config/config.yaml")
    if p.exists():
        with open(p) as f:
            return yaml.safe_load(f)
    return {
        "model": {"provider": "ollama", "name": "mistral", "temperature": 0.7},
        "agents": {"debate_rounds": 3, "enable_expert_injection": True, "enable_reflection": True},
        "scoring": {"relevance_weight": 0.4, "coherence_weight": 0.4, "contradiction_penalty": 0.2},
        "memory": {"enabled": True},
    }


config = load_config()
manager = DebateManager(config)
memory = MemoryStore()
kg = KnowledgeGraph()
preference_model = PreferenceModel()
persona_evolver = PersonaEvolver()

app = FastAPI(
    title="Cognitive Conflict AI",
    description="Production-grade multi-agent adversarial reasoning system.",
    version="2.0.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class QueryRequest(BaseModel):
    query: str
    context: Optional[str] = None

class FeedbackRequest(BaseModel):
    query_id: str
    rating: int
    helpful: bool
    comment: Optional[str] = None
    domain: Optional[str] = None

class KGAddRequest(BaseModel):
    content: str
    node_type: str = "fact"
    confidence: float = 0.8
    source: str = "manual"
    tags: list = []


@app.get("/")
def root():
    return {"service": "Cognitive Conflict AI", "version": "2.0.0", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "healthy", "features": [
        "anti_herding", "tree_of_thoughts", "self_reflection", "working_memory",
        "tool_agents", "fact_checking", "hallucination_detection",
        "uncertainty_decomposition", "audit_logging", "safety_checking",
        "task_decomposition", "persona_evolution", "knowledge_graph",
        "preference_learning", "hybrid_rag", "temporal_awareness", "red_team",
    ]}

@app.post("/query")
def query(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(400, "Query cannot be empty.")
    output = manager.run(request.query, retrieval_context=request.context)
    return output.to_dict()

@app.post("/feedback")
def submit_feedback(req: FeedbackRequest):
    preference_model.record_feedback(UserFeedback(
        query_id=req.query_id, rating=req.rating, helpful=req.helpful,
        comment=req.comment, domain=req.domain,
    ))
    return {"status": "recorded", "query_id": req.query_id}

@app.get("/feedback/stats")
def feedback_stats():
    return preference_model.get_feedback_stats()

@app.get("/memory/history")
def history(limit: int = 20):
    return memory.get_all(limit=limit)

@app.get("/memory/search")
def search_memory(q: str, limit: int = 5):
    return memory.search_similar(q, limit=limit)

@app.get("/knowledge-graph/stats")
def kg_stats():
    return kg.stats()

@app.get("/knowledge-graph/search")
def kg_search(q: str, limit: int = 5):
    nodes = kg.search(q, limit=limit)
    return [{"id": n.id, "content": n.content, "confidence": n.confidence,
             "type": n.node_type, "tags": n.tags} for n in nodes]

@app.post("/knowledge-graph/add")
def kg_add(req: KGAddRequest):
    import hashlib, time
    from memory.knowledge_graph import KGNode
    node_id = hashlib.md5((req.content + str(time.time())).encode()).hexdigest()[:12]
    node = KGNode(id=node_id, content=req.content, node_type=req.node_type,
                  confidence=req.confidence, source_query=req.source, tags=req.tags)
    kg.add_node(node)
    return {"status": "added", "node_id": node_id}

@app.get("/metrics")
def metrics():
    return get_metrics().summary()

@app.get("/agents/leaderboard")
def agent_leaderboard():
    return persona_evolver.leaderboard()

@app.get("/audit/session")
def audit_session():
    return get_logger().get_session_log()

@app.post("/red-team/run")
def run_red_team():
    agent = RedTeamAgent()
    report = agent.run_full_red_team()
    return report.to_dict()
