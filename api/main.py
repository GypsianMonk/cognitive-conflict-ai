"""
FastAPI Backend — REST API for the Cognitive Conflict AI system.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import yaml
from pathlib import Path

from engine.debate_manager import DebateManager
from scoring.metrics import get_metrics, record_output
from memory.memory_store import MemoryStore


# ── Load config ─────────────────────────────────────────────────────────────
def load_config() -> dict:
    config_path = Path("config/config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {
        "model": {"provider": "ollama", "name": "mistral", "temperature": 0.7},
        "agents": {"debate_rounds": 3, "enable_expert_injection": True},
        "scoring": {"relevance_weight": 0.4, "coherence_weight": 0.4, "contradiction_penalty": 0.2},
        "memory": {"enabled": True},
    }


config = load_config()
manager = DebateManager(config)
memory = MemoryStore() if config.get("memory", {}).get("enabled", True) else None


# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Cognitive Conflict AI",
    description="Multi-agent adversarial reasoning system for validated AI answers.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ──────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    context: Optional[str] = None


class QueryResponse(BaseModel):
    query: str
    final_answer: str
    reasoning_trace: str
    confidence_score: str
    convergence_achieved: bool
    total_debate_rounds: int
    domain: Optional[str]
    complexity: str
    contradictions: list[str]
    agent_responses: list[dict]
    duration_seconds: float


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "Cognitive Conflict AI",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    output = manager.run(request.query, retrieval_context=request.context)
    record_output(output)

    if memory:
        memory.save(output)

    return output.to_dict()


@app.get("/metrics")
def metrics():
    return get_metrics().summary()


@app.get("/history")
def history(limit: int = 20):
    if not memory:
        raise HTTPException(status_code=503, detail="Memory is disabled.")
    return memory.get_all(limit=limit)


@app.get("/health")
def health():
    return {"status": "healthy"}
