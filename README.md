# 🧠 Cognitive Conflict AI

> *Not generating answers faster — but generating answers more intelligently through conflict.*

A **production-grade multi-agent AI reasoning system** that simulates human-like cognitive diversity by generating, debating, and resolving conflicting perspectives before producing a validated final answer.

---

## 🔥 Core Concept

| Traditional AI | Cognitive Conflict AI |
|---|---|
| Query → Answer | Query → Multi-perspective reasoning → Conflict → Resolution → Validated Answer |

---

## 🏗 Architecture

```
User Query
   ↓
Query Understanding Layer
   ↓
Multi-Agent Generation Layer
   ↓
Conflict Engine (Debate System)
   ↓
Re-ranking & Scoring Layer
   ↓
Judge / Decision Engine
   ↓
Confidence Estimation
   ↓
Final Output + Reasoning Trace
```

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/GypsianMonk/cognitive-conflict-ai.git
cd cognitive-conflict-ai

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp config/config.example.yaml config/config.yaml

# Run the API
uvicorn api.main:app --reload

# Or run the CLI
python main.py --query "Your question here"
```

---

## 📁 Project Structure

```
cognitive-conflict-ai/
│
├── agents/               # Individual AI agents (optimist, skeptic, alternative)
│   ├── base_agent.py
│   ├── optimist_agent.py
│   ├── skeptic_agent.py
│   ├── alternative_agent.py
│   └── expert_agent.py
│
├── engine/               # Core conflict & debate engine
│   ├── conflict_engine.py
│   ├── debate_manager.py
│   └── query_understanding.py
│
├── scoring/              # Re-ranking and evaluation
│   ├── scorer.py
│   └── metrics.py
│
├── memory/               # Past queries and learning
│   ├── memory_store.py
│   └── learning_adapter.py
│
├── retrieval/            # RAG - Retrieval Augmented Generation
│   ├── retriever.py
│   └── vector_store.py
│
├── api/                  # FastAPI backend
│   ├── main.py
│   ├── routes.py
│   └── schemas.py
│
├── ui/                   # Frontend interface
│   └── index.html
│
├── config/               # Configuration files
│   └── config.example.yaml
│
├── tests/                # Unit and integration tests
│   ├── test_agents.py
│   ├── test_engine.py
│   └── test_scoring.py
│
├── main.py               # CLI entry point
├── requirements.txt
└── README.md
```

---

## 🔷 System Components

### 1. Query Understanding Layer
- Analyzes query complexity and domain
- Decides reasoning depth (shallow / deep / expert)
- Routes to appropriate agent configuration

### 2. Multi-Agent Generation Layer

| Agent | Role |
|---|---|
| Agent A (Optimist) | Direct / Optimistic Answer |
| Agent B (Skeptic) | Critical / Challenges assumptions |
| Agent C (Alternative) | Lateral / Alternative Perspective |
| Expert Agent (Optional) | Domain-specific injected knowledge |

### 3. Conflict Engine
- Structured multi-round debate between agents
- Contradiction detection and argument refinement
- Configurable debate rounds (default: 3)

### 4. Scoring Layer
Evaluates responses on:
- ✅ Logical consistency
- ✅ Relevance to query
- ✅ Completeness
- ✅ Contradiction level

```python
score = (relevance * 0.4) + (coherence * 0.4) - (contradiction * 0.2)
```

### 5. Judge / Decision Engine
- Aggregates all perspectives
- Resolves conflicts via comparative reasoning
- Produces final answer with full reasoning trace

### 6. Confidence Estimation
- Confidence score (0–100%)
- Agreement level between agents
- Uncertainty indicators

---

## ⚙️ Configuration

```yaml
# config/config.yaml
model:
  provider: "ollama"         # ollama | openai | anthropic
  name: "mistral"
  temperature: 0.7

agents:
  count: 3
  debate_rounds: 3
  enable_expert_injection: true

scoring:
  relevance_weight: 0.4
  coherence_weight: 0.4
  contradiction_penalty: 0.2

memory:
  enabled: true
  max_history: 1000

retrieval:
  enabled: false
  vector_db: "chroma"
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
pytest tests/test_engine.py -v    # Engine tests only
pytest tests/ --cov=.             # With coverage
```

---

## 🔥 Advanced Features

- **Multi-Round Debate** — Agents iteratively refine answers across rounds
- **Dynamic Expert Injection** — Domain-specific agents added based on query topic
- **Memory & Learning** — Stores past conflicts to improve future reasoning
- **RAG Support** — Retrieval-augmented generation for factual grounding
- **Failure Detection** — Auto re-runs on low confidence or high contradiction

---

## 📊 Evaluation Metrics

| Metric | Description |
|---|---|
| Answer Quality | Relevance and completeness of final answer |
| Logical Consistency | Internal coherence of reasoning |
| Conflict Resolution Accuracy | Quality of resolving agent disagreements |
| Latency | End-to-end response time |

---

## 🛠 Tech Stack

- **Backend**: Python, FastAPI
- **LLM**: Ollama (local) / OpenAI / Anthropic
- **Vector DB**: ChromaDB (for RAG)
- **Memory**: SQLite / Redis
- **Testing**: pytest

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
