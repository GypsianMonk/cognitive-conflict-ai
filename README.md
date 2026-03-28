# 🧠 Cognitive Conflict AI

> *Not generating answers faster — generating answers more intelligently through structured conflict.*

A **production-grade multi-agent AI reasoning system** built on a composable pipeline architecture. Multiple agents debate, challenge, and refine answers before a validated final output is produced.

---

## Architecture

```
Query
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│                     PIPELINE (17 steps)                      │
│                                                              │
│  Safety → Query Understanding → Knowledge Graph              │
│  → Task Decomposition → Tree of Thoughts → Build Agents      │
│  → Debate (Blind Round 1, Concurrent) → Self-Reflection      │
│  → Output Safety → Hallucination Detection → Fact Check      │
│  → Uncertainty Decomposition → Scoring → Graph of Thoughts   │
│  → Judge → Persona Evolution → Persist                       │
└─────────────────────────────────────────────────────────────┘
  │
  ▼
Validated Answer + Full Provenance
```

Each step is a discrete, testable unit. The pipeline stops cleanly on safety blocks and continues gracefully past step failures.

---

## Quick Start

```bash
git clone https://github.com/GypsianMonk/cognitive-conflict-ai.git
cd cognitive-conflict-ai
pip install -r requirements.txt
cp config/config.example.yaml config/config.yaml
```

**CLI:**
```bash
python main.py --query "Should I use Python or Rust for my backend?"
python main.py --query "Analyze microservices vs monolith" --verbose
python main.py --query "What is the best ML framework?" --json
```

**API:**
```bash
uvicorn api.main:app --reload
# Open ui/index.html for the live dashboard
```

---

## Configuration

Edit `config/config.yaml`:

```yaml
model:
  provider: "ollama"      # ollama | openai | anthropic | mock
  name: "mistral"
  temperature: 0.7
  max_retries: 3
  timeout: 60

agents:
  debate_rounds: 3
  enable_reflection: true
  reflection_threshold: 0.7
  enable_tools: true

scoring:
  relevance_weight: 0.4
  coherence_weight: 0.4
  contradiction_penalty: 0.2

memory:
  enabled: true
  kg_enabled: true
```

### Provider setup

**Ollama (local, free):**
```bash
ollama pull mistral
# Set provider: "ollama" in config
```

**OpenAI:**
```yaml
model:
  provider: "openai"
  name: "gpt-4o-mini"
  openai_api_key: "sk-..."
```

**Anthropic:**
```yaml
model:
  provider: "anthropic"
  name: "claude-haiku-4-5-20251001"
  anthropic_api_key: "sk-ant-..."
```

---

## Project Structure

```
cognitive-conflict-ai/
│
├── config.py                  # Validated, typed config (single source of truth)
├── main.py                    # CLI entry point
│
├── agents/
│   ├── base_agent.py          # Abstract base + multi-provider LLM with retry
│   ├── optimist_agent.py      # Direct, solution-oriented answers
│   ├── skeptic_agent.py       # Critical adversarial challenger
│   ├── alternative_agent.py   # Lateral thinker + domain expert
│   ├── reflective_agent.py    # Self-reflection loop (RBB-LLM)
│   ├── fact_checker.py        # Live claim verification
│   ├── tool_agent.py          # Calculator, Wikipedia, web search, Python exec
│   ├── persona_evolver.py     # Agents adapt from win/loss history
│   └── red_team_agent.py      # Adversarial vulnerability testing
│
├── engine/
│   ├── pipeline.py            # Pipeline runner + PipelineState
│   ├── steps.py               # 17 discrete, composable pipeline steps
│   ├── debate_manager.py      # Assembles and runs the pipeline
│   ├── query_understanding.py # Domain + complexity classification
│   ├── blind_first_round.py   # Anti-herding: concurrent blind round 1
│   ├── working_memory.py      # Shared inter-round whiteboard
│   ├── tot_engine.py          # Tree of Thoughts (branching + pruning)
│   ├── got_engine.py          # Graph of Thoughts (merge + split)
│   ├── task_decomposer.py     # Multi-step task breakdown
│   ├── safety_checker.py      # Injection guard + output monitoring
│   ├── conflict_engine.py     # Standalone debate (use directly if needed)
│   └── audit_logger.py        # Full decision provenance chain
│
├── scoring/
│   ├── scorer.py              # Weighted multi-metric response scoring
│   ├── hallucination_detector.py  # Cross-agent + spectral checks
│   ├── uncertainty.py         # Epistemic vs aleatoric decomposition
│   ├── preference_model.py    # User feedback → scoring weights
│   └── metrics.py             # System-level performance tracking
│
├── memory/
│   ├── memory_store.py        # SQLite query history
│   └── knowledge_graph.py     # Persistent cross-session validated facts
│
├── retrieval/
│   ├── hybrid_retriever.py    # Dense + BM25 sparse retrieval
│   └── temporal_filter.py     # Recency-aware source scoring
│
├── api/
│   └── main.py                # FastAPI: 15+ endpoints
│
├── ui/
│   └── index.html             # Real-time debate dashboard
│
├── tests/
│   ├── test_pipeline.py       # Pipeline + integration tests (22 tests)
│   ├── test_agents.py         # Agent unit tests
│   ├── test_engine.py         # Engine unit tests
│   └── test_scoring.py        # Scoring unit tests
│
└── config/
    └── config.example.yaml
```

---

## Features

| Feature | Module | Description |
|---|---|---|
| Anti-herding | `blind_first_round.py` | Round 1 agents run in parallel with no shared context |
| Concurrent agents | `blind_first_round.py` | `ThreadPoolExecutor` — all agents run simultaneously |
| Tree of Thoughts | `tot_engine.py` | Branch + prune reasoning paths before debate |
| Graph of Thoughts | `got_engine.py` | Merge agent responses into a synthesis graph |
| Self-reflection | `reflective_agent.py` | Agents critique and refine their own outputs |
| Hallucination detection | `hallucination_detector.py` | Cross-agent factual consistency checks |
| Uncertainty decomposition | `uncertainty.py` | Epistemic vs aleatoric uncertainty |
| Fact checking | `fact_checker.py` | Live claim verification via Wikipedia + web |
| Tool use | `tool_agent.py` | Calculator, Python exec, Wikipedia, web search |
| Persona evolution | `persona_evolver.py` | Agents learn from win/loss history per domain |
| Knowledge graph | `knowledge_graph.py` | Persistent validated facts across sessions |
| Hybrid RAG | `hybrid_retriever.py` | Dense + sparse retrieval with trust scoring |
| Temporal awareness | `temporal_filter.py` | Source recency + domain-specific staleness flags |
| Preference learning | `preference_model.py` | User ratings update scoring weights |
| Safety guard | `safety_checker.py` | Injection detection on both input and output |
| Audit logging | `audit_logger.py` | Full JSON-lines provenance trail |
| Red team agent | `red_team_agent.py` | Adversarial probing for vulnerabilities |
| Task decomposition | `task_decomposer.py` | Complex queries broken into ordered subtasks |
| Working memory | `working_memory.py` | Shared premises/conclusions across rounds |
| Pipeline architecture | `pipeline.py` + `steps.py` | Composable, testable, fail-graceful steps |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/query` | Run the full debate pipeline |
| `POST` | `/feedback` | Submit answer rating (trains scoring weights) |
| `GET` | `/feedback/stats` | Preference model statistics |
| `GET` | `/memory/history` | Past query history |
| `GET` | `/memory/search?q=` | Search past queries |
| `GET` | `/knowledge-graph/stats` | KG node/edge counts |
| `GET` | `/knowledge-graph/search?q=` | Search validated facts |
| `POST` | `/knowledge-graph/add` | Manually add a fact |
| `GET` | `/metrics` | System performance metrics |
| `GET` | `/agents/leaderboard` | Agent win rates by domain |
| `GET` | `/audit/session` | Current session audit log |
| `POST` | `/red-team/run` | Run adversarial vulnerability tests |
| `GET` | `/health` | Health + active features list |

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Pipeline + integration (no LLM needed — uses mock provider)
pytest tests/test_pipeline.py -v

# With coverage
pytest tests/ --cov=. --cov-report=term-missing
```

---

## License

MIT — see [LICENSE](LICENSE) for details.
