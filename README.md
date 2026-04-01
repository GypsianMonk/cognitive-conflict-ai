<div align="center">

<!-- Animated title using SVG -->
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=32&pause=1000&color=7C6AF7&center=true&vCenter=true&width=700&lines=🧠+Cognitive+Conflict+AI;Multi-Agent+Adversarial+Reasoning;Not+faster+answers.+Smarter+ones." alt="Typing SVG" />

<br/>

<img src="https://img.shields.io/badge/version-3.0.0-7C6AF7?style=for-the-badge&logo=git&logoColor=white"/>
<img src="https://img.shields.io/badge/python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/pipeline-17%20steps-3ecf8e?style=for-the-badge&logo=circleci&logoColor=white"/>
<img src="https://img.shields.io/badge/tests-22%20passing-3ecf8e?style=for-the-badge&logo=pytest&logoColor=white"/>
<img src="https://img.shields.io/badge/TurboQuant-Google%20Research-f59e0b?style=for-the-badge&logo=google&logoColor=white"/>
<img src="https://img.shields.io/badge/license-MIT-blue?style=for-the-badge"/>

<br/><br/>

> *"Not generating answers faster — generating answers more intelligently through conflict."*

<br/>

</div>

---

## ⚡ What is this?

Most AI systems do this:

```
Query ──────────────────────────────────► Answer
```

**Cognitive Conflict AI** does this:

```
Query
  │
  ▼
🛡️  Safety Check          ──── blocks injections & harmful queries
  │
  ▼
🔍 Query Understanding    ──── domain, complexity, debate depth
  │
  ▼
🧠 Knowledge Graph        ──── inject validated past facts
  │
  ▼
🌳 Tree of Thoughts       ──── explore branching reasoning paths
  │
  ▼
⚔️  Multi-Agent Debate     ──── 3 agents argue concurrently (blind round 1)
  │           ┌──────────────────────────────────────┐
  │           │  🟢 Optimist  →  direct answer        │
  │           │  🔴 Skeptic   →  challenges it         │
  │           │  🔵 Alt View  →  lateral thinking      │
  │           │  🟡 Expert    →  domain injection      │
  │           └──────────────────────────────────────┘
  │
  ▼
🔁 Self-Reflection        ──── agents critique their own outputs
  │
  ▼
👁️  Hallucination Check   ──── cross-agent factual consistency
  │
  ▼
✅ Fact Verification      ──── live claim checking via web/Wikipedia
  │
  ▼
📊 Uncertainty Split      ──── epistemic vs aleatoric decomposition
  │
  ▼
🏆 Judge + Score          ──── preference-weighted ranking
  │
  ▼
📜 Audit + Persist        ──── full provenance, memory, KG update
  │
  ▼
Validated Answer + Confidence + Full Reasoning Trace
```

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/GypsianMonk/cognitive-conflict-ai.git
cd cognitive-conflict-ai

# 2. Install
pip install -r requirements.txt

# 3. Configure
cp config/config.example.yaml config/config.yaml
# Edit config.yaml — choose your LLM provider (see below)

# 4a. CLI
python main.py --query "Should I use Python or Rust for my backend?"
python main.py --query "Analyze microservices vs monolith" --verbose
python main.py --query "What is the best ML framework?" --json

# 4b. API + Dashboard
uvicorn api.main:app --reload
# Then open ui/index.html in your browser
```

---

## 🔌 LLM Provider Setup

<details>
<summary><b>🦙 Ollama (local, free — recommended)</b></summary>

```bash
# Install Ollama: https://ollama.com
ollama pull mistral         # 4GB — good quality
ollama pull phi3:mini       # 1.6GB — fast, for simple queries
```

```yaml
# config/config.yaml
model:
  provider: "ollama"
  name: "mistral"
  ollama_base_url: "http://localhost:11434"
```

</details>

<details>
<summary><b>🤖 OpenAI</b></summary>

```yaml
model:
  provider: "openai"
  name: "gpt-4o-mini"
  openai_api_key: "sk-..."
```

</details>

<details>
<summary><b>🟣 Anthropic</b></summary>

```yaml
model:
  provider: "anthropic"
  name: "claude-haiku-4-5-20251001"
  anthropic_api_key: "sk-ant-..."
```

</details>

<details>
<summary><b>🧪 Mock (no LLM — for testing)</b></summary>

```yaml
model:
  provider: "mock"
```

All 22 tests pass with mock — no LLM required.

</details>

---

## 🏗️ Architecture

### Pipeline — 17 Composable Steps

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PIPELINE                                     │
│                                                                      │
│  [1] safety           Block injections & harmful queries             │
│  [2] query_understanding   Domain, complexity, debate depth          │
│  [3] knowledge_graph  Inject validated cross-session facts           │
│  [4] task_decomposition    Break complex queries into subtasks       │
│  [5] tree_of_thoughts      Explore branching reasoning paths         │
│  [6] build_agents     Instantiate 1-4 agents based on complexity     │
│  [7] debate           Concurrent blind round 1, then open debate     │
│  [8] reflection       Agents critique their own outputs              │
│  [9] output_safety    Safety check all agent outputs                 │
│  [10] hallucination   Cross-agent factual consistency checks         │
│  [11] fact_check      Live claim verification via web/Wikipedia      │
│  [12] uncertainty     Epistemic vs aleatoric decomposition           │
│  [13] scoring         Preference-weighted multi-metric scoring       │
│  [14] graph_of_thoughts    Synthesise agent ideas into a graph       │
│  [15] judge           Select winner, build reasoning trace           │
│  [16] persona_evolution    Update agent win/loss history             │
│  [17] persist         Save to memory, KG, audit log, metrics         │
│                                                                      │
│  Each step is discrete, testable, and fails gracefully.              │
│  A blocked step never crashes the pipeline.                          │
└─────────────────────────────────────────────────────────────────────┘
```

### Project Structure

```
cognitive-conflict-ai/
│
├── 📄 config.py                   Validated, typed AppConfig
├── 📄 main.py                     CLI entry point
│
├── 🤖 agents/
│   ├── base_agent.py              Multi-provider LLM + retry logic
│   ├── optimist_agent.py          Direct, solution-oriented
│   ├── skeptic_agent.py           Critical adversarial challenger
│   ├── alternative_agent.py       Lateral thinker + domain expert
│   ├── reflective_agent.py        Self-reflection loop (RBB-LLM 2025)
│   ├── fact_checker.py            Live claim verification
│   ├── tool_agent.py              Calculator, Wikipedia, web, Python
│   ├── persona_evolver.py         Agents learn from win/loss history
│   └── red_team_agent.py          Adversarial vulnerability testing
│
├── ⚙️  engine/
│   ├── pipeline.py                Pipeline runner + PipelineState
│   ├── steps.py                   17 discrete, composable steps
│   ├── debate_manager.py          Assembles and runs the pipeline
│   ├── blind_first_round.py       Anti-herding: ThreadPoolExecutor
│   ├── working_memory.py          Shared inter-round whiteboard
│   ├── tot_engine.py              Tree of Thoughts (TotNode)
│   ├── got_engine.py              Graph of Thoughts (GotNode)
│   ├── task_decomposer.py         Multi-step task breakdown
│   ├── safety_checker.py          Injection guard + output monitor
│   ├── query_understanding.py     Domain + complexity detection
│   └── audit_logger.py            JSON-lines full provenance
│
├── 📊 scoring/
│   ├── scorer.py                  Weighted multi-metric scoring
│   ├── hallucination_detector.py  Cross-agent + spectral checks
│   ├── uncertainty.py             Epistemic vs aleatoric split
│   ├── preference_model.py        User feedback → scoring weights
│   └── metrics.py                 System performance tracking
│
├── 🧠 memory/
│   ├── memory_store.py            SQLite query history
│   └── knowledge_graph.py         Persistent cross-session facts
│
├── 🔎 retrieval/
│   ├── turbo_quant.py             TurboQuant (Google Research 2025)
│   ├── hybrid_retriever.py        Dense (TQ-compressed) + BM25
│   └── temporal_filter.py         Recency-aware source scoring
│
├── 🌐 api/
│   └── main.py                    FastAPI: 15+ endpoints
│
├── 🖥️  ui/
│   └── index.html                 Real-time debate dashboard
│
└── 🧪 tests/
    ├── test_pipeline.py            22 tests, 0 LLM calls needed
    ├── test_agents.py
    ├── test_engine.py
    └── test_scoring.py
```

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🔒 Safety & Reliability
- **Prompt injection guard** — 16 pattern classes blocked
- **Output safety checker** — monitors all agent responses
- **Anti-herding protocol** — blind round 1 prevents groupthink
- **Hallucination detection** — cross-agent consistency (AUROC 0.76–0.92)
- **Graceful failure** — pipeline continues past step errors
- **Audit logging** — full JSON-lines provenance trail

</td>
<td width="50%">

### 🧠 Reasoning
- **Tree of Thoughts** — branch + prune reasoning paths
- **Graph of Thoughts** — non-linear idea merging
- **Self-reflection** — RBB-LLM framework (−40% hallucination)
- **Uncertainty decomposition** — epistemic vs aleatoric split
- **Task decomposition** — ordered subtask execution
- **Concurrent agents** — ThreadPoolExecutor, all agents parallel

</td>
</tr>
<tr>
<td width="50%">

### 💾 Memory & Retrieval
- **Knowledge graph** — persistent cross-session validated facts
- **Hybrid RAG** — TurboQuant dense + BM25 sparse retrieval
- **TurboQuant compression** — 8–16× memory reduction
- **Temporal filtering** — domain-specific staleness scoring
- **SQLite history** — keyword + similarity search
- **Transactive memory** — agents share what they know, not what they said

</td>
<td width="50%">

### 📈 Learning & Observability
- **Persona evolution** — agents improve from win/loss history
- **Preference learning** — user feedback trains scoring weights
- **Agent leaderboard** — win rates by domain
- **Red team agent** — continuous adversarial probing
- **15+ API endpoints** — full REST observability
- **Real-time dashboard** — watch the debate live

</td>
</tr>
</table>

---

## 🔬 TurboQuant — Google Research Integration

This system implements **TurboQuant** from Google Research / Google DeepMind:

> *"TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"*
> Zandieh, Daliri, Hadian, Mirrokni — arXiv:2504.19874 (April 2025)

### The Algorithm

```
Input vector x ∈ Rᵈ
        │
        ▼
┌─────────────────────┐
│  Random rotation Π  │  ← makes any vector uniform on unit hypersphere
│  y = Π · x          │    each coordinate follows Beta distribution
└─────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  Lloyd-Max scalar quantisation per coordinate        │
│  idx_j = argmin_k |y_j - c_k|                       │  ← optimal for Beta dist
│  centroids pre-computed via k-means on Eq.(4)        │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐   TurboQuantProd only
│  QJL on residual  r = x - x̂_mse                    │
│  qjl = sign(S · r)  where S ~ N(0,1)                │  ← makes estimator unbiased
│  Guarantees: E[⟨y, x̂⟩] = ⟨y, x⟩                   │
└─────────────────────────────────────────────────────┘
        │
        ▼
  2–4 bits per coordinate (vs 32 bits FP32)
```

### Validated Against Paper Theorems

| Bit-width | Paper MSE | Measured MSE | Status |
|---|---|---|---|
| b=1 | ≈ 0.360 | 0.360 | ✅ |
| b=2 | ≈ 0.117 | 0.115 | ✅ |
| b=3 | ≈ 0.030 | 0.033 | ✅ |
| b=4 | ≈ 0.009 | 0.009 | ✅ |
| Bias (TurboQuantProd) | 0.000 | < 0.001 | ✅ |

Distortion upper bound: **Dmse ≤ √(3π/2) · 4⁻ᵇ** — within 2.7× of the Shannon information-theoretic lower bound (Theorem 3).

### Memory Impact at Scale

| Precision | 1M × 1536-dim vectors | Compression | Quality |
|---|---|---|---|
| FP32 (baseline) | 5,859 MB | 1× | Full |
| **4-bit TurboQuant** | **736 MB** | **8×** | ✅ Quality neutral (LongBench score: 50.06 vs 50.06) |
| 3-bit TurboQuant | 553 MB | 10.6× | Marginal drop |
| 2-bit TurboQuant | 370 MB | 15.8× | Small drop |

### Indexing Speed vs Alternatives (100K vectors, 4-bit, d=1536)

| Method | Time | Notes |
|---|---|---|
| Product Quantisation | 239.75s | Requires k-means training |
| RabitQ | 2,267.59s | No GPU vectorisation |
| **TurboQuant** | **0.0013s** | Data-oblivious, zero training |

**185,000× faster** than Product Quantisation — no training, no calibration, instant indexing.

### Usage

```python
from retrieval.turbo_quant import TurboQuantProd
import numpy as np

tq = TurboQuantProd(dim=1536, bit_width=4)

# Compress a document vector (store this, not the float array)
doc_vec = np.random.randn(1536).astype(np.float32)
q = tq.quantise(doc_vec)          # ~6x less memory than float32

# Unbiased similarity — no dequantisation needed
query_vec = np.random.randn(1536).astype(np.float32)
score = tq.inner_product(query_vec, q)   # E[score] = true inner product
```

---

## 🌐 API Reference

```bash
# Start the server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/query` | Run full 17-step debate pipeline |
| `POST` | `/feedback` | Submit rating → trains scoring weights |
| `GET` | `/feedback/stats` | Preference model statistics |
| `GET` | `/memory/history` | Past query history |
| `GET` | `/memory/search?q=` | Semantic search past queries |
| `GET` | `/knowledge-graph/stats` | KG node + edge counts |
| `GET` | `/knowledge-graph/search?q=` | Search validated facts |
| `POST` | `/knowledge-graph/add` | Manually add a fact node |
| `GET` | `/metrics` | System performance metrics |
| `GET` | `/agents/leaderboard` | Agent win rates by domain |
| `GET` | `/audit/session` | Current session full audit log |
| `POST` | `/red-team/run` | Run adversarial vulnerability tests |
| `GET` | `/health` | Health check + active features list |

<details>
<summary><b>Example: Full query response</b></summary>

```json
{
  "query": "Should I use Python or Rust?",
  "final_answer": "...",
  "confidence_score": "76.2%",
  "total_debate_rounds": 2,
  "convergence_achieved": true,
  "domain": "technology",
  "complexity": "complex",
  "hallucination_risk": 0.042,
  "uncertainty": {
    "total_uncertainty": 0.18,
    "epistemic": 0.15,
    "aleatoric": 0.03,
    "dominant_type": "epistemic",
    "recommendation": "Knowledge gap — recommend retrieving additional sources."
  },
  "fact_check": {
    "overall_credibility": 0.91,
    "verified_count": 3,
    "unverified_count": 1,
    "disputed_count": 0
  },
  "got_summary": {
    "total_nodes": 5,
    "merge_nodes": 1,
    "cycles_detected": 0
  },
  "agent_responses": [...],
  "pipeline_errors": []
}
```

</details>

---

## ⚙️ Configuration

```yaml
# config/config.yaml

model:
  provider: "ollama"          # ollama | openai | anthropic | mock
  name: "mistral"
  temperature: 0.7
  max_retries: 3              # retry on LLM failure
  timeout: 60                 # seconds per LLM call
  max_tokens: 1024

agents:
  debate_rounds: 3            # max rounds (stops early on convergence)
  enable_reflection: true     # RBB-LLM self-critique loop
  reflection_threshold: 0.7  # only reflect if confidence < this
  enable_expert_injection: true
  enable_tools: true          # calculator, wikipedia, web, python

scoring:
  relevance_weight: 0.4
  coherence_weight: 0.4
  contradiction_penalty: 0.2

memory:
  enabled: true
  max_history: 1000
  kg_enabled: true            # knowledge graph
```

---

## 🧪 Testing

```bash
# All tests (22 tests, runs in < 1 second, no LLM needed)
pytest tests/ -v

# Pipeline + integration only
pytest tests/test_pipeline.py -v

# With coverage
pytest tests/ --cov=. --cov-report=term-missing
```

Tests use the `mock` provider — no LLM, no network, instant results.

What's covered:

- Config validation and defaults
- Pipeline state and step ordering
- Safety blocking (injections, harmful content)
- Domain detection and complexity classification
- Multi-round debate execution
- All 14 output fields present
- GoT summary, uncertainty decomposition
- Task decomposition triggering
- Legacy dict config backward compatibility

---

## 📊 Benchmarks

| Query type | Agents | Rounds | Avg latency (Ollama Mistral 7B) |
|---|---|---|---|
| Simple | 1 | 1 | ~3s |
| Medium | 2 | 2 | ~12s |
| Complex | 3–4 | 3 | ~28s |
| Injections | 0 | 0 | < 1ms (blocked) |

All agents run **concurrently** via `ThreadPoolExecutor`. 3 agents in parallel = same wall time as 1 agent.

---

## 🤝 Contributing

```bash
# Fork → clone → branch
git checkout -b feature/your-feature

# Make changes, run tests
pytest tests/ -v

# Commit with context
git commit -m "feat: description of what and why"

# Push + PR
git push origin feature/your-feature
```

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with adversarial reasoning, Google Research quantisation, and 2025 multi-agent literature.**

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=14&pause=2000&color=7C6AF7&center=true&vCenter=true&width=600&lines=The+difference+between+a+chatbot+and+a+thinking+system.;Query+→+Debate+→+Conflict+→+Resolution+→+Truth." alt="Footer typing" />

</div>
