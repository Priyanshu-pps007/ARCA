<div align="center">

<img src="https://img.shields.io/badge/status-in%20development-blue?style=for-the-badge" />
<img src="https://img.shields.io/badge/python-3.11+-brightgreen?style=for-the-badge&logo=python" />
<img src="https://img.shields.io/badge/LangGraph-powered-orange?style=for-the-badge" />
<img src="https://img.shields.io/badge/license-MIT-lightgrey?style=for-the-badge" />

<br /><br />

```
   █████╗ ██████╗  ██████╗ █████╗
  ██╔══██╗██╔══██╗██╔════╝██╔══██╗
  ███████║██████╔╝██║     ███████║
  ██╔══██║██╔══██╗██║     ██╔══██║
  ██║  ██║██║  ██║╚██████╗██║  ██║
  ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝
```

### **The AI Agent Runtime with Semantic Memory**
*Natural language → deployed AI agent in under 60 seconds*

[Getting Started](#-getting-started) · [Architecture](#-architecture) · [Memory Layer](#-the-memory-layer) · [Roadmap](#-roadmap) · [Contributing](#-contributing)

</div>

---

## What is ARCA?

Most AI agent platforms give you **execution**. ARCA gives you **execution + memory**.

Every time your agent runs, it learns. It stores what worked, what failed, and what it discovered — and retrieves that knowledge on the next run. The result: agents that compound intelligence over time, like a senior engineer who gets better with every project.

```bash
# From natural language to a live, deployed agent
$ agentctl create "Monitor my GitHub PRs and summarize them every morning"

✓ Understanding intent...
✓ Building agent graph...
✓ Deploying runtime...
✓ Agent live in 47 seconds

Agent ID: agt_pr_monitor_x7k2
Endpoint: POST /agents/agt_pr_monitor_x7k2/run
```

---

## Why ARCA?

| | Claude Code | LangGraph Cloud | ARCA |
|---|:---:|:---:|:---:|
| Natural language → agent | ✅ | ❌ | ✅ |
| Managed deployment | ❌ | ✅ | ✅ |
| Semantic memory across runs | ❌ | ❌ | ✅ |
| Per-tenant memory isolation | ❌ | ❌ | ✅ |
| Open source | ✅ | ❌ | ✅ |
| Self-hostable | ✅ | ❌ | ✅ |

The key insight: **agents should get smarter with every run, not reset.**

---

## Architecture

ARCA is built in 4 layers, each with a single responsibility:

```
┌─────────────────────────────────────────────────────────────┐
│                        INTERFACE LAYER                       │
│              CLI (agentctl)  +  REST API                     │
├─────────────────────────────────────────────────────────────┤
│                        BUILDER LAYER                         │
│     Natural Language → agent.json (Pydantic schema)         │
│     RAG Pipeline (BGE-M3 + pgvector) + GPT-4o + LangGraph   │
├─────────────────────────────────────────────────────────────┤
│                        RUNTIME LAYER                         │
│         LangGraph Executor + Celery Workers + Docker         │
├─────────────────────────────────────────────────────────────┤
│                        MEMORY LAYER                          │
│    PostgreSQL (state)  +  Redis (cache)  +  pgvector (semantic) │
└─────────────────────────────────────────────────────────────┘
```

### Layer Breakdown

**Builder** — Translates natural language into a validated `agent.json` schema using a RAG pipeline (Playwright crawler → BGE-M3 embeddings → pgvector retrieval) + GPT-4o with retry loops. The agent graph is then constructed dynamically via LangGraph.

**Runtime** — Each agent runs in an isolated Docker container. Celery workers handle async task execution and scheduling. LangGraph manages the agent's internal state machine.

**Memory** — The core differentiator. Three types of memory per agent:
- `episodic` → what happened during each run
- `semantic` → what the agent learned (abstracted)
- `procedural` → what tool call patterns worked

**Interface** — `agentctl` CLI for developers. REST API for integrations.

---

## The Memory Layer

This is what separates ARCA from everything else.

**The problem with existing runtimes:**
```
Run 1:  Agent executes → finishes → state discarded
Run 2:  Agent starts from zero → same mistakes → same cost
Run N:  Still starting from zero
```

**The ARCA approach:**
```
Run 1:  Agent executes → learnings extracted → embedded → stored in pgvector
Run 2:  Task arrives → query embedded → top-K similar memories retrieved
        → injected into context → agent starts informed
Run N:  Agent has accumulated knowledge → better outputs → lower token cost
```

```python
# Memory write path (after every run)
async def form_memories(trace: ExecutionTrace, agent_id: str):
    learnings = await extract_learnings(trace)        # LLM distillation
    vectors   = embedder.encode(learnings)            # BGE-M3
    await pgvector.insert(
        agent_id    = agent_id,
        content     = learnings,
        vector      = vectors,
        memory_type = "semantic",                     # episodic | semantic | procedural
        run_id      = trace.run_id,
    )

# Memory read path (before every run)
async def recall(task: str, agent_id: str, top_k: int = 5):
    query_vec = embedder.encode(task)
    return await pgvector.similarity_search(
        vector   = query_vec,
        agent_id = agent_id,                          # tenant-isolated
        top_k    = top_k,                             # IVFFlat index — O(log n)
    )
```

At 100K stored memories, retrieval stays fast because of the **IVFFlat index on pgvector** — cost stays flat while quality compounds.

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| API | FastAPI | Async-first, production-grade |
| Agent graphs | LangGraph | Stateful, cyclical agent workflows |
| Task queue | Celery + Redis | Async execution at scale |
| Vector store | pgvector (IVFFlat) | O(log n) retrieval, same DB as state |
| Embeddings | BGE-M3 | Best-in-class multilingual embeddings |
| State storage | PostgreSQL | 14-table schema with multi-tenancy + RLS |
| Containers | Docker | Isolated agent runtimes |
| LLM | GPT-4o (builder) | Reliable schema generation with retry loops |

---

## Getting Started

```bash
# Clone the repo
git clone https://github.com/Priyanshu-pps007/arca.git
cd arca

# Set up environment
cp .env.example .env
# Add your OPENAI_API_KEY and DATABASE_URL

# Start the stack
docker-compose up -d

# Install CLI
pip install -e .

# Create your first agent
agentctl create "Summarize my top 5 Hacker News posts every morning"
```

> **Requirements:** Python 3.11+, Docker, PostgreSQL with pgvector extension

---

## Project Structure

```
arca/
├── builder/
│   ├── rag/              # Playwright crawler + BGE-M3 + pgvector
│   ├── schema/           # agent.json Pydantic models
│   └── graph.py          # LangGraph builder with retry loops
├── runtime/
│   ├── executor.py       # LangGraph execution engine
│   ├── workers/          # Celery task workers
│   └── docker/           # Agent container templates
├── memory/
│   ├── store.py          # pgvector read/write
│   ├── embedder.py       # BGE-M3 wrapper
│   └── recall.py         # Similarity search + injection
├── interface/
│   ├── api/              # FastAPI REST endpoints
│   └── cli/              # agentctl commands
└── db/
    └── migrations/       # 14-table PostgreSQL schema
```

---

## Roadmap

- [x] PostgreSQL schema (14 tables, RLS, multi-tenancy)
- [x] Celery runtime + Redis task queue
- [x] RAG pipeline (Playwright + BGE-M3 + pgvector)
- [x] agent.json Pydantic schema
- [x] LangGraph builder layer with GPT-4o + retry loops
- [ ] LangGraph executor (replacing stub)
- [ ] Memory write path (post-run learning)
- [ ] `agentctl` CLI polish
- [ ] MCP registry integration
- [ ] HackerNews open-source launch

---

## The Bigger Picture

ARCA is built on a simple thesis:

> *The gap between a working AI agent and a useful AI agent is memory.*

A customer support agent that remembers your product's edge cases. A research agent that builds on what it found last week. A coding agent that learns your codebase conventions. None of this is possible when agents reset after every run.

ARCA's semantic memory layer is the infrastructure that makes agents actually get smarter — the same way a senior employee compounds knowledge over years, not a contractor who starts fresh every engagement.

---

## Contributing

ARCA is in active development. If you're interested in:
- AI agent infrastructure
- Vector search and memory systems
- LangGraph internals
- Production MLOps

...open an issue or reach out directly.

---

## Author

**Priyanshu Pratap Singh** — Backend & AI Engineer

Built production multi-agent systems at [Svahnar](http://svahnar.com). ARCA is the distillation of everything learned building LangGraph + MCP + RAG systems in production.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/priyanshu-pratap-singh-84bb692b7)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/Priyanshu-pps007)

---

<div align="center">
<sub>Built with obsession. Designed to compound.</sub>
</div>
