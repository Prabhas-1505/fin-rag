# 🚀 FIN-RAG Master Supervisor: Multi-Agent Financial RAG Pipeline

A production-grade, multi-agent Retrieval-Augmented Generation (RAG) system built to query unstructured financial reports (PDFs) and structured market metrics (SQL). 

Featuring a deterministic Python control loop, structured LLM routing contracts, a two-stage hybrid retrieval core, semantic caching, and a premium interactive dashboard interface.

---

## 🏗️ System Architecture

The project orchestrates two specialized worker agents through a structured Master Supervisor:

```mermaid
graph TD
    User([User Query]) --> Cache{Semantic Cache Check}
    Cache -- Hit (Similarity >= 95%) --> ReturnCached[Return Cached Answer]
    Cache -- Miss --> SupLoop[Supervisor Loop]
    
    subgraph Supervisor Loop (Max 5 Iterations)
        SupLoop --> Super[Supervisor Agent]
        Super --> Route{Router Decision}
        
        Route -- CALL_SQL_AGENT --> SQLAgent[SQL Specialist Agent]
        SQLAgent --> SQLTool[SQL Executor Tool]
        SQLTool --> SQLConn{Postgres Online?}
        SQLConn -- Yes --> PG[(Postgres DB)]
        SQLConn -- No/Fallback --> SQLite[(In-Memory SQLite)]
        PG & SQLite --> SQLResult[SQL Results]
        SQLResult --> SQLAgent
        SQLAgent --> StateUpdate[Append to Conversation History]
        
        Route -- CALL_PDF_AGENT --> PDFAgent[PDF Specialist Agent]
        PDFAgent --> PDFTool[PDF Search Tool]
        PDFTool --> Hybrid[Two-Stage Hybrid Search]
        Hybrid --> HybridConn{Qdrant Online?}
        HybridConn -- Yes --> Qdrant[(Qdrant Vector DB)]
        HybridConn -- No/Fallback --> MockPDF[Local BM25 + Mock Chunks]
        Qdrant & MockPDF --> RRF[Reciprocal Rank Fusion]
        RRF --> Rerank[Cross-Encoder Reranker]
        Rerank --> TopChunks[Top 4 Context Chunks]
        TopChunks --> PDFAgent
        PDFAgent --> StateUpdate
        
        StateUpdate --> SupLoop
        
        Route -- FINISH --> Synthesizer[Synthesize Final Response]
    end
    
    Synthesizer --> SetCache[Save in Semantic Cache]
    SetCache --> Response([Final Answer to User])
```

---

## 🌟 Key Features

*   **Deterministic Supervisor Loop**: Replaces unpredictable graph framework states with a clean, native Python loop protected by a hard-capped iteration limit to permanently block infinite execution loops.
*   **Structured Pydantic Routing**: Uses OpenAI structured output endpoints to enforce rigid JSON contracts, preventing routing hallucinations and invalid workers.
*   **Two-Stage Hybrid Retrieval Core**:
    1.  **Stage 1 (Retrieval)**: Executes concurrent dense vector search (Qdrant + `bge-large-en-v1.5`) and sparse keyword search (BM25), ranking candidates via **Reciprocal Rank Fusion (RRF)**.
    2.  **Stage 2 (Re-ranking)**: Passes candidate text chunks through a Cross-Encoder model (`bge-reranker-large`) to evaluate exact token relevance, narrowing search results to the top 4 highly accurate chunks.
*   **Built-in Production Fallbacks**: Automatically degrades gracefully during database connectivity losses. Falls back to **in-memory SQLite**, **local BM25 dict search**, and **RAM-based caching** if PostgreSQL, Qdrant, or Redis databases are offline.
*   **Redis Semantic Cache**: Embeds user queries and queries cached database records via Cosine similarity. Hits matching a $\ge 95\%$ threshold skip the agent loop entirely, cutting P95 latency down to milliseconds.
*   **Premium Interactive Web Dashboard**: Built with a dark glassmorphism theme to display agent routing steps, trace logs, database query returns, and cache hit metrics.

---

## 📂 Project Structure

```text
├── main.py                     # CLI entrypoint
├── requirements.txt            # Python dependencies
├── docker-compose.yml          # Container configuration (Qdrant, Postgres, Redis)
└── src/
    ├── agents/
    │   ├── tools.py            # LangChain tools with production fallbacks
    │   ├── specialists.py      # worker agents (PDF Research, SQL Analytics)
    │   └── supervisor_loop.py  # Master Supervisor orchestration
    ├── api/
    │   ├── redis_cache.py      # Semantic cache (Redis client & NumPy Cosine math)
    │   ├── server.py           # FastAPI backend
    │   └── static/             # Frontend Dashboard UI (HTML, CSS, JS)
    ├── retrieval/
    │   ├── hybrid_search.py    # Dense/Sparse RRF logic
    │   └── reranker.py         # Cross-Encoder re-ranker
    ├── ingestion/
    │   ├── database_setup.py   # Database seeder
    │   └── document_processor.py # PDF parsing & sliding window chunking
    └── evals/
        └── run_evals.py        # Automated RAGAS evaluation pipeline
```

---

## 🚀 Setup & Installation

### 1. Clone & Set Up Environment
```bash
# Clone the repository
git clone https://github.com/your-username/datascience-rag.git
cd datascience-rag

# Create virtual environment and install packages
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the root directory by copying the example template:
```bash
cp .env.example .env
```
Open `.env` and fill in your `OPENAI_API_KEY`.

### 3. Spin Up Database Infrastructure
Two supported paths, no Docker required for either:

**Option A — Homebrew (used in this project's actual verified setup):**
```bash
brew install postgresql@16 redis
brew services start postgresql@16
brew services start redis
psql postgres -c "CREATE ROLE admin WITH LOGIN PASSWORD 'password' SUPERUSER;"
psql postgres -c "CREATE DATABASE rag_db OWNER admin;"
PYTHONPATH=. python src/ingestion/database_setup.py  # creates tables
PYTHONPATH=. python src/ingestion/seed_real_financials.py  # seeds real quarterly data (5 companies, sourced from stockanalysis.com)
```
Qdrant runs in embedded local-file mode automatically (set via `QDRANT_LOCAL_PATH` in `.env`) — no separate install needed.

**Option B — Docker**, if you'd rather use the original `docker-compose.yml`:
```bash
docker-compose up -d
```

*(If neither Postgres nor Redis nor Qdrant is reachable, the project gracefully falls back to SQLite, local BM25 indexing, and in-memory caching — but the real, verified setup for this project uses Option A.)*

---

## 🏃 Running the Application

### Option A: Open the Interactive Web Dashboard (Recommended)
Start the FastAPI backend server:
```bash
PYTHONPATH=. python src/api/server.py
```
Or use uvicorn directly:
```bash
PYTHONPATH=. uvicorn src.api.server:app --reload --host 127.0.0.1 --port 8000
```
Open **[http://127.0.0.1:8000/](http://127.0.0.1:8000/)** in your web browser.

### Option B: Terminal CLI Chat
Start the interactive terminal session:
```bash
python main.py
```

---

## 🧪 Running Automated Evaluations (Ragas)
To assess Faithfulness (grounding) and Context Recall metrics using your golden test dataset, run:
```bash
PYTHONPATH=. python src/evals/run_evals.py
```
