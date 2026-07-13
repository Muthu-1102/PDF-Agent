# Agentic RAG System (LangGraph)

An Agentic Retrieval-Augmented Generation system built to demonstrate
**LangGraph orchestration, multi-agent collaboration, and evidence-verified
answer generation** — not infrastructure plumbing.

Traditional RAG: `Retrieve -> Generate -> Return`

This project: `Retrieve -> Evaluate Evidence -> Retry if Needed -> Generate -> Return`

See [`Architecture.txt`](./Architecture.txt) for the full design and
[`docs/ROADMAP.md`](./docs/ROADMAP.md) for the step-by-step build plan.

## Stack

| Layer | Tech |
|---|---|
| Agent orchestration | LangGraph |
| LLM | Groq API |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector store | ChromaDB (embedded, file-persisted — no separate DB service) |
| Backend API | FastAPI |
| Frontend | React (minimal — this project's focus is the agentic backend) |

## Project layout

```
backend/
  app/
    api/            FastAPI routes (thin, stateless)
    core/            config
    ingestion/       loader + chunker
    vectorstore/     embedded Chroma client
    graph/           <-- the interesting part
      state.py       LangGraph state schema
      nodes.py       retrieve / evidence / retry / report nodes
      orchestrator.py  builds + compiles the StateGraph, run_query()
      agents/
        evidence_agent.py
        report_agent.py
    llm/             Groq client
    main.py
frontend/            minimal React chat UI
docs/
  ROADMAP.md         step-by-step build plan
```

## Quickstart

```bash
# Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your GROQ_API_KEY
uvicorn app.main:app --reload

# Frontend (separate terminal)
cd frontend
npm install
npm run dev
```

Upload a PDF/DOCX/TXT, then ask a question. Watch the terminal / add
logging in `orchestrator.py` to see the retrieve → evidence → retry loop
in action.
