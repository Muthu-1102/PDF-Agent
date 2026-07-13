# Build Roadmap — Agentic RAG (LangGraph)

Scaffolding for all steps below already exists as temp/skeleton code in
this repo. Work through the phases in order — each one is independently
runnable/testable before you move to the next, so you always have
something working to demo.

Focus areas to showcase: **LangGraph orchestration, agent design,
retrieval-retry reasoning, evidence-grounded generation.**
Deliberately out of scope: complex DB routing/microservices — Chroma runs
embedded, in-process.

---

## Phase 0 — Environment Setup
- [ ] `cd backend && python -m venv .venv && source .venv/bin/activate`
- [ ] `pip install -r requirements.txt`
- [ ] Get a free Groq API key at console.groq.com, copy `.env.example` → `.env`
- [ ] Sanity check: `uvicorn app.main:app --reload` then hit `GET /health`

**Done when:** server boots, `/health` returns `{"status": "ok"}`.

---

## Phase 1 — Ingestion Pipeline (Upload Workflow)
Files: `app/ingestion/loader.py`, `app/ingestion/chunker.py`,
`app/vectorstore/chroma_client.py`, `app/api/routes_upload.py`

- [ ] Verify PDF text extraction (`load_pdf`) against a real multi-page PDF
- [ ] Verify DOCX and TXT loaders
- [ ] Tune `CHUNK_SIZE_WORDS` / `CHUNK_OVERLAP_WORDS` in `core/config.py`
      for your typical document type
- [ ] Confirm chunks land in Chroma with correct metadata
      (document_name, page_number, chunk_index, upload_timestamp)
- [ ] Test via `curl -F "file=@sample.pdf" http://localhost:8000/upload/`

**Done when:** you can upload a file and see `chunks_indexed > 0` in the
response, and inspect `backend/data/chroma_store/` growing on disk.

---

## Phase 2 — LangGraph State & Nodes
Files: `app/graph/state.py`, `app/graph/nodes.py`

- [ ] Read through `RAGState` — this is the contract every node shares
- [ ] Implement/verify `retrieve_node`: queries Chroma, populates
      `retrieved_chunks` + `similarity_scores`
- [ ] Implement/verify `retry_node`: increments `retry_count`, widens
      `current_top_k` (capped at `MAX_TOP_K`)
- [ ] Write a standalone test script that builds a fake `RAGState` and
      runs `retrieve_node` in isolation against your indexed docs

**Done when:** `retrieve_node` returns real chunks + scores for a
sample question, independent of the rest of the graph.

---

## Phase 3 — Evidence Agent
Files: `app/graph/agents/evidence_agent.py`

- [ ] Tune `SIMILARITY_THRESHOLD` and `MIN_TRUSTED_CHUNKS` in
      `core/config.py` against real queries — this is the knob that
      controls how "picky" the system is before it answers
- [ ] Log `evidence_status` for a batch of test questions (some
      in-KB, some deliberately out-of-KB) and confirm it correctly
      flags "insufficient" for the latter
- [ ] (Stretch) Add an LLM-judged relevance re-rank step here instead of
      pure cosine-similarity filtering

**Done when:** questions answerable from your docs return
`evidence_status: "sufficient"`; unrelated questions return
`"insufficient"`.

---

## Phase 4 — Report Agent
Files: `app/graph/agents/report_agent.py`, `app/llm/groq_client.py`

- [ ] Confirm Groq call works end-to-end (check `.env` key)
- [ ] Review `REPORT_PROMPT` — this is where citation discipline is
      enforced ("only answer from evidence, cite chunk ids")
- [ ] Verify the zero-evidence fallback path (empty `trusted_chunks` ->
      canned "insufficient evidence" answer, confidence 0.0)
- [ ] Confirm `confidence` score and `sources` list are populated
      correctly from `trusted_chunks`

**Done when:** a well-formed answer comes back with inline citations
and a matching `sources` array.

---

## Phase 5 — Orchestrator (wire the graph)
Files: `app/graph/orchestrator.py`

- [ ] Walk through `build_graph()` against Architecture.md section 8 —
      confirm the conditional edge (`route_on_evidence`) actually loops
      `evidence -> retry -> retrieve -> evidence` on insufficient evidence
- [ ] Add a `print`/log statement in each node temporarily to watch the
      loop execute for a deliberately hard/ambiguous question
- [ ] Confirm the `MAX_RETRIES` circuit breaker stops infinite loops
- [ ] (Optional but great for demos) Use `graph.get_graph().draw_mermaid()`
      to export a visual of the compiled LangGraph and drop it in
      `docs/graph.png` for your README/portfolio

**Done when:** you can watch a question retry retrieval 1–2 times
before either finding evidence or gracefully giving up.

---

## Phase 6 — API Wiring
Files: `app/api/routes_query.py`, `app/main.py`

- [ ] `curl -X POST http://localhost:8000/query/ -H "Content-Type: application/json" -d '{"question":"..."}'`
- [ ] Confirm response shape: `answer`, `sources`, `confidence`,
      `evidence_status`, `retries`
- [ ] Add basic error handling (empty question, no documents uploaded yet)

**Done when:** full upload → query round trip works purely via HTTP.

---

## Phase 7 — Frontend Wiring
Files: `frontend/`

- [ ] `npm install && npm run dev`
- [ ] Confirm the Vite proxy forwards `/upload` and `/query` to
      `localhost:8000` (see `vite.config.js`)
- [ ] Upload a doc, ask a question, confirm sources render
- [ ] (Optional) Surface `retries` / `evidence_status` in the UI so the
      "agentic" retry behavior is visibly demoable, not just backend logs

**Done when:** you can demo the whole flow from the browser with zero
terminal commands.

---

## Phase 8 — Multi-Document / Conversation Flow
Matches Architecture.md section 13.

- [ ] Upload 2–3 different documents, confirm queries search across all
      of them (shared Chroma collection handles this automatically)
- [ ] Add a follow-up question that depends on `conversation_history`
      (currently stored in state but not yet fed into the prompt —
      wire it into `REPORT_PROMPT` for multi-turn context)

**Done when:** a second question can reference the first answer.

---

## Phase 9 — Polish for Demo / Portfolio
- [ ] Write 2–3 sample questions with expected "sufficient" vs
      "insufficient" behavior into `docs/demo-script.md`
- [ ] Record a short screen capture showing the retry loop firing
- [ ] Add the LangGraph mermaid diagram (Phase 5 optional step) to the README
- [ ] Clean up TODOs, remove dead code, pin exact dependency versions

**Done when:** a stranger could clone the repo, follow `README.md`, and
reproduce your demo in under 10 minutes.

---

## Future Enhancements (post-MVP, from Architecture.md §16)
Pick 1–2 to extend the project further once the core loop is solid:
- Query rewriting before retrieval retry (rather than just widening Top-K)
- Cross-encoder reranking of retrieved chunks before the Evidence Agent
- Citation verification (check generated citations actually appear in
  trusted_chunks)
- Hallucination detection pass on the Report Agent's output
- Streaming responses (SSE/WebSocket) from `/query/`
- Multi-user workspaces (namespaced Chroma collections per user)
