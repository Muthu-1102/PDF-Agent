# Phase 1 Findings: Working State & Reality Check

**Date:** 2026-10-21  
**Repository:** https://github.com/Muthu-1102/PDF-Agent  
**Evaluator Branch:** `eval/evidence`  
**Baseline Commit:** `b94ce12ec174dce95034f5b29540f8ec5803707f` ("Dockerized version")

---

## 1. Environment & Dependencies

| Component | Target / Pinned | Observed Reality | Verification Command |
|---|---|---|---|
| Python Version | >=3.10 | Python 3.11.9 (virtualenv `.venv`) / System Python 3.10.8 | `.\.venv\Scripts\python.exe --version` |
| fastapi | 0.115.0 | 0.115.0 installed | `.\.venv\Scripts\pip.exe list` |
| langgraph | 0.2.34 | 0.2.34 installed | `.\.venv\Scripts\pip.exe list` |
| chromadb | 0.5.5 | 0.5.5 installed | `.\.venv\Scripts\pip.exe list` |
| groq | 0.11.0 | 0.11.0 installed | `.\.venv\Scripts\pip.exe list` |
| sentence-transformers | 3.1.1 | 3.1.1 installed | `.\.venv\Scripts\pip.exe list` |
| pymupdf | 1.24.10 | 1.24.10 installed | `.\.venv\Scripts\pip.exe list` |
| python-docx | 1.1.2 | 1.1.2 installed | `.\.venv\Scripts\pip.exe list` |
| Node.js / Frontend | Modern Node | Node v22.15.0, npm 11.6.0 | `node -v ; npm -v` |
| Frontend Build | Vite React build | PASS: Built in 1.51s (`dist/assets/index-CSkFMlMj.js` 149.52 kB) | `npm --prefix frontend run build` |

---

## 2. Test Suite Check

- **Command:** `python -m unittest discover`
- **Output:** `Ran 0 tests in 0.000s. OK.`
- **Finding:** No test suite exists anywhere in `backend/` or `frontend/`. Only an orphaned `.pyc` from an earlier untracked `tests.py` was detected in `backend/app/__pycache__/`.

---

## 3. LLM API & Model Availability

- **GROQ_API_KEY:** Present in `backend/.env` (length: 56 characters, git-ignored).
- **Default Configured Model:** `llama-3.3-70b-versatile` (specified in `backend/app/core/config.py` and `backend/.env.example`).
- **Live Check with Default Model:**
  ```text
  groq.NotFoundError: Error code: 404 - {'error': {'message': 'The model `llama-3.3-70b-versatile` does not exist or you do not have access to it.', 'type': 'invalid_request_error', 'code': 'model_not_found'}}
  ```
- **Live Check of Available Models via `GET https://api.groq.com/openai/v1/models`:**
  - Active models returned: `qwen/qwen3.8-27b`, `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `allam-2-7b`, `canopylabs/orpheus-v1-english`, `whisper-large-v3`.
- **Operational Model Selection:**
  - `qwen/qwen3.8-27b` tested and verified working for zero-shot JSON evidence judging and report generation.
  - Active Rate Limits on this account: `1,000 requests/min`, `8,000 tokens/min` (with fast bucket refill ~134ms-577ms).

---

## 4. Hypothesis H1 Immediate Verification (Embedding Truncation)

- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **`model.max_seq_length`:** 256 tokens.
- **Configured Chunk Size (`CHUNK_SIZE_WORDS`):** 500 words.
- **Natural Language 500-Word Chunk Tokenization:**
  - 500 words of technical/policy text produces **~680 WordPiece tokens**.
  - Truncation point: First 256 tokens are embedded; the remaining **~424 tokens (62.35% of the chunk)** are silently discarded by the embedding model tokenizer.
  - Warning emitted by Hugging Face tokenizer:
    `Token indices sequence length is longer than the specified maximum sequence length for this model (680 > 256). Running this sequence through the model will result in indexing errors`
- **Conclusion:** **H1 is CONFIRMED**. Any facts appearing in the second half of a 500-word chunk are invisible to vector search.

---

## 5. End-to-End Smoke Test Findings

- **Setup:** Isolated temporary Chroma directory (`mkdtemp`), synthetic file `sample_smoke.txt` with 1 fact:
  `"Project Alpha is an internal initiative led by Dr. Evelyn Reed. The launch date is November 15, 2026."`
- **`GET /health`:** Returned HTTP 200 `{"status": "ok"}`.
- **`POST /upload/`:** Returned HTTP 200 `{"document_id": "...", "document_name": "sample_smoke.txt", "chunks_indexed": 1}`.
- **`POST /query/`:** `"Who leads Project Alpha and when is the launch date?"`
  - Returned HTTP 200:
    ```json
    {
      "answer": "Project Alpha is led by Dr. Evelyn Reed, and the launch date is November 15, 2026 [sample_smoke.txt, chunk 0].",
      "sources": [{"document": "sample_smoke.txt", "page_number": 1, "chunk_id": 0, "score": 0.8644}],
      "confidence": 0.86,
      "evidence_status": "insufficient",
      "retries": 3
    }
    ```
- **Live Architectural Observations:**
  1. **H6 Confirmed:** The single indexed chunk was 100% relevant and sufficient, but because `MIN_TRUSTED_CHUNKS = 2`, `evidence_status` was set to `"insufficient"`.
  2. **H4 Confirmed:** Graph executed 3 retries (top-k 4 -> 8 -> 12 -> 12). Logs showed `n_results = 1` for 4, 8, 12, 12. Retry 3 retrieved at top-k 12 identically to Retry 2.
  3. **H5 Confirmed:** When retries exhausted, `route_on_evidence` routed to `report_node`. Because `trusted_chunks` was non-empty (1 chunk), `report_agent` generated a full, cited answer with `confidence = 0.86`, despite `evidence_status == "insufficient"`.

---

## 6. Documentation Claims vs. Reality

| Claim in Docs | Location | Actual Reality |
|---|---|---|
| Architecture file | `README.md` L11: `See Architecture.txt` | `Architecture.txt` does not exist in root; actual design file is `docs/Architecture.md`. |
| Demo script | `docs/ROADMAP.md` Phase 9 | `docs/demo-script.md` does not exist. |
| Conversation History | `app/graph/state.py` & ROADMAP §8 | `conversation_history` exists in `RAGState` but is never passed to LLM prompts. |
| Query Rewriting on Retry | ROADMAP §16 Future Enhancements | Not implemented. `retry_node` only increments `current_top_k`. |
| Unit / Integration Tests | Standard project structure | Zero automated tests in codebase. |
| `.env` location handling | `backend/app/core/config.py` | Calls `load_dotenv()` without explicit path; fails when running from repository root unless `PYTHONPATH` or `CWD` is `backend/`. |
