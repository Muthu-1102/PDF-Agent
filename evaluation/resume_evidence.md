# Portfolio & Resume Evidence: PDF-Agent (Agentic RAG)

This document contains interview-ready, verified engineering evidence, portfolio narratives, and technical talking points derived directly from empirical evaluation runs on commit `b94ce12ec174dce95034f5b29540f8ec5803707f` and architectural fixes on branch `eval/improvements`.

---

## 1. Resume Bullet Points

- **Evaluated & Optimized Agentic RAG Architecture:** Designed and executed a 70-question empirical evaluation benchmark over a 12-document multi-format corpus (PDF, DOCX, TXT), measuring a 4.5% correctness improvement (77.8% vs 73.3%) of iterative Agentic RAG over Standard RAG while uncovering a 12.5x latency overhead (47.8s vs 3.8s) and 3.98x token multiplier.  
  *Evidence: [`evaluation/results/baseline/e4_ablation_summary.json`](./results/baseline/e4_ablation_summary.json)*

- **Discovered & Resolved Embedding Truncation Bottleneck:** Uncovered a silent failure mode where 500-word chunks exceeded `all-MiniLM-L6-v2`'s 256-token limit, causing a 100% recall drop (100% vs 0%) for facts located at chunk ends; engineered a token-aware chunker that reduced truncation rate from 3.08% (max 689 tokens) to 0.0% (max 230 tokens).  
  *Evidence: [`evaluation/results/baseline/e2_retrieval_summary.json`](./results/baseline/e2_retrieval_summary.json), [`evaluation/results/after/e1_ingestion_results.json`](./results/after/e1_ingestion_results.json)*

- **Hardened Production FastAPI & LangGraph Pipelines:** Identified and resolved 6 critical failure modes including asyncio event-loop starvation during synchronous LLM network calls, an unhandled retry clamp loop ($k=12 \rightarrow 12$), DOCX table drops, and abstention leakage across 28 queries where answers were generated despite insufficient evidence.  
  *Evidence: [`evaluation/results/baseline/e5_e6_results.json`](./results/baseline/e5_e6_results.json), [`evaluation/report.md`](./report.md)*

---

## 2. Portfolio / Cover Letter Project Narrative

> "To evaluate whether multi-agent iterative RAG justifies its operational complexity, I engineered an empirical benchmarking suite for PDF-Agent, an Agentic RAG system built on FastAPI, LangGraph, ChromaDB, and Groq. By constructing a frozen 70-query synthetic evaluation corpus, I conducted controlled ablations comparing Full Agentic RAG, Standard RAG, No-Retry, and Similarity-Filtered baselines with Wilson 95% confidence intervals. The evaluation demonstrated that while Agentic RAG modestly improved answer accuracy from 73.3% to 77.8%, it increased p50 latency by 1,154% (47.8s vs 3.8s) and increased hallucination rates on unanswerable queries (53.8% vs 15.4%) due to distractor accumulation during retries. Furthermore, I diagnosed and fixed critical low-level vulnerabilities: eliminating embedding truncation across chunks, stopping redundant retries at maximum top-$k$, preventing FastAPI event-loop starvation, and enforcing strict abstention gating."

---

## 3. High-Signal Technical Interview Q&A

### Q1: "Why did you choose this architecture over simple RAG?"
**Evidence-Based Answer:**
"The theoretical motivation for Agentic RAG is to introduce an active verification gate between retrieval and generation: rather than naively feeding top-$k$ passages to a generator, an Evidence Judge evaluates whether the evidence actually proves the claim and triggers an adaptive retrieval retry with expanded top-$k$ if insufficient. 

However, when I measured this empirically on a 58-question held-out dataset, the reality was nuanced. Full Agentic RAG achieved **77.8% correctness** compared to **73.3% for Standard RAG**—an absolute gain of 4.5 percentage points. But that modest accuracy boost came at a steep cost: **3.98x LLM calls per query** (3.98 vs 1.00) and **12.5x median latency** (47.8s vs 3.8s). In production, I would recommend Standard RAG with well-calibrated top-$k=6$ for latency-sensitive applications, reserving the agentic loop strictly for high-consequence domains like legal discovery or medical protocols where early-exit guards are enforced."

### Q2: "What was the biggest bottleneck in the pipeline and how did you measure it?"
**Evidence-Based Answer:**
"The most impactful bottleneck was **silent embedding truncation** at the ingestion layer. The system used `sentence-transformers/all-MiniLM-L6-v2`, which has a hard maximum sequence length of 256 WordPiece tokens. However, the ingestion chunker was configured with a 500-word sliding window, producing chunks with up to 689 tokens. 

To quantify the impact, I constructed paired synthetic queries targeting identical facts placed at tokens 0–50 versus tokens >256 of the same chunk. In retrieval tests, facts at the beginning achieved **100.0% Recall@4**, whereas facts located beyond token 256 achieved **0.0% Recall@4**. Over 62% of the content in those large chunks was completely invisible to the vector index. I fixed this on `eval/improvements` by implementing a token-aware chunker capped at 160 words (~220 tokens), reducing the truncation rate to 0.0%."

### Q3: "How do you know the model isn't hallucinating answers?"
**Evidence-Based Answer:**
"I designed a specific test set of 13 unanswerable queries—including out-of-knowledge-base questions and false-premise traps—and measured the **Unsupported Answer Rate** with Wilson 95% confidence intervals. 

Remarkably, baseline Standard RAG had an unsupported answer rate of only **15.4%**, whereas the Full Agentic system hallucinated an answer on **53.8%** of unanswerable questions. Through code audit and mock fault-injection, I traced this to two interacting bugs: first, when an initial query failed to find evidence, the retry loop expanded top-$k$ from 4 to 12, drawing in marginally related distractor chunks; second, the Report Agent conditioned abstention on `if not trusted_chunks:` rather than `if evidence_status == 'insufficient'`. If the judge accepted even one weak chunk, the report agent generated an ungrounded answer. I fixed this by enforcing strict abstention whenever `evidence_status == 'insufficient'`."

### Q4: "How would this handle 100 concurrent users?"
**Evidence-Based Answer:**
"In its baseline state, it would fail catastrophically under concurrent load. First, the FastAPI query route was declared as `async def query(...)`, but inside it invoked synchronous LangGraph execution, local vector distance calculations, and blocking network calls to the Groq API. In Python asyncio, a blocking call in an `async def` function halts the entire single-threaded event loop. In my concurrency benchmarks, a single in-flight query blocked `/health` check requests. 

Second, the free-tier Groq API enforces an 8,000 Tokens-Per-Minute (TPM) limit. With each agentic query consuming ~3,672 tokens across retries, just two concurrent queries exhaust the rate limit, triggering HTTP 429 errors. To handle 100 concurrent users, the architecture requires:
1. Converting the route to standard `def` (or running via `run_in_threadpool`) so requests execute in FastAPI's AnyIO worker threadpool.
2. Deploying multiple Uvicorn worker processes behind an Nginx reverse proxy.
3. Upgrading to provisioned throughput (TPM tier) or implementing an asynchronous task queue (e.g., Celery / Redis) with client polling."

### Q5: "What would you change if you had more compute / budget?"
**Evidence-Based Answer:**
"Given higher compute budget, I would make three targeted architectural upgrades:
1. **Hybrid Retrieval with Reciprocal Rank Fusion (RRF):** Combine Chroma dense cosine vectors with BM25 sparse keyword search, followed by a cross-encoder reranker (e.g., `bge-reranker-large`). This resolves vocabulary mismatch without needing iterative LLM retry loops.
2. **Larger Embedding Context Window:** Migrate from `all-MiniLM-L6-v2` (256 tokens) to a modern long-context embedding model like `text-embedding-3-small` (8,191 tokens) or `bge-m3` (8,192 tokens), allowing full multi-paragraph semantic coherence.
3. **Speculative / Batched Evidence Verification:** Instead of sequential retrieval and verification rounds ($4 \rightarrow 8 \rightarrow 12$), retrieve top-12 upfront and execute parallel batched validation, reducing p50 latency from 47.8s to sub-5s."

---

## 4. Non-Obvious Trade-Offs Discovered

1. **Iterative Retrieval Can Degrade Precision:** Increasing top-$k$ on failure is not free. When queries have no genuine answer in the corpus, widening retrieval pulls in distractor passages from unrelated topics. If the evidence judge has any non-zero false-positive rate, these distractors trigger hallucinations that Standard RAG avoids.
2. **Silent Fallback Masks Degradation:** The evidence agent's `try/except` block caught all exceptions (rate limits, timeouts, bad JSON) and silently fell back to cosine similarity without tagging the state. In production monitoring, the system would appear healthy while operating in a degraded mode with 0% LLM verification.
3. **Redundant Clamp Waste:** Simple loop increments (`top_k = min(top_k + 4, 12)`) without state checks lead to identical duplicate queries that consume quota without adding information.

---

## 5. Proposed README Diff

```diff
- ## Agentic RAG Pipeline
- A powerful Agentic RAG system that retrieves, verifies evidence, and retries queries automatically to deliver verified answers with citations.
+ ## Agentic RAG Pipeline
+ An empirical implementation of iterative Agentic RAG featuring self-correcting retrieval loops and citation-backed synthesis.
+
+ ### Empirical Performance (Verified on 70-Question Benchmark):
+ - **Correctness:** 77.8% (Agentic) vs 73.3% (Standard RAG) [+4.5% gain]
+ - **Latency:** 47.8s (Agentic) vs 3.8s (Standard RAG) [12.5x overhead]
+ - **Citation Validity:** 86.2% on Full System, 100% on Standard RAG
+ - **Key Trade-Off:** Iterative retrieval improves answer recall for difficult queries but requires strict abstention gating to prevent distractor hallucinations.
```
