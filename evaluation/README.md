# Agentic RAG Evaluation Framework

This directory contains the independent, reproducible evaluation framework for the `PDF-Agent` repository.

## Methodology Overview

The evaluation measures the empirical performance, reliability, and architectural trade-offs of the LangGraph-orchestrated Agentic RAG system versus standard baseline architectures.

### Core Architecture Evaluated
- **Graph Orchestration:** `START -> retrieve -> evidence -> (sufficient: report | insufficient: retry -> retrieve) -> END`.
- **Evidence Verification:** LLM judge evaluating retrieved chunks against an evidence candidate floor, falling back to cosine similarity threshold on failure.
- **Retry Mechanism:** Dynamic Top-$K$ expansion ($4 \to 8 \to 12$).
- **Report Generation:** Citation-grounded synthesis using only `trusted_chunks`.

### Evaluated Configurations (Arms)
1. **Arm A (Full System - Default):** LLM Evidence Judge (`qwen/qwen3.8-27b`), default retries ($k \in [4, 8, 12]$), `SIMILARITY_THRESHOLD=0.35`, `MIN_TRUSTED_CHUNKS=2`.
2. **Arm B (Standard RAG Baseline):** Traditional single-shot RAG without evidence filtering or retry loops. Retrieves top-$K$ chunks and directly invokes the report agent prompt.
3. **Arm C (LLM Judge, No Retry):** Evidence judge enabled with `MAX_RETRIES=0`. Measures the isolated benefit of iterative retrieval.
4. **Arm D (Similarity-Only Filter):** `EVIDENCE_JUDGE_MODE="similarity"`. Replaces the LLM judge with cosine similarity thresholding ($0.35$).

---

## Dataset Design

All test documents and queries are 100% synthetic, procedurally generated with fixed random seeds (`SEED=42`), ensuring:
- **Zero Parametric Memory Leakage:** Fictional organizations, fictional policies, and synthetic numbers that cannot be memorized by the LLM.
- **Multimodal Formats:** Corpus consists of PDF (multi-page, single-page, long 50+ page manual), DOCX (including tables), and TXT files.
- **Adversarial & Edge Cases:**
  - Near-duplicate conflicting policies (e.g., Policy v1.0 vs v2.4 with updated expense caps).
  - Position-sensitive pairs: identical target facts positioned at the beginning (tokens 0-100) vs the tail (tokens 400-500) of 500-word chunks (testing embedding and judge truncation).
  - Out-of-knowledge-base questions and false-premise queries to test abstention.
- **Gold Evidence Mapping:** Post-chunking exact substring matching against the system's actual indexed chunks to ensure unambiguous, deterministic Recall@k ground truth.

---

## Reproducibility & Safety

1. **ChromaDB Isolation:** Every experiment run creates and destroys an isolated temporary directory for `CHROMA_PERSIST_DIR`. The project's persistent `./data/chroma_store` is never modified or read.
2. **Non-Destructive Instrumentation:** Application source files remain unaltered on `eval/evidence`. All metrics, node timings, and fallback tracking are captured via harness monkeypatching.
3. **Paced Rate Limiting:** All Groq API calls are paced with exponential backoff to operate strictly within the free-tier limit of 8,000 tokens/minute.
4. **Strict Authenticity:** No fabricated data, mock runs presented as real, or hardcoded benchmark metrics. Raw traces are saved in JSONL format under `evaluation/results/baseline/`.
