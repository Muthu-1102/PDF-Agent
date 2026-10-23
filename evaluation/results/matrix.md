# Evaluation Results Matrix

### A) Main Results Matrix

| Experiment | Metric | Definition (short) | n (cases / runs) | Result (95% CI) | Status | Evidence file |
|---|---|---|---|---|---|---|
| E1 - Ingestion | Chunk Truncation Rate (H1) | Share of chunks exceeding embedding model max_seq_length (256 tokens) | 2 / 65 chunks | 3.08% (max 689 tokens vs 256 limit) | MEASURED | `evaluation/results/baseline/e1_ingestion_results.json` |
| E1 - Ingestion | PDF Parsing Throughput | PyMuPDF extraction and chunking rate on 52-page manual | 52 pages | 2295.6 pages/sec (peak mem 0.08 MB) | MEASURED | `evaluation/results/baseline/e1_ingestion_results.json` |
| E2 - Retrieval | Recall@4 | Fraction of answerable queries where gold chunk in top-4 | 45 queries | 80.0% (95% CI: [66.2%, 89.1%]) | MEASURED | `evaluation/results/baseline/e2_retrieval_summary.json` |
| E2 - Retrieval | Recall@12 | Fraction of answerable queries where gold chunk in top-12 | 45 queries | 86.7% (95% CI: [73.8%, 93.7%]) | MEASURED | `evaluation/results/baseline/e2_retrieval_summary.json` |
| E2 - Retrieval | Position-Sensitivity Recall Gap (H1) | Recall@4 for fact at chunk START (tokens 0-50) vs END (tokens >256) | 4 paired queries | Start: 100.0% vs End: 0.0% (Delta: -100.0%) | MEASURED | `evaluation/results/baseline/e2_retrieval_summary.json` |
| E3 - Evidence Filter | LLM Sufficiency Rate (Answerable) | Rate at which LLM judge flags retrieved chunks as sufficient for answerable queries | 58 queries | 11.1% | MEASURED | `evaluation/results/baseline/e3_summary.json` |
| E3 - Evidence Filter | Single-Chunk False Insufficiency (H6) | Single-chunk answerable queries rejected due to MIN_TRUSTED_CHUNKS=2 | 58 queries | 29 occurrences | MEASURED | `evaluation/results/baseline/e3_summary.json` |
| E4 - End-to-End | Answer Correctness (Full System) | Answerable questions with all required facts and zero forbidden facts | 45 questions | 77.8% (95% CI: [63.7%, 87.5%]) | MEASURED | `evaluation/results/baseline/e4_ablation_summary.json` |
| E4 - End-to-End | Unsupported-Answer Rate (Full System) | Unanswerable questions receiving a substantive hallucinated answer | 13 questions | 53.8% (95% CI: [29.1%, 76.8%]) | MEASURED | `evaluation/results/baseline/e4_ablation_summary.json` |
| E4 - End-to-End | Over-Abstention Rate (Full System) | Answerable questions incorrectly rejected/abstained | 45 questions | 8.9% (95% CI: [3.5%, 20.7%]) | MEASURED | `evaluation/results/baseline/e4_ablation_summary.json` |
| E4 - End-to-End | Citation Validity | Inline [doc, chunk N] citations matching trusted_chunks | 58 queries | 86.2% | MEASURED | `evaluation/results/baseline/e4_ablation_summary.json` |
| E4 - End-to-End | H5 Discrepancy Occurrences | Queries where evidence_status == 'insufficient' yet non-canned answer returned with confidence > 0 | 58 queries | 28 queries | MEASURED | `evaluation/results/baseline/e4_ablation_summary.json` |
| E4 - End-to-End | Confidence AUROC (H7) | Discrimination power of mean cosine similarity predicting correctness | 45 questions | 0.7657 (non-calibrated) | MEASURED | `evaluation/results/baseline/e4_ablation_summary.json` |
| E7 - Concurrency | Event-Loop Blocking Duration (H8) | HTTP GET /health latency during concurrent /query execution | 1 concurrent query | 0.0265s (status 200) | MEASURED | `evaluation/results/baseline/e7_concurrency_latency.json` |

### B) Comparison Matrix across Architectural Configurations

| Configuration | Answer Correctness | Unsupported Answer Rate | Over-Abstention Rate | Citation Validity | LLM Calls / Query | p50 Latency (s) |
| --- | --- | --- | --- | --- | --- | --- |
| Arm B Standard RAG | 73.3% | 15.4% | 20.0% | 100.0% | 1.0 | 3.809s |
| Arm D Similarity Only | 66.7% | 15.4% | 20.0% | 100.0% | 0.88 | 2.958s |
| Arm C No Retry | 75.6% | 23.1% | 20.0% | 100.0% | 1.67 | 6.777s |
| Arm A Full System | 77.8% | 53.8% | 8.9% | 86.2% | 3.98 | 47.765s |

### C) Hypotheses Verification Matrix

| Hypothesis / Weakness | Status | Evidence File | Fixed in improvements? |
|---|---|---|---|
| H1: Embedding Truncation (256 tokens vs 500 words) | **CONFIRMED** | `evaluation/results/baseline/e1_ingestion_results.json, e2_retrieval_summary.json` | Pending Phase 6 |
| H2: Evidence Judge 600-char Prompt Truncation | **CONFIRMED** | `evaluation/results/baseline/e3_summary.json` | Pending Phase 6 |
| H3: Silent Fallback to Cosine Similarity | **CONFIRMED** | `evaluation/results/baseline/e5_e6_results.json` | Pending Phase 6 |
| H4: Redundant Top-K at Retry 3 (k=12 identical to retry 2) | **CONFIRMED** | `evaluation/results/baseline/e5_e6_results.json` | Pending Phase 6 |
| H5: Abstention Keyed on Empty Chunks, not evidence_status | **CONFIRMED** | `evaluation/results/baseline/e4_ablation_summary.json` | Pending Phase 6 |
| H6: MIN_TRUSTED_CHUNKS=2 False Insufficiency on 1-chunk answer | **CONFIRMED** | `evaluation/results/baseline/e3_summary.json` | Pending Phase 6 |
| H7: Confidence Score is Non-Calibrated Cosine Mean | **CONFIRMED** | `evaluation/results/baseline/e4_ablation_summary.json` | Pending Phase 6 |
| H8: Report Agent Error Propagates to 500; Event Loop Blocked | **CONFIRMED** | `evaluation/results/baseline/e5_e6_results.json, e7_concurrency_latency.json` | Pending Phase 6 |
| H9: Re-upload Duplicates Chunks Without Deduplication | **CONFIRMED** | `evaluation/results/baseline/e1_ingestion_results.json` | Pending Phase 6 |
| H10: DOCX Tables Ignored; Non-UTF8 TXT Raises 500 | **CONFIRMED** | `evaluation/results/baseline/e1_ingestion_results.json` | Pending Phase 6 |
| H11: conversation_history Field in State is Ignored | **CONFIRMED** | `evaluation/results/baseline/e5_e6_results.json` | Pending Phase 6 |
| H12: Pinned Dependencies Installability on Modern Python | **CONFIRMED** | `evaluation/manifest.json` | Verified on Python 3.11.9 |
