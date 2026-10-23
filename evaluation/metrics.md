# Operational Metric Definitions and Specifications

This document defines every metric used in the evaluation of the Agentic RAG system.
All metric computations are deterministic and automated in `evaluation/harness/metrics.py`.

---

## 1. Retrieval Metrics (E2)

### 1.1 Recall@k (k ∈ {4, 8, 12})
- **Operational Definition:** Binary indicator per question whether at least one gold chunk ID exists in the top-$k$ retrieved chunks.
- **Formula:**
  $$\text{Recall@}k = \frac{1}{|Q_{\text{ans}}|} \sum_{q \in Q_{\text{ans}}} \mathbb{I}\left( \text{GoldChunks}(q) \cap \text{Retrieved}_k(q) \neq \emptyset \right)$$
- **Denominator:** Total answerable questions $|Q_{\text{ans}}|$.
- **Target / Threshold:** Higher is better (ideal: 1.0).

### 1.2 Multi-Chunk Recall@k
- **Operational Definition:** For questions requiring multiple supporting chunks, the fraction of required gold chunks that were retrieved in top-$k$.
- **Formula:**
  $$\text{MultiRecall@}k = \frac{1}{|Q_{\text{multi}}|} \sum_{q \in Q_{\text{multi}}} \frac{|\text{GoldChunks}(q) \cap \text{Retrieved}_k(q)|}{|\text{GoldChunks}(q)|}$$
- **Denominator:** Multi-chunk answerable questions $|Q_{\text{multi}}|$.

### 1.3 Mean Reciprocal Rank (MRR)
- **Operational Definition:** Reciprocal of the 1-indexed rank of the first relevant gold chunk retrieved in top-$k$. If no gold chunk is retrieved, reciprocal rank is 0.
- **Formula:**
  $$\text{MRR} = \frac{1}{|Q_{\text{ans}}|} \sum_{q \in Q_{\text{ans}}} \frac{1}{\min \{ \text{rank}(c) : c \in \text{GoldChunks}(q) \cap \text{Retrieved}(q) \} \cup \{\infty\}}$$

### 1.4 Context Precision@k
- **Operational Definition:** Fraction of retrieved chunks in top-$k$ that are true gold chunks.
- **Formula:**
  $$\text{Precision@}k = \frac{1}{|Q_{\text{ans}}|} \sum_{q \in Q_{\text{ans}}} \frac{|\text{GoldChunks}(q) \cap \text{Retrieved}_k(q)|}{k}$$

---

## 2. Evidence Filtering Metrics (E3)

### 2.1 Evidence Precision and Recall
- **Definition:** Precision and Recall of the `trusted_chunks` set relative to the `GoldChunks(q)`.
- **Formulas:**
  $$\text{Evidence Precision} = \frac{|\text{Trusted}(q) \cap \text{GoldChunks}(q)|}{|\text{Trusted}(q)|} \quad (\text{if } |\text{Trusted}| > 0 \text{ else } 0)$$
  $$\text{Evidence Recall} = \frac{|\text{Trusted}(q) \cap \text{GoldChunks}(q)|}{|\text{GoldChunks}(q)|}$$

### 2.2 Sufficiency Confusion Matrix
- **TP:** Answerable question where gold chunks retrieved AND `evidence_status == "sufficient"`.
- **FP:** Unanswerable/out-of-KB question where `evidence_status == "sufficient"`.
- **TN:** Unanswerable/out-of-KB question where `evidence_status == "insufficient"`.
- **FN:** Answerable question where gold chunks retrieved BUT `evidence_status == "insufficient"`.

### 2.3 Single-Chunk False Insufficient Rate (H6)
- **Operational Definition:** Fraction of single-chunk answerable questions where the gold chunk was retrieved and judged relevant, but `evidence_status` was declared `"insufficient"` due to `MIN_TRUSTED_CHUNKS >= 2`.

---

## 3. End-to-End Generation & Safety Metrics (E4)

### 3.1 Answer Correctness
- **Operational Definition:** For answerable questions, binary indicator that:
  1. ALL required facts ($F_{\text{req}}$) are present in the answer text (case-insensitive normalized substring or numerical equivalence).
  2. NO forbidden facts ($F_{\text{forbid}}$, e.g. distractor values from superseded document versions) are asserted in the answer.
- **Denominator:** Answerable questions $|Q_{\text{ans}}|$.

### 3.2 Abstention Correctness
- **Operational Definition:** For unanswerable or false-premise questions, binary indicator that the system refused or declined to answer.
- **Matching Rule:** Answer matches the canned fallback string (`"I don't have sufficient verified evidence in the knowledge base to answer this question confidently."`) OR contains refusal phrases (`"does not contain"`, `"cannot be answered"`, `"not mentioned"`, `"not found"`, `"insufficient evidence"`, `"no information"`, `"not provided"`).
- **Denominator:** Total unanswerable + false-premise questions $|Q_{\text{unans}}|$.

### 3.3 Unsupported-Answer Rate
- **Operational Definition:** Fraction of unanswerable or false-premise questions that received a substantive answer rather than an abstention.
- **Formula:**
  $$\text{Unsupported-Answer Rate} = 1 - \text{Abstention Correctness}$$
- **Target:** Lower is better (ideal: 0.0).

### 3.4 Over-Abstention Rate
- **Operational Definition:** Fraction of answerable questions where the system returned an abstention despite having valid evidence in the corpus.
- **Target:** Lower is better.

### 3.5 Citation Validity
- **Operational Definition:** Fraction of parsed citations in the format `[<doc_name>, chunk <idx>]` that match a chunk actually present in `trusted_chunks`.
- **Denominator:** Total citations parsed across all generated answers.

### 3.6 Citation Coverage
- **Operational Definition:** Fraction of non-empty sentences in the generated answer that contain at least one bracketed citation.
- **Denominator:** Total sentences across all generated answers.

### 3.7 Deterministic Faithfulness (Grounding)
- **Operational Definition:** For all numeric values (e.g. `$42,000`, `14.5%`, `2026`) and proper nouns asserted in the generated answer, the fraction that appear literally within the cited chunk text.
- **Denominator:** Total extracted numeric/named tokens in generated answers.

### 3.8 H5 Discrepancy Rate
- **Operational Definition:** Fraction of runs where `evidence_status == "insufficient"` yet `confidence > 0.0` and a non-empty, non-canned answer was returned.
- **Denominator:** All completed queries where `evidence_status == "insufficient"`.

### 3.9 Confidence Discrimination (AUROC - H7)
- **Operational Definition:** Area Under the ROC Curve treating `confidence` as a continuous predictor of binary `Answer Correctness`.
- **Target:** $>0.50$ (random guessing = 0.50). Bootstrap 95% CI computed with 1,000 resamples.

---

## 4. Efficiency & Robustness Metrics (E5, E6, E7)

### 4.1 Redundant Retrieval Rate (H4)
- **Operational Definition:** Fraction of queries executing Retry 3 where `current_top_k` was identical to Retry 2 (`MAX_TOP_K = 12`), resulting in a 100% duplicate retrieval set.

### 4.2 LLM Call Count & Token Consumption
- **Definition:** Mean LLM calls and total prompt + completion tokens per query per arm.

### 4.3 Latency Profiles
- **Definition:** p50, p95, and max latency in seconds, broken down by node: `retrieve`, `evidence`, `report`, and `total_end_to_end`.

---

## 5. Statistical Estimation Rules
- **Proportions:** All rates and percentages are reported with Wilson score 95% confidence intervals:
  $$\hat{p} \pm \frac{z^2}{2n} \pm \frac{z}{1 + z^2/n} \sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}$$
- **Continuous / AUROC:** 95% confidence intervals derived from 1,000 bootstrap resamples.
- **Comparative Hypotheses:** Paired McNemar's test for binary outcomes across matched question pairs; two-tailed paired bootstrap for continuous differences.
