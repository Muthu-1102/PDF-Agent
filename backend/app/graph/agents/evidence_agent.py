"""
Evidence Agent
--------------
Responsibility: decide whether retrieved chunks are actually good enough
to answer the question. Never generates the answer itself.

Strategy: LLM-judged relevance + sufficiency.
    1. A loose similarity floor (EVIDENCE_CANDIDATE_FLOOR) trims obvious
       noise before spending LLM tokens.
    2. The remaining candidates are shown to the LLM, which returns
       structured JSON: which chunks are actually relevant to the
       question, whether that's sufficient to answer, and why.
    3. If the LLM call or JSON parsing fails for any reason, the agent
       automatically falls back to pure similarity-threshold filtering
       (the old behavior) so the graph never breaks.
"""

import json
import re

from app.core.config import settings
from app.graph.state import RAGState
from app.llm.groq_client import call_llm

EVIDENCE_PROMPT = """You are the Evidence Agent in an Agentic RAG system.
Your only job is to judge retrieved passages -- you do NOT answer the
question yourself.

Question: {question}

Candidate passages:
{candidates}

For each passage, decide if it is actually relevant and useful for
answering the question (not just topically similar). Then decide if the
relevant passages, taken together, are SUFFICIENT to answer the question
completely and accurately.

Respond with ONLY a JSON object, no other text, in this exact shape:
{{
  "relevant_indices": [list of integer indices of relevant passages],
  "sufficient": true or false,
  "reasoning": "one sentence explaining the sufficiency decision"
}}"""


def _format_candidates(chunks) -> str:
    lines = []
    for i, c in enumerate(chunks):
        meta = c["metadata"]
        # H2 Fix: Do not aggressively truncate chunks to 600 characters; allow full chunk up to 2000 chars
        text_snippet = c["text"][:2000]
        lines.append(
            f"[{i}] ({meta.get('document_name')}, page {meta.get('page_number')}): "
            f"{text_snippet}"
        )
    return "\n\n".join(lines)


def _to_trusted(chunk, score) -> dict:
    metadata = chunk["metadata"]
    return {
        "text": chunk["text"],
        "doc_name": metadata.get("document_name"),
        "page_number": metadata.get("page_number"),
        "chunk_id": metadata.get("chunk_index"),
        "score": round(score, 4),
    }


def _similarity_fallback(state: RAGState) -> RAGState:
    """Old pure-threshold behavior. Used if the LLM judging path fails."""
    trusted = [
        _to_trusted(chunk, score)
        for chunk, score in zip(state["retrieved_chunks"], state["similarity_scores"])
        if score >= settings.SIMILARITY_THRESHOLD
    ]
    state["trusted_chunks"] = trusted
    state["evidence_mode"] = "similarity_fallback"  # H3 Fix
    state["evidence_status"] = (
        "sufficient" if len(trusted) >= settings.MIN_TRUSTED_CHUNKS else "insufficient"
    )
    return state


def _llm_judge(state: RAGState) -> RAGState:
    chunks = state["retrieved_chunks"]
    scores = state["similarity_scores"]

    # Loose pre-filter: cut obvious noise, keep everything else for the LLM.
    candidates = [
        (c, s) for c, s in zip(chunks, scores) if s >= settings.EVIDENCE_CANDIDATE_FLOOR
    ]

    if not candidates:
        state["trusted_chunks"] = []
        state["evidence_status"] = "insufficient"
        state["evidence_mode"] = "llm"  # H3 Fix
        return state

    candidate_chunks = [c for c, _ in candidates]
    prompt = EVIDENCE_PROMPT.format(
        question=state["question"],
        candidates=_format_candidates(candidate_chunks),
    )

    raw_response = call_llm(prompt, temperature=0.0)

    # Be tolerant of models that wrap JSON in prose or code fences.
    match = re.search(r"\{.*\}", raw_response, re.DOTALL)
    if not match:
        raise ValueError("Evidence Agent: no JSON object found in LLM response")

    verdict = json.loads(match.group(0))
    relevant_indices = set(verdict.get("relevant_indices", []))
    llm_sufficient = bool(verdict.get("sufficient", False))

    trusted = [
        _to_trusted(*candidates[i])
        for i in relevant_indices
        if 0 <= i < len(candidates)
    ]

    state["trusted_chunks"] = trusted
    state["evidence_mode"] = "llm"  # H3 Fix
    # Require both the LLM's judgment AND a minimum evidence count --
    # guards against the LLM saying "sufficient" off a single thin chunk.
    state["evidence_status"] = (
        "sufficient"
        if llm_sufficient and len(trusted) >= settings.MIN_TRUSTED_CHUNKS
        else "insufficient"
    )
    return state


def evaluate_evidence(state: RAGState) -> RAGState:
    if settings.EVIDENCE_JUDGE_MODE != "llm":
        return _similarity_fallback(state)

    try:
        return _llm_judge(state)
    except Exception as exc:
        # H3 Fix: record judge error and degrade gracefully to similarity fallback
        state["judge_error"] = f"{type(exc).__name__}: {str(exc)}"
        return _similarity_fallback(state)