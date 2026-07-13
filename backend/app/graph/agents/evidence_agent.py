"""
Evidence Agent
--------------
Responsibility: decide whether retrieved chunks are actually good enough
to answer the question. Never generates the answer itself.

Current strategy: similarity-threshold filtering + minimum trusted-chunk
count. TODO (see ROADMAP): replace/augment with an LLM-judged relevance
pass for cases where cosine similarity is misleading.
"""

from app.core.config import settings
from app.graph.state import RAGState


def evaluate_evidence(state: RAGState) -> RAGState:
    trusted = []

    for chunk, score in zip(state["retrieved_chunks"], state["similarity_scores"]):
        if score >= settings.SIMILARITY_THRESHOLD:
            metadata = chunk["metadata"]
            trusted.append(
                {
                    "text": chunk["text"],
                    "doc_name": metadata.get("document_name"),
                    "page_number": metadata.get("page_number"),
                    "chunk_id": metadata.get("chunk_index"),
                    "score": round(score, 4),
                }
            )

    state["trusted_chunks"] = trusted
    state["evidence_status"] = (
        "sufficient" if len(trusted) >= settings.MIN_TRUSTED_CHUNKS else "insufficient"
    )
    return state
