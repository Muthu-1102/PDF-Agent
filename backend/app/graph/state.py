"""
LangGraph state definition for the Agentic RAG workflow.

This is the single source of truth passed between every node
(retrieve -> evidence -> retry|report). Keep it flat and serializable.
"""

from typing import TypedDict, List, Dict, Any, Optional


class TrustedChunk(TypedDict):
    text: str
    doc_name: str
    page_number: Optional[int]
    chunk_id: Any
    score: float


class RAGState(TypedDict):
    # Input
    question: str
    conversation_history: List[Dict[str, str]]

    # Retrieval
    retrieved_chunks: List[Dict[str, Any]]
    similarity_scores: List[float]
    current_top_k: int
    retry_count: int

    # Evidence Agent output
    trusted_chunks: List[TrustedChunk]
    evidence_status: str  # "pending" | "sufficient" | "insufficient"
    evidence_mode: Optional[str]  # "llm" | "similarity_fallback" (H3 fix)
    judge_error: Optional[str]

    # Report Agent output
    generated_answer: Optional[str]
    sources: List[Dict[str, Any]]
    confidence: Optional[float]


def build_initial_state(question: str, top_k: int) -> RAGState:
    return {
        "question": question,
        "conversation_history": [],
        "retrieved_chunks": [],
        "similarity_scores": [],
        "current_top_k": top_k,
        "retry_count": 0,
        "trusted_chunks": [],
        "evidence_status": "pending",
        "evidence_mode": None,
        "judge_error": None,
        "generated_answer": None,
        "sources": [],
        "confidence": None,
    }
