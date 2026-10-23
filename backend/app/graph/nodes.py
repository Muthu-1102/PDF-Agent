"""
LangGraph node functions. Each takes the full RAGState and returns an
updated RAGState. Kept separate from agents/ so retrieval mechanics
(talking to Chroma) stay independent of agent reasoning logic.
"""

from app.core.config import settings
from app.vectorstore.chroma_client import get_collection
from app.graph.state import RAGState
from app.graph.agents.evidence_agent import evaluate_evidence
from app.graph.agents.report_agent import generate_report


def retrieve_node(state: RAGState) -> RAGState:
    collection = get_collection()
    results = collection.query(
        query_texts=[state["question"]],
        n_results=state["current_top_k"],
    )

    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    state["retrieved_chunks"] = [
        {"text": d, "metadata": m} for d, m in zip(docs, metadatas)
    ]
    # cosine distance -> similarity score
    state["similarity_scores"] = [1 - dist for dist in distances]
    return state


def evidence_node(state: RAGState) -> RAGState:
    return evaluate_evidence(state)


def report_node(state: RAGState) -> RAGState:
    return generate_report(state)


def retry_node(state: RAGState) -> RAGState:
    """Improve-retrieval step: widen Top-K and loop back to retrieve."""
    state["retry_count"] += 1
    state["current_top_k"] = min(
        state["current_top_k"] + settings.TOP_K_STEP, settings.MAX_TOP_K
    )
    return state


def route_on_evidence(state: RAGState) -> str:
    """Conditional edge: decide retry vs report after evidence check."""
    if state["evidence_status"] == "sufficient":
        return "report"
    if state["retry_count"] >= settings.MAX_RETRIES:
        # Give up retrying -- report_node will produce a low/zero-confidence
        # answer rather than looping forever.
        return "report"
    if state.get("current_top_k", settings.DEFAULT_TOP_K) >= settings.MAX_TOP_K:
        # H4 Fix: Top-K is already at maximum (e.g. 12).
        # Retrying would re-query the exact same top-12 chunks and waste an LLM call.
        return "report"
    return "retry"
