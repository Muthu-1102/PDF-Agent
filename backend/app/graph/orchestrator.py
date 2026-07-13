"""
RAG Orchestrator
----------------
Builds and compiles the LangGraph StateGraph described in
Architecture.md section 8:

    START -> retrieve -> evidence --sufficient--> report -> END
                             |
                       insufficient
                             |
                           retry -> retrieve (loop)

The orchestrator only wires nodes together and maintains state; it never
generates answers itself (that's the Report Agent's job).
"""

from langgraph.graph import StateGraph, END

from app.core.config import settings
from app.graph.state import RAGState, build_initial_state
from app.graph.nodes import (
    retrieve_node,
    evidence_node,
    retry_node,
    report_node,
    route_on_evidence,
)


def build_graph():
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("evidence", evidence_node)
    graph.add_node("retry", retry_node)
    graph.add_node("report", report_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "evidence")

    graph.add_conditional_edges(
        "evidence",
        route_on_evidence,
        {"retry": "retry", "report": "report"},
    )

    graph.add_edge("retry", "retrieve")
    graph.add_edge("report", END)

    return graph.compile()


# Compile once at import time; reuse across requests.
_compiled_graph = build_graph()


def run_query(question: str) -> dict:
    initial_state = build_initial_state(question, top_k=settings.DEFAULT_TOP_K)
    final_state: RAGState = _compiled_graph.invoke(initial_state)

    return {
        "answer": final_state["generated_answer"],
        "sources": final_state["sources"],
        "confidence": final_state["confidence"],
        "evidence_status": final_state["evidence_status"],
        "retries": final_state["retry_count"],
    }
