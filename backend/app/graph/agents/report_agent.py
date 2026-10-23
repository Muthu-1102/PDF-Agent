"""
Report Agent
------------
Responsibility: generate the final answer, but ONLY from trusted_chunks
handed to it by the Evidence Agent. Always attaches sources + a
confidence score derived from retrieval similarity.
"""

from app.graph.state import RAGState
from app.llm.groq_client import call_llm

REPORT_PROMPT = """You are the Report Agent in an Agentic RAG system.
Answer the user's question using ONLY the evidence provided below.
Do not use outside knowledge. If the evidence does not fully answer the
question, say explicitly what is missing. Cite the document name and
chunk id inline for every claim, like [doc_name, chunk 3].

Question: {question}

Evidence:
{evidence}

Answer:"""


def _format_evidence(trusted_chunks) -> str:
    return "\n\n".join(
        f"[{c['doc_name']} | chunk {c['chunk_id']} | page {c['page_number']}]\n{c['text']}"
        for c in trusted_chunks
    )


def generate_report(state: RAGState) -> RAGState:
    trusted_chunks = state.get("trusted_chunks", [])
    evidence_status = state.get("evidence_status", "insufficient")

    # H5 Fix: If evidence was judged insufficient (even after retries) or no trusted chunks exist,
    # cleanly abstain rather than hallucinating an answer.
    if not trusted_chunks or evidence_status == "insufficient":
        state["generated_answer"] = (
            "I don't have sufficient verified evidence in the knowledge "
            "base to answer this question confidently."
        )
        state["confidence"] = 0.0
        state["sources"] = []
        return state

    prompt = REPORT_PROMPT.format(
        question=state["question"], evidence=_format_evidence(trusted_chunks)
    )
    answer = call_llm(prompt)

    state["generated_answer"] = answer
    state["confidence"] = round(
        sum(c["score"] for c in trusted_chunks) / len(trusted_chunks), 2
    )
    state["sources"] = [
        {
            "document": c["doc_name"],
            "page_number": c["page_number"],
            "chunk_id": c["chunk_id"],
            "score": c["score"],
        }
        for c in trusted_chunks
    ]
    return state
