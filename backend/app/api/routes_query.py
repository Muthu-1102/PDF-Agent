from fastapi import APIRouter
from pydantic import BaseModel

from app.graph.orchestrator import run_query

router = APIRouter(prefix="/query", tags=["query"])


class QueryRequest(BaseModel):
    question: str


@router.post("/")
def query(request: QueryRequest):
    """Runs the full Agentic RAG LangGraph workflow for a single question.
    H8 Fix: Using standard 'def' offloads the synchronous blocking LangGraph workflow
    and LLM network I/O to a worker threadpool, avoiding event loop starvation."""
    return run_query(request.question)
