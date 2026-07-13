from fastapi import APIRouter
from pydantic import BaseModel

from app.graph.orchestrator import run_query

router = APIRouter(prefix="/query", tags=["query"])


class QueryRequest(BaseModel):
    question: str


@router.post("/")
async def query(request: QueryRequest):
    """Runs the full Agentic RAG LangGraph workflow for a single question."""
    return run_query(request.question)
