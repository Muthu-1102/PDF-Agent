from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes_upload import router as upload_router
from app.api.routes_query import router as query_router

app = FastAPI(
    title="PDF Agent",
    description="LangGraph-orchestrated, evidence-verified RAG",
    version="1.0",
)

# FastAPI stays intentionally stateless/thin -- all workflow state lives
# in LangGraph (see app/graph/state.py).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router)
app.include_router(query_router)


@app.get("/health")
def health():
    return {"status": "ok"}
