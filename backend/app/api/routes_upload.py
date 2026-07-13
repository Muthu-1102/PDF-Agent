import os
import uuid
import tempfile
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.ingestion.loader import load_document
from app.ingestion.chunker import chunk_document
from app.vectorstore.chroma_client import get_collection

router = APIRouter(prefix="/upload", tags=["upload"])

ALLOWED_TYPES = {"pdf", "docx", "txt"}


@router.post("/")
async def upload_document(file: UploadFile = File(...)):
    file_type = file.filename.split(".")[-1].lower()
    if file_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_type}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp:
        tmp.write(await file.read())
        temp_path = tmp.name

    try:
        pages = load_document(temp_path, file_type)
        chunks = chunk_document(pages)  # List[(page_number, chunk_text)]

        if not chunks:
            raise HTTPException(status_code=400, detail="No extractable text found in document")

        doc_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        collection = get_collection()

        ids, docs, metadatas = [], [], []
        for idx, (page_number, chunk_text) in enumerate(chunks):
            ids.append(f"{doc_id}_{idx}")
            docs.append(chunk_text)
            metadatas.append(
                {
                    "document_id": doc_id,
                    "document_name": file.filename,
                    "chunk_index": idx,
                    "page_number": page_number,
                    "upload_timestamp": timestamp,
                }
            )

        collection.add(ids=ids, documents=docs, metadatas=metadatas)

        return {
            "document_id": doc_id,
            "document_name": file.filename,
            "chunks_indexed": len(chunks),
        }
    finally:
        os.remove(temp_path)
