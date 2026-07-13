"""
Document loaders. Each function returns page-aware text where possible,
so upstream chunking can retain accurate page_number metadata.
"""

from typing import List, Tuple

import fitz  # PyMuPDF
import docx


def load_pdf(file_path: str) -> List[Tuple[int, str]]:
    """Returns list of (page_number, page_text)."""
    pages = []
    with fitz.open(file_path) as doc:
        for i, page in enumerate(doc, start=1):
            pages.append((i, page.get_text()))
    return pages


def load_docx(file_path: str) -> List[Tuple[int, str]]:
    """DOCX has no native page concept -> treat whole doc as page 1."""
    document = docx.Document(file_path)
    text = "\n".join(p.text for p in document.paragraphs)
    return [(1, text)]


def load_txt(file_path: str) -> List[Tuple[int, str]]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [(1, f.read())]


def load_document(file_path: str, file_type: str) -> List[Tuple[int, str]]:
    file_type = file_type.lower()
    if file_type == "pdf":
        return load_pdf(file_path)
    if file_type == "docx":
        return load_docx(file_path)
    if file_type == "txt":
        return load_txt(file_path)
    raise ValueError(f"Unsupported file type: {file_type}")
