"""
Simple word-window chunker with overlap. Swap for a smarter
(semantic / recursive-character) splitter later -- see docs/ROADMAP.md
Future Enhancements.
"""

from typing import List, Tuple

from app.core.config import settings


def chunk_page_text(
    page_number: int,
    text: str,
    chunk_size: int = settings.CHUNK_SIZE_WORDS,
    overlap: int = settings.CHUNK_OVERLAP_WORDS,
) -> List[Tuple[int, str]]:
    """Returns list of (page_number, chunk_text)."""
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append((page_number, chunk))
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def chunk_document(pages: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    """Chunk every page and flatten into a single (page_number, chunk) list."""
    all_chunks = []
    for page_number, text in pages:
        all_chunks.extend(chunk_page_text(page_number, text))
    return all_chunks
