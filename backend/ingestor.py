from dataclasses import dataclass, field
from typing import List
import re
import uuid
from pypdf import PdfReader
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150



@dataclass
class Chunk:
    id: str
    text: str
    source: str        # original filename
    page: int           # 1-indexed page number this chunk starts on
    chunk_index: int    # position of chunk within the document


@dataclass
class IngestResult:
    filename: str
    chunks: List[Chunk] = field(default_factory=list)
    error: str = None
    is_empty: bool = False



def extract_pages(file_path: str) -> List[str]:
    """Return a list of per-page text strings for a PDF."""
    reader = PdfReader(file_path)
    pages = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append(text)
    return pages



def _split_into_windows(text: str, size: int, overlap: int) -> List[str]:
    """Sliding window split, snapping to the nearest sentence boundary."""
    if len(text) <= size:
        return [text] if text.strip() else []
    windows = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        # try to snap to a sentence/paragraph boundary near end
        if end < n:
            snap_window = text[end:end + 200]
            match = re.search(r"[.!?]\s", snap_window)
            if match:
                end = end + match.end()
        chunk_text = text[start:end].strip()
        if chunk_text:
            windows.append(chunk_text)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return windows


def ingest_pdf(file_path: str, filename: str) -> IngestResult:
    """Load a single PDF, extract text per page, and chunk it."""
    try:
        pages = extract_pages(file_path)
    except Exception as e:
        return IngestResult(filename=filename, error=f"Failed to read PDF: {e}")
    full_text_len = sum(len(p.strip()) for p in pages)
    if full_text_len == 0:
        return IngestResult(
            filename=filename,
            is_empty=True,
            error="No extractable text found (PDF may be scanned/image-only).",
        )
    chunks: List[Chunk] = []
    chunk_idx = 0
    for page_num, page_text in enumerate(pages, start=1):
        if not page_text.strip():
            continue
        for window in _split_into_windows(page_text, CHUNK_SIZE, CHUNK_OVERLAP):
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    text=window,
                    source=filename,
                    page=page_num,
                    chunk_index=chunk_idx,
                )
            )
            chunk_idx += 1
    return IngestResult(filename=filename, chunks=chunks)


def ingest_many(file_paths_and_names: List[tuple]) -> List[IngestResult]:
    """file_paths_and_names: list of (path_on_disk, original_filename)."""
    return [ingest_pdf(path, name) for path, name in file_paths_and_names]


