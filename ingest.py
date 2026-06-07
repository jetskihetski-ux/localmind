"""Document loading and chunking for LocalMind.

Supports PDF, plain text, and Markdown. Everything runs locally — no files
ever leave your machine.
"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

# File extensions we know how to read.
TEXT_SUFFIXES = {".txt", ".md", ".markdown"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | {".pdf"}


def read_pdf(path: Path) -> str:
    """Extract text from every page of a PDF."""
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def read_text(path: Path) -> str:
    """Read a plain-text or Markdown file."""
    return path.read_text(encoding="utf-8", errors="ignore")


def load_file(path: Path) -> str:
    """Load a single supported file into raw text."""
    if path.suffix.lower() == ".pdf":
        return read_pdf(path)
    return read_text(path)


def load_documents(folder: str | Path) -> list[tuple[str, str]]:
    """Load every supported file in ``folder`` (recursively).

    Returns a list of ``(text, source_name)`` tuples.
    """
    folder = Path(folder)
    docs: list[tuple[str, str]] = []
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            text = load_file(path).strip()
            if text:
                docs.append((text, path.name))
    return docs


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Split text into overlapping word-based chunks.

    Overlap keeps context from spilling across chunk boundaries so retrieval
    stays coherent.
    """
    words = text.split()
    if not words:
        return []

    step = max(chunk_size - overlap, 1)
    chunks = []
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size])
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
    return chunks


def build_chunks(folder: str | Path) -> list[dict]:
    """Load all documents and split them into retrievable chunks.

    Returns a list of ``{"text": ..., "source": ...}`` dicts.
    """
    records: list[dict] = []
    for text, source in load_documents(folder):
        for chunk in chunk_text(text):
            records.append({"text": chunk, "source": source})
    return records
