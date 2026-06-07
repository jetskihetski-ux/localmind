"""Local retrieval-augmented generation powered entirely by Ollama.

Embeddings and chat both run through your local Ollama instance, so nothing
is ever sent to a third-party API.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np
import ollama

from ingest import build_chunks

# Defaults — override with env vars or the Streamlit sidebar.
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2"
INDEX_DIR = Path(".localmind_index")

SYSTEM_PROMPT = (
    "You are LocalMind, a helpful assistant that answers questions using only "
    "the provided context from the user's documents. If the answer is not in "
    "the context, say so plainly instead of guessing. Cite the source file "
    "names you used."
)


def embed(texts: list[str], model: str = EMBED_MODEL) -> np.ndarray:
    """Embed a list of strings into a normalized matrix for cosine similarity."""
    vectors = []
    for text in texts:
        response = ollama.embeddings(model=model, prompt=text)
        vectors.append(response["embedding"])
    matrix = np.array(vectors, dtype=np.float32)
    # Normalize so a plain dot product equals cosine similarity.
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class RAG:
    """A tiny, dependency-light vector store + chat wrapper over Ollama."""

    def __init__(self, embed_model: str = EMBED_MODEL, chat_model: str = CHAT_MODEL):
        self.embed_model = embed_model
        self.chat_model = chat_model
        self.embeddings: np.ndarray | None = None
        self.chunks: list[dict] = []

    # ----- index lifecycle -------------------------------------------------
    def build(self, docs_folder: str = "docs") -> int:
        """Build the index from every document in ``docs_folder``."""
        self.chunks = build_chunks(docs_folder)
        if not self.chunks:
            self.embeddings = None
            return 0
        self.embeddings = embed([c["text"] for c in self.chunks], self.embed_model)
        return len(self.chunks)

    def save(self, index_dir: Path = INDEX_DIR) -> None:
        """Persist the index to disk so we don't re-embed on every launch."""
        index_dir.mkdir(parents=True, exist_ok=True)
        if self.embeddings is not None:
            np.save(index_dir / "embeddings.npy", self.embeddings)
        (index_dir / "chunks.json").write_text(
            json.dumps(self.chunks, ensure_ascii=False), encoding="utf-8"
        )

    def load(self, index_dir: Path = INDEX_DIR) -> bool:
        """Load a previously saved index. Returns False if none exists."""
        emb_path = index_dir / "embeddings.npy"
        chunks_path = index_dir / "chunks.json"
        if not emb_path.exists() or not chunks_path.exists():
            return False
        self.embeddings = np.load(emb_path)
        self.chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        return True

    # ----- retrieval + generation -----------------------------------------
    def retrieve(self, query: str, k: int = 4) -> list[dict]:
        """Return the ``k`` chunks most similar to ``query``."""
        if self.embeddings is None or not self.chunks:
            return []
        q = embed([query], self.embed_model)[0]
        scores = self.embeddings @ q
        top = np.argsort(scores)[::-1][:k]
        return [self.chunks[i] for i in top]

    def answer(self, query: str, k: int = 4) -> Iterator[str]:
        """Stream an answer grounded in the retrieved context."""
        context_chunks = self.retrieve(query, k)
        if not context_chunks:
            yield "No documents indexed yet. Add files to ./docs and rebuild the index."
            return

        context = "\n\n".join(
            f"[Source: {c['source']}]\n{c['text']}" for c in context_chunks
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            },
        ]
        stream = ollama.chat(model=self.chat_model, messages=messages, stream=True)
        for part in stream:
            yield part["message"]["content"]
