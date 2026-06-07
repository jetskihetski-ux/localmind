"""Cross-validation test suite for LocalMind (10 cases).

Ollama needs a live server, so we inject a deterministic fake `ollama` module
before importing the app code. The fake uses a hashing vectorizer for
embeddings, which means retrieval ranking is genuinely tested: chunks that
share words with the query really do score higher.

Run:  python test_localmind.py
"""

import hashlib
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Inject a fake `ollama` module BEFORE importing rag/ingest.
# ---------------------------------------------------------------------------
_DIM = 64


def _hash_vec(text: str) -> list[float]:
    """Deterministic bag-of-words hashing vector (mimics real embeddings)."""
    v = np.zeros(_DIM, dtype=float)
    for tok in text.lower().split():
        idx = int(hashlib.md5(tok.encode()).hexdigest(), 16) % _DIM
        v[idx] += 1.0
    return v.tolist()


_fake = types.ModuleType("ollama")
_fake.embeddings = lambda model, prompt: {"embedding": _hash_vec(prompt)}


def _fake_chat(model, messages, stream=False):
    user_content = messages[-1]["content"]

    def gen():
        # Echo the context back so tests can assert it was passed through.
        yield {"message": {"content": f"[answer based on] {user_content}"}}

    return gen() if stream else {"message": {"content": "[answer]"}}


_fake.chat = _fake_chat
sys.modules["ollama"] = _fake

# Now safe to import the real modules.
import ingest  # noqa: E402
import rag  # noqa: E402


def make_docs(tmp: Path) -> Path:
    """Create a small multi-file corpus for retrieval tests."""
    docs = tmp / "docs"
    docs.mkdir()
    (docs / "cats.txt").write_text(
        "Cats are small domesticated feline animals that purr and meow."
    )
    (docs / "finance.txt").write_text(
        "The quarterly revenue report shows profit margins increased sharply."
    )
    (docs / "passphrase.txt").write_text(
        "The secret passphrase for the demo is blue penguin forty two."
    )
    (docs / "ignore.csv").write_text("a,b,c\n1,2,3")  # unsupported -> skipped
    return docs


class TestChunking(unittest.TestCase):
    def test_01_chunk_count_and_size(self):
        words = " ".join(f"w{i}" for i in range(2000))
        chunks = ingest.chunk_text(words, chunk_size=800, overlap=150)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(len(chunks[0].split()), 800)

    def test_02_chunk_overlap_is_preserved(self):
        words = " ".join(f"w{i}" for i in range(2000))
        chunks = ingest.chunk_text(words, chunk_size=800, overlap=150)
        self.assertEqual(chunks[0].split()[-150:], chunks[1].split()[:150])

    def test_03_empty_text_returns_no_chunks(self):
        self.assertEqual(ingest.chunk_text(""), [])
        self.assertEqual(ingest.chunk_text("   \n  "), [])

    def test_04_short_text_is_single_chunk(self):
        chunks = ingest.chunk_text("just a few words here", chunk_size=800)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], "just a few words here")


class TestIngest(unittest.TestCase):
    def test_05_load_documents_skips_unsupported(self):
        with tempfile.TemporaryDirectory() as d:
            docs = make_docs(Path(d))
            loaded = ingest.load_documents(docs)
            sources = {name for _, name in loaded}
            self.assertEqual(sources, {"cats.txt", "finance.txt", "passphrase.txt"})

    def test_06_build_chunks_shape(self):
        with tempfile.TemporaryDirectory() as d:
            docs = make_docs(Path(d))
            records = ingest.build_chunks(docs)
            self.assertTrue(records)
            for r in records:
                self.assertIn("text", r)
                self.assertIn("source", r)


class TestEmbeddings(unittest.TestCase):
    def test_07_embeddings_are_normalized(self):
        mat = rag.embed(["hello world", "another sentence here"])
        norms = np.linalg.norm(mat, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)


class TestRAG(unittest.TestCase):
    def test_08_retrieval_ranks_relevant_chunk_first(self):
        with tempfile.TemporaryDirectory() as d:
            docs = make_docs(Path(d))
            r = rag.RAG()
            r.build(str(docs))
            top = r.retrieve("what is the secret passphrase", k=1)
            self.assertEqual(top[0]["source"], "passphrase.txt")
            # A different query should surface a different document.
            top_cats = r.retrieve("tell me about cats", k=1)
            self.assertEqual(top_cats[0]["source"], "cats.txt")

    def test_09_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            docs = make_docs(Path(d))
            idx = Path(d) / "index"
            r = rag.RAG()
            n = r.build(str(docs))
            r.save(idx)

            r2 = rag.RAG()
            self.assertTrue(r2.load(idx))
            self.assertEqual(len(r2.chunks), n)
            np.testing.assert_array_equal(r.embeddings, r2.embeddings)

    def test_10_answer_handles_empty_and_grounds_response(self):
        # Empty index -> graceful message, no crash.
        empty = rag.RAG()
        msg = "".join(empty.answer("anything"))
        self.assertIn("No documents indexed", msg)

        # Populated index -> answer is grounded in retrieved context.
        with tempfile.TemporaryDirectory() as d:
            docs = make_docs(Path(d))
            r = rag.RAG()
            r.build(str(docs))
            out = "".join(r.answer("what is the secret passphrase"))
            self.assertIn("passphrase", out.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
