<div align="center">

# 🧠 LocalMind

### Chat with your documents — 100% offline, 100% private.

Drop in your PDFs, notes, and Markdown files, then ask questions in plain English.
Powered by [Ollama](https://ollama.com) — **no API keys, no cloud, nothing ever leaves your machine.**

![Tests](https://github.com/jetskihetski-ux/localmind/actions/workflows/tests.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)
![Offline](https://img.shields.io/badge/100%25-offline-brightgreen.svg)
![Powered by Ollama](https://img.shields.io/badge/powered%20by-Ollama-black.svg)

</div>

---

## ✨ Why LocalMind?

- 🔒 **Truly private** — your documents never touch a third-party server.
- 💸 **Free forever** — no OpenAI bill, no rate limits. Runs on your own hardware.
- ⚡ **Dead simple** — three commands and you're chatting with your files.
- 📚 **Cites its sources** — answers tell you which file they came from.
- 🪶 **Lightweight** — a tiny NumPy vector store, no heavy database to set up.

> Perfect for searching study notes, research papers, contracts, or any pile of
> documents you'd rather not upload to the cloud.

## 🎬 Demo

> _Add a screen recording here once you run it — a GIF in the first screen is the
> single biggest driver of GitHub stars._

```
You:        What is the secret passphrase?
LocalMind:  The secret passphrase is "blue penguin 42", according to sample.md.
```

## 🚀 Quickstart

### 1. Install Ollama and pull the models

Download Ollama from [ollama.com](https://ollama.com), then:

```bash
ollama pull llama3.2          # the chat model
ollama pull nomic-embed-text  # the embedding model
```

### 2. Install LocalMind

```bash
git clone https://github.com/jetskihetski-ux/localmind.git
cd localmind
pip install -r requirements.txt
```

### 3. Add your documents and run

```bash
# Drop your PDFs / .txt / .md files into the docs/ folder, then:
streamlit run app.py
```

Click **(Re)build index** in the sidebar, and start asking questions. That's it!

## 🧩 How it works

LocalMind uses **RAG** (Retrieval-Augmented Generation):

```
Your docs ──chunk──▶ embed ──▶ [vectors stored on disk]
                                      │
Your question ──embed──▶ compare ─────┘──▶ top matches ──▶ LLM ──▶ grounded answer
```

1. **Ingest** — documents are split into overlapping chunks ([ingest.py](ingest.py)).
2. **Embed** — each chunk becomes a vector via `nomic-embed-text` ([rag.py](rag.py)).
3. **Retrieve** — your question is matched against chunks by cosine similarity.
4. **Generate** — the most relevant chunks are handed to the chat model to answer.

All three stages run locally through Ollama.

## ⚙️ Configuration

Change models or retrieval depth right in the sidebar, or edit the defaults at
the top of [rag.py](rag.py):

| Setting       | Default            | Description                      |
|---------------|--------------------|----------------------------------|
| `CHAT_MODEL`  | `llama3.2`         | Any chat model you've pulled     |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model for retrieval    |
| `top_k`       | `4`                | How many chunks to feed the LLM  |

## 📂 Project structure

```
localmind/
├── app.py            # Streamlit chat UI
├── rag.py            # embedding, retrieval, and generation
├── ingest.py         # document loading + chunking
├── docs/             # ← put your files here
├── requirements.txt
└── README.md
```

## 🗺️ Roadmap

- [ ] Drag-and-drop file upload in the UI
- [ ] Chat history persistence
- [ ] Support for `.docx` and `.html`
- [ ] Per-source filtering

## 🤝 Contributing

Issues and pull requests are welcome! If LocalMind is useful to you, please
consider leaving a ⭐ — it really helps.

## 📜 License

[MIT](LICENSE) — free to use, modify, and share.
