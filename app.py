"""LocalMind — chat with your documents, 100% offline.

Run with:  streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from rag import CHAT_MODEL, EMBED_MODEL, INDEX_DIR, RAG

st.set_page_config(page_title="LocalMind", page_icon="🧠", layout="wide")


@st.cache_resource(show_spinner=False)
def get_rag(embed_model: str, chat_model: str) -> RAG:
    rag = RAG(embed_model=embed_model, chat_model=chat_model)
    rag.load(INDEX_DIR)
    return rag


# ----- sidebar ------------------------------------------------------------
with st.sidebar:
    st.title("🧠 LocalMind")
    st.caption("Chat with your documents — fully offline, powered by Ollama.")

    chat_model = st.text_input("Chat model", value=CHAT_MODEL)
    embed_model = st.text_input("Embedding model", value=EMBED_MODEL)
    top_k = st.slider("Chunks to retrieve", 1, 10, 4)

    st.divider()
    st.markdown(f"**Documents folder:** `./docs`")
    docs_count = len(list(Path("docs").rglob("*"))) if Path("docs").exists() else 0
    st.caption(f"{docs_count} file(s) found in ./docs")

    if st.button("🔄 (Re)build index", use_container_width=True):
        with st.spinner("Embedding documents locally…"):
            rag = RAG(embed_model=embed_model, chat_model=chat_model)
            n = rag.build("docs")
            rag.save(INDEX_DIR)
        get_rag.clear()
        if n:
            st.success(f"Indexed {n} chunks.")
        else:
            st.warning("No documents found. Add files to ./docs first.")

    st.divider()
    st.markdown(
        "Made with [Ollama](https://ollama.com). "
        "[⭐ Star on GitHub](https://github.com/)"
    )


# ----- main chat ----------------------------------------------------------
rag = get_rag(embed_model, chat_model)

st.title("Chat with your documents")

if rag.embeddings is None:
    st.info(
        "No index found yet. Drop PDFs, `.txt`, or `.md` files into the **./docs** "
        "folder, then click **(Re)build index** in the sidebar."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask anything about your documents…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = st.write_stream(rag.answer(prompt, k=top_k))

    st.session_state.messages.append({"role": "assistant", "content": response})
