import json
import sys
import tempfile
import shutil
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).parent))
from config import DATA_DIR, TESTS_DIR
from src.ingestor import ingest_all_pdfs, save_chunks, TESTS_DIR
from src.indexer import get_chroma_client, get_or_create_collection, index_to_chroma, build_bm25_index, load_bm25_index
from src.generator import ask
from src.assignment import generate_assignment


# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-PDF RAG",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Multi-PDF Intelligence Platform")
st.caption("Upload PDFs → Chat with them → Generate assignments")


# ── Session state ─────────────────────────────────────────────────
if "collection"  not in st.session_state: st.session_state.collection  = None
if "bm25"        not in st.session_state: st.session_state.bm25        = None
if "meta"        not in st.session_state: st.session_state.meta        = None
if "indexed"     not in st.session_state: st.session_state.indexed     = False
if "chat_history" not in st.session_state: st.session_state.chat_history = []


# ── Sidebar: PDF Upload & Indexing ────────────────────────────────
with st.sidebar:
    st.header("📂 Upload PDFs")

    uploaded_files = st.file_uploader(
        "Drop your PDFs here",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        for uf in uploaded_files:
            dest = DATA_DIR / uf.name
            if not dest.exists():
                with open(dest, "wb") as f:
                    f.write(uf.read())

        st.success(f"{len(uploaded_files)} PDF(s) uploaded")

    if st.button("⚙️ Index PDFs", use_container_width=True, type="primary"):
        pdfs = list(DATA_DIR.glob("*.pdf"))
        if not pdfs:
            st.error("No PDFs found. Upload at least one PDF first.")
        else:
            with st.spinner("Extracting text and building indexes..."):
                chunks = ingest_all_pdfs(DATA_DIR)

                if chunks:
                    TESTS_DIR.mkdir(exist_ok=True)
                    save_chunks(chunks, TESTS_DIR / "chunks.json")

                    client_db  = get_chroma_client()
                    collection = get_or_create_collection(client_db)
                    index_to_chroma(collection, chunks)
                    bm25 = build_bm25_index(chunks)
                    bm25_index, meta = load_bm25_index()

                    st.session_state.collection = collection
                    st.session_state.bm25       = bm25_index
                    st.session_state.meta       = meta
                    st.session_state.indexed    = True

                    child = sum(1 for c in chunks if c.chunk_type == "child")
                    st.success(f"Indexed {len(pdfs)} PDF(s) → {child} chunks")

    # Load existing index if already built
    if not st.session_state.indexed:
        try:
            client_db  = get_chroma_client()
            collection = get_or_create_collection(client_db)
            bm25, meta = load_bm25_index()
            if collection.count() > 0:
                st.session_state.collection = collection
                st.session_state.bm25       = bm25
                st.session_state.meta       = meta
                st.session_state.indexed    = True
        except Exception:
            pass

    st.divider()

    if st.session_state.indexed:
        st.success(f"✅ Index ready — {st.session_state.collection.count()} chunks")
        pdfs = list(DATA_DIR.glob("*.pdf"))
        if pdfs:
            st.caption("Loaded PDFs:")
            for p in pdfs:
                st.caption(f"  • {p.name}")
    else:
        st.info("Upload PDFs and click Index to begin")


# ── Main tabs ─────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["💬 Chat", "📝 Assignment Generator"])


# ── Tab 1: Chat ───────────────────────────────────────────────────
with tab1:
    st.subheader("Ask anything about your PDFs")

    if not st.session_state.indexed:
        st.warning("Index your PDFs first using the sidebar.")
    else:
        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and "sources" in msg:
                    with st.expander("📄 Sources"):
                        for s in msg["sources"]:
                            st.caption(f"• {s['source']}  —  page {s['page']}")

        # Chat input
        question = st.chat_input("Ask a question about your documents...")

        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Searching and generating answer..."):
                    result = ask(
                        question,
                        st.session_state.collection,
                        st.session_state.bm25,
                        st.session_state.meta
                    )

                st.markdown(result["answer"])

                with st.expander("📄 Sources"):
                    for s in result["sources"]:
                        st.caption(f"• {s['source']}  —  page {s['page']}")

            st.session_state.chat_history.append({
                "role":    "assistant",
                "content": result["answer"],
                "sources": result["sources"]
            })

        if st.session_state.chat_history:
            if st.button("🗑️ Clear chat"):
                st.session_state.chat_history = []
                st.rerun()


# ── Tab 2: Assignment Generator ───────────────────────────────────
with tab2:
    st.subheader("Generate assignments from your PDFs")

    if not st.session_state.indexed:
        st.warning("Index your PDFs first using the sidebar.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            topic = st.text_input(
                "Topic",
                placeholder="e.g. machine learning types, neural networks"
            )
            difficulty = st.selectbox(
                "Difficulty",
                ["beginner", "intermediate", "advanced"]
            )

        with col2:
            q_type = st.selectbox(
                "Question type",
                ["mcq", "short", "essay"],
                format_func=lambda x: {
                    "mcq":   "Multiple Choice (MCQ)",
                    "short": "Short Answer",
                    "essay": "Essay"
                }[x]
            )
            num_q = st.slider(
                "Number of questions",
                min_value=1, max_value=10, value=3,
                disabled=(q_type == "essay")
            )

        if st.button("🎓 Generate Assignment", type="primary", use_container_width=True):
            if not topic.strip():
                st.error("Enter a topic first.")
            else:
                with st.spinner(f"Generating {q_type.upper()} questions on '{topic}'..."):
                    result = generate_assignment(
                        topic         = topic,
                        difficulty    = difficulty,
                        q_type        = q_type,
                        num_questions = num_q if q_type != "essay" else 1,
                        collection    = st.session_state.collection,
                        bm25          = st.session_state.bm25,
                        meta_list     = st.session_state.meta
                    )

                st.success(f"Generated {len(result['questions'])} question(s)")

                # Display questions
                for i, q in enumerate(result["questions"], 1):
                    with st.expander(f"Question {i}: {q['question'][:80]}...", expanded=True):
                        st.markdown(f"**{q['question']}**")

                        if q_type == "mcq":
                            for letter, text in q["options"].items():
                                is_correct = letter == q["correct_answer"]
                                icon = "✅" if is_correct else "◦"
                                st.markdown(f"{icon} **{letter}.** {text}")
                            st.divider()
                            st.markdown(f"**Answer:** {q['correct_answer']}")
                            st.markdown(f"**Explanation:** {q['explanation']}")
                            st.caption(f"Source: page {q['source_page']}")

                        elif q_type == "short":
                            st.markdown(f"**Model answer:** {q['model_answer']}")
                            st.markdown("**Key points:**")
                            for pt in q["key_points"]:
                                st.markdown(f"  - {pt}")
                            st.caption(f"Source: page {q['source_page']}")

                        elif q_type == "essay":
                            st.markdown(f"**Guidance:** {q['guidance']}")
                            st.markdown("**Key themes:**")
                            for t in q["key_themes"]:
                                st.markdown(f"  - {t}")
                            st.markdown(f"**Suggested length:** {q['suggested_length']}")
                            st.caption(f"Source pages: {q['source_pages']}")

                # Sources
                with st.expander("📄 PDF sources used"):
                    for s in result["sources"]:
                        st.caption(f"• {s['source']}  —  page {s['page']}")

                # Download button
                out_json = json.dumps(result, indent=2, ensure_ascii=False)
                st.download_button(
                    label     = "⬇️ Download assignment (JSON)",
                    data      = out_json,
                    file_name = f"assignment_{topic[:20]}.json",
                    mime      = "application/json"
                )