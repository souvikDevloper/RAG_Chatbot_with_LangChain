import streamlit as st
from api_client import create_collection, list_collections, upload_document, list_documents, get_job, chat

st.set_page_config(page_title="Agentic Hybrid RAG", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --bg: #0f172a;
        --panel: #111827;
        --accent: #2496ed;
        --text: #e5e7eb;
        --muted: #94a3b8;
        --line: rgba(148, 163, 184, 0.2);
    }
    .stApp {
        background: radial-gradient(circle at top left, #0b1f3a 0%, #0f172a 45%, #0b1020 100%);
        color: var(--text);
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        background: rgba(17, 24, 39, 0.92);
        border-radius: 18px;
        box-shadow: 0 18px 48px rgba(2, 6, 23, 0.65);
        border: 1px solid rgba(148, 163, 184, 0.08);
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
        border-right: 1px solid var(--line);
    }
    h1, h2, h3, h4, h5, h6 {
        color: var(--text);
    }
    .stCaption, .stMarkdown p {
        color: var(--muted);
    }
    .stButton > button {
        background: linear-gradient(135deg, #2496ed 0%, #3aa9ff 100%);
        color: #0b1020;
        border-radius: 10px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        box-shadow: 0 10px 22px rgba(36, 150, 237, 0.35);
    }
    .stTextInput input, .stSelectbox select, .stTextArea textarea {
        background: #0b1020;
        color: var(--text);
        border-radius: 10px;
        border: 1px solid var(--line);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Agentic Hybrid RAG")

with st.sidebar:
    st.subheader("BYOK Provider")
    llm_api_key = st.text_input("LLM API key", type="password")
    embed_api_key = st.text_input("Embeddings API key (for ingestion + chat)", type="password") or llm_api_key
    base_url = st.text_input("OpenAI-compatible base_url", value="https://api.openai.com/v1")
    llm_model = st.text_input("LLM model", value="gpt-4o-mini")
    emb_model = st.text_input("Embedding model", value="text-embedding-3-small")

    use_rerank = st.checkbox("Use rerank (Cohere BYOK)", value=False)
    rerank_key = st.text_input("Cohere API key", type="password") if use_rerank else ""
    rerank_model = st.text_input("Cohere rerank model", value="rerank-english-v3.0") if use_rerank else ""

st.divider()

cols = st.columns([1, 2])
with cols[0]:
    st.subheader("Collections")
    try:
        cols_list = list_collections()
    except Exception as e:
        st.error(f"API not reachable: {e}")
        st.stop()

    names = {c["name"]: c["id"] for c in cols_list}
    sel_name = st.selectbox("Select", ["(create new)"] + list(names.keys()))
    if sel_name == "(create new)":
        new_name = st.text_input("New collection name")
        if st.button("Create") and new_name.strip():
            c = create_collection(new_name.strip())
            st.success(f"Created: {c['name']}")
            st.rerun()
        st.stop()
    collection_id = names[sel_name]
    st.code(collection_id)

with cols[1]:
    st.subheader("Upload documents")
    if not embed_api_key:
        st.warning("Add your Embeddings API key in the sidebar to ingest documents (BYOK).")
    up = st.file_uploader("Upload PDF/TXT/CSV/DOCX", accept_multiple_files=True)
    if up and embed_api_key:
        for f in up:
            if st.button(f"Upload: {f.name}", key=f"up_{f.name}"):
                res = upload_document(collection_id, f.getvalue(), f.name, embed_api_key, base_url, emb_model)
                st.success(f"Queued job: {res['job_id']}")
                st.session_state.setdefault("jobs", []).append(res["job_id"])
    st.write("Jobs:")
    jobs = st.session_state.get("jobs", [])
    for jid in jobs[-10:]:
        try:
            j = get_job(jid)
            st.write(f"{jid[:8]}…  {j['status']}  {int(j['progress']*100)}%")
        except Exception:
            pass

st.divider()
st.subheader("Chat")

docs = list_documents(collection_id)
ready_docs = [d for d in docs if d["status"] == "ready"]
st.caption(f"Docs: {len(docs)} total, {len(ready_docs)} ready")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask something about your uploaded docs…")
if prompt:
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not llm_api_key:
        with st.chat_message("assistant"):
            st.error("Add your LLM API key in the sidebar (BYOK).")
        st.stop()

    payload = {
        "collection_id": collection_id,
        "session_id": st.session_state.session_id,
        "message": prompt,
        "filters": {"document_ids": [], "filename_contains": None},
        "provider": {
            "llm_base_url": base_url,
            "llm_model": llm_model,
            "embed_base_url": base_url,
            "embed_model": emb_model,
            "rerank_provider": "cohere" if use_rerank else None,
            "rerank_model": rerank_model if use_rerank else None,
        },
        "keys": {
            "llm_api_key": llm_api_key,
            "embed_api_key": embed_api_key or llm_api_key,
            "rerank_api_key": rerank_key if use_rerank else None,
        },
        "options": {"k": 12, "use_rerank": bool(use_rerank)},
    }

    with st.chat_message("assistant"):
        try:
            res = chat(payload)
            st.markdown(res["answer"])
            with st.expander("Citations"):
                for c in res.get("citations", []):
                    st.write(f"- {c['filename']} p{c.get('page_start')}–{c.get('page_end')}  (chunk {c['chunk_id'][:8]}…) score={c['score']:.3f}")
            with st.expander("Agent trace"):
                st.json(res.get("agent_trace", {}))
            st.session_state.messages.append({"role":"assistant","content":res["answer"]})
        except Exception as e:
            st.error(str(e))
