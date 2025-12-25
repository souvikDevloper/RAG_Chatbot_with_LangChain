"""
Core RAG helpers shared by Streamlit UI and FastAPI API
------------------------------------------------------

• Builds (or reloads) the FAISS vector store.
• Creates a Retrieval-QA chain.
• Provides two thin wrappers:
      query(question)  -> str
      add_document(txt)   # persist to store
"""
import os
from typing import Tuple, List

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# --- Config -----------------------------------------------------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_STORE_PATH    = "faiss_store"              # Folder created on first run
OPENAI_MODEL_NAME    = "gpt-3.5-turbo"            # or "gpt-4o-mini" / etc.
# ---------------------------------------------------------------------------

def _build_vector_store(embeddings: HuggingFaceEmbeddings) -> FAISS:
    """Load existing FAISS index if present; else create an empty one."""
    if os.path.exists(VECTOR_STORE_PATH):
        return FAISS.load_local(VECTOR_STORE_PATH, embeddings)
    # Empty store (you can add docs later through /upload endpoints)
    return FAISS.from_texts([], embeddings)

def _build_chain() -> Tuple[RetrievalQA, FAISS]:
    embeddings  = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = _build_vector_store(embeddings)

    llm   = OpenAI(model_name=OPENAI_MODEL_NAME, temperature=0.0)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=False,
    )
    return chain, vectorstore


_chain, _vs = _build_chain()   # singleton – created once when module loads
# ---------------------------------------------------------------------------

def query(question: str) -> str:
    """Ask a question, get a text answer."""
    return _chain.run(question)

def add_document(text: str) -> None:
    """Add a plain-text document to the vector DB and persist it."""
    _vs.add_texts([text])
    _vs.save_local(VECTOR_STORE_PATH)
