from typing import Callable, Optional, List
from db.db import get_conn
from rag.util.time import now_iso
from rag.util.ids import new_id

from rag.ingest.loader import load_text
from rag.ingest.chunker import chunk_pages
from rag.index.vector_store import get_chroma_collection
from rag.llm.embeddings import embed_texts

def ingest_path(
    collection_id: str,
    document_id: str,
    filename: str,
    path: str,
    on_progress: Optional[Callable[[float], None]] = None,
) -> None:
    if on_progress:
        on_progress(0.05)

    pages = load_text(path)
    if on_progress:
        on_progress(0.15)

    page_tuples = [(p.text, p.page) for p in pages]
    chunks = chunk_pages(page_tuples, chunk_size=1200, overlap=200)
    if on_progress:
        on_progress(0.30)

    conn = get_conn()
    try:
        # clear existing (idempotent)
        existing = conn.execute("SELECT id FROM chunks WHERE document_id=?", (document_id,)).fetchall()
        for r in existing:
            conn.execute("DELETE FROM chunks_fts WHERE chunk_id=?", (r["id"],))
        conn.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))
        conn.commit()

        # insert chunks
        ids: List[str] = []
        texts: List[str] = []
        metas: List[dict] = []

        for i, ch in enumerate(chunks):
            cid = new_id()
            ids.append(cid)
            texts.append(ch.text)
            metas.append({
                "chunk_id": cid,
                "document_id": document_id,
                "collection_id": collection_id,
                "filename": filename,
                "page_start": ch.page_start if ch.page_start is not None else -1,
                "page_end": ch.page_end if ch.page_end is not None else -1,
                "chunk_index": i,
            })
            conn.execute(
                "INSERT INTO chunks (id, document_id, chunk_index, text, page_start, page_end, section_title, char_start, char_end, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (cid, document_id, i, ch.text, ch.page_start, ch.page_end, None, ch.char_start, ch.char_end, now_iso()),
            )
            conn.execute(
                "INSERT INTO chunks_fts (text, chunk_id, document_id, filename) VALUES (?, ?, ?, ?)",
                (ch.text, cid, document_id, filename),
            )
        conn.commit()
    finally:
        conn.close()

    if on_progress:
        on_progress(0.45)

    # embeddings
    embs = embed_texts(texts)
    if on_progress:
        on_progress(0.75)

    # index into chroma
    coll = get_chroma_collection(collection_id)
    # ensure no duplicate ids
    try:
        coll.delete(ids=ids)
    except Exception:
        pass
    coll.add(ids=ids, embeddings=embs, metadatas=metas, documents=texts)

    if on_progress:
        on_progress(0.95)
