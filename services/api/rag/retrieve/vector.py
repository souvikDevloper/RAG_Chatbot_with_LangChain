from typing import List, Tuple
from rag.index.vector_store import get_chroma_collection

def vector_search(collection_id: str, query_emb: list[float], k: int, document_ids: list[str] | None = None, filename_contains: str | None = None) -> List[Tuple[str, float]]:
    coll = get_chroma_collection(collection_id)

    where = None
    if document_ids:
        where = {"document_id": {"$in": document_ids}}

    # fetch a bit more if we need to post-filter by filename
    n_results = int(k * 3) if filename_contains else int(k)

    res = coll.query(
        query_embeddings=[query_emb],
        n_results=n_results,
        where=where,
        include=["distances", "metadatas"],
    )

    ids = res.get("ids", [[]])[0] or []
    dists = res.get("distances", [[]])[0] or []
    metas = res.get("metadatas", [[]])[0] or []

    out: List[Tuple[str, float]] = []
    for i, cid in enumerate(ids):
        meta = metas[i] if i < len(metas) else {}
        if filename_contains and filename_contains.lower() not in str(meta.get("filename","")).lower():
            continue
        dist = float(dists[i]) if i < len(dists) else 0.0
        score = 1.0 / (1.0 + max(0.0, dist))
        out.append((cid, score))
        if len(out) >= int(k):
            break
    return out
